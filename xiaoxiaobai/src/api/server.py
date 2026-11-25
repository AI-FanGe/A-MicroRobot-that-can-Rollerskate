# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

"""Robot API server: REST (/api/v1), Server-Sent Events and the web UI, all on one port.

Run on the robot:   PYTHONPATH=src .venv/bin/python -m api
Try it on a PC:     PYTHONPATH=src python -m api --mock
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import logging
import mimetypes
import os
import queue
import re
import signal
import socket
import threading
import time
import traceback
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, unquote, urlsplit

from api import API_VERSION, llm, memo
from api.backend import REPO, Backend, MockBackend, RealBackend
from api.ble import BleLink
from api.chat import ChatService
from api.events import EventBus
from api.interaction import Interaction, SystemInfo, VoiceSession
from api.knowledge import KnowledgeBase
from api.memo import MemoStore
from api.memory import ConversationStore, MemoryStore
from api.motion import ACTIONS, MotionController
from api.music import MusicService
from api.netease import QUALITIES
from api.protocol import ApiError, CommonState, Reason, TaskRegistry, error_response, response
from api.store import DEFAULT_PERSONA, ROBOT_NAME, Settings, is_masked, mask_secret
from api.wifi import MockWifi, Wifi
from api.workbuddy import WorkBuddy, normalize_url

WEBUI = REPO / "webui"
MAX_BODY = 4_000_000
DEFAULT_PORT = 8000
REMIND_POLL_S = 15.0
# Streams and files: HTTP over the Wi-Fi LAN only, never the app's Bluetooth link.
HTTP_ONLY = {"SubscribeEvents", "Chat", "GetCameraStream", "GetCameraSnapshot", "PreviewVoice", "ExportMemos", "GetWorkBuddyInstaller",
             "GetWalkRecording", "GetWalkDiagnostics"}
# Per-tick recordings of the walks and dances (walk_recorder.py; the control loop runs in REPO).
WALK_REC_DIR = REPO / "logs" / "walk_rec"
WALK_REC_NAME = re.compile(r"^[a-z_]+_\d{8}_\d{6}\.(npz|part)$")
WALK_REC_LIST_MAX = 100
WALK_REC_BUNDLE_MAX = 10
BACKGROUND_NICE = 15  # downloads read the SD card beside the control loop: at the lowest priority


def _background_thread() -> None:
    """Lower the calling thread (one per request) under the control loop and the rest of the API."""
    try:
        os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), BACKGROUND_NICE)
    except (AttributeError, OSError):
        pass
log = logging.getLogger("xiaoxiaobai.api")


@dataclass
class Route:
    method: str
    path: str
    handler: Callable[["Request"], Any]
    name: str
    group: str
    summary: str
    example: Any = None
    regex: re.Pattern = field(init=False)

    def __post_init__(self) -> None:
        self.regex = re.compile("^" + re.sub(r"\{(\w+)\}", r"(?P<\1>[^/]+)", self.path) + "$")


@dataclass
class Request:
    params: dict[str, str]
    query: dict[str, list[str]]
    body: Any
    client: str = ""  # caller's IP over HTTP; empty over Bluetooth
    host: str = ""  # the Host header: how the caller reaches this robot

    def arg(self, name: str, default: str | None = None) -> str | None:
        values = self.query.get(name)
        return values[0] if values else default

    def json(self) -> dict[str, Any]:
        if self.body is None:
            return {}
        if not isinstance(self.body, dict):
            raise ApiError("请求体需要是 JSON 对象", CommonState.INVALID)
        return self.body


@dataclass
class Raw:
    body: bytes
    content_type: str
    status: int = 200
    filename: str = ""  # sent as an attachment (download) under this name


@dataclass
class Stream:
    write: Callable[["Handler"], None]


def robot_id() -> str:
    """Stable id of this robot for the phone app (its saved pairing), from the machine id."""
    try:
        seed = Path("/etc/machine-id").read_text(encoding="ascii").strip()
    except OSError:
        seed = ""
    return hashlib.sha256((seed or socket.gethostname()).encode()).hexdigest()[:12]


def _num(values: dict[str, Any], key: str, lo: float, hi: float, cast: type = float) -> None:
    if key in values:
        try:
            values[key] = cast(min(hi, max(lo, float(values[key]))))
        except (TypeError, ValueError):
            raise ApiError(f"{key} 需要是数字", CommonState.INVALID) from None


class App:
    def __init__(self, mock: bool, ble: bool = False, port: int = DEFAULT_PORT) -> None:
        self.mock = mock
        self.port = port
        self.backend: Backend = MockBackend() if mock else RealBackend()
        self.wifi: Wifi = MockWifi() if mock else Wifi()
        self.events = EventBus()
        self.tasks = TaskRegistry()
        self.settings = Settings()
        self.motion = MotionController(self.backend, self.events, self.tasks, self.settings)
        self.interaction = Interaction(self.backend, self.events)
        self.memory = MemoryStore()
        self.memos = MemoStore(timezone=lambda: self.settings.section("memo").get("timezone", ""),
                               on_change=lambda action, item: self.events.publish("memo", action=action, id=item.get("id")))
        self.conversations = ConversationStore()
        self.knowledge = KnowledgeBase()
        self.workbuddy = WorkBuddy(self.settings, self.events, self._on_workbuddy_done, mock=mock, speak=self._workbuddy_speak)
        self.chat = ChatService(
            self.settings, self.motion, self.interaction, self.memory, self.conversations, self.knowledge, self.memos,
            workbuddy=self.workbuddy, demo=mock,
        )
        self._voice_conversation: str | None = None
        self.voice = VoiceSession(self.backend, self.events, self.settings, self._on_voice_turn, self.chat.voice_instructions)
        self.music = MusicService(self.backend, self.events, self.settings, self.interaction, self.voice)
        self.chat.music = self.music
        self.music.motion = self.motion
        self.motion.music = self.music
        self.system = SystemInfo(self.backend)
        self.shutting_down = threading.Event()
        self._snapshot: tuple[float, dict[str, Any]] = (0.0, {})
        self._snapshot_lock = threading.Lock()
        self._pair_board = False
        self.routes = self._routes()
        self.ble = BleLink(self.call, self.events, self.snapshot, self.ble_info, self.setup_mode, self._show_pair_code) if ble else None

    def start(self) -> None:
        self._apply_brightness()
        self.motion.start()
        threading.Thread(target=self._remind_loop, name="xiaoxiaobai-api-memo", daemon=True).start()
        if self.ble is not None:
            self.ble.start()

    def _remind_loop(self) -> None:
        """A memo whose time has come: chime, the memo on the face screen and a web UI alert."""
        while not self.shutting_down.wait(REMIND_POLL_S):
            cfg = self.settings.section("memo")
            if not cfg.get("enabled", True) or not cfg.get("remind", True):
                continue
            due = self.memos.due()
            if not due:
                continue
            self.chat.show_memos(None, None, "提醒", "reminder", title="提醒", title_color="pink", items=due, seconds=45)
            try:
                self.interaction.play_sound("chime", None, 70)
            except ApiError:
                pass
            for m in due:
                self.events.alert("warning", f"提醒：{m['time']} {m['text']}")

    def close(self) -> None:
        self.shutting_down.set()
        if self.ble is not None:
            self.ble.close()
        self.motion.close()
        self.music.close()
        if self.voice.status()["running"]:
            self.voice.stop()

    # ---- helpers ---------------------------------------------------------------------------

    def _apply_brightness(self) -> None:
        """Hands the saved screen brightness to the display loop, which polls the file."""
        from esp32_link import write_brightness

        try:
            self.backend.paths.root.mkdir(parents=True, exist_ok=True)
            write_brightness(self.settings.section("display").get("brightness", 100), self.backend.paths.brightness)
        except OSError as exc:
            log.warning("cannot write the screen brightness: %s", exc)

    def _on_voice_turn(self, role: str, text: str, interrupted: bool) -> None:
        if not text:
            return
        if self._voice_conversation is None or not self.conversations.exists(self._voice_conversation):
            self._voice_conversation = self.conversations.create("voice")["id"]
        meta = {"interrupted": True} if interrupted else {}
        self.conversations.append(self._voice_conversation, role, text, **meta)
        robot_name = self.settings.section("robot").get("name") or ROBOT_NAME
        if role == "assistant":
            self.workbuddy.voice_said()
        elif self.workbuddy.voice_reply(text, robot_name):
            return  # the voice model was told to say it passes the answer on
        self.chat.voice_turn(role, text, interrupted)
        if role == "user" and self.workbuddy.match_voice(text):
            try:
                self.workbuddy.submit(text, "voice", robot_name)
                self.interaction.play_sound("confirm", None, 60)
            except ApiError as exc:
                self.events.alert("warning", exc.message)

    def _workbuddy_speak(self, text: str) -> bool:
        """A PC task's question: the voice model asks it aloud, or a chime when no voice session runs."""
        if self.voice.prompt(text):
            return True
        try:
            self.interaction.play_sound("chime", None, 70)
        except ApiError:
            pass
        return False

    def _on_workbuddy_done(self, task: dict[str, Any]) -> None:
        ok = task["status"] == "done"
        summary = (task["result"] if ok else task["error"] or task["result"] or "")[:120]
        self.events.alert("info" if ok else "error", f"电脑任务{'完成' if ok else '没有完成'}：{summary}")
        if self.settings.section("workbuddy").get("announce", True) and task["status"] != "cancelled":
            self.interaction.play_sound("happy" if ok else "sad", None, 60)
        if task["origin"] == "voice" and self._voice_conversation and self.conversations.exists(self._voice_conversation):
            self.conversations.append(self._voice_conversation, "assistant", f"[电脑助手] {summary}")

    def snapshot(self) -> dict[str, Any]:
        with self._snapshot_lock:
            ts, data = self._snapshot
            if time.monotonic() - ts < 0.5:
                return data
            voice = self.voice.status()
            voice.pop("voices", None)
            data = {
                "action": self.motion.get_action(),
                "battery": self.motion.battery(),
                "input_source": self.motion.sources()["current"],
                "expression": self.interaction.current_expression(),
                "voice": voice,
                "music": self.music.status(),
                "robot_name": self.settings.section("robot").get("name") or ROBOT_NAME,
                "llm_ready": self.chat.provider() is not None,
                "mock": self.mock,
                "ble": self.ble.summary() if self.ble is not None else None,
            }
            self._snapshot = (time.monotonic(), data)
            return data

    def public_settings(self) -> dict[str, Any]:
        data = self.settings.get()
        data["llm"]["providers"] = [llm.public_provider(p) for p in data["llm"].get("providers", [])]
        key, origin = self.voice.api_key()
        data["voice"]["api_key"] = mask_secret(data["voice"].get("api_key", ""))
        data["voice"]["has_key"] = bool(key)
        data["voice"]["key_source"] = origin
        volc_key, volc_origin = self.voice.volc_key()
        data["voice"]["volc_api_key"] = mask_secret(data["voice"].get("volc_api_key", ""))
        data["voice"]["has_volc_key"] = bool(volc_key)
        data["voice"]["volc_key_source"] = volc_origin
        data["robot"]["default_persona"] = DEFAULT_PERSONA
        data["workbuddy"]["has_password"] = bool(data["workbuddy"].get("password"))
        data["workbuddy"]["password"] = mask_secret(data["workbuddy"].get("password", ""))
        return data

    def call(self, method: str, path: str, query: str = "", body: Any = None) -> tuple[int, dict[str, Any]]:
        """One API call outside HTTP (the app's Bluetooth link): HTTP status and CommonResponse."""
        route, params, known = self.match(method, path.rstrip("/") or "/")
        if route is None:
            err = ApiError("方法不支持" if known else f"没有接口 {path}", CommonState.INVALID, Reason.NOT_FOUND, 405 if known else 404)
            return err.http_status, error_response(err)
        if route.name in HTTP_ONLY:
            err = ApiError("这个接口传的是数据流或文件，请经 Wi-Fi（HTTP）调用", CommonState.INVALID, Reason.NOT_SUPPORTED)
            return err.http_status, error_response(err)
        try:
            result = route.handler(Request(params, parse_qs(query), body))
        except ApiError as exc:
            return exc.http_status, error_response(exc)
        except Exception as exc:  # noqa: BLE001
            log.error("BLE %s %s failed:\n%s", method, path, traceback.format_exc())
            return 500, error_response(ApiError(f"内部错误：{exc}", CommonState.FAILURE, Reason.HARDWARE, 500))
        if isinstance(result, dict) and "header" in result and "state" in result:
            return 200, result
        return 200, response(result)

    # ---- network and the phone app ---------------------------------------------------------

    def setup_mode(self) -> bool:
        """No Wi-Fi yet: a new phone may pair over Bluetooth without the on-screen code."""
        try:
            return self.wifi.available() and not self.wifi.status()["connected"]
        except ApiError:
            return False

    def ble_info(self) -> dict[str, Any]:
        """What the app reads before pairing (the INFO characteristic): enough to reach the HTTP API."""
        try:
            wifi = self.wifi.status()
        except ApiError:
            wifi = {"connected": False, "ip": None}
        ips = self.system.ips()
        return {
            "id": robot_id(),
            "name": self.settings.section("robot").get("name") or ROBOT_NAME,
            "ip": wifi.get("ip") or (ips[0] if ips else None),
            "ips": ips[:3],
            "port": self.port,
            "setup": self.wifi.available() and not wifi.get("connected"),
            "api": API_VERSION,
            "mock": self.mock,
        }

    def _show_pair_code(self, code: str | None) -> bool:
        if code is None:
            if self._pair_board:
                self._pair_board = False
                self.interaction.show_board({"kind": "clear"}, "ble")
            return True
        # The web UI is trusted (it controls everything anyway); never sent over Bluetooth.
        self.events.alert("info", f"有手机正在用蓝牙配对，配对码 {code}", ble=False)
        try:
            self._pair_board = self.backend.ensure_display() and self.interaction.show_board(
                {"kind": "list", "title": "蓝牙配对", "subtitle": "在小小白 App 里输入", "title_color": "blue",
                 "items": [{"text": code}], "seconds": 120}, "ble")
        except Exception:  # noqa: BLE001 - the code is in the web UI's alert too
            log.warning("pairing code not shown:\n%s", traceback.format_exc())
            self._pair_board = False
        return self._pair_board

    def wifi_connect(self, req: Request) -> Any:
        body = req.json()
        status = self.wifi.connect(str(body.get("ssid", "")), str(body.get("password") or ""), bool(body.get("hidden")))
        self.events.publish("wifi", connected=status["connected"], ssid=status["ssid"], ip=status["ip"])
        return response(status, message=f"已连上 {status['ssid']}，IP {status['ip'] or '获取中'}")

    def ble_status(self) -> dict[str, Any]:
        if self.ble is None:
            return {"enabled": False, "running": False, "error": "API 启动时关掉了蓝牙（--no-ble）", "sessions": [], "clients": []}
        return self.ble.status()

    def ble_revoke(self, client: str | None) -> Any:
        if self.ble is None:
            raise ApiError("蓝牙没有开启", CommonState.NOT_READY, Reason.NOT_CONFIGURED)
        count = self.ble.revoke(client)
        if client is not None and not count:
            raise ApiError("没有这台手机", CommonState.INVALID, Reason.NOT_FOUND, 404)
        return response({"revoked": count}, message="已取消配对")

    def _provider_from(self, body: dict[str, Any]) -> dict[str, Any]:
        existing = next((p for p in self.chat.providers() if p["id"] == body.get("id")), None)
        if existing is None and body.get("id") and not body.get("base_url"):
            raise ApiError("没有这个模型配置", CommonState.INVALID, Reason.NOT_FOUND, 404)
        return llm.normalize_provider(body, existing)

    # ---- handlers --------------------------------------------------------------------------

    def get_action(self, req: Request) -> Any:
        return self.motion.get_action()

    def set_action(self, req: Request) -> Any:
        body = req.json()
        return self.motion.set_action(body.get("action", body.get("value")), source=str(body.get("source") or "app_proxy"))

    def velocity(self, req: Request) -> Any:
        b = req.json()
        return self.motion.velocity(str(b.get("source") or "app_proxy"), b.get("vx", 0), b.get("vy", 0), b.get("vtheta", 0),
                                    b.get("head_yaw"))

    def walk(self, req: Request) -> Any:
        b = req.json()
        return self.motion.walk_for(
            float(b.get("vx", 0)), float(b.get("vy", 0)), float(b.get("vtheta", 0)), float(b.get("duration_s", 2)),
            source=str(b.get("source") or "app_proxy"),
        )

    def register_source(self, req: Request) -> Any:
        b = req.json()
        return self.motion.register_source(
            str(b.get("name", "")), int(b.get("priority", 40)), int(b.get("timeout_ms", 1000)), str(b.get("description", "")),
            bool(b.get("head_front", False)),
        )

    def task(self, req: Request) -> Any:
        try:
            task = self.tasks.get(int(req.params["task_id"]))
        except ValueError:
            task = None
        if task is None:
            raise ApiError("没有这个任务", CommonState.INVALID, Reason.NOT_FOUND, 404)
        return response(task, state=CommonState[task["state"]], message=task["message"], task_id=task["task_id"])

    def camera(self, req: Request) -> Any:
        port = self.backend.camera_port()
        esp32 = self.backend.esp32_info() or {}
        return {
            "available": port is not None,
            "stream_url": "/api/v1/hal/camera/stream" if port else None,
            "snapshot_url": "/api/v1/hal/camera/snapshot" if port else None,
            # The ESP32's own MJPEG stream (http://<ip>:81/stream), reachable without the Pi in the path.
            "esp32_ip": esp32.get("ip"),
            "note": None if port else ("模拟模式没有摄像头" if self.mock else "摄像头服务未运行（显示进程启动后自动开启）"),
        }

    def _camera_url(self, path: str) -> str:
        port = self.backend.camera_port()
        if port is None:
            raise ApiError("摄像头服务未运行", CommonState.NOT_READY, Reason.HARDWARE, 503)
        return f"http://127.0.0.1:{port}{path}"

    def camera_snapshot(self, req: Request) -> Any:
        try:
            with urllib.request.urlopen(self._camera_url("/snapshot.jpg"), timeout=8) as resp:
                return Raw(resp.read(), resp.headers.get("Content-Type", "image/jpeg"))
        except (urllib.error.URLError, TimeoutError) as exc:
            raise ApiError(f"取图失败：{exc}", CommonState.FAILURE, Reason.HARDWARE, 502) from None

    def camera_stream(self, req: Request) -> Any:
        url = self._camera_url("/stream")
        try:
            upstream = urllib.request.urlopen(url, timeout=10)
        except (urllib.error.URLError, TimeoutError) as exc:
            raise ApiError(f"连接摄像头失败：{exc}", CommonState.FAILURE, Reason.HARDWARE, 502) from None

        def write(h: Handler) -> None:
            with upstream:
                h.send_response(200)
                h.send_header("Content-Type", upstream.headers.get("Content-Type", "multipart/x-mixed-replace"))
                h.send_header("Cache-Control", "no-store")
                h.send_header("Connection", "close")
                h.end_headers()
                h.close_connection = True
                while not self.shutting_down.is_set():
                    chunk = upstream.read1(65536) if hasattr(upstream, "read1") else upstream.read(4096)
                    if not chunk:
                        break
                    h.wfile.write(chunk)

        return Stream(write)

    def logs(self, req: Request) -> Any:
        lines = max(10, min(2000, int(req.arg("lines", "300") or 300)))
        return {"name": req.params["name"], "text": self.system.log(req.params["name"], lines)}

    @staticmethod
    def _walk_recordings() -> list[Path]:
        try:
            files = [p for p in WALK_REC_DIR.iterdir() if WALK_REC_NAME.match(p.name)]
        except OSError:
            return []
        return sorted(files, key=lambda p: p.name.rsplit("_", 2)[-2:], reverse=True)

    def walk_rec_list(self, req: Request) -> Any:
        items = []
        for path in self._walk_recordings()[:WALK_REC_LIST_MAX]:
            try:
                stat = path.stat()
            except OSError:
                continue
            items.append({"name": path.name, "size": stat.st_size, "mtime": stat.st_mtime})
        return {"items": items, "dir": str(WALK_REC_DIR.relative_to(REPO))}

    def walk_rec_file(self, req: Request) -> Any:
        name = req.params["name"]
        path = WALK_REC_DIR / name
        if not WALK_REC_NAME.match(name) or not path.is_file():
            raise ApiError(f"没有录像 {name}", CommonState.INVALID, Reason.NOT_FOUND, 404)
        _background_thread()
        return Raw(path.read_bytes(), "application/octet-stream", filename=name)

    def walk_rec_bundle(self, req: Request) -> Any:
        """The .npz recordings are already compressed: stored as they are, they cost the Pi no CPU.
        Only the raw .part of a cut (or still running) session is deflated, at the fastest level."""
        try:
            count = max(1, min(WALK_REC_BUNDLE_MAX, int(req.arg("count", "5") or 5)))
        except ValueError:
            raise ApiError("count 应为整数", CommonState.INVALID) from None
        _background_thread()
        stamp = time.strftime("%Y%m%d_%H%M%S")
        state = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "walk_tuning": self.motion.walk_tuning(),
                 "action": self.motion.get_action(), "system": self.system.info()}
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for path in self._walk_recordings()[:count]:
                try:
                    if path.suffix == ".npz":
                        zf.write(path, f"walk_rec/{path.name}", compress_type=zipfile.ZIP_STORED)
                    else:
                        zf.write(path, f"walk_rec/{path.name}", compress_type=zipfile.ZIP_DEFLATED,
                                 compresslevel=1)
                except OSError:
                    continue
            zf.writestr("run.log", self.system.log("run", 5000), compress_type=zipfile.ZIP_DEFLATED)
            zf.writestr("state.json", json.dumps(state, ensure_ascii=False, indent=1, default=str),
                        compress_type=zipfile.ZIP_DEFLATED)
        return Raw(buf.getvalue(), "application/zip", filename=f"xiaoxiaobai_diag_{stamp}.zip")

    def power(self, req: Request) -> Any:
        action = req.json().get("action")
        if action not in ("shutdown", "reboot"):
            raise ApiError("action 只能是 shutdown 或 reboot", CommonState.INVALID)
        self.motion.emergency_stop(source="power")

        def later() -> None:
            time.sleep(1.5)
            out = self.backend.power(action)
            if out:
                self.events.alert("warning", f"{action}: {out}")

        threading.Thread(target=later, daemon=True).start()
        return response({"action": action}, state=CommonState.RUNNING, message="即将关机" if action == "shutdown" else "即将重启")

    def events_stream(self, req: Request) -> Any:
        def write(h: Handler) -> None:
            sub = self.events.subscribe()
            try:
                h.start_sse()
                h.sse({"type": "hello", "api_version": API_VERSION, "snapshot": self.snapshot(), "recent": self.events.recent({"alert"}, 20)})
                last = time.monotonic()
                while not self.shutting_down.is_set():
                    try:
                        h.sse(sub.get(timeout=1.0))
                    except queue.Empty:
                        pass
                    if time.monotonic() - last >= 1.0:
                        h.sse({"type": "status", **self.snapshot()})
                        last = time.monotonic()
            finally:
                self.events.unsubscribe(sub)

        return Stream(write)

    def chat_stream(self, req: Request) -> Any:
        body = req.json()
        gen = self.chat.stream(body.get("conversation_id"), str(body.get("message", "")), body.get("provider_id"))
        first = next(gen)

        def write(h: Handler) -> None:
            h.start_sse()
            try:
                h.sse(first)
                for event in gen:
                    h.sse(event)
            finally:
                gen.close()

        return Stream(write)

    def workbuddy_submit(self, req: Request) -> Any:
        body = req.json()
        task = self.workbuddy.submit(str(body.get("text", "")), "api", self.settings.section("robot").get("name") or ROBOT_NAME,
                                     reply_to=body.get("reply_to"))
        return response(task, state=CommonState.RUNNING, message="已交给电脑")

    def workbuddy_answer(self, req: Request) -> Any:
        body = req.json()
        task = self.workbuddy.answer(req.params["task_id"], body.get("answers"), str(body.get("text") or ""))
        q = task["question"]
        complete = q is None or all(item["id"] in q["answers"] for item in q["items"])
        return response(task, message="已回答" if complete else "已记下，还有问题没回答")

    def workbuddy_installer(self, req: Request) -> Any:
        """The PC gateway script with this robot's address and the pairing password already set,
        so the PC only runs `irm <url> | iex` (Windows) or `curl <url> | bash` (macOS / Linux)."""
        kind = req.params["kind"]
        if kind not in ("ps1", "sh"):
            raise ApiError("只有 install.ps1（Windows）和 install.sh（macOS / Linux）", CommonState.INVALID, Reason.NOT_FOUND, 404)
        if not re.fullmatch(r"[A-Za-z0-9.\-]+(:\d+)?|\[[0-9A-Fa-f:.]+\](:\d+)?", req.host):
            raise ApiError("请用机器人的局域网地址打开（例如 http://192.168.2.15:8000）", CommonState.INVALID)
        robot = f"http://{req.host}"
        password = self.workbuddy.pairing()["password"]
        auto = "1" if req.arg("auto") == "1" else "0"
        script = (WEBUI / "downloads" / f"workbuddy-gateway.{kind}").read_text(encoding="utf-8")
        if kind == "ps1":
            q = lambda s: "'" + s.replace("'", "''") + "'"  # noqa: E731
            head = f"$env:XXB_ROBOT={q(robot)}; $env:XXB_PASSWORD={q(password)}; $env:XXB_AUTO={q(auto)}\n"
        else:
            q = lambda s: "'" + s.replace("'", "'\\''") + "'"  # noqa: E731
            head = f"XXB_ROBOT={q(robot)}; XXB_PASSWORD={q(password)}; XXB_AUTO={q(auto)}\n"
        return Raw((head + script).encode("utf-8"), "text/plain; charset=utf-8")

    def voice_start(self, req: Request) -> Any:
        self._voice_conversation = None
        if self.music.active:
            self.music.stop(resume_voice=False)
        return self.voice.start(req.json().get("overrides") or None)

    def voice_preview(self, req: Request) -> Any:
        body = req.json()
        rate = body.get("speech_rate")
        return Raw(self.voice.preview(str(body.get("voice") or ""), body.get("text"), float(rate) if rate is not None else None), "audio/wav")

    def update_settings(self, req: Request) -> Any:
        section = req.params["section"]
        values = dict(req.json())
        if section == "robot":
            values = {k: str(v)[:4000] for k, v in values.items() if k in ("name", "persona")}
            if "name" in values:
                values["name"] = values["name"].strip()[:24] or ROBOT_NAME
        elif section == "llm":
            values = {k: v for k, v in values.items() if k not in ("providers", "active")}
            _num(values, "temperature", 0, 2)
            _num(values, "max_tokens", 64, 8192, int)
            _num(values, "history_turns", 0, 30, int)
            _num(values, "kb_top_k", 0, 10, int)
        elif section == "voice":
            for secret in ("api_key", "volc_api_key"):
                if is_masked(values.get(secret)):
                    values.pop(secret)
                elif secret in values:
                    values[secret] = str(values[secret] or "").strip()
            if "route" in values and values["route"] not in ("esp32", "pi"):
                raise ApiError("route 只能是 esp32 或 pi", CommonState.INVALID)
            if "voice" in values:
                values["voice"] = str(values["voice"] or "").strip()[:120] or "Mochi"
            if "volc_resource" in values:
                values["volc_resource"] = str(values["volc_resource"] or "").strip()[:60]
            for derived in ("has_key", "key_source", "has_volc_key", "volc_key_source"):
                values.pop(derived, None)
            _num(values, "volume", 0, 100, int)
            _num(values, "mic_gain", 0.5, 8)
            _num(values, "speech_rate", 0.5, 2.0)
        elif section == "safety":
            _num(values, "low_battery_v", 5.5, 8.0)
            _num(values, "warn_battery_v", 5.5, 8.4)
            merged = {**self.settings.section("safety"), **values}
            if merged["warn_battery_v"] <= merged["low_battery_v"]:
                raise ApiError("提醒电压需要高于保护电压", CommonState.INVALID)
        elif section == "display":
            values = {k: v for k, v in values.items() if k == "brightness"}
            _num(values, "brightness", 10, 100, int)
        elif section == "workbuddy":
            values = {k: v for k, v in values.items() if k in ("enabled", "url", "password", "voice_trigger", "triggers", "chat_tool", "announce", "timeout_s")}
            if is_masked(values.get("password")):
                values.pop("password")
            if "url" in values:
                values["url"] = normalize_url(str(values["url"]))
            if "triggers" in values:
                items = values["triggers"]
                if isinstance(items, str):
                    items = re.split(r"[,，、\n]", items)
                values["triggers"] = [str(t).strip()[:20] for t in items if str(t).strip()][:20]
            _num(values, "timeout_s", 30, 3600, int)
        elif section == "memo":
            values = {k: v for k, v in values.items() if k in ("enabled", "timezone", "show_on_screen", "board_seconds", "voice_capture", "remind")}
            if "timezone" in values:
                values["timezone"] = str(values["timezone"] or "").strip()
                if values["timezone"] and not memo.valid_timezone(values["timezone"]):
                    raise ApiError(f"不认识的时区 {values['timezone']}（例如 Asia/Shanghai）", CommonState.INVALID)
            _num(values, "board_seconds", 5, 120, int)
        elif section == "music":
            values = {k: v for k, v in values.items()
                      if k in ("enabled", "volume", "quality", "voice_control", "resume_voice", "show_on_screen", "groove", "groove_size",
                                "groove_offset_ms", "dance_music", "dance_volume")}
            if "quality" in values and values["quality"] not in QUALITIES:
                raise ApiError(f"quality 只能是 {' / '.join(QUALITIES)}", CommonState.INVALID)
            _num(values, "volume", 0, 100, int)
            _num(values, "dance_volume", 0, 100, int)
            _num(values, "groove_size", 0, 100, int)
            _num(values, "groove_offset_ms", -300, 300, int)
            for key in ("groove", "dance_music"):
                if key in values:
                    values[key] = bool(values[key])
        else:
            raise ApiError(f"没有设置分区 {section}", CommonState.INVALID, Reason.NOT_FOUND, 404)
        self.settings.update(section, values)
        if section == "display":
            self._apply_brightness()
        self.events.publish("settings", section=section)
        return response(self.public_settings()[section], message="已保存")

    def add_provider(self, req: Request) -> Any:
        provider = llm.normalize_provider(req.json())
        cfg = self.settings.section("llm")
        providers = [*cfg.get("providers", []), provider]
        self.settings.update("llm", {"providers": providers, "active": cfg.get("active") or provider["id"]})
        self.events.publish("settings", section="llm")
        return response(llm.public_provider(provider), message="已添加")

    def update_provider(self, req: Request) -> Any:
        cfg = self.settings.section("llm")
        providers = cfg.get("providers", [])
        for i, p in enumerate(providers):
            if p["id"] == req.params["provider_id"]:
                providers[i] = llm.normalize_provider(req.json(), p)
                self.settings.update("llm", {"providers": providers})
                self.events.publish("settings", section="llm")
                return response(llm.public_provider(providers[i]), message="已保存")
        raise ApiError("没有这个模型配置", CommonState.INVALID, Reason.NOT_FOUND, 404)

    def delete_provider(self, req: Request) -> Any:
        cfg = self.settings.section("llm")
        providers = [p for p in cfg.get("providers", []) if p["id"] != req.params["provider_id"]]
        if len(providers) == len(cfg.get("providers", [])):
            raise ApiError("没有这个模型配置", CommonState.INVALID, Reason.NOT_FOUND, 404)
        active = cfg.get("active")
        if active == req.params["provider_id"]:
            active = providers[0]["id"] if providers else ""
        self.settings.update("llm", {"providers": providers, "active": active})
        self.events.publish("settings", section="llm")
        return response(None, message="已删除")

    def set_active_provider(self, req: Request) -> Any:
        provider_id = req.json().get("id")
        if not any(p["id"] == provider_id for p in self.chat.providers()):
            raise ApiError("没有这个模型配置", CommonState.INVALID, Reason.NOT_FOUND, 404)
        self.settings.update("llm", {"active": provider_id})
        self.events.publish("settings", section="llm")
        return response({"active": provider_id}, message="已设为默认模型")

    def test_provider(self, req: Request) -> Any:
        return llm.test_provider(self._provider_from(req.json()))

    def provider_models(self, req: Request) -> Any:
        try:
            return {"models": llm.list_models(self._provider_from(req.json()))}
        except llm.LlmError as exc:
            raise ApiError(f"获取模型列表失败：{exc}", CommonState.FAILURE, Reason.UPSTREAM, 502) from None

    def memo_list(self, req: Request) -> Any:
        now = self.memos.now()
        items = self.memos.list(req.arg("start"), req.arg("end"), req.arg("q", "") or "", req.arg("status", "all") or "all",
                                notes=req.arg("notes", "1") != "0")
        return {"items": items, "today": now.date().isoformat(), "now": self.chat.now_text(),
                "timezone": self.settings.section("memo").get("timezone", "")}

    def memo_show(self, req: Request) -> Any:
        body = req.json()

        def day(key: str) -> date | None:
            value = str(body.get(key) or body.get("date") or "").strip()
            try:
                return date.fromisoformat(value) if value else None
            except ValueError:
                raise ApiError(f"{key} 要写成 YYYY-MM-DD", CommonState.INVALID) from None

        if body.get("clock"):
            shown = self.chat.show_clock("app_proxy")
        else:
            shown = self.chat.show_range(day("start"), day("end"), "app_proxy")
        if not shown:
            raise ApiError("没有显示：显示进程没在运行，或者设置里关了「在屏幕上显示」", CommonState.FAILURE, Reason.HARDWARE)
        return response({"shown": True}, message="已显示在屏幕上")

    def memo_export(self, req: Request) -> Any:
        body, content_type, filename = self.memos.export(req.arg("format", "csv") or "csv")
        return Raw(body, content_type, filename=filename)

    def memo_heard(self, req: Request) -> Any:
        text = str(req.json().get("text", "")).strip()
        query = memo.parse_query(text, self.memos.today())
        self.chat.memo_heard(text, "voice")
        return {
            "query": None if query is None else {"start": query[0], "end": query[1], "label": query[2]},
            "clock": memo.asks_clock(text),
            "capture": memo.wants_capture(text),
        }

    def heard(self, req: Request) -> Any:
        text = str(req.json().get("text", "")).strip()
        if not text:
            raise ApiError("text 不能为空", CommonState.INVALID)
        self.chat.voice_turn("user", text, False)
        return response({"text": text}, message="已当作语音处理")

    def api_list(self, req: Request) -> Any:
        return [
            {"method": r.method, "path": r.path, "name": r.name, "group": r.group, "summary": r.summary, "example": r.example,
             "ble": r.name not in HTTP_ONLY}
            for r in self.routes
        ]

    # ---- route table -----------------------------------------------------------------------

    def _routes(self) -> list[Route]:
        m, i, k, mem, conv, mu = self.motion, self.interaction, self.knowledge, self.memory, self.conversations, self.music
        ok = lambda data=None, message="": response(data, message=message)  # noqa: E731
        R = Route
        return [
            # motion control (mc), after the PrimeBot SDK
            R("GET", "/api/v1/mc/action", self.get_action, "GetMcAction", "运动控制", "当前运动模式、可切换目标和控制循环状态"),
            R("POST", "/api/v1/mc/action", self.set_action, "SetMcAction", "运动控制", "切换运动模式（异步，返回 task_id）",
              {"action": "BIPED_STAND_DEFAULT"}),
            R("GET", "/api/v1/mc/actions", lambda r: m.catalog(), "GetMcActionList", "运动控制", "全部运动模式和状态机"),
            R("POST", "/api/v1/mc/estop", lambda r: m.emergency_stop(str(r.json().get("source") or "app_proxy")),
              "EmergencyStop", "运动控制", "急停：停止控制循环并断开力矩（任何状态都可调用）"),
            R("POST", "/api/v1/mc/locomotion/velocity", self.velocity, "SetLocomotionVelocity", "运动控制",
              "走跑速度（-1..1 比例），需按输入源超时周期持续发送；head_front 输入源可带 head_yaw（度，+左）转头",
              {"source": "app_proxy", "vx": 0.4, "vy": 0, "vtheta": 0}),
            R("POST", "/api/v1/mc/walk", self.walk, "WalkFor", "运动控制", "以给定速度走一段时间后自动停下（异步）",
              {"vx": 0.5, "vy": 0, "vtheta": 0, "duration_s": 2}),
            R("GET", "/api/v1/mc/input_source", lambda r: m.sources(), "GetMcInputSource", "运动控制", "输入源列表和当前控制者"),
            R("POST", "/api/v1/mc/input_source", self.register_source, "SetMcInputSource", "运动控制",
              "注册自定义输入源（优先级 0–59；head_front 让头在它控制时保持朝前）",
              {"name": "my_app", "priority": 40, "timeout_ms": 1000, "head_front": False}),
            R("DELETE", "/api/v1/mc/input_source/{name}", lambda r: m.delete_source(r.params["name"]),
              "DeleteMcInputSource", "运动控制", "删除自定义输入源"),
            R("POST", "/api/v1/mc/lead", lambda r: m.set_lead(bool(r.json().get("enabled")), str(r.json().get("source") or "app_proxy")),
              "SetHandLead", "运动控制", "牵手走：走跑·上肢自由下手臂放软，拉手臂前后走、侧抬转弯，松手后手臂慢慢回原位并停下（站立 / 走跑中会自动切过去）",
              {"enabled": True}),
            R("POST", "/api/v1/mc/hug", lambda r: m.set_hug(bool(r.json().get("enabled")), str(r.json().get("source") or "app_proxy")),
              "SetHug", "运动控制", "抱住面前的腿不放（App 跟屁虫用）：双臂合拢夹紧，头左右蹭蹭再抬头看人，随机节奏和表情；"
              "关掉松手并回到原来的模式（站立 / 走跑中会自动切到走跑·上肢自由）", {"enabled": True}),
            R("POST", "/api/v1/mc/propup", lambda r: m.set_propup(bool(r.json().get("down")), str(r.json().get("source") or "app_proxy")),
              "SetPropUp", "运动控制", "撑起/躺下：仰躺时 down=false 撑起并保持，down=true 躺回去（不在该模式时先切过去再撑起）；"
              "也可以重复设置 BIPED_PROPUP 一键切换", {"down": False}),
            R("POST", "/api/v1/mc/head", lambda r: m.set_head(bool(r.json().get("enabled"))), "SetHeadMotion", "运动控制",
              "开关头部随动摆动", {"enabled": True}),
            R("GET", "/api/v1/mc/walk_tuning", lambda r: m.walk_tuning(), "GetWalkTuning", "运动控制",
              "走路 / 滑板策略（可选模型）和 IMU 安装校正"),
            R("POST", "/api/v1/mc/walk_tuning", lambda r: m.set_walk_tuning(r.json()), "SetWalkTuning", "运动控制",
              "切换走路 / 滑板策略、调 IMU 横滚/俯仰校正（±6°，走路中实时生效并保存）；mirror_policy=true 把策略左右镜像"
              "（诊断走歪是策略还是机器人本身，不保存，重启后关闭）；kp_vin_comp 开关电压补偿 KP（只在走动时按电压调 KP，"
              "让舵机一直像 7.0 V 时一样硬，站着不动不超过原 KP）；imu_auto_trim 开关 IMU 自动修正（稳态直行时横滚拉回"
              "左右直立、俯仰保持开机刚走时的姿态，需先做 IMU 安装标定，修不动自动停用，关掉即清零）；imu_mount_*_deg 为 IMU 安装偏差"
              "（±10°，一般由 imu_calibrate 测出）；reset=true 恢复默认（保留安装标定）",
              {"walk_agent": 5, "skate_agent": 0, "imu_roll_corr_deg": 0.0, "imu_pitch_corr_deg": 0.0, "mirror_policy": False,
               "kp_vin_comp": True, "imu_auto_trim": True}),
            R("POST", "/api/v1/mc/imu_calibrate", lambda r: m.imu_calibrate(), "ImuCalibrate", "运动控制",
              "IMU 安装标定：机器人在「位控站立」模式放在平地上别碰，约 3 秒，测出 IMU 相对躯干的横滚/俯仰安装偏差并保存，"
              "手动校正归零（换 IMU、拆装后做一次）", {}),
            R("POST", "/api/v1/mc/upper_body", lambda r: m.upper_body(r.json().get("targets_deg") or {}), "SetUpperBodyJoints",
              "运动控制", "上肢控制模式下设置 13–20 号舵机目标角（度）", {"targets_deg": {"13": 20, "17": -20}}),
            R("GET", "/api/v1/mc/gestures", lambda r: m.gesture_catalog(), "GetGestureList", "运动控制",
              "走跑·上肢自由模式的全部拟人动作和 13–20 号关节范围"),
            R("POST", "/api/v1/mc/gesture", lambda r: m.play_gesture(r.json().get("name", ""), r.json().get("size", 1.0),
                                                                      str(r.json().get("source") or "app_proxy")),
              "PlayGesture", "运动控制", "做一个动作（站立 / 走跑中会自动切到走跑·上肢自由）；name=random 随机",
              {"name": "wave", "size": 1.0}),
            R("POST", "/api/v1/mc/gesture/stop", lambda r: m.stop_gesture(), "StopGesture", "运动控制", "停止当前动作，回到自然姿态"),
            R("POST", "/api/v1/mc/upper", lambda r: m.upper_settings(r.json().get("auto"), r.json().get("manual_deg"),
                                                                     r.json().get("release")),
              "SetUpperBodyFree", "运动控制", "走跑·上肢自由：空闲小动作开关、手动保持某些关节角（关节角，度）、释放",
              {"auto": True, "manual_deg": {"19": 20, "20": -10}, "release": None}),
            # move editor: poses in the gestures' authoring space (rp/rr/re, lp/lr/le, yaw, tilt; degrees)
            R("GET", "/api/v1/mc/editor", lambda r: m.editor_info(), "GetMoveEditor", "动作编辑",
              "编辑器用的关节范围、缓动方式和已保存的自定义动作"),
            R("GET", "/api/v1/mc/editor/pose", lambda r: m.upper_pose(), "GetUpperPose", "动作编辑",
              "上肢当前姿态（示教时为实测角度）"),
            R("POST", "/api/v1/mc/editor/pose", lambda r: m.set_upper_pose(r.json().get("pose")), "SetUpperPose", "动作编辑",
              "把上肢摆到某个姿态（上肢控制 / 走跑·上肢自由）", {"pose": {"rp": 90, "re": 40, "yaw": 10}}),
            R("POST", "/api/v1/mc/editor/release", lambda r: m.release_pose(), "ReleaseUpperPose", "动作编辑",
              "走跑·上肢自由：松开编辑器保持的关节"),
            R("POST", "/api/v1/mc/editor/teach", lambda r: m.set_teach(bool(r.json().get("enabled"))), "SetTeachMode", "动作编辑",
              "上肢控制：放松 13–20 号电机用手掰姿势（enabled=false 锁定当前姿势）", {"enabled": True}),
            R("POST", "/api/v1/mc/editor/preview", lambda r: m.play_custom(None, r.json().get("keys"), "app_proxy"),
              "PreviewMove", "动作编辑", "按原样播放还没保存的关键帧",
              {"keys": [{"dur": 0.6, "pose": {"rp": 120, "re": 60}, "hold": 0.5, "ease": "io"}]}),
            R("GET", "/api/v1/mc/custom_gestures", lambda r: m.custom.list(), "GetCustomGestureList", "动作编辑", "全部自定义动作"),
            R("POST", "/api/v1/mc/custom_gestures", lambda r: m.save_custom(r.json()), "SaveCustomGesture", "动作编辑",
              "保存自定义动作（带 id 就是修改）；之后说出名字机器人就会做",
              {"name": "举手投降", "keys": [{"dur": 0.6, "pose": {"rp": 150, "lp": 150}, "hold": 1.0, "ease": "io"}]}),
            R("GET", "/api/v1/mc/custom_gestures/{move_id}", lambda r: m.custom.get(r.params["move_id"]), "GetCustomGesture",
              "动作编辑", "一个自定义动作的关键帧"),
            R("DELETE", "/api/v1/mc/custom_gestures/{move_id}", lambda r: m.delete_custom(r.params["move_id"]),
              "DeleteCustomGesture", "动作编辑", "删除自定义动作"),
            R("POST", "/api/v1/mc/custom_gestures/{move_id}/play", lambda r: m.play_custom(r.params["move_id"],
                                                                                       source=str(r.json().get("source") or "app_proxy")),
              "PlayCustomGesture", "动作编辑", "做一个自定义动作（上肢控制下直接做；站立 / 走跑下切到走跑·上肢自由做完再回来）", {}),
            # interaction
            R("GET", "/api/v1/interaction/emotions",
              lambda r: [{k: e[k] for k in ("name", "label", "index", "code")} for e in i.expressions()]
              if r.arg("brief") == "1" else i.expressions(),
              "GetEmotionList", "交互", "全部表情（含点阵预览；?brief=1 只要名字，蓝牙用）"),
            R("GET", "/api/v1/interaction/emotion", lambda r: {"name": i.current_expression()}, "GetEmotion", "交互", "当前表情"),
            R("POST", "/api/v1/interaction/emotion", lambda r: i.set_expression(str(r.json().get("name", ""))),
              "SetEmotion", "交互", "切换表情", {"name": "happy"}),
            R("GET", "/api/v1/interaction/emotion/{name}/frames", lambda r: i.expression_loop(r.params["name"]),
              "GetEmotionFrames", "交互", "表情动画的全部帧"),
            R("GET", "/api/v1/interaction/sounds", lambda r: i.sounds(), "GetSoundList", "交互", "内置提示音"),
            R("POST", "/api/v1/interaction/sound", lambda r: i.play_sound(r.json().get("name"), r.json().get("notes"), r.json().get("volume", 60)),
              "PlaySound", "交互", "播放提示音或自定义音符 [[Hz, ms], …]", {"name": "chime", "volume": 60}),
            R("GET", "/api/v1/interaction/voice", lambda r: self.voice.status(), "GetVoiceChatState", "交互", "实时语音对话状态"),
            R("POST", "/api/v1/interaction/voice/start", self.voice_start, "StartVoiceChat", "交互", "开始实时语音对话（千问 Omni）", {}),
            R("POST", "/api/v1/interaction/voice/stop", lambda r: self.voice.stop(), "StopVoiceChat", "交互", "结束实时语音对话"),
            R("POST", "/api/v1/interaction/voice/preview", self.voice_preview, "PreviewVoice", "交互",
              "用某个音色朗读一句话（试听），返回 24 kHz WAV", {"voice": "zh_female_vv_uranus_bigtts", "text": "你好"}),
            # music
            R("GET", "/api/v1/music", lambda r: mu.status(), "GetMusicState", "音乐", "播放状态：当前歌曲、进度、音量"),
            R("GET", "/api/v1/music/queue", lambda r: mu.queue(), "GetMusicQueue", "音乐", "播放列表"),
            R("GET", "/api/v1/music/search", lambda r: mu.search(r.arg("q", "") or "", int(r.arg("limit", "20") or 20)),
              "SearchMusic", "音乐", "在网易云搜歌（?q=晴天），带 playable：能不能播"),
            R("POST", "/api/v1/music/play", lambda r: mu.play(r.json(), "app_proxy"), "PlayMusic", "音乐",
              "播放：query 搜索并播放、playlist 为 hot / daily / liked、tracks+index 播一个列表、只给 index 跳到列表里那首",
              {"query": "晴天 周杰伦"}),
            R("POST", "/api/v1/music/pause", lambda r: mu.pause(), "PauseMusic", "音乐", "暂停"),
            R("POST", "/api/v1/music/resume", lambda r: mu.resume(), "ResumeMusic", "音乐", "继续播放"),
            R("POST", "/api/v1/music/stop", lambda r: mu.stop(), "StopMusic", "音乐", "停止（语音对话是因为音乐停的会恢复）"),
            R("POST", "/api/v1/music/next", lambda r: mu.jump(1), "NextMusic", "音乐", "下一首"),
            R("POST", "/api/v1/music/previous", lambda r: mu.jump(-1), "PreviousMusic", "音乐", "上一首"),
            R("POST", "/api/v1/music/volume", lambda r: mu.set_volume(r.json().get("volume", 60)), "SetMusicVolume", "音乐",
              "音乐音量 0–100", {"volume": 60}),
            R("GET", "/api/v1/music/account", lambda r: mu.account(r.arg("refresh") == "1"), "GetMusicAccount", "音乐",
              "网易云登录状态（昵称、头像、是否 VIP）"),
            R("POST", "/api/v1/music/login/qr", lambda r: mu.qr_start(), "StartMusicQrLogin", "音乐",
              "生成网易云扫码登录二维码（返回 key 和 SVG）", {}),
            R("GET", "/api/v1/music/login/qr/{key}", lambda r: mu.qr_check(r.params["key"]), "CheckMusicQrLogin", "音乐",
              "查询扫码状态：801 等待扫码、802 待确认、803 成功、800 过期；被风控拦住时带 verify（滑块或安全验证二维码）"),
            R("GET", "/api/v1/music/login/verify/{key}", lambda r: mu.verify_check(r.params["key"]), "CheckMusicVerify", "音乐",
              "查询网易云安全验证（-462）：0 等待扫码、10 已扫码、20 通过（自动重做被拦的那一步，结果在 retry）、21 过期"),
            R("POST", "/api/v1/music/login/verify", lambda r: mu.verify_slider(str(r.json().get("key", "")), str(r.json().get("validate", ""))),
              "SubmitMusicSlider", "音乐", "提交网页里拖完的网易云滑块验证（-462），通过后自动重做被拦的那一步，结果在 retry",
              {"key": "00.40....", "validate": "CN31_..."}),
            R("POST", "/api/v1/music/login/sms/send", lambda r: mu.sms_send(str(r.json().get("phone", "")), str(r.json().get("country") or "86")),
              "SendMusicSmsCode", "音乐", "发送网易云登录验证码", {"phone": "13800000000"}),
            R("POST", "/api/v1/music/login/sms", lambda r: mu.sms_login(str(r.json().get("phone", "")), str(r.json().get("code", "")),
                                                                         str(r.json().get("country") or "86")),
              "MusicSmsLogin", "音乐", "用手机验证码登录网易云", {"phone": "13800000000", "code": "1234"}),
            R("POST", "/api/v1/music/login/cookie", lambda r: mu.cookie_login(str(r.json().get("cookie", ""))), "MusicCookieLogin",
              "音乐", "用浏览器里复制的 Cookie（或 MUSIC_U）登录网易云", {"cookie": "MUSIC_U=..."}),
            R("POST", "/api/v1/music/logout", lambda r: mu.logout(), "MusicLogout", "音乐", "退出网易云账号"),
            # hardware
            R("GET", "/api/v1/hal/bms/state", lambda r: m.battery(), "GetBmsState", "硬件", "电池电压和估算电量"),
            R("GET", "/api/v1/hal/joint/state", lambda r: m.joint_state(), "GetJointState", "硬件",
              "全部关节位置 / 目标 / 速度（rad），控制循环运行时 5 Hz 更新"),
            R("GET", "/api/v1/hal/imu/state", lambda r: m.imu_state(), "GetImuState", "硬件", "IMU 姿态角、角速度和四元数"),
            R("GET", "/api/v1/hal/camera", self.camera, "GetCameraInfo", "硬件", "摄像头是否可用"),
            R("GET", "/api/v1/hal/camera/snapshot", self.camera_snapshot, "GetCameraSnapshot", "硬件", "抓一张 JPEG"),
            R("GET", "/api/v1/hal/camera/stream", self.camera_stream, "GetCameraStream", "硬件", "MJPEG 视频流"),
            # system
            R("GET", "/api/v1/system/state", lambda r: self.snapshot(), "GetRobotState", "系统", "一次取全：模式、电量、表情、语音"),
            R("GET", "/api/v1/system/info", lambda r: self.system.info(), "GetSystemInfo", "系统", "主机、网络、温度、版本"),
            R("GET", "/api/v1/system/logs/{name}", self.logs, "GetLog", "系统", "日志尾部：run / display / voice / music / upper_body_slider / api"),
            R("GET", "/api/v1/system/walk_rec", self.walk_rec_list, "ListWalkRecordings", "系统",
              "机器人上的走路 / 做动作录像（每次走路自动记录的逐帧数据，最新在前）"),
            R("GET", "/api/v1/system/walk_rec/file/{name}", self.walk_rec_file, "GetWalkRecording", "系统",
              "下载一段录像（.npz；.part 是断电或崩溃中断、或正在录的一段）"),
            R("GET", "/api/v1/system/walk_rec/bundle", self.walk_rec_bundle, "GetWalkDiagnostics", "系统",
              f"诊断包（.zip）：最近 count 段录像（默认 5，最多 {WALK_REC_BUNDLE_MAX}）、控制循环日志、当前调校和状态"),
            R("POST", "/api/v1/system/power", self.power, "SetPower", "系统", "关机或重启树莓派", {"action": "reboot"}),
            # network: the Pi's Wi-Fi and the phone app's Bluetooth link
            R("GET", "/api/v1/system/wifi", lambda r: self.wifi.status(), "GetWifiState", "网络",
              "机器人的 Wi-Fi：是否连上、SSID、IP、已保存的网络"),
            R("GET", "/api/v1/system/wifi/scan", lambda r: self.wifi.scan(), "ScanWifi", "网络",
              "扫描附近的 Wi-Fi（主控只支持 2.4 GHz）"),
            R("POST", "/api/v1/system/wifi/connect", self.wifi_connect, "ConnectWifi", "网络",
              "连接并保存一个 Wi-Fi，拿到 IP 才返回；经 Wi-Fi 调用时连接会断开，App 首次配网走蓝牙",
              {"ssid": "Home-2.4G", "password": "12345678"}),
            R("DELETE", "/api/v1/system/wifi/{uuid}", lambda r: self.wifi.forget(r.params["uuid"]), "ForgetWifi", "网络",
              "删除一个已保存的网络"),
            R("GET", "/api/v1/system/ble", lambda r: self.ble_status(), "GetBleState", "网络",
              "手机 App 的蓝牙：是否在广播、连着的手机、配对过的手机"),
            R("DELETE", "/api/v1/system/ble/clients/{client_id}", lambda r: self.ble_revoke(r.params["client_id"]),
              "RevokeBleClient", "网络", "取消一台手机的配对（下次要重新输入屏幕上的配对码）"),
            R("DELETE", "/api/v1/system/ble/clients", lambda r: self.ble_revoke(None), "RevokeAllBleClients", "网络",
              "取消全部手机的配对"),
            R("GET", "/api/v1/tasks", lambda r: self.tasks.recent(50), "GetTaskList", "系统", "最近的异步任务"),
            R("GET", "/api/v1/tasks/{task_id}", self.task, "GetTask", "系统", "查询异步任务结果"),
            R("GET", "/api/v1/events", self.events_stream, "SubscribeEvents", "系统", "事件流（SSE）：状态、告警、语音、表情"),
            # assistant
            R("POST", "/api/v1/chat", self.chat_stream, "Chat", "AI 助手", "文字对话（SSE 流式，含工具调用）",
              {"message": "你好，站起来吧", "conversation_id": None}),
            R("GET", "/api/v1/conversations", lambda r: conv.list(r.arg("channel", "") or ""), "GetConversationList", "AI 助手", "会话列表（chat / voice）"),
            R("POST", "/api/v1/conversations", lambda r: conv.create("chat", str(r.json().get("title", ""))), "CreateConversation", "AI 助手", "新建会话", {}),
            R("GET", "/api/v1/conversations/{conv_id}", lambda r: conv.get(r.params["conv_id"]), "GetConversation", "AI 助手", "会话全部消息"),
            R("PATCH", "/api/v1/conversations/{conv_id}", lambda r: conv.rename(r.params["conv_id"], str(r.json().get("title", ""))),
              "RenameConversation", "AI 助手", "重命名会话", {"title": "新的标题"}),
            R("DELETE", "/api/v1/conversations/{conv_id}", lambda r: ok(conv.delete(r.params["conv_id"]), "已删除"),
              "DeleteConversation", "AI 助手", "删除会话"),
            R("GET", "/api/v1/memory", lambda r: mem.list(r.arg("q", "") or ""), "GetMemoryList", "AI 助手", "长期记忆"),
            R("POST", "/api/v1/memory", lambda r: mem.add(str(r.json().get("text", "")), "manual", r.json().get("tags"), bool(r.json().get("pinned"))),
              "AddMemory", "AI 助手", "添加记忆", {"text": "主人叫小明，喜欢恐龙", "pinned": True}),
            R("PATCH", "/api/v1/memory/{memory_id}", lambda r: mem.update(r.params["memory_id"], r.json()), "UpdateMemory", "AI 助手", "修改记忆"),
            R("DELETE", "/api/v1/memory/{memory_id}", lambda r: ok(mem.delete(r.params["memory_id"]), "已删除"), "DeleteMemory", "AI 助手", "删除记忆"),
            R("DELETE", "/api/v1/memory", lambda r: ok({"deleted": mem.clear()}, "已清空"), "ClearMemory", "AI 助手", "清空全部记忆"),
            R("GET", "/api/v1/knowledge", lambda r: {"documents": k.list(), "stats": k.stats()}, "GetKnowledgeList", "AI 助手", "知识库文档"),
            R("POST", "/api/v1/knowledge", lambda r: k.add(str(r.json().get("title", "")), str(r.json().get("text", "")), str(r.json().get("source", "manual"))),
              "AddKnowledge", "AI 助手", "添加文档（纯文本 / Markdown）", {"title": "机器人说明书", "text": "小小白有 19 个舵机……"}),
            R("GET", "/api/v1/knowledge/search", lambda r: k.search(r.arg("q", "") or "", int(r.arg("k", "5") or 5)),
              "SearchKnowledge", "AI 助手", "检索知识库（?q=…&k=5）"),
            R("GET", "/api/v1/knowledge/{doc_id}", lambda r: k.get(r.params["doc_id"]), "GetKnowledge", "AI 助手", "文档全文"),
            R("PATCH", "/api/v1/knowledge/{doc_id}", lambda r: k.update(r.params["doc_id"], r.json()), "UpdateKnowledge", "AI 助手", "修改文档 / 启用停用"),
            R("DELETE", "/api/v1/knowledge/{doc_id}", lambda r: ok(k.delete(r.params["doc_id"]), "已删除"), "DeleteKnowledge", "AI 助手", "删除文档"),
            # calendar and memos
            R("GET", "/api/v1/memo", self.memo_list, "GetMemoList", "备忘录",
              "备忘列表（?start=YYYY-MM-DD&end=…&q=关键词&status=all|open|done&notes=0 不要随手记）"),
            R("POST", "/api/v1/memo", lambda r: self.memos.add(str(r.json().get("text", "")), r.json().get("date"), r.json().get("time"), "manual"),
              "AddMemo", "备忘录", "添加备忘（date、time 可省略：没有日期就是随手记）", {"text": "和小王开会", "date": "2026-11-08", "time": "15:00"}),
            R("PATCH", "/api/v1/memo/{memo_id}", lambda r: self.memos.update(r.params["memo_id"], r.json()), "UpdateMemo", "备忘录",
              "修改备忘 / 标记完成", {"done": True}),
            R("DELETE", "/api/v1/memo/{memo_id}", lambda r: ok(self.memos.delete(r.params["memo_id"]), "已删除"), "DeleteMemo", "备忘录", "删除备忘"),
            R("DELETE", "/api/v1/memo", lambda r: ok({"deleted": self.memos.clear_done()}, "已清除"), "ClearDoneMemos", "备忘录", "清除全部已完成的备忘"),
            R("POST", "/api/v1/memo/show", self.memo_show, "ShowMemoBoard", "备忘录", "在脸上的点阵屏显示某天（或一段日子）的清单，clock=true 显示时钟",
              {"date": "2026-11-08"}),
            R("POST", "/api/v1/memo/hide", lambda r: ok({"hidden": i.show_board({"kind": "clear"})}, "已收起"), "HideMemoBoard", "备忘录",
              "收起屏幕上的清单，回到表情"),
            R("GET", "/api/v1/memo/export", self.memo_export, "ExportMemos", "备忘录", "导出全部备忘（?format=csv|md|json|ics）"),
            R("POST", "/api/v1/interaction/heard", self.heard, "SimulateSpeech", "交互",
              "当作用户对机器人说了一句话（测试用）：备忘、动作（含自定义动作）、点歌都会照常响应", {"text": "做一下螃蟹舞"}),
            R("POST", "/api/v1/memo/heard", self.memo_heard, "SimulateMemoSpeech", "备忘录",
              "当作用户对机器人说了一句话（测试用）：问安排就上屏，记事就记下", {"text": "看一下今天要做的事情"}),
            # computer assistant (WorkBuddy on a PC)
            R("GET", "/api/v1/workbuddy", lambda r: self.workbuddy.status(), "GetWorkBuddyState", "电脑助手",
              "WorkBuddy 连接状态（会实际连一次电脑）"),
            R("GET", "/api/v1/workbuddy/tasks", lambda r: self.workbuddy.tasks(), "GetWorkBuddyTaskList", "电脑助手", "最近交给电脑的任务"),
            R("POST", "/api/v1/workbuddy/tasks", self.workbuddy_submit, "SubmitWorkBuddyTask", "电脑助手",
              "让电脑上的 WorkBuddy 执行一个任务（异步）", {"text": "打开记事本，写一句你好"}),
            R("GET", "/api/v1/workbuddy/tasks/{task_id}", lambda r: self.workbuddy.get(r.params["task_id"]), "GetWorkBuddyTask", "电脑助手",
              "任务进度和结果（含原始事件，便于排查）"),
            R("POST", "/api/v1/workbuddy/tasks/{task_id}/cancel", lambda r: self.workbuddy.cancel(r.params["task_id"]),
              "CancelWorkBuddyTask", "电脑助手", "取消任务"),
            R("POST", "/api/v1/workbuddy/tasks/{task_id}/answer", self.workbuddy_answer, "AnswerWorkBuddyQuestion", "电脑助手",
              "回答电脑上的智能体提出的问题（任务的 question）：answers 按问题 id 给选项文字，或 text 回答下一个问题",
              {"answers": {"q_0": "蓝色"}}),
            R("POST", "/api/v1/workbuddy/pairing", lambda r: self.workbuddy.pairing(), "GetWorkBuddyPairing", "电脑助手",
              "电脑端安装命令要带的网关密码（第一次调用时生成）"),
            R("GET", "/api/v1/workbuddy/install.{kind}", self.workbuddy_installer, "GetWorkBuddyInstaller", "电脑助手",
              "电脑端安装脚本，已带机器人地址和密码（?auto=1 免批准）：Windows 用 install.ps1，macOS / Linux 用 install.sh"),
            R("POST", "/api/v1/workbuddy/register", lambda r: self.workbuddy.register(r.client, r.json()), "RegisterWorkBuddy", "电脑助手",
              "电脑上的网关报到：机器人记下来源 IP，并回连测试", {"port": 8321, "password": "…", "name": "我的电脑"}),
            # settings
            R("GET", "/api/v1/settings", lambda r: self.public_settings(), "GetSettings", "设置", "全部设置（密钥已打码）"),
            R("PATCH", "/api/v1/settings/{section}", self.update_settings, "UpdateSettings", "设置", "修改 robot / llm / voice / safety / workbuddy / memo",
              {"temperature": 0.7}),
            R("GET", "/api/v1/llm/presets", lambda r: llm.PRESETS, "GetLlmPresets", "设置", "常见大模型服务商预设"),
            R("POST", "/api/v1/llm/providers", self.add_provider, "AddLlmProvider", "设置", "添加大模型配置",
              {"preset": "deepseek", "api_key": "sk-…", "model": "deepseek-chat"}),
            R("PATCH", "/api/v1/llm/providers/{provider_id}", self.update_provider, "UpdateLlmProvider", "设置", "修改大模型配置"),
            R("DELETE", "/api/v1/llm/providers/{provider_id}", self.delete_provider, "DeleteLlmProvider", "设置", "删除大模型配置"),
            R("POST", "/api/v1/llm/active", self.set_active_provider, "SetActiveLlm", "设置", "设为默认模型", {"id": "llm_xxxxxxxx"}),
            R("POST", "/api/v1/llm/test", self.test_provider, "TestLlm", "设置", "测试连通性（可传未保存的配置）",
              {"preset": "ollama", "model": "qwen2.5:3b"}),
            R("POST", "/api/v1/llm/models", self.provider_models, "ListLlmModels", "设置", "从服务商拉取可用模型列表", {"id": "llm_xxxxxxxx"}),
            # meta
            R("GET", "/api/v1/meta/apis", self.api_list, "GetApiList", "系统", "本页列出的全部接口"),
            R("GET", "/api/v1/meta/ping", lambda r: {"pong": True, "api_version": API_VERSION, "mock": self.mock, "id": robot_id(),
                                                     "name": self.settings.section("robot").get("name") or ROBOT_NAME},
              "Ping", "系统", "连通性检查（含机器人 id 和名字，App 用来认出同一台机器人）"),
        ]

    def match(self, method: str, path: str) -> tuple[Route | None, dict[str, str], bool]:
        path_known = False
        for route in self.routes:
            found = route.regex.match(path)
            if found:
                path_known = True
                if route.method == method:
                    return route, {k: unquote(v) for k, v in found.groupdict().items()}, True
        return None, {}, path_known


class StaticFiles:
    COMPRESS = {".js", ".css", ".html", ".json", ".svg", ".txt", ".md", ".ps1", ".sh"}

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self._cache: dict[Path, tuple[float, bytes, bytes | None]] = {}
        self._lock = threading.Lock()

    def get(self, url_path: str) -> tuple[Path, bytes, bytes | None] | None:
        rel = unquote(url_path).lstrip("/") or "index.html"
        path = (self.root / rel).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            return None
        mtime = path.stat().st_mtime
        with self._lock:
            cached = self._cache.get(path)
            if cached and cached[0] == mtime:
                return path, cached[1], cached[2]
        raw = path.read_bytes()
        gz = gzip.compress(raw, 6) if path.suffix in self.COMPRESS and len(raw) > 1024 else None
        with self._lock:
            self._cache[path] = (mtime, raw, gz)
        return path, raw, gz


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = f"XiaoxiaobaiAPI/{API_VERSION}"
    app: App
    static: StaticFiles

    def log_request(self, code: Any = "-", size: Any = "-") -> None:
        if isinstance(code, int) and code >= 400:
            log.info("%s %s -> %s", self.command, self.path, code)

    def log_message(self, fmt: str, *args: Any) -> None:
        log.debug(fmt, *args)

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_HEAD(self) -> None:
        self._dispatch("HEAD")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def do_PATCH(self) -> None:
        self._dispatch("PATCH")

    def do_DELETE(self) -> None:
        self._dispatch("DELETE")

    # ---- output ----------------------------------------------------------------------------

    def _send(self, status: int, body: bytes, content_type: str, *, gz: bytes | None = None, cache: str = "no-store",
              filename: str = "", etag: str = "") -> None:
        use_gz = gz is not None and "gzip" in (self.headers.get("Accept-Encoding") or "")
        payload = gz if use_gz else body
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", cache)
        if etag:
            self.send_header("ETag", etag)
        if use_gz:
            self.send_header("Content-Encoding", "gzip")
        if gz is not None:
            self.send_header("Vary", "Accept-Encoding")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(payload)

    def send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        gz = gzip.compress(body, 5) if len(body) > 2048 else None
        self._send(status, body, "application/json; charset=utf-8", gz=gz)

    def start_sse(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

    def sse(self, event: dict[str, Any]) -> None:
        self.wfile.write(f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n".encode("utf-8"))
        self.wfile.flush()

    # ---- dispatch --------------------------------------------------------------------------

    def _dispatch(self, method: str) -> None:
        url = urlsplit(self.path)
        if not url.path.startswith("/api/"):
            self._static(url.path)
            return
        route, params, known = self.app.match("GET" if method == "HEAD" else method, url.path.rstrip("/") or "/")
        if route is None:
            err = ApiError("方法不支持" if known else f"没有接口 {url.path}", CommonState.INVALID, Reason.NOT_FOUND, 405 if known else 404)
            self._drain()
            self.send_json(error_response(err), err.http_status)
            return
        try:
            body = self._read_body(method)
            result = route.handler(Request(params, parse_qs(url.query), body, self.client_address[0], self.headers.get("Host") or ""))
        except ApiError as exc:
            self.send_json(error_response(exc), exc.http_status)
            return
        except Exception as exc:  # noqa: BLE001
            log.error("%s %s failed:\n%s", method, url.path, traceback.format_exc())
            self.send_json(error_response(ApiError(f"内部错误：{exc}", CommonState.FAILURE, Reason.HARDWARE, 500)), 500)
            return
        try:
            if isinstance(result, Stream):
                result.write(self)
            elif isinstance(result, Raw):
                self._send(result.status, result.body, result.content_type, filename=result.filename)
            elif isinstance(result, dict) and "header" in result and "state" in result:
                self.send_json(result)
            else:
                self.send_json(response(result))
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            self.close_connection = True

    def _drain(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        if 0 < length <= MAX_BODY:
            self.rfile.read(length)

    def _read_body(self, method: str) -> Any:
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            self.close_connection = True
            raise ApiError(f"请求体超过 {MAX_BODY // 1_000_000} MB", CommonState.INVALID, http_status=413)
        raw = self.rfile.read(length) if length else b""
        if method == "POST":
            # Only JSON posts: a plain HTML form on another site cannot drive the robot.
            content_type = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if content_type != "application/json":
                raise ApiError("POST 需要 Content-Type: application/json", CommonState.INVALID, http_status=415)
        if not raw:
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ApiError(f"JSON 解析失败：{exc}", CommonState.INVALID) from None

    def _static(self, path: str) -> None:
        if self.command not in ("GET", "HEAD"):
            self._drain()
            self.send_json(error_response(ApiError("方法不支持", CommonState.INVALID, http_status=405)), 405)
            return
        found = self.static.get(path)
        if found is None and "." not in path.rsplit("/", 1)[-1]:
            # /pc is pc.html (typed by hand on the PC); other paths are routes of the web app.
            found = self.static.get(path.rstrip("/") + ".html") or self.static.get("/index.html")
        if found is None:
            self._send(404, "404 Not Found".encode(), "text/plain; charset=utf-8")
            return
        file, raw, gz = found
        content_type = mimetypes.guess_type(file.name)[0] or "application/octet-stream"
        if file.suffix in (".ps1", ".sh"):
            # Piped into PowerShell / bash on the PC; text with a charset keeps the Chinese readable.
            content_type = "text/plain"
        if content_type.startswith("text/") or content_type in ("application/javascript", "application/json"):
            content_type += "; charset=utf-8"
        # Revalidate every load: `make sync` swaps files under a running server and the app's
        # WebView must not keep an old module that no longer matches the API.
        st = file.stat()
        etag = f'W/"{st.st_mtime_ns:x}-{st.st_size:x}"'
        if etag in (self.headers.get("If-None-Match") or ""):
            self.send_response(304)
            self.send_header("ETag", etag)
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            return
        self._send(200, raw, content_type, gz=gz, cache="no-cache", etag=etag)


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def setup_logging(path: Path) -> None:
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    root = logging.getLogger("xiaoxiaobai")
    root.setLevel(logging.INFO)
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    root.addHandler(stream)
    try:
        file = logging.FileHandler(path, encoding="utf-8")
        file.setFormatter(fmt)
        root.addHandler(file)
    except OSError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Xiaoxiaobai robot API + web UI")
    parser.add_argument("--mock", action="store_true", help="simulate the robot (no motors, ESP32 or camera)")
    parser.add_argument("--host", default=os.environ.get("XIAOXIAOBAI_API_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("XIAOXIAOBAI_API_PORT", DEFAULT_PORT)))
    parser.add_argument("--display", action="store_true", help="start the face display right away")
    parser.add_argument("--ble", action=argparse.BooleanOptionalAction, default=None,
                        help="Bluetooth LE for the phone app (default: on, off with --mock; env XIAOXIAOBAI_BLE=0/1)")
    parser.add_argument("--stop-on-exit", action="store_true",
                        help="on exit, stop the control loop (torque off) and the face display this service started "
                             "(`make run`'s Ctrl-C)")
    args = parser.parse_args()

    mimetypes.add_type("application/javascript", ".js")
    mimetypes.add_type("text/css", ".css")
    mimetypes.add_type("application/octet-stream", ".bin")

    ble = args.ble
    if ble is None:
        env = os.environ.get("XIAOXIAOBAI_BLE", "")
        ble = env == "1" if env in ("0", "1") else not args.mock
    app = App(mock=args.mock, ble=ble, port=args.port)
    app.backend.paths.log("api").write_text("", encoding="utf-8")
    setup_logging(app.backend.paths.log("api"))
    Handler.app = app
    Handler.static = StaticFiles(WEBUI)
    server = Server((args.host, args.port), Handler)
    app.start()

    def stop(*_: Any) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    if args.stop_on_exit:
        signal.signal(signal.SIGHUP, stop)  # the SSH session of `make run` dropped
    ips = app.system.info()["ips"] or ["127.0.0.1"]
    log.info("小小白 API %s%s", API_VERSION, "（模拟模式）" if args.mock else "")
    for ip in ips:
        log.info("  web UI: http://%s:%d/", ip, args.port)
    if args.display:
        threading.Thread(target=app.backend.ensure_display, name="xiaoxiaobai-api-display", daemon=True).start()
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        if args.stop_on_exit:
            stop_robot(app)
        app.close()
        if args.stop_on_exit:
            app.backend.stop_display()
        server.server_close()
        log.info("stopped")


def stop_robot(app: App, timeout_s: float = 8.0) -> None:
    """Torque off before exiting; the loop disables torque itself on the stop flag."""
    if app.backend.loop_pid() is None and app.backend.upper_pid() is None:
        return
    log.info("stopping the control loop (torque off)…")
    app.motion.emergency_stop(source="exit")
    deadline = time.time() + timeout_s
    while time.time() < deadline and (app.backend.loop_pid() or app.backend.upper_pid()):
        time.sleep(0.1)


if __name__ == "__main__":
    main()
