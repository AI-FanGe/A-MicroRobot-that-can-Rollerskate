# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

"""Face expressions, speaker tones and the realtime voice session (voice_chat.py)."""

from __future__ import annotations

import asyncio
import codecs
import io
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path
from typing import Any, Callable

from voice_catalog import VOLC_RESOURCES, catalog, engine_of, is_volc, label_of

from api import API_VERSION
from api.backend import REPO, Backend, tail
from api.events import EventBus
from api.protocol import ApiError, CommonState, Reason, response
from api.store import Settings, qwen_api_key

SOUNDS: dict[str, tuple[list[tuple[int, int]], str]] = {
    "chime": ([(523, 160), (659, 160), (784, 160), (1047, 320)], "上行和弦（扬声器测试）"),
    "beep": ([(880, 120)], "短提示音"),
    "confirm": ([(660, 90), (990, 140)], "确认"),
    "alert": ([(988, 140), (0, 60), (988, 140), (0, 60), (988, 220)], "告警"),
    "happy": ([(784, 110), (988, 110), (1175, 110), (1568, 260)], "开心"),
    "sad": ([(523, 220), (466, 220), (392, 420)], "难过"),
}

PREVIEW_TEXT = "你好呀，我是小小白，以后就用这个声音和你聊天啦。"
PREVIEW_CACHE = 24


class Interaction:
    def __init__(self, backend: Backend, events: EventBus) -> None:
        self.backend = backend
        self.events = events
        self._board_until = 0.0  # time.monotonic() when the board on the screen goes away by itself
        self._catalog: list[dict[str, Any]] | None = None
        self._loops: dict[str, dict[str, Any]] = {}

    def expressions(self) -> list[dict[str, Any]]:
        if self._catalog is None:
            from expressions import catalog

            self._catalog = catalog(8)
        return self._catalog

    def expression_names(self) -> list[str]:
        from expressions import EXPRESSIONS

        return [item.name for item in EXPRESSIONS]

    def expression_loop(self, name: str) -> dict[str, Any]:
        if name not in self._loops:
            from expressions import loop_frames

            if name not in self.expression_names():
                raise ApiError(f"没有表情 {name}", CommonState.INVALID, Reason.NOT_FOUND, 404)
            self._loops[name] = loop_frames(name)
        return self._loops[name]

    def current_expression(self) -> str:
        from expressions import read_expression_name

        return read_expression_name(self.backend.paths.expression)

    def set_expression(self, name: str, source: str = "app_proxy") -> dict[str, Any]:
        from expressions import EXPRESSION_BY_NAME, write_expression_name

        if name not in EXPRESSION_BY_NAME:
            raise ApiError(f"没有表情 {name}", CommonState.INVALID, Reason.NOT_FOUND)
        write_expression_name(name, self.backend.paths.expression)
        item = EXPRESSION_BY_NAME[name]
        self.events.publish("expression", name=name, label=item.label, source=source)
        return response({"name": name, "label": item.label, "display_running": self.backend.display_running()})

    def show_board(self, spec: dict[str, Any], source: str = "app_proxy") -> bool:
        """Text on the face screen for a while (board.py); False when the display is not running."""
        from board import request_board

        if not self.backend.display_running():
            return False
        request_board(spec, self.backend.paths.board)
        if spec.get("kind") == "clear":
            self._board_until = 0.0
        else:
            self._board_until = time.monotonic() + float(spec.get("seconds") or 30)
            self.events.publish("board", board=spec.get("kind", "list"), title=spec.get("title", ""), source=source)
        return True

    def board_up(self) -> bool:
        return time.monotonic() < self._board_until

    def sounds(self) -> list[dict[str, Any]]:
        return [{"name": k, "label": v[1], "notes": v[0]} for k, v in SOUNDS.items()]

    def play_sound(self, name: str | None, notes: list | None, volume: int) -> dict[str, Any]:
        from esp32_link import MAX_NOTES, request_sound

        if notes:
            try:
                parsed = [(int(f), int(ms)) for f, ms in notes][:MAX_NOTES]
            except (TypeError, ValueError):
                raise ApiError("notes 格式为 [[频率Hz, 毫秒], ...]", CommonState.INVALID) from None
        elif name in SOUNDS:
            parsed = SOUNDS[name][0]
        else:
            raise ApiError(f"没有提示音 {name}", CommonState.INVALID, Reason.NOT_FOUND)
        if not self.backend.ensure_display():
            raise ApiError("显示/音频进程没有运行，无法播放", CommonState.FAILURE, Reason.HARDWARE)
        volume = max(0, min(100, int(volume)))
        request_sound(parsed, volume, self.backend.paths.sound)
        return response({"notes": parsed, "volume": volume})


class VoiceSession:
    """Runs voice_chat.py (Qwen-Omni realtime) and turns its console output into events."""

    def __init__(
        self,
        backend: Backend,
        events: EventBus,
        settings: Settings,
        on_turn: Callable[[str, str, bool], None],
        instructions: Callable[[], str],
    ) -> None:
        self.backend = backend
        self.events = events
        self.settings = settings
        self._on_turn = on_turn
        self._instructions = instructions
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._state = "idle"
        self._since = time.time()
        self._error: str | None = None
        self._reply = ""
        self._started_with: dict[str, Any] = {}
        self._previews: dict[tuple, bytes] = {}

    def api_key(self) -> tuple[str | None, str]:
        return qwen_api_key(self.settings.section("voice"))

    def volc_key(self) -> tuple[str | None, str]:
        configured = (self.settings.section("voice").get("volc_api_key") or "").strip()
        if configured:
            return configured, "settings"
        env = os.environ.get("VOLC_TTS_API_KEY", "").strip()
        return (env, "env") if env else (None, "none")

    def status(self) -> dict[str, Any]:
        running = self._proc is not None and self._proc.poll() is None
        key, origin = self.api_key()
        volc_key, _ = self.volc_key()
        voice = str(self.settings.section("voice").get("voice") or "Mochi")
        return {
            "running": running,
            "state": self._state if running else ("error" if self._error else "idle"),
            "since": self._since,
            "error": self._error,
            "session": self._started_with if running else None,
            "has_key": bool(key),
            "key_source": origin,
            "has_volc_key": bool(volc_key),
            "voice": voice,
            "voice_label": label_of(voice),
            "voice_engine": engine_of(voice),
            "voices": catalog(),
            "volc_resources": VOLC_RESOURCES,
        }

    def preview(self, voice: str, text: str | None, speech_rate: float | None) -> bytes:
        """WAV of `text` in `voice`, cached per (voice, rate, text)."""
        cfg = self.settings.section("voice")
        voice = (voice or cfg.get("voice") or "Mochi").strip()
        text = (text or PREVIEW_TEXT).strip()[:120]
        rate = max(0.5, min(2.0, float(speech_rate if speech_rate is not None else cfg.get("speech_rate", 1.0))))
        cache_key = (voice, rate, text, cfg.get("volc_resource") or "")
        cached = self._previews.get(cache_key)
        if cached is not None:
            return cached
        if self.backend.mock and not (self.volc_key()[0] if is_volc(voice) else self.api_key()[0]):
            pcm = self._mock_speech(text, voice)
        else:
            from voice_preview import speak

            try:
                pcm = asyncio.run(speak(
                    voice, text, speech_rate=rate, qwen_key=self.api_key()[0] or "", qwen_url=str(cfg.get("url") or ""),
                    volc_key=self.volc_key()[0] or "", volc_resource_id=str(cfg.get("volc_resource") or ""),
                ))
            except PermissionError as exc:
                raise ApiError(str(exc), CommonState.FAILURE, Reason.NOT_CONFIGURED) from None
            except Exception as exc:  # network, auth or protocol errors from either provider
                status = getattr(getattr(exc, "response", None), "status_code", None)
                detail = f"HTTP {status}（检查 API Key）" if status in (401, 403) else str(exc) or type(exc).__name__
                raise ApiError(f"试听失败：{detail}", CommonState.FAILURE, Reason.HARDWARE, 502) from None
            if not pcm:
                raise ApiError("试听失败：服务没有返回音频", CommonState.FAILURE, Reason.HARDWARE, 502)
        out = io.BytesIO()
        with wave.open(out, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes(pcm)
        if len(self._previews) >= PREVIEW_CACHE:
            self._previews.pop(next(iter(self._previews)))
        self._previews[cache_key] = out.getvalue()
        return self._previews[cache_key]

    @staticmethod
    def _mock_speech(text: str, voice: str) -> bytes:
        """Simulation: one short vowel-like blip per character, pitched by the voice name."""
        base = 160 + sum(map(ord, voice)) % 140
        frames = bytearray()
        for i, _ in enumerate(text[:40]):
            f0 = base * (1.0 + 0.12 * math.sin(i * 1.7))
            n = 24000 * 11 // 100
            for t in range(n):
                env = math.sin(math.pi * t / n)
                s = sum(math.sin(2 * math.pi * f0 * k * t / 24000) / k for k in (1, 2, 3))
                frames += int(5000 * env * s).to_bytes(2, "little", signed=True)
        return bytes(frames)

    def _set_state(self, state: str, **extra: Any) -> None:
        if state != self._state:
            self._state, self._since = state, time.time()
        self.events.publish("voice", state=state, **extra)

    def start(self, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return response(self.status(), reason=Reason.TARGET_STATE, message="语音会话已在运行")
            cfg = {**self.settings.section("voice"), **(overrides or {})}
            key, _ = self.api_key()
            if not key and not self.backend.mock:
                raise ApiError("还没有配置千问语音 API Key（设置 → 语音）", CommonState.FAILURE, Reason.NOT_CONFIGURED)
            voice = str(cfg.get("voice") or "Mochi")
            volc_key, _ = self.volc_key()
            if is_volc(voice) and not volc_key and not self.backend.mock:
                raise ApiError("所选的是火山引擎音色，还没有填写火山引擎 API Key（设置 → 语音）", CommonState.FAILURE, Reason.NOT_CONFIGURED)
            if not self.backend.ensure_display():
                raise ApiError("显示/音频进程启动失败，无法使用麦克风和扬声器", CommonState.FAILURE, Reason.HARDWARE)
            args = self._command(cfg)
            env = dict(os.environ, PYTHONPATH=str(REPO / "src"), PYTHONUNBUFFERED="1")
            if key:
                env["DASHSCOPE_API_KEY"] = key
            if volc_key:
                env["VOLC_TTS_API_KEY"] = volc_key
            timezone = self.settings.section("memo").get("timezone")
            if timezone:
                env["TZ"] = timezone
            self._error = None
            self._started_with = {k: cfg.get(k) for k in ("route", "voice", "volume", "camera", "full_duplex", "search")}
            self._started_with |= {"route": self._route(cfg), "voice_label": label_of(voice), "voice_engine": engine_of(voice)}
            self._proc = subprocess.Popen(
                args,
                cwd=REPO,
                env=env,
                stdin=subprocess.PIPE,  # prompt() commands
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self.backend.paths.log("voice").write_text("", encoding="utf-8")
            self._set_state("starting")
            threading.Thread(target=self._pump, args=(self._proc,), name="xiaoxiaobai-api-voice", daemon=True).start()
        return response(self.status(), state=CommonState.RUNNING, message="语音会话启动中")

    @staticmethod
    def _route(cfg: dict[str, Any]) -> str:
        # The ESP32 firmware only talks to Qwen; Volcengine voices are synthesized on the Pi.
        return "pi" if cfg.get("route") == "pi" or is_volc(str(cfg.get("voice") or "")) else "esp32"

    def _command(self, cfg: dict[str, Any]) -> list[str]:
        python = self.backend.python()
        if self.backend.mock:
            return [python, "-u", str(Path(__file__).with_name("mock_voice.py"))]
        voice = str(cfg.get("voice") or "Mochi")
        args = [
            python,
            "-u",
            "src/voice_chat.py",
            "--route",
            self._route(cfg),
            "--voice",
            voice,
            "--volume",
            str(int(cfg.get("volume", 100))),
            "--mic-gain",
            str(float(cfg.get("mic_gain", 2.0))),
            "--speech-rate",
            str(float(cfg.get("speech_rate") or 1.0)),
        ]
        if is_volc(voice) and cfg.get("volc_resource"):
            args += ["--volc-resource", str(cfg["volc_resource"])]
        if cfg.get("share_persona", True):
            args += ["--instructions", self._instructions()]
        if not cfg.get("camera", True):
            args.append("--no-camera")
        if cfg.get("full_duplex"):
            args.append("--full-duplex")
        if cfg.get("search"):
            args.append("--search")
        kb_top_k = int(self.settings.section("llm").get("kb_top_k", 4))
        if kb_top_k > 0:
            args += ["--kb-top-k", str(kb_top_k)]
        if self.settings.section("memo").get("enabled", True):
            args.append("--memo")
        if self.settings.section("llm").get("memory_recall", True):
            args.append("--recall")
        if cfg.get("show_user_text", True):
            args.append("--show-user-text")
        if cfg.get("url"):
            args += ["--url", str(cfg["url"])]
        return args

    def prompt(self, text: str) -> bool:
        """Has the voice model answer `text` aloud now, as if the user had typed it (once the user and
        the robot are quiet). False when no voice session runs."""
        with self._lock:
            proc = self._proc
            if proc is None or proc.poll() is not None or proc.stdin is None:
                return False
            try:
                proc.stdin.write((json.dumps({"prompt": text}, ensure_ascii=False) + "\n").encode())
                proc.stdin.flush()
            except (OSError, ValueError):
                return False
        return True

    def stop(self) -> dict[str, Any]:
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return response(self.status(), reason=Reason.TARGET_STATE, message="语音会话没有运行")
        try:
            os.killpg(proc.pid, signal.SIGINT)
        except OSError:
            pass

        def reap() -> None:
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()

        threading.Thread(target=reap, daemon=True).start()
        return response(self.status(), state=CommonState.RUNNING, message="正在结束语音会话")

    def _pump(self, proc: subprocess.Popen) -> None:
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        line = ""
        emitted = 0
        log_path = self.backend.paths.log("voice")
        assert proc.stdout is not None
        with log_path.open("a", encoding="utf-8") as log:
            while True:
                chunk = os.read(proc.stdout.fileno(), 4096)
                if not chunk:
                    break
                text = decoder.decode(chunk)
                log.write(text)
                log.flush()
                for ch in text:
                    if ch == "\n":
                        self._on_line(line)
                        line, emitted = "", 0
                        continue
                    line += ch
                if line.startswith("机器人: ") and len(line) > max(emitted, 5):
                    start = max(emitted, 5)
                    if emitted == 0:
                        self._set_state("speaking")
                    self.events.publish("voice_reply_delta", text=line[start:])
                    emitted = len(line)
        code = proc.wait()
        if line:
            self._on_line(line)
        if code not in (0, -signal.SIGINT, 130) and not self._error:
            self._error = f"语音进程退出（{code}）：{tail(log_path, 3)}"
        self._set_state("error" if self._error else "idle", error=self._error)
        if self._error:
            self.events.alert("error", self._error)

    def _on_line(self, line: str) -> None:
        line = line.strip()
        if not line:
            return
        if line.startswith("机器人: ") and "你: " in line:
            # The user transcript can land in the middle of a streaming reply line.
            reply, _, user = line.partition("你: ")
            self._on_line("你: " + user)
            line = reply
        if line.startswith("机器人: "):
            text = line[len("机器人: ") :]
            interrupted = text.endswith("[打断]")
            text = text.replace("[打断]", "").strip()
            self.events.publish("voice_reply", text=text, interrupted=interrupted)
            self._on_turn("assistant", text, interrupted)
            self._set_state("ready")
        elif line.startswith("你: "):
            text = line[len("你: ") :].strip()
            self.events.publish("voice_user", text=text)
            self._on_turn("user", text, False)
            self._set_state("thinking")
        elif line.startswith("[ready]"):
            self._set_state("ready", message=line)
        else:
            lowered = line.lower()
            level = "error" if ("http 4" in lowered or "traceback" in lowered or "error" in lowered) else "info"
            if line.startswith("No API key") or "Cannot reach the voice bridge" in line or "is a Volcengine voice" in line:
                self._error = line
                level = "error"
            elif "火山引擎" in line and any(word in line for word in ("失败", "拒绝", "错误")):
                level = "error"
            self.events.publish("voice_log", text=line, level=level)


class SystemInfo:
    LOGS = ("run", "display", "voice", "music", "upper_body_slider", "api")

    def __init__(self, backend: Backend) -> None:
        self.backend = backend
        self.started = time.time()

    @staticmethod
    def _read(path: str) -> str:
        try:
            return Path(path).read_text(encoding="utf-8").strip()
        except OSError:
            return ""

    @staticmethod
    def ips() -> list[str]:
        ips: list[str] = []
        if shutil.which("hostname"):
            try:
                out = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=2).stdout
                ips = [ip for ip in out.split() if ":" not in ip]
            except (OSError, subprocess.TimeoutExpired):
                pass
        if not ips:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                    sock.connect(("10.255.255.255", 1))
                    ips = [sock.getsockname()[0]]
            except OSError:
                pass
        return ips

    def info(self) -> dict[str, Any]:
        mem: dict[str, int] = {}
        for row in self._read("/proc/meminfo").splitlines():
            key, _, value = row.partition(":")
            if key in ("MemTotal", "MemAvailable"):
                mem[key] = int(value.split()[0]) // 1024
        temp = self._read("/sys/class/thermal/thermal_zone0/temp")
        uptime = self._read("/proc/uptime").split()
        disk = shutil.disk_usage(str(REPO))
        git = ""
        try:
            git = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True, timeout=2
            ).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
        ssid = ""
        if shutil.which("iwgetid"):
            try:
                ssid = subprocess.run(["iwgetid", "-r"], capture_output=True, text=True, timeout=2).stdout.strip()
            except (OSError, subprocess.TimeoutExpired):
                pass
        return {
            "hostname": socket.gethostname(),
            "ips": self.ips(),
            "wifi_ssid": ssid or None,
            "mock": self.backend.mock,
            "api_version": API_VERSION,
            "git": git or None,
            "python": sys.version.split()[0],
            "uptime_s": float(uptime[0]) if uptime else None,
            "api_uptime_s": time.time() - self.started,
            "load": os.getloadavg() if hasattr(os, "getloadavg") else None,
            "mem_total_mb": mem.get("MemTotal"),
            "mem_available_mb": mem.get("MemAvailable"),
            "cpu_temp_c": round(int(temp) / 1000, 1) if temp.isdigit() else None,
            "disk_free_gb": round(disk.free / 1e9, 1),
            "display_running": self.backend.display_running(),
            "esp32": self.backend.esp32_info(),
            "gamepad": bool(list(Path("/dev/input").glob("js*"))) if Path("/dev/input").exists() else False,
        }

    def log(self, name: str, lines: int = 300) -> str:
        if name not in self.LOGS:
            raise ApiError(f"没有日志 {name}，可选：{', '.join(self.LOGS)}", CommonState.INVALID, Reason.NOT_FOUND, 404)
        return tail(self.backend.paths.log(name), lines)
