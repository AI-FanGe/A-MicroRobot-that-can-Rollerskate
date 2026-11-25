#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

"""Local Xiaoxiaobai console: connect multiple robots, push code, walk, set faces."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CONSOLE_DIR = Path(__file__).resolve().parent
for path in (str(ROOT), str(SRC), str(CONSOLE_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from expressions import catalog, loop_frames, resolve_expression  # noqa: E402
from sshutil import CONFIG_DIR, SshSession  # noqa: E402

ROBOTS_PATH = CONFIG_DIR / "robots.json"
STATIC_DIR = Path(__file__).resolve().parent / "static"
REMOTE_DIR = "xiaoxiaobai"
UPPER_BODY_UDP_PORT = 8765
RSYNC_EXCLUDES = [
    ".git",
    ".venv",
    "__pycache__",
    "cad",
    "docs",
    "logs",
    "src/debug",
    "src/sim",
    "src/model/mjcf",
    "console/static/.cache",
    ".console-data",
]

_STATUS_SCRIPT = r"""
python3 - <<'PY'
import json, os, socket
from pathlib import Path

def alive(path):
    try:
        pid = int(Path(path).read_text().strip())
        os.kill(pid, 0)
        return True
    except Exception:
        return False

expr = "idle"
try:
    expr = json.loads(Path("/tmp/xiaoxiaobai_expression.json").read_text()).get("name", "idle")
except Exception:
    pass

print(json.dumps({
    "hostname": socket.gethostname(),
    "running": alive("/tmp/xiaoxiaobai_scheduler.pid"),
    "expression": expr,
    "home": str(Path.home()),
}))
PY
"""

_WRITE_CONTROL = r"""
python3 - <<'PY'
import json, time
from pathlib import Path
payload = json.loads(r'''%s''')
payload["ts"] = time.time()
Path("/tmp/xiaoxiaobai_remote_input.json").write_text(json.dumps(payload) + "\n")
if payload.get("expression"):
    Path("/tmp/xiaoxiaobai_expression.json").write_text(
        json.dumps({"name": payload["expression"]}) + "\n"
    )
print("ok")
PY
"""

_START_RUN = r"""
bash -l -c '
set -e
cd "$HOME/xiaoxiaobai"
if [ ! -x .venv/bin/python ]; then
  echo "MISSING_VENV"
  exit 2
fi
if [ -f /tmp/xiaoxiaobai_scheduler.pid ]; then
  pid=$(cat /tmp/xiaoxiaobai_scheduler.pid)
  if kill -0 "$pid" 2>/dev/null; then
    echo "ALREADY $pid"
    exit 0
  fi
fi
nohup env PYTHONPATH=src XIAOXIAOBAI_INPUT=remote .venv/bin/python -u src/main.py \
  >/tmp/xiaoxiaobai_run.log 2>&1 </dev/null &
echo "STARTED $!"
'
"""

_START_DISPLAY = r"""
bash -l -c '
set -e
cd "$HOME/xiaoxiaobai"
if [ ! -x .venv/bin/python ]; then
  echo "MISSING_VENV"
  exit 2
fi
pkill -f src/display_smile.py >/dev/null 2>&1 || true
sleep 0.2
nohup env PYTHONPATH=src .venv/bin/python -u src/display_smile.py \
  >/tmp/xiaoxiaobai_display.log 2>&1 </dev/null &
echo "DISPLAY $!"
'
"""

_START_UPPER_BODY_SLIDER = r"""
bash -l -c '
set -e
cd "$HOME/xiaoxiaobai"
if [ ! -x .venv/bin/python ]; then
  echo "MISSING_VENV"
  exit 2
fi
if [ ! -f src/upper_body_slider.py ]; then
  echo "MISSING_UPPER_BODY_SLIDER"
  exit 2
fi
if [ -f /tmp/xiaoxiaobai_upper_body_slider.pid ]; then
  pid=$(cat /tmp/xiaoxiaobai_upper_body_slider.pid)
  if kill -0 "$pid" 2>/dev/null; then
    echo "UPPER_BODY_SLIDER_ALREADY $pid"
    exit 0
  fi
fi
nohup env PYTHONPATH=src .venv/bin/python -u src/upper_body_slider.py --ids 13-20 --udp-port 8765 \
  >/tmp/xiaoxiaobai_upper_body_slider.log 2>&1 </dev/null &
echo "UPPER_BODY_SLIDER $!"
'
"""

_WRITE_UPPER_BODY_TARGETS = r"""
python3 - <<'PY'
import json, time
from pathlib import Path
payload = json.loads(r'''%s''')
payload["ts"] = time.time()
Path("/tmp/xiaoxiaobai_upper_body_slider.json").write_text(
    json.dumps(payload, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
print("ok")
PY
"""

_STOP_UPPER_BODY_SLIDER = r"""
bash -l -c '
set +e
printf stop > /tmp/xiaoxiaobai_upper_body_slider.stop
if [ -f /tmp/xiaoxiaobai_upper_body_slider.pid ]; then
  pid=$(cat /tmp/xiaoxiaobai_upper_body_slider.pid)
  kill "$pid" >/dev/null 2>&1 || true
fi
sleep 0.3
pkill -f "[u]pper_body_slider.py" >/dev/null 2>&1 || true
echo "UPPER_BODY_STOPPED"
'
"""

_STOP_RUN = r"""
bash -l -c '
cd "$HOME/xiaoxiaobai"
if [ -x .venv/bin/python ]; then
  PYTHONPATH=src .venv/bin/python src/stop.py
else
  python3 - <<'"'"'PY'"'"'
from pathlib import Path
Path("/tmp/xiaoxiaobai_scheduler.stop").write_text("stop\n")
print("stop flag written")
PY
fi
'
"""


def _load_robots() -> list[dict[str, Any]]:
    if not ROBOTS_PATH.exists():
        return []
    try:
        data = json.loads(ROBOTS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(data, list):
        return data
    return []


def _save_robots(robots: list[dict[str, Any]]) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(CONFIG_DIR, 0o700)
    except OSError:
        pass
    ROBOTS_PATH.write_text(json.dumps(robots, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        os.chmod(ROBOTS_PATH, 0o600)
    except OSError:
        pass


def _public_robot(robot: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": robot.get("id"),
        "name": robot.get("name") or robot.get("host"),
        "host": robot.get("host"),
        "user": robot.get("user") or "user",
        "port": int(robot.get("port") or 22),
        "has_password": bool(robot.get("password")),
    }


def _find_robot(robot_id: str) -> dict[str, Any] | None:
    for robot in _load_robots():
        if robot.get("id") == robot_id:
            return robot
    return None


def _session(robot: dict[str, Any]) -> SshSession:
    return SshSession(
        host=str(robot["host"]),
        user=str(robot.get("user") or "user"),
        port=int(robot.get("port") or 22),
        password=robot.get("password") or None,
    )


def _combine_output(proc: subprocess.CompletedProcess[str]) -> str:
    chunks = [proc.stdout or "", proc.stderr or ""]
    return "\n".join(part.strip() for part in chunks if part and part.strip())


def _decode_proc(proc: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    return {
        "ok": proc.returncode == 0,
        "code": proc.returncode,
        "output": _combine_output(proc)[-8000:],
    }


def probe_robot(robot: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    try:
        proc = _session(robot).run(_STATUS_SCRIPT, timeout=12)
    except subprocess.TimeoutExpired:
        return {"ok": False, "online": False, "error": "SSH 超时", "latency_ms": None}
    except Exception as exc:
        return {"ok": False, "online": False, "error": str(exc), "latency_ms": None}
    latency = int((time.time() - started) * 1000)
    if proc.returncode != 0:
        raw = _combine_output(proc) or f"ssh 退出码 {proc.returncode}"
        lowered = raw.lower()
        if any(token in lowered for token in ("permission denied", "publickey", "password", "authentication")):
            raw = "SSH 需要密码。请在上方输入这台 Pi 的登录密码，再点「保存密码并连接」。\n" + raw
        elif "connection refused" in lowered:
            raw = "连不上这台主机的 22 端口。请确认主机名是 xiaoxiaobai.local 或机器人的 IP，不要填 127.0.0.1。\n" + raw
        return {
            "ok": False,
            "online": False,
            "error": raw,
            "latency_ms": latency,
            "need_password": "permission denied" in lowered or "publickey" in lowered,
        }
    payload: dict[str, Any] = {}
    for line in reversed((proc.stdout or "").splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                payload = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
    return {
        "ok": True,
        "online": True,
        "latency_ms": latency,
        "running": bool(payload.get("running")),
        "expression": payload.get("expression") or "happy",
        "hostname": payload.get("hostname"),
    }


def sync_robot(robot: dict[str, Any], setup: bool = False) -> dict[str, Any]:
    session = _session(robot)
    try:
        proc = session.rsync_to_remote(ROOT, REMOTE_DIR, RSYNC_EXCLUDES)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "代码推送超时"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    result = _decode_proc(proc)
    if not result["ok"]:
        result["error"] = result.get("output") or "rsync 失败"
        return result
    if setup:
        try:
            setup_proc = session.run(
                "bash -l -c 'cd \"$HOME/xiaoxiaobai\" && uv sync --frozen'",
                timeout=240,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "依赖安装超时", "output": result["output"]}
        setup_result = _decode_proc(setup_proc)
        setup_result["output"] = ((result["output"] or "") + "\n" + (setup_result["output"] or "")).strip()
        if not setup_result["ok"]:
            setup_result["error"] = setup_result.get("output") or "uv sync 失败"
        return setup_result
    display = start_display(robot)
    result["message"] = "代码已推送到机器人，屏幕已用新表情重启"
    if display.get("output"):
        result["output"] = ((result.get("output") or "") + "\n" + display["output"]).strip()
    return result


def run_robot(robot: dict[str, Any]) -> dict[str, Any]:
    try:
        proc = _session(robot).run(_START_RUN, timeout=20)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    output = _combine_output(proc)
    if "MISSING_VENV" in output:
        return {"ok": False, "error": "机器人上还没有 .venv，请先点「推送并安装依赖」", "output": output}
    return {"ok": proc.returncode == 0, "output": output, "error": None if proc.returncode == 0 else output}


def stop_robot(robot: dict[str, Any]) -> dict[str, Any]:
    try:
        proc = _session(robot).run(_STOP_RUN, timeout=15)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return _decode_proc(proc)


def shutdown_robot(robot: dict[str, Any]) -> dict[str, Any]:
    try:
        proc = _session(robot).run("sudo shutdown -h now", timeout=10, tty=True)
    except subprocess.TimeoutExpired:
        return {"ok": True, "output": "已发送关机命令"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": proc.returncode in (0, 255), "output": _combine_output(proc)}


_LEAD_REMOTE = (
    'bash -l -c \'cd "$HOME/xiaoxiaobai" && '
    'if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python3; fi && '
    'PYTHONPATH=src "$PY" -u src/lead_teach.py {role} --ids {ids}\''
)

_lead_lock = threading.Lock()
_lead_state: dict[str, Any] = {
    "running": False,
    "phase": "stopped",
    "lead": None,
    "follow": None,
    "frames": 0,
    "error": None,
    "ids": [],
}
_lead_stop = threading.Event()
_lead_thread: threading.Thread | None = None
_lead_procs: list[subprocess.Popen[str]] = []
_lead_sessions: list[SshSession] = []


def lead_status() -> dict[str, Any]:
    with _lead_lock:
        return dict(_lead_state)


def _set_lead_phase(phase: str, **extra: Any) -> None:
    with _lead_lock:
        _lead_state["phase"] = phase
        _lead_state.update(extra)


def stop_lead() -> dict[str, Any]:
    _lead_stop.set()
    with _lead_lock:
        if _lead_state["running"]:
            _lead_state["phase"] = "stopping"
    for session in list(_lead_sessions):
        try:
            session.run("pkill -f src/lead_teach.py >/dev/null 2>&1 || true", timeout=6)
        except Exception:
            pass
    for proc in list(_lead_procs):
        if proc.poll() is None:
            try:
                if proc.stdin:
                    proc.stdin.close()
            except OSError:
                pass
            proc.terminate()
    return {"ok": True, "running": False}


def start_lead(lead_robot: dict[str, Any], follow_robot: dict[str, Any], ids: str = "13-18") -> dict[str, Any]:
    global _lead_thread
    if lead_robot.get("id") == follow_robot.get("id"):
        return {"ok": False, "error": "示教机和真机不能是同一台"}
    with _lead_lock:
        if _lead_state["running"]:
            return {"ok": False, "error": "示教已经在跑，先点停止"}
        _lead_state.update(
            {
                "running": True,
                "phase": "starting",
                "lead": _public_robot(lead_robot),
                "follow": _public_robot(follow_robot),
                "frames": 0,
                "error": None,
                "ids": [],
            }
        )
    _lead_stop.clear()

    def worker() -> None:
        global _lead_thread
        lead_ssh = _session(lead_robot)
        follow_ssh = _session(follow_robot)
        _lead_sessions.extend([lead_ssh, follow_ssh])
        try:
            for robot in (lead_robot, follow_robot):
                _set_lead_phase("starting", error=None)
                result = sync_robot(robot, setup=False)
                if not result.get("ok"):
                    raise RuntimeError(result.get("error") or f"{robot.get('name')} 推送失败")
                stopped = stop_robot(robot)
                if not stopped.get("ok"):
                    raise RuntimeError(stopped.get("error") or f"{robot.get('name')} 停控制循环失败")
            if _lead_stop.is_set():
                return
            follow_proc = follow_ssh.popen(_LEAD_REMOTE.format(role="follow", ids=ids))
            lead_proc = lead_ssh.popen(_LEAD_REMOTE.format(role="lead", ids=ids))
            _lead_procs.extend([follow_proc, lead_proc])
            assert follow_proc.stdin is not None and lead_proc.stdout is not None

            def pump_stderr(prefix: str, proc: subprocess.Popen[str]) -> None:
                if proc.stderr is None:
                    return
                for raw in proc.stderr:
                    text = raw.rstrip()
                    if not text:
                        continue
                    lowered = text.lower()
                    if any(token in lowered for token in ("error", "traceback", "exception", "没有可用", "not found")):
                        with _lead_lock:
                            _lead_state["error"] = text

            threading.Thread(target=pump_stderr, args=("lead", lead_proc), daemon=True).start()
            threading.Thread(target=pump_stderr, args=("follow", follow_proc), daemon=True).start()

            for line in lead_proc.stdout:
                if _lead_stop.is_set() or follow_proc.poll() is not None:
                    break
                follow_proc.stdin.write(line)
                follow_proc.stdin.flush()
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if payload.get("error"):
                    raise RuntimeError(str(payload["error"]))
                if payload.get("ready") == "lead" and payload.get("ids"):
                    _set_lead_phase("streaming", ids=payload["ids"])
                    continue
                if payload.get("ids") and payload.get("pos") is not None:
                    with _lead_lock:
                        _lead_state["phase"] = "streaming"
                        _lead_state["frames"] = int(_lead_state["frames"]) + 1
                        _lead_state["ids"] = payload["ids"]
            if follow_proc.poll() not in (None, 0) and not _lead_stop.is_set():
                raise RuntimeError(_lead_state.get("error") or "真机 follow 退出了")
        except Exception as exc:  # noqa: BLE001
            with _lead_lock:
                _lead_state["error"] = str(exc)
        finally:
            stop_lead()
            for proc in list(_lead_procs):
                if proc.poll() is None:
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        proc.kill()
            _lead_procs.clear()
            for session in list(_lead_sessions):
                session.close()
            _lead_sessions.clear()
            with _lead_lock:
                _lead_state["running"] = False
                _lead_state["phase"] = "stopped"
            _lead_thread = None

    _lead_thread = threading.Thread(target=worker, name="xiaoxiaobai-lead", daemon=True)
    _lead_thread.start()
    return {"ok": True, "running": True, "phase": "starting"}


def start_display(robot: dict[str, Any]) -> dict[str, Any]:
    try:
        proc = _session(robot).run(_START_DISPLAY, timeout=15)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    output = _combine_output(proc)
    if "MISSING_VENV" in output:
        return {"ok": False, "error": "机器人上还没有 .venv，请先推送并安装依赖", "output": output}
    return {"ok": proc.returncode == 0, "output": output}


def start_upper_body_slider(robot: dict[str, Any]) -> dict[str, Any]:
    try:
        proc = _session(robot).run(_START_UPPER_BODY_SLIDER, timeout=12)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    output = _combine_output(proc)
    if "MISSING_VENV" in output:
        return {"ok": False, "error": "机器人上还没有 .venv，请先点「推送并安装依赖」", "output": output}
    if "MISSING_UPPER_BODY_SLIDER" in output:
        return {"ok": False, "error": "机器人上还没有 slider 控制脚本，请先点「推送代码」", "output": output}
    return {"ok": proc.returncode == 0, "output": output, "error": None if proc.returncode == 0 else output}


def stop_upper_body_slider(robot: dict[str, Any]) -> dict[str, Any]:
    try:
        proc = _session(robot).run(_STOP_UPPER_BODY_SLIDER, timeout=8)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return _decode_proc(proc)


def _upper_body_targets(body: dict[str, Any]) -> dict[str, float]:
    targets_deg = body.get("targets_deg") or {}
    if not isinstance(targets_deg, dict):
        raise ValueError("targets_deg 必须是对象")
    targets: dict[str, float] = {}
    for motor_id in range(13, 21):
        raw = targets_deg.get(str(motor_id))
        if raw is None:
            continue
        value = float(raw)
        if not -120.0 <= value <= 120.0:
            raise ValueError(f"ID {motor_id} 超出范围：{value:.1f}°")
        targets[str(motor_id)] = value * 3.141592653589793 / 180.0
    if not targets:
        raise ValueError("没有可下发的 13-20 目标角")
    return targets


def set_upper_body_targets(robot: dict[str, Any], body: dict[str, Any]) -> dict[str, Any]:
    try:
        targets = _upper_body_targets(body)
    except (TypeError, ValueError) as exc:
        return {"ok": False, "error": str(exc)}
    payload = json.dumps({"targets": targets, "ts": time.time()}, ensure_ascii=False).encode("utf-8")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.sendto(payload, (str(robot["host"]), UPPER_BODY_UDP_PORT))
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "targets": targets}


def send_control(robot: dict[str, Any], body: dict[str, Any]) -> dict[str, Any]:
    moves = []
    if body.get("walk"):
        moves.append("walk")
    if body.get("head"):
        moves.append("head")
    expression = body.get("expression")
    if expression:
        expression = resolve_expression(str(expression)).name
    payload = {
        "active_moves": moves,
        "velocity": {
            "vx": float(body.get("vx", 0.0)),
            "vy": float(body.get("vy", 0.0)),
            "vtheta": float(body.get("vtheta", 0.0)),
        },
        "expression": expression,
    }
    script = _WRITE_CONTROL % json.dumps(payload)
    try:
        proc = _session(robot).run(script, timeout=8)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    result = _decode_proc(proc)
    if expression and result["ok"]:
        result["expression"] = expression
    return result


def set_expression(robot: dict[str, Any], name: str) -> dict[str, Any]:
    resolved = resolve_expression(name).name
    script = (
        "python3 - <<'PY'\n"
        "from pathlib import Path\n"
        f"Path('/tmp/xiaoxiaobai_expression.json').write_text('{{\\\"name\\\": \\\"{resolved}\\\"}}\\n')\n"
        "print('ok')\n"
        "PY\n"
    )
    try:
        proc = _session(robot).run(script, timeout=8)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    result = _decode_proc(proc)
    result["expression"] = resolved
    return result


def discover_robots() -> list[dict[str, str]]:
    found: dict[str, dict[str, str]] = {}

    def add(host: str, name: str = "") -> None:
        host = host.strip().rstrip(".")
        if not host:
            return
        found[host] = {"host": host, "name": name or host.split(".")[0]}

    if shutil.which("avahi-browse"):
        try:
            proc = subprocess.run(
                ["avahi-browse", "-prt", "_ssh._tcp"],
                text=True,
                capture_output=True,
                timeout=4,
            )
            for line in (proc.stdout or "").splitlines():
                if not line.startswith("="):
                    continue
                parts = line.split(";")
                if len(parts) >= 8:
                    add(parts[6], parts[3])
        except (subprocess.TimeoutExpired, OSError):
            pass

    for candidate in ("xiaoxiaobai.local", "xiaoxiaobai-ext.local"):
        try:
            socket.getaddrinfo(candidate, 22)
            add(candidate)
        except OSError:
            pass
    return list(found.values())


def _json_bytes(payload: Any, status: int = 200) -> tuple[int, bytes, str]:
    return status, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8"


def _normalize_robot(body: dict[str, Any], existing: dict[str, Any] | None = None) -> dict[str, Any]:
    current = dict(existing or {})
    host = str(body.get("host") or current.get("host") or "").strip()
    if not host:
        raise ValueError("请填写主机名或 IP")
    robot = {
        "id": current.get("id") or uuid.uuid4().hex[:10],
        "name": str(body.get("name") or current.get("name") or host).strip(),
        "host": host,
        "user": str(body.get("user") or current.get("user") or "user").strip(),
        "port": int(body.get("port") or current.get("port") or 22),
    }
    if "password" in body:
        password = str(body.get("password") or "")
        if password:
            robot["password"] = password
        elif "password" in current:
            robot.pop("password", None)
    elif current.get("password"):
        robot["password"] = current["password"]
    return robot


class ConsoleHandler(BaseHTTPRequestHandler):
    server_version = "XiaoxiaobaiConsole/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        if not raw:
            return {}
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("JSON 必须是对象")
        return data

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/":
            self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
            return
        if path.startswith("/static/"):
            relative = path[len("/static/") :]
            target = (STATIC_DIR / relative).resolve()
            if not str(target).startswith(str(STATIC_DIR.resolve())) or not target.is_file():
                self._send(*_json_bytes({"error": "not found"}, 404))
                return
            types = {".css": "text/css; charset=utf-8", ".js": "text/javascript; charset=utf-8"}
            self._send_file(target, types.get(target.suffix, "application/octet-stream"))
            return
        if path == "/api/expressions":
            self._send(*_json_bytes({"items": catalog(8), "cols": 64, "rows": 48}))
            return
        match = re.fullmatch(r"/api/expressions/([^/]+)/loop", path)
        if match:
            self._send(*_json_bytes(loop_frames(match.group(1), pixels=True)))
            return
        if path == "/api/robots":
            self._send(*_json_bytes({"robots": [_public_robot(item) for item in _load_robots()]}))
            return
        if path == "/api/discover":
            self._send(*_json_bytes({"hosts": discover_robots()}))
            return
        if path == "/api/lead":
            self._send(*_json_bytes(lead_status()))
            return
        match = re.fullmatch(r"/api/robots/([^/]+)/status", path)
        if match:
            robot = _find_robot(match.group(1))
            if robot is None:
                self._send(*_json_bytes({"error": "机器人不存在"}, 404))
                return
            self._send(*_json_bytes(probe_robot(robot)))
            return
        self._send(*_json_bytes({"error": "not found"}, 404))

    def do_DELETE(self) -> None:
        match = re.fullmatch(r"/api/robots/([^/]+)", urlparse(self.path).path)
        if not match:
            self._send(*_json_bytes({"error": "not found"}, 404))
            return
        robots = [item for item in _load_robots() if item.get("id") != match.group(1)]
        _save_robots(robots)
        self._send(*_json_bytes({"ok": True}))

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            body = self._read_json()
        except (ValueError, json.JSONDecodeError) as exc:
            self._send(*_json_bytes({"error": str(exc)}, 400))
            return

        if path == "/api/lead/start":
            lead_robot = _find_robot(str(body.get("lead_id") or ""))
            follow_robot = _find_robot(str(body.get("follow_id") or ""))
            if lead_robot is None or follow_robot is None:
                self._send(*_json_bytes({"error": "请先在列表里选好示教机和真机"}, 400))
                return
            ids = str(body.get("ids") or "13-18").strip() or "13-18"
            result = start_lead(lead_robot, follow_robot, ids)
            status = 400 if not result.get("ok") else 200
            self._send(*_json_bytes(result, status))
            return
        if path == "/api/lead/stop":
            self._send(*_json_bytes(stop_lead()))
            return

        if path == "/api/robots":
            try:
                robot = _normalize_robot(body)
            except ValueError as exc:
                self._send(*_json_bytes({"error": str(exc)}, 400))
                return
            robots = _load_robots()
            robots.append(robot)
            _save_robots(robots)
            self._send(*_json_bytes({"robot": _public_robot(robot)}))
            return

        match = re.fullmatch(
            r"/api/robots/([^/]+)/(probe|sync|setup|run|stop|shutdown|control|expression|display|upper-body|upper-body-start|upper-body-stop)",
            path,
        )
        if match:
            robot = _find_robot(match.group(1))
            if robot is None:
                self._send(*_json_bytes({"error": "机器人不存在"}, 404))
                return
            action = match.group(2)
            if action == "probe":
                self._send(*_json_bytes(probe_robot(robot)))
                return
            if action == "sync":
                self._send(*_json_bytes(sync_robot(robot, setup=False)))
                return
            if action == "setup":
                self._send(*_json_bytes(sync_robot(robot, setup=True)))
                return
            if action == "run":
                self._send(*_json_bytes(run_robot(robot)))
                return
            if action == "stop":
                self._send(*_json_bytes(stop_robot(robot)))
                return
            if action == "shutdown":
                self._send(*_json_bytes(shutdown_robot(robot)))
                return
            if action == "display":
                self._send(*_json_bytes(start_display(robot)))
                return
            if action == "upper-body-start":
                self._send(*_json_bytes(start_upper_body_slider(robot)))
                return
            if action == "upper-body-stop":
                self._send(*_json_bytes(stop_upper_body_slider(robot)))
                return
            if action == "upper-body":
                result = set_upper_body_targets(robot, body)
                self._send(*_json_bytes(result, 400 if not result.get("ok") else 200))
                return
            if action == "control":
                self._send(*_json_bytes(send_control(robot, body)))
                return
            if action == "expression":
                name = str(body.get("name") or "")
                self._send(*_json_bytes(set_expression(robot, name)))
                return

        match = re.fullmatch(r"/api/robots/([^/]+)", path)
        if match:
            existing = _find_robot(match.group(1))
            if existing is None:
                self._send(*_json_bytes({"error": "机器人不存在"}, 404))
                return
            try:
                updated = _normalize_robot(body, existing)
            except ValueError as exc:
                self._send(*_json_bytes({"error": str(exc)}, 400))
                return
            robots = [updated if item.get("id") == updated["id"] else item for item in _load_robots()]
            _save_robots(robots)
            self._send(*_json_bytes({"robot": _public_robot(updated)}))
            return

        self._send(*_json_bytes({"error": "not found"}, 404))

    def _send_file(self, path: Path, content_type: str) -> None:
        if not path.is_file():
            self._send(*_json_bytes({"error": "not found"}, 404))
            return
        data = path.read_bytes()
        self._send(200, data, content_type)


def _existing_console_url(host: str, port: int) -> str | None:
    try:
        with socket.create_connection((host, port), timeout=0.4) as sock:
            sock.sendall(b"GET /api/expressions HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
            body = sock.recv(512).decode("utf-8", errors="replace")
    except OSError:
        return None
    if "200" in body.split("\n", 1)[0] and ("items" in body or "application/json" in body):
        return f"http://{host}:{port}"
    return None


def _stop_console(port: int) -> None:
    subprocess.run(
        ["fuser", "-k", f"{port}/tcp"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(20):
        if _existing_console_url("127.0.0.1", port) is None:
            return
        time.sleep(0.1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Xiaoxiaobai 多机控制台")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--restart", action="store_true", help="关掉旧控制台再启动，加载最新表情")
    args = parser.parse_args()
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"配置目录: {CONFIG_DIR}", flush=True)
    if args.restart:
        _stop_console(args.port)
    ThreadingHTTPServer.allow_reuse_address = True
    try:
        httpd = ThreadingHTTPServer((args.host, args.port), ConsoleHandler)
    except OSError as exc:
        if getattr(exc, "errno", None) != 98:
            raise
        existing = _existing_console_url(args.host, args.port)
        if existing and not args.restart:
            print(f"控制台已经在运行: {existing}", flush=True)
            print("浏览器打开上面的地址即可。若刚加过表情，请执行: make console", flush=True)
            return
        print(
            f"端口 {args.port} 已被占用。可以先执行:  fuser -k {args.port}/tcp",
            file=sys.stderr,
            flush=True,
        )
        print(f"或者换端口:  python3 console/server.py --port 8081", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc
    print(f"Xiaoxiaobai 控制台: http://{args.host}:{args.port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已退出", flush=True)


if __name__ == "__main__":
    main()
