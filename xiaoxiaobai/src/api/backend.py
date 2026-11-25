# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

"""Processes the robot API drives: the control loop (main.py), the upper-body slider and the
display loop. RealBackend runs them on the Pi; MockBackend simulates them so the web UI can
be developed on a PC with no robot attached (`make api-dev`)."""

from __future__ import annotations

import ast
import fcntl
import json
import math
import os
import random
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
UPPER_BODY_UDP_PORT = 8765


def read_constants(*names: str) -> dict[str, Any]:
    """Literal values from constants.py, read without importing it (it needs numpy)."""
    tree = ast.parse((SRC / "constants.py").read_text(encoding="utf-8"))
    found: dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = [getattr(t, "id", None) for t in node.targets]
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [getattr(node.target, "id", None)]
        else:
            continue
        for name in names:
            if name in targets:
                found[name] = ast.literal_eval(node.value)
    missing = set(names) - set(found)
    if missing:
        raise RuntimeError(f"{', '.join(sorted(missing))} not found in constants.py")
    return found


JOINT_IDS: dict[str, int] = dict(read_constants("MOTOR_TO_ID")["MOTOR_TO_ID"])
CHILL_CLIPS: dict[str, tuple] = read_constants("CHILL_CLIPS")["CHILL_CLIPS"]
_chill_clips: dict[str, dict[str, Any]] = {}


def _chill_clip(name: str) -> dict[str, Any]:
    """A chill clip for the mock loop to play (moves/chill.py)."""
    if name not in _chill_clips:
        import bisect

        import numpy as np

        d = np.load(SRC / "agents" / CHILL_CLIPS[name][0])
        fps = float(d["fps"])
        starts = [round(float(s) * fps) for s in d["section_starts"]]
        labels = [str(s) for s in d["section_names"]]
        _chill_clips[name] = {
            "names": [str(n) for n in d["joint_names"]] + ["head_pitch"],
            "q": np.concatenate([d["joint_pos"], d["head_pitch"][:, None]], axis=1),
            "fps": fps,
            "loop_start": int(d["loop_start"]),
            "section": lambda f: labels[max(0, bisect.bisect_right(starts, f) - 1)],
        }
    return _chill_clips[name]


@dataclass(frozen=True)
class RuntimePaths:
    root: Path

    @property
    def remote_input(self) -> Path:
        return self.root / "xiaoxiaobai_remote_input.json"

    @property
    def status(self) -> Path:
        return self.root / "xiaoxiaobai_status.json"

    @property
    def expression(self) -> Path:
        return self.root / "xiaoxiaobai_expression.json"

    @property
    def sound(self) -> Path:
        return self.root / "xiaoxiaobai_sound.json"

    @property
    def board(self) -> Path:
        return self.root / "xiaoxiaobai_board.json"

    @property
    def brightness(self) -> Path:
        return self.root / "xiaoxiaobai_brightness.json"

    @property
    def scheduler_pid(self) -> Path:
        return self.root / "xiaoxiaobai_scheduler.pid"

    @property
    def scheduler_stop(self) -> Path:
        return self.root / "xiaoxiaobai_scheduler.stop"

    @property
    def upper_pid(self) -> Path:
        return self.root / "xiaoxiaobai_upper_body_slider.pid"

    @property
    def upper_stop(self) -> Path:
        return self.root / "xiaoxiaobai_upper_body_slider.stop"

    @property
    def upper_state(self) -> Path:
        return self.root / "xiaoxiaobai_upper_body_slider_state.json"

    @property
    def display_lock(self) -> Path:
        return self.root / "xiaoxiaobai_display.lock"

    @property
    def esp32_info(self) -> Path:
        return self.root / "xiaoxiaobai_esp32.json"

    def log(self, name: str) -> Path:
        return self.root / f"xiaoxiaobai_{name}.log"


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def read_pid(path: Path) -> int | None:
    try:
        pid = int(path.read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        return None
    return pid if pid_alive(pid) else None


def tail(path: Path, lines: int = 200) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 64 * 1024))
            data = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
    data = data.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(data.splitlines()[-lines:])


class Backend:
    mock = False
    paths: RuntimePaths

    def python(self) -> str:
        venv = REPO / ".venv" / "bin" / "python"
        return str(venv) if venv.exists() else sys.executable

    def env(self, **extra: str) -> dict[str, str]:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(SRC)
        env.update(extra)
        return env

    def read_status(self) -> dict[str, Any] | None:
        try:
            data = json.loads(self.paths.status.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None

    def esp32_info(self) -> dict[str, Any] | None:
        try:
            return json.loads(self.paths.esp32_info.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    # Overridden below.
    def loop_pid(self) -> int | None: ...
    def start_loop(self) -> None: ...
    def stop_loop(self) -> None: ...
    def upper_pid(self) -> int | None: ...
    def start_upper(self) -> None: ...
    def stop_upper(self) -> None: ...
    def send_upper(self, targets_rad: dict[int, float], limp: list[int] | None = None) -> None: ...

    def upper_state(self) -> dict[str, Any] | None:
        """What upper_body_slider.py last wrote: motor positions [rad] (measured on limp motors,
        commanded on the others) and the limp ids."""
        try:
            data = json.loads(self.paths.upper_state.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None

    def display_running(self) -> bool: ...
    def ensure_display(self) -> bool: ...
    def stop_display(self) -> None: ...
    def camera_port(self) -> int | None: ...
    def power(self, action: str) -> str: ...


class RealBackend(Backend):
    def __init__(self) -> None:
        self.paths = RuntimePaths(Path("/tmp"))
        self._loop_proc: subprocess.Popen | None = None
        self._upper_proc: subprocess.Popen | None = None
        self._display_proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    def _spawn(self, args: list[str], log_name: str, **env: str) -> subprocess.Popen:
        log = self.paths.log(log_name).open("ab")
        try:
            return subprocess.Popen(
                [self.python(), "-u", *args],
                cwd=REPO,
                env=self.env(**env),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            log.close()

    @staticmethod
    def _reap(proc: subprocess.Popen | None) -> int | None:
        if proc is not None and proc.poll() is None:
            return proc.pid
        return None

    def loop_pid(self) -> int | None:
        return self._reap(self._loop_proc) or read_pid(self.paths.scheduler_pid)

    def start_loop(self) -> None:
        with self._lock:
            if self.loop_pid():
                return
            self.paths.scheduler_stop.unlink(missing_ok=True)
            self.paths.status.unlink(missing_ok=True)
            self.paths.log("run").write_bytes(b"")
            self._loop_proc = self._spawn(["src/main.py"], "run", XIAOXIAOBAI_INPUT="auto")

    def stop_loop(self) -> None:
        pid = self.loop_pid()
        self.paths.scheduler_stop.write_text("stop\n", encoding="ascii")
        if pid is None:
            return

        def escalate() -> None:
            # The loop polls the stop flag every tick and disables torque itself; only a
            # hung loop gets SIGINT, then SIGTERM followed by stop.py's direct torque-off.
            for sig, wait_s in ((None, 3.0), (signal.SIGINT, 3.0), (signal.SIGTERM, 1.0)):
                if sig is not None:
                    try:
                        os.kill(pid, sig)
                    except OSError:
                        return
                deadline = time.time() + wait_s
                while time.time() < deadline:
                    if not pid_alive(pid) or (self._loop_proc and self._loop_proc.poll() is not None):
                        return
                    time.sleep(0.05)
            subprocess.run([self.python(), "src/stop.py"], cwd=REPO, env=self.env(), timeout=20, check=False)

        threading.Thread(target=escalate, name="xiaoxiaobai-api-stop", daemon=True).start()

    def upper_pid(self) -> int | None:
        return self._reap(self._upper_proc) or read_pid(self.paths.upper_pid)

    def start_upper(self) -> None:
        with self._lock:
            if self.upper_pid():
                return
            self.paths.upper_stop.unlink(missing_ok=True)
            self._upper_proc = self._spawn(
                ["src/upper_body_slider.py", "--ids", "13-20", "--udp-port", str(UPPER_BODY_UDP_PORT)], "upper_body_slider"
            )

    def stop_upper(self) -> None:
        self.paths.upper_stop.write_text("stop\n", encoding="ascii")
        pid = self.upper_pid()
        if pid:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass

    def send_upper(self, targets_rad: dict[int, float], limp: list[int] | None = None) -> None:
        body: dict[str, Any] = {"targets": {str(k): v for k, v in targets_rad.items()}, "ts": time.time()}
        if limp is not None:
            body["limp"] = list(limp)
        payload = json.dumps(body).encode()
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.sendto(payload, ("127.0.0.1", UPPER_BODY_UDP_PORT))

    def display_running(self) -> bool:
        try:
            fd = os.open(self.paths.display_lock, os.O_CREAT | os.O_RDWR, 0o666)
        except OSError:
            return False
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        else:
            fcntl.flock(fd, fcntl.LOCK_UN)
            return False
        finally:
            os.close(fd)

    def ensure_display(self) -> bool:
        """The display loop owns the ESP32: face, speaker, microphone (voice) and camera."""
        with self._lock:
            if self.display_running():
                return True
            self._display_proc = self._spawn(["src/display_smile.py"], "display")
        for _ in range(30):
            if self.display_running():
                return True
            if self._display_proc.poll() is not None:
                return False
            time.sleep(0.1)
        return self.display_running()

    def stop_display(self) -> None:
        """Stop the display loop if this service started it (a boot display service is left alone)."""
        proc, self._display_proc = self._display_proc, None
        if proc is None or proc.poll() is not None:
            return
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            proc.kill()

    def camera_port(self) -> int | None:
        from camera_stream import camera_port

        return camera_port()

    def power(self, action: str) -> str:
        command = {"shutdown": ["sudo", "-n", "shutdown", "-h", "now"], "reboot": ["sudo", "-n", "reboot"]}[action]
        self.stop_loop()
        proc = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
        return (proc.stdout + proc.stderr).strip()


class MockBackend(Backend):
    """Simulated control loop with the same files and timings as the real one."""

    mock = True

    def __init__(self) -> None:
        self.paths = RuntimePaths(Path(os.environ.get("XIAOXIAOBAI_MOCK_DIR") or "/tmp/xiaoxiaobai_mock"))
        self.paths.root.mkdir(parents=True, exist_ok=True)
        self._loop: threading.Thread | None = None
        self._loop_stop = threading.Event()
        self._upper = False
        self.upper_targets: dict[int, float] = {}
        self.upper_limp: set[int] = set()
        self._voltage = 8.05
        self._pid = 0

    def loop_pid(self) -> int | None:
        return self._pid if self._loop is not None and self._loop.is_alive() else None

    def start_loop(self) -> None:
        if self.loop_pid():
            return
        self.paths.status.unlink(missing_ok=True)
        self.paths.scheduler_stop.unlink(missing_ok=True)
        self._loop_stop.clear()
        self._pid = random.randint(20000, 60000)
        self._loop = threading.Thread(target=self._run_loop, name="xiaoxiaobai-mock-loop", daemon=True)
        self._loop.start()

    def stop_loop(self) -> None:
        self.paths.scheduler_stop.write_text("stop\n", encoding="ascii")

    def _write(self, payload: dict[str, Any]) -> None:
        tmp = self.paths.status.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, self.paths.status)

    def _run_loop(self) -> None:
        import gestures
        from input.remote_input import RemoteInputSource

        source = RemoteInputSource(self.paths.remote_input)
        upper = gestures.UpperBody()
        upper_seq = None
        held: dict[str, float] = {}  # upper joints while limp, drifting as if moved by hand
        with self.paths.log("run").open("a", encoding="utf-8") as log:
            log.write("[mock] Ramping all motors to neutral position...\n[mock] loading policies\n")
        time.sleep(2.5)  # ramp_to_neutral + onnx preload
        moves = {"head": "inactive", "walk": "inactive", "skate": "inactive",
                 "gymnastics": "inactive", "peek": "inactive", "propup": "inactive", "upper": "inactive",
                 **{name: "inactive" for name in CHILL_CLIPS}}
        since = {name: 0.0 for name in moves}
        propup_t: float | None = None  # reference time the mock 撑起/躺下 shows (moves/propup.py)
        chill_settle: float | None = None  # when the mock chill move began settling (moves/chill.py)
        calib_seq, calib_start, calib_result = 0.0, None, None  # mock IMU mount calibration (imu_calib.py)
        t0 = time.time()
        reason = "stop_flag"
        while True:
            if self.paths.scheduler_stop.exists():
                self.paths.scheduler_stop.unlink(missing_ok=True)
                break
            now = time.time()
            user = source.read()
            for name in moves:
                want = name in user.active_moves
                state = moves[name]
                ramp = 0.5
                if want and state in ("inactive", "stopping"):
                    moves[name], since[name] = "starting", now
                elif not want and state in ("starting", "active"):
                    moves[name], since[name] = "stopping", now
                elif state == "starting" and now - since[name] > ramp:
                    moves[name] = "active"
                elif state == "stopping" and now - since[name] > 0.4:
                    moves[name] = "inactive"
            walking = "active" in (moves["walk"], moves["skate"])
            propup_status = None
            if moves["propup"] == "active":
                end, hold, up = 8.86, 5.2, 3.5
                down = "propup_down" in user.active_moves
                propup_t = 0.0 if propup_t is None or (propup_t >= end and not down) else propup_t
                stop = end if down or propup_t > hold else hold
                propup_t = min(stop, propup_t + 0.05)
                phase = ("lying" if propup_t >= end else "lowering") if propup_t > hold else "up" if propup_t >= up else "rising"
                propup_status = {"phase": phase, "t_s": round(propup_t, 2), "duration_s": end, "refused": False}
            elif moves["propup"] == "inactive":
                propup_t = None
            chill_name = next((n for n in CHILL_CLIPS if moves[n] == "active"), None)
            chill_status = None
            chill_targets: dict[str, float] = {}
            if chill_name is not None:
                clip = _chill_clip(chill_name)
                t = now - since[chill_name] - 0.5
                frame = int(max(0.0, t - 1.5) * clip["fps"])
                n, loop = len(clip["q"]), clip["loop_start"]
                done = loop >= n and frame >= n - 1  # played once (chill_export.py --once)
                frame = frame if frame < n else n - 1 if loop >= n else loop + (frame - loop) % (n - loop)
                if "chill_settle" in user.active_moves:
                    chill_settle = chill_settle or now
                    phase, frame = ("settled" if now - chill_settle > 1.2 else "settling"), 0
                else:
                    chill_settle = None
                    phase = "easing" if t < 1.5 else "done" if done else "playing"
                chill_targets = dict(zip(clip["names"], clip["q"][frame].tolist()))
                chill_status = {"phase": phase, "section": clip["section"](frame), "t_s": round(frame / clip["fps"], 2),
                                "loop_s": round((n - loop) / clip["fps"], 2), "refused": False}
            speed = abs(user.velocity["vx"]) + abs(user.velocity["vy"]) + 0.3 * abs(user.velocity["vtheta"])
            current = 2.2 + (3.0 + 6.0 * speed if walking else 0.0)
            self._voltage = max(6.2, self._voltage - (0.00002 + 0.00001 * current))
            sway = math.sin((now - t0) * 6.0) * (2.0 if walking else 0.3)
            phase = (now - t0) * 2 * math.pi * 1.8
            upper_targets: dict[str, float] = {}
            upper_status = None
            if moves["upper"] != "inactive":
                cmd = user.upper or {}
                if cmd.get("seq") != upper_seq:
                    upper_seq = cmd.get("seq")
                    if now - float(cmd.get("gesture_ts") or 0.0) < 2.0:
                        upper.command(cmd, now)
                manual = {k: float(v) for k, v in (cmd.get("manual") or {}).items() if k in gestures.REST}
                still = abs(user.velocity["vx"]) + abs(user.velocity["vy"]) < 0.02
                limp = bool(cmd.get("limp"))
                if limp:
                    if not held:
                        upper.stop()
                        held = gestures.to_joints(gestures.REST)
                    for i, name in enumerate(gestures.UPPER_JOINTS):
                        held[name] += 0.004 * math.sin((now - t0) * 0.7 + i)
                    upper_targets = gestures.to_joints(gestures.from_joints(held))
                else:
                    held = {}
                    groove = cmd.get("groove") if isinstance(cmd.get("groove"), dict) else None
                    pose = upper.update(now, 0.05, gestures.REST, gestures.UpperBodyInput(
                        manual, bool(cmd.get("auto", True)) and still, groove=groove, wall=time.time()))
                    upper_targets = gestures.to_joints(pose)
                upper_status = {"gesture": upper.playing, "face": upper.face, "auto": bool(cmd.get("auto", True)),
                                "manual": sorted(manual),
                                "weight": 1.0, "scale": 1.0, "seq": upper_seq, "limp": limp,
                                "groove": round(upper.groove.weight, 2), "groove_move": upper.groove.move,
                                "groove_section": upper.groove.section}
            joints = {}
            for name in JOINT_IDS:
                target = chill_targets.get(name, upper_targets.get(name, 0.0))
                if walking and ("hip_pitch" in name or "knee" in name or "ankle_pitch" in name):
                    sign = 1.0 if name.startswith("right") else -1.0
                    gain = {"hip_pitch": -0.35, "knee": 0.7, "ankle_pitch": -0.35}[name.split("_", 1)[1]]
                    target = sign * gain * max(0.0, math.sin(phase + (0.0 if sign > 0 else math.pi)))
                pos = target + random.gauss(0.0, 0.006) - 0.02 * math.copysign(1.0, target) * (target != 0.0)
                joints[name] = [round(pos, 4), round(target, 4), round(random.gauss(0.0, 0.05), 3), None, None]
            roll, pitch = math.radians(sway), math.radians(sway * 0.5)
            if user.imu_calib_seq != calib_seq:
                calib_seq = user.imu_calib_seq
                calib_start = now if calib_seq and now - calib_seq < 10.0 else None
            if calib_start is not None:
                standing = all(state == "inactive" for name, state in moves.items() if name != "head")
                if not standing:
                    calib_result, calib_start = {"seq": calib_seq, "ok": False, "error": "not_standing"}, None
                elif now - calib_start >= 3.0:
                    calib_result = {"seq": calib_seq, "ok": True, "error": "", "roll_deg": 1.84, "pitch_deg": -2.31,
                                    "lean_right": -1.84, "lean_fwd": -2.31, "ref_right": 0.0, "ref_fwd": 0.0}
                    calib_start = None
            self._write(
                {
                    "ts": now,
                    "pid": self._pid,
                    "running": True,
                    "moves": dict(moves),
                    "active_moves": sorted(user.active_moves),
                    "velocity": {k: round(v, 3) for k, v in user.velocity.items()},
                    "current_a": round(current + random.uniform(-0.3, 0.3), 2),
                    "voltage_v": round(self._voltage, 2),
                    "supply_v": round(self._voltage, 2),
                    "kp_written": round(125 * min(max(7.0 / self._voltage, 0.85), 1.25 if speed > 0 else 1.0))
                    if user.kp_vin_comp and self._voltage > 0 else 125,
                    "imu_trim_deg": [0.0, 0.0],
                    "imu_mount_deg": [user.imu_mount_roll_deg, user.imu_mount_pitch_deg],
                    "imu_calib": calib_result,
                    "imu_calib_running": calib_start is not None,
                    "roll_deg": round(sway, 1),
                    "pitch_deg": round(sway * 0.5, 1),
                    "loop_ms": round(random.uniform(9.0, 13.0), 2),
                    "overruns": 0,
                    "joints": joints,
                    **({"upper": upper_status} if upper_status else {}),
                    **({"propup": propup_status} if propup_status else {}),
                    **({chill_name: chill_status} if chill_status else {}),
                    "gyro": [round(random.gauss(0.0, 0.02) + (0.6 * math.cos(phase) if walking else 0.0), 4),
                             round(random.gauss(0.0, 0.02), 4), round(random.gauss(0.0, 0.01), 4)],
                    "quat": [round(math.cos(roll / 2) * math.cos(pitch / 2), 5), round(math.sin(roll / 2) * math.cos(pitch / 2), 5),
                             round(math.cos(roll / 2) * math.sin(pitch / 2), 5), round(-math.sin(roll / 2) * math.sin(pitch / 2), 5)],
                }
            )
            if self._loop_stop.wait(0.05):
                reason = "interrupted"
                break
        self._write({"ts": time.time(), "pid": self._pid, "running": False, "stop_reason": reason})
        with self.paths.log("run").open("a", encoding="utf-8") as log:
            log.write("[mock] Torque disabled on all motors\n")

    def upper_pid(self) -> int | None:
        return 1 if self._upper else None

    def start_upper(self) -> None:
        self._upper = True

    def stop_upper(self) -> None:
        self._upper = False
        self.upper_limp = set()

    def send_upper(self, targets_rad: dict[int, float], limp: list[int] | None = None) -> None:
        self.upper_targets.update(targets_rad)
        if limp is not None:
            self.upper_limp = set(limp)

    def upper_state(self) -> dict[str, Any] | None:
        if not self._upper:
            return None
        # Limp motors sag a little under gravity, as if moved by hand.
        positions = {str(k): v + (random.gauss(0.0, 0.01) if k in self.upper_limp else 0.0)
                     for k, v in self.upper_targets.items()}
        return {"running": True, "positions": positions, "limp": sorted(self.upper_limp), "ts": time.time()}

    def display_running(self) -> bool:
        return True

    def ensure_display(self) -> bool:
        return True

    def stop_display(self) -> None:
        pass

    def camera_port(self) -> int | None:
        return None

    def power(self, action: str) -> str:
        return f"模拟模式：忽略 {action}"
