#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

"""Linked MuJoCo + real robot joint slider comparison tool.

Run from the xiaoxiaobai repository on the Ubuntu machine:

    PYTHONPATH=src uv run --group sim src/debug/mujoco_joint_slider_compare.py --host xiaoxiaobai

The script opens a local MuJoCo viewer and a Tk slider window. Moving a slider
sets the same named joint angle in MuJoCo and on the Raspberry Pi robot, then
reads the real motor position back and displays target/real/error in degrees.
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Iterable, Mapping

import mujoco
import mujoco.viewer

try:
    import tkinter as tk
except ImportError as exc:  # pragma: no cover - depends on local desktop setup
    raise SystemExit("tkinter is required for the slider UI.") from exc

from constants import KP_DEFAULT, MOTOR_SERIAL_PORT, MOTOR_SIGN, MOTOR_TO_ID, NEUTRAL_POSE, PRESENT_CURRENT_UNIT_A
from neutral_offset import NEUTRAL_OFFSET_PATH, load_neutral_offsets, save_neutral_offsets


XIAOXIAOBAI_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MJCF = XIAOXIAOBAI_REPO_ROOT / "src/model/mjcf/scene.xml"
DEFAULT_ROOT_Z = 0.181
DEFAULT_SLIDER_LIMIT_DEG = 30.0


REMOTE_CODE_TEMPLATE = r"""
import json
import os
import sys
import time
from pathlib import Path

from rustypot import Xl330PyController

MOTOR_SERIAL_PORT = __MOTOR_SERIAL_PORT__
MOTOR_TO_ID = __MOTOR_TO_ID__
MOTOR_SIGN = __MOTOR_SIGN__
KP_DEFAULT = __KP_DEFAULT__
PRESENT_CURRENT_UNIT_A = __PRESENT_CURRENT_UNIT_A__
STARTUP_RAMP_SECONDS = __STARTUP_RAMP_SECONDS__
HOME_TARGETS = __HOME_TARGETS__
PID_FILE = Path("/tmp/xiaoxiaobai_scheduler.pid")


def emit(payload):
    print(json.dumps(payload), flush=True)


def another_session_running():
    if not PID_FILE.exists():
        return False
    try:
        pid = int(PID_FILE.read_text(encoding="ascii").strip())
        os.kill(pid, 0)
    except (ValueError, OSError):
        return False
    return True


if another_session_running():
    emit({
        "type": "error",
        "message": "A xiaoxiaobai control loop is already running. Run `make stop` first.",
    })
    sys.exit(2)


controller = Xl330PyController(serial_port=MOTOR_SERIAL_PORT, baudrate=1_000_000, timeout=0.1)
all_ids = list(MOTOR_TO_ID.values())
id_to_name = {motor_id: name for name, motor_id in MOTOR_TO_ID.items()}
id_to_sign = {motor_id: float(MOTOR_SIGN[name]) for name, motor_id in MOTOR_TO_ID.items()}


def write_named_targets(targets):
    ids = []
    positions = []
    for name, target in targets.items():
        if name not in MOTOR_TO_ID:
            continue
        motor_id = MOTOR_TO_ID[name]
        ids.append(motor_id)
        positions.append(float(target) * id_to_sign[motor_id])
    if ids:
        controller.sync_write_goal_position(ids, positions)


def scalar(value):
    if isinstance(value, (list, tuple)):
        return float(value[0])
    return float(value)


def read_named_positions():
    raw_positions = controller.sync_read_present_position(all_ids)
    if len(raw_positions) != len(all_ids):
        raise RuntimeError(
            "Expected {} motor positions, got {}".format(len(all_ids), len(raw_positions))
        )
    return {
        id_to_name[motor_id]: float(position) * id_to_sign[motor_id]
        for motor_id, position in zip(all_ids, raw_positions)
    }


def read_named_currents():
    try:
        raw_currents = controller.sync_read_present_current(all_ids)
    except Exception:
        return {}
    return {
        id_to_name[motor_id]: scalar(current) * PRESENT_CURRENT_UNIT_A
        for motor_id, current in zip(all_ids, raw_currents)
    }


def read_named_voltages():
    try:
        raw_voltages = controller.sync_read_present_input_voltage(all_ids)
    except Exception:
        return {}
    return {
        id_to_name[motor_id]: scalar(voltage)
        for motor_id, voltage in zip(all_ids, raw_voltages)
    }


def read_named_eeprom_limits():
    # Position mode clamps goal positions to these EEPROM registers.
    try:
        mins = controller.sync_read_min_position_limit(all_ids)
        maxs = controller.sync_read_max_position_limit(all_ids)
        raw_mins = controller.sync_read_raw_min_position_limit(all_ids)
        raw_maxs = controller.sync_read_raw_max_position_limit(all_ids)
        modes = controller.sync_read_operating_mode(all_ids)
    except Exception as exc:
        return {"error": str(exc)}
    return {
        id_to_name[motor_id]: {
            "min": scalar(lo), "max": scalar(hi), "raw_min": int(scalar(raw_lo)), "raw_max": int(scalar(raw_hi)),
            "mode": int(scalar(mode)),
        }
        for motor_id, lo, hi, raw_lo, raw_hi, mode in zip(all_ids, mins, maxs, raw_mins, raw_maxs, modes)
    }


def read_named_state():
    return {
        "positions": read_named_positions(),
        "currents": read_named_currents(),
        "voltages": read_named_voltages(),
    }


def ramp_named_targets(start_targets, end_targets, duration_s):
    if duration_s <= 0.0:
        write_named_targets(end_targets)
        return

    steps = max(1, int(duration_s * 50.0))
    delay = duration_s / steps
    for step in range(steps + 1):
        alpha = step / steps
        targets = {
            name: start + (float(end_targets.get(name, 0.0)) - start) * alpha
            for name, start in start_targets.items()
        }
        write_named_targets(targets)
        time.sleep(delay)


torque_enabled = False
last_targets = None

try:
    controller.sync_write_status_return_level(all_ids, [1] * len(all_ids))
    controller.sync_write_position_p_gain(all_ids, [KP_DEFAULT] * len(all_ids))

    current_targets = read_named_positions()
    write_named_targets(current_targets)
    controller.sync_write_torque_enable(all_ids, [True] * len(all_ids))
    torque_enabled = True
    last_targets = dict(current_targets)

    home_targets = {name: float(HOME_TARGETS.get(name, 0.0)) for name in current_targets}
    ramp_named_targets(current_targets, home_targets, STARTUP_RAMP_SECONDS)
    last_targets = dict(home_targets)

    emit({"type": "ready", "eeprom": read_named_eeprom_limits(), **read_named_state()})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        message = json.loads(line)
        message_type = message.get("type")

        if message_type == "targets":
            last_targets = dict(message["targets"])
            write_named_targets(last_targets)
            emit({"type": "positions", **read_named_state()})
        elif message_type == "set_home":
            incoming = message.get("targets") or {}
            for name, value in incoming.items():
                HOME_TARGETS[str(name)] = float(value)
        elif message_type == "read":
            emit({"type": "positions", **read_named_state()})
        elif message_type == "stop":
            break
finally:
    try:
        if torque_enabled:
            zero = {name: float(HOME_TARGETS.get(name, 0.0)) for name in MOTOR_TO_ID}
            if last_targets is not None:
                ramp_named_targets(last_targets, zero, STARTUP_RAMP_SECONDS)
            else:
                write_named_targets(zero)
                time.sleep(0.2)
    finally:
        if torque_enabled:
            controller.sync_write_torque_enable(all_ids, [False] * len(all_ids))
    emit({"type": "stopped"})
"""


def build_remote_code(
    startup_ramp_seconds: float,
    home_targets: Mapping[str, float] | None = None,
    motor_signs: Mapping[str, float] | None = None,
) -> str:
    return (
        REMOTE_CODE_TEMPLATE.replace("__MOTOR_SERIAL_PORT__", repr(MOTOR_SERIAL_PORT))
        .replace("__MOTOR_TO_ID__", repr(dict(MOTOR_TO_ID)))
        .replace("__MOTOR_SIGN__", repr(dict(motor_signs or MOTOR_SIGN)))
        .replace("__KP_DEFAULT__", repr(int(KP_DEFAULT)))
        .replace("__PRESENT_CURRENT_UNIT_A__", repr(float(PRESENT_CURRENT_UNIT_A)))
        .replace("__STARTUP_RAMP_SECONDS__", repr(float(startup_ramp_seconds)))
        .replace("__HOME_TARGETS__", repr(dict(home_targets or {})))
    )


class PiMotorBridge:
    """Line-delimited JSON bridge to a small Python process running on the Pi."""

    def __init__(
        self,
        host: str,
        startup_ramp_seconds: float,
        ready_timeout_s: float = 30.0,
        home_targets: Mapping[str, float] | None = None,
        motor_signs: Mapping[str, float] | None = None,
    ) -> None:
        self._write_lock = threading.Lock()
        self._messages: queue.Queue[dict] = queue.Queue()
        self._ready = threading.Event()
        self._stopped = threading.Event()
        self._last_positions: dict[str, float] = {}
        self._last_currents: dict[str, float] = {}
        self._last_voltages: dict[str, float] = {}
        self._error: str | None = None
        self.ready_info: dict = {}

        code_arg = shlex.quote(
            build_remote_code(
                startup_ramp_seconds,
                home_targets=home_targets,
                motor_signs=motor_signs,
            )
        )
        remote_shell = (
            'if [ -x "$HOME/xiaoxiaobai/.venv/bin/python" ]; then '
            f'cd "$HOME/xiaoxiaobai" && PYTHONPATH=src exec .venv/bin/python -u -c {code_arg}; '
            "elif command -v python3 >/dev/null 2>&1; then "
            f"exec python3 -u -c {code_arg}; "
            "elif command -v python >/dev/null 2>&1; then "
            f"exec python -u -c {code_arg}; "
            "else "
            "echo 'No Python interpreter found on Pi' >&2; exit 127; "
            "fi"
        )
        self._proc = subprocess.Popen(
            ["ssh", host, f"bash -lc {shlex.quote(remote_shell)}"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        self._reader = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader.start()
        self._wait_ready(ready_timeout_s)

    @property
    def last_positions(self) -> dict[str, float]:
        return dict(self._last_positions)

    @property
    def last_currents(self) -> dict[str, float]:
        return dict(self._last_currents)

    @property
    def last_voltages(self) -> dict[str, float]:
        return dict(self._last_voltages)

    def send_targets(self, targets: dict[str, float]) -> None:
        self._send_message({"type": "targets", "targets": targets})

    def set_home(self, targets: dict[str, float]) -> None:
        self._send_message({"type": "set_home", "targets": targets})

    def request_positions(self) -> None:
        self._send_message({"type": "read"})

    def close(self) -> None:
        if self._proc.poll() is None:
            try:
                self._send_message({"type": "stop"})
            except RuntimeError:
                pass
        try:
            self._proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            self._proc.terminate()

    def _reader_loop(self) -> None:
        assert self._proc.stdout is not None
        for raw_line in self._proc.stdout:
            line = raw_line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                print(f"[Pi] {line}", file=sys.stderr)
                continue

            message_type = message.get("type")
            if message_type in ("ready", "positions"):
                positions = message.get("positions")
                if isinstance(positions, dict):
                    self._last_positions = {str(name): float(value) for name, value in positions.items()}
                currents = message.get("currents")
                if isinstance(currents, dict):
                    self._last_currents = {str(name): float(value) for name, value in currents.items()}
                voltages = message.get("voltages")
                if isinstance(voltages, dict):
                    self._last_voltages = {str(name): float(value) for name, value in voltages.items()}
            if message_type == "ready":
                self.ready_info = message
                self._ready.set()
            elif message_type == "error":
                self._error = str(message.get("message", "unknown remote error"))
                self._ready.set()
            elif message_type == "stopped":
                self._stopped.set()
            self._messages.put(message)

    def _wait_ready(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._ready.wait(timeout=0.05):
                if self._error is not None:
                    raise RuntimeError(self._error)
                return
            if self._proc.poll() is not None:
                raise RuntimeError(f"Pi bridge exited with code {self._proc.returncode}.")
        raise TimeoutError("Timed out waiting for the Pi motor bridge to become ready.")

    def _send_message(self, message: dict) -> None:
        if self._proc.poll() is not None:
            raise RuntimeError(f"Pi bridge exited with code {self._proc.returncode}.")
        assert self._proc.stdin is not None
        with self._write_lock:
            self._proc.stdin.write(json.dumps(message) + "\n")
            self._proc.stdin.flush()


class JointCompareApp:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        viewer,
        joint_names: Iterable[str],
        slider_limit_deg: float | None,
        bridge: PiMotorBridge | None,
        read_interval_ms: int,
        home_targets: Mapping[str, float] | None = None,
        neutral_offsets: Mapping[str, float] | None = None,
        root_z: float = DEFAULT_ROOT_Z,
        title: str = "Xiaoxiaobai joint slider compare",
        flipped_names: set[str] | None = None,
        intro: str | None = None,
        annotations: Mapping[str, str] | None = None,
    ) -> None:
        self._model = model
        self._data = data
        self._viewer = viewer
        self._bridge = bridge
        self._home_targets = {name: float((home_targets or {}).get(name, 0.0)) for name in joint_names}
        self._targets = dict(self._home_targets)
        self._neutral_offsets = {
            name: float((neutral_offsets or {}).get(name, 0.0)) for name in joint_names
        }
        self._capture_enabled = max((abs(value) for value in self._home_targets.values()), default=0.0) < 1e-3
        self._root_z = root_z
        self._joint_qpos: dict[str, int] = {}
        self._actuator_ids: dict[str, int] = {}
        self._sliders: dict[str, tk.Scale] = {}
        self._target_vars: dict[str, tk.StringVar] = {}
        self._real_vars: dict[str, tk.StringVar] = {}
        self._error_vars: dict[str, tk.StringVar] = {}
        self._current_vars: dict[str, tk.StringVar] = {}
        self._voltage_vars: dict[str, tk.StringVar] = {}
        self._dirty = True
        self._closed = False
        self._read_interval_ms = read_interval_ms
        self._last_read_request = 0.0
        self._flipped_names = set(flipped_names or ())
        self._annotations = {str(name): str(text) for name, text in (annotations or {}).items()}

        for name in joint_names:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"Joint {name!r} not found in MuJoCo model.")
            self._joint_qpos[name] = int(model.jnt_qposadr[joint_id])

            actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if actuator_id >= 0:
                self._actuator_ids[name] = actuator_id

        self._root = tk.Tk()
        self._root.title(title)
        self._root.protocol("WM_DELETE_WINDOW", self.close)

        tk.Label(
            self._root,
            text=intro
            or (
                "滑块是相对已保存中立位的角度。MuJoCo 走模型角度，真机额外加上中立位偏差。\n"
                "调到左右对称后点“设置当前为新的中立位”。target/real/error 为度，current 为 A，voltage 为 V。"
            ),
            justify="left",
        ).grid(row=0, column=0, columnspan=8, padx=8, pady=(8, 4), sticky="w")

        headers = ("joint", "slider", "target", "real", "error", "current", "voltage", "")
        for column, title in enumerate(headers):
            tk.Label(self._root, text=title).grid(row=1, column=column, padx=4, sticky="w")

        for row, name in enumerate(joint_names, start=2):
            low_deg, high_deg = self._slider_range_deg(name, slider_limit_deg)
            flip_marker = "  FLIP" if name in self._flipped_names else ""
            note = self._annotations.get(name, "")
            note_text = f"  {note}" if note else ""
            tk.Label(
                self._root,
                text=(
                    f"ID {MOTOR_TO_ID[name]:02d}  {name}{flip_marker}  "
                    f"[{low_deg:+.1f}, {high_deg:+.1f}]{note_text}"
                ),
            ).grid(row=row, column=0, padx=8, sticky="w")

            target_var = tk.StringVar(value="+0.0")
            real_var = tk.StringVar(value="--")
            error_var = tk.StringVar(value="--")
            current_var = tk.StringVar(value="--")
            voltage_var = tk.StringVar(value="--")
            slider = tk.Scale(
                self._root,
                from_=low_deg,
                to=high_deg,
                resolution=0.5,
                orient=tk.HORIZONTAL,
                length=260,
                showvalue=False,
                command=lambda raw, n=name, v=target_var: self._set_target_deg(n, raw, v),
            )
            slider.grid(row=row, column=1, padx=4, sticky="ew")
            tk.Label(self._root, textvariable=target_var, width=8).grid(row=row, column=2, padx=4)
            tk.Label(self._root, textvariable=real_var, width=8).grid(row=row, column=3, padx=4)
            tk.Label(self._root, textvariable=error_var, width=8).grid(row=row, column=4, padx=4)
            tk.Label(self._root, textvariable=current_var, width=8).grid(row=row, column=5, padx=4)
            tk.Label(self._root, textvariable=voltage_var, width=8).grid(row=row, column=6, padx=4)
            tk.Button(self._root, text="zero", command=lambda n=name: self._zero_one(n)).grid(
                row=row, column=7, padx=4
            )

            self._sliders[name] = slider
            self._target_vars[name] = target_var
            self._real_vars[name] = real_var
            self._error_vars[name] = error_var
            self._current_vars[name] = current_var
            self._voltage_vars[name] = voltage_var
            slider.set(math.degrees(self._targets[name]))

        control_row = len(self._targets) + 2
        tk.Button(self._root, text="home all", command=self._home_all).grid(
            row=control_row, column=0, padx=8, pady=8, sticky="ew"
        )
        tk.Button(self._root, text="zero all", command=self._zero_all).grid(
            row=control_row, column=1, padx=8, pady=8, sticky="ew"
        )
        tk.Button(self._root, text="quit", command=self.close).grid(
            row=control_row, column=7, padx=8, pady=8, sticky="ew"
        )
        self._offset_status = tk.StringVar()
        self._refresh_offset_status()
        if self._capture_enabled:
            tk.Button(
                self._root,
                text="设置当前为新的中立位",
                command=self._capture_neutral,
            ).grid(row=control_row + 1, column=0, columnspan=2, padx=8, pady=(0, 8), sticky="ew")
        tk.Label(
            self._root,
            textvariable=self._offset_status,
            justify="left",
        ).grid(row=control_row + 1, column=2, columnspan=6, padx=8, pady=(0, 8), sticky="w")

        self._apply_targets(send_to_bridge=False)
        if self._bridge is not None:
            self._update_real_labels(
                self._bridge.last_positions,
                self._bridge.last_currents,
                self._bridge.last_voltages,
            )
        self._root.after(20, self._tick)

    def _slider_range_deg(self, name: str, slider_limit_deg: float | None) -> tuple[float, float]:
        joint_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if self._model.jnt_limited[joint_id]:
            low, high = (float(value) for value in self._model.jnt_range[joint_id])
            if slider_limit_deg is not None:
                limit = math.radians(slider_limit_deg)
                low = max(low, -limit)
                high = min(high, limit)
        else:
            limit = math.radians(slider_limit_deg if slider_limit_deg is not None else 180.0)
            low = -limit
            high = limit
        return math.degrees(low), math.degrees(high)

    def _set_target_deg(self, name: str, raw_value: str, value: tk.StringVar) -> None:
        degrees = float(raw_value)
        value.set(f"{degrees:+.1f}")
        self._targets[name] = math.radians(degrees)
        self._dirty = True

    def _zero_one(self, name: str) -> None:
        self._sliders[name].set(0.0)

    def _zero_all(self) -> None:
        for slider in self._sliders.values():
            slider.set(0.0)
        self._dirty = True

    def _home_all(self) -> None:
        for name, slider in self._sliders.items():
            slider.set(math.degrees(self._home_targets[name]))
        self._dirty = True

    def _hardware_targets(self) -> dict[str, float]:
        return {
            name: angle + self._neutral_offsets.get(name, 0.0)
            for name, angle in self._targets.items()
        }

    def _refresh_offset_status(self) -> None:
        active = [
            f"{name} {math.degrees(value):+.2f}°"
            for name, value in self._neutral_offsets.items()
            if abs(value) > 1e-6
        ]
        if active:
            text = "已保存中立位偏差: " + ", ".join(active)
        else:
            text = "已保存中立位偏差: 无"
        self._offset_status.set(text)

    def _capture_neutral(self) -> None:
        updated = dict(self._neutral_offsets)
        for name, angle in self._targets.items():
            updated[name] = updated.get(name, 0.0) + angle
        path = save_neutral_offsets(updated)
        self._neutral_offsets = {name: float(updated.get(name, 0.0)) for name in self._targets}
        if self._bridge is not None:
            self._bridge.set_home(dict(self._neutral_offsets))
        for slider in self._sliders.values():
            slider.set(0.0)
        self._dirty = True
        self._refresh_offset_status()
        print(f"Saved neutral offsets to {path}", flush=True)
        for name, value in self._neutral_offsets.items():
            if abs(value) > 1e-6:
                print(f"  {name}: {math.degrees(value):+.2f} deg", flush=True)

    def _apply_targets(self, send_to_bridge: bool = True) -> None:
        if self._data.qpos.size >= 3:
            self._data.qpos[2] = self._root_z
        for name, angle in self._targets.items():
            self._data.qpos[self._joint_qpos[name]] = angle
            actuator_id = self._actuator_ids.get(name)
            if actuator_id is not None:
                self._data.ctrl[actuator_id] = angle
        mujoco.mj_forward(self._model, self._data)
        self._viewer.sync()
        if send_to_bridge and self._bridge is not None:
            self._bridge.send_targets(self._hardware_targets())

    def _update_real_labels(
        self,
        positions: dict[str, float],
        currents: dict[str, float] | None = None,
        voltages: dict[str, float] | None = None,
    ) -> None:
        currents = currents or {}
        voltages = voltages or {}
        for name, target in self._targets.items():
            if name in positions:
                real = positions[name] - self._neutral_offsets.get(name, 0.0)
                error = real - target
                self._real_vars[name].set(f"{math.degrees(real):+.1f}")
                self._error_vars[name].set(f"{math.degrees(error):+.1f}")
            if name in currents:
                self._current_vars[name].set(f"{currents[name]:+.3f}")
            if name in voltages:
                self._voltage_vars[name].set(f"{voltages[name]:.2f}")

    def _tick(self) -> None:
        if self._closed:
            return
        if not self._viewer.is_running():
            self.close()
            return

        try:
            if self._dirty:
                self._dirty = False
                self._apply_targets()

            if self._bridge is not None:
                now = time.monotonic()
                if (now - self._last_read_request) * 1000.0 >= self._read_interval_ms:
                    self._last_read_request = now
                    self._bridge.request_positions()
                self._update_real_labels(
                    self._bridge.last_positions,
                    self._bridge.last_currents,
                    self._bridge.last_voltages,
                )
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            self.close()
            return

        self._root.after(20, self._tick)

    def run(self) -> None:
        self._root.mainloop()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._targets = dict(self._home_targets)
            self._apply_targets(send_to_bridge=False)
        except Exception as exc:
            print(f"Warning: failed to reset MuJoCo targets: {exc}", file=sys.stderr)
        if self._bridge is not None:
            self._bridge.close()
        try:
            self._viewer.close()
        finally:
            self._root.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="xiaoxiaobai", help="SSH host for the Raspberry Pi.")
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF, help="Path to the MuJoCo scene XML.")
    parser.add_argument(
        "--limit-deg",
        type=float,
        default=DEFAULT_SLIDER_LIMIT_DEG,
        help=(
            "Optional symmetric slider limit around zero before joint-range clipping. "
            "Use 0 or a negative value for full MJCF limits."
        ),
    )
    parser.add_argument(
        "--startup-ramp-seconds",
        type=float,
        default=2.0,
        help="Seconds used to ramp the real robot to zero on start and shutdown.",
    )
    parser.add_argument(
        "--read-interval-ms",
        type=int,
        default=150,
        help="How often to ask the real robot for present positions.",
    )
    parser.add_argument(
        "--no-robot",
        action="store_true",
        help="Open the MuJoCo sliders without connecting to the Pi.",
    )
    parser.add_argument(
        "--flip",
        action="append",
        default=[],
        metavar="JOINT",
        help=(
            "Temporarily flip the real-robot motor sign for this joint in the slider test only. "
            "Can be passed multiple times; does not modify constants.py."
        ),
    )
    return parser.parse_args()


def pi_connect_error(host: str, exc: BaseException) -> str:
    """SSH itself may have worked; only say 'could not connect' when it did not."""
    text = str(exc)
    if "already running" in text:
        return (
            f"{text}\n"
            "SSH reached the Pi. This tool stopped because that control loop already owns the motors, "
            "and the two cannot share the serial port.\n"
            f"Stop it first, then run this again: make stop HOST={host}"
        )
    return (
        f"Could not connect to the Pi over SSH using host {host!r}: {exc}\n"
        "Make sure the Ubuntu computer and Pi are on the same network, then pass the reachable SSH host, "
        "for example: --host xiaoxiaobai.local or --host xiaoxiaobai@192.168.43.23"
    )


def motor_signs_with_flips(flip_names: Iterable[str]) -> dict[str, float]:
    signs = dict(MOTOR_SIGN)
    unknown = sorted(set(flip_names) - set(MOTOR_TO_ID))
    if unknown:
        raise ValueError(f"Unknown --flip joint(s): {', '.join(unknown)}")
    for name in flip_names:
        signs[name] = -float(signs[name])
    return signs


def main() -> None:
    args = parse_args()
    slider_limit_deg = args.limit_deg if args.limit_deg > 0.0 else None
    flipped_names = set(args.flip)
    motor_signs = motor_signs_with_flips(flipped_names)

    model = mujoco.MjModel.from_xml_path(str(args.mjcf))
    data = mujoco.MjData(model)
    joint_names = []
    skipped_names = []

    for name, _motor_id in sorted(MOTOR_TO_ID.items(), key=lambda item: item[1]):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id >= 0:
            joint_names.append(name)
        else:
            skipped_names.append(name)

    for name, angle in NEUTRAL_POSE.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id >= 0:
            data.qpos[model.jnt_qposadr[joint_id]] = angle
    if data.qpos.size >= 3:
        data.qpos[2] = DEFAULT_ROOT_Z
    mujoco.mj_forward(model, data)

    print(f"Loaded MJCF: {args.mjcf.resolve()}")
    print(f"Slider safety limit: {slider_limit_deg} deg" if slider_limit_deg else "Slider safety limit: full MJCF")
    if flipped_names:
        print("Temporary real-robot sign flips:", ", ".join(sorted(flipped_names)))
    if skipped_names:
        print("Skipped MOTOR_TO_ID entries with no MJCF joint:", ", ".join(skipped_names))

    neutral_offsets = load_neutral_offsets()
    if neutral_offsets:
        print(f"Loaded neutral offsets from {NEUTRAL_OFFSET_PATH}")
        for name, value in sorted(neutral_offsets.items()):
            if abs(value) > 1e-6:
                print(f"  {name}: {math.degrees(value):+.2f} deg")
    else:
        print(f"No neutral offsets at {NEUTRAL_OFFSET_PATH}")

    # Connect before opening the viewer: closing the viewer on the way out of a refused
    # connection segfaults (make reports error 139).
    bridge = None
    if not args.no_robot:
        print(f"Connecting to Pi over SSH: {args.host}")
        print("If prompted, enter the Pi password in this terminal.")
        try:
            bridge = PiMotorBridge(
                args.host,
                startup_ramp_seconds=args.startup_ramp_seconds,
                home_targets={name: float(neutral_offsets.get(name, 0.0)) for name in joint_names},
                motor_signs=motor_signs,
            )
        except Exception as exc:
            raise SystemExit(pi_connect_error(args.host, exc)) from exc

    print("Opening MuJoCo viewer...")
    viewer = mujoco.viewer.launch_passive(model, data)

    app = JointCompareApp(
        model=model,
        data=data,
        viewer=viewer,
        joint_names=joint_names,
        slider_limit_deg=slider_limit_deg,
        bridge=bridge,
        read_interval_ms=args.read_interval_ms,
        neutral_offsets=neutral_offsets,
        title=f"Xiaoxiaobai joint slider compare - {args.mjcf.name}",
        flipped_names=flipped_names,
    )
    app.run()


if __name__ == "__main__":
    main()
