#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

"""MuJoCo + real robot motor direction checker.

Run locally on the Ubuntu machine:

    PYTHONPATH=src uv run --group sim src/debug/mujoco_motor_direction_check.py --host xiaoxiaobai

The script opens a local MuJoCo viewer and a small slider window. Moving a
slider sets the same named joint angle in MuJoCo and on the Raspberry Pi robot.
If the real joint moves opposite to MuJoCo, flip MOTOR_SIGN for that joint. If a
different joint moves, fix MOTOR_TO_ID.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import mujoco
import mujoco.viewer
import onnxruntime as ort

try:
    import tkinter as tk
except ImportError as exc:  # pragma: no cover - depends on local desktop setup
    raise SystemExit("tkinter is required for the slider UI.") from exc

from constants import KP_DEFAULT, MOTOR_SERIAL_PORT, MOTOR_SIGN, MOTOR_TO_ID, NEUTRAL_POSE


XIAOXIAOBAI_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MJCF = XIAOXIAOBAI_REPO_ROOT / "src/model/mjcf/scene.xml"
DEFAULT_ROOT_Z = 0.181


REMOTE_CODE_TEMPLATE = r"""
import json
import math
import sys
import time

from rustypot import Xl330PyController

MOTOR_SERIAL_PORT = __MOTOR_SERIAL_PORT__
MOTOR_TO_ID = __MOTOR_TO_ID__
MOTOR_SIGN = __MOTOR_SIGN__
KP_DEFAULT = __KP_DEFAULT__
STARTUP_RAMP_SECONDS = 2.0

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


def ramp_named_targets(start_targets, end_targets, duration_s):
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
    ramp_named_targets(current_targets, {name: 0.0 for name in current_targets}, STARTUP_RAMP_SECONDS)
    last_targets = {name: 0.0 for name in current_targets}
    print("REMOTE_READY", flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        message = json.loads(line)
        if message.get("type") == "targets":
            last_targets = dict(message["targets"])
            write_named_targets(last_targets)
        elif message.get("type") == "stop":
            break
finally:
    try:
        if torque_enabled:
            zero = {name: 0.0 for name in MOTOR_TO_ID}
            if last_targets is not None:
                ramp_named_targets(last_targets, zero, STARTUP_RAMP_SECONDS)
            else:
                write_named_targets(zero)
                time.sleep(0.2)
    finally:
        if torque_enabled:
            controller.sync_write_torque_enable(all_ids, [False] * len(all_ids))
    print("REMOTE_STOPPED", flush=True)
"""


def build_remote_code() -> str:
    return (
        REMOTE_CODE_TEMPLATE
        .replace("__MOTOR_SERIAL_PORT__", repr(MOTOR_SERIAL_PORT))
        .replace("__MOTOR_TO_ID__", repr(dict(MOTOR_TO_ID)))
        .replace("__MOTOR_SIGN__", repr(dict(MOTOR_SIGN)))
        .replace("__KP_DEFAULT__", repr(int(KP_DEFAULT)))
    )


class PiMotorBridge:
    """Line-delimited JSON bridge to a small Python process running on the Pi."""

    def __init__(self, host: str) -> None:
        code_arg = shlex.quote(build_remote_code())
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
            text=True,
        )

    def send_targets(self, targets: dict[str, float]) -> None:
        if self._proc.poll() is not None:
            raise RuntimeError(f"Pi bridge exited with code {self._proc.returncode}.")
        assert self._proc.stdin is not None
        self._proc.stdin.write(json.dumps({"type": "targets", "targets": targets}) + "\n")
        self._proc.stdin.flush()

    def close(self) -> None:
        if self._proc.poll() is None and self._proc.stdin is not None:
            try:
                self._proc.stdin.write(json.dumps({"type": "stop"}) + "\n")
                self._proc.stdin.flush()
            except BrokenPipeError:
                pass
        try:
            self._proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            self._proc.terminate()


class DirectionCheckApp:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        viewer,
        joint_names: Iterable[str],
        angle_limit_deg: float | None,
        policy_default_pose: dict[str, float] | None,
        bridge: PiMotorBridge | None,
    ) -> None:
        self._model = model
        self._data = data
        self._viewer = viewer
        self._bridge = bridge
        self._targets = {name: 0.0 for name in joint_names}
        self._policy_default_pose = {
            name: angle
            for name, angle in (policy_default_pose or {}).items()
            if name in self._targets
        }
        self._joint_qpos: dict[str, int] = {}
        self._sliders: dict[str, tk.Scale] = {}
        self._dirty = True
        self._closed = False

        for name in joint_names:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"Joint {name!r} not found in MuJoCo model.")
            self._joint_qpos[name] = int(model.jnt_qposadr[joint_id])

        self._root = tk.Tk()
        self._root.title("Xiaoxiaobai motor direction check")
        self._root.protocol("WM_DELETE_WINDOW", self.close)

        tk.Label(
            self._root,
            text=(
                "Move one slider at a time. Sliders are sorted by motor ID and use the MJCF joint limits.\n"
                "Compare the real robot with MuJoCo.\n"
                "Opposite direction => flip MOTOR_SIGN. Wrong joint => fix MOTOR_TO_ID."
            ),
            justify="left",
        ).grid(row=0, column=0, columnspan=4, padx=8, pady=(8, 4), sticky="w")

        for row, name in enumerate(joint_names, start=1):
            low_deg, high_deg = self._slider_range_deg(name, angle_limit_deg)
            tk.Label(self._root, text=f"ID {MOTOR_TO_ID[name]:02d}  {name}  [{low_deg:+.1f}, {high_deg:+.1f}]").grid(
                row=row, column=0, padx=8, sticky="w"
            )
            value = tk.StringVar(value="0.0 deg")
            slider = tk.Scale(
                self._root,
                from_=low_deg,
                to=high_deg,
                resolution=0.5,
                orient=tk.HORIZONTAL,
                length=260,
                showvalue=False,
                command=lambda raw, n=name, v=value: self._set_target_deg(n, raw, v),
            )
            slider.grid(row=row, column=1, padx=4, sticky="ew")
            tk.Label(self._root, textvariable=value, width=9).grid(row=row, column=2, padx=4)
            tk.Button(
                self._root,
                text="zero",
                command=lambda n=name: self._zero_one(n),
            ).grid(row=row, column=3, padx=4)
            self._sliders[name] = slider
            slider.set(0.0)

        control_row = len(self._targets) + 1
        tk.Button(self._root, text="zero all", command=self._zero_all).grid(
            row=control_row, column=0, padx=8, pady=8, sticky="ew"
        )
        if self._policy_default_pose:
            tk.Button(self._root, text="policy default", command=self._policy_default).grid(
                row=control_row, column=1, padx=8, pady=8, sticky="ew"
            )
        tk.Button(self._root, text="quit", command=self.close).grid(
            row=control_row, column=3, padx=8, pady=8, sticky="ew"
        )

        self._apply_targets()
        self._root.after(20, self._tick)

    def _slider_range_deg(self, name: str, angle_limit_deg: float | None) -> tuple[float, float]:
        joint_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if self._model.jnt_limited[joint_id]:
            low, high = (float(value) for value in self._model.jnt_range[joint_id])
            if angle_limit_deg is not None:
                limit = math.radians(angle_limit_deg)
                low = max(low, -limit)
                high = min(high, limit)
        else:
            limit = math.radians(angle_limit_deg if angle_limit_deg is not None else 180.0)
            low = -limit
            high = limit
        return math.degrees(low), math.degrees(high)

    def _set_target_deg(self, name: str, raw_value: str, value: tk.StringVar) -> None:
        degrees = float(raw_value)
        value.set(f"{degrees:+.1f} deg")
        self._targets[name] = math.radians(degrees)
        self._dirty = True

    def _zero_one(self, name: str) -> None:
        self._sliders[name].set(0.0)

    def _zero_all(self) -> None:
        for slider in self._sliders.values():
            slider.set(0.0)
        self._dirty = True

    def _policy_default(self) -> None:
        for name, angle in self._policy_default_pose.items():
            slider = self._sliders[name]
            slider.set(math.degrees(angle))
        self._dirty = True

    def _apply_targets(self, send_to_bridge: bool = True) -> None:
        self._data.qpos[2] = DEFAULT_ROOT_Z
        for name, angle in self._targets.items():
            self._data.qpos[self._joint_qpos[name]] = angle
        mujoco.mj_forward(self._model, self._data)
        self._viewer.sync()
        if send_to_bridge and self._bridge is not None:
            self._bridge.send_targets(self._targets)

    def _tick(self) -> None:
        if self._closed:
            return
        if not self._viewer.is_running():
            self.close()
            return
        if self._dirty:
            self._dirty = False
            try:
                self._apply_targets()
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
            self._targets = {name: 0.0 for name in self._targets}
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
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF, help="Path to scene.xml.")
    parser.add_argument(
        "--limit-deg",
        type=float,
        default=None,
        help="Optional symmetric slider limit around zero before joint-range clipping. Defaults to full MJCF limits.",
    )
    parser.add_argument(
        "--no-robot",
        action="store_true",
        help="Open the MuJoCo sliders without connecting to the Pi.",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=XIAOXIAOBAI_REPO_ROOT / "src/agents/walk_low_battery_1.onnx",
        help="ONNX policy whose default_joint_pos metadata can be sent with the policy default button.",
    )
    return parser.parse_args()


def load_policy_default_pose(policy_path: Path) -> dict[str, float]:
    if not policy_path.exists():
        print(f"Warning: policy not found, no policy default button: {policy_path}", file=sys.stderr)
        return {}
    session = ort.InferenceSession(str(policy_path))
    meta = session.get_modelmeta().custom_metadata_map
    if "joint_names" not in meta or "default_joint_pos" not in meta:
        print(f"Warning: policy has no default_joint_pos metadata: {policy_path}", file=sys.stderr)
        return {}
    names = meta["joint_names"].split(",")
    positions = [float(value) for value in meta["default_joint_pos"].split(",")]
    return dict(zip(names, positions))


def main() -> None:
    args = parse_args()

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
    data.qpos[2] = DEFAULT_ROOT_Z
    mujoco.mj_forward(model, data)

    print("Opening MuJoCo viewer...")
    print(f"Loaded MJCF: {args.mjcf.resolve()}")
    print("Slider joints:")
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        low, high = model.jnt_range[joint_id]
        axis = model.jnt_axis[joint_id]
        print(
            f"  ID {MOTOR_TO_ID[name]:02d}: {name} "
            f"[{math.degrees(low):+.1f}, {math.degrees(high):+.1f}] deg "
            f"axis=({axis[0]:+.0f},{axis[1]:+.0f},{axis[2]:+.0f})"
        )
    if skipped_names:
        print("Skipped MOTOR_TO_ID entries with no MJCF joint:", ", ".join(skipped_names))
    policy_default_pose = load_policy_default_pose(args.policy)
    viewer = mujoco.viewer.launch_passive(model, data)

    bridge = None
    if not args.no_robot:
        print(f"Connecting to Pi over SSH: {args.host}")
        print("If prompted, enter the Pi password in this terminal.")
        bridge = PiMotorBridge(args.host)

    app = DirectionCheckApp(
        model=model,
        data=data,
        viewer=viewer,
        joint_names=joint_names,
        angle_limit_deg=args.limit_deg,
        policy_default_pose=policy_default_pose,
        bridge=bridge,
    )
    app.run()


if __name__ == "__main__":
    main()
