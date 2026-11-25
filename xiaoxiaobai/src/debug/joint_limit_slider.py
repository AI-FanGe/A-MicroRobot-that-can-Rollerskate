#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

"""Measure joint limits for the get-up training with unlimited sliders on MuJoCo and the real robot.

The MuJoCo window is the training model. Sliders are training-model joint angles over the
full motor turn, not clipped to the MJCF ranges; the same angle goes to the real robot
(through MOTOR_SIGN and the saved neutral offsets). Drag a joint until it touches the
mechanical stop or another part, then press 记下限 / 记上限: the slider angle and the
measured real angle are saved to calibration/getup_joint_limits.json. The walk keeps its
own limits; these are only for the get-up task.

    make joint-limits HOST=user@192.168.2.118
    PYTHONPATH=src uv run --group sim src/debug/joint_limit_slider.py --no-robot
"""

from __future__ import annotations

import argparse
import json
import math
import time
import tkinter as tk
from pathlib import Path

import mujoco
import mujoco.viewer

import mujoco_joint_slider_compare as msc
from constants import KP_DEFAULT, MOTOR_SIGN, MOTOR_TO_ID, NEUTRAL_POSE
from mujoco_joint_slider_compare import DEFAULT_ROOT_Z, XIAOXIAOBAI_REPO_ROOT, JointCompareApp, PiMotorBridge
from neutral_offset import load_neutral_offsets
from training_direction_slider import TRAINING_SCENE

LIMITS_PATH = XIAOXIAOBAI_REPO_ROOT / "calibration/getup_joint_limits.json"
SLIDER_SPAN_DEG = 180.0
CURRENT_WARN_A = 0.8

INTRO = (
    "关节限位测量（只用于摔倒爬起训练，步态不改）。滑块 = 训练模型关节角，全行程 ±180°，不受模型限位裁剪。\n"
    "拖到碰到机械限位或别的零件为止，点“记下限/记上限”。保存滑块角度和真机实测角度。\n"
    "“训练范围”列变红 = 超出当前训练模型的限位。电流长时间超过 "
    f"{CURRENT_WARN_A} A 说明顶住了，退回一点。真机最好悬空，每次只动一个关节。"
)


def eeprom_in_model_frame(entry: dict, sign: float, offset: float) -> tuple[float, float]:
    """EEPROM position limits [rad, motor frame] -> training-model joint angles [deg]."""
    lo, hi = sorted((entry["min"] * sign, entry["max"] * sign))
    return math.degrees(lo - offset), math.degrees(hi - offset)


class JointLimitApp(JointCompareApp):
    def __init__(self, *args, limits: dict, eeprom: dict, **kwargs) -> None:
        self._limits = limits
        self._eeprom = eeprom
        super().__init__(*args, **kwargs)
        names = list(self._targets)
        self._range_vars: dict[str, tk.StringVar] = {}
        self._range_labels: dict[str, tk.Label] = {}
        self._limit_vars: dict[str, tk.StringVar] = {}

        tk.Label(self._root, text="训练范围").grid(row=1, column=8, padx=4, sticky="w")
        tk.Label(self._root, text="已记录 下限 / 上限 (滑块|实测)").grid(row=1, column=13, padx=4, sticky="w")
        for row, name in enumerate(names, start=2):
            self._sliders[name].configure(length=420)
            self._range_vars[name] = tk.StringVar()
            self._range_labels[name] = tk.Label(self._root, textvariable=self._range_vars[name], width=16)
            self._range_labels[name].grid(row=row, column=8, padx=4)
            tk.Button(self._root, text="-1°", command=lambda n=name: self._nudge(n, -1.0)).grid(row=row, column=9)
            tk.Button(self._root, text="+1°", command=lambda n=name: self._nudge(n, +1.0)).grid(row=row, column=10)
            tk.Button(self._root, text="记下限", command=lambda n=name: self._record(n, "low")).grid(row=row, column=11)
            tk.Button(self._root, text="记上限", command=lambda n=name: self._record(n, "high")).grid(row=row, column=12)
            self._limit_vars[name] = tk.StringVar()
            tk.Label(self._root, textvariable=self._limit_vars[name], width=34, anchor="w").grid(
                row=row, column=13, padx=4, sticky="w")
            self._refresh_range(name)
            self._refresh_limit_label(name)

        status_row = len(names) + 4
        self._collision_var = tk.StringVar()
        tk.Label(self._root, textvariable=self._collision_var, justify="left", fg="#b00000").grid(
            row=status_row, column=0, columnspan=14, padx=8, sticky="w")
        self._current_var = tk.StringVar()
        tk.Label(self._root, textvariable=self._current_var, justify="left", fg="#b00000").grid(
            row=status_row + 1, column=0, columnspan=14, padx=8, pady=(0, 8), sticky="w")
        self._viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        self._refresh_collisions()

    def _slider_range_deg(self, name: str, slider_limit_deg: float | None) -> tuple[float, float]:
        return -SLIDER_SPAN_DEG, SLIDER_SPAN_DEG

    def _set_target_deg(self, name: str, raw_value: str, value: tk.StringVar) -> None:
        super()._set_target_deg(name, raw_value, value)
        if hasattr(self, "_range_vars"):
            self._refresh_range(name)

    def _apply_targets(self, send_to_bridge: bool = True) -> None:
        super()._apply_targets(send_to_bridge)
        if hasattr(self, "_collision_var"):
            self._refresh_collisions()

    def _update_real_labels(self, positions, currents=None, voltages=None) -> None:
        super()._update_real_labels(positions, currents, voltages)
        if not hasattr(self, "_current_var"):
            return
        high = [f"{n} {a:+.2f} A" for n, a in (currents or {}).items() if n in self._targets and abs(a) > CURRENT_WARN_A]
        self._current_var.set(f"电流过大（顶住了？）: {', '.join(high)}" if high else "")

    def _nudge(self, name: str, delta_deg: float) -> None:
        degrees = max(-SLIDER_SPAN_DEG, min(SLIDER_SPAN_DEG, math.degrees(self._targets[name]) + delta_deg))
        self._sliders[name].set(degrees)
        self._set_target_deg(name, str(degrees), self._target_vars[name])

    def _refresh_range(self, name: str) -> None:
        joint_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
        low, high = (math.degrees(v) for v in self._model.jnt_range[joint_id])
        self._range_vars[name].set(f"[{low:+.0f}, {high:+.0f}]")
        target = math.degrees(self._targets[name])
        outside = target < low - 0.25 or target > high + 0.25
        self._range_labels[name].configure(fg="#d00000" if outside else "black")

    def _refresh_collisions(self) -> None:
        pairs = {}
        for contact in self._data.contact[: self._data.ncon]:
            b1, b2 = self._model.geom_bodyid[contact.geom1], self._model.geom_bodyid[contact.geom2]
            if b1 == 0 or b2 == 0 or contact.dist >= 0.0:
                continue
            key = tuple(sorted(mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_BODY, b) for b in (b1, b2)))
            pairs[key] = min(pairs.get(key, 0.0), contact.dist)
        text = ", ".join(f"{a} <-> {b} {-d * 1000:.1f} mm" for (a, b), d in sorted(pairs.items()))
        self._collision_var.set(f"MuJoCo 自碰撞: {text}" if text else "MuJoCo 自碰撞: 无")

    def _record(self, name: str, side: str) -> None:
        # Re-read so entries edited outside this window since startup are kept.
        if LIMITS_PATH.exists():
            self._limits.update(json.loads(LIMITS_PATH.read_text(encoding="utf-8")))
        entry = self._limits.setdefault(name, {})
        entry[f"{side}_deg"] = round(math.degrees(self._targets[name]), 1)
        if self._bridge is not None and name in self._bridge.last_positions:
            real = self._bridge.last_positions[name] - self._neutral_offsets.get(name, 0.0)
            entry[f"{side}_real_deg"] = round(math.degrees(real), 1)
        entry[f"{side}_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        LIMITS_PATH.parent.mkdir(parents=True, exist_ok=True)
        LIMITS_PATH.write_text(json.dumps(self._limits, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        self._refresh_limit_label(name)
        print(f"{name} {side}: {entry}  -> {LIMITS_PATH}", flush=True)

    def _refresh_limit_label(self, name: str) -> None:
        entry = self._limits.get(name, {})

        def side(key: str) -> str:
            if f"{key}_deg" not in entry:
                return "--"
            real = entry.get(f"{key}_real_deg")
            return f"{entry[f'{key}_deg']:+.1f}" + (f"|{real:+.1f}" if real is not None else "")

        self._limit_vars[name].set(f"{side('low')}  /  {side('high')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="user@xiaoxiaobai.local", help="SSH host for the Raspberry Pi.")
    parser.add_argument("--mjcf", type=Path, default=TRAINING_SCENE, help="Training MuJoCo scene.")
    parser.add_argument("--kp", type=int, default=KP_DEFAULT,
                        help="Motor position P gain on the robot; lower it to push softer against the stops.")
    parser.add_argument("--startup-ramp-seconds", type=float, default=2.0)
    parser.add_argument("--read-interval-ms", type=int, default=150)
    parser.add_argument("--no-robot", action="store_true", help="Only the MuJoCo sliders, no Pi connection.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.mjcf))
    data = mujoco.MjData(model)
    joint_names = [n for n, _ in sorted(MOTOR_TO_ID.items(), key=lambda item: item[1])
                   if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) >= 0]
    home = {name: float(NEUTRAL_POSE.get(name, 0.0)) for name in joint_names}
    offsets = load_neutral_offsets()
    for name, angle in home.items():
        data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]] = angle
    data.qpos[2] = DEFAULT_ROOT_Z
    mujoco.mj_forward(model, data)

    limits = json.loads(LIMITS_PATH.read_text(encoding="utf-8")) if LIMITS_PATH.exists() else {}
    print(f"训练场景: {args.mjcf.resolve()}")
    print(f"限位文件: {LIMITS_PATH}（已有 {len(limits)} 个关节）")

    bridge = None
    eeprom: dict = {}
    if not args.no_robot:
        msc.KP_DEFAULT = args.kp
        print(f"正在通过 SSH 连接真机: {args.host}（kp {args.kp}），要密码就在这个终端输入。", flush=True)
        bridge = PiMotorBridge(
            args.host,
            startup_ramp_seconds=args.startup_ramp_seconds,
            ready_timeout_s=180.0,
            home_targets={name: home[name] + float(offsets.get(name, 0.0)) for name in joint_names},
        )
        eeprom = bridge.ready_info.get("eeprom") or {}
        if "error" in eeprom:
            print(f"读不到舵机 EEPROM 限位: {eeprom['error']}")
            eeprom = {}
        print("舵机 EEPROM 位置限位（训练模型角度，位置模式下目标会被裁到这里）:")
        for name in joint_names:
            if name in eeprom:
                lo, hi = eeprom_in_model_frame(eeprom[name], float(MOTOR_SIGN[name]), float(offsets.get(name, 0.0)))
                e = eeprom[name]
                narrow = "  <- 比滑块窄" if lo > -SLIDER_SPAN_DEG + 2 or hi < SLIDER_SPAN_DEG - 2 else ""
                print(f"  ID {MOTOR_TO_ID[name]:02d} {name:22s} [{lo:+7.1f}, {hi:+7.1f}]°  "
                      f"raw [{e['raw_min']}, {e['raw_max']}]  mode {e['mode']}{narrow}")

    viewer = mujoco.viewer.launch_passive(model, data)
    annotations = {}
    for name in joint_names:
        if name in eeprom:
            lo, hi = eeprom_in_model_frame(eeprom[name], float(MOTOR_SIGN[name]), float(offsets.get(name, 0.0)))
            annotations[name] = f"舵机 [{lo:+.0f}, {hi:+.0f}]"
    app = JointLimitApp(
        model=model,
        data=data,
        viewer=viewer,
        joint_names=joint_names,
        slider_limit_deg=None,
        bridge=bridge,
        read_interval_ms=args.read_interval_ms,
        home_targets=home,
        neutral_offsets=offsets,
        root_z=DEFAULT_ROOT_Z,
        title="关节限位测量（爬起训练）",
        intro=INTRO,
        annotations=annotations,
        limits=limits,
        eeprom=eeprom,
    )
    app.run()


if __name__ == "__main__":
    main()
