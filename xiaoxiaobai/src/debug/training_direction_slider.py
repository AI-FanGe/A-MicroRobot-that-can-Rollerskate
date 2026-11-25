#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

"""Compare training-policy joint directions with the real robot.

The MuJoCo window is the model used by the current training run. Each slider
is that model's joint angle: dragging toward the positive end is the positive
direction the training policy commands. The same angle is sent to the real
robot through MOTOR_SIGN, so a joint that turns the other way on the hardware
has the wrong sign.

Run from the xiaoxiaobai repository:

    PYTHONPATH=src uv run --group sim src/debug/training_direction_slider.py --host user@xiaoxiaobai.local
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import mujoco
import mujoco.viewer

from constants import MOTOR_TO_ID, NEUTRAL_POSE
from neutral_offset import NEUTRAL_OFFSET_PATH, load_neutral_offsets

try:
    from .mujoco_joint_slider_compare import (
        DEFAULT_ROOT_Z,
        DEFAULT_SLIDER_LIMIT_DEG,
        JointCompareApp,
        PiMotorBridge,
    )
except ImportError:
    from mujoco_joint_slider_compare import (
        DEFAULT_ROOT_Z,
        DEFAULT_SLIDER_LIMIT_DEG,
        JointCompareApp,
        PiMotorBridge,
    )


TRAINING_SCENE = (
    Path(__file__).resolve().parents[3]
    / "mjlab_xiaoxiaobai"
    / "src"
    / "mjlab_xiaoxiaobai"
    / "robot"
    / "xiaoxiaobai"
    / "scene.xml"
)

INTRO = (
    "滑块角度就是当前训练模型的关节角。往正方向拖，是训练策略命令的正方向。\n"
    "MuJoCo 用这个角度摆姿态，真机收到同一角度（经过 MOTOR_SIGN，并加上已保存的中立位偏差）。\n"
    "两边转的方向不一致，就说明这个关节的 MOTOR_SIGN 和训练模型相反。每次只动一个滑块。"
)


def axis_label(model: mujoco.MjModel, name: str) -> str:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    axis = model.jnt_axis[joint_id]
    dominant = max(range(3), key=lambda index: abs(float(axis[index])))
    sign = "+" if float(axis[dominant]) >= 0.0 else "-"
    return f"训练正方向 {sign}{'xyz'[dominant]}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="user@xiaoxiaobai.local", help="SSH host for the Raspberry Pi.")
    parser.add_argument("--mjcf", type=Path, default=TRAINING_SCENE, help="Training MuJoCo scene.")
    parser.add_argument(
        "--limit-deg",
        type=float,
        default=DEFAULT_SLIDER_LIMIT_DEG,
        help="Symmetric slider limit around zero before MJCF joint-range clipping. <=0 uses full MJCF limits.",
    )
    parser.add_argument(
        "--startup-ramp-seconds",
        type=float,
        default=2.0,
        help="Seconds used to ramp the real robot to the training home pose on start and shutdown.",
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
        help="Open the training MuJoCo sliders without connecting to the Pi.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.mjcf.exists():
        raise SystemExit(f"Training scene not found: {args.mjcf}")

    slider_limit_deg = args.limit_deg if args.limit_deg > 0.0 else None
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
    if not joint_names:
        raise SystemExit(f"No MOTOR_TO_ID joints found in {args.mjcf}")

    home_targets = {name: float(NEUTRAL_POSE.get(name, 0.0)) for name in joint_names}
    neutral_offsets = load_neutral_offsets()
    hardware_home = {
        name: home_targets[name] + float(neutral_offsets.get(name, 0.0)) for name in joint_names
    }

    for name, angle in home_targets.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[joint_id]] = angle
    if data.qpos.size >= 3:
        data.qpos[2] = DEFAULT_ROOT_Z
    mujoco.mj_forward(model, data)

    print("训练仿真 vs 真机：关节正方向对照")
    print(f"训练场景: {args.mjcf.resolve()}")
    print(
        f"滑块安全限位: ±{slider_limit_deg:.0f}°（再裁到模型关节限位内）"
        if slider_limit_deg
        else "滑块安全限位: 模型关节全行程"
    )
    print("初始姿态是训练 HOME_FRAME。滑块往正方向 = 训练策略的正方向。")
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        low, high = model.jnt_range[joint_id]
        print(
            f"  ID {MOTOR_TO_ID[name]:02d}: {name}  {axis_label(model, name)}  "
            f"home={math.degrees(home_targets[name]):+.1f}°  "
            f"range=[{math.degrees(low):+.1f}, {math.degrees(high):+.1f}]°"
        )
    if skipped_names:
        print("模型里没有这些电机:", ", ".join(skipped_names))
    if neutral_offsets:
        print(f"已加载真机中立位偏差 {NEUTRAL_OFFSET_PATH}")
    else:
        print(f"没有真机中立位偏差文件 {NEUTRAL_OFFSET_PATH}")

    bridge = None
    if not args.no_robot:
        print(f"正在通过 SSH 连接真机: {args.host}", flush=True)
        print("如果终端要求密码，在这个终端里输入。", flush=True)
        try:
            bridge = PiMotorBridge(
                args.host,
                startup_ramp_seconds=args.startup_ramp_seconds,
                ready_timeout_s=180.0,
                home_targets=hardware_home,
            )
        except Exception as exc:
            raise SystemExit(
                f"连不上真机 {args.host!r}: {exc}\n"
                "先确认机器人和这台电脑在同一网络，并且没有正在跑的控制程序"
                "（make stop HOST=user@xiaoxiaobai.local）。"
            ) from exc

    viewer = mujoco.viewer.launch_passive(model, data)

    print("滑块窗口已打开。往正方向拖一个关节，对照 MuJoCo 和真机是不是同向。", flush=True)
    app = JointCompareApp(
        model=model,
        data=data,
        viewer=viewer,
        joint_names=joint_names,
        slider_limit_deg=slider_limit_deg,
        bridge=bridge,
        read_interval_ms=args.read_interval_ms,
        home_targets=home_targets,
        neutral_offsets=neutral_offsets,
        root_z=DEFAULT_ROOT_Z,
        title="训练仿真 vs 真机 关节方向",
        intro=INTRO,
        annotations={name: axis_label(model, name) for name in joint_names},
    )
    app.run()


if __name__ == "__main__":
    main()
