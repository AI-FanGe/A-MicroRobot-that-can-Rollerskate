#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

"""MuJoCo + real robot joint slider comparison around a walk policy's default pose.

Run locally on the Ubuntu machine:

    PYTHONPATH=src uv run --group sim src/debug/policy_joint_slider_compare.py --host user@xiaoxiaobai.local

Both the MuJoCo model and the real robot start at the ONNX policy's
``default_joint_pos``; each slider then drives the same joint on both, so a
joint that moves the other way (or by a different amount) on the robot stands
out. This is a hardware test tool only and does not touch the runtime moves.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import mujoco
import mujoco.viewer
import onnxruntime as ort

from constants import MOTOR_TO_ID, OBSERVATION_DOF_ORDER
from neutral_offset import load_neutral_offsets
try:
    from .mujoco_joint_slider_compare import (
        DEFAULT_MJCF,
        DEFAULT_SLIDER_LIMIT_DEG,
        JointCompareApp,
        XIAOXIAOBAI_REPO_ROOT,
        PiMotorBridge,
    )
except ImportError:
    from mujoco_joint_slider_compare import (
        DEFAULT_MJCF,
        DEFAULT_SLIDER_LIMIT_DEG,
        JointCompareApp,
        XIAOXIAOBAI_REPO_ROOT,
        PiMotorBridge,
    )

DEFAULT_POLICY = XIAOXIAOBAI_REPO_ROOT / "src/agents/walk_low_battery_1.onnx"


def load_policy_metadata(policy_path: Path) -> tuple[dict[str, float], str]:
    if not policy_path.exists():
        raise FileNotFoundError(f"ONNX policy not found: {policy_path}")
    session = ort.InferenceSession(str(policy_path))
    meta = session.get_modelmeta().custom_metadata_map
    if "joint_names" not in meta or "default_joint_pos" not in meta:
        raise ValueError(f"Policy has no joint_names/default_joint_pos metadata: {policy_path}")
    names = meta["joint_names"].split(",")
    positions = [float(value) for value in meta["default_joint_pos"].split(",")]
    return dict(zip(names, positions)), meta.get("run_path", "?")


def root_z_on_ground(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """Root height that puts the lowest foot collision point on z = 0."""
    data.qpos[2] = 0.0
    mujoco.mj_forward(model, data)
    lowest = math.inf
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if "foot" not in name or "collision" not in name:
            continue
        if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_BOX:
            continue
        rotation = data.geom_xmat[geom_id].reshape(3, 3)
        center = data.geom_xpos[geom_id]
        half = model.geom_size[geom_id]
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    corner = center + rotation @ (half * (sx, sy, sz))
                    lowest = min(lowest, float(corner[2]))
    return -lowest if math.isfinite(lowest) else 0.186


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="xiaoxiaobai", help="SSH host for the Raspberry Pi.")
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF, help="MuJoCo scene XML (training model).")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY, help="Walk ONNX policy.")
    parser.add_argument(
        "--limit-deg",
        type=float,
        default=DEFAULT_SLIDER_LIMIT_DEG,
        help="Symmetric slider limit around zero before MJCF joint-range clipping; <=0 uses full MJCF limits.",
    )
    parser.add_argument(
        "--startup-ramp-seconds",
        type=float,
        default=2.0,
        help="Seconds used to ramp the real robot to the policy default on start and shutdown.",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    slider_limit_deg = args.limit_deg if args.limit_deg > 0.0 else None
    policy_default_pose, run_path = load_policy_metadata(args.policy)
    joint_names = [
        name
        for name, _motor_id in sorted(MOTOR_TO_ID.items(), key=lambda item: item[1])
        if name in OBSERVATION_DOF_ORDER
    ]
    home_targets = {name: float(policy_default_pose.get(name, 0.0)) for name in joint_names}
    neutral_offsets = load_neutral_offsets()
    hardware_home = {
        name: home_targets[name] + float(neutral_offsets.get(name, 0.0)) for name in joint_names
    }

    model = mujoco.MjModel.from_xml_path(str(args.mjcf))
    data = mujoco.MjData(model)
    for name, angle in home_targets.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id >= 0:
            data.qpos[model.jnt_qposadr[joint_id]] = angle
    root_z = root_z_on_ground(model, data)
    data.qpos[2] = root_z
    mujoco.mj_forward(model, data)

    print("Opening policy joint compare viewer...")
    print(f"MJCF: {args.mjcf.resolve()}")
    print(f"Policy: {args.policy.resolve()} (run {run_path})")
    print(f"Slider safety limit: {slider_limit_deg} deg" if slider_limit_deg else "Slider safety limit: full MJCF")
    print("Slider joints (start at policy default):")
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        low, high = model.jnt_range[joint_id]
        axis = model.jnt_axis[joint_id]
        print(
            f"  ID {MOTOR_TO_ID[name]:02d}: {name} "
            f"default={math.degrees(home_targets[name]):+.1f} deg "
            f"range=[{math.degrees(low):+.1f}, {math.degrees(high):+.1f}] deg "
            f"axis=({axis[0]:+.0f},{axis[1]:+.0f},{axis[2]:+.0f})"
        )

    viewer = mujoco.viewer.launch_passive(model, data)
    bridge = None
    if not args.no_robot:
        print(f"Connecting to Pi over SSH: {args.host}")
        print("If prompted, enter the Pi password in this terminal.")
        try:
            bridge = PiMotorBridge(
                args.host,
                startup_ramp_seconds=args.startup_ramp_seconds,
                ready_timeout_s=180.0,
                home_targets=hardware_home,
            )
        except Exception as exc:
            viewer.close()
            raise SystemExit(
                f"Could not connect to the Pi over SSH using host {args.host!r}: {exc}\n"
                "Make sure no control loop is running; run `make stop HOST=...` first."
            ) from exc

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
        root_z=root_z,
        title="Xiaoxiaobai walk policy joint compare",
    )
    app.run()


if __name__ == "__main__":
    main()
