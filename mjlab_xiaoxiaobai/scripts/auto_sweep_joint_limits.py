from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))

from calibrate_joint_limits_ui import (
    DEFAULT_XML,
    JointControl,
    build_controls,
    load_model,
    reset_to_home,
    save_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Automatically sweep each actuated hinge joint around the home pose and "
            "record the last collision-free angle in both directions."
        )
    )
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("joint_limit_calibration/auto_sweep"),
    )
    parser.add_argument(
        "--wide-ranges",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use a temporary XML with hinge ranges widened to +/- pi.",
    )
    parser.add_argument("--step-deg", type=float, default=1.0)
    parser.add_argument("--safety-margin-deg", type=float, default=2.0)
    parser.add_argument(
        "--contact-tolerance",
        type=int,
        default=0,
        help="Allow this many contacts above the home-pose contact count.",
    )
    return parser.parse_args()


def contact_count(model: mujoco.MjModel, data: mujoco.MjData, qpos: np.ndarray) -> int:
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    return int(data.ncon)


def sweep_direction(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    control: JointControl,
    home_qpos: np.ndarray,
    direction: int,
    step: float,
    contact_limit: int,
) -> float:
    assert direction in (-1, 1)
    joint_min, joint_max = model.jnt_range[control.joint_id]
    end = float(joint_min if direction < 0 else joint_max)
    current = control.home
    last_safe = current

    while (direction < 0 and current > end) or (direction > 0 and current < end):
        current += direction * step
        if direction < 0:
            current = max(current, end)
        else:
            current = min(current, end)

        qpos = home_qpos.copy()
        qpos[control.qposadr] = current
        if contact_count(model, data, qpos) > contact_limit:
            break
        last_safe = current

        if current == end:
            break

    return last_safe


def apply_safety_margin(control: JointControl, margin: float) -> None:
    if control.observed_min < control.home:
        control.observed_min = min(control.home, control.observed_min + margin)
    if control.observed_max > control.home:
        control.observed_max = max(control.home, control.observed_max - margin)


def main() -> None:
    args = parse_args()
    xml_path = args.xml.resolve()
    output_dir = args.output_dir.resolve()
    model, loaded_xml = load_model(xml_path, output_dir, args.wide_ranges)
    data = mujoco.MjData(model)
    reset_to_home(model, data)

    controls = build_controls(model, data)
    if not controls:
        raise RuntimeError("No hinge-joint actuators found.")

    home_qpos = data.qpos.copy()
    base_contacts = contact_count(model, data, home_qpos)
    contact_limit = base_contacts + args.contact_tolerance
    step = math.radians(args.step_deg)
    margin = math.radians(args.safety_margin_deg)

    print(f"[INFO] Source XML: {xml_path}")
    print(f"[INFO] Loaded XML: {loaded_xml}")
    print(f"[INFO] Home contact count: {base_contacts}")
    print(f"[INFO] Sweep step: {args.step_deg} deg, safety margin: {args.safety_margin_deg} deg")

    for control in controls:
        control.observed_min = sweep_direction(
            model,
            data,
            control,
            home_qpos,
            direction=-1,
            step=step,
            contact_limit=contact_limit,
        )
        control.observed_max = sweep_direction(
            model,
            data,
            control,
            home_qpos,
            direction=1,
            step=step,
            contact_limit=contact_limit,
        )
        apply_safety_margin(control, margin)
        print(
            f"{control.joint_name:24s} "
            f"[{math.degrees(control.observed_min):8.2f}, "
            f"{math.degrees(control.observed_max):8.2f}] deg"
        )

    save_outputs(output_dir, controls, xml_path, loaded_xml)


if __name__ == "__main__":
    main()
