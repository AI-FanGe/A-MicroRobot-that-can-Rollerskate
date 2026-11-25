#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

"""Phase 2 of motor_health_check.py (mirrored +-step_deg leg steps, robot hanging) on the
MuJoCo BAM model, printed in the same table so it can be compared with the real robot.

    PYTHONPATH=src uv run --group sim src/debug/sim_motor_check.py [--step-deg 10]

The trunk is pinned in the air after every physics step.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from sim_replay import _NoViewer  # noqa: E402

from constants import KP_RL, MOTOR_TO_ID, NEUTRAL_POSE  # noqa: E402
from motor_health_check import LEG_JOINTS, print_step_header, print_step_row, step_metrics  # noqa: E402
from sim.mujoco_controller import MuJoCoController  # noqa: E402

DT = 0.02


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step-deg", type=float, default=10.0)
    parser.add_argument("--act-delay-steps", type=int, default=2)
    parser.add_argument("--leg-mass-scale", type=float, default=1.0, help="scale mass and inertia of leg bodies")
    args = parser.parse_args()

    mujoco.viewer.launch_passive = lambda *a, **k: _NoViewer()
    ctrl = MuJoCoController(mjcf_path="src/model/mjcf/scene.xml", root_z=0.5, use_bam=True,
                            delay_act_steps=args.act_delay_steps)
    model = ctrl._model
    for b in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if model.body_parentid[b] != 0 and any(k in name for k in ("hip", "femur", "tibia", "ankle", "foot")):
            model.body_mass[b] *= args.leg_mass_scale
            model.body_inertia[b] *= args.leg_mass_scale
    pinned = ctrl._data.qpos[:7].copy()
    raw_step = mujoco.mj_step

    def pinned_step(model, data):
        raw_step(model, data)
        data.qpos[:7] = pinned
        data.qvel[:6] = 0.0

    mujoco.mj_step = pinned_step

    names = list(MOTOR_TO_ID)
    ids = [MOTOR_TO_ID[n] for n in names]
    neutral = np.array([NEUTRAL_POSE[n] for n in names])
    ctrl.sync_write_kp(ids, [KP_RL] * len(ids))

    def run(target: np.ndarray, ticks: int):
        ts, qs, cs = [], [], []
        for k in range(ticks):
            ctrl.sync_write_goal_position(ids, target.tolist())
            ts.append((k + 1) * DT)
            qs.append(ctrl.sync_read_present_position(ids))
            cs.append(ctrl.sync_read_present_current(ids))
        return np.array(ts), np.array(qs), np.array(cs)

    run(neutral, 100)
    print(f"=== sim: mirrored +-{args.step_deg:.0f} deg steps (KP={KP_RL}, trunk pinned) ===")
    print_step_header()
    step = math.radians(args.step_deg)
    for j in LEG_JOINTS:
        kl, kr = names.index(f"left_{j}"), names.index(f"right_{j}")
        for sign in (+1, -1):
            target = neutral.copy()
            target[kl] += sign * step
            target[kr] -= sign * step
            start = np.array(ctrl.sync_read_present_position(ids))
            t, q, c = run(target, 40)
            ml = step_metrics(t, q[:, kl], c[:, kl], start[kl], target[kl])
            mr = step_metrics(t, q[:, kr], c[:, kr], start[kr], target[kr])
            print_step_row(j, sign, ml, mr)
            run(neutral, 40)


if __name__ == "__main__":
    main()
