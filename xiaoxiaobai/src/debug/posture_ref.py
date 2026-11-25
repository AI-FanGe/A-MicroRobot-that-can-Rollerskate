#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

"""Write src/agents/posture_ref.json: the trunk attitude each walk policy keeps in sim.

    PYTHONPATH=src uv run --group sim src/debug/posture_ref.py [--agents walk_low_battery_1.onnx ...]

Runs the deployment code path (Observer + WalkMove + MuJoCoController with the BAM servo
model, supply KP_VIN_REF) headless. For every walk policy, at forward speeds VX_BINS (0:
standing), the lean right / forward [deg] averaged over MEASURE_S once settled, as the
scheduler computes them from the projected gravity; then again with the IMU corrections at
+GAIN_PROBE_DEG roll and pitch, which gives how many degrees of lean a degree of correction
moves the policy to (gain_roll, gain_pitch). Also the trunk attitude of the robot standing on
flat soles at NEUTRAL_POSE, from the kinematics, with its derivative per leg joint ("neutral"),
the reference of the IMU mount calibration (imu_calib.py).
tilt_trim.py holds the robot at these attitudes; rerun after adding or retraining a policy.
"""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


class _NoViewer:
    def __init__(self) -> None:
        self.opt = mujoco.MjvOption()

    def is_running(self) -> bool:
        return True

    def sync(self) -> None:
        pass

    def close(self) -> None:
        pass


mujoco.viewer.launch_passive = lambda *a, **k: _NoViewer()

from constants import (KP_VIN_REF, MOTOR_TO_ID, NEUTRAL_POSE, WALK_AGENTS,  # noqa: E402
                       WALK_UPPER_AGENT)
from imu_reader import apply_mount_correction  # noqa: E402
from input.input_source import UserInput  # noqa: E402
from moves.move import MotorCommand, MoveState  # noqa: E402
from moves.walk import WalkMove  # noqa: E402
from observer import Observation, Observer  # noqa: E402
from sim.mujoco_controller import MuJoCoController  # noqa: E402

OUT = Path("src/agents/posture_ref.json")
VX_BINS = [0.0, 0.1, 0.2, 0.3, 0.4, -0.1, -0.2, -0.3, -0.4]  # m/s, in the order walked
SETTLE_S = 3.0
MEASURE_S = 6.0
GAIN_PROBE_DEG = 2.0
DT = 0.02
FOOT_SOLES = ("left_foot_collision", "right_foot_collision")
LEG_JOINTS = [name for name in NEUTRAL_POSE if any(part in name for part in ("hip", "knee", "ankle"))]


def _controller() -> MuJoCoController:
    controller = MuJoCoController("src/model/mjcf/scene.xml", root_z=0.181, use_bam=True, delay_act_steps=2,
                                  delay_vel_ticks=1, delay_gyro_ticks=3, delay_quat_ticks=4)
    controller._bam.model.actuator.vin = KP_VIN_REF
    return controller


def _lean(gravity) -> tuple[float, float]:
    gx, gy, _ = gravity
    return math.degrees(math.asin(max(-1.0, min(1.0, -gy)))), math.degrees(math.asin(max(-1.0, min(1.0, gx))))


def _kinematic_lean(model: mujoco.MjModel, data: mujoco.MjData, pose: dict[str, float]) -> np.ndarray:
    """Lean right / forward [deg] of the trunk standing on both soles flat at this joint pose,
    averaged over the two feet (kinematics only: the pose need not balance)."""
    data.qpos[:] = model.qpos0
    for name, q in pose.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            data.qpos[model.jnt_qposadr[jid]] = q
    mujoco.mj_kinematics(model, data)
    trunk = data.xmat[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")].reshape(3, 3)
    leans = []
    for name in FOOT_SOLES:
        sole = data.geom_xmat[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)].reshape(3, 3)
        # The sole's normal (local z) is the world up of a sole standing flat.
        leans.append(_lean(-(trunk.T @ sole[:, 2])))
    return np.mean(leans, axis=0)


def neutral() -> dict:
    """The trunk attitude of the robot standing at NEUTRAL_POSE, with how it moves per radian
    of each leg joint (the robot sags a little off the pose)."""
    model = mujoco.MjModel.from_xml_path("src/model/mjcf/scene.xml")
    data = mujoco.MjData(model)
    base = _kinematic_lean(model, data, NEUTRAL_POSE)
    jac_right, jac_fwd = {}, {}
    for name in LEG_JOINTS:
        step = 0.01
        plus = _kinematic_lean(model, data, {**NEUTRAL_POSE, name: NEUTRAL_POSE[name] + step})
        minus = _kinematic_lean(model, data, {**NEUTRAL_POSE, name: NEUTRAL_POSE[name] - step})
        jac_right[name], jac_fwd[name] = (round(float(v), 3) for v in (plus - minus) / (2 * step))
    return {"lean_right": round(float(base[0]), 3), "lean_fwd": round(float(base[1]), 3),
            "jac_right": jac_right, "jac_fwd": jac_fwd}


def walk(agent: str, offset_deg: float) -> tuple[dict[float, tuple[float, float]], bool]:
    controller = _controller()
    observer = Observer(controller)
    move = WalkMove(controller=controller, agents=[(agent, agent)])
    move.state = MoveState.STARTING
    leans: dict[float, list] = {vx: [] for vx in VX_BINS}
    fell = False
    t = 0.0
    for vx in VX_BINS:
        for k in range(int((SETTLE_S + MEASURE_S) / DT)):
            state = observer.read_state(DT)
            state.time_s = t
            raw = list(state.projected_gravity)
            if t > 1.0 and raw[2] > -0.5:  # the delayed IMU reads (1, 0, 0) at first
                fell = True
            state.projected_gravity = apply_mount_correction(raw, offset_deg, offset_deg)
            user = UserInput(active_moves={"walk"}, velocity={"vx": vx, "vy": 0.0, "vtheta": 0.0}, kp_vin_comp=False)
            command = MotorCommand()
            obs = Observation(robot_state=state, user_input=user)
            if move.state == MoveState.STARTING:
                move.on_start(obs, command)
            else:
                move.step(obs, command)
            controller.sync_write_goal_position([MOTOR_TO_ID[n] for n in command.target_angles],
                                                list(command.target_angles.values()))
            if k * DT >= SETTLE_S:
                leans[vx].append(_lean(raw))
            t += DT
    return {vx: tuple(np.mean(v, axis=0)) for vx, v in leans.items()}, fell


def agent_ref(agent: str) -> dict:
    base, fell = walk(agent, 0.0)
    probe, fell_probe = walk(agent, GAIN_PROBE_DEG)
    vxs = sorted(base)
    walking = [vx for vx in vxs if vx != 0.0]
    gain_roll = float(np.mean([-(probe[vx][0] - base[vx][0]) for vx in walking])) / GAIN_PROBE_DEG
    gain_pitch = float(np.mean([probe[vx][1] - base[vx][1] for vx in walking])) / GAIN_PROBE_DEG
    print(f"{agent}: gain roll {gain_roll:.2f} pitch {gain_pitch:.2f}"
          + ("  FELL" if fell or fell_probe else "") + "  " +
          ", ".join(f"{vx:+.1f}: {base[vx][0]:+.1f}/{base[vx][1]:+.1f}" for vx in vxs), flush=True)
    return {
        "vx": vxs,
        "lean_right": [round(float(base[vx][0]), 3) for vx in vxs],
        "lean_fwd": [round(float(base[vx][1]), 3) for vx in vxs],
        "gain_roll": round(gain_roll, 3),
        "gain_pitch": round(gain_pitch, 3),
        "fell": fell or fell_probe,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agents", nargs="*", help="policy files in src/agents (default: every walk policy)")
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    agents = args.agents or [f for _, f in WALK_AGENTS] + [WALK_UPPER_AGENT]
    out = json.loads(OUT.read_text()) if OUT.exists() else {"agents": {}}
    with ProcessPoolExecutor(args.jobs) as pool:
        neutral_ref = pool.submit(neutral)
        refs = dict(zip(agents, pool.map(agent_ref, agents)))
        out["neutral"] = neutral_ref.result()
    out["vin"] = KP_VIN_REF
    out["agents"].update(refs)
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"neutral: {out['neutral']}\nwrote {OUT}")


if __name__ == "__main__":
    main()
