"""Run the deployment pipeline (Observer + WalkMove + MuJoCoController) headless.

Compares the deployment IMU observations against the training definitions
(projected gravity in the trunk frame, gyro at the imu site) and reports
whether the policy stands and walks through the deployment code path.
"""

import argparse
import math
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


mujoco.viewer.launch_passive = lambda *a, **k: _NoViewer()

from constants import MOTOR_TO_ID, NEUTRAL_POSE  # noqa: E402
from debug.policy_joint_slider_compare import DEFAULT_MJCF, root_z_on_ground  # noqa: E402
from input.input_source import UserInput  # noqa: E402
from moves.move import MotorCommand, MoveState  # noqa: E402
from moves.walk import WalkMove  # noqa: E402
from observer import Observation, Observer  # noqa: E402
from sim.mujoco_controller import MuJoCoController  # noqa: E402


def trunk_projected_gravity(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
    rot = data.xmat[trunk].reshape(3, 3)
    return rot.T @ np.array([0.0, 0.0, -1.0])


def imu_site_gyro(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
    adr = model.sensor_adr[sid]
    return np.array(data.sensordata[adr:adr + 3])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", default="src/agents/walk_low_battery_1.onnx")
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF)
    parser.add_argument("--vx", type=float, default=0.25)
    parser.add_argument("--stand-s", type=float, default=3.0)
    parser.add_argument("--walk-s", type=float, default=8.0)
    parser.add_argument("--start-from", choices=("neutral", "default"), default="neutral")
    parser.add_argument("--delay-act-steps", type=int, default=0)
    parser.add_argument("--root-pitch-deg", type=float, default=0.0)
    parser.add_argument("--actuator", choices=("bam", "xml"), default="xml")
    args = parser.parse_args()

    walk = WalkMove(agent_path=args.agent)
    default_pose = dict(walk._default_pose)
    initial = dict(NEUTRAL_POSE) if args.start_from == "neutral" else {**NEUTRAL_POSE, **default_pose}
    half_pitch = 0.5 * math.radians(args.root_pitch_deg)
    root_quat = (math.cos(half_pitch), 0.0, math.sin(half_pitch), 0.0)

    probe_model = mujoco.MjModel.from_xml_path(str(args.mjcf))
    probe_data = mujoco.MjData(probe_model)
    probe_data.qpos[3:7] = root_quat
    for name, angle in initial.items():
        jid = mujoco.mj_name2id(probe_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            probe_data.qpos[probe_model.jnt_qposadr[jid]] = angle
    root_z = root_z_on_ground(probe_model, probe_data) + 0.002

    controller = MuJoCoController(
        str(args.mjcf),
        initial_pose=initial,
        root_z=root_z,
        root_quat=root_quat,
        delay_act_steps=args.delay_act_steps,
        use_bam=args.actuator == "bam",
    )
    walk._controller = controller
    observer = Observer(controller)
    model, data = controller._model, controller._data
    ids = list(MOTOR_TO_ID.values())
    controller.sync_write_kp(ids, [400] * len(ids))

    dt = 0.02
    sim_t = 0.0
    total_ticks = int((args.stand_s + args.walk_s) / dt) + 10
    grav_err, gyro_err = [], []
    walk_start_xy = None
    phase_rows: dict[str, list] = {"stand": [], "walk": []}
    fell_at = None
    walk.state = MoveState.STARTING

    for tick in range(total_ticks):
        state = observer.read_state(dt)
        state.time_s = sim_t
        in_walk = walk.state == MoveState.ACTIVE and sim_t >= args.stand_s
        vx = args.vx if in_walk else 0.0
        obs = Observation(robot_state=state, user_input=UserInput(active_moves={"walk"}, velocity={"vx": vx, "vy": 0.0, "vtheta": 0.0}))

        grav_err.append(float(np.max(np.abs(np.array(state.projected_gravity) - trunk_projected_gravity(model, data)))))
        gyro_err.append(float(np.max(np.abs(np.array(state.gyro) - imu_site_gyro(model, data)))))

        command = MotorCommand()
        if walk.state == MoveState.STARTING:
            walk.on_start(obs, command)
        else:
            walk.step(obs, command)
        controller.sync_write_goal_position([MOTOR_TO_ID[n] for n in command.target_angles], list(command.target_angles.values()))
        sim_t += dt

        pg = trunk_projected_gravity(model, data)
        pitch = math.degrees(math.atan2(-pg[0], -pg[2]))
        roll = math.degrees(math.atan2(pg[1], -pg[2]))
        height = float(data.qpos[2])
        if fell_at is None and (pg[2] > -0.5 or height < 0.10):
            fell_at = sim_t
        if walk.state == MoveState.ACTIVE:
            if in_walk:
                if walk_start_xy is None:
                    walk_start_xy = data.qpos[:2].copy()
                phase_rows["walk"].append((pitch, roll, height))
            else:
                phase_rows["stand"].append((pitch, roll, height))

    print(f"agent={args.agent} start_from={args.start_from} delay_act_steps={args.delay_act_steps} root_z0={root_z:.4f}")
    print(f"deploy-vs-training projected_gravity max abs err: {max(grav_err):.2e}")
    print(f"deploy-vs-training gyro max abs err:              {max(gyro_err):.2e}")
    for phase, rows in phase_rows.items():
        if not rows:
            continue
        arr = np.array(rows)
        print(f"{phase}: pitch mean {arr[:, 0].mean():+.1f} deg (min {arr[:, 0].min():+.1f}, max {arr[:, 0].max():+.1f}), "
              f"roll mean {arr[:, 1].mean():+.1f} deg, height mean {arr[:, 2].mean():.3f} m")
    if walk_start_xy is not None:
        disp = data.qpos[:2] - walk_start_xy
        trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        fwd = data.xmat[trunk].reshape(3, 3)[:, 0]
        print(f"walk displacement xy=({disp[0]:+.3f}, {disp[1]:+.3f}) m over {args.walk_s:.1f}s -> "
              f"{np.linalg.norm(disp) / args.walk_s:.3f} m/s, trunk x-axis world=({fwd[0]:+.2f}, {fwd[1]:+.2f})")
    print(f"fell_at: {fell_at if fell_at is not None else 'never'}")


if __name__ == "__main__":
    main()
