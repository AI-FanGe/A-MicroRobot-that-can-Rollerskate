"""Drive the training env with the unmodified deployment code path.

Each env gets its own deployment ``WalkMove`` and ``Observer``; the Observer
reads joints and the IMU (framequat + gyro at the ``imu`` site) through a fake
controller backed by the training sim, so the policy sees exactly what the
robot-side code would build. The deployment observation is compared against
the training observation every step, and the deployment target angles are
converted back into env actions.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch

DEPLOY_SRC = Path(__file__).resolve().parents[2] / "xiaoxiaobai" / "src"
sys.path.insert(0, str(DEPLOY_SRC))
for stub in ("bmi088", "bmi088.bmi088", "smbus3", "ahrs", "ahrs.common", "ahrs.common.orientation"):
    module = types.ModuleType(stub)
    module.BMI088 = object
    module.SMBus = object
    module.acc2q = lambda *a, **k: None
    sys.modules[stub] = module

import mjlab.tasks  # noqa: E402,F401
import mjlab_xiaoxiaobai.tasks  # noqa: E402,F401
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402

from constants import ID_TO_MOTOR, OBSERVATION_DOF_ORDER  # noqa: E402
from input.input_source import UserInput  # noqa: E402
from moves.move import MotorCommand, MoveState  # noqa: E402
from moves.skate import SkateMove  # noqa: E402
from moves.walk import WalkMove  # noqa: E402
from observer import Observation, Observer  # noqa: E402

TASK = "Mjlab-Velocity-Xiaoxiaobai"


class SimBackedController:
    def __init__(self, env: ManagerBasedRlEnv, index: int, gyro_adr: int, quat_adr: int) -> None:
        robot = env.scene["robot"]
        self._robot = robot
        self._env = env
        self._i = index
        self._joint = {name: robot.joint_names.index(name) for name in robot.joint_names}
        self._gyro_adr = gyro_adr
        self._quat_adr = quat_adr

    # Motors the sim model has no joint for (head_pitch) read 0.
    def sync_read_present_position(self, ids):
        q = self._robot.data.joint_pos[self._i]
        return [float(q[self._joint[ID_TO_MOTOR[mid]]]) if ID_TO_MOTOR[mid] in self._joint else 0.0 for mid in ids]

    def sync_read_present_velocity(self, ids):
        v = self._robot.data.joint_vel[self._i]
        return [float(v[self._joint[ID_TO_MOTOR[mid]]]) if ID_TO_MOTOR[mid] in self._joint else 0.0 for mid in ids]

    def read_acc(self):
        return (0.0, 0.0, -1.0)

    def read_gyro(self):
        s = self._env.sim.data.sensordata[self._i]
        return tuple(float(x) for x in s[self._gyro_adr:self._gyro_adr + 3])

    def read_quat(self, dt):
        s = self._env.sim.data.sensordata[self._i]
        return tuple(float(x) for x in s[self._quat_adr:self._quat_adr + 4])


def sensor_adr(env: ManagerBasedRlEnv, suffix: str) -> int:
    import mujoco

    model = env.sim.mj_model
    for sid in range(model.nsensor):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, sid) or ""
        if name == suffix or name.endswith("/" + suffix):
            return int(model.sensor_adr[sid])
    raise KeyError(suffix)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=TASK)
    parser.add_argument("--onnx", default=None)
    parser.add_argument("--move", choices=("walk", "skate"), default="walk")
    parser.add_argument("--wz", type=float, default=0.0)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--stand-s", type=float, default=3.0)
    parser.add_argument("--walk-s", type=float, default=8.0)
    parser.add_argument("--vx", type=float, default=0.25)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    env_cfg = load_env_cfg(args.task, play=True)
    env_cfg.scene.num_envs = int(args.num_envs)
    env_cfg.commands["twist"].debug_vis = False
    env_cfg.commands["twist"].resampling_time_range = (1.0e9, 1.0e9)
    env_cfg.events["push_robot"].interval_range_s = (1.0e9, 1.0e9)
    env_cfg.events.pop("encoder_bias", None)
    env_cfg.terminations.clear()
    for group in env_cfg.observations.values():
        group.enable_corruption = False
    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
    robot = env.scene["robot"]
    n = env.num_envs

    action_term = env.action_manager.get_term("joint_pos")
    action_joints = list(action_term.target_names) if hasattr(action_term, "target_names") else list(OBSERVATION_DOF_ORDER)
    env_default = {name: float(robot.data.default_joint_pos[0, robot.joint_names.index(name)]) for name in robot.joint_names}

    gyro_adr = sensor_adr(env, "angular-velocity")
    quat_adr = sensor_adr(env, "orientation")
    obs_dict, _ = env.reset()
    move_cls = SkateMove if args.move == "skate" else WalkMove
    if args.onnx is None:
        args.onnx = str(DEPLOY_SRC / "agents" / f"{args.move}.onnx")
    walks = [move_cls(agent_path=args.onnx) for _ in range(n)]
    observers = [Observer(SimBackedController(env, i, gyro_adr, quat_adr)) for i in range(n)]
    for walk in walks:
        walk.state = MoveState.ACTIVE
    default_mismatch = max(abs(walks[0]._default_pose[k] - env_default[k]) for k in OBSERVATION_DOF_ORDER)

    rate_cfg = (env.action_manager.get_term("stroke_rate").cfg
                if "stroke_rate" in env.action_manager.active_terms else None)
    rates: list[float] = []
    term = env.command_manager.get_term("twist")
    stand_steps = int(round(args.stand_s / env.step_dt))
    walk_steps = int(round(args.walk_s / env.step_dt))
    obs_err = np.zeros(env.observation_manager.group_obs_dim["actor"][0])
    alive = np.ones(n, dtype=bool)
    start_xy = None
    pitches = {"stand": [], "walk": []}
    gravity_w = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(n, 3)
    from mjlab.utils.lab_api.math import quat_apply_inverse

    for step in range(stand_steps + walk_steps):
        vx = 0.0 if step < stand_steps else args.vx
        wz = 0.0 if step < stand_steps else args.wz
        vel = torch.tensor([vx, 0.0, wz], device=env.device, dtype=term.vel_command_b.dtype)
        term.vel_command_b[:] = vel
        for flag in ("is_standing_env", "is_heading_env", "is_world_env", "is_forward_env", "is_rotation_env"):
            if hasattr(term, flag):
                getattr(term, flag)[:] = False
        if step == stand_steps:
            start_xy = robot.data.root_link_pos_w[:, :2].clone()

        env_obs = env.observation_manager.compute()["actor"].cpu().numpy()
        actions = torch.zeros((n, env.action_manager.total_action_dim), device=env.device)
        for i in range(n):
            state = observers[i].read_state(env.step_dt)
            obs = Observation(robot_state=state, user_input=UserInput(active_moves={args.move}, velocity={"vx": vx, "vy": 0.0, "vtheta": wz}))
            deploy_obs = np.asarray(walks[i].build_observation(obs), dtype=np.float64)
            obs_err = np.maximum(obs_err, np.abs(deploy_obs - env_obs[i]))
            command = MotorCommand()
            walks[i].step(obs, command)
            for j, name in enumerate(action_joints):
                actions[i, j] = float(command.target_angles[name]) - env_default[name]
            if rate_cfg is not None:
                # The env's stroke clock runs at the rate the deployment phase just took.
                actions[i, len(action_joints)] = (walks[i]._stroke_rate - 1.0) / rate_cfg.scale
                rates.append(walks[i]._stroke_rate)
        env.step(actions)

        g_b = quat_apply_inverse(robot.data.root_link_quat_w, gravity_w)
        pitch = torch.rad2deg(torch.asin(g_b[:, 0].clamp(-1, 1))).cpu().numpy()
        height = robot.data.root_link_pos_w[:, 2].cpu().numpy()
        alive &= ~((np.abs(pitch) > 50) | (height < 0.10))
        pitches["stand" if step < stand_steps else "walk"].append(np.where(alive, pitch, np.nan))

    disp = (robot.data.root_link_pos_w[:, :2] - start_xy).cpu().numpy()
    speed = np.linalg.norm(disp, axis=1) / args.walk_s
    names = ["gyro"] * 3 + ["proj_grav"] * 3 + ["joint_pos"] * 18 + ["joint_vel"] * 18 + ["last_action"] * 18 + ["command"] * 3
    names += ["gait_phase"] * (len(obs_err) - len(names))
    per_group = {}
    for name, err in zip(names, obs_err):
        per_group[name] = max(per_group.get(name, 0.0), float(err))
    result = {
        "onnx": args.onnx,
        "action_joint_order_equals_deploy": action_joints == list(OBSERVATION_DOF_ORDER),
        "default_pose_max_abs_diff_rad": default_mismatch,
        "deploy_vs_training_obs_max_abs_err": per_group,
        "fall_rate": float((~alive).mean()),
        "stand_pitch_mean_deg": float(np.nanmean(np.array(pitches["stand"]))),
        "walk_pitch_mean_deg": float(np.nanmean(np.array(pitches["walk"]))),
        "walk_speed_mean_mps": float(speed[alive].mean()) if alive.any() else None,
        "walk_vx_cmd": args.vx,
        "walk_wz_cmd": args.wz,
        "stroke_rate_mean": float(np.mean(rates)) if rates else None,
    }
    print("RESULT " + json.dumps(result, indent=2), flush=True)
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    env.close()


if __name__ == "__main__":
    main()
