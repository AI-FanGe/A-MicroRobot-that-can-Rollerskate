"""Drive Mjlab-Dance-Xiaoxiaobai with the deployment DanceMove (xiaoxiaobai src/moves/dance.py).

Like deploy_pipeline_in_training_env.py: each env gets a DanceMove and an Observer reading the
sim through a fake controller. The deployment observation is compared against the training one
(with --yaw 0 the start is aligned and they must match), the whole reference is played and the
falls are counted.

usage: dance_deploy_check.py [--onnx ../xiaoxiaobai/src/agents/dance.onnx] [--num-envs 64] [--yaw 0.2]
env:   TASK (Mjlab-Dance-Xiaoxiaobai); for a task that plays the head from the reference
       (Mjlab-Gymnastics-Xiaoxiaobai) the deployment's head target is checked against the training one.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deploy_pipeline_in_training_env import DEPLOY_SRC, SimBackedController, sensor_adr  # noqa: E402

import mjlab_xiaoxiaobai.tasks  # noqa: E402,F401
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402
from mjlab.utils.lab_api.math import quat_apply_inverse  # noqa: E402

from constants import OBSERVATION_DOF_ORDER  # noqa: E402
from input.input_source import UserInput  # noqa: E402
from moves.dance import DanceMove  # noqa: E402
from moves.move import MotorCommand, MoveState  # noqa: E402
from constants import ID_TO_MOTOR  # noqa: E402
from observer import Observation, Observer  # noqa: E402


class Controller(SimBackedController):
    """Motors the sim robot lacks (head_pitch on the 24996 base) read 0."""

    def _read(self, values, ids):
        return [float(values[self._joint[ID_TO_MOTOR[m]]]) if ID_TO_MOTOR[m] in self._joint else 0.0 for m in ids]

    def sync_read_present_position(self, ids):
        return self._read(self._robot.data.joint_pos[self._i], ids)

    def sync_read_present_velocity(self, ids):
        return self._read(self._robot.data.joint_vel[self._i], ids)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=str(DEPLOY_SRC / "agents" / "dance.onnx"))
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--yaw", type=float, default=0.2, help="start yaw noise range [rad]")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    cfg = load_env_cfg(os.environ.get("TASK", "Mjlab-Dance-Xiaoxiaobai"), play=True)
    cfg.scene.num_envs = args.num_envs
    cfg.episode_length_s = 1.0e6
    cfg.terminations = {k: v for k, v in cfg.terminations.items() if k == "time_out"}
    cfg.events.pop("encoder_bias", None)
    cfg.events.pop("push_robot", None)
    cmd_cfg = cfg.commands["motion"]
    cmd_cfg.debug_vis = False
    cmd_cfg.pose_range = {**cmd_cfg.pose_range, "yaw": (-args.yaw, args.yaw)}
    for group in cfg.observations.values():
        group.enable_corruption = False
        for term in group.terms.values():
            term.delay_min_lag = term.delay_max_lag = 0
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    robot = env.scene["robot"]
    n = env.num_envs
    action_joints = list(env.action_manager.get_term("joint_pos").target_names)
    assert action_joints == list(OBSERVATION_DOF_ORDER), action_joints
    default = {nm: float(robot.data.default_joint_pos[0, robot.joint_names.index(nm)]) for nm in robot.joint_names}

    gyro_adr, quat_adr = sensor_adr(env, "angular-velocity"), sensor_adr(env, "orientation")
    env.reset()
    moves = [DanceMove(agent_path=args.onnx) for _ in range(n)]
    observers = [Observer(Controller(env, i, gyro_adr, quat_adr)) for i in range(n)]
    sim_step = [0]
    for m in moves:
        m.state = MoveState.ACTIVE
        m.clock = lambda: sim_step[0] * env.step_dt
    pose_err = max(abs(moves[0]._default_pose[k] - default[k]) for k in OBSERVATION_DOF_ORDER)

    total = moves[0].total_frames
    obs_err = np.zeros(102)
    alive = np.ones(n, dtype=bool)
    fall_t = np.full(n, np.nan)
    gravity_w = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(n, 3)
    where = np.zeros((102, 3), dtype=int)
    user = UserInput(active_moves={"dance"}, imu_roll_corr_deg=0.0, imu_pitch_corr_deg=0.0)
    head = robot.joint_names.index("head")
    head_err = 0.0
    gyro_errs = []
    # The training command resamples once its frame reaches the end; the deployment holds it.
    for step in range(total - 2):
        sim_step[0] = step
        env_obs = env.observation_manager.compute()["actor"].cpu().numpy()
        actions = torch.zeros((n, len(action_joints)), device=env.device)
        for i in range(n):
            o = Observation(robot_state=observers[i].read_state(env.step_dt), user_input=user)
            if step == 0:
                moves[i].on_start(o, MotorCommand())
            d = np.asarray(moves[i].build_observation(o))
            c = MotorCommand()
            moves[i].step(o, c)
            if alive[i]:
                e = np.abs(d - env_obs[i])
                gyro_errs.append(float(e[:3].max()))
                worse = e > obs_err
                where[worse] = (step, i, moves[i]._frame - 1 if step > 0 else 1)
                obs_err = np.maximum(obs_err, e)
            for j, nm in enumerate(action_joints):
                actions[i, j] = float(c.target_angles[nm]) - default[nm]
        # The training head target for this step was set by the command at the end of the last one.
        if "head" in c.target_angles:
            head_err = max(head_err, abs(float(c.target_angles["head"]) - float(robot.data.joint_pos_target[n - 1, head])))
        env.step(actions)
        g = quat_apply_inverse(robot.data.root_link_quat_w, gravity_w)
        tilt = torch.rad2deg(torch.acos((-g[:, 2]).clamp(-1, 1))).cpu().numpy()
        h = robot.data.root_link_pos_w[:, 2].cpu().numpy()
        new = alive & ((tilt > 50) | (h < 0.10))
        fall_t[new] = step * env.step_dt
        alive &= ~new

    names = ["gyro"] * 3 + ["proj_grav"] * 3 + ["joint_pos"] * 18 + ["joint_vel"] * 18 + ["actions"] * 18 \
        + ["motion_cmd_pos"] * 18 + ["motion_cmd_vel"] * 18 + ["anchor_ori"] * 6
    per = {}
    for k, (nm, e) in enumerate(zip(names, obs_err)):
        if float(e) > per.get(nm, (0.0,))[0]:
            per[nm] = (round(float(e), 6), f"obs {k} step/env/frame {where[k].tolist()}")
    result = {
        "onnx": args.onnx,
        "yaw_noise": args.yaw,
        "default_pose_max_abs_diff_rad": pose_err,
        "deploy_vs_training_obs_max_abs_err": per,
        "head_target_max_abs_err_rad": round(head_err, 6),
        "gyro_err_p50_p99": [round(float(np.percentile(gyro_errs, q)), 6) for q in (50, 99)],
        "fall_rate": float((~alive).mean()),
        "fall_times_s": sorted(np.round(fall_t[~alive], 2).tolist()),
        "done": all(m.done for m in moves),
    }
    print("RESULT " + json.dumps(result, indent=2), flush=True)
    env.close()


if __name__ == "__main__":
    main()
