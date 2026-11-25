"""Drive Mjlab-PropUp-Xiaoxiaobai with the deployment PropUpMove (xiaoxiaobai src/moves/propup.py).

Like dance_deploy_check.py, each env gets the deployment move and an Observer reading the sim
through a fake controller. The robot props up, holds, lies down, lies still and props up again
on the button (propup_down in the active moves), as the API drives it. Until the first hold
the deployment observation is compared against the training one (with --yaw 0 they must
match); throughout, the trunk's orientation error to the reference frame the move shows is
tracked, and an env fails past 45 deg.

usage: propup_deploy_check.py [--onnx ../xiaoxiaobai/src/agents/propup.onnx] [--num-envs 64] [--yaw 3.14]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dance_deploy_check import Controller  # noqa: E402
from deploy_pipeline_in_training_env import DEPLOY_SRC, sensor_adr  # noqa: E402

import mjlab_xiaoxiaobai.tasks  # noqa: E402,F401
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402

from constants import OBSERVATION_DOF_ORDER, PROPUP_DOWN  # noqa: E402
from input.input_source import UserInput  # noqa: E402
from moves.move import MotorCommand, MoveState  # noqa: E402
from moves.propup import PropUpMove  # noqa: E402
from observer import Observation, Observer  # noqa: E402

# (button down?, phase to reach, seconds to stay there once reached)
PLAN = [(False, "up", 2.0), (True, "lying", 1.0), (False, "up", 1.0)]
FAIL_DEG = 45.0


def rot_err_deg(obs: np.ndarray) -> float:
    c = obs[-6:].reshape(3, 2)
    r = np.column_stack([c, np.cross(c[:, 0], c[:, 1])])
    return float(np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=str(DEPLOY_SRC / "agents" / "propup.onnx"))
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--yaw", type=float, default=3.14, help="start yaw noise range [rad]")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    cfg = load_env_cfg("Mjlab-PropUp-Xiaoxiaobai", play=True)
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
    cmd = env.command_manager.get_term("motion")
    # The deployment holds frames; past its end the training command must not reset the robot.
    cmd._resample_command = lambda ids: cmd.time_steps.__setitem__(ids, cmd.motion.time_step_total - 1)
    moves = [PropUpMove(agent_path=args.onnx) for _ in range(n)]
    observers = [Observer(Controller(env, i, gyro_adr, quat_adr)) for i in range(n)]
    sim_step = [0]
    for m in moves:
        m.state = MoveState.ACTIVE
        m.clock = lambda: sim_step[0] * env.step_dt
    hold = moves[0]._hold_frame

    obs_err = np.zeros(102)
    worst = np.zeros(n)
    fail_t = np.full(n, np.nan)
    stage = np.zeros(n, dtype=int)
    reached = np.full(n, np.nan)
    stage_times = [[] for _ in PLAN]
    refused = 0
    step = 0
    max_steps = int(40.0 / env.step_dt)
    while (stage < len(PLAN)).any() and step < max_steps:
        sim_step[0] = step
        env_obs = env.observation_manager.compute()["actor"].cpu().numpy()
        actions = torch.zeros((n, len(action_joints)), device=env.device)
        for i in range(n):
            down, _, _ = PLAN[min(stage[i], len(PLAN) - 1)]
            user = UserInput(active_moves={"propup", PROPUP_DOWN} if down else {"propup"},
                             imu_roll_corr_deg=0.0, imu_pitch_corr_deg=0.0)
            o = Observation(robot_state=observers[i].read_state(env.step_dt), user_input=user)
            if step == 0:
                moves[i].on_start(o, MotorCommand())
                refused += moves[i].refused
            d = np.asarray(moves[i].build_observation(o))
            c = MotorCommand()
            moves[i].step(o, c)
            if step < hold - 1:
                obs_err = np.maximum(obs_err, np.abs(d - env_obs[i]))
            e = rot_err_deg(d)
            worst[i] = max(worst[i], e)
            if e > FAIL_DEG and np.isnan(fail_t[i]):
                fail_t[i] = step * env.step_dt
            if stage[i] < len(PLAN):
                _, phase, stay = PLAN[stage[i]]
                if np.isnan(reached[i]) and moves[i].status()["phase"] == phase:
                    reached[i] = step * env.step_dt
                    stage_times[stage[i]].append(reached[i])
                if not np.isnan(reached[i]) and step * env.step_dt - reached[i] >= stay:
                    stage[i] += 1
                    reached[i] = np.nan
            for j, nm in enumerate(action_joints):
                actions[i, j] = float(c.target_angles[nm]) - default[nm]
        env.step(actions)
        step += 1

    names = ["gyro"] * 3 + ["proj_grav"] * 3 + ["joint_pos"] * 18 + ["joint_vel"] * 18 + ["actions"] * 18 \
        + ["motion_cmd_pos"] * 18 + ["motion_cmd_vel"] * 18 + ["anchor_ori"] * 6
    per = {}
    for nm, e in zip(names, obs_err):
        per[nm] = max(per.get(nm, 0.0), round(float(e), 6))
    result = {
        "onnx": args.onnx,
        "yaw_noise": args.yaw,
        "refused": refused,
        "deploy_vs_training_obs_max_abs_err_until_hold": per,
        "stages_completed": int((stage >= len(PLAN)).sum()),
        "stage_reached_s_median": [round(float(np.median(t)), 2) if t else None for t in stage_times],
        "trunk_vs_ref_err_deg_p50_max": [round(float(np.median(worst)), 1), round(float(worst.max()), 1)],
        "fail_rate": float((~np.isnan(fail_t)).mean()),
        "fail_times_s": sorted(np.round(fail_t[~np.isnan(fail_t)], 2).tolist()),
        "sim_s": round(step * env.step_dt, 1),
    }
    print("RESULT " + json.dumps(result, indent=2), flush=True)
    env.close()


if __name__ == "__main__":
    main()
