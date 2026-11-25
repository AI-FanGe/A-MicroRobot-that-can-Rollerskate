"""Fit a walk policy's arm joint angles as a linear function of its leg state and command.

Rolls the checkpoint out at 7.2 V on flat floor (no pushes) with commands redrawn every 2-5 s
(vx, vy, wz from the training ranges, 10% standing), and fits, per arm joint, least squares on
  [1, vx, |vx|, vy, |vy|, wz, |wz|] x [1, hip pitch R, hip pitch L, hip roll R, hip roll L, knee R, knee L]
(the products included), over the steps after SETTLE_S. Prints R^2 and the rms residual [deg]
per joint on held-out envs, and saves the weights to OUT (npz: names, features, W).

usage: XIAOXIAOBAI_BASE=24996 arm_ref_fit.py <ckpt> [OUT]
"""
import os
import sys
from dataclasses import asdict

import numpy as np
import torch
import mjlab_xiaoxiaobai.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab_xiaoxiaobai.tasks.mdp import arm_reference_features

TASK = "Mjlab-Velocity-Xiaoxiaobai"
ckpt = sys.argv[1]
out_path = sys.argv[2] if len(sys.argv) > 2 else None
seconds, settle_s = float(os.environ.get("SECONDS", 30)), 2.0

cfg = load_env_cfg(TASK, play=True)
cfg.scene.num_envs = int(os.environ.get("NUM_ENVS", 1024))
cfg.episode_length_s = seconds + 5.0
cfg.events.pop("push_robot", None)
cfg.events["base_com"].params["ranges"] = {0: (0.0, 0.0), 1: (0.0, 0.0), 2: (0.0, 0.0)}
cfg.commands["twist"].resampling_time_range = (2.0, 5.0)
cfg.commands["twist"].rel_standing_envs = 0.1
gen = cfg.scene.terrain.terrain_generator
gen.sub_terrains = {"flat": gen.sub_terrains["flat"]}
gen.sub_terrains["flat"].proportion = 1.0
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device="cuda:0"))
u = env.unwrapped
n = u.num_envs
runner = load_runner_cls(TASK)(env, asdict(load_rl_cfg(TASK)), device="cuda:0")
runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
policy = runner.get_inference_policy("cuda:0")
robot = u.scene["robot"]
for act in robot.actuators:
    act.cfg.vin_min = 4.0
    act.vin_tensor[:] = 7.2
cmd = u.command_manager.get_term("twist")

names = list(robot.joint_names)
LEG = ["right_hip_pitch", "left_hip_pitch", "right_hip_roll", "left_hip_roll", "right_knee", "left_knee"]
ARM = ["right_shoulder_pitch", "left_shoulder_pitch", "right_shoulder_roll", "left_shoulder_roll", "right_elbow", "left_elbow"]
leg_ids, arm_ids = [names.index(x) for x in LEG], [names.index(x) for x in ARM]

X, Y, E = [], [], []
obs, _ = env.reset()
alive = torch.ones(n, dtype=torch.bool, device=u.device)
with torch.no_grad():
    for t in range(int(seconds / u.step_dt)):
        obs, _, dones, extras = env.step(policy(obs))
        alive &= ~(dones.bool() & ~extras["time_outs"].bool())
        if t * u.step_dt < settle_s:
            continue
        q = robot.data.joint_pos - robot.data.default_joint_pos
        X.append(arm_reference_features(cmd.command, q[:, leg_ids])[alive])
        Y.append(q[:, arm_ids][alive])
        E.append(torch.arange(n, device=u.device)[alive])
X, Y, E = torch.cat(X).double(), torch.cat(Y).double(), torch.cat(E)
train, test = E % 5 != 0, E % 5 == 0
Xt = X[train]
W = torch.linalg.solve(Xt.T @ Xt + 1e-6 * torch.eye(X.shape[1], dtype=X.dtype, device=X.device), Xt.T @ Y[train])
res = Y[test] - X[test] @ W
r2 = 1 - res.var(0) / Y[test].var(0)
print(f"{ckpt}: {int(train.sum())} train / {int(test.sum())} test samples")
for j, name in enumerate(ARM):
    print(f"  {name:22s} R^2 {r2[j].item():.3f}  residual rms {np.rad2deg(res[:, j].square().mean().sqrt().item()):5.2f} deg  (std {np.rad2deg(Y[test][:, j].std().item()):5.2f})")
if out_path:
    np.savez(out_path, legs=np.array(LEG), arms=np.array(ARM), W=W.float().cpu().numpy())
    print("saved", out_path)
