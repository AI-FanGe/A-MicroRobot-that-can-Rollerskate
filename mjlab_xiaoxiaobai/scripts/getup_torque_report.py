"""Per-motor torque use of a get-up policy from flat starts (play mode, no assist): how
close each motor comes to its current limit (max_current * kt) while getting up.

usage: getup_torque_report.py <ckpt>
env:   SUPINE=<0..1> (default 1)  SECONDS=<s> (default 12)  FRICTION=<mu>
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

TASK = "Mjlab-Getup-Xiaoxiaobai"
seconds = float(os.environ.get("SECONDS", 12))
cfg = load_env_cfg(TASK, play=True)
cfg.scene.num_envs = 1024
cfg.episode_length_s = seconds + 5.0
cfg.events["reset_getup"].params.update(
    {"standing_fraction": 0.0, "flat_fraction": 1.0, "supine_fraction": float(os.environ.get("SUPINE", 1.0))}
)
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device="cuda:0"))
u = env.unwrapped
if os.environ.get("FRICTION"):
    u.sim.model.geom_friction[:, :, 0] = float(os.environ["FRICTION"])
runner = load_runner_cls(TASK)(env, asdict(load_rl_cfg(TASK)), device="cuda:0")
runner.load(sys.argv[1], load_cfg={"actor": True}, strict=True, map_location="cuda:0")
policy = runner.get_inference_policy("cuda:0")
robot = u.scene["robot"]
act = robot.actuators[0]
limit = act.cfg.max_current * act._bam_model.kt.value
names = list(act.target_names) if hasattr(act, "target_names") else getattr(act, "_target_names", None)
obs = env.get_observations()
U, H, G = [], [], []
with torch.no_grad():
    for _ in range(int(seconds / u.step_dt)):
        obs, *_ = env.step(policy(obs))
        U.append((act._prev_motor_torque.abs() / limit).cpu().numpy())
        H.append((robot.data.root_link_pos_w[:, 2] - u.scene.env_origins[:, 2]).cpu().numpy())
        G.append(robot.data.projected_gravity_b[:, 2].cpu().numpy())
U, H, G = np.stack(U), np.stack(H), np.stack(G)          # (T, N, J), (T, N)
stand = (H > 0.16) & (G < -0.95)
first = np.where(stand.any(0), stand.argmax(0), U.shape[0])
moving = np.arange(U.shape[0])[:, None] < first[None, :]  # before first standing
print(f"stood {stand.any(0).mean():.2f}, motor torque limit {limit:.3f} Nm (= 100%)")
if names is None or len(names) != U.shape[2]:
    names = [f"j{j}" for j in range(U.shape[2])]
peak = np.where(moving[..., None], U, 0).max(0)            # (N, J)
sat = (np.where(moving[..., None], U, 0) > 0.9).sum(0) / np.maximum(moving.sum(0), 1)[:, None]
print(f"{'joint':22s} peak p50/p90   time >90% of limit (mean over envs)")
for j in np.argsort(-np.median(peak, 0)):
    print(f"{names[j]:22s} {np.median(peak[:, j]) * 100:4.0f}%/{np.percentile(peak[:, j], 90) * 100:4.0f}%   {sat[:, j].mean() * 100:5.1f}%")
any_sat = (np.where(moving[..., None], U, 0) > 0.9).any(-1)
print(f"time with any motor >90% of limit while getting up: {(any_sat.sum(0) / np.maximum(moving.sum(0), 1)).mean() * 100:.1f}%")
