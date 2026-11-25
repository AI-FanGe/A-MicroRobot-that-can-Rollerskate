"""How much a policy chatters at a fixed command, with the actor's sensor noise and delays on:
action change per tick, its second difference, and the share of the joint targets' power
above 6 Hz (the robot's control loop runs at 50 Hz).

usage: policy_smoothness.py <task> <ckpt>
env:   VX=<m/s> (0)  WZ=<rad/s> (0)  SECONDS (10)  NUM_ENVS (256)
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
from mjlab_xiaoxiaobai.tasks.mdp import _twist_cfgs, set_command_velocity

TASK, CKPT = sys.argv[1], sys.argv[2]
VX, WZ = float(os.environ.get("VX", 0.0)), float(os.environ.get("WZ", 0.0))
cfg = load_env_cfg(TASK, play=True)
cfg.scene.num_envs = int(os.environ.get("NUM_ENVS", 256))
cfg.observations["actor"].enable_corruption = True
cfg.episode_length_s = 1e3
cfg.terminations.pop("out_of_terrain_bounds", None)
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device="cuda:0"))
u = env.unwrapped
standing = 1.0 if VX == 0 and WZ == 0 else 0.0
set_command_velocity(u, lin_vel_x=(VX, VX), lin_vel_y=(0.0, 0.0), ang_vel_z=(WZ, WZ))
for c in _twist_cfgs(u):
    c.rel_standing_envs = standing
    c.rel_rotation_envs = 0.0
u.reset()
runner = load_runner_cls(TASK)(env, asdict(load_rl_cfg(TASK)), device="cuda:0")
runner.load(CKPT, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
policy = runner.get_inference_policy("cuda:0")
robot = u.scene["robot"]
names = list(u.action_manager.get_term("joint_pos").target_names)
legs = np.array([("hip" in n or "knee" in n or "ankle" in n) for n in names])

T = int(float(os.environ.get("SECONDS", 10)) / u.step_dt)
A, GYRO = [], []
alive = np.ones(u.num_envs, bool)
obs = env.get_observations()
with torch.no_grad():
    for _ in range(T):
        action = policy(obs)
        obs, _, dones, _ = env.step(action)
        alive &= ~dones.cpu().numpy().astype(bool)
        A.append(action.cpu().numpy())
        GYRO.append(robot.data.root_link_ang_vel_b.cpu().numpy())
A, GYRO = np.stack(A)[T // 5:, alive], np.stack(GYRO)[T // 5:, alive]
d1 = np.diff(A, axis=0)
d2 = np.diff(A, 2, axis=0)
spec = np.abs(np.fft.rfft(A - A.mean(0), axis=0)) ** 2
freqs = np.fft.rfftfreq(A.shape[0], u.step_dt)
hf = spec[freqs > 6.0].sum(0) / spec[freqs > 0.0].sum(0).clip(min=1e-12)
print(f"{TASK} {os.path.basename(CKPT)} vx {VX} wz {WZ} | alive {alive.mean():.2f}")
for label, m in (("legs", legs), ("arms", ~legs)):
    print(f"  {label}: |da| rms {np.sqrt((d1[..., m] ** 2).mean()):.4f} rad/tick"
          f"  |d2a| rms {np.sqrt((d2[..., m] ** 2).mean()):.4f}  power >6 Hz {hf[:, m].mean():.2f}")
print(f"  trunk gyro rms {np.sqrt((GYRO ** 2).mean(0)).mean(0).round(3)} rad/s")
