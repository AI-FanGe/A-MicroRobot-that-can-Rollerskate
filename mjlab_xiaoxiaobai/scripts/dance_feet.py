"""Foot lifts of a dance checkpoint against its reference: how high each sole goes per step of the
reference, and how far the planted feet slide.

usage: dance_feet.py <ckpt>
env:   NUM_ENVS (64)
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

TASK = "Mjlab-Dance-Xiaoxiaobai"
cfg = load_env_cfg(TASK, play=True)
cfg.scene.num_envs = int(os.environ.get("NUM_ENVS", 64))
cfg.scene.terrain.terrain_type = "plane"
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device="cuda:0"))
u = env.unwrapped
runner = load_runner_cls(TASK)(env, asdict(load_rl_cfg(TASK)), device="cuda:0")
runner.load(sys.argv[1], load_cfg={"actor": True}, strict=True, map_location="cuda:0")
policy = runner.get_inference_policy("cuda:0")

cmd = u.command_manager.get_term("motion")
robot = u.scene["robot"]
feet = [cmd.cfg.body_names.index(n) for n in ("foot_2", "foot")]
T = cmd.motion.time_step_total - 1
fps = 1.0 / u.step_dt
REF, ROB, VEL, ALIVE = [], [], [], []
alive = np.ones(u.num_envs, bool)
obs = env.get_observations()
with torch.no_grad():
    for t in range(T):
        obs, _, dones, _ = env.step(policy(obs))
        alive &= ~dones.cpu().numpy().astype(bool)
        REF.append(cmd.body_pos_w[:, feet, 2].cpu().numpy() - u.scene.env_origins[:, None, 2].cpu().numpy())
        ROB.append(cmd.robot_body_pos_w[:, feet, 2].cpu().numpy() - u.scene.env_origins[:, None, 2].cpu().numpy())
        VEL.append(torch.norm(robot.data.body_link_lin_vel_w[:, [robot.body_names.index(n) for n in ("foot_2", "foot")], :2], dim=-1).cpu().numpy())
        ALIVE.append(alive.copy())
REF, ROB, VEL, ALIVE = map(np.stack, (REF, ROB, VEL, ALIVE))
ref = REF[:, 0]
base = np.percentile(ref, 5, axis=0)
rob_base = np.percentile(ROB, 5, axis=0)  # per robot and foot
print(f"{alive.mean():.0%} of {u.num_envs} danced it through | foot origin height when planted: ref {base * 1000} mm")

for i, side in enumerate(("left", "right")):
    lift = ref[:, i] - base[i]
    up = lift > 0.006
    edges = np.flatnonzero(np.diff(up.astype(int)))
    starts, ends = edges[::2] + 1, edges[1::2] + 1
    print(f"{side} foot: reference lifts > 6 mm: {len(ends)}")
    print("    t[s]   dur[s]  ref peak[mm]  robot peak[mm] (median, p10..p90)  robot planted slide[mm/s]")
    for a, b in zip(starts, ends):
        if b - a < 3:
            continue
        rob = (ROB[a:b, :, i] - rob_base[:, i])[:, ALIVE[b]]
        peaks = rob.max(0) * 1000
        print(f"  {a / fps:6.1f}  {(b - a) / fps:6.2f}  {lift[a:b].max() * 1000:12.0f}  "
              f"{np.median(peaks):14.0f} ({np.percentile(peaks, 10):.0f}..{np.percentile(peaks, 90):.0f})")
    planted = ~up
    v = VEL[:, :, i][planted][:, ALIVE[-1]] * 1000
    print(f"  while the reference foot is planted, the robot's foot moves {np.median(v):.0f} mm/s median, p90 {np.percentile(v, 90):.0f}")
