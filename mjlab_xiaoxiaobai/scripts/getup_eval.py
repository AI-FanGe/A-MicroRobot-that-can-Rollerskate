"""Evaluate a get-up checkpoint in play mode (no assist): success by starting pose, time to
stand, and how dynamic the get-up is.

usage: getup_eval.py <ckpt>
env:   FRICTION=<mu>  every geom's sliding friction
       FLAT=<0..1>    share of fallen starts lying flat on the back or front
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
cfg = load_env_cfg(TASK, play=True)
cfg.scene.num_envs = 1024
if os.environ.get("FLAT"):
    cfg.events["reset_getup"].params["flat_fraction"] = float(os.environ["FLAT"])
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device="cuda:0"))
u = env.unwrapped
if os.environ.get("FRICTION"):
    u.sim.model.geom_friction[:, :, 0] = float(os.environ["FRICTION"])
runner = load_runner_cls(TASK)(env, asdict(load_rl_cfg(TASK)), device="cuda:0")
runner.load(sys.argv[1], load_cfg={"actor": True}, strict=True, map_location="cuda:0")
policy = runner.get_inference_policy("cuda:0")
robot = u.scene["robot"]
act = [i for i, n in enumerate(robot.joint_names) if not (n.startswith("head") or n.startswith("leg_flex"))]
obs = env.get_observations()
T = 495
H, G, GX, W, VB, VZ, QD, C = ([] for _ in range(8))
with torch.no_grad():
    for t in range(T):
        obs, *_ = env.step(policy(obs))
        g = robot.data.projected_gravity_b
        H.append((robot.data.root_link_pos_w[:, 2] - u.scene.env_origins[:, 2]).cpu().numpy())
        G.append(g[:, 2].cpu().numpy())
        GX.append(g[:, 0].cpu().numpy())
        W.append(robot.data.root_link_ang_vel_b.norm(dim=-1).cpu().numpy())
        VB.append(robot.data.body_link_lin_vel_w.norm(dim=-1).max(dim=1).values.cpu().numpy())
        VZ.append(robot.data.root_link_lin_vel_w[:, 2].cpu().numpy())
        QD.append(robot.data.joint_vel[:, act].abs().max(dim=1).values.cpu().numpy())
        C.append((u.sim.data.actuator_force.abs() / 0.366).sum(dim=1).cpu().numpy())
H, G, GX, W, VB, VZ, QD, C = map(np.stack, (H, G, GX, W, VB, VZ, QD, C))
stand = (H > 0.16) & (G < -0.95)
started = stand[0]
fallen = ~started
print(f"started fallen {fallen.sum()}: ever stood {stand.any(0)[fallen].mean():.2f}, standing at end {stand[-1][fallen].mean():.2f}"
      f" | started standing {started.sum()}: still standing {stand[-1][started].mean():.2f}")
gx0 = GX[5]
for name, m in (("on back", gx0 < -0.7), ("on front", gx0 > 0.7), ("other", np.abs(gx0) <= 0.7)):
    m = m & fallen
    if m.sum():
        print(f"  {name:9s} n={m.sum():4d} standing at end {stand[-1][m].mean():.2f}")
first = np.where(stand.any(0), stand.argmax(0), -1)[fallen]
ok = first[first >= 0]
if len(ok):
    print("time to first stand p25/50/75 [s]", (np.percentile(ok, [25, 50, 75]) * 0.02).round(2))
win = slice(10, 150)
pk = lambda X: (np.median(X[win][:, fallen].max(0)), np.percentile(X[win][:, fallen].max(0), 90))
print("getting up (0.2-3 s), peak per env p50/p90: trunk ang vel %.1f/%.1f rad/s | body speed %.2f/%.2f m/s | trunk vz %.2f/%.2f m/s | joint vel %.1f/%.1f rad/s | current %.1f/%.1f A" % (
    *pk(W), *pk(VB), *pk(VZ), *pk(QD), *pk(C)))
