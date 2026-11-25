"""Roll-over then get-up, as the robot runs them (xiaoxiaobai GetupMove): from flat starts, the
roll-over policy runs until the robot lies face down and still, then the get-up policy.
Play mode, no assist.

usage: getup_chain_eval.py <rollover_ckpt> <getup_ckpt>
env:   SUPINE=<0..1>  share of flat starts on the back (default 1)
       FRICTION=<mu>  every geom's sliding friction
       NUM_ENVS=<n>   (default 1024; fewer to fit next to a training run)
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
from mjlab_xiaoxiaobai.tasks.xiaoxiaobai_rollover_env_cfg import HANDOVER_HOLD_S, HANDOVER_MAX_ANG_VEL, HANDOVER_MIN_GX

TASK = "Mjlab-Getup-Xiaoxiaobai"
cfg = load_env_cfg(TASK, play=True)
cfg.scene.num_envs = int(os.environ.get("NUM_ENVS", 1024))
cfg.episode_length_s = 20.0
cfg.events["reset_getup"].params.update(
    {"standing_fraction": 0.0, "flat_fraction": 1.0, "supine_fraction": float(os.environ.get("SUPINE", 1.0))}
)
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device="cuda:0"))
u = env.unwrapped
if os.environ.get("FRICTION"):
    u.sim.model.geom_friction[:, :, 0] = float(os.environ["FRICTION"])


def load(ckpt):
    runner = load_runner_cls(TASK)(env, asdict(load_rl_cfg(TASK)), device="cuda:0")
    runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
    return runner.get_inference_policy("cuda:0")


rollover, getup = load(sys.argv[1]), load(sys.argv[2])
robot = u.scene["robot"]
obs = env.get_observations()
T = int(cfg.episode_length_s / u.step_dt) - 5
hold = round(HANDOVER_HOLD_S / u.step_dt)
g = robot.data.projected_gravity_b
on_back = g[:, 0] < 0.0
rolling = g[:, 0] <= HANDOVER_MIN_GX
ticks = torch.zeros(u.num_envs, dtype=torch.long, device=u.device)
handover = torch.full((u.num_envs,), -1, dtype=torch.long, device=u.device)
H, G = [], []
with torch.no_grad():
    for t in range(T):
        g = robot.data.projected_gravity_b
        ready = (g[:, 0] > HANDOVER_MIN_GX) & (robot.data.root_link_ang_vel_b.norm(dim=-1) < HANDOVER_MAX_ANG_VEL)
        ticks = torch.where(rolling & ready, ticks + 1, 0)
        done = rolling & (ticks >= hold)
        handover[done] = t
        rolling &= ~done
        action = torch.where(rolling.unsqueeze(-1), rollover(obs), getup(obs))
        obs, *_ = env.step(action)
        H.append((robot.data.root_link_pos_w[:, 2] - u.scene.env_origins[:, 2]).cpu().numpy())
        G.append(robot.data.projected_gravity_b[:, 2].cpu().numpy())
H, G = np.stack(H), np.stack(G)
stand = (H > 0.16) & (G < -0.95)
back = on_back.cpu().numpy()
ho = handover.cpu().numpy()
first = np.where(stand.any(0), stand.argmax(0), -1)
for name, m in (("on back", back), ("on front", ~back)):
    if m.sum():
        rolled = (ho[m] >= 0).mean() if name == "on back" else float("nan")
        ok = first[m][first[m] >= 0]
        t50 = np.median(ok) * u.step_dt if len(ok) else float("nan")
        print(f"{name:9s} n={m.sum():4d} rolled {rolled:.2f} | standing at end {stand[-1][m].mean():.2f}"
              f" | median time to stand {t50:.1f} s")
ok = ho[back & (ho >= 0)]
if len(ok):
    print("roll-over time p25/50/75 [s]", (np.percentile(ok, [25, 50, 75]) * u.step_dt).round(2))
