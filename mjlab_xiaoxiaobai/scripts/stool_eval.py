"""Sit down on the box behind, rest, then stand back up (Mjlab-Sit-Xiaoxiaobai), as the robot
would be commanded: sit for SIT_S, then stand for STAND_S. Play mode, deterministic policy.
Results per seat height.

usage: stool_eval.py <ckpt>
env:   TASK=<id>      Mjlab-Sit-Xiaoxiaobai (default) or Mjlab-Sit-Xiaoxiaobai-Backup
       SIT_S=<s>      seconds of the sit command (default 6)
       STAND_S=<s>    seconds of the stand command after it (default 6)
       NUM_ENVS=<n>   (default 1024; fewer to fit next to a training run)
       VIDEO=<dir>    instead of the statistics, one clip per episode into <dir>
       EPISODES=<n>   clips with VIDEO (default 4)
"""
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import mjlab_xiaoxiaobai.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab_xiaoxiaobai.tasks import mdp as xiaoxiaobai_mdp

TASK = os.environ.get("TASK", "Mjlab-Sit-Xiaoxiaobai")
sit_s = float(os.environ.get("SIT_S", 6.0))
stand_s = float(os.environ.get("STAND_S", 6.0))
video = os.environ.get("VIDEO")
cfg = load_env_cfg(TASK, play=True)
cfg.scene.num_envs = 1 if video else int(os.environ.get("NUM_ENVS", 1024))
cfg.episode_length_s = sit_s + stand_s + 5.0
cfg.commands["sit"].resampling_time_range = (1e6, 1e6)
cfg.commands["sit"].start_sit_prob = 1.0
if video:
    cfg.viewer.width, cfg.viewer.height = 640, 480
u = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode="rgb_array" if video else None)
env = RslRlVecEnvWrapper(u)
runner = load_runner_cls(TASK)(env, asdict(load_rl_cfg(TASK)), device="cuda:0")
runner.load(sys.argv[1], load_cfg={"actor": True}, strict=True, map_location="cuda:0")
policy = runner.get_inference_policy("cuda:0")
robot = u.scene["robot"]
command = u.command_manager.get_term("sit")
switch = int(sit_s / u.step_dt)
steps = switch + int(stand_s / u.step_dt)


def run(obs, record=None):
    H, G, S, C, F = [], [], [], [], []
    with torch.no_grad():
        for t in range(steps):
            if t == switch:
                command.sit[:] = 0.0
                obs = env.get_observations()
            obs, _, dones, _ = env.step(policy(obs))
            H.append((robot.data.root_link_pos_w[:, 2] - u.scene.env_origins[:, 2]).cpu().numpy())
            G.append(robot.data.projected_gravity_b[:, 2].cpu().numpy())
            S.append(xiaoxiaobai_mdp._seat_support(u, "seat_contact").cpu().numpy())
            C.append(xiaoxiaobai_mdp.total_current_excess(u, max_current=0.0).cpu().numpy())
            F.append(u.termination_manager.terminated.cpu().numpy())
            if record is not None:
                record.append_data(np.asarray(u.render(), dtype=np.uint8))
    return np.stack(H), np.stack(G), np.stack(S), np.stack(C), np.stack(F)


if video:
    import imageio

    out = Path(video)
    out.mkdir(parents=True, exist_ok=True)
    for ep in range(int(os.environ.get("EPISODES", 4))):
        obs, _ = env.reset()
        command.sit[:] = 1.0
        seat_h = 2 * float(u._seat_half_size[0, 2])
        path = out / f"{ep:02d}_seat{seat_h * 100:.0f}cm.mp4"
        with imageio.get_writer(str(path), fps=round(1 / u.step_dt), codec="libx264", quality=8, macro_block_size=1) as w:
            H, G, S, C, F = run(obs, w)
        stood = np.flatnonzero((H[switch:, 0] > 0.16) & (G[switch:, 0] < -0.95))
        print(f"{path.name}: seat {seat_h * 100:.1f} cm | on seat {S[switch - 1, 0]:.2f} of the weight, "
              f"current {C[switch - 50:switch, 0].mean():.2f} A | fell {bool(F.any())} | stood after "
              f"{stood[0] * u.step_dt if len(stood) else None} s", flush=True)
    sys.exit()

obs = env.get_observations()
seat_h = 2 * u._seat_half_size[:, 2].cpu().numpy()
gap = u._seat_gap.cpu().numpy()
H, G, S, C, F = run(obs)
second = round(1.0 / u.step_dt)
fell_sit = F[:switch].any(0)
fell = F.any(0)
seated = ~fell_sit & (S[switch - second:switch].mean(0) > 0.5) & (G[switch - 1] < -0.8)
sit_current = C[switch - second:switch].mean(0)
stood = ~fell & (H[-1] > 0.16) & (G[-1] < -0.95)
stand = (H[switch:] > 0.16) & (G[switch:] < -0.95)
first = np.where(stand.any(0), stand.argmax(0), -1)
stand_current = C[-second:].mean(0)
print(f"n={u.num_envs}  sit {sit_s:.0f} s, then stand {stand_s:.0f} s")
print("seat [cm]   n    seated  fell sitting  current seated  stood up  fell standing  time to stand")
for lo, hi in ((0.08, 0.10), (0.10, 0.12), (0.12, 0.14), (0.08, 0.14)):
    m = (seat_h >= lo) & (seat_h < hi + 1e-9)
    ok = first[m & stood]
    print(f"{lo * 100:4.0f}-{hi * 100:<4.0f} {m.sum():5d}   {seated[m].mean():.2f}    {fell_sit[m].mean():.2f}"
          f"         {np.median(sit_current[m & seated]) if (m & seated).any() else float('nan'):.2f} A"
          f"        {stood[m].mean():.2f}      {(fell & ~fell_sit)[m].mean():.2f}"
          f"           {np.median(ok) * u.step_dt if len(ok) else float('nan'):.1f} s")
print("box behind the trunk [cm]   n    seated  fell sitting  stood up")
for lo, hi in ((0.03, 0.09), (0.09, 0.16), (0.16, 0.23), (0.23, 0.30)):
    m = (gap >= lo) & (gap < hi)
    if m.any():
        print(f"   {lo * 100:4.0f}-{hi * 100:<4.0f}              {m.sum():5d}   {seated[m].mean():.2f}"
              f"    {fell_sit[m].mean():.2f}          {stood[m].mean():.2f}")
print(f"standing current at the end p50 {np.median(stand_current[stood]) if stood.any() else float('nan'):.2f} A")
