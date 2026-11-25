"""Video of roll-over then get-up as the robot runs them (xiaoxiaobai GetupMove), one clip per
episode from flat on the back (or front), deterministic policies, play mode, no assist.

usage: getup_chain_video.py <rollover_ckpt> <getup_ckpt> <out_dir>
env:   EPISODES=<n> (default 4)  SUPINE=<0..1> (default 1)  SECONDS=<s> (default 12)
"""
import os
import sys
from dataclasses import asdict
from pathlib import Path

import imageio
import numpy as np
import torch
import mjlab_xiaoxiaobai.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab_xiaoxiaobai.tasks.xiaoxiaobai_rollover_env_cfg import HANDOVER_HOLD_S, HANDOVER_MAX_ANG_VEL, HANDOVER_MIN_GX

TASK = "Mjlab-Getup-Xiaoxiaobai"
seconds = float(os.environ.get("SECONDS", 12))
cfg = load_env_cfg(TASK, play=True)
cfg.scene.num_envs = 1
cfg.episode_length_s = seconds + 5.0
cfg.events["reset_getup"].params.update(
    {"standing_fraction": 0.0, "flat_fraction": 1.0, "supine_fraction": float(os.environ.get("SUPINE", 1.0))}
)
cfg.viewer.width, cfg.viewer.height = 640, 480
u = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode="rgb_array")
env = RslRlVecEnvWrapper(u)


def load(ckpt):
    runner = load_runner_cls(TASK)(env, asdict(load_rl_cfg(TASK)), device="cuda:0")
    runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
    return runner.get_inference_policy("cuda:0")


rollover, getup = load(sys.argv[1]), load(sys.argv[2])
out = Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
robot = u.scene["robot"]
hold = round(HANDOVER_HOLD_S / u.step_dt)
for ep in range(int(os.environ.get("EPISODES", 4))):
    obs, _ = env.reset()
    g = robot.data.projected_gravity_b[0]
    start = "back" if g[0] < 0 else "front"
    rolling = bool(g[0] <= HANDOVER_MIN_GX)
    ticks, handover, stood = 0, None, None
    path = out / f"{ep:02d}_{start}.mp4"
    with imageio.get_writer(str(path), fps=round(1 / u.step_dt), codec="libx264", quality=8, macro_block_size=1) as w:
        for t in range(int(seconds / u.step_dt)):
            g = robot.data.projected_gravity_b[0]
            if rolling:
                still = robot.data.root_link_ang_vel_b[0].norm() < HANDOVER_MAX_ANG_VEL
                ticks = ticks + 1 if (g[0] > HANDOVER_MIN_GX and still) else 0
                if ticks >= hold:
                    rolling, handover = False, t * u.step_dt
            with torch.no_grad():
                obs, *_ = env.step(rollover(obs) if rolling else getup(obs))
            h = float(robot.data.root_link_pos_w[0, 2] - u.scene.env_origins[0, 2])
            if stood is None and h > 0.16 and float(robot.data.projected_gravity_b[0, 2]) < -0.95:
                stood = t * u.step_dt
            w.append_data(np.asarray(u.render(), dtype=np.uint8))
    print(f"{path.name}: start on {start}, handed over at {handover} s, stood at {stood} s", flush=True)
