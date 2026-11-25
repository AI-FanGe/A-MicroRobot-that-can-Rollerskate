"""Walk policy at fixed supply voltages: falls, trunk pitch and speed tracking (play mode,
random walk commands, no pushes, the usual load-dependent voltage drop on top).

usage: XIAOXIAOBAI_BASE=24996 walk_voltage_eval.py <ckpt> [vin ...]   (default 7.5 6.5 6.0 5.5)
env:   NUM_ENVS=<n> (default 1024)  SECONDS=<s> (default 20)
       TASK=<id> (default Mjlab-Velocity-Xiaoxiaobai; Mjlab-Velocity-Xiaoxiaobai-FreeArms for the
       free-arms walk, whose arms play the motion bank on the same supply)

Pitch is + leaning forward. fwd / back: envs commanded vx > 0.2 / < -0.2; sway is the std of
each env's pitch over time (trunk rocking), averaged over the envs.
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

TASK = os.environ.get("TASK", "Mjlab-Velocity-Xiaoxiaobai")
ckpt = sys.argv[1]
vins = [float(v) for v in sys.argv[2:]] or [7.5, 6.5, 6.0, 5.5]
seconds = float(os.environ.get("SECONDS", 20))
cfg = load_env_cfg(TASK, play=True)
cfg.scene.num_envs = int(os.environ.get("NUM_ENVS", 1024))
cfg.episode_length_s = seconds + 5.0
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device="cuda:0"))
u = env.unwrapped
runner = load_runner_cls(TASK)(env, asdict(load_rl_cfg(TASK)), device="cuda:0")
runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
policy = runner.get_inference_policy("cuda:0")
robot = u.scene["robot"]
acts = robot.actuators
# The training floor (vin_min) would lift every voltage below it back up.
for act in acts:
    act.cfg.vin_min = 4.0
cmd = u.command_manager.get_term("twist")
print(f"{TASK} {ckpt}")
for vin in vins:
    for act in acts:
        act.vin_tensor[:] = vin
    obs, _ = env.reset()
    fell = torch.zeros(u.num_envs, dtype=torch.bool, device=u.device)
    pitch, err = [], []
    with torch.no_grad():
        for t in range(int(seconds / u.step_dt)):
            obs, _, dones, extras = env.step(policy(obs))
            fell |= dones.bool() & ~extras["time_outs"].bool()
            if t >= 50:
                g = robot.data.projected_gravity_b
                pitch.append(torch.rad2deg(torch.atan2(g[:, 0], -g[:, 2])).cpu().numpy())
                v = robot.data.root_link_lin_vel_b[:, :2]
                err.append((v - cmd.command[:, :2]).norm(dim=-1).cpu().numpy())
    pitch, err = np.stack(pitch), np.stack(err)
    vx = cmd.command[:, 0].cpu().numpy()
    fwd, back = vx > 0.2, vx < -0.2
    up = ~fell.cpu().numpy()
    sway = pitch[:, up].std(axis=0).mean()
    print(f"vin {vin:.1f} V: fell {fell.float().mean().item() * 100:5.1f}% | pitch mean {pitch.mean():+.1f} deg"
          f" (fwd {pitch[:, fwd].mean():+.1f}, back {pitch[:, back].mean():+.1f}), |pitch| p95 {np.percentile(np.abs(pitch), 95):.1f}"
          f", sway {sway:.2f} | speed error {err.mean():.3f} m/s", flush=True)
