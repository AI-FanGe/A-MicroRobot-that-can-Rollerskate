"""Standing push recovery and standing jitter under supply voltages.

Every env stands (zero command). After SETTLE_S it gets one push, a trunk velocity change of
one of the MAGS [m/s] in one of 8 directions, and the falls over the next HOLD_S are counted
per magnitude and direction. Before the push it records the quiet-standing jitter: the mean
per-tick change of the policy action and the RMS trunk angular velocity.

usage: XIAOXIAOBAI_BASE=24996 stand_push_eval.py <ckpt> [vin ...]   (default 7.2 6.0 5.0)
env:   NUM_ENVS (default 2048)  SETTLE_S (default 3)  HOLD_S (default 3)
       MAGS (default 0.3,0.5,0.7,0.9)
"""
import os
import sys
from dataclasses import asdict

import torch
import mjlab_xiaoxiaobai.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK = "Mjlab-Velocity-Xiaoxiaobai"
DIRS = ("fwd", "fwd-L", "left", "back-L", "back", "back-R", "right", "fwd-R")
ckpt = sys.argv[1]
vins = [float(v) for v in sys.argv[2:]] or [7.2, 6.0, 5.0]
settle_s, hold_s = float(os.environ.get("SETTLE_S", 3)), float(os.environ.get("HOLD_S", 3))
mags = [float(m) for m in os.environ.get("MAGS", "0.3,0.5,0.7,0.9").split(",")]

cfg = load_env_cfg(TASK, play=True)
cfg.scene.num_envs = int(os.environ.get("NUM_ENVS", 2048))
cfg.episode_length_s = settle_s + hold_s + 5.0
cfg.events.pop("push_robot", None)
cfg.commands["twist"].resampling_time_range = (1e6, 1e6)
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device="cuda:0"))
u = env.unwrapped
n = u.num_envs
runner = load_runner_cls(TASK)(env, asdict(load_rl_cfg(TASK)), device="cuda:0")
runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
policy = runner.get_inference_policy("cuda:0")
robot = u.scene["robot"]
for act in robot.actuators:
    act.cfg.vin_min = 4.0
cmd = u.command_manager.get_term("twist")

idx = torch.arange(n, device=u.device)
dir_group, mag_group = idx % len(DIRS), (idx // len(DIRS)) % len(mags)
angle = dir_group.float() * (2 * torch.pi / len(DIRS))
mag = torch.tensor(mags, device=u.device)[mag_group]
push = torch.zeros(n, 6, device=u.device)
push[:, 0], push[:, 1] = mag * torch.cos(angle), mag * torch.sin(angle)

print(f"{ckpt}  (push at {settle_s:g} s, falls within {hold_s:g} s)")
for vin in vins:
    for act in robot.actuators:
        act.vin_tensor[:] = vin
    obs, _ = env.reset()
    fell_before = torch.zeros(n, dtype=torch.bool, device=u.device)
    fell = torch.zeros(n, dtype=torch.bool, device=u.device)
    rate, gyro = [], []
    prev = None
    push_tick = int(settle_s / u.step_dt)
    with torch.no_grad():
        for t in range(push_tick + int(hold_s / u.step_dt)):
            cmd.vel_command_b[:] = 0.0
            if t == push_tick:
                vel = robot.data.root_link_vel_w.clone()
                vel[:, :2] += push[:, :2]
                robot.write_root_link_velocity_to_sim(vel)
            action = policy(obs)
            if t < push_tick and t * u.step_dt >= 1.0:
                if prev is not None:
                    rate.append((action - prev).abs().mean(dim=1))
                gyro.append(robot.data.root_link_ang_vel_b.square().sum(dim=1))
            prev = action
            obs, _, dones, extras = env.step(action)
            down = dones.bool() & ~extras["time_outs"].bool()
            if t < push_tick:
                fell_before |= down
            else:
                fell |= down & ~fell_before
    ok = ~fell_before
    rate, gyro = torch.stack(rate).mean(0), torch.stack(gyro).mean(0).sqrt()
    print(f"vin {vin:.1f} V  fell before push {fell_before.float().mean().item() * 100:.1f}%"
          f" | standing jitter: action change {rate[ok].mean().item():.4f}/tick, trunk ang vel rms {gyro[ok].mean().item():.3f} rad/s")
    print("  fell %   " + " ".join(f"{d:>6s}" for d in DIRS) + "    all")
    for m, value in enumerate(mags):
        row = [fell[ok & (mag_group == m) & (dir_group == d)].float().mean().item() * 100 for d in range(len(DIRS))]
        all_ = fell[ok & (mag_group == m)].float().mean().item() * 100
        print(f"  {value:.1f} m/s " + " ".join(f"{r:6.0f}" for r in row) + f" {all_:6.1f}", flush=True)
