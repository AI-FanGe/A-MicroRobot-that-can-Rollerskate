"""Falls and posture while the battery runs down mid-walk.

Every env walks for DURATION_S while the supply voltage falls linearly from V_START to V_END
(BAM's load-dependent sag on top, as in training). The command changes every 1.5-4 s to one of
vx +-0.4 / +-0.2, a turn at +-0.75 rad/s while walking, or standing, through the robot's command
ramp (moves/walk.py). Per voltage band it prints the share of the envs still up at the start of
the band that fell in it, and while walking: the speed reached as a share of the command, knee
flexion and trunk pitch (+ forward) at vx > 0 / < 0, and the trunk height. A fallen env is not
counted again.

usage: XIAOXIAOBAI_BASE=24996 walk_discharge_eval.py <ckpt> [ckpt ...]
env:   NUM_ENVS (default 3000)  DURATION_S (default 90)  V_START (7.6)  V_END (5.0)  SEED (0)
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
ckpts = sys.argv[1:]
duration = float(os.environ.get("DURATION_S", 90))
v_start, v_end = float(os.environ.get("V_START", 7.6)), float(os.environ.get("V_END", 5.0))
seed = int(os.environ.get("SEED", 0))

cfg = load_env_cfg(TASK, play=True)
cfg.scene.num_envs = int(os.environ.get("NUM_ENVS", 3000))
cfg.episode_length_s = duration + 5.0
cfg.events.pop("push_robot", None)
cfg.events.pop("battery_discharge", None)
cfg.events.pop("battery_discharge_step", None)
cfg.commands["twist"].resampling_time_range = (1e6, 1e6)
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device="cuda:0"))
u = env.unwrapped
n, dev, dt = u.num_envs, u.device, u.step_dt
robot = u.scene["robot"]
for act in robot.actuators:
    act.cfg.vin_min = 4.0
cmd = u.command_manager.get_term("twist")
runner = load_runner_cls(TASK)(env, asdict(load_rl_cfg(TASK)), device="cuda:0")
knee_ids = robot.find_joints((r".*knee.*",))[0]
knee_sign = torch.sign(robot.data.joint_pos_limits[0, knee_ids].sum(dim=-1))

full_speed = torch.tensor([0.5, 0.3, 1.5], device=dev)
choices = torch.tensor(
    [[0.4, 0, 0], [-0.4, 0, 0], [0.2, 0, 0], [-0.2, 0, 0], [0.3, 0, 0.75], [0.3, 0, -0.75], [-0.3, 0, 0.75], [0, 0, 0]],
    device=dev,
)
bands = [(7.6, 7.0), (7.0, 6.5), (6.5, 6.0), (6.0, 5.5), (5.5, 5.0)]


def ramp(cur: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    slowing = (target.abs() < cur.abs()) | (target * cur < 0)
    rate = full_speed * torch.where(slowing, 1.0 / 0.8, 1.0 / 1.0)
    return cur + torch.clamp(target - cur, -rate * dt, rate * dt)


print(f"battery {v_start} -> {v_end} V over {duration:g} s, {n} envs; per band: fell % of those up | "
      "speed/cmd, knee fwd/back deg, pitch fwd/back deg, height mm")
for ckpt in ckpts:
    runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
    policy = runner.get_inference_policy("cuda:0")
    gen = torch.Generator(device=dev).manual_seed(seed)
    for act in robot.actuators:
        act.vin_tensor[:] = v_start
    obs, _ = env.reset()
    command = torch.zeros(n, 3, device=dev)
    target = choices[torch.randint(len(choices), (n,), device=dev, generator=gen)]
    next_change = torch.rand(n, device=dev, generator=gen) * 2.5 + 1.5
    down = torch.zeros(n, dtype=torch.bool, device=dev)
    stats = {b: {k: [] for k in ("up", "fell", "speed", "kf", "kb", "pf", "pb", "h")} for b in bands}
    with torch.no_grad():
        for t in range(int(duration / dt)):
            time = t * dt
            vin = v_start + (v_end - v_start) * time / duration
            for act in robot.actuators:
                act.vin_tensor[:] = vin
            band = next(b for b in bands if b[1] <= vin <= b[0] + 1e-6)
            change = time >= next_change
            if change.any():
                target[change] = choices[torch.randint(len(choices), (int(change.sum()),), device=dev, generator=gen)]
                next_change[change] = time + torch.rand(int(change.sum()), device=dev, generator=gen) * 2.5 + 1.5
            command = ramp(command, target)
            cmd.vel_command_b[:] = command
            obs, _, dones, extras = env.step(policy(obs))
            fell = dones.bool() & ~extras["time_outs"].bool() & ~down
            s = stats[band]
            if not s["up"]:
                s["up"].append((~down).sum().item())
            s["fell"].append(fell.sum().item())
            down |= fell
            up = ~down
            g = robot.data.projected_gravity_b
            pitch = torch.rad2deg(torch.atan2(g[:, 0], -g[:, 2]))
            knee = torch.rad2deg((robot.data.joint_pos[:, knee_ids] * knee_sign).mean(dim=1))
            vx = robot.data.root_link_lin_vel_b[:, 0]
            fwd, back = up & (command[:, 0] > 0.15), up & (command[:, 0] < -0.15)
            straight = (fwd | back) & (command[:, 2].abs() < 0.05) & (command[:, 0].abs() > 0.35)
            if straight.any():
                s["speed"].append((vx[straight] / command[straight, 0]).mean().item())
            if fwd.any():
                s["kf"].append(knee[fwd].mean().item())
                s["pf"].append(pitch[fwd].mean().item())
            if back.any():
                s["kb"].append(knee[back].mean().item())
                s["pb"].append(pitch[back].mean().item())
            s["h"].append(robot.data.root_link_pos_w[up, 2].mean().item() * 1000)
    print(f"{ckpt}  total fell {down.float().mean().item() * 100:.1f}%")
    for b in bands:
        s = stats[b]
        if not s["up"]:
            continue
        m = lambda k: sum(s[k]) / max(len(s[k]), 1)  # noqa: E731
        print(f"  {b[0]:.1f}-{b[1]:.1f} V  fell {sum(s['fell']) / max(s['up'][0], 1) * 100:5.1f}% | speed {m('speed'):4.2f}"
              f"  knee {m('kf'):5.1f} /{m('kb'):5.1f}  pitch {m('pf'):5.1f} /{m('pb'):5.1f}  height {m('h'):5.1f}", flush=True)
