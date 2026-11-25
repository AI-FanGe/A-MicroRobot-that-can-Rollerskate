"""Falls on sudden direction changes while walking.

Every env walks a fixed command for SWITCH_S, switches to a second one, switches back at
2 * SWITCH_S and walks on to 3 * SWITCH_S. Scenarios (VX m/s, WZ rad/s, turn sign alternating):
  fwd>back   +VX -> -VX            back>fwd   -VX -> +VX
  fwd>turn   +VX -> +VX, +-WZ      back>turn  -VX -> -VX, +-WZ
  fwd>back+turn  +VX -> -VX, +-WZ
each either through the robot's command ramp (moves/walk.py: full speed 0.5 / 0.3 / 1.5 in
1.0 s speeding up, 0.8 s slowing down) or as an instant step. Per supply voltage it prints
the falls before the first switch (steady walking) and after it, the mean time until the
trunk's vx has half the new command's sign and size (reversals), and the pitch overshoot
after the first switch: how far the trunk pitch leaves the range between the mean pitch of
the last second before the switch and of the last second before switching back (a
reversal moves the trunk between two leans on purpose).

usage: XIAOXIAOBAI_BASE=24996 walk_switch_eval.py <ckpt> [vin ...]   (default 7.2 6.0 5.0)
env:   NUM_ENVS (default 4000)  SWITCH_S (default 4)  VX (default 0.4)  WZ (default 0.75)
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
ckpt = sys.argv[1]
vins = [float(v) for v in sys.argv[2:]] or [7.2, 6.0, 5.0]
switch_s = float(os.environ.get("SWITCH_S", 4))
vx, wz = float(os.environ.get("VX", 0.4)), float(os.environ.get("WZ", 0.75))

cfg = load_env_cfg(TASK, play=True)
cfg.scene.num_envs = int(os.environ.get("NUM_ENVS", 4000))
cfg.episode_length_s = 3 * switch_s + 5.0
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
dev = u.device

scenarios = [  # name, first command, second command (turn sign set per env)
    ("fwd>back", (vx, 0.0), (-vx, 0.0)),
    ("back>fwd", (-vx, 0.0), (vx, 0.0)),
    ("fwd>turn", (vx, 0.0), (vx, wz)),
    ("back>turn", (-vx, 0.0), (-vx, wz)),
    ("fwd>back+turn", (vx, 0.0), (-vx, wz)),
]
idx = torch.arange(n, device=dev)
scen, ramped = idx % len(scenarios), (idx // len(scenarios)) % 2 == 0
turn_sign = torch.where((idx // (2 * len(scenarios))) % 2 == 0, 1.0, -1.0).to(dev)
first, second = torch.zeros(n, 3, device=dev), torch.zeros(n, 3, device=dev)
for s, (_, a, b) in enumerate(scenarios):
    first[scen == s, 0], first[scen == s, 2] = a[0], a[1]
    second[scen == s, 0], second[scen == s, 2] = b[0], b[1]
second[:, 2] *= turn_sign
reversal = (first[:, 0] * second[:, 0]) < 0

full_speed = torch.tensor([0.5, 0.3, 1.5], device=dev)


def ramp(cur: torch.Tensor, target: torch.Tensor, dt: float) -> torch.Tensor:
    slowing = (target.abs() < cur.abs()) | (target * cur < 0)
    rate = full_speed * torch.where(slowing, 1.0 / 0.8, 1.0 / 1.0)
    return cur + torch.clamp(target - cur, -rate * dt, rate * dt)


print(f"{ckpt}  (vx {vx}, wz {wz}, switch every {switch_s:g} s)")
for vin in vins:
    for act in robot.actuators:
        act.vin_tensor[:] = vin
    obs, _ = env.reset()
    command = first.clone()
    fell_before = torch.zeros(n, dtype=torch.bool, device=dev)
    fell_after = torch.zeros(n, dtype=torch.bool, device=dev)
    reach = torch.full((n,), float("nan"), device=dev)
    pitch_before, pitch_after = [], []
    steps = int(3 * switch_s / u.step_dt)
    with torch.no_grad():
        for t in range(steps):
            time = t * u.step_dt
            target = second if switch_s <= time < 2 * switch_s else first
            command = torch.where(ramped[:, None], ramp(command, target, u.step_dt), target)
            cmd.vel_command_b[:] = command
            obs, _, dones, extras = env.step(policy(obs))
            fell = dones.bool() & ~extras["time_outs"].bool()
            if time < switch_s:
                fell_before |= fell
            else:
                fell_after |= fell & ~fell_before
            g = robot.data.projected_gravity_b
            pitch = torch.rad2deg(torch.atan2(g[:, 0], -g[:, 2]))
            if switch_s - 1.0 <= time < switch_s:
                pitch_before.append(pitch)
            if switch_s <= time < 2 * switch_s:
                pitch_after.append(pitch)
                v = robot.data.root_link_lin_vel_b[:, 0]
                hit = reversal & torch.isnan(reach) & (v * torch.sign(second[:, 0]) > 0.5 * second[:, 0].abs())
                reach[hit] = time - switch_s
    after = torch.stack(pitch_after)
    ref1 = torch.stack(pitch_before).mean(0)
    ref2 = after[-int(1.0 / u.step_dt):].mean(0)
    lo, hi = torch.minimum(ref1, ref2), torch.maximum(ref1, ref2)
    overshoot = torch.maximum(after - hi, lo - after).clamp(min=0.0).amax(0)
    print(f"vin {vin:.1f} V   steady falls {fell_before.float().mean().item() * 100:.1f}%")
    for s, (name, _, _) in enumerate(scenarios):
        row = []
        for r, label in ((True, "ramp"), (False, "step")):
            sel = (scen == s) & (ramped == r) & ~fell_before
            line = f"{label} fell {fell_after[sel].float().mean().item() * 100:5.1f}%"
            ok = sel & ~fell_after
            if reversal[sel].any():
                line += f" reach {torch.nanmean(reach[ok]).item():4.2f} s"
            line += f" overshoot {overshoot[ok].mean().item():4.1f} deg"
            row.append(line)
        print(f"  {name:14s} " + "  |  ".join(row), flush=True)
