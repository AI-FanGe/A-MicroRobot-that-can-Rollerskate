"""Walk, then stop and stand: does the walk policy hold its height when the command goes to zero?

Walks forward at VX for WALK_S, ramps the command to 0 in 0.8 s like the deploy side
(VELOCITY_RAMP_DOWN_S), then stands for STAND_S. No pushes. Per supply voltage it prints falls,
trunk height and leg joint offsets over the stand, leg tremble (joint speed and action change in
the last 5 s) and the motor current.

usage: XIAOXIAOBAI_BASE=24996 walk_stop_eval.py <ckpt> [vin ...]   (default 7.5 6.5 5.5)
env:   NUM_ENVS (default 256)  VX (default 0.3)  WALK_S (default 3)  STAND_S (default 20)
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
vins = [float(v) for v in sys.argv[2:]] or [7.5, 6.5, 5.5]
vx = float(os.environ.get("VX", 0.3))
walk_s, stand_s = float(os.environ.get("WALK_S", 3)), float(os.environ.get("STAND_S", 20))
cfg = load_env_cfg(TASK, play=True)
cfg.scene.num_envs = int(os.environ.get("NUM_ENVS", 256))
cfg.episode_length_s = walk_s + stand_s + 10.0
cfg.events.pop("push_robot", None)
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device="cuda:0"))
u = env.unwrapped
runner = load_runner_cls(TASK)(env, asdict(load_rl_cfg(TASK)), device="cuda:0")
runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
policy = runner.get_inference_policy("cuda:0")
robot = u.scene["robot"]
act = robot.actuators[0]
act.cfg.vin_min = 4.0
kt = act._bam_model.kt.value
cmd = u.command_manager.get_term("twist")
names = robot.joint_names
groups = {g: [i for i, n in enumerate(names) if g in n] for g in ("hip_pitch", "knee", "ankle")}
legs = [i for i, n in enumerate(names) if any(k in n for k in ("hip", "knee", "ankle"))]
dt = u.step_dt
print(ckpt)
for vin in vins:
    act.vin_tensor[:] = vin
    obs, _ = env.reset()
    fell = torch.zeros(u.num_envs, dtype=torch.bool, device=u.device)
    v = vx
    rows, tail_vel, tail_da, tail_i, tail_imax = [], [], [], [], []
    last = None
    n_walk, n_stand = int(walk_s / dt), int(stand_s / dt)
    with torch.no_grad():
        for t in range(n_walk + n_stand):
            if t >= n_walk:
                v = max(0.0, v - vx / 0.8 * dt)
            cmd.command[:] = 0.0
            cmd.command[:, 0] = v
            obs["actor"][:, 60:63] = cmd.command
            a = policy(obs)
            obs, _, dones, extras = env.step(a)
            fell |= dones.bool() & ~extras["time_outs"].bool()
            ok = ~fell
            q = robot.data.joint_pos - robot.data.default_joint_pos
            s = t - n_walk
            if s >= 0 and s % int(2.0 / dt) == 0:
                z = robot.data.root_link_pos_w[:, 2] - u.scene.env_origins[:, 2]
                rows.append((s * dt, z[ok].mean().item(), *(q[ok][:, groups[g]].mean().item() for g in groups)))
            if t >= n_walk + n_stand - int(5.0 / dt):
                tail_vel.append(robot.data.joint_vel[ok][:, legs].abs().mean().item())
                if last is not None:
                    tail_da.append((a - last)[ok].abs().mean().item())
                cur = act._prev_motor_torque[ok].abs() / kt
                tail_i.append(cur.sum(-1).mean().item())
                tail_imax.append(cur.max(-1).values.mean().item())
            last = a
    mean = lambda x: sum(x) / max(len(x), 1)
    print(f"vin {vin:.1f} V: fell {fell.float().mean().item() * 100:5.1f}% | last 5 s: leg |qd| {mean(tail_vel):.3f} rad/s,"
          f" |da| {mean(tail_da):.3f}, current total {mean(tail_i):.2f} A, hottest motor {mean(tail_imax):.2f} A", flush=True)
    for r in rows:
        print(f"    stand {r[0]:4.1f} s: trunk z {r[1] * 100:5.2f} cm", flush=True)
    q = (robot.data.joint_pos - robot.data.default_joint_pos)[~fell].mean(0)
    target = act._prev_motor_torque[~fell].abs().mean(0) / kt
    print("    end of stand, offset from default [rad] / current [A]: " + ", ".join(
        f"{names[i]} {q[i]:+.3f}" + (f"/{target[k]:.2f}" if k < target.numel() else "")
        for k, i in enumerate(legs)), flush=True)
