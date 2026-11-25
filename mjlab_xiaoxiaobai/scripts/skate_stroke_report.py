"""Per-stance analysis of a skate checkpoint: every time a skate is on the ground, from
touchdown to lift-off: how long, where it sets down and lifts off relative to the trunk, how
much forward impulse the floor gives it, and how the robot's speed holds up, per floor.

usage: skate_stroke_report.py <ckpt>
env:   VX=<m/s> (0.3)  SECONDS (12)  NUM_ENVS (512)
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
from mjlab_xiaoxiaobai.tasks.mdp import _twist_cfgs, set_command_velocity
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, yaw_quat
from mjlab_xiaoxiaobai.robot.xiaoxiaobai_constants import SKATE_WHEEL_RADIUS

TASK = os.environ.get("TASK", "Mjlab-Skate-Xiaoxiaobai")
VX = float(os.environ.get("VX", 0.3))
cfg = load_env_cfg(TASK, play=True)
cfg.scene.num_envs = int(os.environ.get("NUM_ENVS", 512))
cmd = cfg.commands["twist"]
cmd.ranges.lin_vel_x, cmd.ranges.lin_vel_y, cmd.ranges.ang_vel_z = (VX, VX), (0.0, 0.0), (0.0, 0.0)
cmd.rel_standing_envs = 0.0
cfg.episode_length_s = 1e3
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device="cuda:0"))
u = env.unwrapped
# The live command term keeps the cfg it was registered with; set that one.
set_command_velocity(u, lin_vel_x=(VX, VX), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0))
for c in _twist_cfgs(u):
    c.rel_standing_envs = 1.0 if VX == 0 and 0.0 == 0 else 0.0
u.reset()
robot = u.scene["robot"]
runner = load_runner_cls(TASK)(env, asdict(load_rl_cfg(TASK)), device="cuda:0")
runner.load(sys.argv[1], load_cfg={"actor": True}, strict=True, map_location="cuda:0")
policy = runner.get_inference_policy("cuda:0")

n, dt = u.num_envs, u.step_dt
wheel_geoms = robot.indexing.geom_ids[robot.find_geoms(r".*_wheel_.*_collision")[0]].long()
crr = (u.sim.model.geom_friction[:, wheel_geoms[0], 2] / SKATE_WHEEL_RADIUS).cpu().numpy()
weight = 1.178 * 9.81
sites = [robot.site_names.index(s) for s in ("left_foot", "right_foot")]
out_sign = torch.tensor([1.0, -1.0], device=u.device)
contact = u.scene.sensors["feet_ground_contact"]

T = int(float(os.environ.get("SECONDS", 12)) / dt)
DOWN, X, Y, FWD, TOE, TILT, VXS, ALIVE = ([] for _ in range(8))
alive = torch.ones(n, dtype=torch.bool, device=u.device)
obs = env.get_observations()
with torch.no_grad():
    for t in range(T):
        obs, _, dones, _ = env.step(policy(obs))
        alive &= ~dones.bool()
        found = contact.data.found
        found = (found.any(-1) if found.dim() == 3 else found > 0)
        heading = yaw_quat(robot.data.root_link_quat_w).unsqueeze(1).expand(-1, 2, -1).reshape(-1, 4)
        rel = (robot.data.site_pos_w[:, sites] - robot.data.root_link_pos_w.unsqueeze(1)).reshape(-1, 3)
        rel_h = quat_apply_inverse(heading, rel).reshape(n, 2, 3)
        f_h = quat_apply_inverse(heading, -contact.data.force.reshape(-1, 3)).reshape(n, 2, 3)
        q_site = robot.data.site_quat_w[:, sites].reshape(-1, 4)
        x_h = quat_apply_inverse(heading, quat_apply(q_site, torch.tensor([[1.0, 0, 0]], device=u.device).expand(2 * n, -1)))
        y_w = quat_apply(q_site, torch.tensor([[0, 1.0, 0]], device=u.device).expand(2 * n, -1))
        DOWN.append(found)
        X.append(rel_h[..., 0])
        Y.append(rel_h[..., 1] * out_sign)
        FWD.append(f_h[..., 0] / weight)
        TOE.append(torch.atan2(x_h[:, 1], x_h[:, 0]).reshape(n, 2) * out_sign)
        TILT.append(torch.asin(torch.clamp(y_w[:, 2].reshape(n, 2) * out_sign, -1, 1)))
        VXS.append(robot.data.root_link_lin_vel_b[:, 0])
        ALIVE.append(alive.clone())
DOWN, X, Y, FWD, TOE, TILT, VXS, ALIVE = (torch.stack(a).cpu().numpy() for a in (DOWN, X, Y, FWD, TOE, TILT, VXS, ALIVE))

# Stances that start and end inside the second half of the rollout.
start = T // 3
stances = []  # (env, foot, duration, x_td, x_lo, y_td, impulse, toe, tilt, double share)
for e in range(n):
    for f in range(2):
        d = DOWN[:, e, f]
        t = start
        while t < T:
            if d[t] and not d[t - 1]:
                t1 = t
                while t1 < T and d[t1]:
                    t1 += 1
                if t1 >= T or not ALIVE[t1 - 1, e]:
                    break
                sl = slice(t, t1)
                stances.append((e, f, (t1 - t) * dt, X[t, e, f], X[t1 - 1, e, f], Y[t, e, f],
                                FWD[sl, e, f].sum() * dt, TOE[sl, e, f].mean(), TILT[sl, e, f].mean(),
                                DOWN[sl, e, 1 - f].mean()))
                t = t1
            t += 1
S = np.array(stances)

print(f"command vx {VX}, no pushes | {len(S)} stances")
for name, envs in (("tile (c_rr < 0.015)", crr < 0.015), ("carpet (c_rr > 0.04)", crr > 0.04), ("all", np.ones(n, bool))):
    live = envs & ALIVE[-1]
    v = VXS[start:, live]
    print(f"\n  {name}: n={live.sum()} | vx mean {v.mean():.3f}, time with vx < 0.05: {(v < 0.05).mean():.2f},"
          f" vx < 0: {(v < 0).mean():.2f} | per-robot mean vx p10/p50/p90 {np.percentile(v.mean(0), [10, 50, 90]).round(3)}")
    s = S[envs[S[:, 0].astype(int)]]
    if not len(s):
        continue
    dur = s[:, 2]
    print(f"    stance duration p10/p50/p90 {np.percentile(dur, [10, 50, 90]).round(2)} s")
    for lo, hi in ((0, 0.15), (0.15, 0.4), (0.4, 10)):
        m = (dur >= lo) & (dur < hi)
        if not m.any():
            continue
        print(f"    stances {lo:.2f}-{hi:.2f} s ({m.mean():.0%}): set down x {s[m, 3].mean() * 1000:+.0f} mm,"
              f" lift off x {s[m, 4].mean() * 1000:+.0f} mm, out {s[m, 5].mean() * 1000:+.0f} mm |"
              f" forward impulse {s[m, 6].mean() * 1000:+.1f} mN*s/N (sum share {s[m, 6].sum() / max(s[:, 6].sum(), 1e-9):+.0%})"
              f" | toe-out {np.rad2deg(s[m, 7].mean()):+.0f} deg tilt-in {np.rad2deg(s[m, 8].mean()):+.0f} deg"
              f" | other skate down {s[m, 9].mean():.0%}")
