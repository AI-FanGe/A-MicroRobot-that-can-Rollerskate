"""Posture of a skate checkpoint at a fixed command and floor: knee bend, hip pitch, shoulder
pitch, where the elbows sit ahead of the trunk and the trunk height, averaged over the robots
still up in the second half of the run.

usage: skate_posture.py <ckpt> [<ckpt> ...]
env:   VX (0.5)  WZ (0)  CRR (0.08)  BEARING (1e-3)  DURATION (6)  NUM_ENVS (128)
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
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat
from mjlab_xiaoxiaobai.robot.xiaoxiaobai_constants import SKATE_WHEEL_RADIUS
from mjlab_xiaoxiaobai.tasks.mdp import _twist_cfgs, set_command_velocity

TASK = "Mjlab-Skate-Xiaoxiaobai"
VX, WZ = float(os.environ.get("VX", 0.5)), float(os.environ.get("WZ", 0.0))
cfg = load_env_cfg(TASK, play=True)
cfg.scene.num_envs = int(os.environ.get("NUM_ENVS", 128))
cfg.episode_length_s = 1e3
cfg.terminations.pop("out_of_terrain_bounds", None)
rolling = float(os.environ.get("CRR", 0.08)) * SKATE_WHEEL_RADIUS
bearing = float(os.environ.get("BEARING", 1e-3))
cfg.events["wheel_rolling_resistance"].params["ranges"] = (rolling, rolling)
cfg.events["wheel_bearing_friction"].params["ranges"] = (bearing, bearing)
cfg.events["wheel_bearing_damping"].params["ranges"] = (0.0, 0.0)
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device="cuda:0"))
u = env.unwrapped
set_command_velocity(u, lin_vel_x=(VX, VX), lin_vel_y=(0.0, 0.0), ang_vel_z=(WZ, WZ))
for c in _twist_cfgs(u):
    c.rel_standing_envs = 0.0
robot = u.scene["robot"]
runner = load_runner_cls(TASK)(env, asdict(load_rl_cfg(TASK)), device="cuda:0")

# Positive is a bent knee, a hip flexed forward, an arm swung forward, rolled out and a bent
# elbow on both sides.
signed = {"knee": ("right_knee", "left_knee", (1.0, -1.0)),
          "hip pitch": ("right_hip_pitch", "left_hip_pitch", (-1.0, 1.0)),
          "shoulder pitch": ("right_shoulder_pitch", "left_shoulder_pitch", (1.0, -1.0)),
          "shoulder roll": ("right_shoulder_roll", "left_shoulder_roll", (1.0, -1.0)),
          "elbow": ("right_elbow", "left_elbow", (-1.0, 1.0))}
jnt = {k: [robot.joint_names.index(r), robot.joint_names.index(l)] for k, (r, l, _) in signed.items()}
sgn = {k: torch.tensor(s, device=u.device) for k, (_, _, s) in signed.items()}
elbows = [robot.body_names.index(b) for b in ("radius", "radius_2")]
T = int(float(os.environ.get("DURATION", 6)) / u.step_dt)

print(f"command vx {VX} wz {WZ} | c_rr {os.environ.get('CRR', 0.08)} bearing {bearing:g} | {u.num_envs} envs")
for ckpt in sys.argv[1:]:
    runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
    policy = runner.get_inference_policy("cuda:0")
    u.reset()
    obs = env.get_observations()
    alive = np.ones(u.num_envs, bool)
    rec = {k: [] for k in (*signed, "elbow x", "height", "vx")}
    with torch.no_grad():
        for t in range(T):
            obs, _, dones, _ = env.step(policy(obs))
            alive &= ~dones.cpu().numpy().astype(bool)
            if t < T // 2:
                continue
            for k in signed:
                rec[k].append((robot.data.joint_pos[:, jnt[k]] * sgn[k]).mean(1).cpu().numpy())
            rel = robot.data.body_link_pos_w[:, elbows] - robot.data.root_link_pos_w.unsqueeze(1)
            q = yaw_quat(robot.data.root_link_quat_w).repeat_interleave(2, 0)
            rec["elbow x"].append(quat_apply_inverse(q, rel.reshape(-1, 3)).reshape(-1, 2, 3)[..., 0].mean(1).cpu().numpy())
            rec["height"].append(robot.data.root_link_pos_w[:, 2].cpu().numpy())
            rec["vx"].append(robot.data.root_link_lin_vel_b[:, 0].cpu().numpy())
    r = {k: np.stack(v)[:, alive] for k, v in rec.items()}
    name = "/".join(ckpt.split("/")[-2:])
    print(f"{name}\n  knee {np.degrees(r['knee'].mean()):5.1f} deg  hip pitch {np.degrees(r['hip pitch'].mean()):5.1f}"
          f"  shoulder pitch {np.degrees(r['shoulder pitch'].mean()):+6.1f} (swing std {np.degrees(r['shoulder pitch'].std(0).mean()):4.1f})"
          f"  shoulder roll {np.degrees(r['shoulder roll'].mean()):+5.1f}  elbow {np.degrees(r['elbow'].mean()):+5.1f}"
          f"  elbow x {r['elbow x'].mean() * 1000:+5.1f} mm  trunk z {r['height'].mean() * 1000:5.1f} mm"
          f"  vx {r['vx'].mean():.3f}  up {alive.mean():.2f}")
