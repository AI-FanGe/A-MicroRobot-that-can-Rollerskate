"""Stroke-clock analysis of a Mjlab-Skate-Stroke-Xiaoxiaobai checkpoint at a fixed command: per
tenth of the stroke cycle, how often each skate is down, how high it is, and how fast the
trunk goes, next to the schedule the clock asks for.

usage: skate_stroke_phase_report.py <ckpt>
env:   VX=<m/s> (0.3)  WZ=<rad/s> (0)  SECONDS (10)  NUM_ENVS (256)
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
from mjlab_xiaoxiaobai.tasks.mdp import _twist_cfgs, gait_clock_steps, set_command_velocity
from mjlab_xiaoxiaobai.tasks.xiaoxiaobai_skate_stroke_env_cfg import STROKE_PERIOD_STEPS, STROKE_PUSH_FRACTION

TASK = os.environ.get("TASK", "Mjlab-Skate-Stroke-Xiaoxiaobai")
VX = float(os.environ.get("VX", 0.3))
WZ = float(os.environ.get("WZ", 0.0))
SECONDS = float(os.environ.get("SECONDS", 10))
BINS = 10

cfg = load_env_cfg(TASK, play=True)
cfg.scene.num_envs = int(os.environ.get("NUM_ENVS", 256))
cmd = cfg.commands["twist"]
cmd.ranges.lin_vel_x, cmd.ranges.lin_vel_y, cmd.ranges.ang_vel_z = (VX, VX), (0.0, 0.0), (WZ, WZ)
cmd.rel_standing_envs = 0.0
cfg.episode_length_s = 1e3
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device="cuda:0"))
u = env.unwrapped
set_command_velocity(u, lin_vel_x=(VX, VX), lin_vel_y=(0.0, 0.0), ang_vel_z=(WZ, WZ))
for c in _twist_cfgs(u):
    c.rel_standing_envs = 0.0
u.reset()
robot = u.scene["robot"]
runner = load_runner_cls(TASK)(env, asdict(load_rl_cfg(TASK)), device="cuda:0")
runner.load(sys.argv[1], load_cfg={"actor": True}, strict=True, map_location="cuda:0")
policy = runner.get_inference_policy("cuda:0")

sensor = u.scene.sensors["feet_ground_contact"]
height = u.scene["foot_height_scan"]
joints = ["right_hip_yaw", "left_hip_yaw", "right_hip_roll", "left_hip_roll"]
jids = [robot.joint_names.index(n) for n in joints]

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat
feet = SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))
feet.resolve(u.scene)  # model order: right, left


def heading(vec):
    q = yaw_quat(robot.data.root_link_quat_w).unsqueeze(1).expand(-1, vec.shape[1], -1)
    return quat_apply_inverse(q.reshape(-1, 4), vec.reshape(-1, 3)).reshape(vec.shape)


down_sum = np.zeros((BINS, 2)); h_sum = np.zeros((BINS, 2)); n_bin = np.zeros(BINS)
rel_sum = np.zeros((BINS, 3)); vx_sum = np.zeros(BINS)  # right skate off the trunk (x, y, z), its vx
from mjlab_xiaoxiaobai.tasks.xiaoxiaobai_skate_stroke_env_cfg import STROKE_LEG_JOINTS, STROKE_REFERENCE
leg_ids = [robot.joint_names.index(f"right_{j}") for j in STROKE_LEG_JOINTS]
left_ids = [robot.joint_names.index(f"left_{j}") for j in STROKE_LEG_JOINTS]
leg_sum = np.zeros((BINS, len(leg_ids))); left_sum = np.zeros((BINS, len(leg_ids)))
rel_l_sum = np.zeros((BINS, 3)); yaw_rate_sum = np.zeros(BINS)
wyaw = []; byaw = []
speed = []; single = []; both = []; air_any = []; jpos = []
glide_num = 0.0; glide_den = 0.0; touch_ahead = []
obs = env.get_observations()
steps = int(SECONDS / u.step_dt)
for t in range(steps):
    with torch.no_grad():
        obs, _, _, _ = env.step(policy(obs))
    if t < 50:
        continue
    found = sensor.data.found
    found = (found.any(-1) if found.dim() == 3 else found > 0).float()  # right, left
    h = height.data.heights[:, [1, 0]]  # right, left
    phase = (gait_clock_steps(u) % STROKE_PERIOD_STEPS).float() / STROKE_PERIOD_STEPS
    b = torch.clamp((phase * BINS).long(), max=BINS - 1)
    rel_both = heading(robot.data.site_pos_w[:, feet.site_ids] - robot.data.root_link_pos_w.unsqueeze(1))
    rel_r = rel_both[:, 0]
    vx_r = heading(robot.data.site_lin_vel_w[:, feet.site_ids])[:, 0, 0]
    for k in range(BINS):
        m = b == k
        if m.any():
            down_sum[k] += found[m].sum(0).cpu().numpy()
            h_sum[k] += h[m].sum(0).cpu().numpy()
            n_bin[k] += m.sum().item()
            rel_sum[k] += rel_r[m].sum(0).cpu().numpy()
            vx_sum[k] += vx_r[m].sum().item()
            q = robot.data.joint_pos[m][:, leg_ids] - robot.data.default_joint_pos[m][:, leg_ids]
            leg_sum[k] += q.sum(0).cpu().numpy()
            ql = robot.data.joint_pos[m][:, left_ids] - robot.data.default_joint_pos[m][:, left_ids]
            left_sum[k] += ql.sum(0).cpu().numpy()
            rel_l_sum[k] += rel_both[m][:, 1].sum(0).cpu().numpy()
            yaw_rate_sum[k] += robot.data.root_link_ang_vel_w[m][:, 2].sum().item()
    speed.append(robot.data.root_link_lin_vel_b[:, 0].mean().item())
    wyaw.append(robot.data.root_link_ang_vel_w[:, 2].mean().item()); byaw.append(robot.data.root_link_ang_vel_b[:, 2].mean().item())
    s = found.sum(-1)
    foot_v = heading(robot.data.site_lin_vel_w[:, feet.site_ids])[..., 0]
    trunk_v = heading(robot.data.root_link_lin_vel_w.unsqueeze(1))[:, 0, 0]
    alone = (found > 0) & (s == 1).unsqueeze(-1) & (trunk_v > 0.05).unsqueeze(-1)
    glide_num += (foot_v * alone).sum().item()
    glide_den += (trunk_v.unsqueeze(-1) * alone).sum().item()
    first = sensor.compute_first_contact(dt=u.step_dt)
    rel = heading(robot.data.site_pos_w[:, feet.site_ids] - robot.data.root_link_pos_w.unsqueeze(1))[..., 0]
    touch_ahead += rel[first].tolist()
    single.append((s == 1).float().mean().item()); both.append((s == 2).float().mean().item())
    air_any.append((s == 0).float().mean().item())
    jpos.append(robot.data.joint_pos[:, jids].mean(0).cpu().numpy())

d = 2 * STROKE_PUSH_FRACTION
print(f"VX={VX} WZ={WZ}  trunk vx mean {np.mean(speed):.3f} m/s  world yaw rate {np.mean(wyaw):+.3f} rad/s"
      f"  body-z rate {np.mean(byaw):+.3f}")
print(f"stance: both down {np.mean(both):.1%}  one down {np.mean(single):.1%}  none {np.mean(air_any):.1%}")
print(f"glide: skate alone on the floor moves at {glide_num / max(glide_den, 1e-9):.0%} of the trunk's speed "
      f"(walking ~0%, gliding ~100%)")
if touch_ahead:
    print(f"touchdown: {1000 * np.mean(touch_ahead):+.1f} mm ahead of the trunk (median "
          f"{1000 * np.median(touch_ahead):+.1f})")
print("phase   want(R,L)   down R   down L   h R[mm]  h L[mm]   R off trunk x,y,z [mm]   R vx")
for k in range(BINS):
    p = (k + 0.5) / BINS
    want_r = "down" if (p < STROKE_PUSH_FRACTION or p >= 0.5) else "AIR "
    lp = (p + 0.5) % 1.0
    want_l = "down" if (lp < STROKE_PUSH_FRACTION or lp >= 0.5) else "AIR "
    nb = max(n_bin[k], 1)
    dn = down_sum[k] / nb; hh = 1000 * h_sum[k] / nb; rel = 1000 * rel_sum[k] / nb
    print(f"{p:4.2f}   {want_r},{want_l}   {dn[0]:6.2f}   {dn[1]:6.2f}   {hh[0]:7.1f}  {hh[1]:7.1f}"
          f"   {rel[0]:+6.0f} {rel[1]:+5.0f} {rel[2]:+5.0f}   {vx_sum[k] / nb:+.2f}")
knots = [k[0] for k in STROKE_REFERENCE] + [1.0]
vals = np.array([k[1] for k in STROKE_REFERENCE] + [STROKE_REFERENCE[0][1]])
print("right leg [deg off default], actual (reference):", ", ".join(STROKE_LEG_JOINTS))
for k in range(BINS):
    p = (k + 0.5) / BINS
    ref = [np.interp(p, knots, vals[:, j]) for j in range(vals.shape[1])]
    act = np.degrees(leg_sum[k] / max(n_bin[k], 1))
    print(f"{p:4.2f}  " + "  ".join(f"{a:+6.1f}({r:+5.1f})" for a, r in zip(act, ref)))
print("mirror check, each leg at its own phase (left mirrored; a symmetric stroke matches):")
print("own   R x,y [mm]   L x,-y [mm]   |  R / -L deg:", ", ".join(STROKE_LEG_JOINTS))
half = BINS // 2
for k in range(BINS):
    kl = (k + half) % BINS  # the clock bin where the left leg is at own phase k
    nr, nl = max(n_bin[k], 1), max(n_bin[kl], 1)
    rr, rl = 1000 * rel_sum[k] / nr, 1000 * rel_l_sum[kl] / nl
    jr, jl = np.degrees(leg_sum[k] / nr), -np.degrees(left_sum[kl] / nl)
    print(f"{(k + 0.5) / BINS:4.2f}  {rr[0]:+5.0f} {rr[1]:+5.0f}   {rl[0]:+5.0f} {-rl[1]:+5.0f}    |  "
          + "  ".join(f"{a:+5.1f}/{b:+5.1f}" for a, b in zip(jr, jl)))
print("trunk yaw rate per clock bin [rad/s]:", " ".join(f"{yaw_rate_sum[k] / max(n_bin[k], 1):+.2f}" for k in range(BINS)))
jp = np.degrees(np.mean(jpos, axis=0))
print("mean joint deg:", {n: round(float(v), 1) for n, v in zip(joints, jp)})
