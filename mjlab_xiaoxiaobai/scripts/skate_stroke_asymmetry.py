"""Per-robot left/right asymmetry of a Mjlab-Skate-Stroke-Xiaoxiaobai checkpoint going straight.

For each robot, over whole stroke cycles: the mean of each right leg joint against the mirrored
left one (a symmetric stroke matches), the trunk speed gained in the right and in the left push,
how long each skate glides alone, and the drift in heading. Asymmetry is set against the robot's
encoder bias (observed = true + b), which the policy cannot see.

usage: skate_stroke_asymmetry.py <ckpt>
env:   VX (0.3)  SECONDS (12)  NUM_ENVS (512)  NO_BIAS=1 (drop encoder_bias)  NO_COM=1 (centre base_com)
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
from mjlab_xiaoxiaobai.tasks.xiaoxiaobai_skate_stroke_env_cfg import STROKE_LEG_JOINTS, STROKE_PERIOD_STEPS

TASK = os.environ.get("TASK", "Mjlab-Skate-Stroke-Xiaoxiaobai")
VX = float(os.environ.get("VX", 0.3))
SECONDS = float(os.environ.get("SECONDS", 12))

cfg = load_env_cfg(TASK, play=True)
cfg.scene.num_envs = int(os.environ.get("NUM_ENVS", 512))
cmd = cfg.commands["twist"]
cmd.ranges.lin_vel_x, cmd.ranges.lin_vel_y, cmd.ranges.ang_vel_z = (VX, VX), (0.0, 0.0), (0.0, 0.0)
cmd.rel_standing_envs = 0.0
cfg.episode_length_s = 1e3
if os.environ.get("NO_BIAS") == "1":
    cfg.events.pop("encoder_bias", None)
if os.environ.get("NO_COM") == "1":
    cfg.events.pop("base_com", None)
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device="cuda:0"))
u = env.unwrapped
set_command_velocity(u, lin_vel_x=(VX, VX), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0))
for c in _twist_cfgs(u):
    c.rel_standing_envs = 0.0
u.reset()
robot = u.scene["robot"]
runner = load_runner_cls(TASK)(env, asdict(load_rl_cfg(TASK)), device="cuda:0")
runner.load(sys.argv[1], load_cfg={"actor": True}, strict=True, map_location="cuda:0")
policy = runner.get_inference_policy("cuda:0")
sensor = u.scene.sensors["feet_ground_contact"]

r_ids = [robot.joint_names.index(f"right_{j}") for j in STROKE_LEG_JOINTS]
l_ids = [robot.joint_names.index(f"left_{j}") for j in STROKE_LEG_JOINTS]
n = u.num_envs
dev = u.device
sum_r = torch.zeros(n, len(r_ids), device=dev); sum_l = torch.zeros_like(sum_r); count = torch.zeros(n, device=dev)
gain = torch.zeros(n, 2, device=dev)  # trunk speed gained during the right, the left push
alone = torch.zeros(n, 2, device=dev)  # steps with only the right / only the left skate down
yaw = torch.zeros(n, device=dev)
alive = torch.ones(n, dtype=torch.bool, device=dev)
prev_vx = None

obs = env.get_observations()
steps = int(SECONDS / u.step_dt)
warm = 100
# Whole cycles after the warm-up at the base rate; the rate output varies it a little.
for t in range(steps):
    with torch.no_grad():
        obs, _, dones, _ = env.step(policy(obs))
    alive &= ~dones.bool()
    vx = robot.data.root_link_lin_vel_b[:, 0]
    if t < warm:
        prev_vx = vx.clone()
        continue
    q = robot.data.joint_pos - robot.data.default_joint_pos
    sum_r += q[:, r_ids]; sum_l += -q[:, l_ids]; count += 1
    phase = (gait_clock_steps(u) % STROKE_PERIOD_STEPS) / STROKE_PERIOD_STEPS
    dv = vx - prev_vx
    gain[:, 0] += torch.where(phase < 0.25, dv, 0.0)
    gain[:, 1] += torch.where((phase >= 0.5) & (phase < 0.75), dv, 0.0)
    found = sensor.data.found
    found = (found.any(-1) if found.dim() == 3 else found > 0)  # right, left
    one = found.sum(-1) == 1
    alone += (found & one.unsqueeze(-1)).float()
    yaw += robot.data.root_link_ang_vel_w[:, 2]
    prev_vx = vx.clone()

m = alive.cpu().numpy()
asym = np.degrees(((sum_r - sum_l) / count.unsqueeze(-1)).cpu().numpy())[m]
secs = (steps - warm) * u.step_dt
gain_np = (gain / secs).cpu().numpy()[m]  # m/s gained per second of riding, by push side
alone_np = (alone * u.step_dt / secs).cpu().numpy()[m]
yaw_np = (yaw / (steps - warm)).cpu().numpy()[m]
bias = getattr(robot.data, "encoder_bias", None)
print(f"VX={VX}  robots alive {m.sum()}/{n}"
            + "".join(f"  {k}={os.environ[k]}" for k in ("NO_BIAS", "NO_COM") if k in os.environ))
print("joint           mean(R - mirrored L) [deg]   |asym| p50   p90   max", end="")
corr = bias is not None and os.environ.get("NO_BIAS") != "1"
print("   corr with bias(R) + bias(-L)" if corr else "")
for j, name in enumerate(STROKE_LEG_JOINTS):
    a = asym[:, j]
    line = f"{name:14s}  {a.mean():+6.2f}                       {np.percentile(np.abs(a), 50):5.2f} {np.percentile(np.abs(a), 90):5.2f} {np.abs(a).max():5.2f}"
    if corr:
        b = np.degrees((bias[:, r_ids[j]] + bias[:, l_ids[j]]).cpu().numpy())[m]
        line += f"   r={np.corrcoef(a, b)[0, 1]:+.2f}  bias|p50| {np.percentile(np.abs(b), 50):.2f}"
    print(line)
g = gain_np
print(f"push gain [m/s per s]: right {g[:, 0].mean():.3f}  left {g[:, 1].mean():.3f}"
      f"  per-robot |R-L| p50 {np.percentile(np.abs(g[:, 0] - g[:, 1]), 50):.3f} p90 {np.percentile(np.abs(g[:, 0] - g[:, 1]), 90):.3f}")
a = alone_np
print(f"gliding alone [fraction]: right {a[:, 0].mean():.3f}  left {a[:, 1].mean():.3f}"
      f"  per-robot |R-L| p50 {np.percentile(np.abs(a[:, 0] - a[:, 1]), 50):.3f} p90 {np.percentile(np.abs(a[:, 0] - a[:, 1]), 90):.3f}")
print(f"heading drift [rad/s]: mean {yaw_np.mean():+.3f}  |drift| p50 {np.percentile(np.abs(yaw_np), 50):.3f}"
      f" p90 {np.percentile(np.abs(yaw_np), 90):.3f}")
