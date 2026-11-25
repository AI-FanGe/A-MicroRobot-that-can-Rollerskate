"""How a skate checkpoint pushes: skate angle and tilt, forces and slip on the ground, split
into pushing (floor pushes the skate forward) and gliding samples, per floor.

usage: skate_push_report.py <ckpt>
env:   VX=<m/s> (0.3)  SECONDS (10)  NUM_ENVS (512)
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

TASK = "Mjlab-Skate-Xiaoxiaobai"
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

n = u.num_envs
wheel_geoms = robot.indexing.geom_ids[robot.find_geoms(r".*_wheel_.*_collision")[0]].long()
crr = (u.sim.model.geom_friction[:, wheel_geoms[0], 2] / SKATE_WHEEL_RADIUS).cpu().numpy()
mu = u.sim.model.geom_friction[:, wheel_geoms[0], 0].cpu().numpy()
weight = 1.178 * 9.81
sites = [robot.site_names.index(s) for s in ("left_foot", "right_foot")]
# Outward is +y for the left skate and -y for the right one.
out_sign = torch.tensor([1.0, -1.0], device=u.device)
contact = u.scene.sensors["feet_ground_contact"]

rec = {k: [] for k in ("down", "toe_out", "tilt_in", "fwd", "out", "normal", "lat_ratio", "slip", "alive")}
alive = torch.ones(n, dtype=torch.bool, device=u.device)
obs = env.get_observations()
T = int(float(os.environ.get("SECONDS", 10)) / u.step_dt)
with torch.no_grad():
    for t in range(T):
        obs, _, dones, _ = env.step(policy(obs))
        alive &= ~dones.bool()
        if t < T // 3:
            continue
        found = contact.data.found
        found = (found.any(-1) if found.dim() == 3 else found > 0)
        q_site = robot.data.site_quat_w[:, sites]  # [n, 2, 4]
        heading = yaw_quat(robot.data.root_link_quat_w).unsqueeze(1).expand(-1, 2, -1)
        x_axis = quat_apply(q_site.reshape(-1, 4), torch.tensor([[1.0, 0, 0]], device=u.device).expand(2 * n, -1))
        y_axis = quat_apply(q_site.reshape(-1, 4), torch.tensor([[0, 1.0, 0]], device=u.device).expand(2 * n, -1))
        x_h = quat_apply_inverse(heading.reshape(-1, 4), x_axis).reshape(n, 2, 3)
        y_w = y_axis.reshape(n, 2, 3)
        yaw = torch.atan2(x_h[..., 1], x_h[..., 0])
        # Toe out: the skate's nose turned away from the other foot.
        toe_out = yaw * out_sign
        # Tilt onto the inner edge: the skate's inward axis pointing down.
        tilt_in = torch.asin(torch.clamp(y_w[..., 2] * out_sign, -1, 1))
        f_w = -contact.data.force  # floor on robot
        f_h = quat_apply_inverse(heading.reshape(-1, 4), f_w.reshape(-1, 3)).reshape(n, 2, 3)
        f_skate = quat_apply_inverse(q_site.reshape(-1, 4), f_w.reshape(-1, 3)).reshape(n, 2, 3)
        v_skate = quat_apply_inverse(q_site.reshape(-1, 4), robot.data.site_lin_vel_w[:, sites].reshape(-1, 3)).reshape(n, 2, 3)
        normal = f_h[..., 2].clamp(min=1e-3)
        rec["down"].append(found)
        rec["toe_out"].append(toe_out)
        rec["tilt_in"].append(tilt_in)
        rec["fwd"].append(f_h[..., 0] / weight)
        rec["out"].append(f_h[..., 1] * out_sign / weight)
        rec["normal"].append(normal / weight)
        rec["lat_ratio"].append(f_skate[..., 1].abs() / normal)
        rec["slip"].append(v_skate[..., 1].abs())
R = {k: torch.stack(v).cpu().numpy() for k, v in rec.items() if v}
alive = alive.cpu().numpy()

print(f"command vx {VX} | push = floor pushes the skate forward by > 3% of body weight")
for name, envs in (("tile (c_rr < 0.015)", crr < 0.015), ("carpet (c_rr > 0.04)", crr > 0.04), ("all", np.ones(n, bool))):
    sel = (envs & alive)[None, :, None] & R["down"]
    push = sel & (R["fwd"] > 0.03)
    brake = sel & (R["fwd"] < -0.03)
    deg = lambda x, m: np.rad2deg(x[m].mean()) if m.any() else float("nan")
    avg = lambda x, m: x[m].mean() if m.any() else float("nan")
    steps = len(R["fwd"]) * (envs & alive).sum()
    print(f"  {name} (n={(envs & alive).sum()}, wheel grip mu {mu[envs].mean():.2f})")
    print(f"    time a skate pushes {push.sum() / steps:.2f} / brakes {brake.sum() / steps:.2f} per robot-step |"
          f" forward impulse from pushes {R['fwd'][push].sum() / steps:+.3f}, brakes {R['fwd'][brake].sum() / steps:+.3f} body weights")
    for label, m in (("pushing", push), ("gliding", sel & ~push & ~brake), ("braking", brake)):
        print(f"    {label:8s} toe-out {deg(R['toe_out'], m):+5.1f} deg  tilt-in {deg(R['tilt_in'], m):+5.1f} deg  "
              f"fwd {avg(R['fwd'], m):+.3f} W  outward {avg(R['out'], m):+.3f} W  normal {avg(R['normal'], m):.2f} W  "
              f"|lat|/normal {avg(R['lat_ratio'], m):.2f}  side slip {avg(R['slip'], m):.3f} m/s")
