"""Evaluate a skate checkpoint at a fixed command: survival, speed tracking, how the skates
are used, and the motor load, split by floor (rolling resistance of the wheels).

usage: skate_eval.py <ckpt>
env:   VX=<m/s> (0.3)  WZ=<rad/s> (0)  DURATION=<s> (10)  NUM_ENVS (1024)
       ZERO=1  zero actions instead of the policy (the ready pose rolling on its skates)
       CRR=<c_rr>  BEARING=<N m>  KP_SCALE=<x>  VIN=<V>  fix the floor's rolling resistance,
       the bearings' friction, scale the motors' P gain or fix the supply voltage for every robot
"""
import os
import sys
from dataclasses import asdict

import numpy as np
import torch
import mjlab_xiaoxiaobai.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab_xiaoxiaobai.tasks.mdp import _twist_cfgs, set_command_velocity
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat
from mjlab_xiaoxiaobai.robot.xiaoxiaobai_constants import SKATE_WHEEL_RADIUS
from mjlab_xiaoxiaobai.tasks.mdp import _motor_load

TASK = os.environ.get("TASK", "Mjlab-Skate-Xiaoxiaobai")
VX, WZ = float(os.environ.get("VX", 0.3)), float(os.environ.get("WZ", 0.0))
cfg = load_env_cfg(TASK, play=True)
cfg.scene.num_envs = int(os.environ.get("NUM_ENVS", 1024))
cmd = cfg.commands["twist"]
cmd.ranges.lin_vel_x, cmd.ranges.lin_vel_y, cmd.ranges.ang_vel_z = (VX, VX), (0.0, 0.0), (WZ, WZ)
cmd.rel_standing_envs = 1.0 if VX == 0 and WZ == 0 else 0.0
cfg.episode_length_s = 1e3
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device="cuda:0"))
u = env.unwrapped
# The live command term keeps the cfg it was registered with; set that one.
set_command_velocity(u, lin_vel_x=(VX, VX), lin_vel_y=(0.0, 0.0), ang_vel_z=(WZ, WZ))
for c in _twist_cfgs(u):
    c.rel_standing_envs = 1.0 if VX == 0 and WZ == 0 else 0.0
u.reset()
robot = u.scene["robot"]
if os.environ.get("ZERO") == "1":
    policy = lambda obs: torch.zeros(u.num_envs, u.action_manager.total_action_dim, device=u.device)
else:
    runner = load_runner_cls(TASK)(env, asdict(load_rl_cfg(TASK)), device="cuda:0")
    runner.load(sys.argv[1], load_cfg={"actor": True}, strict=True, map_location="cuda:0")
    policy = runner.get_inference_policy("cuda:0")

wheel_geoms = robot.indexing.geom_ids[robot.find_geoms(r".*_wheel_.*_collision")[0]].long()
wheel_dofs = robot.indexing.joint_v_adr[robot.find_joints(r".*_wheel_.*")[0]].long()
if "CRR" in os.environ:
    u.sim.model.geom_friction[:, wheel_geoms, 2] = float(os.environ["CRR"]) * SKATE_WHEEL_RADIUS
if "BEARING" in os.environ:
    u.sim.model.dof_frictionloss[:, wheel_dofs] = float(os.environ["BEARING"])
if "KP_SCALE" in os.environ:
    for actuator in robot.actuators:
        actuator.set_gains(slice(None), kp_scale=float(os.environ["KP_SCALE"]))
if "VIN" in os.environ:
    for actuator in robot.actuators:
        if getattr(actuator, "vin_tensor", None) is not None:
            actuator.vin_tensor[:] = float(os.environ["VIN"])
            # The training floor (vin_min) would lift a supply below it back up.
            actuator.cfg.vin_min = 4.0
# Rolling resistance coefficient c_rr of each robot's floor (shared by its wheels).
resistance = (u.sim.model.geom_friction[:, wheel_geoms[0], 2] / SKATE_WHEEL_RADIUS).cpu().numpy()
# The bearings' Coulomb friction as extra c_rr, at the ~1.4 N a wheel carries in double support.
wheel_dof = wheel_dofs[0]
bearing = (u.sim.model.dof_frictionloss[:, wheel_dof] / (1.45 * SKATE_WHEEL_RADIUS)).cpu().numpy()
resistance = resistance + bearing
contact = u.scene.sensors["feet_ground_contact"]
# In the contact sensor's order (bodies foot, foot_2: right, left).
site_ids = [robot.site_names.index(s) for s in ("right_foot", "left_foot")]

T = int(float(os.environ.get("DURATION", 10)) / u.step_dt)
alive = np.ones(u.num_envs, bool)
# A robot that skates off its terrain patch ends early too; that is not a fall.
fell = np.zeros(u.num_envs, bool)
out_of_bounds = np.zeros(u.num_envs, bool)
VXS, WZS, ONE, NONE, SIDE, LOAD, AIR, DOWN, TDX = ([] for _ in range(9))
action_term = u.action_manager.get_term("joint_pos")
yaw_names = [n for n in action_term.target_names if "hip_yaw" in n]
yaw_act = [list(action_term.target_names).index(n) for n in yaw_names]
yaw_jnt = [robot.joint_names.index(n) for n in yaw_names]
YAW_CMD, YAW_POS = [], []
# Outward (spreading) hip roll from zero: the ready pose rolls the left hip negative.
roll_jnt = [robot.joint_names.index(n) for n in ("left_hip_roll", "right_hip_roll")]
roll_out = torch.tensor([-1.0, 1.0], device=u.device)
ROLL = []
RATE = []  # stroke clock rate (StrokeRateAction); 1 without one
obs = env.get_observations()
with torch.no_grad():
    for t in range(T):
        action = policy(obs)
        obs, _, dones, _ = env.step(action)
        YAW_CMD.append(action[:, yaw_act].cpu().numpy())
        YAW_POS.append((robot.data.joint_pos[:, yaw_jnt] - robot.data.default_joint_pos[:, yaw_jnt]).cpu().numpy())
        ROLL.append((robot.data.joint_pos[:, roll_jnt] * roll_out).cpu().numpy())
        fell |= alive & u.termination_manager.get_term("fell_over").cpu().numpy()
        out_of_bounds |= alive & u.termination_manager.get_term("out_of_terrain_bounds").cpu().numpy()
        alive &= ~dones.cpu().numpy().astype(bool)
        vel_b = robot.data.root_link_lin_vel_b
        VXS.append(vel_b[:, 0].cpu().numpy())
        WZS.append(robot.data.root_link_ang_vel_b[:, 2].cpu().numpy())
        found = contact.data.found
        found = (found.any(-1) if found.dim() == 3 else found > 0)
        ONE.append((found.sum(-1) == 1).cpu().numpy())
        DOWN.append(found.cpu().numpy())
        rel = robot.data.site_pos_w[:, site_ids] - robot.data.root_link_pos_w.unsqueeze(1)
        TDX.append(quat_apply_inverse(yaw_quat(robot.data.root_link_quat_w).repeat_interleave(2, 0), rel.reshape(-1, 3))
                   .reshape(u.num_envs, 2, 3)[..., 0].cpu().numpy())
        NONE.append((found.sum(-1) == 0).cpu().numpy())
        q = robot.data.site_quat_w[:, site_ids].reshape(-1, 4)
        v = robot.data.site_lin_vel_w[:, site_ids].reshape(-1, 3)
        side = quat_apply_inverse(q, v).reshape(u.num_envs, 2, 3)[..., 1].abs()
        SIDE.append(((side * found).sum(-1) / found.sum(-1).clamp(min=1)).cpu().numpy())
        LOAD.append(_motor_load(u, SceneEntityCfg("robot")).max(dim=1).values.cpu().numpy())
        AIR.append(contact.data.last_air_time.max(dim=1).values.cpu().numpy())
        RATE.append(u.__dict__.get("_stroke_rate", torch.ones(u.num_envs, device=u.device)).cpu().numpy())
VXS, WZS, ONE, NONE, SIDE, LOAD, AIR, DOWN, TDX, RATE = map(np.stack, (VXS, WZS, ONE, NONE, SIDE, LOAD, AIR, DOWN, TDX, RATE))
late = slice(T // 3, None)
# Touchdowns per second per foot: one push-glide cycle per foot each.
first = (DOWN[1:] & ~DOWN[:-1])[late]
touchdowns = first.sum(0) / ((T - T // 3) * u.step_dt)
# Where the skates set down, ahead (+) of the trunk [m].
td_x = TDX[1:][late]

YAW_CMD, YAW_POS = np.stack(YAW_CMD)[late], np.stack(YAW_POS)[late]
ROLL = np.degrees(np.stack(ROLL)[late][:, alive])
print(f"command vx {VX} wz {WZ} | {T * u.step_dt:.0f} s | {u.num_envs} envs"
      + "".join(f" | {k} {os.environ[k]}" for k in ("CRR", "BEARING", "KP_SCALE", "VIN") if k in os.environ))
print(f"  hip yaw swing (std, deg): target {np.degrees(YAW_CMD[:, alive].std(0)).mean():.1f}"
          f"  achieved {np.degrees(YAW_POS[:, alive].std(0)).mean():.1f}")
if ROLL.size:
    print(f"  hip roll outward (deg): p50 {np.percentile(ROLL, 50):.1f}  p95 {np.percentile(ROLL, 95):.1f}"
          f"  p99 {np.percentile(ROLL, 99):.1f}  max {ROLL.max():.1f}"
          f"  time >25 {(ROLL > 25).mean():.3f} >30 {(ROLL > 30).mean():.3f} >32 {(ROLL > 32).mean():.3f}")
floors = (("tile (c_rr < 0.015)", resistance < 0.015),
          ("carpet (c_rr 0.04-0.12)", (resistance > 0.04) & (resistance < 0.12)),
          ("thick carpet (c_rr > 0.12)", resistance > 0.12), ("all", np.ones_like(alive)))
for name, m in floors:
    a = m & alive
    if not m.sum():
        continue
    print(f"  {name:30s} n={m.sum():4d} survived {alive[m].mean():.2f} fell {fell[m].mean():.2f}"
          f" off terrain {out_of_bounds[m].mean():.2f}"
          + (f" | vx {VXS[late][:, a].mean():.3f} wz {WZS[late][:, a].mean():+.3f}"
             f" | one skate down {ONE[late][:, a].mean():.2f} touchdowns/s L {touchdowns[a, 1].mean():.2f} R {touchdowns[a, 0].mean():.2f}"
             f" none {NONE[late][:, a].mean():.2f} touchdown x {td_x[:, a][first[:, a]].mean() * 1000:+.0f} mm"
             f" | side slip {SIDE[late][:, a].mean():.3f} m/s | max motor load p50 {np.median(LOAD[late][:, a]):.2f}"
             f" sat>0.9 {(LOAD[late][:, a] > 0.9).mean():.2f} | stroke rate {RATE[late][:, a].mean():.2f}" if a.sum() else ""))
