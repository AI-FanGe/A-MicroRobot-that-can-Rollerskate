"""Walk posture under fixed commands, lateral CoM errors and supply voltages.

Every env gets one fixed command (vx = +VX, -VX or 0) and a fixed lateral trunk CoM offset
(-COM_MM, 0, +COM_MM): a robot whose weight sits off to one side tips toward it unless the
policy rights it from the IMU. Per supply voltage and command it prints falls, the trunk
roll/pitch held after SETTLE_S (true orientation), and the arms: shoulder roll outward from
the default side (0 = joint zero, default pose +5 deg) and the shoulder pitch swing (std).

roll slope: change of the mean trunk roll per mm of lateral CoM offset; the closer to 0, the
better the policy rights a lopsided robot. Pitch is + leaning forward, roll + leaning left.

usage: XIAOXIAOBAI_BASE=24996 walk_posture_eval.py <ckpt> [vin ...]   (default 7.2 6.0 5.0)
env:   NUM_ENVS (default 1152)  SECONDS (default 15)  SETTLE_S (default 3)  VX (default 0.4)
       COM_MM (default 8)
"""
import os
import sys
from dataclasses import asdict

import numpy as np
import torch
import mjlab_xiaoxiaobai.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp import dr
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK = "Mjlab-Velocity-Xiaoxiaobai"
ckpt = sys.argv[1]
vins = [float(v) for v in sys.argv[2:]] or [7.2, 6.0, 5.0]
seconds, settle_s = float(os.environ.get("SECONDS", 15)), float(os.environ.get("SETTLE_S", 3))
vx_mag, com_mm = float(os.environ.get("VX", 0.4)), float(os.environ.get("COM_MM", 8))

cfg = load_env_cfg(TASK, play=True)
cfg.scene.num_envs = int(os.environ.get("NUM_ENVS", 1152))
cfg.episode_length_s = seconds + 5.0
cfg.events.pop("push_robot", None)
cfg.events["base_com"].params["ranges"] = {0: (0.0, 0.0), 1: (0.0, 0.0), 2: (0.0, 0.0)}
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
vx_levels = torch.tensor([vx_mag, -vx_mag, 0.0], device=u.device)
com_levels = [-com_mm * 1e-3, 0.0, com_mm * 1e-3]
vx_group, com_group = idx % 3, (idx // 3) % 3
trunk_cfg = SceneEntityCfg("robot", body_names=("trunk",))
trunk_cfg.resolve(u.scene)
for g, dy in enumerate(com_levels):
    if dy != 0.0:
        dr.body_com_offset(u, idx[com_group == g], ranges={1: (dy, dy)}, asset_cfg=trunk_cfg, operation="add")
command = torch.zeros(n, 3, device=u.device)
command[:, 0] = vx_levels[vx_group]

names = list(robot.joint_names)
roll_ids = [names.index("right_shoulder_roll"), names.index("left_shoulder_roll")]
pitch_ids = [names.index("right_shoulder_pitch"), names.index("left_shoulder_pitch")]
out_sign = torch.tensor([1.0, -1.0], device=u.device)

print(f"{ckpt}  (vx +-{vx_mag}, lateral CoM +-{com_mm:g} mm)")
for vin in vins:
    for act in robot.actuators:
        act.vin_tensor[:] = vin
    obs, _ = env.reset()
    fell = torch.zeros(n, dtype=torch.bool, device=u.device)
    roll, pitch, sh_roll, sh_pitch = [], [], [], []
    with torch.no_grad():
        for t in range(int(seconds / u.step_dt)):
            cmd.vel_command_b[:] = command
            obs, _, dones, extras = env.step(policy(obs))
            fell |= dones.bool() & ~extras["time_outs"].bool()
            if t * u.step_dt < settle_s:
                continue
            g = robot.data.projected_gravity_b
            roll.append(torch.rad2deg(torch.atan2(-g[:, 1], -g[:, 2])))
            pitch.append(torch.rad2deg(torch.atan2(g[:, 0], -g[:, 2])))
            q = robot.data.joint_pos
            sh_roll.append(torch.rad2deg(q[:, roll_ids] * out_sign).mean(dim=1))
            sh_pitch.append(torch.rad2deg(q[:, pitch_ids]))
    roll, pitch, sh_roll = torch.stack(roll).mean(0), torch.stack(pitch).mean(0), torch.stack(sh_roll).mean(0)
    swing = torch.stack(sh_pitch).std(dim=0).mean(dim=1)
    up = ~fell
    print(f"vin {vin:.1f} V")
    for vg, label in enumerate(("fwd ", "back", "stand")):
        sel = (vx_group == vg) & up
        rolls = [roll[sel & (com_group == c)].mean().item() for c in range(3)]
        slope = (rolls[2] - rolls[0]) / (2 * com_mm)
        falls = "/".join(f"{fell[(vx_group == vg) & (com_group == c)].float().mean().item() * 100:.0f}" for c in range(3))
        print(f"  {label:5s} fell {fell[vx_group == vg].float().mean().item() * 100:5.1f}% (CoM -/0/+ {falls}%) | pitch {pitch[sel].mean().item():+5.1f} deg"
              f" | roll at CoM -/0/+ {rolls[0]:+5.2f} {rolls[1]:+5.2f} {rolls[2]:+5.2f} deg, slope {slope:+.3f} deg/mm"
              f" | shoulder roll out {sh_roll[sel].mean().item():5.1f} deg, pitch swing {swing[sel].mean().item():4.1f} deg", flush=True)
