"""Dragging a load backward (Mjlab-Drag-Xiaoxiaobai): falls and speed per load and floor.

Every env walks backward at VX for EPISODE_S with a load from MASSES on a floor friction from
FRICTIONS (a grid over the envs). Per supply voltage and load it prints the falls and, for
those still up, the mean backward speed over the last half as a share of the command.

usage: XIAOXIAOBAI_BASE=24996 drag_eval.py <ckpt> [vin ...]   (default 8.0 7.2 6.0 5.0)
env:   NUM_ENVS (default 4000)  VX (default -0.2)  EPISODE_S (default 8)
       MASSES (default 0.1,0.25,0.5,0.75,1,1.5,2,3)  FRICTIONS (default 0.2,0.5,0.8)
"""
import os
import sys
from dataclasses import asdict

import torch
import mjlab_xiaoxiaobai.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import RecomputeLevel
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

from mjlab_xiaoxiaobai.tasks.xiaoxiaobai_drag_env_cfg import LOAD_UNIT_INERTIA

TASK = "Mjlab-Drag-Xiaoxiaobai"
ckpt = sys.argv[1]
vins = [float(v) for v in sys.argv[2:]] or [8.0, 7.2, 6.0, 5.0]
vx = float(os.environ.get("VX", -0.2))
episode_s = float(os.environ.get("EPISODE_S", 8))
masses = [float(m) for m in os.environ.get("MASSES", "0.1,0.25,0.5,0.75,1,1.5,2,3").split(",")]
frictions = [float(f) for f in os.environ.get("FRICTIONS", "0.2,0.5,0.8").split(",")]

cfg = load_env_cfg(TASK, play=True)
cfg.scene.num_envs = int(os.environ.get("NUM_ENVS", 4000))
cfg.episode_length_s = episode_s + 2.0
cfg.events.pop("push_robot", None)
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
mjm = u.sim.mj_model
load_body = mjm.body("load/load").id
sled = mjm.geom("load/load_sled").id

idx = torch.arange(n, device=dev)
mass_id = idx % len(masses)
fric_id = (idx // len(masses)) % len(frictions)
mass = torch.tensor(masses, device=dev)[mass_id]
fric = torch.tensor(frictions, device=dev)[fric_id]

print(f"{ckpt}  (vx {vx}, {episode_s:g} s)")
for vin in vins:
    for act in robot.actuators:
        act.vin_tensor[:] = vin
    obs, _ = env.reset()
    u.sim.model.body_mass[:, load_body] = mass
    u.sim.model.body_inertia[:, load_body] = mass[:, None] * torch.tensor(LOAD_UNIT_INERTIA, device=dev)
    u.sim.model.geom_friction[:, sled, 0] = fric
    u.sim.recompute_constants(RecomputeLevel.set_const)
    fell = torch.zeros(n, dtype=torch.bool, device=dev)
    speed = torch.zeros(n, device=dev)
    pitch = torch.zeros(n, device=dev)
    steps = int(episode_s / u.step_dt)
    with torch.no_grad():
        for t in range(steps):
            cmd.vel_command_b[:] = torch.tensor([vx, 0.0, 0.0], device=dev)
            obs, _, dones, extras = env.step(policy(obs))
            fell |= dones.bool() & ~extras["time_outs"].bool()
            if t >= steps // 2:
                speed += robot.data.root_link_lin_vel_b[:, 0] / (steps - steps // 2)
                g = robot.data.projected_gravity_b
                pitch += torch.rad2deg(torch.atan2(g[:, 0], -g[:, 2])) / (steps - steps // 2)
    print(f"vin {vin:.1f} V   " + "   ".join(f"mu {f:.1f}" for f in frictions) + "   (fell %, speed / command, pitch deg)")
    for m, km in enumerate(masses):
        row = []
        for f in range(len(frictions)):
            sel = (mass_id == m) & (fric_id == f)
            ok = sel & ~fell
            ratio = (speed[ok] / vx).mean().item() if ok.any() else float("nan")
            row.append(f"{fell[sel].float().mean().item() * 100:5.1f}% {ratio:5.2f} {pitch[ok].mean().item() if ok.any() else float('nan'):5.1f}")
        print(f"  {km:4.2f} kg  " + "  |  ".join(row), flush=True)
