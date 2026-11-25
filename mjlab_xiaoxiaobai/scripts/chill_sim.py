"""Play a chill_motion.py loop in physics: its joint angles as position targets at KP_GETUP (no
policy), from its first pose, on the prop-up task's robot (BAM motors, every body touching the
floor). Prints how the trunk drifts and the motor load, and saves robot 0 as a raw motion with
the loop's head pitch for gymnastics_preview.py.

usage: chill_sim.py <raw.npz> <out_raw.npz> [--num-envs 8] [--kp N]
The raw motion is first turned into the task's motion file (dance_motion_npz.py --no-ready). Its
prop_boxes (a chair) are added to the scene as colliding geoms.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("raw", type=Path)
ap.add_argument("out", type=Path)
ap.add_argument("--num-envs", type=int, default=8)
ap.add_argument("--kp", type=int, default=None, help="motor KP (default the task's KP_GETUP)")
args = ap.parse_args()

motion = args.raw.with_name(args.raw.stem.replace("_raw", "") + "_50hz.npz")
subprocess.run([sys.executable, str(Path(__file__).with_name("dance_motion_npz.py")), str(args.raw), str(motion),
                "--no-ready"], check=True)
os.environ["XIAOXIAOBAI_PROPUP_MOTION"] = str(motion.resolve())

import torch  # noqa: E402
import mjlab_xiaoxiaobai.tasks  # noqa: E402,F401
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.managers.scene_entity_config import SceneEntityCfg  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402
from mjlab.utils.lab_api.math import quat_error_magnitude  # noqa: E402

from mjlab_xiaoxiaobai.tasks.mdp import _motor_load  # noqa: E402

cfg = load_env_cfg("Mjlab-PropUp-Xiaoxiaobai", play=True)
cfg.scene.num_envs = args.num_envs
if args.kp:
    from dataclasses import replace
    art = cfg.scene.entities["robot"].articulation
    art.actuators = tuple(replace(a, kp_fw=args.kp) for a in art.actuators)
props = np.load(args.raw).get("prop_boxes")
if props is not None:
    import mujoco

    def add_props(spec):
        for k, box in enumerate(props):
            spec.worldbody.add_geom(name=f"prop_{k}", type=mujoco.mjtGeom.mjGEOM_BOX, pos=box[:3], size=box[3:6],
                                    rgba=box[6:10], quat=box[10:14] if len(box) >= 14 else (1, 0, 0, 0),
                                    contype=1, conaffinity=1, friction=(0.8, 0.005, 0.0001))

    cfg.scene.spec_fn = add_props
    # The props sit at the world origin: every robot at the origin too, each in its own world.
    cfg.scene.env_spacing = cfg.scene.terrain.env_spacing = 0.0
cfg.terminations = {k: v for k, v in cfg.terminations.items() if k == "time_out"}
cfg.events.pop("push_robot", None)
cmd_cfg = cfg.commands["motion"]
cmd_cfg.debug_vis = False
cmd_cfg.pose_range = {k: (0.0, 0.0) for k in cmd_cfg.pose_range}
cmd_cfg.velocity_range = {k: (0.0, 0.0) for k in cmd_cfg.velocity_range}
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
env.reset()
cmd = env.command_manager.get_term("motion")
cmd._resample_command = lambda ids: cmd.time_steps.__setitem__(ids, cmd.motion.time_step_total - 1)
act = env.action_manager.get_term("joint_pos")
robot = env.scene["robot"]
jids = [robot.joint_names.index(n) for n in act.target_names]
load_names = [n for a in robot.actuators if getattr(a.cfg, "max_current", None) is not None for n in a.target_names]
T = cmd.motion.time_step_total - 1
fps = 1.0 / env.step_dt

quat0 = robot.data.root_link_quat_w.clone()
pos0 = robot.data.root_link_pos_w.clone()
saved, rows = [], []
with torch.no_grad():
    for t in range(T):
        cmd.time_steps[:] = t
        target = cmd.motion.joint_pos[min(t + 1, T)][jids].expand(env.num_envs, -1)
        env.step((target - act.offset) / act.scale)
        rot = torch.rad2deg(quat_error_magnitude(robot.data.root_link_quat_w, quat0))
        slide = torch.norm((robot.data.root_link_pos_w - pos0)[:, :2], dim=-1) * 1e3
        jerr = torch.rad2deg((robot.data.joint_pos[:, jids] - target).abs()).mean(0)
        load = _motor_load(env, SceneEntityCfg("robot")).amax(0)
        rows.append((rot.cpu().numpy(), slide.cpu().numpy(), jerr.cpu().numpy(), load.cpu().numpy()))
        saved.append((robot.data.joint_pos[0].cpu().numpy(), (robot.data.root_link_pos_w[0] - env.scene.env_origins[0]).cpu().numpy(),
                      robot.data.root_link_quat_w[0].cpu().numpy()))

print(f"{args.raw.name}: open loop at KP {args.kp or 'GETUP'} over {T / fps:.0f} s, {env.num_envs} robots (friction randomised)")
print("  sec  trunk turn [deg] mean/max   slide [mm] mean/max   worst joints err [deg]          max motor load")
for s in range(0, len(rows), int(2 * fps)):
    chunk = rows[s:s + int(2 * fps)]
    rot = np.max([r[0] for r in chunk], axis=0)
    slide = np.max([r[1] for r in chunk], axis=0)
    jerr = np.mean([r[2] for r in chunk], axis=0)
    load = np.max([r[3] for r in chunk], axis=0)
    jb = " ".join(f"{act.target_names[i]} {jerr[i]:.0f}" for i in np.argsort(jerr)[::-1][:2])
    lb = " ".join(f"{load_names[i]} {load[i]:.2f}" for i in np.argsort(load)[::-1][:2])
    print(f"  {s / fps:4.0f}  {rot.mean():8.1f} {rot.max():6.1f}      {slide.mean():8.0f} {slide.max():6.0f}    {jb:32s} {lb}")

final = rows[-1][0]
print("  trunk turn at the end per robot [deg]:", np.round(final, 0).tolist())
src = np.load(args.raw)
q, p, r = map(np.stack, zip(*saved))
n = min(len(q), len(src["head_pitch"]))
np.savez(args.out, fps=fps, joint_names=np.array(robot.joint_names), joint_pos=q[:n], root_pos=p[:n], root_quat=r[:n],
         head_pitch=src["head_pitch"][1:n + 1] if len(src["head_pitch"]) > n else src["head_pitch"][:n],
         section_names=src["section_names"], section_starts=src["section_starts"],
         **({"prop_boxes": props} if props is not None else {}))
print("saved", args.out)
