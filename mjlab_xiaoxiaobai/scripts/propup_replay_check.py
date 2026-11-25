"""Play a tracking task's reference open loop (the reference joint angles as position targets, no
policy) from its first frame: how far the bodies drift off it each second and when each
termination would fire. A reference the robot cannot hold this way is where the policy has to
do the work; one that falls at once (penetration at reset, a pose the motors cannot hold) needs
fixing before training. BOOST > 0 adds that much feedback on the joint error to the targets
(target = ref + BOOST * (ref - q)), a crude stand-in for a policy pushing past the reference
where the motors fall behind; the motor load column shows whether they still have current left.

usage: propup_replay_check.py
START=<s> starts the replay at that time of the reference and HOLD=1 keeps its pose as the
target (can the motors hold it?).

env:   TASK (Mjlab-PropUp-Xiaoxiaobai)  NUM_ENVS (16)  BOOST (0)  START (0)  HOLD (0)  SAVE=<npz> robot 0 as a raw motion
       (scripts/gymnastics_preview.py renders it)
"""
import os

import numpy as np
import torch
import mjlab_xiaoxiaobai.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.lab_api.math import quat_error_magnitude

from mjlab_xiaoxiaobai.tasks.mdp import _motor_load

TASK = os.environ.get("TASK", "Mjlab-PropUp-Xiaoxiaobai")
cfg = load_env_cfg(TASK, play=True)
cfg.scene.num_envs = int(os.environ.get("NUM_ENVS", 16))
BOOST = float(os.environ.get("BOOST", 0))
HOLD = os.environ.get("HOLD") == "1"
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
env.reset()
cmd = env.command_manager.get_term("motion")
act = env.action_manager.get_term("joint_pos")
robot = env.scene["robot"]
jids = [robot.joint_names.index(n) for n in act.target_names]
load_names = [n for a in robot.actuators if getattr(a.cfg, "max_current", None) is not None for n in a.target_names]
terms = [n for n in env.termination_manager.active_terms if n != "time_out"]
bodies = list(cmd.cfg.body_names)
T = cmd.motion.time_step_total - 1
fps = 1.0 / env.step_dt
T0 = int(float(os.environ.get("START", 0)) * fps)
if T0:
    ids = torch.arange(env.num_envs, device=env.device)
    cmd.time_steps[:] = T0
    cmd._write_reference_state_to_sim(ids, cmd.body_pos_w[:, 0], cmd.body_quat_w[:, 0], cmd.body_lin_vel_w[:, 0],
                                      cmd.body_ang_vel_w[:, 0], cmd.joint_pos, cmd.joint_vel)

first = {}
rows = []
saved = []
with torch.no_grad():
    for t in range(T0, T):
        cmd.time_steps[:] = T0 if HOLD else t
        target = cmd.motion.joint_pos[T0 if HOLD else min(t + 1, T)][jids].expand(env.num_envs, -1)
        target = target + BOOST * (target - robot.data.joint_pos[:, jids])
        env.step((target - act.offset) / act.scale)
        cmd.time_steps[:] = T0 if HOLD else min(t + 1, T)
        for n in terms:
            if n not in first and env.termination_manager.get_term(n).any():
                first[n] = t / fps
        dz = (cmd.body_pos_relative_w - cmd.robot_body_pos_w)[..., 2].abs().mean(0).cpu().numpy()
        pos = torch.norm(cmd.body_pos_relative_w - cmd.robot_body_pos_w, dim=-1).mean().item()
        rot = quat_error_magnitude(cmd.anchor_quat_w, cmd.robot_anchor_quat_w).mean().item()
        jerr = (robot.data.joint_pos[:, jids] - cmd.joint_pos[:, jids]).abs().mean(0).cpu().numpy()
        load = _motor_load(env, SceneEntityCfg("robot")).amax(0).cpu().numpy()
        rows.append((pos, rot, dz, jerr, load))
        if HOLD and t - T0 >= 3 * fps:
            break
        saved.append((robot.data.joint_pos[0].cpu().numpy(), (robot.data.root_link_pos_w[0] - env.scene.env_origins[0]).cpu().numpy(),
                      robot.data.root_link_quat_w[0].cpu().numpy()))

print(f"{TASK}: open loop (BOOST {BOOST}) over {T / fps:.1f} s, {env.num_envs} robots")
print("  first firing of each termination:", {n: round(s, 2) for n, s in first.items()} or "none")
print("  sec  body_err[mm]  trunk_rot[deg]  worst z [mm]                 worst joints [deg]                     max motor load")
step = int(fps)
for s in range(0, len(rows), step):
    pos, rot, dz, jerr = (np.mean([r[i] for r in rows[s:s + step]], axis=0) for i in range(4))
    load = np.max([r[4] for r in rows[s:s + step]], axis=0)
    zb = " ".join(f"{bodies[i][:10]} {dz[i] * 1e3:.0f}" for i in np.argsort(dz)[::-1][:2])
    jb = " ".join(f"{act.target_names[i].replace('_joint', '')} {np.degrees(jerr[i]):.0f}"
                  for i in np.argsort(jerr)[::-1][:3])
    lb = " ".join(f"{load_names[i].replace('_joint', '')} {load[i]:.2f}" for i in np.argsort(load)[::-1][:2])
    print(f"  {(s + T0) / fps:4.0f}  {pos * 1e3:10.0f}  {np.degrees(rot):12.1f}   {zb:28s} {jb:38s} {lb}")
if os.environ.get("ARMS"):
    print("  t[s]  trunk_rot  load (r shoulder_pitch, r shoulder_roll, r elbow, r hip_pitch, r knee)  err [deg]")
    pick = [act.target_names.index(f"right_{j}_joint") if f"right_{j}_joint" in act.target_names else act.target_names.index(f"right_{j}")
            for j in ("shoulder_pitch", "shoulder_roll", "elbow", "hip_pitch", "knee")]
    pick_load = [load_names.index(act.target_names[i]) for i in pick]
    for k in range(max(int(3 * fps) - T0, 0), min(int(8 * fps) - T0, len(rows)), int(fps / 4)):
        pos, rot, dz, jerr, load = rows[k]
        print(f"  {(k + T0) / fps:4.2f}  {np.degrees(rot):8.0f}   " + " ".join(f"{load[i]:.2f}" for i in pick_load)
              + "   " + " ".join(f"{np.degrees(jerr[i]):3.0f}" for i in pick))

if os.environ.get("SAVE"):
    q, p, r = map(np.stack, zip(*saved))
    np.savez(os.environ["SAVE"], fps=fps, joint_names=np.array(robot.joint_names), joint_pos=q, root_pos=p,
             root_quat=r, section_names=np.array(["回放"]), section_starts=np.array([0.0]))
    print("saved", os.environ["SAVE"])
