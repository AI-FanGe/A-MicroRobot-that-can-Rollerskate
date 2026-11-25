"""Evaluate a dance checkpoint over the whole motion from its start: where the robots fall, how
far they are off the reference each second, and which joints miss it most.

usage: dance_eval.py <ckpt>
env:   TASK (Mjlab-Dance-Xiaoxiaobai)  NUM_ENVS (256)  PUSH=1 keep the training pushes
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
from mjlab.utils.lab_api.math import quat_error_magnitude

TASK = os.environ.get("TASK", "Mjlab-Dance-Xiaoxiaobai")
cfg = load_env_cfg(TASK, play=True)
cfg.scene.num_envs = int(os.environ.get("NUM_ENVS", 256))
if os.environ.get("PUSH") == "1":
    cfg.events["push_robot"] = load_env_cfg(TASK).events["push_robot"]
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device="cuda:0"))
u = env.unwrapped
runner = load_runner_cls(TASK)(env, asdict(load_rl_cfg(TASK)), device="cuda:0")
runner.load(sys.argv[1], load_cfg={"actor": True}, strict=True, map_location="cuda:0")
policy = runner.get_inference_policy("cuda:0")

cmd = u.command_manager.get_term("motion")
robot = u.scene["robot"]
names = list(u.action_manager.get_term("joint_pos").target_names)
jids = [robot.joint_names.index(n) for n in names]
T = cmd.motion.time_step_total - 1
fps = 1.0 / u.step_dt
terms = [n for n in u.termination_manager.active_terms if n != "time_out"]

alive = np.ones(u.num_envs, bool)
fail_t = np.full(u.num_envs, -1)
fail_term = np.full(u.num_envs, "", dtype=object)
POS, ROT, JERR, ACT = [], [], [], []
ee_terms = [n for n in terms if n.startswith("ee_")]
ee_cfgs = [u.termination_manager.get_term_cfg(n).params for n in ee_terms]
ee_names = [b for p in ee_cfgs for b in p["body_names"]]
ee_tol = np.array([p["threshold"] for p in ee_cfgs for _ in p["body_names"]])
ee = [cmd.cfg.body_names.index(n) for n in ee_names]
ee_fail = []
ee_dz = np.zeros((u.num_envs, len(ee)))
obs = env.get_observations()
with torch.no_grad():
    for t in range(T):
        obs, _, dones, _ = env.step(policy(obs))
        d = dones.cpu().numpy().astype(bool)
        for n in terms:
            hit = alive & d & u.termination_manager.get_term(n).cpu().numpy().astype(bool)
            fail_term[hit & (fail_term == "")] = n
        # Which end effector left its height band (the check runs on the state before the reset).
        hit = alive & np.isin(fail_term, ee_terms) & (fail_t < 0) & d
        if hit.any():
            for i in np.flatnonzero(hit):
                ee_fail.append((t, ee_names[int(np.argmax(ee_dz[i] / ee_tol))]))
        ee_dz = (cmd.body_pos_relative_w[:, ee] - cmd.robot_body_pos_w[:, ee])[..., 2].abs().cpu().numpy()
        fail_t[alive & d] = t
        alive &= ~d
        POS.append(torch.norm(cmd.body_pos_relative_w - cmd.robot_body_pos_w, dim=-1).mean(-1).cpu().numpy())
        ROT.append(quat_error_magnitude(cmd.anchor_quat_w, cmd.robot_anchor_quat_w).cpu().numpy())
        JERR.append((robot.data.joint_pos[:, jids] - cmd.joint_pos[:, jids]).abs().cpu().numpy())
        ACT.append(alive.copy())
POS, ROT, JERR, ACT = map(np.stack, (POS, ROT, JERR, ACT))

print(f"{u.num_envs} robots, {T / fps:.1f} s from the start | survived {alive.mean():.0%}")
reasons = {n: int((fail_term == n).sum()) for n in terms}
print("  first termination:", reasons)
if ee_fail:
    print("  end effector off its height (body: count, seconds):")
    for n in ee_names:
        ts = [t / fps for t, b in ee_fail if b == n]
        if ts:
            print(f"    {n:12s} {len(ts):3d}  " + " ".join(f"{s:.1f}" for s in sorted(ts)))
print("  sec  falls  alive  body_pos_err[mm]  anchor_rot_err[deg]  worst joints [deg]")
for s in range(int(np.ceil(T / fps))):
    sl = slice(int(s * fps), min(int((s + 1) * fps), T))
    a = ACT[sl]
    n_fall = int(((fail_t >= sl.start) & (fail_t < sl.stop)).sum())
    pos = POS[sl][a].mean() * 1000 if a.any() else float("nan")
    rot = np.degrees(ROT[sl][a].mean()) if a.any() else float("nan")
    je = np.degrees(JERR[sl][a].mean(0)) if a.any() else np.zeros(len(names))
    worst = ", ".join(f"{names[j]} {je[j]:.0f}" for j in np.argsort(-je)[:3])
    print(f"  {s:3d}  {n_fall:5d}  {a[-1].mean():5.0%}  {pos:16.1f}  {rot:19.1f}  {worst}")
foot_geoms = robot.indexing.geom_ids[robot.find_geoms(r".*(left|right)_foot_collision.*")[0]].long()
mu = u.sim.model.geom_friction[:, foot_geoms[0], 0].cpu().numpy()
print("  survival by foot friction:", "  ".join(
    f"{lo:.1f}-{hi:.1f}: {alive[(mu >= lo) & (mu < hi)].mean():.0%} (n={((mu >= lo) & (mu < hi)).sum()})"
    for lo, hi in ((0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 2.0)) if ((mu >= lo) & (mu < hi)).any()))
je = np.degrees(JERR[ACT].mean(0))
print("  mean joint error over the dance [deg]:")
for j in np.argsort(-je):
    print(f"    {names[j]:22s} {je[j]:5.1f}")
