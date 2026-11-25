"""How hard the walk policy squeezes its arms against the trunk (shoulder_roll).

Walks WALK_S under the play commands (no pushes), then stands STAND_S with a zero command, at
supply voltage VIN. Per phase: shoulder roll angle and its position target, measured outward
from the trunk (0 = arm on the trunk stop, default pose 0.087 rad), the share of time the
target is inside the trunk (< 0), and the shoulder roll motor current.

usage: XIAOXIAOBAI_BASE=24996 arm_roll_eval.py <ckpt> [<ckpt> ...]
env:   NUM_ENVS (default 512)  VIN (default 7.0)  WALK_S (default 15)  STAND_S (default 10)
"""
import os
import sys
from dataclasses import asdict

import torch
import mjlab_xiaoxiaobai.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK = "Mjlab-Velocity-Xiaoxiaobai"
vin = float(os.environ.get("VIN", 7.0))
walk_s, stand_s = float(os.environ.get("WALK_S", 15)), float(os.environ.get("STAND_S", 10))
cfg = load_env_cfg(TASK, play=True)
cfg.scene.num_envs = int(os.environ.get("NUM_ENVS", 512))
cfg.episode_length_s = walk_s + stand_s + 10.0
cfg.events.pop("push_robot", None)
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device="cuda:0"))
u = env.unwrapped
robot = u.scene["robot"]
act = robot.actuators[0]
act.cfg.vin_min = 4.0
kt = act._bam_model.kt.value
term = u.action_manager.get_term("joint_pos")
cmd = u.command_manager.get_term("twist")
names = list(robot.joint_names)
act_names = list(act._target_names)
roll = {side: names.index(f"{side}_shoulder_roll") for side in ("right", "left")}
sign = {"right": 1.0, "left": -1.0}
target_col = {s: term.target_ids.tolist().index(i) for s, i in roll.items()}
for ckpt in sys.argv[1:]:
    runner = load_runner_cls(TASK)(env, asdict(load_rl_cfg(TASK)), device="cuda:0")
    runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
    policy = runner.get_inference_policy("cuda:0")
    act.vin_tensor[:] = vin
    obs, _ = env.reset()
    stats = {p: {"q": [], "tgt": [], "inside": [], "cur": []} for p in ("walk", "stand")}
    n_walk, n_stand = int(walk_s / u.step_dt), int(stand_s / u.step_dt)
    with torch.no_grad():
        for t in range(n_walk + n_stand):
            if t >= n_walk:
                cmd.command[:] = 0.0
                obs["actor"][:, 60:63] = 0.0
            obs, _, _, _ = env.step(policy(obs))
            if t % n_walk < 50:
                continue
            s = stats["walk" if t < n_walk else "stand"]
            for side, j in roll.items():
                q = sign[side] * robot.data.joint_pos[:, j]
                tgt = sign[side] * term._processed_actions[:, target_col[side]]
                s["q"].append(q.mean().item())
                s["tgt"].append(tgt.mean().item())
                s["inside"].append((tgt < 0).float().mean().item())
                s["cur"].append((act._prev_motor_torque[:, act_names.index(f"{side}_shoulder_roll")].abs() / kt).mean().item())
    mean = lambda x: sum(x) / max(len(x), 1)
    print(ckpt.split("/")[-2][-28:] + "/" + ckpt.split("/")[-1], f"(vin {vin} V)")
    for p, s in stats.items():
        print(f"    {p:5s}: shoulder roll {mean(s['q']):+.3f} rad, target {mean(s['tgt']):+.3f} rad,"
              f" target inside the trunk {mean(s['inside']) * 100:4.1f}% of the time, current {mean(s['cur']):.2f} A", flush=True)
