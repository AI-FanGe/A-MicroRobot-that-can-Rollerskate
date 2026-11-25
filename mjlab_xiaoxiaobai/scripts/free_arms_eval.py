"""Falls of a walk checkpoint while the arms play the motion bank (free-arms walk).

usage: free_arms_eval.py <ckpt>
env:   TASK (Mjlab-Velocity-Xiaoxiaobai-FreeArms)  SECONDS (20)  NUM_ENVS (1024)
       STAND=1  zero velocity commands (gestures are mostly played standing)

Random walk commands from the training ranges, no pushes. Also checks that the actor sees
no arm state and that the arms follow their bank targets.
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
from mjlab_xiaoxiaobai.robot.xiaoxiaobai_constants import ARM_JOINTS
from mjlab_xiaoxiaobai.tasks.mdp import _twist_cfgs

TASK = os.environ.get("TASK", "Mjlab-Velocity-Xiaoxiaobai-FreeArms")
STAND = os.environ.get("STAND") == "1"
cfg = load_env_cfg(TASK, play=True)
cfg.scene.num_envs = int(os.environ.get("NUM_ENVS", 1024))
cfg.episode_length_s = 1e3
if STAND:
    cmd = cfg.commands["twist"]
    cmd.ranges.lin_vel_x, cmd.ranges.lin_vel_y, cmd.ranges.ang_vel_z = (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)
    cmd.rel_standing_envs = 1.0
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device="cuda:0"))
u = env.unwrapped
if STAND:
    for c in _twist_cfgs(u):
        c.rel_standing_envs = 1.0
u.reset()
runner = load_runner_cls(TASK)(env, asdict(load_rl_cfg(TASK)), device="cuda:0")
runner.load(sys.argv[1], load_cfg={"actor": True}, strict=True, map_location="cuda:0")
policy = runner.get_inference_policy("cuda:0")

robot = u.scene["robot"]
term = u.action_manager.get_term("joint_pos")
names = list(term.target_names)
arm_cols = [i for i, n in enumerate(names) if __import__("re").match(ARM_JOINTS, n)]
arm_ids = term.target_ids[arm_cols]
print("action order:", names)

T = int(float(os.environ.get("SECONDS", 20)) / u.step_dt)
alive = np.ones(u.num_envs, bool)
fell_at = np.full(u.num_envs, np.nan)
err, reach = [], []
obs = env.get_observations()
actor_terms = dict(zip(u.observation_manager.active_terms["actor"], u.observation_manager.group_obs_term_dim["actor"]))
with torch.no_grad():
    for t in range(T):
        obs, _, dones, _ = env.step(policy(obs))
        fell = alive & u.termination_manager.get_term("fell_over").cpu().numpy()
        fell_at[fell] = t * u.step_dt
        alive &= ~dones.cpu().numpy().astype(bool)
        if t > 100:
            target = term._processed_actions[:, arm_cols]
            q = robot.data.joint_pos[:, arm_ids]
            err.append((target - q).abs().mean().item())
            reach.append(target.abs().max().item())
        if t == 200 and "Free" in TASK:
            a = obs["actor"]
            i, out = 0, {}
            for name, dim in actor_terms.items():
                n = int(np.prod(dim))
                out[name] = a[:, i:i + n]
                i += n
            arm_obs = torch.cat([out["joint_pos"][:, arm_cols], out["joint_vel"][:, arm_cols], out["actions"][:, arm_cols]], 1)
            print(f"actor arm obs max |x| at t=4s: {arm_obs.abs().max().item():.4f} (noise only)")

n = u.num_envs
print(f"{TASK} {os.path.basename(sys.argv[1])} {'stand' if STAND else 'walk'}: "
      f"fell {np.isfinite(fell_at).sum()}/{n} ({100 * np.isfinite(fell_at).mean():.1f}%) in {T * u.step_dt:.0f} s")
if np.isfinite(fell_at).any():
    print("  fall times p10/p50/p90:", np.round(np.nanpercentile(fell_at, [10, 50, 90]), 1))
print(f"  arm tracking error mean {np.degrees(np.mean(err)):.1f} deg, largest target {np.degrees(max(reach)):.0f} deg")
