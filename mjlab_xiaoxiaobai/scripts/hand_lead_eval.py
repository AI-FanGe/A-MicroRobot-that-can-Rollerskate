"""Free-arms walk led by the hand (xiaoxiaobai 牵手走): falls at fixed supply voltages.

usage: XIAOXIAOBAI_BASE=24996 hand_lead_eval.py <ckpt> [vin ...]   (default 7.5 6.0 5.0)
env:   NUM_ENVS (1024)  SECONDS (20)  FMAX=<N> every hand pulls at most that (default: the
       training range)

Every episode is led (XIAOXIAOBAI_HAND_LEAD=1): the arms limp, one or both hands taken and let go
at random, commands from the lead's ranges (vx -0.2..0.3 m/s, vtheta +-0.6 rad/s, no side
steps), no pushes. Reported: falls, the pull while held, the share of the time held and the
speed error.
"""
import os
import sys

os.environ["XIAOXIAOBAI_HAND_LEAD"] = "1"

from dataclasses import asdict  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import mjlab_xiaoxiaobai.tasks  # noqa: E402,F401
import mjlab_xiaoxiaobai.tasks.xiaoxiaobai_free_arms_env_cfg as free_arms  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls  # noqa: E402

TASK = "Mjlab-Velocity-Xiaoxiaobai-FreeArms"
ckpt = sys.argv[1]
vins = [float(v) for v in sys.argv[2:]] or [7.5, 6.0, 5.0]
seconds = float(os.environ.get("SECONDS", 20))
if os.environ.get("FMAX"):
    free_arms.LEAD_MAX_FORCE = (float(os.environ["FMAX"]),) * 2

cfg = free_arms.make_xiaoxiaobai_free_arms_env_cfg(play=True)
cfg.scene.num_envs = int(os.environ.get("NUM_ENVS", 1024))
cfg.episode_length_s = seconds + 5.0
cmd_cfg = cfg.commands["twist"]
cmd_cfg.ranges.lin_vel_x, cmd_cfg.ranges.lin_vel_y, cmd_cfg.ranges.ang_vel_z = (-0.2, 0.3), (0.0, 0.0), (-0.6, 0.6)
cmd_cfg.rel_standing_envs = 0.15
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device="cuda:0"))
u = env.unwrapped
runner = load_runner_cls(TASK)(env, asdict(load_rl_cfg(TASK)), device="cuda:0")
runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
policy = runner.get_inference_policy("cuda:0")
robot = u.scene["robot"]
for act in robot.actuators:
    act.cfg.vin_min = 4.0
lead = u.action_manager.get_term("joint_pos")._lead
cmd = u.command_manager.get_term("twist")
print(f"hand lead {ckpt} (pull at most {free_arms.LEAD_MAX_FORCE} N)")
for vin in vins:
    for act in robot.actuators:
        act.vin_tensor[:] = vin
    obs, _ = env.reset()
    fell = torch.zeros(u.num_envs, dtype=torch.bool, device=u.device)
    pull, held_share, err = [], [], []
    with torch.no_grad():
        for t in range(int(seconds / u.step_dt)):
            obs, _, dones, extras = env.step(policy(obs))
            fell |= dones.bool() & ~extras["time_outs"].bool()
            if t >= 50:
                f = lead.force.norm(dim=-1)
                held = lead.held & lead.holding[:, None]
                if held.any():
                    pull.append(f[held].cpu().numpy())
                held_share.append(held.any(dim=1).float().mean().item())
                v = robot.data.root_link_lin_vel_b[:, :2]
                err.append((v - cmd.command[:, :2]).norm(dim=-1).mean().item())
    pull = np.concatenate(pull)
    print(f"vin {vin:.1f} V: fell {fell.float().mean().item() * 100:5.1f}% | pull mean {pull.mean():.2f} N, p90 "
          f"{np.percentile(pull, 90):.2f} N | held {np.mean(held_share) * 100:.0f}% of the time | speed error "
          f"{np.mean(err):.3f} m/s", flush=True)
