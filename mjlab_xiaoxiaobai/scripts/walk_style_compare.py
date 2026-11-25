"""Arm swing and gait style of several walk checkpoints under the same fixed commands.

Flat floor, no pushes, no CoM offsets. Every env walks one fixed command; per supply voltage,
command and checkpoint it prints, over the envs still up after SETTLE_S:
  arms   shoulder pitch swing (std over time, and p95-p5 range), shoulder roll held outward
         (mean, 0 = joint zero) and its swing (std), elbow flexion (mean, std), and the swing
         of the shoulder pitch target (raw action std): the commanded swing vs the achieved one
  gait   touchdowns per foot per second, mean air time, swing foot height (p95), stance width,
         knee flexion, hip pitch range (p95-p5), trunk pitch/roll mean and std, trunk angular
         velocity rms, action rate rms (legs / arms, deg per tick), achieved vx
Then, on the first checkpoint's rollouts, the rms difference of every other checkpoint's
mean action from the first one's on the very same observations, per joint group [deg]:
how differently the networks answer the same state, the dynamics left out.

usage: XIAOXIAOBAI_BASE=24996 walk_style_compare.py <ckpt> [<ckpt> ...] [-- vin ...]
env:   NUM_ENVS (default 1200)  SECONDS (default 12)  SETTLE_S (default 3)
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

TASK = "Mjlab-Velocity-Xiaoxiaobai"
args = sys.argv[1:]
if "--" in args:
    ckpts, vins = args[: args.index("--")], [float(v) for v in args[args.index("--") + 1 :]]
else:
    ckpts, vins = args, [7.2, 6.0, 5.0]
seconds, settle_s = float(os.environ.get("SECONDS", 12)), float(os.environ.get("SETTLE_S", 3))

COMMANDS = [  # label, vx, wz
    ("fwd 0.2", 0.2, 0.0),
    ("fwd 0.4", 0.4, 0.0),
    ("back 0.2", -0.2, 0.0),
    ("back 0.4", -0.4, 0.0),
    ("turn", 0.2, 0.75),
    ("stand", 0.0, 0.0),
]

cfg = load_env_cfg(TASK, play=True)
cfg.scene.num_envs = int(os.environ.get("NUM_ENVS", 1200))
cfg.episode_length_s = seconds + 5.0
cfg.events.pop("push_robot", None)
cfg.events["base_com"].params["ranges"] = {0: (0.0, 0.0), 1: (0.0, 0.0), 2: (0.0, 0.0)}
cfg.commands["twist"].resampling_time_range = (1e6, 1e6)
gen = cfg.scene.terrain.terrain_generator
gen.sub_terrains = {"flat": gen.sub_terrains["flat"]}
gen.sub_terrains["flat"].proportion = 1.0
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device="cuda:0"))
u = env.unwrapped
n, dev = u.num_envs, u.device
robot = u.scene["robot"]
for act in robot.actuators:
    act.cfg.vin_min = 4.0
cmd = u.command_manager.get_term("twist")
contact = u.scene["feet_ground_contact"]

policies = []
for c in ckpts:
    runner = load_runner_cls(TASK)(env, asdict(load_rl_cfg(TASK)), device="cuda:0")
    runner.load(c, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
    policies.append(runner.get_inference_policy("cuda:0"))

names = list(robot.joint_names)
act_names = list(u.action_manager.get_term("joint_pos")._target_names)
J = lambda *ns: [names.index(x) for x in ns]  # noqa: E731
A = lambda *ns: [act_names.index(x) for x in ns]  # noqa: E731
sp_q, sr_q, el_q = J("right_shoulder_pitch", "left_shoulder_pitch"), J("right_shoulder_roll", "left_shoulder_roll"), J("right_elbow", "left_elbow")
knee_q, hp_q = J("right_knee", "left_knee"), J("right_hip_pitch", "left_hip_pitch")
out_sign = torch.tensor([1.0, -1.0], device=dev)  # shoulder roll outward, elbow and knee flexion: right +, left -
flex_sign = torch.tensor([-1.0, 1.0], device=dev)
knee_sign = torch.tensor([1.0, -1.0], device=dev)
groups = {
    "sh_pitch": A("right_shoulder_pitch", "left_shoulder_pitch"),
    "sh_roll": A("right_shoulder_roll", "left_shoulder_roll"),
    "elbow": A("right_elbow", "left_elbow"),
    "legs": [i for i, x in enumerate(act_names) if not any(k in x for k in ("shoulder", "elbow"))],
}
arm_a = groups["sh_pitch"] + groups["sh_roll"] + groups["elbow"]
leg_a = groups["legs"]
foot_sites = [robot.site_names.index(s) for s in ("right_foot", "left_foot")]

idx = torch.arange(n, device=dev)
cgroup = idx % len(COMMANDS)
command = torch.zeros(n, 3, device=dev)
for g, (_, vx, wz) in enumerate(COMMANDS):
    command[cgroup == g, 0], command[cgroup == g, 2] = vx, wz * torch.where((idx[cgroup == g] // len(COMMANDS)) % 2 == 0, 1.0, -1.0)
deg = np.rad2deg


def rollout(policy, others=()):
    obs, _ = env.reset()
    fell = torch.zeros(n, dtype=torch.bool, device=dev)
    rec = {k: [] for k in ("sp", "sr", "el", "knee", "hp", "pitch", "roll", "angvel", "sp_act", "da_leg", "da_arm", "vx", "footz", "width")}
    diffs = [{k: [] for k in groups} for _ in others]
    touchdowns = torch.zeros(n, 2, device=dev)
    air = torch.zeros(n, device=dev)
    air_n = torch.zeros(n, device=dev)
    prev_found, prev_act = None, None
    steps = int(seconds / u.step_dt)
    with torch.no_grad():
        for t in range(steps):
            cmd.vel_command_b[:] = command
            action = policy(obs)
            if t * u.step_dt >= settle_s:
                for d, other in zip(diffs, others):
                    delta = other(obs) - action
                    for k, cols in groups.items():
                        d[k].append(delta[:, cols].square().mean(dim=1))
            obs, _, dones, extras = env.step(action)
            fell |= dones.bool() & ~extras["time_outs"].bool()
            found = (contact.data.found > 0).reshape(n, -1)[:, :2]
            if t * u.step_dt < settle_s:
                prev_found, prev_act = found, action
                continue
            down = found & ~prev_found
            touchdowns += down.float()
            last_air = contact.data.last_air_time.reshape(n, -1)[:, :2]
            air += (last_air * down).sum(dim=1)
            air_n += down.float().sum(dim=1)
            prev_found = found
            q = robot.data.joint_pos
            rec["sp"].append(q[:, sp_q])
            rec["sr"].append(q[:, sr_q] * out_sign)
            rec["el"].append(q[:, el_q] * flex_sign)
            rec["knee"].append((q[:, knee_q] * knee_sign).mean(dim=1))
            rec["hp"].append(q[:, hp_q])
            g = robot.data.projected_gravity_b
            rec["pitch"].append(torch.atan2(g[:, 0], -g[:, 2]))
            rec["roll"].append(torch.atan2(-g[:, 1], -g[:, 2]))
            rec["angvel"].append(robot.data.root_link_ang_vel_b.square().sum(dim=1))
            rec["sp_act"].append(action[:, groups["sh_pitch"]])
            rec["da_leg"].append((action - prev_act)[:, leg_a].square().mean(dim=1))
            rec["da_arm"].append((action - prev_act)[:, arm_a].square().mean(dim=1))
            prev_act = action
            rec["vx"].append(robot.data.root_link_lin_vel_b[:, 0])
            fp = robot.data.site_pos_w[:, foot_sites]
            rec["footz"].append(fp[:, :, 2] - u.scene.env_origins[:, None, 2])
            w, x, y, z = robot.data.root_link_quat_w.unbind(-1)
            yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            d = fp[:, 0, :2] - fp[:, 1, :2]
            rec["width"].append((d[:, 1] * torch.cos(yaw) - d[:, 0] * torch.sin(yaw)).abs())
    rec = {k: torch.stack(v) for k, v in rec.items()}
    T = rec["sp"].shape[0] * u.step_dt
    out = {
        "fell": fell,
        "sp_std": deg(rec["sp"].std(0).mean(1).cpu().numpy()),
        "sp_rng": deg((rec["sp"].quantile(0.95, 0) - rec["sp"].quantile(0.05, 0)).mean(1).cpu().numpy()),
        "sr_out": deg(rec["sr"].mean(0).mean(1).cpu().numpy()),
        "sr_std": deg(rec["sr"].std(0).mean(1).cpu().numpy()),
        "el_mean": deg(rec["el"].mean(0).mean(1).cpu().numpy()),
        "el_std": deg(rec["el"].std(0).mean(1).cpu().numpy()),
        "sp_act_std": deg(rec["sp_act"].std(0).mean(1).cpu().numpy()),
        "steps_hz": (touchdowns.mean(1) / T).cpu().numpy(),
        "air_s": (air / air_n.clamp(min=1)).cpu().numpy(),
        "foot_h_mm": 1000 * (rec["footz"].quantile(0.95, 0) - rec["footz"].quantile(0.05, 0)).mean(1).cpu().numpy(),
        "width_mm": 1000 * rec["width"].mean(0).cpu().numpy(),
        "knee": deg(rec["knee"].mean(0).cpu().numpy()),
        "hp_rng": deg((rec["hp"].quantile(0.95, 0) - rec["hp"].quantile(0.05, 0)).mean(1).cpu().numpy()),
        "pitch": deg(rec["pitch"].mean(0).cpu().numpy()),
        "pitch_std": deg(rec["pitch"].std(0).cpu().numpy()),
        "roll_std": deg(rec["roll"].std(0).cpu().numpy()),
        "angvel": rec["angvel"].mean(0).sqrt().cpu().numpy(),
        "da_leg": deg(rec["da_leg"].mean(0).sqrt().cpu().numpy()),
        "da_arm": deg(rec["da_arm"].mean(0).sqrt().cpu().numpy()),
        "vx": rec["vx"].mean(0).cpu().numpy(),
    }
    diffs = [{k: deg(torch.stack(v).mean(0).sqrt().cpu().numpy()) for k, v in d.items()} for d in diffs]
    return out, diffs


ARM = [("sp_std", "swing", 4.1), ("sp_rng", "range", 4.1), ("sp_act_std", "tgt", 4.1), ("sr_out", "roll", 5.1), ("sr_std", "rsw", 4.1), ("el_mean", "elb", 5.1), ("el_std", "esw", 4.1)]
GAIT = [("steps_hz", "Hz", 4.2), ("air_s", "air", 5.3), ("foot_h_mm", "lift", 4.1), ("width_mm", "width", 4.1), ("knee", "knee", 5.1), ("hp_rng", "hip", 5.1),
        ("pitch", "pitch", 5.1), ("pitch_std", "p_sd", 4.1), ("roll_std", "r_sd", 4.1), ("angvel", "w", 4.2), ("da_leg", "dLeg", 4.1), ("da_arm", "dArm", 4.1), ("vx", "vx", 5.2)]
short = [os.path.basename(os.path.dirname(c)).split("_", 2)[-1] + "/" + os.path.basename(c).removesuffix(".pt") for c in ckpts]
print("checkpoints: " + "  |  ".join(f"[{i}] {s}" for i, s in enumerate(short)))
print("arms deg: swing = shoulder pitch std, range = p95-p5, tgt = pitch target std, roll = shoulder roll out, rsw = its std, elb/esw = elbow flexion mean/std")
print("gait: Hz = touchdowns per foot per s, air s, lift = foot height range mm, width mm, knee/hip deg, pitch deg (+ fwd), p_sd/r_sd deg, w rad/s rms, dLeg/dArm deg per tick")
for vin in vins:
    for act in robot.actuators:
        act.vin_tensor[:] = vin
    results = []
    for i, p in enumerate(policies):
        results.append(rollout(p, others=policies[1:] if i == 0 else ()))
    print(f"\n===== vin {vin:.1f} V =====")
    for g, (label, _, _) in enumerate(COMMANDS):
        print(f"-- {label}")
        for i, (out, _) in enumerate(results):
            sel = ((cgroup == g) & ~out["fell"]).cpu().numpy()
            falls = out["fell"][cgroup == g].float().mean().item() * 100
            fmt = lambda keys: " ".join(f"{name} {np.mean(out[k][sel]):{w}f}" for k, name, w in keys)  # noqa: E731
            print(f"  [{i}] fell {falls:4.1f}% | {fmt(ARM)} | {fmt(GAIT)}")
    for j, d in enumerate(results[0][1], start=1):
        per = []
        for g, (label, _, _) in enumerate(COMMANDS):
            sel = (cgroup == g).cpu().numpy()
            per.append(f"{label}: " + " ".join(f"{k} {np.mean(v[sel]):4.1f}" for k, v in d.items()))
        print(f"  action rms diff [{j}] vs [0] on [0]'s states, deg: " + "  ".join(per))
