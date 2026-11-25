"""Replay a velocity checkpoint under specific hardware-mismatch cases.

Cases cover joint zero errors (applied via mjlab encoder_bias: observed = true + b,
physical = command - b) and extra control latency (actions held for N policy steps).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import deque
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

import mjlab.tasks  # noqa: F401
import mjlab_xiaoxiaobai.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.torch import configure_torch_backends

TASK = "Mjlab-Velocity-Xiaoxiaobai"


def set_command(env: ManagerBasedRlEnv, vx: float) -> None:
    term = env.command_manager.get_term("twist")
    vel = torch.tensor([vx, 0.0, 0.0], device=env.device, dtype=term.vel_command_b.dtype)
    term.teleop_vel[:] = vel.detach().cpu().numpy()
    term.vel_command_b[:] = vel
    term.vel_command_w[:] = vel
    term.is_standing_env[:] = False
    term.is_heading_env[:] = False
    term.is_world_env[:] = False
    term.is_forward_env[:] = False
    if hasattr(term, "is_rotation_env"):
        term.is_rotation_env[:] = False


def build_cases(offsets_deg: dict[str, float]) -> list[dict]:
    toes_up = {"right_ankle_pitch": -1.0, "left_ankle_pitch": 1.0}
    cases = [
        {"name": "nominal", "physical_error_deg": {}, "extra_delay_steps": 0},
        {"name": "saved_offsets_are_wrong(+)", "physical_error_deg": dict(offsets_deg), "extra_delay_steps": 0},
        {"name": "saved_offsets_are_wrong(-)", "physical_error_deg": {k: -v for k, v in offsets_deg.items()}, "extra_delay_steps": 0},
        {"name": "right_hip_pitch_-5", "physical_error_deg": {"right_hip_pitch": -5.0}, "extra_delay_steps": 0},
        {"name": "right_hip_pitch_+5", "physical_error_deg": {"right_hip_pitch": 5.0}, "extra_delay_steps": 0},
        {"name": "both_ankles_toes_up_4", "physical_error_deg": {k: 4.0 * s for k, s in toes_up.items()}, "extra_delay_steps": 0},
        {"name": "both_ankles_toes_down_4", "physical_error_deg": {k: -4.0 * s for k, s in toes_up.items()}, "extra_delay_steps": 0},
        {"name": "hip_roll_outward_3", "physical_error_deg": {"right_hip_roll": 3.0, "left_hip_roll": -3.0}, "extra_delay_steps": 0},
        {"name": "extra_delay_20ms", "physical_error_deg": {}, "extra_delay_steps": 1},
        {"name": "extra_delay_40ms", "physical_error_deg": {}, "extra_delay_steps": 2},
    ]
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--offsets", required=True, help="neutral_offset.json from the robot")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--commands", type=float, nargs="*", default=(0.0, 0.25))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    raw = json.loads(Path(args.offsets).read_text())["joints"]
    offsets_deg = {k: math.degrees(v) for k, v in raw.items() if abs(v) > 1e-6 and k != "head"}

    configure_torch_backends()
    env_cfg = load_env_cfg(TASK, play=True)
    agent_cfg = load_rl_cfg(TASK)
    env_cfg.scene.num_envs = int(args.num_envs)
    env_cfg.commands["twist"].teleop = False
    env_cfg.commands["twist"].debug_vis = False
    env_cfg.commands["twist"].resampling_time_range = (1.0e9, 1.0e9)
    env_cfg.events["push_robot"].interval_range_s = (1.0e9, 1.0e9)
    env_cfg.events["push_robot"].params["velocity_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0)}
    env_cfg.events.pop("encoder_bias", None)
    env_cfg.terminations.clear()

    base_env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(env, asdict(agent_cfg), device=args.device)
    runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)

    robot = base_env.scene["robot"]
    joint_index = {name: i for i, name in enumerate(robot.joint_names)}
    site_index = {name: i for i, name in enumerate(robot.site_names)}
    feet = [site_index["left_foot"], site_index["right_foot"]]
    n = base_env.num_envs
    gravity_w = torch.tensor([0.0, 0.0, -1.0], device=base_env.device).expand(n, 3)
    steps = int(round(args.seconds / base_env.step_dt))

    results = {"checkpoint": str(Path(args.checkpoint).resolve()), "offsets_deg": offsets_deg, "cases": []}
    for case in build_cases(offsets_deg):
        bias = torch.zeros_like(robot.data.encoder_bias)
        for name, err in case["physical_error_deg"].items():
            bias[:, joint_index[name]] = -math.radians(err)
        for vx in args.commands:
            robot.data.encoder_bias[:] = bias
            obs, _ = env.reset()
            robot.data.encoder_bias[:] = bias
            queue: deque[torch.Tensor] = deque()
            alive = np.ones(n, dtype=bool)
            last_pitch = np.zeros(n)
            pitch_sum = np.zeros(n)
            lat_sum = np.zeros(n)
            count = np.zeros(n)
            lat_max = np.zeros(n)
            fore_max = np.zeros(n)
            for _ in range(steps):
                set_command(base_env, vx)
                with torch.no_grad():
                    action = policy(obs)
                queue.append(action)
                applied = queue.popleft() if len(queue) > case["extra_delay_steps"] else torch.zeros_like(action)
                with torch.no_grad():
                    obs, _, _, _ = env.step(applied)
                quat = robot.data.root_link_quat_w
                g_b = quat_apply_inverse(quat, gravity_w)
                pitch = torch.rad2deg(torch.asin(g_b[:, 0].clamp(-1, 1))).cpu().numpy()
                rel = robot.data.site_pos_w[:, feet, :] - robot.data.root_link_pos_w[:, None, :]
                site_b = quat_apply_inverse(
                    quat[:, None, :].expand(-1, 2, -1).reshape(-1, 4), rel.reshape(-1, 3)
                ).reshape(n, 2, 3)
                lat = (site_b[:, 0, 1] - site_b[:, 1, 1]).cpu().numpy()
                fore = (site_b[:, 0, 0] - site_b[:, 1, 0]).abs().cpu().numpy()
                height = robot.data.root_link_pos_w[:, 2].cpu().numpy()
                fallen = alive & ((height < 0.12) | (np.abs(pitch) > 50))
                alive &= ~fallen
                last_pitch = np.where(alive | fallen, pitch, last_pitch)
                pitch_sum += np.where(alive, pitch, 0.0)
                lat_sum += np.where(alive, lat, 0.0)
                count += alive
                lat_max = np.where(alive, np.maximum(lat_max, lat), lat_max)
                fore_max = np.where(alive, np.maximum(fore_max, fore), fore_max)
            fell = ~alive
            safe = np.maximum(count, 1)
            row = {
                "case": case["name"],
                "vx": vx,
                "fall_rate": float(fell.mean()),
                "backward_fall_fraction": float((last_pitch[fell] < 0).mean()) if fell.any() else None,
                "pitch_mean_deg": float((pitch_sum / safe).mean()),
                "foot_lateral_sep_mean_mm": float((lat_sum / safe).mean() * 1000),
                "foot_lateral_sep_max_mm": float(lat_max.mean() * 1000),
                "foot_fore_aft_sep_max_mm": float(fore_max.mean() * 1000),
            }
            results["cases"].append(row)
            print(json.dumps(row), flush=True)

    env.close()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
