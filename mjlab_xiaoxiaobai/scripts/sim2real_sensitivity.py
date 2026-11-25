"""Probe how a velocity checkpoint reacts to likely sim-to-real mismatches.

Each env gets a trunk COM x shift and a foot friction (sampled by startup events)
plus an IMU pitch bias injected into the actor's projected-gravity observation.
"""

from __future__ import annotations

import argparse
import json
import math
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
IMU_BIASES_DEG = (-6.0, -3.0, 0.0, 3.0, 6.0)


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


def actor_obs(obs):
    if isinstance(obs, torch.Tensor):
        return obs
    return obs["actor"]


def apply_imu_bias(obs, bias_rad: torch.Tensor):
    actor = actor_obs(obs)
    gx = actor[:, 3].clone()
    gz = actor[:, 5].clone()
    c = torch.cos(bias_rad)
    s = torch.sin(bias_rad)
    actor[:, 3] = gx * c - gz * s
    actor[:, 5] = gz * c + gx * s
    return obs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-envs", type=int, default=1000)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--com-x-range", type=float, nargs=2, default=(-0.030, 0.020))
    parser.add_argument("--friction-range", type=float, nargs=2, default=(0.25, 1.2))
    parser.add_argument("--commands", type=float, nargs="*", default=(0.0, 0.25))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    configure_torch_backends()
    env_cfg = load_env_cfg(TASK, play=True)
    agent_cfg = load_rl_cfg(TASK)
    env_cfg.scene.num_envs = int(args.num_envs)
    env_cfg.commands["twist"].teleop = False
    env_cfg.commands["twist"].debug_vis = False
    env_cfg.commands["twist"].resampling_time_range = (1.0e9, 1.0e9)
    env_cfg.events["push_robot"].interval_range_s = (1.0e9, 1.0e9)
    env_cfg.events["push_robot"].params["velocity_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0)}
    env_cfg.events["base_com"].params["ranges"] = {
        0: tuple(args.com_x_range),
        1: (0.0, 0.0),
        2: (0.0, 0.0),
    }
    env_cfg.events["foot_friction"].params["ranges"] = tuple(args.friction_range)
    env_cfg.events["foot_friction"].params["shared_random"] = True
    env_cfg.terminations.clear()

    base_env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(env, asdict(agent_cfg), device=args.device)
    runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)

    robot = base_env.scene["robot"]
    model = base_env.sim.model
    trunk_body = robot.indexing.body_ids[robot.body_names.index("trunk")]
    com_x = model.body_ipos[:, trunk_body, 0].detach().cpu().numpy().copy()
    nominal_com_x = float(np.median(com_x) - np.mean(args.com_x_range))
    com_dx = com_x - nominal_com_x
    geom_names = robot.geom_names
    foot_geom = robot.indexing.geom_ids[geom_names.index("left_foot_collision")]
    friction = model.geom_friction[:, foot_geom, 0].detach().cpu().numpy().copy()

    n = base_env.num_envs
    bias_deg = np.array([IMU_BIASES_DEG[i % len(IMU_BIASES_DEG)] for i in range(n)], dtype=np.float32)
    bias_rad = torch.tensor(np.deg2rad(bias_deg), device=base_env.device)

    site_index = {name: i for i, name in enumerate(robot.site_names)}
    feet = [site_index["left_foot"], site_index["right_foot"]]
    gravity_w = torch.tensor([0.0, 0.0, -1.0], device=base_env.device).expand(n, 3)
    steps = int(round(args.seconds / base_env.step_dt))

    results = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "nominal_trunk_com_x": nominal_com_x,
        "imu_bias_convention": "positive = IMU reports trunk pitched further forward than it is",
        "commands": [],
    }
    for vx in args.commands:
        obs, _ = env.reset()
        alive = np.ones(n, dtype=bool)
        fall_time = np.full(n, np.nan)
        pitch_hist, lat_hist, fore_hist = [], [], []
        for step in range(steps):
            set_command(base_env, vx)
            obs = apply_imu_bias(obs, bias_rad)
            with torch.no_grad():
                obs, _, _, _ = env.step(policy(obs))
            quat = robot.data.root_link_quat_w
            g_b = quat_apply_inverse(quat, gravity_w)
            pitch = torch.rad2deg(torch.asin(g_b[:, 0].clamp(-1, 1))).cpu().numpy()
            site_w = robot.data.site_pos_w[:, feet, :]
            rel = site_w - robot.data.root_link_pos_w[:, None, :]
            site_b = quat_apply_inverse(quat[:, None, :].expand(-1, 2, -1).reshape(-1, 4), rel.reshape(-1, 3)).reshape(n, 2, 3)
            lat = (site_b[:, 0, 1] - site_b[:, 1, 1]).cpu().numpy()
            fore = (site_b[:, 0, 0] - site_b[:, 1, 0]).abs().cpu().numpy()
            height = robot.data.root_link_pos_w[:, 2].cpu().numpy()
            fallen_now = alive & ((height < 0.12) | (np.abs(pitch) > 50))
            fall_time[fallen_now] = step * base_env.step_dt
            alive &= ~fallen_now
            pitch_hist.append(np.where(alive, pitch, np.nan))
            lat_hist.append(np.where(alive, lat, np.nan))
            fore_hist.append(np.where(alive, fore, np.nan))
        pitch_arr = np.asarray(pitch_hist)
        lat_arr = np.asarray(lat_hist)
        fore_arr = np.asarray(fore_hist)
        with np.errstate(all="ignore"):
            pitch_mean = np.nanmean(pitch_arr, axis=0)
            lat_mean = np.nanmean(lat_arr, axis=0)
            lat_max = np.nanmax(lat_arr, axis=0)
            fore_max = np.nanmax(fore_arr, axis=0)
        fell = ~alive
        fall_pitch_sign = []

        def summarize(mask: np.ndarray) -> dict[str, float]:
            if not mask.any():
                return {"n": 0}
            with np.errstate(all="ignore"):
                return {
                    "n": int(mask.sum()),
                    "fall_rate": float(fell[mask].mean()),
                    "pitch_mean_deg": float(np.nanmean(pitch_mean[mask])),
                    "foot_lateral_sep_mean_mm": float(np.nanmean(lat_mean[mask]) * 1000),
                    "foot_lateral_sep_max_mm": float(np.nanmean(lat_max[mask]) * 1000),
                    "foot_fore_aft_sep_max_mm": float(np.nanmean(fore_max[mask]) * 1000),
                }

        by_bias = {f"{b:+.0f}": summarize(bias_deg == b) for b in IMU_BIASES_DEG}
        com_edges = np.linspace(args.com_x_range[0], args.com_x_range[1], 6)
        by_com = {}
        for lo, hi in zip(com_edges[:-1], com_edges[1:]):
            mask = (com_dx >= lo) & (com_dx < hi + 1e-9) & (bias_deg == 0.0)
            by_com[f"{lo*1000:+.0f}..{hi*1000:+.0f}mm"] = summarize(mask)
        fr_edges = np.linspace(args.friction_range[0], args.friction_range[1], 5)
        by_friction = {}
        for lo, hi in zip(fr_edges[:-1], fr_edges[1:]):
            mask = (friction >= lo) & (friction < hi + 1e-9) & (bias_deg == 0.0)
            by_friction[f"{lo:.2f}..{hi:.2f}"] = summarize(mask)
        # Which way do falls go?
        last_pitch = np.array([
            pitch_arr[np.nanargmax(np.isnan(pitch_arr[:, i])) - 1, i] if fell[i] else np.nan for i in range(n)
        ])
        with np.errstate(all="ignore"):
            backward_falls = float(np.nanmean(last_pitch[fell] < 0)) if fell.any() else None
        results["commands"].append(
            {
                "vx": vx,
                "overall": summarize(np.ones(n, dtype=bool)),
                "backward_fall_fraction": backward_falls,
                "by_imu_bias_deg": by_bias,
                "by_trunk_com_dx_imu0": by_com,
                "by_foot_friction_imu0": by_friction,
            }
        )
        print(json.dumps(results["commands"][-1], indent=2), flush=True)

    env.close()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
