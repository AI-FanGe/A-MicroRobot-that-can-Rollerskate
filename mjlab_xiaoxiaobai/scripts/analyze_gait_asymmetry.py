"""Analyze left/right gait asymmetry for a Xiaoxiaobai velocity checkpoint."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
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


@dataclass(frozen=True)
class Scenario:
    name: str
    lin_vel_x: float
    seconds: float


SCENARIOS = (
    Scenario("forward_0p25", 0.25, 10.0),
    Scenario("backward_0p25", -0.25, 10.0),
)


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


def summarize_series(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def summarize_pair(left: np.ndarray, right: np.ndarray) -> dict[str, object]:
    left_summary = summarize_series(left)
    right_summary = summarize_series(right)
    return {
        "left": left_summary,
        "right": right_summary,
        "left_minus_right_mean": float(left_summary["mean"] - right_summary["mean"]),
        "abs_mean_diff": float(abs(left_summary["mean"] - right_summary["mean"])),
    }


def mean_run_duration(contact: np.ndarray, value: bool, dt: float) -> float | None:
    durations: list[float] = []
    target = bool(value)
    # contact shape: [time, env]
    for env_idx in range(contact.shape[1]):
        series = contact[:, env_idx].astype(bool)
        if series.size == 0:
            continue
        start = 0
        current = series[0]
        for idx in range(1, series.size):
            if series[idx] != current:
                if current == target:
                    durations.append((idx - start) * dt)
                start = idx
                current = series[idx]
        if current == target:
            durations.append((series.size - start) * dt)
    return float(np.mean(durations)) if durations else None


def mean_touchdown_period(contact: np.ndarray, dt: float) -> float | None:
    periods: list[float] = []
    for env_idx in range(contact.shape[1]):
        series = contact[:, env_idx].astype(bool)
        touchdowns = np.flatnonzero((~series[:-1]) & series[1:]) + 1
        if touchdowns.size >= 2:
            periods.extend(np.diff(touchdowns).astype(np.float64) * dt)
    return float(np.mean(periods)) if periods else None


def mean_lr_touchdown_offset(left_contact: np.ndarray, right_contact: np.ndarray, dt: float) -> float | None:
    offsets: list[float] = []
    for env_idx in range(left_contact.shape[1]):
        left_series = left_contact[:, env_idx].astype(bool)
        right_series = right_contact[:, env_idx].astype(bool)
        left_td = np.flatnonzero((~left_series[:-1]) & left_series[1:]) + 1
        right_td = np.flatnonzero((~right_series[:-1]) & right_series[1:]) + 1
        if left_td.size == 0 or right_td.size == 0:
            continue
        for idx in left_td:
            offsets.append(float(np.min(np.abs(right_td - idx)) * dt))
    return float(np.mean(offsets)) if offsets else None


def transition_positions(x_body: np.ndarray, contact: np.ndarray, touchdown: bool) -> np.ndarray:
    positions: list[float] = []
    for env_idx in range(contact.shape[1]):
        series = contact[:, env_idx].astype(bool)
        if touchdown:
            indices = np.flatnonzero((~series[:-1]) & series[1:]) + 1
        else:
            indices = np.flatnonzero(series[:-1] & (~series[1:])) + 1
        positions.extend(float(x_body[idx, env_idx]) for idx in indices)
    return np.asarray(positions, dtype=np.float64)


def configure_env(num_envs: int):
    env_cfg = load_env_cfg("Mjlab-Velocity-Xiaoxiaobai", play=True)
    agent_cfg = load_rl_cfg("Mjlab-Velocity-Xiaoxiaobai")
    env_cfg.scene.num_envs = int(num_envs)
    env_cfg.commands["twist"].teleop = False
    env_cfg.commands["twist"].debug_vis = False
    env_cfg.commands["twist"].resampling_time_range = (1.0e9, 1.0e9)
    if "push_robot" in env_cfg.events:
        env_cfg.events["push_robot"].interval_range_s = (1.0e9, 1.0e9)
        env_cfg.events["push_robot"].params["velocity_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0)}
    # Avoid reset discontinuities while measuring gait timing.
    env_cfg.terminations.clear()
    return env_cfg, agent_cfg


def evaluate_scenario(policy, env, base_env: ManagerBasedRlEnv, scenario: Scenario, warmup_steps: int) -> dict[str, object]:
    robot = base_env.scene["robot"]
    joint_index = {name: i for i, name in enumerate(robot.joint_names)}
    site_index = {name: i for i, name in enumerate(robot.site_names)}
    left_site_id = site_index["left_foot"]
    right_site_id = site_index["right_foot"]
    contact_sensor = base_env.scene.sensors["feet_ground_contact"]

    obs, _ = env.reset()
    for _ in range(warmup_steps):
        set_command(base_env, scenario.lin_vel_x)
        with torch.no_grad():
            obs, _, _, _ = env.step(policy(obs))

    steps = int(round(scenario.seconds / base_env.step_dt))
    foot_x_body: list[np.ndarray] = []
    joints: dict[str, list[np.ndarray]] = {
        "left_hip_pitch": [],
        "right_hip_pitch": [],
        "left_knee": [],
        "right_knee": [],
        "left_ankle_pitch": [],
        "right_ankle_pitch": [],
    }
    contacts: list[np.ndarray] = []
    root_start = robot.data.root_link_pos_w[:, :2].detach().clone()

    for _ in range(steps):
        set_command(base_env, scenario.lin_vel_x)
        with torch.no_grad():
            obs, _, _, _ = env.step(policy(obs))

        site_w = robot.data.site_pos_w[:, [left_site_id, right_site_id], :]
        rel_w = site_w - robot.data.root_link_pos_w[:, None, :]
        root_quat = robot.data.root_link_quat_w[:, None, :].expand(-1, 2, -1)
        site_b = quat_apply_inverse(root_quat.reshape(-1, 4), rel_w.reshape(-1, 3)).reshape(-1, 2, 3)
        foot_x_body.append(site_b[..., 0].detach().cpu().numpy())

        q = robot.data.joint_pos.detach().cpu().numpy()
        for name in joints:
            joints[name].append(q[:, joint_index[name]])

        found = contact_sensor.data.found
        if found.dim() == 3:
            found = found.any(dim=-1)
        # Contact sensor pattern order is right foot body, then left foot body.
        right_contact = found[:, 0].detach().cpu().numpy().astype(bool)
        left_contact = found[:, 1].detach().cpu().numpy().astype(bool)
        contacts.append(np.stack([left_contact, right_contact], axis=-1))

    root_end = robot.data.root_link_pos_w[:, :2].detach()
    root_delta = (root_end - root_start).detach().cpu().numpy()
    foot_x = np.asarray(foot_x_body, dtype=np.float64)
    contact = np.asarray(contacts, dtype=bool)
    dt = float(base_env.step_dt)

    left_x = foot_x[:, :, 0]
    right_x = foot_x[:, :, 1]
    left_amp = left_x.max(axis=0) - left_x.min(axis=0)
    right_amp = right_x.max(axis=0) - right_x.min(axis=0)
    left_touch = transition_positions(left_x, contact[:, :, 0], touchdown=True)
    right_touch = transition_positions(right_x, contact[:, :, 1], touchdown=True)
    left_lift = transition_positions(left_x, contact[:, :, 0], touchdown=False)
    right_lift = transition_positions(right_x, contact[:, :, 1], touchdown=False)

    joint_arrays = {name: np.rad2deg(np.asarray(values, dtype=np.float64)) for name, values in joints.items()}
    # Mirror right-side pitch signs into the left-side convention; knees use flexion magnitude.
    joint_pairs = {
        "hip_pitch_equiv_deg": (joint_arrays["left_hip_pitch"], -joint_arrays["right_hip_pitch"]),
        "knee_flexion_deg": (np.abs(joint_arrays["left_knee"]), np.abs(joint_arrays["right_knee"])),
        "ankle_pitch_equiv_deg": (joint_arrays["left_ankle_pitch"], -joint_arrays["right_ankle_pitch"]),
    }

    joint_summary: dict[str, object] = {}
    for name, (left, right) in joint_pairs.items():
        left_env_amp = left.max(axis=0) - left.min(axis=0)
        right_env_amp = right.max(axis=0) - right.min(axis=0)
        joint_summary[name] = {
            "angle": summarize_pair(left.reshape(-1), right.reshape(-1)),
            "amplitude": summarize_pair(left_env_amp, right_env_amp),
        }

    left_contact = contact[:, :, 0]
    right_contact = contact[:, :, 1]
    contact_summary = {
        "left_fraction": float(left_contact.mean()),
        "right_fraction": float(right_contact.mean()),
        "fraction_diff_left_minus_right": float(left_contact.mean() - right_contact.mean()),
        "left_stance_mean_s": mean_run_duration(left_contact, True, dt),
        "right_stance_mean_s": mean_run_duration(right_contact, True, dt),
        "left_swing_mean_s": mean_run_duration(left_contact, False, dt),
        "right_swing_mean_s": mean_run_duration(right_contact, False, dt),
        "left_touchdown_period_s": mean_touchdown_period(left_contact, dt),
        "right_touchdown_period_s": mean_touchdown_period(right_contact, dt),
        "nearest_lr_touchdown_offset_s": mean_lr_touchdown_offset(left_contact, right_contact, dt),
    }
    if contact_summary["left_stance_mean_s"] is not None and contact_summary["right_stance_mean_s"] is not None:
        contact_summary["stance_diff_left_minus_right_s"] = float(
            contact_summary["left_stance_mean_s"] - contact_summary["right_stance_mean_s"]
        )
    if contact_summary["left_swing_mean_s"] is not None and contact_summary["right_swing_mean_s"] is not None:
        contact_summary["swing_diff_left_minus_right_s"] = float(
            contact_summary["left_swing_mean_s"] - contact_summary["right_swing_mean_s"]
        )

    return {
        "name": scenario.name,
        "command_vx_mps": scenario.lin_vel_x,
        "seconds": scenario.seconds,
        "root_progress_x_m": summarize_series(root_delta[:, 0]),
        "root_progress_y_m": summarize_series(root_delta[:, 1]),
        "foot_relative_x_m": {
            "left": {
                "min_mean": float(left_x.min(axis=0).mean()),
                "max_mean": float(left_x.max(axis=0).mean()),
                "amplitude_mean": float(left_amp.mean()),
                "amplitude_std": float(left_amp.std()),
                "touchdown_x_mean": float(left_touch.mean()) if left_touch.size else None,
                "liftoff_x_mean": float(left_lift.mean()) if left_lift.size else None,
            },
            "right": {
                "min_mean": float(right_x.min(axis=0).mean()),
                "max_mean": float(right_x.max(axis=0).mean()),
                "amplitude_mean": float(right_amp.mean()),
                "amplitude_std": float(right_amp.std()),
                "touchdown_x_mean": float(right_touch.mean()) if right_touch.size else None,
                "liftoff_x_mean": float(right_lift.mean()) if right_lift.size else None,
            },
            "amplitude_diff_left_minus_right_m": float(left_amp.mean() - right_amp.mean()),
            "amplitude_abs_diff_m": float(abs(left_amp.mean() - right_amp.mean())),
        },
        "contacts": contact_summary,
        "joints": joint_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--warmup-seconds", type=float, default=2.0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    configure_torch_backends()
    env_cfg, agent_cfg = configure_env(args.num_envs)
    base_env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)

    runner = (load_runner_cls("Mjlab-Velocity-Xiaoxiaobai") or MjlabOnPolicyRunner)(
        env, asdict(agent_cfg), device=args.device
    )
    runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)
    warmup_steps = int(round(args.warmup_seconds / base_env.step_dt))

    try:
        results = {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "num_envs": int(args.num_envs),
            "step_dt": float(base_env.step_dt),
            "warmup_seconds": float(args.warmup_seconds),
            "scenarios": [
                evaluate_scenario(policy, env, base_env, scenario, warmup_steps)
                for scenario in SCENARIOS
            ],
        }
    finally:
        env.close()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
