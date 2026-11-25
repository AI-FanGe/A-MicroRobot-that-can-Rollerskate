from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import imageio
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
    lin_vel_y: float
    ang_vel_z: float
    push: bool = False


# Speeds stay inside the trained command box: vx ±0.5, vy ±0.3, wz ±0.75.
SCENARIOS = (
    Scenario("stand_no_push", 0.0, 0.0, 0.0),
    Scenario("forward_slow_no_push", 0.2, 0.0, 0.0),
    Scenario("forward_medium_no_push", 0.35, 0.0, 0.0),
    Scenario("forward_fast_no_push", 0.5, 0.0, 0.0),
    Scenario("backward_slow_no_push", -0.2, 0.0, 0.0),
    Scenario("backward_fast_no_push", -0.5, 0.0, 0.0),
    Scenario("lateral_left_slow_no_push", 0.0, 0.15, 0.0),
    Scenario("lateral_left_fast_no_push", 0.0, 0.3, 0.0),
    Scenario("lateral_right_slow_no_push", 0.0, -0.15, 0.0),
    Scenario("lateral_right_fast_no_push", 0.0, -0.3, 0.0),
    Scenario("rotate_left_no_push", 0.0, 0.0, 0.75),
    Scenario("rotate_right_no_push", 0.0, 0.0, -0.75),
    Scenario("forward_left_turn_no_push", 0.35, 0.0, 0.5),
    Scenario("forward_right_turn_no_push", 0.35, 0.0, -0.5),
    Scenario("diagonal_left_no_push", 0.3, 0.15, 0.0),
    Scenario("diagonal_right_no_push", 0.3, -0.15, 0.0),
    Scenario("forward_medium_with_push", 0.35, 0.0, 0.0, push=True),
    Scenario("lateral_left_with_push", 0.0, 0.2, 0.0, push=True),
    Scenario("rotate_left_with_push", 0.0, 0.0, 0.75, push=True),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="Mjlab-Velocity-Xiaoxiaobai",
        help="Registered mjlab task id.",
    )
    parser.add_argument(
        "--checkpoint",
        default=(
            "logs/rsl_rl/mjlab_xiaoxiaobai_velocity/"
            "2026-08-25_01-53-24/model_16299.pt"
        ),
        help="Checkpoint file to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "logs/rsl_rl/mjlab_xiaoxiaobai_velocity/"
            "2026-08-25_01-53-24/videos/eval_fixed/model_16299"
        ),
        help="Directory where videos will be written.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=None,
        help="Optional scenario names to record. Defaults to all scenarios.",
    )
    return parser.parse_args()


def set_push(env: ManagerBasedRlEnv, enabled: bool) -> None:
    push_cfg = env.event_manager.get_term_cfg("push_robot")
    if enabled:
        push_cfg.params["velocity_range"] = {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
        }
        push_cfg.interval_range_s = (1.0, 2.0)
    else:
        push_cfg.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
        }
        push_cfg.interval_range_s = (1.0e9, 1.0e9)


def set_command(env: ManagerBasedRlEnv, scenario: Scenario) -> None:
    term = env.command_manager.get_term("twist")
    vel = np.array(
        [scenario.lin_vel_x, scenario.lin_vel_y, scenario.ang_vel_z],
        dtype=np.float32,
    )
    if hasattr(term, "teleop_vel"):
        term.teleop_vel[:] = vel
    term.vel_command_b[:] = torch.tensor(vel, device=env.device, dtype=term.vel_command_b.dtype)
    term.vel_command_w[:] = term.vel_command_b
    term.is_standing_env[:] = False
    term.is_heading_env[:] = False
    term.is_world_env[:] = False
    term.is_forward_env[:] = False
    if hasattr(term, "is_rotation_env"):
        term.is_rotation_env[:] = False
    # Hold this command for the whole clip. Otherwise the command term resamples
    # and the policy acts on a different velocity than the one being recorded.
    term.time_left[:] = 1.0e6


def frame_to_uint8(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 4:
        frame = frame[0]
    if frame.dtype != np.uint8:
        frame = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)
    return frame


def main() -> None:
    args = parse_args()
    configure_torch_backends()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios = SCENARIOS
    if args.scenarios:
        requested = set(args.scenarios)
        scenarios = tuple(scenario for scenario in SCENARIOS if scenario.name in requested)
        missing = requested - {scenario.name for scenario in scenarios}
        if missing:
            raise ValueError(f"Unknown scenarios: {sorted(missing)}")

    env_cfg = load_env_cfg(args.task, play=True)
    agent_cfg = load_rl_cfg(args.task)
    env_cfg.scene.num_envs = 1
    # One nominal robot, so the clip shows the policy rather than one random draw.
    for name in ("dof_armature_randomization", "dof_friction_randomization"):
        if name in env_cfg.events:
            env_cfg.events[name].params["ranges"] = (1.0, 1.0)
    if "base_com" in env_cfg.events:
        env_cfg.events["base_com"].params["ranges"] = {0: (0.0, 0.0), 1: (0.0, 0.0), 2: (0.0, 0.0)}
    if "foot_friction" in env_cfg.events:
        env_cfg.events["foot_friction"].params["ranges"] = (1.0, 1.0)
    env_cfg.viewer.width = args.width
    env_cfg.viewer.height = args.height
    env_cfg.commands["twist"].teleop = False
    env_cfg.commands["twist"].debug_vis = False

    base_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)

    runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
        str(checkpoint),
        load_cfg={"actor": True},
        strict=True,
        map_location=device,
    )
    policy = runner.get_inference_policy(device=device)

    fps = args.fps or base_env.metadata.get("render_fps", 50)
    steps = int(round(args.seconds / base_env.step_dt))
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": str(checkpoint.resolve()),
        "fps": fps,
        "width": args.width,
        "height": args.height,
        "seconds": args.seconds,
        "step_dt": base_env.step_dt,
        "scenarios": [],
    }

    try:
        for index, scenario in enumerate(scenarios, start=1):
            set_push(base_env, scenario.push)
            obs, _ = env.reset()
            set_command(base_env, scenario)

            for _ in range(args.warmup_steps):
                set_command(base_env, scenario)
                with torch.no_grad():
                    actions = policy(obs)
                obs, _, _, _ = env.step(actions)

            video_path = output_dir / f"{index:02d}_{scenario.name}.mp4"
            print(
                "[INFO] Recording",
                video_path,
                "command=",
                (scenario.lin_vel_x, scenario.lin_vel_y, scenario.ang_vel_z),
                "push=",
                scenario.push,
                flush=True,
            )
            robot = base_env.scene["robot"]
            joint_index = {name: i for i, name in enumerate(robot.joint_names)}
            site_index = {name: i for i, name in enumerate(robot.site_names)}
            contact = base_env.scene.sensors.get("feet_ground_contact")
            start_xy = robot.data.root_link_pos_w[0, :2].detach().cpu().numpy().copy()
            traces: dict[str, list[float]] = {name: [] for name in (
                "left_hip_pitch", "right_hip_pitch",
                "left_knee", "right_knee",
                "left_ankle_pitch", "right_ankle_pitch",
                "left_hip_roll", "right_hip_roll",
                "left_shoulder_pitch", "right_shoulder_pitch",
                "left_foot_y", "right_foot_y",
                "foot_center_x",
            )}
            contact_hist: list[list[float]] = []

            with imageio.get_writer(
                str(video_path),
                fps=fps,
                codec="libx264",
                quality=8,
                macro_block_size=1,
            ) as writer:
                for _ in range(steps):
                    set_command(base_env, scenario)
                    with torch.no_grad():
                        actions = policy(obs)
                    obs, _, _, _ = env.step(actions)
                    q = robot.data.joint_pos[0].detach().cpu().numpy()
                    for name in (
                        "left_hip_pitch", "right_hip_pitch",
                        "left_knee", "right_knee",
                        "left_ankle_pitch", "right_ankle_pitch",
                        "left_hip_roll", "right_hip_roll",
                        "left_shoulder_pitch", "right_shoulder_pitch",
                    ):
                        traces[name].append(float(np.rad2deg(q[joint_index[name]])))
                    left_id = site_index["left_foot"]
                    right_id = site_index["right_foot"]
                    site_w = robot.data.site_pos_w[:, [left_id, right_id], :]
                    rel = site_w - robot.data.root_link_pos_w[:, None, :]
                    root_quat = robot.data.root_link_quat_w[:, None, :].expand(-1, 2, -1)
                    site_b = quat_apply_inverse(
                        root_quat.reshape(-1, 4),
                        rel.reshape(-1, 3),
                    ).reshape(1, 2, 3)
                    traces["left_foot_y"].append(float(site_b[0, 0, 1].item()))
                    traces["right_foot_y"].append(float(site_b[0, 1, 1].item()))
                    traces["foot_center_x"].append(
                        float(0.5 * (site_b[0, 0, 0] + site_b[0, 1, 0]).item())
                    )
                    if contact is not None:
                        found = contact.data.found[0].detach().cpu().numpy().reshape(-1)
                        contact_hist.append(found.astype(float).tolist())
                    frame = base_env.render()
                    if frame is not None:
                        writer.append_data(frame_to_uint8(frame))

            end_xy = robot.data.root_link_pos_w[0, :2].detach().cpu().numpy()
            metrics = {
                "dx_m": float(end_xy[0] - start_xy[0]),
                "dy_m": float(end_xy[1] - start_xy[1]),
            }
            for name, series in traces.items():
                arr = np.asarray(series, dtype=np.float32)
                metrics[name] = {
                    "mean": float(arr.mean()),
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                    "amp": float(arr.max() - arr.min()),
                }
            if contact_hist:
                contact_arr = np.asarray(contact_hist, dtype=np.float32)
                metrics["contact_frac"] = contact_arr.mean(axis=0).tolist()
            print("[INFO] metrics", scenario.name, json.dumps(metrics, ensure_ascii=False), flush=True)

            manifest["scenarios"].append(
                {
                    **asdict(scenario),
                    "video": str(video_path.resolve()),
                    "metrics": metrics,
                }
            )

        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"[INFO] Wrote manifest: {manifest_path}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
