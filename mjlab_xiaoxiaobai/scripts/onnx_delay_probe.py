"""Run an exported ONNX velocity policy under extra control latency.

Works with whichever ``mjlab_xiaoxiaobai`` package is first on ``sys.path``, so the
same script can evaluate the upstream policy in the upstream env by putting the
original repo's ``src`` on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

import mjlab.tasks  # noqa: F401
import mjlab_xiaoxiaobai
import mjlab_xiaoxiaobai.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.lab_api.math import quat_apply_inverse

TASK = "Mjlab-Velocity-Xiaoxiaobai"


def set_command(env: ManagerBasedRlEnv, vx: float) -> None:
    term = env.command_manager.get_term("twist")
    vel = torch.tensor([vx, 0.0, 0.0], device=env.device, dtype=term.vel_command_b.dtype)
    term.vel_command_b[:] = vel
    term.vel_command_w[:] = vel
    term.is_standing_env[:] = False
    term.is_heading_env[:] = False
    term.is_world_env[:] = False
    term.is_forward_env[:] = False
    if hasattr(term, "is_rotation_env"):
        term.is_rotation_env[:] = False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=TASK)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--commands", type=float, nargs="*", default=(0.0, 0.25))
    parser.add_argument("--delays", type=int, nargs="*", default=(0, 1, 2))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    print(f"[INFO] mjlab_xiaoxiaobai from {Path(mjlab_xiaoxiaobai.__file__).resolve()}", flush=True)
    env_cfg = load_env_cfg(args.task, play=True)
    env_cfg.scene.num_envs = int(args.num_envs)
    env_cfg.commands["twist"].debug_vis = False
    env_cfg.commands["twist"].resampling_time_range = (1.0e9, 1.0e9)
    if hasattr(env_cfg.commands["twist"], "teleop"):
        env_cfg.commands["twist"].teleop = False
    env_cfg.events["push_robot"].interval_range_s = (1.0e9, 1.0e9)
    env_cfg.events["push_robot"].params["velocity_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0)}
    env_cfg.events.pop("encoder_bias", None)
    env_cfg.terminations.clear()
    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)

    session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    def policy(obs: torch.Tensor) -> torch.Tensor:
        batch = obs.detach().cpu().numpy().astype(np.float32)
        out = np.stack([session.run(None, {input_name: batch[i : i + 1]})[0][0] for i in range(batch.shape[0])])
        return torch.from_numpy(out).to(env.device)

    robot = env.scene["robot"]
    site_index = {name: i for i, name in enumerate(robot.site_names)}
    feet = [site_index["left_foot"], site_index["right_foot"]]
    n = env.num_envs
    gravity_w = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(n, 3)
    steps = int(round(args.seconds / env.step_dt))

    rows = []
    for delay in args.delays:
        for vx in args.commands:
            obs_dict, _ = env.reset()
            obs = obs_dict["actor"]
            queue: deque[torch.Tensor] = deque()
            alive = np.ones(n, dtype=bool)
            last_pitch = np.zeros(n)
            pitch_sum = np.zeros(n)
            com_sum = np.zeros(n)
            lat_sum = np.zeros(n)
            count = np.zeros(n)
            lat_max = np.zeros(n)
            fore_max = np.zeros(n)
            joint_sum = torch.zeros_like(robot.data.joint_pos[0])
            joint_count = 0
            for _ in range(steps):
                set_command(env, vx)
                action = policy(obs)
                queue.append(action)
                applied = queue.popleft() if len(queue) > delay else torch.zeros_like(action)
                obs_dict, _, _, _, _ = env.step(applied)
                obs = obs_dict["actor"]
                quat = robot.data.root_link_quat_w
                g_b = quat_apply_inverse(quat, gravity_w)
                pitch = torch.rad2deg(torch.asin(g_b[:, 0].clamp(-1, 1))).cpu().numpy()
                rel = robot.data.site_pos_w[:, feet, :] - robot.data.root_link_pos_w[:, None, :]
                site_b = quat_apply_inverse(
                    quat[:, None, :].expand(-1, 2, -1).reshape(-1, 4), rel.reshape(-1, 3)
                ).reshape(n, 2, 3)
                lat = (site_b[:, 0, 1] - site_b[:, 1, 1]).cpu().numpy()
                fore = (site_b[:, 0, 0] - site_b[:, 1, 0]).abs().cpu().numpy()
                com_w = robot.data.data.subtree_com[:, robot.data.indexing.root_body_id]
                mid_w = robot.data.site_pos_w[:, feet, :].mean(dim=1)
                w, x, y, z = quat.unbind(-1)
                yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
                d = com_w[:, :2] - mid_w[:, :2]
                com_x = (torch.cos(yaw) * d[:, 0] + torch.sin(yaw) * d[:, 1]).cpu().numpy()
                height = robot.data.root_link_pos_w[:, 2].cpu().numpy()
                fallen = alive & ((height < 0.6 * height.max()) | (np.abs(pitch) > 50))
                alive &= ~fallen
                last_pitch = np.where(alive | fallen, pitch, last_pitch)
                pitch_sum += np.where(alive, pitch, 0.0)
                com_sum += np.where(alive, com_x, 0.0)
                lat_sum += np.where(alive, lat, 0.0)
                count += alive
                lat_max = np.where(alive, np.maximum(lat_max, lat), lat_max)
                fore_max = np.where(alive, np.maximum(fore_max, fore), fore_max)
                joint_sum += (robot.data.joint_pos - robot.data.default_joint_pos).mean(dim=0)
                joint_count += 1
            fell = ~alive
            row = {
                "label": args.label,
                "extra_delay_ms": delay * 20,
                "vx": vx,
                "fall_rate": float(fell.mean()),
                "backward_fall_fraction": float((last_pitch[fell] < 0).mean()) if fell.any() else None,
                "pitch_mean_deg": float((pitch_sum / np.maximum(count, 1)).mean()),
                "com_ahead_of_feet_mm": float((com_sum / np.maximum(count, 1)).mean() * 1000),
                "foot_lateral_sep_mean_mm": float((lat_sum / np.maximum(count, 1)).mean() * 1000),
                "foot_lateral_sep_max_mm": float(lat_max.mean() * 1000),
                "foot_fore_aft_sep_max_mm": float(fore_max.mean() * 1000),
                "mean_joint_offset_from_default_deg": {
                    name: round(float(torch.rad2deg(value / max(joint_count, 1))), 2)
                    for name, value in zip(robot.joint_names, joint_sum)
                    if any(key in name for key in ("hip", "knee", "ankle"))
                },
            }
            rows.append(row)
            print(json.dumps(row), flush=True)

    env.close()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
