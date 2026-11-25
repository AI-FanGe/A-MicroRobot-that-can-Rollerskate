"""Measure signed left/right drift of a checkpoint under a pure forward command."""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict

import numpy as np
import torch

import mjlab.tasks  # noqa: F401
import mjlab_xiaoxiaobai.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.torch import configure_torch_backends


def yaw_from_quat_wxyz(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def set_forward_command(env: ManagerBasedRlEnv, vx: float) -> None:
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


def evaluate(checkpoint: str, vx: float, seconds: float, num_envs: int, device: str) -> dict:
    env_cfg = load_env_cfg("Mjlab-Velocity-Xiaoxiaobai", play=True)
    agent_cfg = load_rl_cfg("Mjlab-Velocity-Xiaoxiaobai")
    env_cfg.scene.num_envs = num_envs
    env_cfg.commands["twist"].teleop = False
    env_cfg.commands["twist"].debug_vis = False

    base_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    runner = (load_runner_cls("Mjlab-Velocity-Xiaoxiaobai") or MjlabOnPolicyRunner)(
        env, asdict(agent_cfg), device=device
    )
    runner.load(checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)

    robot = base_env.scene["robot"]
    joint_index = {name: i for i, name in enumerate(robot.joint_names)}
    site_index = {name: i for i, name in enumerate(robot.site_names)}
    contact = base_env.scene.sensors["feet_ground_contact"]
    left_id, right_id = site_index["left_foot"], site_index["right_foot"]

    obs, _ = env.reset()
    set_forward_command(base_env, vx)
    for _ in range(30):
        set_forward_command(base_env, vx)
        with torch.no_grad():
            obs, _, _, _ = env.step(policy(obs))

    start_xy = robot.data.root_link_pos_w[:, :2].clone()
    start_yaw = yaw_from_quat_wxyz(robot.data.root_link_quat_w)

    vy_b, yaw_rate, hip_yaw_l, hip_yaw_r = [], [], [], []
    foot_yaw_l, foot_yaw_r = [], []
    contact_l, contact_r = [], []
    last_air = []

    steps = int(round(seconds / base_env.step_dt))
    for _ in range(steps):
        set_forward_command(base_env, vx)
        with torch.no_grad():
            obs, _, _, _ = env.step(policy(obs))

        vy_b.append(robot.data.root_link_lin_vel_b[:, 1].detach())
        yaw_rate.append(robot.data.root_link_ang_vel_b[:, 2].detach())
        q = robot.data.joint_pos
        hip_yaw_l.append(q[:, joint_index["left_hip_yaw"]].detach())
        hip_yaw_r.append(q[:, joint_index["right_hip_yaw"]].detach())

        site_quat = robot.data.site_quat_w[:, [left_id, right_id], :]
        local_fwd = torch.zeros(site_quat.shape[0], 2, 3, device=site_quat.device, dtype=site_quat.dtype)
        local_fwd[..., 0] = 1.0
        from mjlab.utils.lab_api.math import quat_apply

        site_fwd_w = quat_apply(site_quat.reshape(-1, 4), local_fwd.reshape(-1, 3)).reshape(-1, 2, 3)
        root_quat = robot.data.root_link_quat_w[:, None, :].expand(-1, 2, -1)
        site_fwd_b = quat_apply_inverse(root_quat.reshape(-1, 4), site_fwd_w.reshape(-1, 3)).reshape(-1, 2, 3)
        foot_yaw_l.append(torch.atan2(site_fwd_b[:, 0, 1], site_fwd_b[:, 0, 0]).detach())
        foot_yaw_r.append(torch.atan2(site_fwd_b[:, 1, 1], site_fwd_b[:, 1, 0]).detach())

        found = contact.data.found
        if found.dim() == 3:
            found = found.any(dim=-1)
        # Contact primary is foot (right), foot_2 (left).
        contact_r.append(found[:, 0].float().detach())
        contact_l.append(found[:, 1].float().detach())
        if contact.data.last_air_time is not None:
            air = contact.data.last_air_time
            if air.dim() == 3:
                air = air[..., 0]
            last_air.append(air.detach())

    end_xy = robot.data.root_link_pos_w[:, :2]
    end_yaw = yaw_from_quat_wxyz(robot.data.root_link_quat_w)
    delta = end_xy - start_xy
    heading = start_yaw
    forward = torch.cos(heading) * delta[:, 0] + torch.sin(heading) * delta[:, 1]
    # +Y is left in the robot frame; world left of start heading:
    leftward = -torch.sin(heading) * delta[:, 0] + torch.cos(heading) * delta[:, 1]

    def mean_std(xs: list[torch.Tensor]) -> tuple[float, float]:
        t = torch.stack(xs, dim=0).float()
        return float(t.mean().item()), float(t.std().item())

    vy_m, vy_s = mean_std(vy_b)
    wz_m, wz_s = mean_std(yaw_rate)
    lhy_m, _ = mean_std(hip_yaw_l)
    rhy_m, _ = mean_std(hip_yaw_r)
    lfy_m, _ = mean_std(foot_yaw_l)
    rfy_m, _ = mean_std(foot_yaw_r)
    cl_m, _ = mean_std(contact_l)
    cr_m, _ = mean_std(contact_r)
    air_t = torch.stack(last_air, dim=0).float() if last_air else None

    env.close()
    return {
        "checkpoint": checkpoint,
        "vx_command": vx,
        "seconds": seconds,
        "num_envs": num_envs,
        "forward_progress_m": float(forward.mean().item()),
        "leftward_progress_m": float(leftward.mean().item()),
        "rightward_progress_m": float((-leftward).mean().item()),
        "yaw_change_deg": float(torch.rad2deg(end_yaw - start_yaw).mean().item()),
        "body_vy_mean": vy_m,
        "body_vy_std": vy_s,
        "yaw_rate_mean": wz_m,
        "yaw_rate_std": wz_s,
        "left_hip_yaw_deg": math.degrees(lhy_m),
        "right_hip_yaw_deg": math.degrees(rhy_m),
        "left_foot_yaw_deg": math.degrees(lfy_m),
        "right_foot_yaw_deg": math.degrees(rfy_m),
        "left_contact_frac": cl_m,
        "right_contact_frac": cr_m,
        "left_last_air_s": float(air_t[..., 1].mean().item()) if air_t is not None else None,
        "right_last_air_s": float(air_t[..., 0].mean().item()) if air_t is not None else None,
        "frac_envs_ending_right": float((leftward < 0).float().mean().item()),
        "frac_envs_ending_left": float((leftward > 0).float().mean().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vx", type=float, default=0.4)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    configure_torch_backends()
    stats = evaluate(args.checkpoint, args.vx, args.seconds, args.num_envs, args.device)
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
