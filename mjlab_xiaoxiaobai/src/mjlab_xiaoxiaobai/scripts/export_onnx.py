# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

"""Export a trained .pt checkpoint to ONNX.

Usage:
    # Latest checkpoint of the latest run:
    uv run python src/mjlab_xiaoxiaobai/scripts/export_onnx.py

    # Specific checkpoint file:
    uv run python src/mjlab_xiaoxiaobai/scripts/export_onnx.py --checkpoint logs/rsl_rl/mjlab_xiaoxiaobai_velocity/2026-06-25_21-24-43/model_5000.pt

    # Straight into the deployment repo:
    uv run python src/mjlab_xiaoxiaobai/scripts/export_onnx.py --output ../xiaoxiaobai/src/agents/walk_low_battery_1.onnx
"""

import argparse
from dataclasses import asdict
from pathlib import Path

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import get_checkpoint_path

TASK = "Mjlab-Velocity-Xiaoxiaobai"
LOG_ROOT = Path("logs/rsl_rl/mjlab_xiaoxiaobai_velocity")
OUTPUT_FILENAME = "policy.onnx"


def get_latest_model_checkpoint(log_root: Path) -> Path:
    checkpoints = sorted(log_root.glob("**/model_*.pt"), key=lambda path: path.stat().st_mtime)
    if checkpoints:
        return checkpoints[-1]
    return get_checkpoint_path(log_root)


def actuated_joint_metadata(env: ManagerBasedRlEnv, run_path: str) -> dict:
    """Base metadata restricted to joints that have a motor.

    The passive head_wobble_* joints have no actuator, which the shared mjlab exporter
    (patched process-wide by other installed task packages) cannot handle.
    """
    robot = env.scene["robot"]
    ctrl_id = {a.target.split("/")[-1]: a.id for a in robot.spec.actuators}
    keep = [i for i, n in enumerate(robot.joint_names) if n in ctrl_id]
    names = [robot.joint_names[i] for i in keep]
    ids = [ctrl_id[n] for n in names]
    default = robot.data.default_joint_pos[0].cpu().tolist()
    action = env.action_manager.get_term("joint_pos")
    scale = action._scale
    metadata = {
        "run_path": run_path,
        "joint_names": names,
        "joint_stiffness": env.sim.mj_model.actuator_gainprm[ids, 0].tolist(),
        "joint_damping": (-env.sim.mj_model.actuator_biasprm[ids, 2]).tolist(),
        "default_joint_pos": [default[i] for i in keep],
        "command_names": list(env.command_manager.active_terms),
        "observation_names": env.observation_manager.active_terms["actor"],
        "action_scale": scale[0].cpu().tolist() if hasattr(scale, "cpu") else scale,
    }
    terms = dict(zip(env.observation_manager.active_terms["actor"], env.observation_manager._group_obs_term_cfgs["actor"]))
    if "gait_phase" in terms:
        metadata["phase_period_steps"] = terms["gait_phase"].params["period_steps"]
    if "stroke_rate" in env.action_manager.active_terms:
        # One output after the joints: the phase advances clip(1 + scale * a, lo, hi) a tick.
        rate = env.action_manager.get_term("stroke_rate").cfg
        metadata["stroke_rate_scale"] = rate.scale
        metadata["stroke_rate_range"] = list(rate.rate_range)
    return metadata


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", type=str, default=TASK, help=f"Registered task id (default: {TASK})")
    p.add_argument("--log-root", type=Path, default=LOG_ROOT, help=f"RSL-RL log root (default: {LOG_ROOT})")
    p.add_argument("--checkpoint", type=Path, default=None, help="Path to a specific .pt file (default: latest)")
    p.add_argument("--output", type=Path, default=Path(OUTPUT_FILENAME), help=f"Output ONNX path (default: {OUTPUT_FILENAME})")
    p.add_argument("--device", type=str, default="cpu", help="Device for model loading (default: cpu)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    else:
        checkpoint_path = get_latest_model_checkpoint(args.log_root)
    if checkpoint_path.suffix != ".pt":
        raise FileNotFoundError(f"No model_*.pt checkpoint found under {args.log_root}")

    print(f"[INFO] Checkpoint: {checkpoint_path}")

    env_cfg = load_env_cfg(args.task, play=True)
    agent_cfg = load_rl_cfg(args.task)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
    env = RslRlVecEnvWrapper(env)

    runner_cls = load_runner_cls(args.task)
    runner = runner_cls(env, asdict(agent_cfg), device=args.device)
    runner.load(str(checkpoint_path), load_cfg={"actor": True}, strict=True, map_location=args.device)

    onnx_path = args.output
    if not onnx_path.is_absolute():
        onnx_path = Path.cwd() / onnx_path
    output_dir = onnx_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    runner.export_policy_to_onnx(str(output_dir), onnx_path.name)

    metadata = actuated_joint_metadata(env.unwrapped, run_path=str(checkpoint_path.parent.name))
    attach_metadata_to_onnx(str(onnx_path), metadata)

    print(f"[INFO] Exported: {onnx_path}")

    env.close()


if __name__ == "__main__":
    main()
