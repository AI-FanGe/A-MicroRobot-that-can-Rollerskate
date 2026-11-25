"""Export a dance checkpoint for the robot: the actor as ONNX (one `obs` input, like the walk)
and the reference it tracks as a .npz, both into the deployment repo.

usage: export_dance.py <ckpt> [--out-dir ../xiaoxiaobai/src/agents] [--name dance]
env:   TASK (Mjlab-Dance-Xiaoxiaobai)

The .npz holds, per 50 Hz frame, the reference joint positions and velocities of the actuated
joints (`joint_names` order) and the trunk orientation (w, x, y, z); the deployment builds the
motion_command and motion_anchor_ori_b observations from them (xiaoxiaobai src/moves/dance.py).
It also holds the head yaw motor's reference (head_pos), which no policy drives: the routine
tasks (Mjlab-Gymnastics-Xiaoxiaobai) play it on the head; for the dances it stays at 0.
"""
import argparse
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

import mjlab_xiaoxiaobai.tasks  # noqa: F401
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/mjlab_xiaoxiaobai/scripts"))
from export_onnx import actuated_joint_metadata  # noqa: E402

TASK = os.environ.get("TASK", "Mjlab-Dance-Xiaoxiaobai")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[2] / "xiaoxiaobai/src/agents")
    ap.add_argument("--name", default="dance")
    args = ap.parse_args()

    cfg = load_env_cfg(TASK, play=True)
    cfg.scene.num_envs = 1
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device="cpu"))
    u = env.unwrapped
    runner = load_runner_cls(TASK)(env, asdict(load_rl_cfg(TASK)), device="cpu")
    runner.load(str(args.checkpoint), load_cfg={"actor": True}, strict=True, map_location="cpu")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = args.out_dir / f"{args.name}.onnx"
    # The plain actor (MotionTrackingOnPolicyRunner bundles the motion behind a time_step input).
    MjlabOnPolicyRunner.export_policy_to_onnx(runner, str(args.out_dir), onnx_path.name)
    meta = actuated_joint_metadata(u, run_path=f"{args.checkpoint.parent.name}/{args.checkpoint.name}")
    cmd = u.command_manager.get_term("motion")
    meta["motion_file"] = os.path.basename(cmd.cfg.motion_file)
    attach_metadata_to_onnx(str(onnx_path), meta)

    robot = u.scene["robot"]
    action = u.action_manager.get_term("joint_pos")
    names = list(action.target_names)
    ids = [robot.joint_names.index(n) for n in names]
    motion = cmd.motion
    anchor = cmd.cfg.body_names.index(cmd.cfg.anchor_body_name)
    ref_path = args.out_dir / f"{args.name}_ref.npz"
    np.savez(
        ref_path,
        fps=np.float32(1.0 / u.step_dt),
        joint_names=np.array(names),
        joint_pos=motion.joint_pos[:, ids].cpu().numpy().astype(np.float32),
        joint_vel=motion.joint_vel[:, ids].cpu().numpy().astype(np.float32),
        anchor_quat=motion.body_quat_w[:, anchor].cpu().numpy().astype(np.float32),
        head_pos=motion.joint_pos[:, robot.joint_names.index("head")].cpu().numpy().astype(np.float32),
    )
    print(f"[INFO] {onnx_path} ({len(meta['observation_names'])} obs terms: {meta['observation_names']})")
    print(f"[INFO] {ref_path}: {motion.joint_pos.shape[0]} frames at {1.0 / u.step_dt:.0f} Hz, joints {names}")
    env.close()


if __name__ == "__main__":
    main()
