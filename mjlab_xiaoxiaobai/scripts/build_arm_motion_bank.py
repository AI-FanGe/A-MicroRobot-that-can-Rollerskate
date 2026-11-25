"""Arm trajectories for the free-arms walk (Mjlab-Velocity-Xiaoxiaobai-FreeArms).

The robot's upper body runs xiaoxiaobai's gesture engine (xiaoxiaobai/src/gestures.py) while the
walk policy drives the legs only. This records that engine, plus random poses it can be
sent to by hand, as joint targets at the policy rate:

  gestures  random gestures every 0-4 s, idle gestures and the alive layer in between
  poses     random poses anywhere in the arm ranges, reached in 0.25-1.5 s and held 0-6 s
  rest      the alive layer only

Usage: python scripts/build_arm_motion_bank.py [--streams 384] [--seconds 60]
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "xiaoxiaobai" / "src"))

import gestures  # noqa: E402
from gestures import POSE_LIMITS, REST, UpperBody, UpperBodyInput, to_joints  # noqa: E402

DT = 0.02
ARMS = ("right_shoulder_pitch", "right_shoulder_roll", "right_elbow",
        "left_shoulder_pitch", "left_shoulder_roll", "left_elbow")
ARM_KEYS = ("rp", "rr", "re", "lp", "lr", "le")
OUT = ROOT / "src" / "mjlab_xiaoxiaobai" / "tasks" / "data" / "arm_motion_bank.npz"


def gesture_stream(rng: random.Random, steps: int) -> list[dict[str, float]]:
    engine = UpperBody(rng.randrange(1 << 30))
    auto = rng.random() < 0.7
    next_play: float | None = rng.uniform(0.0, 2.0)
    out = []
    for i in range(steps):
        now = i * DT
        if engine.playing is None:
            if next_play is None:
                next_play = now + rng.uniform(0.0, 4.0)
            elif now >= next_play:
                engine.play("random", now, size=rng.uniform(0.6, 1.0))
                next_play = None
        out.append(engine.update(now, DT, REST, UpperBodyInput(auto=auto)))
    return out


def _ease(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * x * (x * (6.0 * x - 15.0) + 10.0)


def pose_stream(rng: random.Random, steps: int) -> list[dict[str, float]]:
    """Smooth moves between random poses; each arm moves on its own or both together."""
    def random_pose(current: dict[str, float]) -> dict[str, float]:
        pose = dict(current)
        sides = rng.choice((("r",), ("l",), ("r", "l"), ("r", "l")))
        for side in sides:
            if rng.random() < 0.15:
                for k in ("p", "r", "e"):
                    pose[side + k] = REST[side + k]
                continue
            lo, hi = POSE_LIMITS[side + "p"]
            pose[side + "p"] = rng.uniform(lo, hi) if rng.random() < 0.3 else rng.uniform(-60.0, hi)
            pose[side + "r"] = rng.uniform(*POSE_LIMITS[side + "r"])
            pose[side + "e"] = rng.uniform(*POSE_LIMITS[side + "e"])
        return pose

    out = []
    pose = dict(REST)
    start, target = dict(pose), dict(pose)
    t0, dur, hold = 0.0, 0.0, rng.uniform(0.0, 1.0)
    step_limit = gestures.MAX_SPEED_DEG_S * DT
    for i in range(steps):
        now = i * DT
        if now >= t0 + dur + hold:
            start, target = dict(pose), random_pose(pose)
            t0, dur = now, rng.uniform(0.25, 1.5)
            hold = rng.uniform(0.0, 6.0) if rng.random() < 0.4 else rng.uniform(0.0, 1.0)
        a = _ease((now - t0) / dur) if dur > 0 else 1.0
        for k in ARM_KEYS:
            want = start[k] + (target[k] - start[k]) * a
            pose[k] = min(max(want, pose[k] - step_limit), pose[k] + step_limit)
        out.append(dict(pose))
    return out


def rest_stream(rng: random.Random, steps: int) -> list[dict[str, float]]:
    engine = UpperBody(rng.randrange(1 << 30))
    return [engine.update(i * DT, DT, REST, UpperBodyInput(auto=False)) for i in range(steps)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--streams", type=int, default=384)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    steps = int(round(args.seconds / DT))
    kinds = ["gestures"] * 65 + ["poses"] * 27 + ["rest"] * 8
    bank = np.zeros((args.streams, steps, len(ARMS)), dtype=np.float16)
    counts: dict[str, int] = {}
    for s in range(args.streams):
        kind = kinds[s * len(kinds) // args.streams]
        counts[kind] = counts.get(kind, 0) + 1
        poses = {"gestures": gesture_stream, "poses": pose_stream, "rest": rest_stream}[kind](rng, steps)
        for t, pose in enumerate(poses):
            joints = to_joints({**REST, **pose})
            bank[s, t] = [joints[name] for name in ARMS]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, targets=bank, joints=np.array(ARMS), dt=np.array(DT))
    deg = np.degrees(bank.astype(np.float32))
    speed = np.abs(np.diff(deg, axis=1)).max() / DT
    print(f"{OUT}: {bank.shape} {counts}")
    for j, name in enumerate(ARMS):
        print(f"  {name:22s} {deg[..., j].min():7.1f} .. {deg[..., j].max():6.1f} deg")
    print(f"  max speed {speed:.0f} deg/s, {math.prod(bank.shape) * 2 / 1e6:.1f} MB raw")


if __name__ == "__main__":
    main()
