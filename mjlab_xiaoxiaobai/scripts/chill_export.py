"""Turn a chill_motion.py loop into the robot's clip (xiaoxiaobai moves/chill.py): the first second
eases in from the base pose and is played once, the rest loops. Its last LOOP_BLEND_S seconds
are blended into the frames that follow the ease-in, so the loop wraps without a jump.
With --once (chill_motion.py wake) the clip is played once as it is: loop_start is its length.

usage: chill_export.py <raw.npz> <clip.npz> [--once] [--preview preview_raw.npz]
--preview also writes a raw motion of the ease-in and two loops, for chill_sim.py.
"""
import argparse
from pathlib import Path

import numpy as np

EASE_IN_S = 1.0  # chill_motion.py eases from the base pose over the first second
LOOP_BLEND_S = 2.0


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--once", action="store_true", help="played once, not looped")
    ap.add_argument("--preview", type=Path)
    args = ap.parse_args()

    d = np.load(args.raw)
    fps = float(d["fps"])
    q = np.concatenate([d["joint_pos"], d["head_pitch"][:, None]], axis=1).astype(np.float64)
    n = len(q)
    step = np.abs(np.diff(q, axis=0)).max()
    if args.once:
        loop_start = n
        print(f"{args.raw.name}: {n / fps:.1f} s, played once (largest frame step {np.degrees(step):.2f} deg)")
    else:
        e, b = round(EASE_IN_S * fps), round(LOOP_BLEND_S * fps)
        loop_start = e + b
        w = smoothstep((np.arange(b) + 1) / b)[:, None]
        q[n - b:] = (1 - w) * q[n - b:] + w * q[e:loop_start]
        jump = np.abs(q[loop_start] - q[-1]).max()
        step = np.abs(np.diff(q[loop_start:], axis=0)).max()
        print(f"{args.raw.name}: {n / fps:.0f} s, loop from {loop_start / fps:.1f} s, wrap step "
              f"{np.degrees(jump):.2f} deg (largest frame step {np.degrees(step):.2f} deg)")

    names = [str(x) for x in d["joint_names"]]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, fps=np.float32(fps), joint_names=np.array(names), joint_pos=q[:, :-1].astype(np.float32),
             head_pitch=q[:, -1].astype(np.float32), loop_start=np.int32(loop_start),
             section_names=d["section_names"], section_starts=d["section_starts"].astype(np.float32))
    print("saved", args.out)

    if args.preview is not None:
        p = q if args.once else np.concatenate([q, q[loop_start:], q[loop_start:]])
        m = len(p)
        extra = {"prop_boxes": d["prop_boxes"]} if "prop_boxes" in d.files else {}
        np.savez(args.preview, fps=np.float32(fps), joint_names=np.array(names), joint_pos=p[:, :-1].astype(np.float32),
                 root_pos=np.repeat(d["root_pos"][:1], m, 0), root_quat=np.repeat(d["root_quat"][:1], m, 0),
                 head_pitch=p[:, -1].astype(np.float32), section_names=d["section_names"],
                 section_starts=d["section_starts"], **extra)
        print("saved", args.preview)


if __name__ == "__main__":
    main()
