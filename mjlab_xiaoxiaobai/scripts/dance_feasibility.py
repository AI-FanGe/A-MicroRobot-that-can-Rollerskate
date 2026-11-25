"""Second-by-second feasibility of a G1 dance CSV (LAFAN1 layout) for Xiaoxiaobai: hops, fast turns,
deep crouches and hip angles past Xiaoxiaobai's joint ranges, then the clip's windows with the
fewest problem seconds.

usage: dance_feasibility.py <csv> [<csv> ...]
env:   WINDOW (30)  TOP (3)
"""
import os
import sys
from pathlib import Path

import mujoco
import numpy as np

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec

FPS = 30
WINDOW = int(os.environ.get("WINDOW", 30))
TOP = int(os.environ.get("TOP", 3))
# Xiaoxiaobai's hip ranges in anatomical terms [deg]: abduction out/in, yaw out/in, knee flexion.
HIP_ABD_OUT, HIP_ABD_IN = 32.0, 18.0
HIP_YAW_OUT, HIP_YAW_IN = 60.0, 30.0
KNEE_MAX = 80.0
TURN_MAX = 200.0  # deg/s, over 0.2 s
CROUCH = 0.8  # pelvis below this fraction of standing height
AIR_CLEARANCE = 0.03  # both soles this far above the floor: a hop

model = get_spec().compile()
data = mujoco.MjData(model)
jid = {model.joint(i).name: model.jnt_qposadr[i] for i in range(model.njnt)}
foot_geoms = [[g for g in range(model.ngeom) if (model.geom(g).name or "").startswith(f"{side}_foot")]
              for side in ("left", "right")]


def sole_heights() -> list[float]:
    """Lowest point of each foot's collision capsules [m]: a raised heel keeps the toe down."""
    out = []
    for geoms in foot_geoms:
        low = np.inf
        for g in geoms:
            axis_z = data.geom_xmat[g].reshape(3, 3)[2, 2]
            low = min(low, data.geom_xpos[g, 2] - abs(axis_z) * model.geom_size[g, 1] - model.geom_size[g, 0])
        out.append(low)
    return out


def deg(q, name):
    return np.degrees(q[:, jid[f"{name}_joint"]])


def analyse(csv: Path) -> None:
    raw = np.loadtxt(csv, delimiter=",")
    q = np.concatenate([raw[:, :3], raw[:, [6, 3, 4, 5]], raw[:, 7:]], axis=1)
    n = len(q)
    sole_z = np.empty((n, 2))
    for i in range(n):
        data.qpos[:] = q[i]
        mujoco.mj_kinematics(model, data)
        sole_z[i] = sole_heights()
    air = sole_z.min(1) > np.percentile(sole_z, 2) + AIR_CLEARANCE
    w, x, y, z = q[:, 3:7].T
    yaw = np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    turn = np.degrees(np.abs(yaw[6:] - yaw[:-6]) / 0.2)
    turn = np.concatenate([turn, np.repeat(turn[-1], 6)])
    pelvis = q[:, 2]
    crouch = pelvis < CROUCH * np.median(pelvis)
    abd_l, abd_r = deg(q, "left_hip_roll"), -deg(q, "right_hip_roll")
    yaw_l, yaw_r = deg(q, "left_hip_yaw"), -deg(q, "right_hip_yaw")
    knee = np.maximum(deg(q, "left_knee"), deg(q, "right_knee"))
    hip = ((np.maximum(abd_l, abd_r) > HIP_ABD_OUT) | (np.minimum(abd_l, abd_r) < -HIP_ABD_IN)
           | (np.maximum(yaw_l, yaw_r) > HIP_YAW_OUT) | (np.minimum(yaw_l, yaw_r) < -HIP_YAW_IN))
    flags = {"跳": air, "快转": turn > TURN_MAX, "深蹲": crouch | (knee > KNEE_MAX), "髋超限": hip}
    secs = n // FPS
    per_sec = {k: v[: secs * FPS].reshape(secs, FPS).any(1) for k, v in flags.items()}
    bad = np.any(np.stack(list(per_sec.values())), axis=0)
    waist = np.degrees(np.abs(q[:, jid["waist_yaw_joint"]]))

    print(f"\n== {csv.stem}: {secs} s, 有问题的秒数 {bad.sum()} ({bad.mean() * 100:.0f}%)"
          + "".join(f" | {k} {v.sum()} s" for k, v in per_sec.items())
          + f" | 腰扭转 >30° 的时间 {(waist > 30).mean() * 100:.0f}%（Xiaoxiaobai 没有腰，要靠髋转）")
    line = "".join("x" if b else "." for b in bad)
    for s in range(0, secs, 60):
        print(f"  {s // 60}:{s % 60:02d} {line[s:s + 60]}")
    if secs < WINDOW:
        return
    counts = np.convolve(bad.astype(int), np.ones(WINDOW, int), mode="valid")
    picked = []
    for s in np.argsort(counts, kind="stable"):
        if all(abs(s - p) >= WINDOW for p in picked):
            picked.append(int(s))
        if len(picked) == TOP:
            break
    for s in picked:
        e = s + WINDOW
        sl = slice(s * FPS, e * FPS)
        print(f"  最可行 {WINDOW} 秒: {s // 60}:{s % 60:02d}-{e // 60}:{e % 60:02d}  问题秒数 {counts[s]}"
              + "".join(f" {k} {per_sec[k][s:e].sum()}" for k in flags)
              + f" | 转身最快 {turn[sl].max():.0f}°/s, 骨盆最低 {pelvis[sl].min() / np.median(pelvis):.0%} 站高"
              + f", 手臂: 肩最大 {max(np.abs(deg(q, 'left_shoulder_pitch')[sl]).max(), np.abs(deg(q, 'right_shoulder_pitch')[sl]).max()):.0f}°")


for path in sys.argv[1:]:
    analyse(Path(path))
