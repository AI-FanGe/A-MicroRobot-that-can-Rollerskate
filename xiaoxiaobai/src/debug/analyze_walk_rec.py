#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

"""Analyse a walk recording made by `make run` (logs/walk_rec/walk_*.npz, or a walk_*.part left by
a session cut by a power loss).

    PYTHONPATH=src uv run --group sim src/debug/analyze_walk_rec.py [file.npz]

Without an argument the newest recording is used. Reports, per constant-command
segment, trunk lean, yaw-rate tracking and a mirrored left/right comparison of every
leg joint (training mirror: swap L/R and negate). Over the whole recording it also
compares the tracking error and response lag of each left/right motor pair.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from walk_recorder import load_recording

LEG = ["hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch", "ankle_roll"]
DT = 0.02


def load(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    data, meta = load_recording(path)
    cols, i = {}, 0
    for name, width in meta["columns"].items():
        cols[name] = data[:, i] if width == 1 else data[:, i:i + width]
        i += width
    return cols, meta


def quat_to_mat(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def segments(cmd: np.ndarray, valid: np.ndarray, min_s: float, settle_s: float) -> list[tuple[int, int]]:
    out, start = [], None
    for k in range(len(cmd) + 1):
        same = k < len(cmd) and valid[k] and start is not None and np.allclose(cmd[k], cmd[start], atol=1e-4)
        if same:
            continue
        if start is not None and (k - start) * DT >= min_s:
            out.append((start + int(settle_s / DT), k))
        start = k if k < len(cmd) and valid[k] else None
    return out


def lag_ticks(target: np.ndarray, q: np.ndarray, max_lag: int = 10) -> int:
    a, b = target - target.mean(), q - q.mean()
    scores = [np.dot(a[: len(a) - s], b[s:]) for s in range(max_lag + 1)]
    return int(np.argmax(scores))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", nargs="?", type=Path)
    parser.add_argument("--min-seg-s", type=float, default=2.0)
    parser.add_argument("--settle-s", type=float, default=0.6)
    args = parser.parse_args()

    path = args.file or max([*Path("logs/walk_rec").glob("walk_*.npz"), *Path("logs/walk_rec").glob("walk_*.part")],
                            key=lambda p: p.stat().st_mtime)
    c, meta = load(path)
    joints = meta["joints"]
    jx = {n: i for i, n in enumerate(joints)}
    ran = c["policy_ran"] > 0.5
    t = c["t"] - c["t"][0]

    local_onnx = Path("src/agents/walk_low_battery_1.onnx")
    local_md5 = hashlib.md5(local_onnx.read_bytes()).hexdigest() if local_onnx.exists() else None
    offsets = {k: round(float(np.degrees(v)), 2) for k, v in meta["neutral_offsets"].items() if abs(v) > 1e-6}
    print(f"file: {path}")
    print(f"start: {meta['start_wallclock']}  duration: {t[-1]:.1f} s  ticks: {len(t)}  stop: {meta['stop_reason']}")
    print(f"policy md5: {meta['agent_md5']} ({'same as' if meta['agent_md5'] == local_md5 else 'DIFFERENT from'} local src/agents/walk_low_battery_1.onnx)")
    print(f"neutral offsets [deg]: {offsets or 'none'}")
    print(f"policy not running (fallen / starting): {np.sum(~ran)} ticks;  loop overruns >20 ms: {np.sum(c['loop_ms'] > 20)}"
          f";  proxy current p95/max: {np.nanpercentile(c['current_proxy_a'], 95):.1f}/{np.nanmax(c['current_proxy_a']):.1f} A")

    r_mount = quat_to_mat(meta["imu_mount_quat"])
    gyro_b = c["gyro"] @ r_mount.T
    g = c["projected_gravity"]
    lean_right = np.degrees(np.arcsin(np.clip(-g[:, 1], -1, 1)))
    pitch_fwd = np.degrees(np.arcsin(np.clip(g[:, 0], -1, 1)))

    print("\n=== Segments (lean_R > 0: trunk leans to the robot's right; pitch > 0: leans forward) ===")
    for a, b in segments(c["cmd"], ran, args.min_seg_s, args.settle_s):
        vx, vy, wz = c["cmd"][a]
        print(f"\n[{t[a]:6.1f}-{t[b - 1]:6.1f} s] cmd vx={vx:+.2f} vy={vy:+.2f} wz={wz:+.2f}  "
              f"lean_R={lean_right[a:b].mean():+5.1f}±{lean_right[a:b].std():.1f}°  "
              f"pitch={pitch_fwd[a:b].mean():+5.1f}°  yaw_rate={gyro_b[a:b, 2].mean():+.2f} rad/s")
        print(f"  {'joint':12s} {'mean L / -R [deg]':>20s} {'range L / R':>14s} {'track_err L / R':>17s}")
        for j in LEG:
            ql, qr = c["q"][a:b, jx[f"left_{j}"]], c["q"][a:b, jx[f"right_{j}"]]
            el = c["target"][a:b, jx[f"left_{j}"]] - ql
            er = c["target"][a:b, jx[f"right_{j}"]] - qr
            rng = lambda x: np.degrees(np.percentile(x, 95) - np.percentile(x, 5))
            rms = lambda x: np.degrees(np.sqrt(np.nanmean(x ** 2)))
            print(f"  {j:12s} {np.degrees(ql.mean()):+8.1f} / {np.degrees(-qr.mean()):+8.1f}"
                  f"   {rng(ql):5.1f} / {rng(qr):5.1f}   {rms(el):6.1f} / {rms(er):6.1f}")

    moving = ran & (np.abs(c["cmd"]).sum(axis=1) > 1e-3)
    if moving.sum() > 100:
        print("\n=== Motor pairs over all moving ticks (target lead ≈ load; lag = response delay) ===")
        print(f"  {'joint':12s} {'rms(target-q) L / R':>21s} {'lag L / R [ticks]':>19s} {'|dq| rms L / R':>16s}")
        for j in LEG:
            vals = []
            for side in ("left", "right"):
                k = jx[f"{side}_{j}"]
                q, tg, dq = c["q"][moving, k], c["target"][moving, k], c["dq"][moving, k]
                vals.append((np.degrees(np.sqrt(np.nanmean((tg - q) ** 2))), lag_ticks(tg, q), np.sqrt(np.mean(dq ** 2))))
            (el, ll, vl), (er, lr, vr) = vals
            flag = "  <-- asymmetric" if max(el, er) > 1.5 * min(el, er) + 1.0 or abs(ll - lr) >= 2 else ""
            print(f"  {j:12s} {el:9.1f} / {er:<9.1f} {ll:8d} / {lr:<8d} {vl:6.2f} / {vr:<6.2f}{flag}")


if __name__ == "__main__":
    main()
