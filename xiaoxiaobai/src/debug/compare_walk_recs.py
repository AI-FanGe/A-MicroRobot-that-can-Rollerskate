#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

"""Left/right indicators of every walk recording side by side, each next to a nominal sim
replay of the same commands.

    PYTHONPATH=src uv run --group sim src/debug/compare_walk_recs.py [rec.npz ...]

Asymmetries are "left minus mirrored right" in degrees: positive hip_roll_q means the right
hip is more abducted, positive knee_q means the right knee is more bent.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze_walk_rec import load  # noqa: E402
from sim_replay import run_replay  # noqa: E402

MAX_LEAD = np.radians(60.0)


def strength(c: dict, k: int, mask: np.ndarray) -> float:
    """Joint speed per unit lead (target - q), best over 1..5 ticks of delay."""
    lead = c["target"][:, k] - c["q"][:, k]
    dq = np.gradient(c["q"][:, k]) / 0.02
    best = None
    for d in range(1, 6):
        idx = np.nonzero(mask[:-d] & mask[d:] & (np.abs(lead[:-d]) < MAX_LEAD))[0]
        x, y = lead[idx], dq[idx + d]
        a = float(np.dot(x, y) / np.dot(x, x))
        r2 = 1 - np.sum((y - a * x) ** 2) / np.sum((y - y.mean()) ** 2)
        if best is None or r2 > best[1]:
            best = (a, r2)
    return best[0]


def indicators(c: dict, meta: dict) -> dict:
    jx = {n: i for i, n in enumerate(meta["joints"])}
    ran = c["policy_ran"] > 0.5
    ran[:25] = False
    still = ran & (np.abs(c["cmd"]).sum(1) < 1e-3)
    moving = ran & ~still
    g = c["projected_gravity"]
    lean = np.degrees(np.arcsin(np.clip(-g[:, 1], -1, 1)))

    def asym(field: str, joint: str, mask: np.ndarray) -> float:
        x = c["target"] - c["q"] if field == "lead" else c[field]
        if not mask.any():
            return float("nan")
        return float(np.degrees(x[mask, jx[f"left_{joint}"]].mean() + x[mask, jx[f"right_{joint}"]].mean()))

    return {
        "seconds": len(c["t"]) * 0.02,
        "lean_R stand": float(lean[still].mean()) if still.any() else float("nan"),
        "lean_R move": float(lean[moving].mean()),
        "hip_roll_q stand": asym("q", "hip_roll", still),
        "hip_roll_q move": asym("q", "hip_roll", moving),
        "hip_roll_lead": asym("lead", "hip_roll", moving),
        "knee_q": asym("q", "knee", moving),
        "knee strength R/L": strength(c, jx["right_knee"], moving) / strength(c, jx["left_knee"], moving),
        "ankle_roll_lead": asym("lead", "ankle_roll", moving),
        "ankle_pitch_lead": asym("lead", "ankle_pitch", moving),
        "target jitter": float(np.degrees(np.abs(np.diff(c["target"][moving], axis=0))).mean()),
    }


def main() -> None:
    recs = sys.argv[1:] or sorted(glob.glob("logs/walk_rec/walk_*.npz"))
    rows = []
    for path in recs:
        real, meta = load(Path(path))
        sim, sim_meta = run_replay(real["cmd"])
        offsets = {k: round(float(np.degrees(v)), 1) for k, v in meta["neutral_offsets"].items() if abs(v) > 1e-6}
        rows.append((Path(path).stem.removeprefix("walk_"), indicators(real, meta), indicators(sim, sim_meta), offsets))

    names = list(rows[0][1])
    print(__doc__.strip().splitlines()[-2] + " " + __doc__.strip().splitlines()[-1])
    print(f"{'indicator':18s}" + "".join(f" | {n:>15s}" for n, *_ in rows) + " | sim replays (mean)")
    for k in names:
        sim_mean = np.nanmean([s[k] for _, _, s, _ in rows])
        print(f"{k:18s}" + "".join(f" | {r[k]:15.2f}" for _, r, _, _ in rows) + f" | {sim_mean:8.2f}")
    for name, *_, offsets in rows:
        print(f"neutral offsets {name}: {offsets or 'none'}")


if __name__ == "__main__":
    main()
