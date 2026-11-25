#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

"""Which joint zero offset reproduces a real walk recording? Replays the recording's commands
in the deploy sim with candidate offsets and ranks them against the real left/right indicators.

    PYTHONPATH=src uv run --group sim src/debug/fit_zero_offsets.py logs/walk_rec/walk_X.npz
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze_walk_rec import load  # noqa: E402
from compare_walk_recs import indicators  # noqa: E402
from sim_replay import run_replay  # noqa: E402

KEYS = ["lean_R stand", "lean_R move", "hip_roll_q stand", "hip_roll_q move", "hip_roll_lead",
        "knee_q", "ankle_roll_lead", "ankle_pitch_lead"]


def main() -> None:
    real, meta = load(Path(sys.argv[1]))
    target = indicators(real, meta)
    cases = {"nominal": {}}
    for joint in ("right_hip_roll", "left_hip_roll", "right_ankle_roll", "left_ankle_roll"):
        for e in (-4.0, -2.0, 2.0, 4.0):
            cases[f"{joint} {e:+.0f}"] = {joint: e}
    results = {}
    for name, offsets in cases.items():
        sim, sim_meta = run_replay(real["cmd"], offsets_deg=offsets)
        results[name] = indicators(sim, sim_meta)
    real_v = np.array([target[k] for k in KEYS])
    print(f"{'case':24s}" + "".join(f" {k[:13]:>13s}" for k in KEYS) + "   distance")
    print(f"{'REAL':24s}" + "".join(f" {v:13.2f}" for v in real_v))
    ranked = sorted(results.items(), key=lambda kv: np.linalg.norm(np.array([kv[1][k] for k in KEYS]) - real_v))
    for name, r in ranked:
        v = np.array([r[k] for k in KEYS])
        print(f"{name:24s}" + "".join(f" {x:13.2f}" for x in v) + f"   {np.linalg.norm(v - real_v):8.2f}")


if __name__ == "__main__":
    main()
