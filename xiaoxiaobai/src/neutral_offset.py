# SPDX-License-Identifier: GPL-3.0-or-later

"""Persistent joint-space offsets for the real robot's visual neutral pose.

Policy and MuJoCo keep using the model zero. On the hardware, a saved offset
is added to every goal and removed from every position read, so the calibrated
pose is what the policy sees as zero.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from constants import MOTOR_TO_ID


NEUTRAL_OFFSET_PATH = Path(__file__).resolve().parents[1] / "calibration" / "neutral_offset.json"


def load_neutral_offsets(path: Path = NEUTRAL_OFFSET_PATH) -> dict[str, float]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("joints", payload) if isinstance(payload, dict) else {}
    if not isinstance(raw, dict):
        raise ValueError(f"Neutral offset file has no joint map: {path}")
    offsets: dict[str, float] = {}
    for name, value in raw.items():
        if name not in MOTOR_TO_ID:
            continue
        offsets[str(name)] = float(value)
    return offsets


def save_neutral_offsets(
    offsets: Mapping[str, float],
    path: Path = NEUTRAL_OFFSET_PATH,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joints = {
        name: float(offsets[name])
        for name in sorted(offsets)
        if name in MOTOR_TO_ID
    }
    payload = {"unit": "rad", "joints": joints}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
