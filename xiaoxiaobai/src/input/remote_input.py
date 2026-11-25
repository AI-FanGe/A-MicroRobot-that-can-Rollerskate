# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

"""File-backed remote input used by the console and the robot API to drive the control loop."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

from input.input_source import InputSource, UserInput

REMOTE_INPUT_PATH = Path("/tmp/xiaoxiaobai_remote_input.json")
STALE_S = 0.8


def write_remote_input(
    *,
    active_moves: list[str] | set[str] | None = None,
    velocity: dict[str, float] | None = None,
    upper: dict | None = None,
    tuning: dict | None = None,
    head_yaw: float | None = None,
    path: Path = REMOTE_INPUT_PATH,
) -> None:
    """Write one remote snapshot. velocity=None keeps the local (gamepad) velocity; tuning
    ({"walk_agent", "skate_agent", "imu_roll_corr_deg", "imu_pitch_corr_deg", "mirror_policy", "kp_vin_comp",
    "imu_auto_trim", "imu_mount_roll_deg", "imu_mount_pitch_deg", "imu_calib_seq"}) replaces the keyboard's;
    head_yaw [deg, + left] points the head of the walk there (with the head_front move)."""
    payload = {
        "active_moves": sorted(set(active_moves or ())),
        "velocity": None
        if velocity is None
        else {
            "vx": float(velocity.get("vx", 0.0)),
            "vy": float(velocity.get("vy", 0.0)),
            "vtheta": float(velocity.get("vtheta", 0.0)),
        },
        "ts": time.time(),
    }
    if upper is not None:
        payload["upper"] = upper
    if tuning is not None:
        payload["tuning"] = tuning
    if head_yaw is not None:
        payload["head_yaw"] = float(head_yaw)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    os.replace(tmp, path)


class RemoteInputSource(InputSource):
    """Read walk / velocity commands written by the console or the robot API."""

    def __init__(self, path: str | Path = REMOTE_INPUT_PATH, stale_s: float = STALE_S) -> None:
        self.path = Path(path)
        self.stale_s = stale_s

    def read(self) -> UserInput:
        return self.read_fresh() or UserInput()

    def read_fresh(self) -> UserInput | None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict):
            return None
        try:
            age = time.time() - float(data.get("ts", 0.0))
        except (TypeError, ValueError):
            return None
        if age > self.stale_s:
            return None
        velocity = data.get("velocity")
        moves = data.get("active_moves") or []
        user = UserInput(
            active_moves=set(str(name) for name in moves),
            velocity={
                "vx": float((velocity or {}).get("vx", 0.0)),
                "vy": float((velocity or {}).get("vy", 0.0)),
                "vtheta": float((velocity or {}).get("vtheta", 0.0)),
            },
            has_velocity=isinstance(velocity, dict),
            upper=data.get("upper") if isinstance(data.get("upper"), dict) else None,
        )
        try:
            head_yaw = float(data["head_yaw"])
        except (KeyError, TypeError, ValueError):
            pass
        else:
            if math.isfinite(head_yaw):
                user.head_yaw = head_yaw
        tuning = data.get("tuning")
        if isinstance(tuning, dict):
            try:
                agent = int(tuning["walk_agent"])
                roll = float(tuning["imu_roll_corr_deg"])
                pitch = float(tuning["imu_pitch_corr_deg"])
            except (KeyError, TypeError, ValueError):
                pass
            else:
                user.walk_agent, user.imu_roll_corr_deg, user.imu_pitch_corr_deg = agent, roll, pitch
                user.has_tuning = True
                user.mirror_policy = tuning.get("mirror_policy") is True
                for key in ("kp_vin_comp", "imu_auto_trim"):
                    if isinstance(tuning.get(key), bool):
                        setattr(user, key, tuning[key])
                for key in ("imu_mount_roll_deg", "imu_mount_pitch_deg", "imu_calib_seq"):
                    try:
                        value = float(tuning[key])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if math.isfinite(value):
                        setattr(user, key, value)
                try:
                    user.skate_agent = int(tuning["skate_agent"])
                except (KeyError, TypeError, ValueError):
                    pass
        return user
