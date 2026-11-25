# SPDX-License-Identifier: GPL-3.0-or-later

"""Automatic trim of the IMU roll / pitch corrections while the walk runs.

A walk policy has no integral action: a constant disturbance it was not trained on (weaker
motors on a low battery, hot motors, a shifted load, a policy that never quite fit the robot)
leaves the trunk leaning a few degrees for good. The policy rights the trunk only part of the
way; this trim does the rest. Over steady stretches of walking (a command of at least MIN_VX
forward or backward, unchanged for IMU_TRIM_STEADY_S, no sideways or turning command; standing,
the trim only holds) it averages the trunk attitude the IMU reads, after the mount calibration
(imu_calib.py), over IMU_TRIM_WINDOW_S (several steps) per speed bin, added up over the short
stretches of a walk back and forth, and compares
- the roll with the one the same policy walks with in simulation, the median over the speeds
  (agents/posture_ref.json, debug/posture_ref.py): about upright;
- the pitch with the one the robot walked with at that speed when the loop started (its first
  IMU_TRIM_BASELINE_WINDOWS windows): the robot's own pitch differs from the simulated one by
  several degrees, and forcing that on it tipped it over; the drift is what is corrected.
The difference, divided by how far the policy moves the trunk per degree of correction (the gain
measured in simulation), is IMU_TRIM_GAIN of the correction that cancels it: an integral loop
around the policy.

Safety: windows swaying more than IMU_TRIM_MAX_STD_DEG are dropped; an axis pinned at its limit
for IMU_TRIM_GIVE_UP_WINDOWS windows while still pushing out is reset and no longer trimmed (the
robot does not follow it); the trim fades in over IMU_TRIM_FADE_S at each walk start.

Switched by UserInput.imu_auto_trim (web UI): switching it off clears everything.
A few sums per tick and one update per window: nothing for the control loop. Each walk policy
and mirroring keeps its own trim.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from constants import (IMU_TRIM_BASELINE_WINDOWS, IMU_TRIM_FADE_S, IMU_TRIM_GAIN, IMU_TRIM_GIVE_UP_WINDOWS,
                       IMU_TRIM_MAX_PITCH_DEG, IMU_TRIM_MAX_ROLL_DEG, IMU_TRIM_MAX_STD_DEG, IMU_TRIM_MAX_STEP_DEG,
                       IMU_TRIM_STEADY_S, IMU_TRIM_WINDOW_S)

POSTURE_REF_PATH = Path(__file__).resolve().parent / "agents" / "posture_ref.json"
STEADY_TOL = 0.03      # command change [m/s, rad/s] that restarts the settling (a hand on a stick wavers)
MIN_VX = 0.1           # m/s: standing or stepping in place the policy does not right the trunk (a trim only winds up)
MAX_VY = 0.02          # m/s: sideways commands lean the trunk on purpose
MAX_VTHETA = 0.15      # rad/s: so do turns (centripetal acceleration)
UPRIGHT_GZ = -0.9      # projected gravity z above this (trunk tilted > 25 deg): no update
VX_BIN = 0.1           # m/s: speeds sharing a pitch baseline
GAIN_RANGE = (0.5, 10.0)  # trust in the simulated gains [deg of lean per deg of correction]
LIMITS = (IMU_TRIM_MAX_ROLL_DEG, IMU_TRIM_MAX_PITCH_DEG)
AXES = ("roll", "pitch")
PRINT_STEP_DEG = 0.5   # report the trim every time it has moved this much


def load_posture_ref(path: Path = POSTURE_REF_PATH) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"Warning: no posture reference ({exc}): IMU auto trim and calibration off", end="\r\n", flush=True)
        return {}


def lean_deg(gravity) -> tuple[float, float]:
    """(lean right, lean forward) [deg] of a projected gravity, as the scheduler prints them."""
    gx, gy = gravity[0], gravity[1]
    return math.degrees(math.asin(max(-1.0, min(1.0, -gy)))), math.degrees(math.asin(max(-1.0, min(1.0, gx))))


@dataclass
class _Trim:
    trim: list[float] = field(default_factory=lambda: [0.0, 0.0])  # roll, pitch [deg]
    pinned: list[int] = field(default_factory=lambda: [0, 0])      # windows pushing past the limit
    off: list[bool] = field(default_factory=lambda: [False, False])  # axis given up
    baselines: dict = field(default_factory=dict)  # speed bin -> [sum of window pitches, count]
    printed: tuple[float, float] = (0.0, 0.0)


class TiltTrim:
    def __init__(self, ref: dict | None = None) -> None:
        self._refs: dict = (load_posture_ref() if ref is None else ref).get("agents", {})
        self._trims: dict[tuple, _Trim] = {}
        self._key: tuple | None = None
        self._trim = _Trim()
        self._ref: dict | None = None
        self._missing: set[str] = set()
        self._active_since: float | None = None
        self._command: np.ndarray | None = None
        self._steady_since = 0.0
        self._last_s: float | None = None
        # speed bin -> [gx, gy, gx^2, gy^2 sums, ticks, seconds] of its window being filled
        self._windows: dict[int, list[float]] = {}

    @property
    def learned(self) -> bool:
        """Some trim or baseline was recorded."""
        return any(t.baselines or any(t.trim) or any(t.off) for t in self._trims.values())

    @property
    def given_up(self) -> list[str]:
        """Axes of the running policy no longer trimmed (the robot did not follow them)."""
        return [axis for axis, off in zip(AXES, self._trim.off) if off]

    def update(self, now_s: float, gravity: list[float], velocity, active: bool, agent: str,
               mirrored: bool) -> tuple[float, float]:
        """One control tick: the (roll, pitch) trim [deg] to add to the IMU corrections.

        gravity: projected gravity after the mount calibration only (the true attitude);
        velocity: (vx, vy, vtheta) the walk policy is fed; active: the walk policy runs;
        agent: its file name in src/agents; mirrored: it runs mirrored left/right.
        """
        if not active:
            self._active_since = None
            self._command = None
            self._last_s = None
            return 0.0, 0.0
        if self._active_since is None:
            self._active_since = now_s
        dt = 0.0 if self._last_s is None else min(max(now_s - self._last_s, 0.0), 0.1)
        self._last_s = now_s
        self._learn(now_s, dt, gravity, velocity, agent, mirrored)
        fade = min(1.0, (now_s - self._active_since) / IMU_TRIM_FADE_S)
        return fade * self._trim.trim[0], fade * self._trim.trim[1]

    def _learn(self, now_s: float, dt: float, gravity: list[float], velocity, agent: str, mirrored: bool) -> None:
        key = (agent, mirrored)
        if key != self._key:
            self._key = key
            self._trim = self._trims.setdefault(key, _Trim())
            self._ref = self._refs.get(agent)
            self._command = None
            self._windows = {}
            if self._ref is None and agent not in self._missing:
                self._missing.add(agent)
                print(f"IMU auto trim: no simulated posture for {agent}, not trimmed", end="\r\n", flush=True)
        if self._ref is None:
            return
        v = np.asarray(velocity, dtype=float)
        if (len(gravity) != 3 or gravity[2] > UPRIGHT_GZ or abs(v[0]) < MIN_VX or abs(v[1]) > MAX_VY
                or abs(v[2]) > MAX_VTHETA):
            self._command = None
            return
        if self._command is None or np.max(np.abs(v - self._command)) > STEADY_TOL:
            self._command = v
            self._steady_since = now_s
            return
        if now_s - self._steady_since < IMU_TRIM_STEADY_S:
            return
        gx, gy = gravity[0], gravity[1]
        s = self._windows.setdefault(int(round(v[0] / VX_BIN)), [0.0] * 6)
        s[0] += gx
        s[1] += gy
        s[2] += gx * gx
        s[3] += gy * gy
        s[4] += 1
        s[5] += dt
        if s[5] >= IMU_TRIM_WINDOW_S:
            n = s[4]
            mean_x, mean_y = s[0] / n, s[1] / n
            sway = math.degrees(math.sqrt(max(0.0, s[2] / n - mean_x * mean_x) + max(0.0, s[3] / n - mean_y * mean_y)))
            s[:] = [0.0] * 6
            if sway <= IMU_TRIM_MAX_STD_DEG:
                self._step(lean_deg((mean_x, mean_y)), float(v[0]), mirrored)

    def _step(self, lean: tuple[float, float], vx: float, mirrored: bool) -> None:
        ref, trim = self._ref, self._trim
        # One roll for every speed: the simulated rolls scatter by +-0.8 deg from one speed to the
        # next, and per-speed targets fight each other on a walk back and forth.
        ref_right = float(np.median(ref["lean_right"]))
        if mirrored:
            ref_right = -ref_right
        base = trim.baselines.setdefault(int(round(vx / VX_BIN)), [0.0, 0])
        if base[1] < IMU_TRIM_BASELINE_WINDOWS:
            base[0] += lean[1]
            base[1] += 1
            error_fwd = 0.0
        else:
            error_fwd = lean[1] - base[0] / base[1]
        errors = (lean[0] - ref_right, error_fwd)
        # Leaning further right than the target: show the policy more right lean (+roll) so it
        # rights to the left; further forward: show it more forward lean (-pitch).
        signs = (1.0, -1.0)
        for i, (error, gain) in enumerate(zip(errors, (ref["gain_roll"], ref["gain_pitch"]))):
            if trim.off[i]:
                continue
            step = signs[i] * min(max(IMU_TRIM_GAIN * error / self._gain(gain), -IMU_TRIM_MAX_STEP_DEG),
                                  IMU_TRIM_MAX_STEP_DEG)
            wanted = trim.trim[i] + step
            trim.trim[i] = min(max(wanted, -LIMITS[i]), LIMITS[i])
            trim.pinned[i] = trim.pinned[i] + 1 if abs(wanted) > LIMITS[i] else 0
            if trim.pinned[i] >= IMU_TRIM_GIVE_UP_WINDOWS:
                trim.trim[i], trim.off[i] = 0.0, True
                print(f"IMU auto trim: {AXES[i]} pinned at {LIMITS[i]:.1f} deg and the robot does not follow: "
                      f"{AXES[i]} trim reset and off for this policy (lean right {lean[0]:+.1f}, fwd {lean[1]:+.1f} deg)",
                      end="\r\n", flush=True)
        if max(abs(trim.trim[0] - trim.printed[0]), abs(trim.trim[1] - trim.printed[1])) >= PRINT_STEP_DEG:
            trim.printed = (trim.trim[0], trim.trim[1])
            print(f"IMU auto trim: roll {trim.trim[0]:+.1f} deg, pitch {trim.trim[1]:+.1f} deg "
                  f"(vx {vx:+.2f}: lean right {lean[0]:+.1f} against {ref_right:+.1f}, "
                  f"pitch drift {error_fwd:+.1f} deg)", end="\r\n", flush=True)

    @staticmethod
    def _gain(gain: float) -> float:
        return min(max(gain, GAIN_RANGE[0]), GAIN_RANGE[1])
