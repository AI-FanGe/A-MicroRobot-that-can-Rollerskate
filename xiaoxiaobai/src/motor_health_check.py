#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

"""Left/right leg motor health check on the real robot.

Hang the robot so both feet are off the ground, then from the PC:

    make motor-check HOST=user@<robot-ip>

Phase 1 reads temperature / hardware error status, ramps to NEUTRAL_POSE with
KP_RL and reports the holding error and current of every motor.
Phase 2 steps every leg joint by +-step_deg (left and right mirrored at the same
time) and compares the response of each left/right pair. A healthy pair should
have nearly identical rise time, steady-state error and peak current.
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

from constants import KP_RL, MOTOR_TO_ID, NEUTRAL_POSE
from robot_controller import RobotController

LEG_JOINTS = ["hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch", "ankle_roll"]
HW_ERROR_BITS = {0: "input_voltage", 2: "overheating", 3: "encoder", 4: "electrical_shock", 5: "overload"}


def _flat(values) -> list[float]:
    return [float(v[0]) if isinstance(v, (list, tuple)) else float(v) for v in values]


def _decode_error(code: int) -> str:
    names = [name for bit, name in HW_ERROR_BITS.items() if code & (1 << bit)]
    return ",".join(names) if names else "ok"


def ramp(controller: RobotController, ids: list[int], target: np.ndarray, duration_s: float) -> None:
    start = np.array(controller.sync_read_present_position(ids))
    t0 = time.perf_counter()
    while (elapsed := time.perf_counter() - t0) < duration_s:
        alpha = min(elapsed / duration_s, 1.0)
        controller.sync_write_goal_position(ids, (start + alpha * (target - start)).tolist())
        time.sleep(0.01)
    controller.sync_write_goal_position(ids, target.tolist())


def record(controller: RobotController, ids: list[int], duration_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ts, qs, cs = [], [], []
    t0 = time.perf_counter()
    while (elapsed := time.perf_counter() - t0) < duration_s:
        ts.append(elapsed)
        qs.append(controller.sync_read_present_position(ids))
        cs.append(controller.sync_read_present_current(ids))
    return np.array(ts), np.array(qs), np.array(cs)


def step_metrics(t: np.ndarray, q: np.ndarray, current: np.ndarray, q0: float, q1: float) -> dict[str, float]:
    """`moved_deg` is the final displacement in the step direction: with KP_RL the servo has
    a 2-3 deg dead band, so even a healthy motor never reaches 90 % of a 10 deg step."""
    delta = q1 - q0
    progress = (q - q0) / delta
    reached = np.nonzero(progress >= 0.5)[0]
    tail = t >= t[-1] - 0.2
    return {
        "t50_ms": float(t[reached[0]] * 1000) if reached.size else float("nan"),
        "overshoot_deg": float(max(0.0, math.degrees((progress.max() - 1.0) * abs(delta)))),
        "moved_deg": float(math.degrees(np.mean(q[tail]) - q0) * np.sign(delta)),
        "peak_a": float(np.abs(current).max()),
    }


def print_step_header() -> None:
    print(f"{'joint':11s} {'dir':>3s} | {'t50_ms L/R':>13s} {'moved_deg L/R':>15s} {'overshoot L/R':>15s} {'peak_A L/R':>13s}")


def print_step_row(joint: str, sign: int, ml: dict, mr: dict) -> None:
    print(
        f"{joint:11s} {'+' if sign > 0 else '-':>3s} | "
        f"{ml['t50_ms']:6.0f}/{mr['t50_ms']:<6.0f} "
        f"{ml['moved_deg']:7.2f}/{mr['moved_deg']:<7.2f} "
        f"{ml['overshoot_deg']:7.2f}/{mr['overshoot_deg']:<7.2f} "
        f"{ml['peak_a']:6.2f}/{mr['peak_a']:<6.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step-deg", type=float, default=10.0)
    parser.add_argument("--hold-s", type=float, default=3.0)
    parser.add_argument("--no-step", action="store_true", help="only run phase 1 (e.g. robot standing on the floor)")
    args = parser.parse_args()

    controller = RobotController()
    names = list(MOTOR_TO_ID)
    ids = [MOTOR_TO_ID[n] for n in names]
    neutral = np.array([NEUTRAL_POSE[n] for n in names])
    raw = controller._controller

    try:
        temps = _flat(raw.sync_read_present_temperature(ids))
        errors = [int(v) for v in _flat(raw.sync_read_hardware_error_status(ids))]
        controller.sync_write_torque_enable(ids, [True] * len(ids))
        controller.sync_write_status_return_level(ids, [1] * len(ids))
        controller.sync_write_kp(ids, [KP_RL] * len(ids))
        ramp(controller, ids, neutral, 2.0)
        time.sleep(1.0)

        _, q_hold, c_hold = record(controller, ids, args.hold_s)
        pwm = [v - 65536 if v > 32767 else v for v in _flat(raw.sync_read_present_pwm(ids))]
        hold_err = np.degrees(q_hold.mean(axis=0) - neutral)
        hold_cur = np.abs(c_hold).mean(axis=0)

        print("\n=== Phase 1: hold NEUTRAL_POSE (KP=%d) ===" % KP_RL)
        print(f"{'joint':10s} | {'side':5s} {'id':>3s} {'temp':>5s} {'err':>14s} {'hold_err':>9s} {'|I|_A':>6s} {'pwm%':>6s}")
        for j in LEG_JOINTS:
            for side in ("left", "right"):
                k = names.index(f"{side}_{j}")
                print(
                    f"{j:10s} | {side:5s} {ids[k]:3d} {temps[k]:5.0f} {_decode_error(errors[k]):>14s} "
                    f"{hold_err[k]:+9.2f} {hold_cur[k]:6.3f} {pwm[k] * 0.113:+6.1f}"
                )

        if args.no_step:
            return

        print(f"\n=== Phase 2: mirrored +-{args.step_deg:.0f} deg steps (robot must hang freely, trunk upright) ===")
        print_step_header()
        step = math.radians(args.step_deg)
        for j in LEG_JOINTS:
            kl, kr = names.index(f"left_{j}"), names.index(f"right_{j}")
            for sign in (+1, -1):
                target = neutral.copy()
                target[kl] += sign * step
                target[kr] -= sign * step
                start = np.array(controller.sync_read_present_position(ids))
                controller.sync_write_goal_position(ids, target.tolist())
                t, q, c = record(controller, ids, 0.8)
                ml = step_metrics(t, q[:, kl], c[:, kl], start[kl], target[kl])
                mr = step_metrics(t, q[:, kr], c[:, kr], start[kr], target[kr])
                print_step_row(j, sign, ml, mr)
                ramp(controller, ids, neutral, 0.5)
                time.sleep(0.3)

        temps_after = _flat(raw.sync_read_present_temperature(ids))
        errors_after = [int(v) for v in _flat(raw.sync_read_hardware_error_status(ids))]
        bad = [f"{n}(id {i}): {_decode_error(e)}" for n, i, e in zip(names, ids, errors_after) if e]
        print("\nTemperature rise during test:",
              ", ".join(f"{n} {a - b:+.0f}C" for n, a, b in zip(names, temps_after, temps) if abs(a - b) >= 2) or "none")
        print("Hardware errors after test:", "; ".join(bad) or "none")
    finally:
        controller.sync_write_torque_enable(ids, [False] * len(ids))
        controller.shutdown()


if __name__ == "__main__":
    main()
