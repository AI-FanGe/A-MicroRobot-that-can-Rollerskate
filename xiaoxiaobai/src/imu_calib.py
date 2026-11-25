# SPDX-License-Identifier: GPL-3.0-or-later

"""IMU mount calibration: the roll / pitch of the IMU on the trunk.

The robot stands still at NEUTRAL_POSE on flat ground (STAND). Its trunk attitude then follows
from the leg kinematics (both soles flat): agents/posture_ref.json "neutral", with how it moves
per radian of each leg joint, for the little the joints sag off the pose. The calibration
averages the IMU's projected gravity over IMU_CALIB_S and returns the (roll, pitch) mount
correction (imu_reader.apply_mount_correction) that makes the IMU read that attitude: the
corrected IMU reads the true trunk attitude, which the automatic trim (tilt_trim.py) compares
with the simulated walk's.

Requested by the robot API (api/motion.py imu_calibrate) through UserInput.imu_calib_seq, the
wall-clock time of the request: a new, fresh value starts one calibration.
"""

from __future__ import annotations

import math
import time

from constants import IMU_CALIB_MAX_DEG, IMU_CALIB_MAX_GYRO, IMU_CALIB_MAX_JOINT_DEG, IMU_CALIB_S, NEUTRAL_POSE
from tilt_trim import lean_deg, load_posture_ref

REQUEST_MAX_AGE_S = 10.0  # an older request (left in the input file) is ignored


class ImuCalib:
    def __init__(self, ref: dict | None = None) -> None:
        self._neutral: dict | None = (load_posture_ref() if ref is None else ref).get("neutral")
        self._seq: float | None = None
        self._start_s: float | None = None
        self._gravity = [0.0, 0.0, 0.0]
        self._joints: dict[str, float] = {}
        self._count = 0
        self.result: dict | None = None  # the last calibration's (status "imu_calib")

    @property
    def running(self) -> bool:
        return self._start_s is not None

    def update(self, now_s: float, seq: float, gravity: list[float], gyro: list[float],
               positions: dict[str, float], standing: bool) -> None:
        """One control tick. gravity: the projected gravity before any correction; standing:
        no move but the head runs (the robot holds NEUTRAL_POSE)."""
        if seq != self._seq:
            self._seq = seq
            if seq and abs(time.time() - seq) < REQUEST_MAX_AGE_S:
                self._begin(now_s)
        if self._start_s is None:
            return
        if self._neutral is None:
            return self._finish(False, "no_ref")
        if not standing:
            return self._finish(False, "not_standing")
        if gyro and math.sqrt(sum(g * g for g in gyro)) > IMU_CALIB_MAX_GYRO:
            return self._finish(False, "moving")
        off = [name for name in self._neutral["jac_fwd"]
               if name in positions and abs(math.degrees(positions[name] - NEUTRAL_POSE[name])) > IMU_CALIB_MAX_JOINT_DEG]
        if off:
            return self._finish(False, "pose", joints=off)
        if len(gravity) != 3:
            return
        for i in range(3):
            self._gravity[i] += gravity[i]
        for name in self._neutral["jac_fwd"]:
            if name in positions:
                self._joints[name] = self._joints.get(name, 0.0) + positions[name]
        self._count += 1
        if now_s - self._start_s >= IMU_CALIB_S:
            self._solve()

    def _begin(self, now_s: float) -> None:
        print("IMU mount calibration: hold still", end="\r\n", flush=True)
        self._start_s = now_s
        self._gravity = [0.0, 0.0, 0.0]
        self._joints = {}
        self._count = 0

    def _solve(self) -> None:
        raw_right, raw_fwd = lean_deg([g / self._count for g in self._gravity])
        neutral = self._neutral
        ref_right, ref_fwd = neutral["lean_right"], neutral["lean_fwd"]
        for name, total in self._joints.items():
            sag = total / self._count - NEUTRAL_POSE[name]
            ref_right += neutral["jac_right"][name] * sag
            ref_fwd += neutral["jac_fwd"][name] * sag
        # apply_mount_correction: +roll reads further right, +pitch further backward.
        roll, pitch = ref_right - raw_right, raw_fwd - ref_fwd
        ok = max(abs(roll), abs(pitch)) <= IMU_CALIB_MAX_DEG
        self._finish(ok, "" if ok else "range", roll_deg=round(roll, 2), pitch_deg=round(pitch, 2),
                     lean_right=round(raw_right, 2), lean_fwd=round(raw_fwd, 2),
                     ref_right=round(ref_right, 2), ref_fwd=round(ref_fwd, 2))

    def _finish(self, ok: bool, error: str, **values) -> None:
        self._start_s = None
        self.result = {"seq": self._seq, "ok": ok, "error": error, **values}
        print(f"IMU mount calibration {'done' if ok else 'failed (' + error + ')'}: {values}", end="\r\n", flush=True)
