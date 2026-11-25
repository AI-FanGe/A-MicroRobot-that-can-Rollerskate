# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

import math
import threading
import time
from dataclasses import dataclass
import numpy as np

from ahrs.common.orientation import acc2q
from bmi088 import BMI088
import bmi088.bmi088 as _bmi_module
from smbus3 import SMBus
from constants import IMU_MOUNT_QUAT


BMI088_ACC_ADDRESSES = (0x19, 0x18)
BMI088_GYRO_ADDRESSES = (0x68, 0x69)


def _i2c_present(bus: SMBus, addr: int) -> bool:
    for method in ("write_quick", "read_byte"):
        try:
            getattr(bus, method)(addr)
            return True
        except Exception:
            pass
    try:
        bus.read_byte_data(addr, 0x00)
        return True
    except Exception:
        return False


def _configure_bmi088_addresses(i2c_bus: int) -> None:
    """Pick the IMU address pair that is actually present on this controller."""
    bus = SMBus(i2c_bus)
    try:
        acc = next((addr for addr in BMI088_ACC_ADDRESSES if _i2c_present(bus, addr)), None)
        gyro = next((addr for addr in BMI088_GYRO_ADDRESSES if _i2c_present(bus, addr)), None)
    finally:
        bus.close()

    if acc is not None:
        _bmi_module.ACC_ADDRESS = acc
    if gyro is not None:
        _bmi_module.GYRO_ADDRESS = gyro


@dataclass(frozen=True)
class IMUSnapshot:
    timestamp_s: float
    quat: tuple[float, float, float, float]
    gyro: tuple[float, float, float]
    acc: tuple[float, float, float]
    valid: bool
    error_count: int


def imu_quat_to_body(
    q: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Convert a quaternion measured in IMU frame to the trunk (body) frame.

    Applies q_body = q_imu * conjugate(IMU_MOUNT_QUAT).
    """
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = IMU_MOUNT_QUAT[0], -IMU_MOUNT_QUAT[1], -IMU_MOUNT_QUAT[2], -IMU_MOUNT_QUAT[3]
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )

def apply_mount_correction(gravity: list[float], roll_deg: float, pitch_deg: float) -> list[float]:
    """Rotate a body-frame gravity vector so the trunk reads `roll_deg` further to the right
    and `pitch_deg` further backward."""
    gx, gy, gz = gravity
    if roll_deg != 0.0:
        c, s = math.cos(math.radians(roll_deg)), math.sin(math.radians(roll_deg))
        gy, gz = gy * c + gz * s, gz * c - gy * s
    if pitch_deg != 0.0:
        c, s = math.cos(math.radians(pitch_deg)), math.sin(math.radians(pitch_deg))
        gx, gz = gx * c + gz * s, gz * c - gx * s
    return [gx, gy, gz]


def quat_apply_inverse(quat: list[float], vec: list[float]) -> list[float]:
    """Apply an inverse quaternion rotation to a vector.

    Args:
        quat: The quaternion in (w, x, y, z) format.
        vec: The vector in (x, y, z) format.

    Returns:
        The rotated vector in (x, y, z) format.
    """
    xyz = quat[1:]
    w = quat[0]
    t = 2 * np.cross(xyz, vec)
    return vec - w * t + np.cross(xyz, t)

def quat_yaw(quat: list[float]) -> float:
    """Heading [rad] of the x axis around the world z axis (counter-clockwise positive) for a
    (w, x, y, z) quaternion."""
    w, x, y, z = quat
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))

class ThreadedIMUReader:
    """Reads the IMU on a dedicated thread and exposes the latest sample snapshot."""

    def __init__(self, i2c_bus: int, frequency_hz: float = 200.0, warn_interval_s: float = 1.0) -> None:
        if frequency_hz <= 0:
            raise ValueError("frequency_hz must be > 0")

        _configure_bmi088_addresses(i2c_bus)
        self._imu = BMI088(i2c_bus=i2c_bus)
        self._period_s = 1.0 / frequency_hz
        self._warn_interval_s = warn_interval_s

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run_loop, name="imu-reader", daemon=True)

        now = time.perf_counter()
        self._snapshot = IMUSnapshot(
            timestamp_s=now,
            quat=(1.0, 0.0, 0.0, 0.0),
            gyro=(0.0, 0.0, 0.0),
            acc=(0.0, 0.0, 0.0),
            valid=False,
            error_count=0,
        )

        self._error_count = 0
        self._last_warn_s = 0.0

    def start(self) -> None:
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self, timeout_s: float = 1.0) -> None:
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=timeout_s)

    def get_latest(self) -> IMUSnapshot:
        with self._lock:
            return self._snapshot

    def get_status(self) -> dict[str, float | int | bool]:
        snap = self.get_latest()
        now = time.perf_counter()
        return {
            "valid": snap.valid,
            "age_s": max(0.0, now - snap.timestamp_s),
            "error_count": snap.error_count,
            "target_frequency_hz": 1.0 / self._period_s,
        }

    def _bootstrap_orientation(self, n_samples: int = 50) -> None:
        """Warm-start the Madgwick filter from a brief static accelerometer average.

        Reads *n_samples* at the normal loop rate, averages them, and uses
        acc2q to compute an initial quaternion (roll/pitch from gravity,yaw=0).
        """
        acc_sum = np.zeros(3)
        count = 0
        for _ in range(n_samples):
            try:
                ax, ay, az = self._imu.read_accelerometer()
                acc_sum += np.array([float(ax), float(ay), float(az)])
                count += 1
            except Exception:
                pass
            time.sleep(self._period_s)
        if count > 0:
            self._imu.q = list(acc2q(acc_sum / count))

    def _run_loop(self) -> None:
        self._bootstrap_orientation()
        next_tick = time.perf_counter()
        last_tick = next_tick

        while not self._stop_event.is_set():
            now = time.perf_counter()
            dt = max(1e-4, now - last_tick)
            last_tick = now

            try:
                w, x, y, z = self._imu.get_quat(dt)
                gx, gy, gz = self._imu.read_gyroscope()
                ax, ay, az = self._imu.read_accelerometer()
                with self._lock:
                    self._snapshot = IMUSnapshot(
                        timestamp_s=now,
                        quat=(float(w), float(x), float(y), float(z)),
                        gyro=(float(gx), float(gy), float(gz)),
                        acc=(float(ax), float(ay), float(az)),
                        valid=True,
                        error_count=self._error_count,
                    )
            except Exception as exc:
                self._error_count += 1
                if (now - self._last_warn_s) >= self._warn_interval_s:
                    print(f"Warning: IMU read failed ({self._error_count}): {exc}", end="\r\n", flush=True)
                    self._last_warn_s = now
                with self._lock:
                    prev = self._snapshot
                    self._snapshot = IMUSnapshot(
                        timestamp_s=prev.timestamp_s,
                        quat=prev.quat,
                        gyro=prev.gyro,
                        acc=prev.acc,
                        valid=prev.valid,
                        error_count=self._error_count,
                    )

            next_tick += self._period_s
            sleep_s = next_tick - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                # Reset cadence anchor when late to avoid accumulating drift.
                next_tick = time.perf_counter()