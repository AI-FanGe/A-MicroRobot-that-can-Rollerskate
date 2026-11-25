# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

import math

from rustypot import Xl330PyController
import numpy as np

from constants import ID_TO_MOTOR, MOTOR_SERIAL_PORT, MOTOR_TO_ID, MOTOR_SIGN, IMU_I2C_BUS, PRESENT_CURRENT_UNIT_A
from imu_reader import ThreadedIMUReader
from neutral_offset import NEUTRAL_OFFSET_PATH, load_neutral_offsets


class RobotController:
    """Wraps Xl330PyController."""

    def __init__(self, serial_port: str = MOTOR_SERIAL_PORT, baudrate: int = 1_000_000, timeout: float = 0.1) -> None:
        self._controller = Xl330PyController(serial_port=serial_port, baudrate=baudrate, timeout=timeout)
        self._id_to_sign: dict[int, float] = {MOTOR_TO_ID[name]: MOTOR_SIGN[name] for name in MOTOR_TO_ID}
        self._offsets: dict[str, float] = load_neutral_offsets()
        self._print_neutral_offsets()
        self._imu_reader = ThreadedIMUReader(i2c_bus=IMU_I2C_BUS, frequency_hz=200.0)
        self._imu_reader.start()

    def _print_neutral_offsets(self) -> None:
        active = [
            f"{name} {math.degrees(value):+.2f} deg"
            for name, value in sorted(self._offsets.items())
            if abs(value) > 1e-6
        ]
        if active:
            print(f"Neutral offset ({NEUTRAL_OFFSET_PATH}): {', '.join(active)}", flush=True)
        else:
            print(f"Neutral offset: none ({NEUTRAL_OFFSET_PATH})", flush=True)

    def _to_hardware(self, motor_id: int, joint_position: float) -> float:
        name = ID_TO_MOTOR[motor_id]
        return (float(joint_position) + self._offsets.get(name, 0.0)) * self._id_to_sign[motor_id]

    def _from_hardware(self, motor_id: int, hardware_position: float) -> float:
        name = ID_TO_MOTOR[motor_id]
        return float(hardware_position) * self._id_to_sign[motor_id] - self._offsets.get(name, 0.0)

    def sync_write_torque_enable(self, ids: list[int], values: list[bool]) -> None:
        self._controller.sync_write_torque_enable(ids, values)

    def sync_write_status_return_level(self, ids: list[int], levels: list[int]) -> None:
        self._controller.sync_write_status_return_level(ids, levels)

    def sync_write_goal_position(self, ids: list[int], positions: list[float]) -> None:
        hw_positions = [self._to_hardware(motor_id, pos) for motor_id, pos in zip(ids, positions)]
        self._controller.sync_write_goal_position(ids, hw_positions)

    def sync_read_present_position(self, ids: list[int]) -> list[float]:
        raw = self._controller.sync_read_present_position(ids)
        return [self._from_hardware(motor_id, r) for motor_id, r in zip(ids, raw)]

    def read_present_position(self, motor_id: int) -> float:
        raw = self._controller.read_present_position(motor_id)
        return self._from_hardware(motor_id, raw)

    def sync_read_present_velocity(self, ids: list[int]) -> list[float]:
        raw = np.array(self._controller.sync_read_present_velocity(ids)) * 0.229 * np.pi / 30
        return [r * self._id_to_sign[motor_id] for motor_id, r in zip(ids, raw)]
    
    def read_present_velocity(self, motor_id: int) -> float:
        raw = self._controller.read_present_velocity(motor_id) * 0.229 * np.pi / 30
        return raw * self._id_to_sign[motor_id]

    def sync_read_present_current(self, ids: list[int]) -> list[float]:
        """Present current per motor, in Amps (signed). Magnitude is what matters for the BMS budget."""
        raw = self._controller.sync_read_present_current(ids)
        return [
            (r[0] if isinstance(r, (list, tuple)) else float(r)) * PRESENT_CURRENT_UNIT_A
            for r in raw
        ]

    def sync_read_present_input_voltage(self, ids: list[int]) -> list[float]:
        raw = self._controller.sync_read_present_input_voltage(ids)
        return [r[0] if isinstance(r, (list, tuple)) else float(r) for r in raw]

    def read_present_input_voltage(self, motor_id: int) -> float:
        raw = self._controller.read_present_input_voltage(motor_id)
        return raw[0] if isinstance(raw, (list, tuple)) else float(raw)

    def sync_read_kp(self, ids: list[int]) -> list[int]:
        return [int(v) for v in self._controller.sync_read_position_p_gain(ids)]

    def sync_write_kp(self, ids: list[int], gains: list[int]) -> None:
        self._controller.sync_write_position_p_gain(ids, gains)

    def read_acc(self) -> tuple[float, float, float]:
        """Return raw accelerometer (ax, ay, az) in g."""
        return self._imu_reader.get_latest().acc

    def read_gyro(self) -> tuple[float, float, float]:
        """Return (gx, gy, gz) in rad/s."""
        return self._imu_reader.get_latest().gyro

    def read_quat(self, dt: float) -> tuple[float, float, float, float]:
        """Return orientation quaternion (w, x, y, z)."""
        _ = dt
        return self._imu_reader.get_latest().quat

    def get_imu_status(self) -> dict[str, float | int | bool]:
        return self._imu_reader.get_status()

    def shutdown(self) -> None:
        self._imu_reader.stop()

    def close(self) -> None:
        self.shutdown()