# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

import time
from collections import deque
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import mujoco.viewer
import numpy as np

if TYPE_CHECKING:
    from sim.mujoco_input import MuJoCoInputSource

from bam.model import load_model as bam_load_model
from bam.mujoco import MujocoController as BamController

from constants import MOTOR_TO_ID, ID_TO_MOTOR, NEUTRAL_POSE, KP_DEFAULT, KP_GAIN_PRM, BAM_VIN, BAM_VOLTAGE_DROP_GAIN, BAM_VIN_MIN, BAM_MAX_CURRENT


class _DelayBuffer:
    """Returns values delayed by n_steps ticks (0 = no delay)."""

    def __init__(self, initial, n_steps: int) -> None:
        size = max(1, n_steps + 1)
        self._buf: deque = deque([initial] * size, maxlen=size)

    def push_and_read(self, value):
        self._buf.appendleft(value)
        return self._buf[-1]

    def fill(self, value) -> None:
        for i in range(len(self._buf)):
            self._buf[i] = value


class MuJoCoController:
    """MuJoCo-backed controller."""

    def __init__(
        self,
        mjcf_path: str,
        key_callback: Callable[[int, int, int, int], None] | None = None,
        stop_flag_path: str = "/tmp/xiaoxiaobai_scheduler.stop",
        reset_source: "MuJoCoInputSource | None" = None,
        # Actuation delay (command → motor), in simulator steps (default timestep: 0.005 s)
        delay_act_steps: int = 0,
        # Sensor delays (motor/IMU → observation), in scheduler ticks (default: 0.02 s)
        delay_pos_ticks: int = 0,
        delay_vel_ticks: int = 0,
        delay_gyro_ticks: int = 0,
        delay_quat_ticks: int = 0,
        trunk_com_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
        initial_pose: Mapping[str, float] | None = None,
        root_z: float = 0.186,
        root_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        # Replace the XML position actuators with torque motors driven by the BAM servo model,
        # as mjlab does during training.
        use_bam: bool = False,
    ) -> None:
        self._stop_flag_path = Path(stop_flag_path)
        self._reset_source = reset_source
        if use_bam:
            spec = mujoco.MjSpec.from_file(mjcf_path)
            for actuator in spec.actuators:
                actuator.set_to_motor()
                actuator.inheritrange = 0.0
                actuator.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
            self._model = spec.compile()
        else:
            self._model = mujoco.MjModel.from_xml_path(mjcf_path)
        self._data = mujoco.MjData(self._model)
        self._initial_pose = dict(initial_pose) if initial_pose is not None else dict(NEUTRAL_POSE)
        self._root_z = root_z
        self._root_quat = tuple(float(v) for v in root_quat)

        self._name_to_actuator_idx: dict[str, int] = {}
        self._name_to_qpos_idx: dict[str, int] = {}
        self._name_to_qvel_idx: dict[str, int] = {}
        # Motors with neither joint nor actuator in the MJCF (head_pitch): they reach their
        # goal at once and carry no mass.
        self._virtual: dict[int, float] = {}
        for name, motor_id in MOTOR_TO_ID.items():
            actuator_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            joint_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if actuator_id < 0 and joint_id < 0:
                self._virtual[motor_id] = float(self._initial_pose.get(name, 0.0))
                continue
            if actuator_id < 0:
                raise ValueError(f"Actuator '{name}' not found in MJCF model {mjcf_path!r}")
            if joint_id < 0:
                raise ValueError(f"Joint '{name}' not found in MJCF model {mjcf_path!r}")
            self._name_to_actuator_idx[name] = actuator_id
            self._name_to_qpos_idx[name] = self._model.jnt_qposadr[joint_id]
            self._name_to_qvel_idx[name] = self._model.jnt_dofadr[joint_id]
        # Number of physics sub-steps per scheduler tick (scheduler runs at 50 Hz)
        self._steps_per_tick = max(1, round(0.02 / self._model.opt.timestep))
        self._torque_interval = 0.1
        self._last_torque_print = 0.0
        self._uses_xml_position_actuators = all(
            int(self._model.actuator_biastype[idx]) == int(mujoco.mjtBias.mjBIAS_AFFINE)
            for idx in self._name_to_actuator_idx.values()
        )

        # Apply CoM offset on trunk body (simulates inertial model error)
        if any(v != 0.0 for v in trunk_com_offset):
            trunk_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
            self._model.body_ipos[trunk_id, 0] += trunk_com_offset[0]
            self._model.body_ipos[trunk_id, 1] += trunk_com_offset[1]
            self._model.body_ipos[trunk_id, 2] += trunk_com_offset[2]

        self._apply_initial_state()

        self._sim_ids = [mid for mid in MOTOR_TO_ID.values() if mid not in self._virtual]
        sim_names = [ID_TO_MOTOR[mid] for mid in self._sim_ids]

        # Delay buffers — simulate sensor/communication latency
        self._delay_pos = {
            mid: _DelayBuffer(
                self._data.qpos[self._name_to_qpos_idx[ID_TO_MOTOR[mid]]],
                delay_pos_ticks,
            )
            for mid in self._sim_ids
        }
        self._delay_vel = {
            mid: _DelayBuffer(0.0, delay_vel_ticks)
            for mid in self._sim_ids
        }
        self._delay_gyro = _DelayBuffer((0.0, 0.0, 0.0), delay_gyro_ticks)
        self._delay_quat = _DelayBuffer((1.0, 0.0, 0.0, 0.0), delay_quat_ticks)
        self._delay_act = {
            mid: _DelayBuffer(
                self._data.qpos[self._name_to_qpos_idx[ID_TO_MOTOR[mid]]],
                delay_act_steps,
            )
            for mid in self._sim_ids
        }

        self._bam = None
        # Per-motor firmware kp, in MOTOR_TO_ID order (the BAM actuator order).
        self._bam_kp = np.full(len(self._sim_ids), float(KP_DEFAULT))
        self._bam_kp_index = {motor_id: i for i, motor_id in enumerate(self._sim_ids)}
        if not self._uses_xml_position_actuators:
            # BAM motor model — servo (DC motor + Stribeck + load-dependent friction)
            bam_model = bam_load_model(motor_name="xl330", model="m6")
            bam_model.actuator.kp = KP_DEFAULT
            bam_model.actuator.vin = BAM_VIN
            try:
                self._bam = BamController(
                    bam_model,
                    sim_names,
                    self._model,
                    self._data,
                    vin_drop_gain=BAM_VOLTAGE_DROP_GAIN,
                    vin_min=BAM_VIN_MIN,
                    max_current=BAM_MAX_CURRENT,
                )
            except TypeError:
                self._bam = BamController(
                    bam_model,
                    sim_names,
                    self._model,
                    self._data,
                    vin_drop_resistance=BAM_VOLTAGE_DROP_GAIN,
                    vin_min=BAM_VIN_MIN,
                )
            if hasattr(self._bam, "reset"):
                self._bam.reset(self._data.qpos)
            # BamController calls mj_setConst, which overwrites qpos with qpos0.
            self._apply_initial_state()

        self._viewer = mujoco.viewer.launch_passive(
            self._model, self._data, key_callback=key_callback
        )

        # Sensor indices for IMU readout
        self._sensor_orientation = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SENSOR, "orientation")
        self._sensor_gyro = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SENSOR, "angular-velocity")

    @property
    def viewer_opt(self) -> mujoco.MjvOption:
        return self._viewer.opt

    def set_kp(self, kp: float) -> None:
        if self._bam is not None:
            self._bam_kp[:] = kp
            self._bam.model.actuator.kp = self._bam_kp.copy()
            return
        for idx in self._name_to_actuator_idx.values():
            xml_kp = kp * KP_GAIN_PRM
            self._model.actuator_gainprm[idx, 0] = xml_kp
            self._model.actuator_biasprm[idx, 1] = -xml_kp

    def sync_read_kp(self, ids: list[int]) -> list[int]:
        if self._bam is not None:
            return [int(self._bam_kp[self._bam_kp_index[motor_id]]) if motor_id in self._bam_kp_index else KP_DEFAULT
                    for motor_id in ids]
        else:
            first_idx = next(iter(self._name_to_actuator_idx.values()))
            kp = int(round(self._model.actuator_gainprm[first_idx, 0] / KP_GAIN_PRM))
        return [kp] * len(ids)

    def sync_write_kp(self, ids: list[int], gains: list[int]) -> None:
        pairs = [(m, g) for m, g in zip(ids, gains) if m not in self._virtual]
        ids, gains = [m for m, _ in pairs], [g for _, g in pairs]
        if self._bam is not None:
            for motor_id, gain in zip(ids, gains):
                self._bam_kp[self._bam_kp_index[motor_id]] = gain
            self._bam.model.actuator.kp = self._bam_kp.copy()
            return
        for motor_id, gain in zip(ids, gains):
            name = ID_TO_MOTOR[motor_id]
            idx = self._name_to_actuator_idx[name]
            xml_kp = gain * KP_GAIN_PRM
            self._model.actuator_gainprm[idx, 0] = xml_kp
            self._model.actuator_biasprm[idx, 1] = -xml_kp

    def sync_write_torque_enable(self, ids: list[int], values: list[bool]) -> None:
        pass

    def sync_write_status_return_level(self, ids: list[int], levels: list[int]) -> None:
        pass

    def sync_write_goal_position(self, ids: list[int], positions: list[float]) -> None:
        if not self._viewer.is_running():
            self._stop_flag_path.write_text("stop\n", encoding="ascii")
            return

        cmd = dict(zip(ids, positions))
        for motor_id in self._virtual.keys() & cmd.keys():
            self._virtual[motor_id] = float(cmd.pop(motor_id))

        for _ in range(self._steps_per_tick):
            for motor_id, pos in cmd.items():
                delayed_pos = self._delay_act[motor_id].push_and_read(pos)
                name = ID_TO_MOTOR[motor_id]
                if self._bam is not None:
                    self._bam.set_q_target(name, delayed_pos)
                else:
                    self._data.ctrl[self._name_to_actuator_idx[name]] = delayed_pos
            if self._bam is not None:
                self._bam.update()
            mujoco.mj_step(self._model, self._data)

        if self._reset_source is not None and self._reset_source.consume_reset():
            self.reset()
            return

        if self._reset_source is not None and self._reset_source.show_torque:
            now = time.monotonic()
            if now - self._last_torque_print >= self._torque_interval:
                total = float(sum(abs(f) for f in self._data.actuator_force))
                print(f"Torque sum: {total:.3f} Nm")
                self._last_torque_print = now

        self._viewer.sync()

    def sync_read_present_position(self, ids: list[int]) -> list[float]:
        return [
            self._virtual[mid] if mid in self._virtual else self._delay_pos[mid].push_and_read(
                self._data.qpos[self._name_to_qpos_idx[ID_TO_MOTOR[mid]]]
            )
            for mid in ids
        ]

    def read_present_position(self, motor_id: int) -> float:
        if motor_id in self._virtual:
            return self._virtual[motor_id]
        name = ID_TO_MOTOR[motor_id]
        return float(self._delay_pos[motor_id].push_and_read(
            self._data.qpos[self._name_to_qpos_idx[name]]
        ))

    def sync_read_present_velocity(self, ids: list[int]) -> list[float]:
        return [
            0.0 if mid in self._virtual else self._delay_vel[mid].push_and_read(
                self._data.qvel[self._name_to_qvel_idx[ID_TO_MOTOR[mid]]]
            )
            for mid in ids
        ]

    def read_present_velocity(self, motor_id: int) -> float:
        if motor_id in self._virtual:
            return 0.0
        name = ID_TO_MOTOR[motor_id]
        return float(self._delay_vel[motor_id].push_and_read(
            self._data.qvel[self._name_to_qvel_idx[name]]
        ))

    def sync_read_present_current(self, ids: list[int]) -> list[float]:
        # Motor current from the torque applied by the bam model: I = torque / kt.
        # ctrl holds the (current-clipped) torque set in MujocoController.update().
        if self._bam is None:
            return [0.0] * len(ids)
        kt = self._bam.model.kt.value
        return [
            0.0 if mid in self._virtual else float(self._data.ctrl[self._name_to_actuator_idx[ID_TO_MOTOR[mid]]] / kt)
            for mid in ids
        ]

    def sync_read_present_input_voltage(self, ids: list[int]) -> list[float]:
        return [BAM_VIN * 10.0] * len(ids)

    def read_present_input_voltage(self, motor_id: int) -> float:
        return BAM_VIN * 10.0

    def read_acc(self) -> tuple[float, float, float]:
        """Return pseudo-accelerometer (ax, ay, az) in g from the 'orientation' sensor."""
        if self._sensor_orientation < 0:
            return 0.0, 0.0, -1.0
        adr = self._model.sensor_adr[self._sensor_orientation]
        w, x, y, z = self._data.sensordata[adr:adr + 4]
        # Gravity in world is (0, 0, -1) g; rotate into IMU frame using conjugate quat
        gx = 2 * (x * z - w * y)
        gy = 2 * (y * z + w * x)
        gz = w * w - x * x - y * y + z * z
        return float(gx), float(gy), float(-gz)

    def read_gyro(self) -> tuple[float, float, float]:
        if self._sensor_gyro < 0:
            current = (0.0, 0.0, 0.0)
        else:
            adr = self._model.sensor_adr[self._sensor_gyro]
            gx, gy, gz = self._data.sensordata[adr:adr + 3]
            current = (float(gx), float(gy), float(gz))
        return self._delay_gyro.push_and_read(current)

    def read_quat(self, dt: float) -> tuple[float, float, float, float]:
        if self._sensor_orientation < 0:
            current = (1.0, 0.0, 0.0, 0.0)
        else:
            adr = self._model.sensor_adr[self._sensor_orientation]
            w, x, y, z = self._data.sensordata[adr:adr + 4]
            current = (float(w), float(x), float(y), float(z))
        return self._delay_quat.push_and_read(current)

    def reset(self) -> None:
        """Reset the simulation to the initial neutral standing pose."""
        self._apply_initial_state()
        if self._bam is not None and hasattr(self._bam, "reset"):
            self._bam.reset(self._data.qpos)
        for mid in self._sim_ids:
            neutral = self._data.qpos[self._name_to_qpos_idx[ID_TO_MOTOR[mid]]]
            self._delay_act[mid].fill(neutral)
        self._viewer.sync()

    def _apply_initial_state(self) -> None:
        """Place the robot in the configured initial pose above the floor."""
        self._data.qpos[:] = 0.0
        self._data.qvel[:] = 0.0
        self._data.ctrl[:] = 0.0
        if self._model.nq >= 7:
            self._data.qpos[3:7] = self._root_quat
        self._data.qpos[2] = self._root_z
        for name, angle in self._initial_pose.items():
            if name in self._name_to_qpos_idx:
                self._data.qpos[self._name_to_qpos_idx[name]] = angle
            if self._uses_xml_position_actuators and name in self._name_to_actuator_idx:
                self._data.ctrl[self._name_to_actuator_idx[name]] = angle
        mujoco.mj_forward(self._model, self._data)