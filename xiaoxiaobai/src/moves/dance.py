# SPDX-License-Identifier: GPL-3.0-or-later

import math
import os
import time

import numpy as np

from constants import DANCE_HEAD_LEAD_FRAMES, OBSERVATION_DOF_ORDER
from controller import ControllerProtocol
from imu_reader import quat_yaw
from moves.move import MotorCommand
from moves.walk import WalkMove
from observer import Observation


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _quat_axis(axis: int, angle: float) -> np.ndarray:
    q = np.zeros(4)
    q[0] = math.cos(angle / 2)
    q[1 + axis] = math.sin(angle / 2)
    return q


def _rot_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def mount_corrected_quat(body_quat: list[float], roll_deg: float, pitch_deg: float) -> np.ndarray:
    """Trunk orientation whose projected gravity is imu_reader.apply_mount_correction's."""
    q_c = _quat_mul(_quat_axis(0, math.radians(roll_deg)), _quat_axis(1, -math.radians(pitch_deg)))
    return _quat_mul(np.asarray(body_quat, dtype=np.float64), q_c)


class DanceMove(WalkMove):
    """Dance by tracking a reference motion with a RL policy trained in simulation.

    The policy sees the walk's observation without the velocity command, followed by the
    reference joint positions (relative to the reference pose) and velocities of the frame, and
    the reference trunk orientation in the trunk frame (first two columns of its rotation
    matrix). The reference yaw is aligned on the trunk's at the start. Once the last frame is
    reached it is held and status() reports done; the API then hands over to the walk.

    A frame per control tick, plus one more on ticks that find the dance behind clock() (a late
    tick): it stays on the soundtrack without the jump a late tick would put in the reference.

    A reference that turns the head (head_pos, the gymnastics) plays it on the head motor, which
    the policy does not drive, DANCE_HEAD_LEAD_FRAMES ahead as in training.
    """

    label = "Dance"
    head_idle = False
    mirrorable = False

    def __init__(self, agent_path: str, controller: ControllerProtocol | None = None) -> None:
        super().__init__(controller=controller, agents=[("dance", os.path.basename(agent_path))])
        ref_path = os.path.splitext(self.agent_path)[0] + "_ref.npz"
        ref = np.load(ref_path)
        if list(ref["joint_names"]) != OBSERVATION_DOF_ORDER:
            raise ValueError(f"{ref_path} joints {list(ref['joint_names'])} != OBSERVATION_DOF_ORDER")
        default = np.array([self._default_pose[n] for n in OBSERVATION_DOF_ORDER])
        self._ref_pos = ref["joint_pos"].astype(np.float64) - default
        self._ref_vel = ref["joint_vel"].astype(np.float64)
        self._ref_quat = ref["anchor_quat"].astype(np.float64)
        head = ref["head_pos"].astype(np.float64) if "head_pos" in ref.files else None
        self._ref_head = head if head is not None and np.abs(head).max() > 1e-6 else None
        self.total_frames = len(self._ref_pos)
        self.fps = float(ref["fps"])
        self._frame = 1
        self.clock = time.monotonic
        self._t0 = 0.0  # clock() when frame 1 was shown
        self._align: np.ndarray | None = None  # yaw rotation from the reference to the robot world
        self.done = False
        self.fallen = False

    def on_start(self, obs: Observation, command: MotorCommand) -> None:
        super().on_start(obs, command)
        # In training the command advances once on reset: the first observation already shows
        # frame 1 while the robot stands in frame 0's pose.
        self._frame = 1
        self._t0 = self.clock()
        self._align = None
        self._last_action = [0.0] * len(OBSERVATION_DOF_ORDER)
        self.done = False
        self.fallen = False

    def status(self) -> dict:
        return {
            "t_s": round(self._frame / self.fps, 2),
            "duration_s": round(self.total_frames / self.fps, 2),
            "done": self.done,
            "fallen": self.fallen,
        }

    def _robot_quat(self, obs: Observation) -> np.ndarray:
        u = obs.user_input
        return mount_corrected_quat(
            obs.robot_state.body_quat,
            getattr(u, "imu_roll_corr_deg", 0.0) + getattr(u, "imu_mount_roll_deg", 0.0),
            getattr(u, "imu_pitch_corr_deg", 0.0) + getattr(u, "imu_mount_pitch_deg", 0.0))

    def step(self, obs: Observation, command: MotorCommand) -> None:
        if obs.robot_state.projected_gravity[2] > self._projected_gravity_z_threshold:
            self.fallen = self.done = True
            return

        self._act(obs, command)
        if self._frame < self.total_frames - 1:
            self._advance(self.total_frames - 1)
        else:
            self.done = True

    def _advance(self, last: int) -> None:
        due = 1 + round((self.clock() - self._t0) * self.fps)
        self._frame = min(last, self._frame + (2 if due > self._frame + 1 else 1))

    def _heading(self, quat: np.ndarray) -> float:
        """The trunk's heading the reference is aligned on."""
        return quat_yaw(list(quat))

    def _act(self, obs: Observation, command: MotorCommand) -> None:
        input_obs = self.build_observation(obs)
        ort_inputs = {self._ort_session.get_inputs()[0].name: [input_obs]}
        action = self._ort_session.run(None, ort_inputs)[0][0]
        self._last_action = action.tolist()
        for i, name in enumerate(OBSERVATION_DOF_ORDER):
            command.target_angles[name] = self._default_pose[name] + action[i] * self.action_scale
        if self._ref_head is not None:
            command.target_angles["head"] = float(
                self._ref_head[min(self._frame + DANCE_HEAD_LEAD_FRAMES, self.total_frames - 1)])

    def build_observation(self, obs: Observation) -> list[float]:
        state = obs.robot_state
        input_obs = list(state.gyro) + list(state.projected_gravity)
        input_obs += [state.motor_positions[n] - self._default_pose[n] for n in OBSERVATION_DOF_ORDER]
        input_obs += [state.motor_velocities[n] for n in OBSERVATION_DOF_ORDER]
        input_obs += list(self._last_action)
        input_obs += self._ref_pos[self._frame].tolist() + self._ref_vel[self._frame].tolist()
        robot_quat = self._robot_quat(obs)
        if self._align is None:
            self._align = _quat_axis(2, self._heading(robot_quat) - self._heading(self._ref_quat[self._frame - 1]))
        rel = _quat_mul(_quat_conj(robot_quat), _quat_mul(self._align, self._ref_quat[self._frame]))
        input_obs += _rot_matrix(rel)[:, :2].reshape(-1).tolist()
        return input_obs
