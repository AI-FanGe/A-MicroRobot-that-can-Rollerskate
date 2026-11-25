# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

import math

from constants import (SKATE_AGENT_DEFAULT, SKATE_AGENTS, SKATE_HEADING_HOLD, SKATE_HEADING_KI, SKATE_HEADING_KP,
                       SKATE_HEADING_MAX_ERROR, SKATE_HEADING_MAX_TRIM, SKATE_HEADING_MIN_VX,
                       SKATE_VTHETA_MAX, SKATE_VX_MAX)
from controller import ControllerProtocol
from imu_reader import quat_yaw
from moves.move import MotorCommand
from moves.walk import WalkMove
from observer import Observation


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class SkateMove(WalkMove):
    """Skate on the roller-skate shoes using a RL policy trained in simulation."""

    label = "Skate"
    head_idle = False
    agent_field = "skate_agent"

    def __init__(self, controller: ControllerProtocol | None = None, agent_path: str | None = None,
                 heading_hold: bool = SKATE_HEADING_HOLD) -> None:
        super().__init__(controller=controller, agent_path=agent_path, agents=SKATE_AGENTS)
        if not agent_path:
            # Start on the default so the first tick does not switch models.
            self._agent_index = SKATE_AGENT_DEFAULT
            self._ort_session = self._sessions[SKATE_AGENT_DEFAULT]
            self.agent_path = self._agent_paths[SKATE_AGENT_DEFAULT]
            self._default_pose = self._default_poses[SKATE_AGENT_DEFAULT]
        self.heading_hold = heading_hold
        self._user_vtheta = 0.0  # ramped user vtheta, before the heading correction
        self._heading_ref: float | None = None
        self._trim = 0.0  # integral term, kept across stops: the veer is the policy's own

    def on_start(self, obs: Observation, command: MotorCommand) -> None:
        super().on_start(obs, command)
        self._user_vtheta = 0.0
        self._heading_ref = None
        self._trim = 0.0

    def command_velocity(self, obs: Observation) -> tuple[float, float, float]:
        # Backward and sideways commands are outside the training ranges; backward means stop.
        v = obs.user_input.velocity
        vx = min(max(v["vx"], 0.0), SKATE_VX_MAX)
        vtheta = min(max(v["vtheta"], -SKATE_VTHETA_MAX), SKATE_VTHETA_MAX)
        return vx, 0.0, vtheta

    def _ramp_velocity(self, obs: Observation, dt: float) -> None:
        # The ramp runs on the user's vtheta; the policy gets it plus the heading correction.
        self._velocity[2] = self._user_vtheta
        super()._ramp_velocity(obs, dt)
        self._user_vtheta = float(self._velocity[2])
        self._velocity[2] = self._hold_heading(obs, dt, self._user_vtheta)

    def _hold_heading(self, obs: Observation, dt: float, vtheta: float) -> float:
        """vtheta for the policy: the user's vtheta plus a PI feedback on the IMU yaw against a
        heading reference that turns at the user's vtheta."""
        body_quat = obs.robot_state.body_quat
        if not self.heading_hold or len(body_quat) != 4 or self._velocity[0] < SKATE_HEADING_MIN_VX:
            self._heading_ref = None
            return vtheta
        yaw = quat_yaw(body_quat)
        if self._heading_ref is None:
            self._heading_ref = yaw
        error = _wrap(self._heading_ref + vtheta * dt - yaw)
        error = min(max(error, -SKATE_HEADING_MAX_ERROR), SKATE_HEADING_MAX_ERROR)
        self._heading_ref = yaw + error
        out = vtheta + SKATE_HEADING_KP * error + self._trim
        if abs(out) < SKATE_VTHETA_MAX or out * error < 0:
            self._trim += SKATE_HEADING_KI * error * dt
            self._trim = min(max(self._trim, -SKATE_HEADING_MAX_TRIM), SKATE_HEADING_MAX_TRIM)
        return min(max(out, -SKATE_VTHETA_MAX), SKATE_VTHETA_MAX)
