# SPDX-License-Identifier: GPL-3.0-or-later

import math

import numpy as np

from constants import (
    CHILL_CLIPS,
    CHILL_MAX_GY,
    KP_GROUND,
    MOTOR_TO_ID,
    OBSERVATION_DOF_ORDER,
    PROPUP_AGENT,
    PROPUP_DOWN,
    PROPUP_HOLD_S,
    PROPUP_MIN_SUPINE_GX,
    PROPUP_UP_S,
)
from controller import ControllerProtocol
from moves.dance import DanceMove, _rot_matrix
from moves.move import MotorCommand
from observer import Observation


class PropUpMove(DanceMove):
    """Prop up on the elbows and hands from lying on the back, and lie back down, with the
    tracking policy of Mjlab-PropUp-Xiaoxiaobai. Its reference lies, props up, holds, lies down and
    lies again, its last frame being its first.

    Up plays the reference to PROPUP_HOLD_S and holds that frame; down (PROPUP_DOWN in
    active_moves) plays on to the last frame and holds it, lying; up again starts over from
    frame 1. The policy only learned the reference forwards: a request made while the motion
    runs the other way waits until it is through (lying down, then up again).

    Unlike the dances it starts and stays on its back: no fall check, KP_GROUND as in training,
    and the reference is aligned on the heading of the trunk's long axis, as the yaw angle is
    undefined lying on the back. It starts lying on its back, or sitting propped up at the hold
    frame; otherwise it holds the measured pose and status() reports refused.
    """

    label = "PropUp"
    kp = KP_GROUND

    def __init__(self, controller: ControllerProtocol | None = None, agent_path: str | None = None) -> None:
        super().__init__(controller=controller, agent_path=agent_path or f"src/agents/{PROPUP_AGENT}")
        self._hold_frame = min(self.total_frames - 1, round(PROPUP_HOLD_S * self.fps))
        self._up_frame = round(PROPUP_UP_S * self.fps)
        self._last = self.total_frames - 1
        self._stop = self._hold_frame
        self.refused = False

    def _write_kp(self) -> None:
        if self._controller is None:
            return
        ids = list(MOTOR_TO_ID.values())
        self._controller.sync_write_kp(ids, [KP_GROUND] * len(ids))

    def on_start(self, obs: Observation, command: MotorCommand) -> None:
        super().on_start(obs, command)
        self._stop = self._hold_frame
        gx, gy = obs.robot_state.projected_gravity[:2]
        _clip, lo, hi = CHILL_CLIPS["chill_sit"]
        if lo < gx < hi and abs(gy) < CHILL_MAX_GY:
            # Sitting propped up (from 坐地休闲): take over at the hold frame, the policy having
            # learned from every frame of the reference.
            self._frame = self._hold_frame
            self._t0 = self.clock() - (self._hold_frame - 1) / self.fps
        self.refused = bool(gx > -PROPUP_MIN_SUPINE_GX and self._frame != self._hold_frame)
        if self.refused:
            print("PropUp: not lying on its back, not starting", end="\r\n", flush=True)
            command.target_angles.update(
                {name: obs.robot_state.motor_positions[name] for name in OBSERVATION_DOF_ORDER})

    def status(self) -> dict:
        if self.refused:
            phase = "refused"
        elif self._frame > self._hold_frame:
            phase = "lying" if self._frame == self._stop else "lowering"
        else:
            phase = "up" if self._frame >= self._up_frame else "rising"
        return {
            "phase": phase,
            "t_s": round(self._frame / self.fps, 2),
            "duration_s": round(self.total_frames / self.fps, 2),
            "refused": self.refused,
        }

    def step(self, obs: Observation, command: MotorCommand) -> None:
        if self.refused:
            return
        down = PROPUP_DOWN in obs.user_input.active_moves
        if self._frame == self._last and not down:
            self._frame = 1
            self._t0 = self.clock()
            self._align = None
        self._stop = self._last if down or self._frame > self._hold_frame else self._hold_frame

        self._act(obs, command)
        if self._frame < self._stop:
            self._advance(self._stop)
        else:
            # Holding: the clock restarts from this frame when the motion goes on.
            self._t0 = self.clock() - (self._frame - 1) / self.fps

    def _heading(self, quat: np.ndarray) -> float:
        # The horizontal direction of the trunk's long axis (z), defined lying and propped up.
        r = _rot_matrix(np.asarray(quat, dtype=np.float64))
        return math.atan2(r[1, 2], r[0, 2])
