# SPDX-License-Identifier: GPL-3.0-or-later

import bisect
import time

import numpy as np

from constants import (
    CHILL_CLIPS,
    CHILL_EASE_DEG_S,
    CHILL_EASE_MIN_S,
    CHILL_KP,
    CHILL_MAX_GY,
    CHILL_SETTLE,
    KP_DEFAULT,
    KP_GROUND,
    MOTOR_TO_ID,
)
from controller import ControllerProtocol
from moves.move import MotorCommand, Move, MoveState
from observer import Observation


def _smoothstep(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


class ChillMove(Move):
    """Lie or sit and fidget: a chill_export.py clip of every motor's angle, played open loop at
    KP_GROUND (or its CHILL_KP) as in its sim check (mjlab_xiaoxiaobai scripts/chill_sim.py).

    It eases from the measured pose to the clip's first pose, plays the clip's ease-in once and
    then loops the rest; a clip whose loop starts at its end (chill_export.py --once) plays once
    and holds its last frame, status() reporting done. With CHILL_SETTLE in active_moves it
    eases back to the first pose and holds it, status() reporting settled: the API waits for
    that before it hands over to another mode (prop up takes over sitting from there). It only
    starts with the trunk in the clip's projected gravity range; otherwise it holds the measured
    pose and status() reports refused.
    """

    label = "Chill"
    kp = KP_GROUND

    def __init__(self, name: str, controller: ControllerProtocol | None = None) -> None:
        super().__init__()
        clip, self._gx_lo, self._gx_hi = CHILL_CLIPS[name]
        self._controller = controller
        self.agent_path = f"src/agents/{clip}"
        d = np.load(self.agent_path)
        self._names = [str(n) for n in d["joint_names"]] + ["head_pitch"]
        unknown = set(self._names) - set(MOTOR_TO_ID)
        if unknown:
            raise ValueError(f"{clip}: no motor for {sorted(unknown)}")
        self._q = np.concatenate([d["joint_pos"], d["head_pitch"][:, None]], axis=1).astype(np.float64)
        self.fps = float(d["fps"])
        self.total_frames = len(self._q)
        self._loop_start = int(d["loop_start"])
        self._once = self._loop_start >= self.total_frames
        self.kp = CHILL_KP.get(name, KP_GROUND)
        self._section_frames = [round(float(s) * self.fps) for s in d["section_starts"]]
        self._section_names = [str(s) for s in d["section_names"]]
        self.clock = time.monotonic
        self.refused = False
        self._frame = 0
        self._t0 = 0.0
        self._ease_s = CHILL_EASE_MIN_S
        self._from = self._q[0]
        self._last = self._q[0]
        self._settle_t0: float | None = None
        self._settled = False

    def _ease_time(self, a: np.ndarray, b: np.ndarray) -> float:
        return max(CHILL_EASE_MIN_S, float(np.degrees(np.abs(b - a).max())) / CHILL_EASE_DEG_S)

    def on_start(self, obs: Observation, command: MotorCommand) -> None:
        if self._controller is not None:
            ids = list(MOTOR_TO_ID.values())
            self._controller.sync_write_kp(ids, [self.kp] * len(ids))
        measured = obs.robot_state.motor_positions
        self._from = np.array([measured.get(n, q) for n, q in zip(self._names, self._q[0])])
        gx, gy = obs.robot_state.projected_gravity[:2]
        self.refused = not (self._gx_lo < gx < self._gx_hi and abs(gy) < CHILL_MAX_GY)
        if self.refused:
            print(f"Chill: trunk gravity ({gx:.2f}, {gy:.2f}) outside its start range, not starting",
                  end="\r\n", flush=True)
        self._restart(self._from)
        self.state = MoveState.ACTIVE
        # Act on the first tick too: the neutral pose at KP_GROUND would yank the limbs.
        self.step(obs, command)

    def _restart(self, start: np.ndarray) -> None:
        self._from = start
        self._ease_s = self._ease_time(start, self._q[0])
        self._t0 = self.clock()
        self._frame = 0
        self._settle_t0 = None
        self._settled = False

    def step(self, obs: Observation, command: MotorCommand) -> None:
        if self.refused:
            target = self._from
        elif CHILL_SETTLE in obs.user_input.active_moves:
            if self._settle_t0 is None:
                self._settle_t0 = self.clock()
                self._from = self._last
                self._ease_s = self._ease_time(self._last, self._q[0])
            w = _smoothstep((self.clock() - self._settle_t0) / self._ease_s)
            self._settled = w >= 1.0
            target = self._from + w * (self._q[0] - self._from)
        else:
            if self._settle_t0 is not None:  # settle called off: carry on from where it got to
                self._restart(self._last)
            t = self.clock() - self._t0
            if t < self._ease_s:
                target = self._from + _smoothstep(t / self._ease_s) * (self._q[0] - self._from)
            else:
                frame = int((t - self._ease_s) * self.fps)
                if frame >= self.total_frames:
                    frame = self.total_frames - 1 if self._once else \
                        self._loop_start + (frame - self._loop_start) % (self.total_frames - self._loop_start)
                self._frame = frame
                target = self._q[frame]
        self._last = target
        command.target_angles.update(zip(self._names, target.tolist()))

    def status(self) -> dict:
        if self.refused:
            phase = "refused"
        elif self._settle_t0 is not None:
            phase = "settled" if self._settled else "settling"
        elif self.clock() - self._t0 < self._ease_s:
            phase = "easing"
        elif self._once and self._frame >= self.total_frames - 1:
            phase = "done"
        else:
            phase = "playing"
        section = self._section_names[max(0, bisect.bisect_right(self._section_frames, self._frame) - 1)] \
            if self._section_names else None
        return {
            "phase": phase,
            "section": section,
            "t_s": round(self._frame / self.fps, 2),
            "loop_s": round((self.total_frames - self._loop_start) / self.fps, 2),
            "refused": self.refused,
        }

    def on_stop(self, obs: Observation, command: MotorCommand) -> None:
        # Hold the measured pose before raising KP, as WalkMove.on_stop does.
        hold = {n: obs.robot_state.motor_positions[n] for n in self._names if n in obs.robot_state.motor_positions}
        command.target_angles.update(hold)
        if self._controller is not None:
            self._controller.sync_write_goal_position([MOTOR_TO_ID[n] for n in hold], list(hold.values()))
            ids = list(MOTOR_TO_ID.values())
            self._controller.sync_write_kp(ids, [KP_DEFAULT] * len(ids))
        self.state = MoveState.INACTIVE
