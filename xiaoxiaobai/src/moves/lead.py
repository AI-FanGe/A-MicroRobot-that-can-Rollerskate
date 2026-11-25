# SPDX-License-Identifier: GPL-3.0-or-later

"""Lead by the hand: in 「走跑·上肢自由」 with the arms soft, the shoulders steer the walk.

Pull an arm forward (shoulder pitch, rp / lp) and the robot walks forward, push it back and it
steps back; lift an arm out sideways (shoulder roll, rr / lr) and it turns on the spot to that
side. Both at once walk an arc. Either arm leads (the one pulled furthest wins); lifting both
out cancels the turn. A hanging arm (pitch about 0, roll a few degrees) means stand still,
with a dead zone wide enough for the arm swinging with the steps. Let go, the arms ease back
to rest by themselves (moves/upper_body.py _spring_back), and the walk slows down to a stop.
"""

import math

from gestures import from_joints

LEAD_JOINTS = ("right_shoulder_pitch", "right_shoulder_roll", "left_shoulder_pitch", "left_shoulder_roll")
PITCH_DEAD_DEG = 15.0
PITCH_FULL_FORWARD_DEG = 50.0
PITCH_FULL_BACK_DEG = 40.0
ROLL_DEAD_DEG = 20.0
ROLL_FULL_DEG = 60.0
# At a full pull [m/s, rad/s], inside the walk policy's ranges (constants.VX_MAX, VTHETA_MAX_MOVING).
VX_FORWARD = 0.3
VX_BACK = 0.2
VTHETA = 0.6
TAU_S = 0.12  # low-pass of the measured angles (sensor noise, the arm bouncing with the steps)


def _pull(angle: float, dead: float, full: float) -> float:
    """0 inside the dead zone, rising to 1 at `full` degrees."""
    return min(1.0, max(0.0, (angle - dead) / (full - dead)))


class HandLead:
    def __init__(self) -> None:
        self._pose: dict[str, float] | None = None
        self.vx = 0.0
        self.vtheta = 0.0

    def reset(self) -> None:
        self._pose = None
        self.vx = self.vtheta = 0.0

    def update(self, joints: dict[str, float], dt: float) -> tuple[float, float]:
        """Measured joint angles [rad] -> (vx [m/s], vtheta [rad/s, + left])."""
        pose = from_joints(joints)
        if self._pose is None:
            self._pose = {key: pose[key] for key in ("rp", "lp", "rr", "lr")}
        else:
            k = 1.0 - math.exp(-dt / TAU_S)
            for key in self._pose:
                self._pose[key] += (pose[key] - self._pose[key]) * k
        p = self._pose
        pitch = max(p["rp"], p["lp"], key=abs)
        if pitch >= 0.0:
            self.vx = VX_FORWARD * _pull(pitch, PITCH_DEAD_DEG, PITCH_FULL_FORWARD_DEG)
        else:
            self.vx = -VX_BACK * _pull(-pitch, PITCH_DEAD_DEG, PITCH_FULL_BACK_DEG)
        turn = _pull(p["lr"], ROLL_DEAD_DEG, ROLL_FULL_DEG) - _pull(p["rr"], ROLL_DEAD_DEG, ROLL_FULL_DEG)
        self.vtheta = VTHETA * turn
        return self.vx, self.vtheta
