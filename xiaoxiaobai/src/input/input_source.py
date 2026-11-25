# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from constants import (
    IMU_AUTO_TRIM,
    IMU_MOUNT_PITCH_DEG,
    IMU_MOUNT_ROLL_DEG,
    IMU_PITCH_CORRECTION_DEG,
    IMU_ROLL_CORRECTION_DEG,
    KP_VIN_COMP,
    SKATE_AGENT_DEFAULT,
    VTHETA_MAX_MOVING,
    VTHETA_MAX_STATIONARY,
    VX_MAX,
    VX_MAX_BACKWARD,
    VY_MAX,
)


@dataclass
class UserInput:
    """Human or agent control state for one scheduler iteration."""

    active_moves: set[str] = field(default_factory=set)
    velocity: dict[str, float] = field(default_factory=lambda: {"vx": 0.0, "vy": 0.0, "vtheta": 0.0})
    show_imu: bool = False
    imu_roll_corr_deg: float = IMU_ROLL_CORRECTION_DEG
    imu_pitch_corr_deg: float = IMU_PITCH_CORRECTION_DEG
    # The IMU mount calibration (imu_calib.py), under the corrections above; imu_calib_seq: the
    # time of a calibration request (0: none).
    imu_mount_roll_deg: float = IMU_MOUNT_ROLL_DEG
    imu_mount_pitch_deg: float = IMU_MOUNT_PITCH_DEG
    imu_calib_seq: float = 0.0
    walk_agent: int = 0  # index into constants.WALK_AGENTS
    skate_agent: int = SKATE_AGENT_DEFAULT  # index into constants.SKATE_AGENTS
    # Run the walk / skate policies mirrored left/right (moves/walk.py), a diagnostic of their bias.
    mirror_policy: bool = False
    # Supply-voltage compensation of the walk's KP (moves/walk.py) and automatic IMU trim (tilt_trim.py).
    kp_vin_comp: bool = KP_VIN_COMP
    imu_auto_trim: bool = IMU_AUTO_TRIM
    # False when a remote snapshot only holds moves: the local velocity is kept.
    has_velocity: bool = True
    # True when a remote snapshot sets walk_agent, skate_agent, the IMU corrections and mount,
    # mirror_policy, kp_vin_comp, imu_auto_trim and imu_calib_seq (web UI).
    has_tuning: bool = False
    # Upper-body commands (moves/upper_body.py): {"seq", "gesture", "size", "manual", "auto"}.
    upper: dict | None = None
    # Where a camera-guided API source wants the walk's head [deg, + left] (moves/walk.py, with HEAD_FRONT).
    head_yaw: float | None = None


def scale_velocity(velocity: dict[str, float]) -> dict[str, float]:
    """Map a normalized velocity command in [-1, 1] per axis to physical limits.

    Applied centrally (in the scheduler) so the limits are identical for every input
    source — keyboard, gamepad, or agent. Forward and backward have different caps, and
    rotation gets a wider range when turning in place (vx = vy = 0) than while translating.
    """
    vx = max(-1.0, min(1.0, velocity.get("vx", 0.0)))
    vy = max(-1.0, min(1.0, velocity.get("vy", 0.0)))
    vtheta = max(-1.0, min(1.0, velocity.get("vtheta", 0.0)))

    moving = abs(vx) > 1e-6 or abs(vy) > 1e-6
    vtheta_max = VTHETA_MAX_MOVING if moving else VTHETA_MAX_STATIONARY
    vx_max = VX_MAX if vx >= 0.0 else VX_MAX_BACKWARD

    return {"vx": vx * vx_max, "vy": vy * VY_MAX, "vtheta": vtheta * vtheta_max}


class InputSource(ABC):
    """Abstract interface for human or agent input. Swap keyboard for gamepad without touching the rest."""

    def start(self) -> None:
        """Start the input source (e.g., launch a background thread)."""

    def stop(self) -> None:
        """Stop the input source and release resources."""

    @abstractmethod
    def read(self) -> UserInput:
        """Return the current input state. Must be non-blocking."""


EXCLUSIVE_MOVES = frozenset({"walk", "skate", "gymnastics", "peek", "propup",
                             "chill_lie", "chill_sit", "chill_chair", "wake_stretch"})
STICK_ACTIVE = 1e-3


class CompositeInputSource(InputSource):
    """Merge several input sources.

    A fresh remote command wins the velocity axes unless a local stick (gamepad, priority
    over the web app) is being moved. When the remote command picks walk or skate, the
    local exclusive move is dropped so the two policies never run together.
    """

    def __init__(self, sources: list[InputSource]) -> None:
        self.sources = sources

    def start(self) -> None:
        for source in self.sources:
            source.start()

    def stop(self) -> None:
        for source in self.sources:
            source.stop()

    def read(self) -> UserInput:
        local = UserInput()
        remote: UserInput | None = None
        for source in self.sources:
            read_fresh = getattr(source, "read_fresh", None)
            if callable(read_fresh):
                snapshot = read_fresh()
                if snapshot is not None:
                    remote = snapshot
                continue
            incoming = source.read()
            local.active_moves |= incoming.active_moves
            local.velocity = dict(incoming.velocity)
            local.show_imu = local.show_imu or incoming.show_imu
            local.imu_roll_corr_deg = incoming.imu_roll_corr_deg
            local.imu_pitch_corr_deg = incoming.imu_pitch_corr_deg
            local.walk_agent = incoming.walk_agent
            local.skate_agent = incoming.skate_agent
            local.mirror_policy = incoming.mirror_policy
            local.kp_vin_comp = incoming.kp_vin_comp
            local.imu_auto_trim = incoming.imu_auto_trim
        if remote is None:
            return local
        local_moves = local.active_moves
        if remote.active_moves & EXCLUSIVE_MOVES:
            local_moves = local_moves - EXCLUSIVE_MOVES
        local_moving = any(abs(v) > STICK_ACTIVE for v in local.velocity.values())
        velocity = remote.velocity if remote.has_velocity and not local_moving else local.velocity
        tuning = remote if remote.has_tuning else local
        return UserInput(
            active_moves=local_moves | remote.active_moves,
            velocity=dict(velocity),
            show_imu=local.show_imu,
            imu_roll_corr_deg=tuning.imu_roll_corr_deg,
            imu_pitch_corr_deg=tuning.imu_pitch_corr_deg,
            imu_mount_roll_deg=tuning.imu_mount_roll_deg,
            imu_mount_pitch_deg=tuning.imu_mount_pitch_deg,
            imu_calib_seq=tuning.imu_calib_seq,
            walk_agent=tuning.walk_agent,
            skate_agent=tuning.skate_agent,
            mirror_policy=tuning.mirror_policy,
            kp_vin_comp=tuning.kp_vin_comp,
            imu_auto_trim=tuning.imu_auto_trim,
            upper=remote.upper,
            head_yaw=remote.head_yaw,
        )
