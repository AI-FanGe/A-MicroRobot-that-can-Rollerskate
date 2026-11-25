# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

"""Upper-body motion for motors 13-20 (arms and head) that looks alive.

A gesture is a list of keyframes in a symmetric "authoring" space (degrees, the same for both
arms). Every time one plays it comes out a little different, as a person's would: its speed,
size and holds vary, each keyframe lands slightly off, the joints do not start together (the
shoulder leads, the elbow follows, the head looks first or catches up), some moves overshoot
and settle, and a few loops repeat a random number of times. Under every gesture an "alive"
layer keeps the body going: breathing, small drifts of the arms and glances of the head.

Pure Python (no numpy, no robot imports): the robot API imports it for the catalogue.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

UPPER_JOINTS = (
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_elbow",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_elbow",
    "head",
    "head_pitch",
)
UPPER_IDS = dict(zip(UPPER_JOINTS, range(13, 21)))

# Joint-space ranges [deg] the upper body may be sent to, as set on the robot: shoulder pitch
# (13/16) +-150, shoulder roll (14/17) 0..160 outward, elbow (15/18) 0..85 bent; head yaw (19)
# +-40 and head pitch (20) -10..+30. mjlab_xiaoxiaobai trains the free-arms walk on the same arm
# ranges (FREE_ARM_RANGES_DEG).
JOINT_LIMITS_DEG: dict[str, tuple[float, float]] = {
    "right_shoulder_pitch": (-150.0, 150.0),
    "right_shoulder_roll": (0.0, 160.0),
    "right_elbow": (-85.0, 0.0),
    "left_shoulder_pitch": (-150.0, 150.0),
    "left_shoulder_roll": (-160.0, 0.0),
    "left_elbow": (0.0, 85.0),
    "head": (-40.0, 40.0),
    "head_pitch": (-10.0, 30.0),
}

# Motor 20 turns the head down for positive angles (it travels 30 deg down, 10 up). If a nod
# lifts the head instead, set +1.
HEAD_PITCH_UP = -1.0

# Authoring space [deg], mirrored so both arms read the same:
#   rp / lp  shoulder pitch: + lifts the arm forward (90 points it ahead, 150 most), - back
#   rr / lr  shoulder roll: + swings the arm out sideways (it cannot cross inward)
#   re / le  elbow: + bends the forearm toward where the shoulder pitch points it (ahead with
#            the arm down, up with the arm held forward, back with the arm up)
#   yaw      head turn, + to the robot's left;   tilt  head pitch, + looks up
POSE_KEYS = ("rp", "rr", "re", "lp", "lr", "le", "yaw", "tilt")
REST: dict[str, float] = {"rp": 0.0, "rr": 6.0, "re": 12.0, "lp": 0.0, "lr": 6.0, "le": 12.0, "yaw": 0.0, "tilt": 0.0}
_MIRROR = {"rp": "lp", "rr": "lr", "re": "le", "lp": "rp", "lr": "rr", "le": "re", "yaw": "yaw", "tilt": "tilt"}

# Fastest a joint is moved [deg/s], whatever a gesture asks. Shoulder pitch (13 / 16) swings
# the whole arm and is held slower: keyframes that would need more are stretched in time.
MAX_SPEED_DEG_S = 420.0
PITCH_MAX_SPEED_DEG_S = 200.0
PITCH_KEYS = ("rp", "lp")
# Steepest slope of each easing over its keyframe (speed = slope * travel / duration).
_EASE_PEAK = {"io": 1.875, "in": 3.0, "out": 3.0, "back": 4.7, "lin": 1.0}
MANUAL_TAU_S = 0.06  # each of the two low-passes manual targets go through
SHAKE_RAMP_S = 0.25  # a Shake fades in and out over this
# A gesture's face is asked for this far ahead: it reaches the display through the control
# loop's status (every 0.2 s) and the API's tick (0.1 s), then blends in over 0.4 s.
FACE_LEAD_S = 0.2


def _clamp(value: float, lo: float, hi: float) -> float:
    return lo if value < lo else hi if value > hi else value


def to_joints(pose: dict[str, float], clamp: bool = True) -> dict[str, float]:
    """Authoring pose [deg] -> joint targets [rad], within JOINT_LIMITS_DEG unless clamp=False."""
    deg = {
        "right_shoulder_pitch": pose["rp"],
        "right_shoulder_roll": pose["rr"],
        "right_elbow": -pose["re"],
        "left_shoulder_pitch": -pose["lp"],
        "left_shoulder_roll": -pose["lr"],
        "left_elbow": pose["le"],
        "head": pose["yaw"],
        "head_pitch": HEAD_PITCH_UP * pose["tilt"],
    }
    if not clamp:
        return {name: math.radians(value) for name, value in deg.items()}
    return {name: math.radians(_clamp(value, *JOINT_LIMITS_DEG[name])) for name, value in deg.items()}


def from_joints(joints: dict[str, float], fallback: dict[str, float] | None = None) -> dict[str, float]:
    """Joint angles [rad] (any subset) -> authoring pose [deg]; missing ones from `fallback` (REST)."""
    pose = dict(fallback or REST)
    sign = {"rp": 1.0, "rr": 1.0, "re": -1.0, "lp": -1.0, "lr": -1.0, "le": 1.0, "yaw": 1.0, "tilt": HEAD_PITCH_UP}
    for key, name in zip(POSE_KEYS, UPPER_JOINTS):
        if name in joints:
            pose[key] = math.degrees(joints[name]) * sign[key]
    return pose


def _pose_limits() -> dict[str, tuple[float, float]]:
    lo = from_joints({n: math.radians(JOINT_LIMITS_DEG[n][0]) for n in UPPER_JOINTS})
    hi = from_joints({n: math.radians(JOINT_LIMITS_DEG[n][1]) for n in UPPER_JOINTS})
    return {key: (min(lo[key], hi[key]), max(lo[key], hi[key])) for key in POSE_KEYS}


POSE_LIMITS = _pose_limits()


def mirror(pose: dict[str, float]) -> dict[str, float]:
    out = {_MIRROR[key]: value for key, value in pose.items()}
    if "yaw" in out:
        out["yaw"] = -out["yaw"]
    return out


# ---------------------------------------------------------------- authoring


@dataclass(frozen=True)
class Key:
    """Move to `pose` (only the joints named; the rest keep going where they were) in `dur` s.
    `face`: the expression (expressions.py) shown from this key on, until another key names one."""

    dur: float
    pose: dict[str, float]
    ease: str = "io"  # io: smooth both ends, out: quick start, in: slow start, back: overshoots, lin
    hold: float = 0.0
    face: str | None = None


@dataclass(frozen=True)
class Shake:
    """Hold where the last key went and tremble all over for `dur` s: every joint shivers on its
    own, about `hz` times a second, `amp` deg on the arms (shoulder pitch, held slower, 3/4 of
    it; the head half). Unlike a key's, its length is played as written."""

    dur: float
    amp: float = 3.0
    hz: float = 8.0
    face: str | None = None


@dataclass(frozen=True)
class Rep:
    """Keys played a random number of times, lo..hi."""

    lo: int
    hi: int
    keys: tuple[Key | Shake, ...]


@dataclass(frozen=True)
class Gesture:
    name: str
    label: str
    steps: tuple[Key | Shake | Rep, ...]
    back: float = 0.6  # time to settle back to the base pose
    mirror: bool = False  # may play with either hand
    idle: bool = False  # among the small things done unasked
    exact: bool = False  # taught by hand: played as recorded, without the human variation


def K(dur: float, ease: str = "io", hold: float = 0.0, face: str | None = None, **pose: float) -> Key:
    return Key(dur, pose, ease, hold, face)


def S(dur: float, amp: float = 3.0, hz: float = 8.0, face: str | None = None) -> Shake:
    return Shake(dur, amp, hz, face)


def R(lo: int, hi: int, *keys: Key | Shake) -> Rep:
    return Rep(lo, hi, keys)


def G(name: str, label: str, *steps: Key | Shake | Rep, back: float = 0.6, mirror: bool = False,
      idle: bool = False) -> Gesture:
    return Gesture(name, label, steps, back, mirror, idle)


# fmt: off
GESTURES: dict[str, Gesture] = {g.name: g for g in (
    # Greetings
    G("wave", "单手打招呼",
      K(0.42, "out", rp=112, rr=86, re=70, lp=-10, lr=12, le=20, yaw=-14, tilt=5),
      R(2, 4, K(0.2, re=30, rr=100), K(0.2, re=85, rr=72)),
      K(0.25, re=65, hold=0.2, tilt=7),
      back=0.65, mirror=True),
    G("wave_both", "双手打招呼",
      K(0.45, "out", rp=112, rr=86, re=65, lp=112, lr=86, le=65, tilt=7),
      R(2, 4, K(0.22, re=28, le=85, rr=100, lr=72, yaw=10), K(0.22, re=85, le=28, rr=72, lr=100, yaw=-10)),
      K(0.25, re=65, le=65, yaw=0, hold=0.15),
      back=0.7),
    G("bye", "挥手再见",
      K(0.5, rp=-12, rr=12, re=15, lp=125, lr=78, le=60, yaw=12, tilt=-4),
      R(3, 5, K(0.3, le=22, lr=95), K(0.3, le=85, lr=64)),
      K(0.4, le=60, tilt=-8, yaw=4, hold=0.3),
      back=0.8, mirror=True),
    G("come_here", "招手过来",
      K(0.4, rp=88, rr=8, re=12, lp=-12, lr=10, le=15, tilt=2, yaw=-6),
      R(2, 4, K(0.18, "out", re=85, rp=96), K(0.24, re=12, rp=84)),
      back=0.55, mirror=True),
    G("bow", "鞠躬致意",
      K(0.5, rp=-30, rr=6, re=20, lp=-30, lr=6, le=20, tilt=-26, yaw=0),
      K(0.25, hold=0.45, tilt=-30, rp=-40, lp=-40),
      K(0.6, "in", tilt=-6, rp=-8, lp=-8, re=15, le=15),
      back=0.5),
    G("salute", "敬礼",
      K(0.32, "back", rp=128, rr=72, re=85, lp=-12, lr=4, le=6, tilt=6, yaw=0),
      K(0.2, hold=0.9, rp=124, tilt=8),
      K(0.35, "in", rp=70, rr=40, re=50),
      back=0.5),
    # Happy
    G("happy", "开心",
      K(0.32, "back", rp=70, rr=45, re=60, lp=70, lr=45, le=60, tilt=8, yaw=8),
      R(2, 3, K(0.2, rp=105, lp=25, rr=62, lr=30, yaw=-12, tilt=4), K(0.2, rp=25, lp=105, rr=30, lr=62, yaw=12, tilt=9)),
      K(0.3, rp=70, lp=70, yaw=0, tilt=6),
      back=0.6),
    G("cheer", "欢呼",
      K(0.3, "back", rp=150, rr=50, re=6, lp=150, lr=50, le=6, tilt=10, yaw=0),
      R(2, 3, K(0.16, "out", rp=115, lp=115, re=50, le=50, rr=30, lr=30), K(0.18, "back", rp=150, lp=150, re=4, le=4, rr=55, lr=55)),
      K(0.3, hold=0.25, rr=70, lr=70),
      back=0.8),
    G("heart", "双手比心",
      K(0.6, rp=128, rr=26, re=62, lp=128, lr=26, le=62, tilt=8, yaw=0),
      K(0.35, "out", rp=145, rr=10, re=82, lp=145, lr=10, le=82, hold=0.25),
      R(2, 3, K(0.12, "out", rr=24, lr=24, re=70, le=70), K(0.16, rr=8, lr=8, re=85, le=85),
        K(0.12, "out", rr=20, lr=20), K(0.3, rr=10, lr=10, hold=0.2)),
      R(1, 2, K(0.4, yaw=14, tilt=5), K(0.4, yaw=-14, tilt=9)),
      K(0.25, yaw=0, hold=0.3),
      back=0.9),
    G("clap", "鼓掌",
      K(0.35, rp=72, rr=22, re=55, lp=72, lr=22, le=55, tilt=5, yaw=0),
      R(4, 7, K(0.1, "out", rr=0, lr=0, re=45, le=45), K(0.13, rr=32, lr=32, re=62, le=62)),
      K(0.2, tilt=8, yaw=6),
      back=0.55),
    G("dance", "跳舞",
      K(0.3, rp=40, rr=40, re=80, lp=40, lr=40, le=80, tilt=4),
      R(3, 5, K(0.28, rp=100, lp=-40, rr=60, lr=20, re=55, le=85, yaw=18, tilt=0),
        K(0.28, rp=-40, lp=100, rr=20, lr=60, re=85, le=55, yaw=-18, tilt=8)),
      K(0.25, rp=40, lp=40, yaw=0),
      back=0.6),
    G("excited", "激动跺脚",
      K(0.22, "out", rp=35, rr=14, re=85, lp=35, lr=14, le=85, tilt=6),
      R(4, 6, K(0.11, rp=75, lp=-20, tilt=2), K(0.11, rp=-20, lp=75, tilt=8)),
      K(0.2, rp=35, lp=35),
      back=0.5),
    G("flex", "秀肌肉",
      K(0.4, "back", rp=95, rr=100, re=85, lp=95, lr=100, le=85, tilt=4, yaw=16),
      R(1, 2, K(0.22, re=60, le=60), K(0.18, "out", re=85, le=85)),
      K(0.3, yaw=-16, hold=0.4),
      back=0.65),
    G("fist_pump", "加油",
      K(0.25, rp=45, rr=14, re=85, lp=-15, lr=10, le=20, tilt=-4, yaw=-8),
      R(2, 3, K(0.14, "out", rp=100, re=85, tilt=-8), K(0.18, rp=30, re=78, tilt=-2)),
      K(0.25, rp=150, rr=30, re=12, tilt=8, hold=0.25),
      back=0.6, mirror=True),
    G("victory", "胜利",
      K(0.35, "back", rp=150, rr=26, re=4, lp=-30, lr=40, le=60, tilt=10, yaw=-12),
      K(0.25, hold=0.6, rp=148, tilt=8),
      R(1, 2, K(0.2, rr=45), K(0.2, rr=18)),
      back=0.7, mirror=True),
    # Angry / annoyed
    G("angry_hips", "生气叉腰",
      K(0.35, "back", rp=-40, rr=62, re=85, lp=-40, lr=62, le=85, tilt=-8, yaw=20),
      K(0.2, hold=0.5, yaw=26),
      R(2, 3, K(0.16, yaw=12, tilt=-5), K(0.16, yaw=28, tilt=-10)),
      K(0.3, yaw=-16, hold=0.4),
      back=0.8),
    G("angry_fists", "气得挥拳",
      K(0.25, rp=50, rr=16, re=85, lp=50, lr=16, le=85, tilt=-10, yaw=0),
      R(3, 5, K(0.12, "out", rp=90, lp=20, re=70, le=85), K(0.12, "out", rp=20, lp=90, re=85, le=70)),
      K(0.25, rp=25, lp=25, re=85, le=85, tilt=-4),
      back=0.6),
    G("no", "摇头说不",
      K(0.3, rp=78, rr=14, re=75, lp=-10, lr=8, le=15, tilt=-2),
      R(3, 4, K(0.17, yaw=-25, rr=36), K(0.17, yaw=25, rr=6)),
      K(0.2, yaw=0),
      back=0.5, mirror=True),
    G("facepalm", "扶额",
      K(0.45, rp=135, rr=14, re=85, lp=-8, lr=10, le=20, tilt=-16, yaw=-8),
      K(0.25, hold=0.4, tilt=-20),
      R(2, 3, K(0.3, yaw=-18), K(0.3, yaw=4)),
      K(0.45, "in", tilt=-6, rp=60, re=55),
      back=0.6, mirror=True),
    # Calm / social
    G("nod", "点头",
      K(0.25, rp=6, rr=10, re=24, lp=6, lr=10, le=24, tilt=2),
      R(2, 3, K(0.16, "out", tilt=-22, re=32, le=32), K(0.2, tilt=2, re=20, le=20)),
      back=0.45, idle=True),
    G("think", "思考",
      K(0.6, rp=115, rr=12, re=85, lp=25, lr=20, le=85, tilt=6, yaw=-18),
      K(0.4, hold=0.8, tilt=8, yaw=-24),
      R(1, 2, K(0.5, yaw=-6, tilt=4), K(0.5, yaw=-26, tilt=9)),
      back=0.8, mirror=True, idle=True),
    G("scratch_head", "挠头",
      K(0.5, rp=150, rr=65, re=85, lp=-6, lr=8, le=18, tilt=-8, yaw=-12),
      R(3, 6, K(0.11, re=62, rr=74), K(0.11, re=85, rr=58)),
      K(0.25, tilt=-4, yaw=-4, hold=0.2),
      back=0.65, mirror=True, idle=True),
    G("shrug", "摊手",
      K(0.35, "back", rp=25, rr=55, re=85, lp=25, lr=55, le=85, tilt=6, yaw=12),
      K(0.25, hold=0.5, rr=68, lr=68, tilt=8),
      K(0.4, "in", rr=35, lr=35, re=65, le=65),
      back=0.55, idle=True),
    G("look_around", "东张西望",
      K(0.4, rp=-15, rr=14, re=30, lp=-15, lr=14, le=30, yaw=35, tilt=0),
      K(0.15, hold=0.6, tilt=5),
      K(0.55, yaw=-36, tilt=-3, rr=10, lr=20),
      K(0.15, hold=0.6, tilt=4),
      K(0.4, yaw=8),
      back=0.5, idle=True),
    G("stretch", "伸懒腰",
      K(1.0, "in", rp=150, rr=45, re=6, lp=150, lr=45, le=6, tilt=8),
      K(0.4, hold=0.7, rr=60, lr=60, tilt=9),
      K(0.7, rr=120, lr=120, rp=90, lp=90, tilt=2),
      K(0.3, hold=0.4, rr=135, lr=135),
      back=1.0, idle=True),
    G("yawn", "打哈欠",
      K(0.6, rp=130, rr=16, re=85, lp=45, lr=28, le=60, tilt=10, yaw=-5),
      K(0.4, hold=0.7, tilt=10),
      K(0.5, rp=150, lp=150, rr=45, lr=45, re=20, le=20, tilt=6),
      back=1.0, mirror=True, idle=True),
    # Still half asleep, fists drawn in; a big stretch up, held while it shivers all over; arms
    # down with a sigh of relief. Held below the joint limits so the shiver is not clipped.
    G("stretch_shiver", "伸懒腰发抖",
      K(0.55, rp=25, rr=8, re=60, lp=25, lr=8, le=60, tilt=-10, yaw=0, face="waking"),
      K(1.1, rp=130, rr=28, re=10, lp=130, lr=28, le=10, tilt=3, face="stretch"),
      K(0.45, "out", hold=0.15, rr=38, lr=38, tilt=3),
      S(2.0, amp=3.0, hz=8.0, face="stretch_shiver"),
      K(1.0, rp=35, rr=55, re=18, lp=35, lr=55, le=18, tilt=-2, hold=0.35, face="relieved"),
      back=0.9),
    G("point", "指向前方",
      K(0.35, "back", rp=95, rr=12, re=4, lp=-12, lr=8, le=18, tilt=2, yaw=-8),
      K(0.2, hold=0.8, yaw=-6),
      R(1, 2, K(0.15, rp=104), K(0.15, rp=92)),
      back=0.55, mirror=True),
    G("hug", "抱抱",
      K(0.5, rp=85, rr=88, re=15, lp=85, lr=88, le=15, tilt=8, yaw=0),
      K(0.25, hold=0.3, rr=98, lr=98),
      K(0.45, "in", rr=8, lr=8, re=65, le=65, tilt=4),
      K(0.25, hold=0.7, re=75, le=75),
      back=0.7),
    # Shy / sad / scared
    G("shy", "害羞",
      K(0.5, rp=65, rr=6, re=85, lp=65, lr=6, le=85, tilt=-16, yaw=-22),
      R(2, 3, K(0.35, rp=50, lp=78, yaw=-30), K(0.35, rp=78, lp=50, yaw=-14)),
      K(0.3, tilt=-20, hold=0.4),
      back=0.8, mirror=True),
    G("sad", "难过",
      K(1.0, "in", rp=-8, rr=2, re=4, lp=-8, lr=2, le=4, tilt=-26, yaw=8),
      K(0.4, hold=0.9, tilt=-30),
      R(1, 2, K(0.6, yaw=-8), K(0.6, yaw=10)),
      back=1.0),
    G("scared", "害怕捂脸",
      K(0.18, "out", rp=142, rr=12, re=85, lp=142, lr=12, le=85, tilt=-18, yaw=14),
      R(4, 7, K(0.06, rr=18, lr=6, yaw=10), K(0.06, rr=6, lr=18, yaw=16)),
      K(0.4, hold=0.4, tilt=-22),
      back=0.9),
    G("surprised", "惊讶",
      K(0.16, "back", rp=50, rr=70, re=60, lp=50, lr=70, le=60, tilt=10, yaw=0),
      K(0.1, hold=0.7, rr=78, lr=78),
      K(0.6, "in", rr=25, lr=25, re=35, le=35, tilt=4),
      back=0.6),
)}
# fmt: on

GESTURE_NAMES = tuple(GESTURES)
IDLE_GESTURES = tuple(name for name, g in GESTURES.items() if g.idle)


def catalog() -> list[dict]:
    return [{"name": g.name, "label": g.label, "index": i} for i, g in enumerate(GESTURES.values(), start=1)]


# A gesture made up on the spot (the voice / chat assistant): at most this many keyframes, each
# move and hold within these bounds, all of it this long at most. Poses are clamped to
# POSE_LIMITS; playback then caps the joint speeds like for any gesture.
CUSTOM = "custom"
CUSTOM_MAX_KEYS = 12
CUSTOM_DUR_S = (0.25, 3.0)
CUSTOM_HOLD_S = (0.0, 3.0)
CUSTOM_MAX_S = 15.0
# Moves taught in the web UI's editor (api/gesture_store.py) may be longer.
TAUGHT_MAX_KEYS = 40
TAUGHT_MAX_S = 90.0


def custom_keys(raw: object, max_keys: int = CUSTOM_MAX_KEYS, max_s: float = CUSTOM_MAX_S) -> list[dict]:
    """Keyframes as sent ([{dur, pose: {key: deg}, ease?, hold?}, ...]) cleaned up: unknown joints
    and bad numbers dropped, values clamped, cut at `max_keys` keys and `max_s` seconds."""
    keys: list[dict] = []
    total = 0.0
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict) or len(keys) >= max_keys:
            continue
        pose = {}
        for name, value in (item.get("pose") or {}).items() if isinstance(item.get("pose"), dict) else ():
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if name in POSE_LIMITS and math.isfinite(value):
                pose[name] = round(_clamp(value, *POSE_LIMITS[name]), 1)
        try:
            dur = _clamp(float(item.get("dur", 0.6)), *CUSTOM_DUR_S)
            hold = _clamp(float(item.get("hold", 0.0) or 0.0), *CUSTOM_HOLD_S)
        except (TypeError, ValueError):
            continue
        if not pose or not (math.isfinite(dur) and math.isfinite(hold)) or total + dur + hold > max_s:
            continue
        ease = item.get("ease") if item.get("ease") in _EASE_PEAK else "io"
        keys.append({"dur": round(dur, 2), "pose": pose, "ease": ease, "hold": round(hold, 2)})
        total += dur + hold
    return keys


def custom_gesture(keys: list[dict], exact: bool = False) -> Gesture:
    clean = custom_keys(keys, TAUGHT_MAX_KEYS, TAUGHT_MAX_S) if exact else custom_keys(keys)
    steps = tuple(Key(k["dur"], k["pose"], k["ease"], k["hold"]) for k in clean)
    return Gesture(CUSTOM, "自编动作", steps, back=0.8, exact=exact)


# ---------------------------------------------------------------- playback


def _ease(kind: str, x: float) -> float:
    x = _clamp(x, 0.0, 1.0)
    if kind == "lin":
        return x
    if kind == "in":
        return x * x * x
    if kind == "out":
        return 1.0 - (1.0 - x) ** 3
    if kind == "back":  # overshoots by ~10 % and settles
        c1 = 1.70158
        return 1.0 + (c1 + 1.0) * (x - 1.0) ** 3 + c1 * (x - 1.0) ** 2
    return x * x * x * (x * (6.0 * x - 15.0) + 10.0)


def _pitch_paced(dur: float, a: dict[str, float], b: dict[str, float], ease: str) -> float:
    """`dur`, or longer if the shoulder pitch would go faster than PITCH_MAX_SPEED_DEG_S."""
    travel = max(abs(b[name] - a[name]) for name in PITCH_KEYS)
    return max(dur, _EASE_PEAK.get(ease, 1.875) * travel / PITCH_MAX_SPEED_DEG_S)


@dataclass
class _Segment:
    t0: float
    dur: float
    start: dict[str, float]
    end: dict[str, float] | None  # None: back to the live base pose
    ease: str
    face: str | None = None
    shake: dict[str, tuple[float, float, float]] | None = None  # joint -> (amp [deg], rad/s, phase)


class Performance:
    """One playing of a gesture: its keyframes turned into a timeline with human variation."""

    def __init__(self, gesture: Gesture, start: dict[str, float], rng: random.Random, size: float = 1.0) -> None:
        self.gesture = gesture
        exact = gesture.exact
        speed = 1.0 if exact else _clamp(math.exp(rng.gauss(0.0, 0.12)), 0.8, 1.25)
        amp = size if exact else rng.uniform(0.88, 1.08) * size
        flip = gesture.mirror and rng.random() < 0.5
        self.segments: list[_Segment] = []
        pose = dict(start)
        t = 0.0
        face = None
        for key in self._expand(gesture.steps, rng):
            face = key.face or face
            if isinstance(key, Shake):
                shake = {}
                for name in POSE_KEYS:
                    share = 0.5 if name in ("yaw", "tilt") else 0.75 if name in PITCH_KEYS else 1.0
                    size_deg = key.amp * size * share * rng.uniform(0.7, 1.2)
                    shake[name] = (size_deg, 2 * math.pi * key.hz * rng.uniform(0.85, 1.2), rng.uniform(0, 2 * math.pi))
                self.segments.append(_Segment(t, key.dur, pose, pose, "lin", face, shake))
                t += key.dur
                continue
            target = dict(pose)
            for name, value in (mirror(key.pose) if flip else key.pose).items():
                jitter = 0.0 if exact else rng.gauss(0.0, 1.5 if name in ("yaw", "tilt") else 2.5)
                target[name] = REST[name] + (value - REST[name]) * amp + jitter
            stretch = 1.0 if exact else rng.uniform(0.85, 1.18)
            dur = _pitch_paced(key.dur * speed * stretch, pose, target, key.ease)
            self.segments.append(_Segment(t, dur, pose, target, key.ease, face))
            t += dur
            if key.hold > 0:
                hold = key.hold * (1.0 if exact else speed * rng.uniform(0.6, 1.5))
                self.segments.append(_Segment(t, hold, target, target, "lin", face))
                t += hold
            pose = target
        back = _pitch_paced(gesture.back * (1.0 if exact else speed * rng.uniform(0.9, 1.3)), pose, start, "io")
        self.segments.append(_Segment(t, back, pose, None, "io", face))
        self.duration = t + back
        # Joints do not move together: the shoulder leads, roll and elbow follow, one arm may
        # trail the other, and the head looks first or catches up.
        if exact:
            self.lag = {name: 0.0 for name in POSE_KEYS}
        else:
            side = {"r": rng.uniform(0.0, 0.07), "l": rng.uniform(0.0, 0.07)}
            self.lag = {
                "rp": side["r"], "rr": side["r"] + rng.uniform(0.0, 0.04), "re": side["r"] + rng.uniform(0.03, 0.09),
                "lp": side["l"], "lr": side["l"] + rng.uniform(0.0, 0.04), "le": side["l"] + rng.uniform(0.03, 0.09),
                "yaw": rng.uniform(-0.12, 0.1), "tilt": rng.uniform(-0.1, 0.1),
            }
        self._start = start
        self._lead = min(0.0, min(self.lag.values()))
        # Joints the gesture hardly uses still take part: a small sway that rises and fades
        # with the gesture, so the whole upper body is in it.
        reach = {name: 0.0 for name in POSE_KEYS}
        for seg in self.segments:
            if seg.end is not None:
                for name in POSE_KEYS:
                    reach[name] = max(reach[name], abs(seg.end[name] - start[name]))
        self.sway = {}
        for name in POSE_KEYS:
            if not exact and reach[name] < 8.0 and self.duration > 0.8:
                head = name in ("yaw", "tilt")
                amp = rng.uniform(3.0, 7.0) if head else rng.uniform(3.0, 6.0)
                self.sway[name] = (amp * size, rng.uniform(0.35, 0.8) * 2 * math.pi, rng.uniform(0, 2 * math.pi))

    @staticmethod
    def _expand(steps: tuple[Key | Shake | Rep, ...], rng: random.Random) -> list[Key | Shake]:
        keys: list[Key | Shake] = []
        for step in steps:
            if isinstance(step, Rep):
                for _ in range(rng.randint(step.lo, step.hi)):
                    keys.extend(step.keys)
            else:
                keys.append(step)
        return keys

    def done(self, t: float) -> bool:
        return t >= self.duration + max(self.lag.values()) - self._lead

    def face(self, t: float) -> str | None:
        """The expression the gesture asks for at t (its keys' `face`), None when it names none."""
        for seg in self.segments:
            if t < seg.t0 + seg.dur:
                return seg.face
        return None

    def pose(self, t: float, base: dict[str, float]) -> dict[str, float]:
        out = {}
        for name in POSE_KEYS:
            tt = t - (self.lag[name] - self._lead)
            if tt <= 0.0:
                out[name] = self._start[name]
                continue
            seg = self.segments[-1]
            for candidate in self.segments:
                if tt < candidate.t0 + candidate.dur:
                    seg = candidate
                    break
            a = seg.start[name]
            b = base[name] if seg.end is None else seg.end[name]
            out[name] = a + (b - a) * _ease(seg.ease, (tt - seg.t0) / seg.dur)
            if seg.shake is not None:
                amp, w, phase = seg.shake[name]
                into = tt - seg.t0
                env = _ease("io", min(into, seg.dur - into) / SHAKE_RAMP_S)
                swell = 0.8 + 0.4 * into / seg.dur  # the shiver grows as the stretch gives out
                out[name] += amp * env * swell * math.sin(w * into + phase)
            if name in self.sway:
                amp, w, phase = self.sway[name]
                env = math.sin(math.pi * _clamp(tt / self.duration, 0.0, 1.0))
                wave = math.sin(w * tt + phase)
                if name in ("rr", "lr", "re", "le"):  # these sit near a limit: sway away from it
                    wave = 0.5 + 0.5 * wave
                out[name] += amp * env * env * wave
        return out


class _Noise:
    """Smooth random wander in [-1, 1]: two sines of random rate and phase."""

    def __init__(self, rng: random.Random, hz: float) -> None:
        self._parts = [(rng.uniform(0.6, 1.4) * hz * 2 * math.pi, rng.uniform(0, 2 * math.pi)) for _ in range(2)]

    def __call__(self, t: float) -> float:
        return sum(math.sin(w * t + p) for w, p in self._parts) / 2.0


class Alive:
    """Breathing, small drifts of the arms and glances of the head, added on top of everything."""

    def __init__(self, rng: random.Random) -> None:
        self._rng = rng
        self._breath = rng.uniform(0, 2 * math.pi)
        self._breath_rate = _Noise(rng, 0.05)
        self._drift = {name: _Noise(rng, rng.uniform(0.08, 0.25)) for name in POSE_KEYS}
        self._gaze = [0.0, 0.0]
        self._gaze_target = [0.0, 0.0]
        self._next_glance = 0.0

    def offsets(self, now: float, dt: float, amount: float) -> dict[str, float]:
        rng = self._rng
        self._breath += 2 * math.pi * 0.24 * (1.0 + 0.2 * self._breath_rate(now)) * dt
        b = math.sin(self._breath)
        if now >= self._next_glance:
            # Like eye fixations: mostly short, now and then a long look; often back to the front.
            self._next_glance = now + _clamp(math.exp(rng.gauss(math.log(2.2), 0.6)), 0.5, 9.0)
            if rng.random() < 0.35:
                self._gaze_target = [rng.gauss(0.0, 2.0), rng.gauss(0.0, 1.0)]
            else:
                self._gaze_target = [_clamp(rng.gauss(0.0, 10.0), -20.0, 20.0), _clamp(rng.gauss(0.0, 3.5), -7.0, 4.0)]
        k = 1.0 - math.exp(-dt / 0.16)
        for i in range(2):
            self._gaze[i] += (self._gaze_target[i] - self._gaze[i]) * k
        d = {name: noise(now) for name, noise in self._drift.items()}
        return {
            "rp": amount * (1.0 * b + 1.6 * d["rp"]),
            "lp": amount * (1.0 * b + 1.6 * d["lp"]),
            "rr": amount * (1.4 * b + 1.4 * d["rr"]),
            "lr": amount * (1.4 * b + 1.4 * d["lr"]),
            "re": amount * (1.5 * math.sin(self._breath + 1.1) + 2.2 * d["re"]),
            "le": amount * (1.5 * math.sin(self._breath + 1.1) + 2.2 * d["le"]),
            "yaw": amount * (self._gaze[0] + 1.0 * d["yaw"]),
            "tilt": amount * (self._gaze[1] + 0.8 * b + 0.6 * d["tilt"]),
        }


# Swaying to music: the beat grid comes from the music player's beat tracker through the API.
GROOVE_LEAD_S = 0.12     # the servos lag their targets: aim this far ahead of the beat
GROOVE_STALE_S = 3.0     # no fresh grid for this long (paused, stopped, hung): fade out
GROOVE_FADE_S = 1.2
GROOVE_LOCK_S = 0.6      # how fast the sway's phase is pulled onto a new grid
GROOVE_SNAP = 0.3        # beats of phase error a still-faded-out sway just jumps across
GROOVE_BLEND_BEATS = 1.0  # a new move, new parameters or a mirrored repeat blend in over this
SECTION_BEATS, PHRASE_BEATS, BAR_BEATS = 16, 8, 4
# Dancing to the song's map (src/song_analysis.py), in dance beats from the section's start: new
# parameters every PLAN_PHRASE, maybe mirrored every PLAN_MIRROR; a map that comes in part way
# through a section is joined on a multiple of PLAN_ENTRY.
PLAN_PHRASE, PLAN_MIRROR, PLAN_ENTRY = 16, 8, 8
PLAN_EARLY = 0.75        # beats: a section's move blends in before its first beat, to be there on it
# How big each kind of section is danced (against its loudness).
PLAN_KIND_SIZE = {"intro": 0.7, "outro": 0.7, "break": 0.5}
# Amplitudes [deg] at size 1, a full-confidence beat and the song's loudest; moves scale them.
GROOVE_DEG = {"pitch": 28.0, "roll": 18.0, "elbow": 28.0, "yaw": 24.0, "nod": 14.0}
# The sway's own speed limit [deg/s], about 60 % of what the engine allows (MAX_SPEED_DEG_S, and
# PITCH_MAX_SPEED_DEG_S for shoulder pitch: the speeds the free-arms walk was trained with), so
# that gestures, held poses and the alive layer on top keep the motors within those.
GROOVE_MAX_SPEED = {"rp": 120.0, "lp": 120.0, "rr": 250.0, "lr": 250.0, "re": 250.0, "le": 250.0,
                    "yaw": 220.0, "tilt": 160.0}


def _hit(x: float) -> float:
    """1 on each beat, 0 half way between, smooth."""
    return (0.5 + 0.5 * math.cos(2.0 * math.pi * x)) ** 2


def _swing(x: float, beats: float) -> float:
    """+1 on beat 0, -1 half a cycle of `beats` later."""
    return math.cos(2.0 * math.pi * x / beats)


# Moves: (beat in the section, its accent, params, amplitudes [deg]) -> pose offsets [deg].
# Each param set (Groove._params) picks the direction, size, posture and asymmetry.

def _move_sway(b: float, j: float, p: dict, A: dict) -> dict[str, float]:
    """Side to side over two beats, along a direction between front-back and sideways."""
    r, l = _swing(b, 2) * p["mirror"], _swing(b - p["lag"], 2) * p["mirror"]
    c, s = math.cos(p["theta"]), math.sin(p["theta"])
    elbow = A["elbow"] * 0.5 * _hit(b - p["elbow_lag"]) * j
    return {"rp": A["pitch"] * c * r, "lp": -A["pitch"] * c * l * p["asym"],
            "rr": A["roll"] * s * (0.5 + 0.5 * r), "lr": A["roll"] * s * (0.5 - 0.5 * l) * p["asym"],
            "re": elbow, "le": elbow * p["asym"],
            "yaw": A["yaw"] * r * p["head"], "tilt": -A["nod"] * 0.5 * _hit(b) * j}


def _move_pump(b: float, j: float, p: dict, A: dict) -> dict[str, float]:
    """Both arms punch forward and up on every beat, the head nods with them."""
    hit = _hit(b) * j
    elbow = A["elbow"] * 0.9 * _hit(b - p["elbow_lag"]) * j
    look = A["yaw"] * 0.5 * _swing(b, 4) * p["mirror"] * p["head"]
    return {"rp": A["pitch"] * 0.8 * hit, "lp": A["pitch"] * 0.8 * hit * p["asym"],
            "rr": A["roll"] * 0.25 * hit, "lr": A["roll"] * 0.25 * hit * p["asym"],
            "re": elbow, "le": elbow * p["asym"], "yaw": look, "tilt": -A["nod"] * hit}


def _move_roll(b: float, j: float, p: dict, A: dict) -> dict[str, float]:
    """The arms circle over two beats, half a turn apart."""
    a = math.pi * b
    a2 = a + math.pi * (1.0 + p["lag"])
    out = {"yaw": A["yaw"] * 0.6 * math.cos(a) * p["head"], "tilt": -A["nod"] * 0.4 * (0.5 + 0.5 * math.sin(a)) * j}
    for side, angle, size in (("r", a, 1.0), ("l", a2, p["asym"])):
        out[side + "p"] = A["pitch"] * 0.7 * math.sin(angle) * p["mirror"] * size
        out[side + "r"] = A["roll"] * (0.5 + 0.5 * math.cos(angle)) * size
        out[side + "e"] = A["elbow"] * 0.5 * (0.5 + 0.5 * math.sin(angle - math.pi * p["elbow_lag"])) * size
    return out


def _move_bounce(b: float, j: float, p: dict, A: dict) -> dict[str, float]:
    """Elbows bouncing (twice a beat for slow, busy songs) over a slow sway."""
    x = 2.0 * b if p["double"] else b
    bounce = _hit(x) * j
    sway = A["pitch"] * 0.35 * _swing(b, 4) * p["mirror"]
    return {"rp": sway, "lp": -sway * p["asym"], "rr": A["roll"] * 0.2 * bounce, "lr": A["roll"] * 0.2 * bounce,
            "re": A["elbow"] * 0.8 * bounce, "le": A["elbow"] * 0.8 * bounce * p["asym"],
            "yaw": A["yaw"] * 0.5 * _swing(b, 4) * p["mirror"] * p["head"], "tilt": -A["nod"] * 0.6 * bounce}


def _move_open(b: float, j: float, p: dict, A: dict) -> dict[str, float]:
    """Arms open out on one beat and close in on the next."""
    o = 0.5 + 0.5 * _swing(b, 2)
    return {"rp": A["pitch"] * 0.35 * (1 - o), "lp": A["pitch"] * 0.35 * (1 - o) * p["asym"],
            "rr": A["roll"] * 1.1 * o * j, "lr": A["roll"] * 1.1 * o * j * p["asym"],
            "re": A["elbow"] * 0.8 * (1 - o), "le": A["elbow"] * 0.8 * (1 - o) * p["asym"],
            "yaw": A["yaw"] * 0.5 * _swing(b, 8) * p["mirror"] * p["head"], "tilt": -A["nod"] * 0.7 * _hit(b) * j}


def _move_point(b: float, j: float, p: dict, A: dict) -> dict[str, float]:
    """One arm up on the bar's first beat, the other on its third; the head looks at it."""
    r = 0.5 + 0.5 * _swing(b, 4) * p["mirror"]
    return {"rp": A["pitch"] * 1.2 * r, "lp": A["pitch"] * 1.2 * (1 - r) * p["asym"],
            "rr": A["roll"] * 0.3 * r, "lr": A["roll"] * 0.3 * (1 - r),
            "re": A["elbow"] * 0.6 * (1 - r), "le": A["elbow"] * 0.6 * r,
            "yaw": -A["yaw"] * 0.8 * (2 * r - 1), "tilt": -A["nod"] * 0.5 * _hit(b) * j}


def _move_chill(b: float, j: float, p: dict, A: dict) -> dict[str, float]:
    """A slow sway over four beats, the head rolling round."""
    s = _swing(b, 4) * p["mirror"]
    l = _swing(b - p["lag"], 4) * p["mirror"]
    return {"rp": A["pitch"] * 0.5 * s, "lp": -A["pitch"] * 0.5 * l * p["asym"],
            "rr": A["roll"] * 0.3 * (0.5 + 0.5 * s), "lr": A["roll"] * 0.3 * (0.5 - 0.5 * l),
            "re": A["elbow"] * 0.3 * _hit(b / 2), "le": A["elbow"] * 0.3 * _hit(b / 2),
            "yaw": A["yaw"] * 0.8 * s * p["head"], "tilt": -A["nod"] * 0.5 * (0.5 + 0.5 * math.sin(math.pi * b / 2))}


GROOVE_MOVES = {"sway": _move_sway, "pump": _move_pump, "roll": _move_roll, "bounce": _move_bounce,
                "open": _move_open, "point": _move_point, "chill": _move_chill}


def _fill(x: float, A: dict, size: float = 1.0) -> dict[str, float]:
    """Both arms up and out, peaking at x = 0 (a section's last beat), down by the next one."""
    f = size * (0.5 + 0.5 * math.cos(math.pi * x)) if abs(x) < 1.0 else 0.0
    return {"rp": A["pitch"] * 0.9 * f, "lp": A["pitch"] * 0.9 * f, "rr": A["roll"] * 0.8 * f, "lr": A["roll"] * 0.8 * f,
            "re": -A["elbow"] * 0.3 * f, "le": -A["elbow"] * 0.3 * f, "tilt": A["nod"] * 0.4 * f}


class Groove:
    """Dancing to music with the arms and head, as authoring-pose offsets [deg].

    The beat phase is a flywheel pulled onto the grid ({"t0": wall time of a beat, "period" [s],
    "confidence", "energy", "busy", "downbeat": wall time of a bar's first beat,
    "bar_confidence", "bar_beats": periods per bar, "size", "ts"}), so a grid that moves a little (every second) or a pause
    does not jerk the arms.

    With the song's map ("index": the map's dance beat at t0, "section" and "next": {"start",
    "end" (dance beats), "label", "variant", "energy", "busy", "kind"}, "end", "song"), the dance
    follows the song's structure: each section starts a move on its first beat (a quiet intro,
    outro or break chills; loud sections pump and open, busy ones bounce), every PLAN_PHRASE
    beats new parameters, every PLAN_MIRROR maybe mirrored, a fill on the section's last beat
    into the next, and the song's last beat ends it. A section that repeats (the second chorus)
    dances what it danced the first time.

    Without one (the first seconds of a song, or no map), beats are counted from the bars the
    beat tracker finds, in bars of four, phrases of eight and sections of sixteen, with the same
    kind of choices made from the music's live loudness and beat; while the map is being made
    ("analysing") only gently.

    The parameters: direction, size, how high and wide the arms are held, which side leads,
    whether the head follows or counters. Slow noise keeps size, direction and posture
    drifting, and every beat lands a little harder or softer."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random()
        self.weight = 0.0
        self._beats = 0.0
        self._period = 0.5
        self._amount = 0.0
        self._gain = 1.0
        self._features = {"energy": 0.5, "confidence": 0.5, "busy": 0.0, "bar_confidence": 0.0, "bpm": 100.0}
        self._energy_fast = self._energy_slow = 0.5
        self._bar_votes = [0.0] * BAR_BEATS
        self._bar = 0
        self._grid_ts: float | None = None
        self._count: int | None = None  # the whole beat last seen
        self._accents: dict[int, float] = {}
        self._voice: dict | None = None
        self._previous: dict | None = None
        self._blend_from = 0.0
        self._out: dict[str, float] = {}
        self._drift = {name: _Noise(self.rng, hz) for name, hz in (("amp", 0.06), ("theta", 0.04), ("lift", 0.03),
                                                                   ("spread", 0.035))}
        self._analysing = False
        self._plan: dict | None = None      # the map around now: "section", "next", "end", "bar"
        self._offset: int | None = None     # the map's beat index less the flywheel's count
        self._song: str | None = None
        self._memory: dict[tuple, dict] = {}  # (label, variant) -> {"move", "phrases": [params]}
        self._section: dict | None = None   # the map's section being danced
        self._level = 0.0                   # how big the map's section is danced: 0-1

    @property
    def move(self) -> str | None:
        return self._voice["move"] if self._voice and self.weight > 0.0 else None

    @property
    def section(self) -> str | None:
        """The map's section being danced (its label and kind), if any."""
        s = self._section if self._plan is not None and self.weight > 0.0 else None
        return f"{s['label']}{s.get('variant', 0) + 1} {s['kind']}" if s else None

    def offsets(self, wall: float, dt: float, grid: dict | None) -> dict[str, float]:
        try:
            period = float(grid["period"]) if grid else 0.0
            fresh = 0.25 < period < 2.0 and wall - float(grid.get("ts") or wall) < GROOVE_STALE_S
        except (TypeError, ValueError, KeyError):
            fresh = False
        if fresh:
            self._follow(wall, dt, grid, period)
            f = self._features
            size = _clamp(float(grid.get("size") or 0.0), 0.0, 1.0)
            if self._plan is not None:
                want = size * self._level
            else:
                # A weak beat (a ballad, speech) still sways, gently.
                want = size * (0.4 + 0.6 * f["confidence"]) * (0.5 + 0.5 * f["energy"]) * (0.6 if self._analysing else 1.0)
        else:
            self._beats += dt / self._period
            want = 0.0
        self._amount += (want - self._amount) * (1.0 - math.exp(-dt / 0.5))
        step = dt / GROOVE_FADE_S
        self.weight = min(1.0, self.weight + step) if fresh else max(0.0, self.weight - step)
        a = self._amount * self.weight
        if a <= 1e-4:
            self._out, self._voice, self._previous, self._count, self._section = {}, None, None, None, None
            return {}
        self._on_beats()
        target = {name: value * a * self._gain for name, value in self._dance(wall).items()}
        # Never faster than GROOVE_MAX_SPEED; hitting it often makes the dance smaller.
        clipped = False
        for name in POSE_KEYS:
            value, last = target.get(name, 0.0), self._out.get(name, 0.0)
            step = GROOVE_MAX_SPEED[name] * dt
            if abs(value - last) > step:
                clipped = True
                value = last + math.copysign(step, value - last)
            self._out[name] = value
        self._gain = max(0.4, self._gain - 0.8 * dt) if clipped else min(1.0, self._gain + 0.1 * dt)
        return dict(self._out)

    def _follow(self, wall: float, dt: float, grid: dict, period: float) -> None:
        """Pulls the flywheel onto the grid; takes the music's features and votes for the bar."""
        target = (wall + GROOVE_LEAD_S - float(grid["t0"])) / period
        if self.weight < 0.05:
            self._beats, self._period = target, period
        self._period += (period - self._period) * min(1.0, dt / GROOVE_LOCK_S)
        self._beats += dt / self._period
        error = (target - self._beats + 0.5) % 1.0 - 0.5
        self._beats += error if abs(error) > GROOVE_SNAP and self.weight < 0.2 else error * min(1.0, dt / GROOVE_LOCK_S)
        f = self._features
        for key in ("energy", "confidence", "busy", "bar_confidence"):
            f[key] = _clamp(float(grid.get(key) or 0.0), 0.0, 1.0)
        f["bpm"] = 60.0 / period
        beat_s = self._period
        self._energy_fast += (f["energy"] - self._energy_fast) * min(1.0, dt / (2 * beat_s))
        self._energy_slow += (f["energy"] - self._energy_slow) * min(1.0, dt / (SECTION_BEATS * beat_s))
        self._analysing = bool(grid.get("analysing"))
        if grid.get("index") is not None:
            # The flywheel's count when the robot is on the map's beat `index`.
            count = self._beats + (float(grid["t0"]) - wall - GROOVE_LEAD_S) / self._period
            self._offset = int(grid["index"]) - round(count)
            if grid.get("song") != self._song:
                self._song, self._memory, self._section = grid.get("song"), {}, None
            self._plan = {"section": grid.get("section"), "next": grid.get("next"),
                          "end": int(grid.get("end") or 0), "bar": max(1, int(grid.get("bar_beats") or BAR_BEATS))}
            s = self._plan_section(math.floor(self._beats) + self._offset)
            self._level = 0.0 if s is None else (0.35 + 0.65 * float(s["energy"])) * PLAN_KIND_SIZE.get(s["kind"], 1.0)
            return
        self._plan = self._offset = None
        ts = grid.get("ts")
        if ts != self._grid_ts and grid.get("downbeat") is not None:
            self._grid_ts = ts
            # The flywheel's count at the downbeat (it runs GROOVE_LEAD_S ahead of the music).
            count = self._beats + (float(grid["downbeat"]) - wall - GROOVE_LEAD_S) / self._period
            self._bar_votes = [v * 0.85 for v in self._bar_votes]
            # A grid at half the song's tempo has two bars in four of its beats: either one will do.
            step = max(1, min(BAR_BEATS, int(grid.get("bar_beats") or BAR_BEATS)))
            for k in range(round(count) % step, BAR_BEATS, step):
                self._bar_votes[k] += f["bar_confidence"]

    def _on_beats(self) -> None:
        """At each whole beat: the bar, phrase and section events."""
        plan = self._plan is not None and self._offset is not None
        count = math.floor(self._beats + (PLAN_EARLY if plan else 0.0))
        if self._count == count and self._voice is not None:
            return
        first = self._voice is None
        self._count = count
        self._accents = {n: v for n, v in self._accents.items() if n >= count - 2}
        for n in (count, count + 1):
            self._accents.setdefault(n, self.rng.uniform(0.85, 1.15))
        if plan:
            self._plan_beat(count, first)
            return
        self._section = None
        best = max(range(BAR_BEATS), key=lambda k: self._bar_votes[k])
        if self._bar_votes[best] > 1.3 * self._bar_votes[self._bar] + 0.05 and best != self._bar:
            self._bar = best
            if not first:  # the beat numbering moved: the same dance, counted from the new bar
                self._start(dict(self._voice, origin=best))
                return
        position = (count - self._bar) % SECTION_BEATS
        jump = self._energy_fast - self._energy_slow
        if first:
            self._start(self._new_voice(self._pick()))
        elif position == 0:
            if self.rng.random() < 0.75 or abs(jump) > 0.2:
                self._start(self._new_voice(self._pick(avoid=self._voice["move"])))
            else:
                self._start(self._new_voice(self._voice["move"]))
        elif position == PHRASE_BEATS:
            if abs(jump) > 0.25:  # the music got much louder or softer
                self._start(self._new_voice(self._pick(avoid=self._voice["move"])))
            elif self.rng.random() < 0.6:
                self._start(self._new_voice(self._voice["move"]))
        elif position % BAR_BEATS == 0 and self.rng.random() < 0.3:
            self._start(self._mirrored(self._voice))

    def _plan_section(self, g: int) -> dict | None:
        """The map's section that dance beat g is in (the current one or the next)."""
        for s in (self._plan["section"], self._plan["next"]) if self._plan else ():
            if s and int(s["start"]) <= g < int(s["end"]):
                return s
        return None

    def _plan_beat(self, n: int, first: bool) -> None:
        """At flywheel beat n with the song's map: sections, phrases, mirrors and fills."""
        g = n + self._offset
        s = self._plan_section(g)
        if s is None:  # past the song's end (the level is 0: the dance fades) or a gap in the map
            return
        pos, length = g - int(s["start"]), int(s["end"]) - int(s["start"])
        origin = n - pos  # the flywheel's count on the section's first beat
        if self._section is None or self._section["start"] != s["start"]:
            if self._section is None and not first and pos % PLAN_ENTRY:
                return  # the map came part way through: the live dance goes on until a phrase starts
            entering = self._section is not None
            previous = self._section
            self._section = s
            key = (s["label"], int(s.get("variant") or 0))
            memo = self._memory.get(key)
            if memo is None:
                memo = self._memory[key] = {"move": self._pick_plan(s, self._voice["move"] if self._voice else None),
                                            "phrases": []}
            self._start(self._plan_voice(memo, s, origin, pos // PLAN_PHRASE))
            if entering and pos == 0:  # the section's first beat lands harder, more so into a louder one
                rise = max(0.0, float(s["energy"]) - float(previous["energy"]))
                self._accents[n] = self.rng.uniform(1.2, 1.35) + 0.8 * rise
            return
        if pos % PLAN_PHRASE == 0:
            memo = self._memory[(s["label"], int(s.get("variant") or 0))]
            self._start(self._plan_voice(memo, s, origin, pos // PLAN_PHRASE))
        elif pos % PLAN_MIRROR == 0 and self._voice["params"].get("flip"):
            self._start(self._mirrored(self._voice))
        if pos == length - 2:
            # Two beats before the end, the map knows what comes next: a fill on the last beat.
            nxt = self._plan["next"]
            last = origin + length - 1
            if nxt is not None and int(nxt["start"]) == int(s["end"]):
                jump = float(nxt["energy"]) - float(s["energy"])
                if nxt["label"] != s["label"] or self._voice["params"]["fill"]:
                    size = 0.5 if nxt["kind"] == "break" else _clamp(0.6 + 1.5 * jump, 0.4, 1.3)
                    self._voice["fill"] = (last, size)
            elif int(s["end"]) >= self._plan["end"]:
                self._accents[last] = 1.4  # the song's last beat

    def _plan_voice(self, memo: dict, s: dict, origin: int, phrase: int) -> dict:
        """The voice of a section's phrase: as danced the last time this section came, or new."""
        while len(memo["phrases"]) <= phrase:
            memo["phrases"].append(self._new_params(float(s["energy"]), float(s["busy"])))
        return {"move": memo["move"], "params": dict(memo["phrases"][phrase]), "origin": origin,
                "bar_len": self._plan["bar"], "cycle": None, "fill": None}

    def _pick_plan(self, s: dict, avoid: str | None) -> str:
        e, u, bpm = float(s["energy"]), float(s["busy"]), self._features["bpm"]
        if s["kind"] in ("intro", "outro", "break"):
            weights = {"chill": 2.0, "sway": 1.0, "roll": 0.3, "point": 0.2}
        else:
            weights = {
                "sway": 0.8,
                "chill": 0.1 + 1.5 * (1 - e),
                "roll": 0.6,
                "pump": 0.2 + 2.0 * e * e,
                "open": 0.2 + 1.2 * e,
                "point": 0.5,
                "bounce": 0.2 + 2.0 * u * e if bpm <= 100 else 0.1,
            }
        if avoid in weights:
            weights[avoid] *= 0.2
        return self.rng.choices(list(weights), weights=list(weights.values()))[0]

    def _mirrored(self, voice: dict) -> dict:
        return dict(voice, params=dict(voice["params"], mirror=-voice["params"]["mirror"]))

    def _pick(self, avoid: str | None = None) -> str:
        f = self._features
        e, c, u, bpm = self._energy_slow, f["confidence"], f["busy"], f["bpm"]
        weights = {
            "sway": 1.0,
            "chill": 0.3 + 2.0 * (1 - e) * (1 - c) + (0.6 if bpm < 75 else 0.0),
            "roll": 0.5 + 0.6 * c,
            "pump": 0.2 + 1.6 * e * c,
            "open": 0.2 + 1.0 * e,
            "point": 0.2 + 0.9 * c * (0.5 + f["bar_confidence"]),
            "bounce": 0.2 + 1.6 * u * e if bpm <= 100 else 0.1,
        }
        if self._analysing:  # the map is on its way: nothing big yet
            weights = {"sway": 1.0, "chill": 1.0}
        if avoid in weights:
            weights[avoid] *= 0.2
        return self.rng.choices(list(weights), weights=list(weights.values()))[0]

    def _new_voice(self, move: str) -> dict:
        return {"move": move, "params": self._new_params(self._energy_slow, self._features["busy"]),
                "origin": self._bar, "bar_len": BAR_BEATS, "cycle": SECTION_BEATS, "fill": None}

    def _new_params(self, energy: float, busy: float) -> dict:
        rng = self.rng
        return {
            "amp": rng.uniform(0.75, 1.25),
            "mirror": rng.choice((-1.0, 1.0)),
            "theta": rng.uniform(0.0, math.pi / 2),       # 0 front-back, pi/2 sideways
            "lag": rng.uniform(0.0, 0.25),                # the left arm behind the right [beats]
            "asym": rng.uniform(0.75, 1.15),              # the left arm's size against the right's
            "head": rng.choice((1.0, 1.0, -1.0, 0.3)),    # follows, counters, or mostly nods
            "elbow_lag": rng.uniform(0.04, 0.15),
            "lift": rng.uniform(-6.0, 16.0),              # arms held this much forward [deg]
            "spread": rng.uniform(0.0, 12.0),             # and this much out
            "double": self._features["bpm"] <= 90.0 and rng.random() < 0.3 + 0.6 * busy,
            "fill": rng.random() < 0.3 + 0.5 * energy,
            "flip": rng.random() < 0.5,                   # mirrored half way through the phrase
        }

    def _start(self, voice: dict) -> None:
        self._previous, self._voice = self._voice, voice
        self._blend_from = self._beats

    def _accent(self, b: float, voice: dict) -> float:
        """The beat nearest b, a little harder or softer; the bar's first harder still."""
        nearest = round(b)
        bar = (nearest - voice["origin"]) % voice["bar_len"] == 0
        return self._accents.get(nearest, 1.0) * (1.25 if bar else 1.0)

    def _voice_pose(self, voice: dict, wall: float) -> dict[str, float]:
        p = dict(voice["params"])
        d = self._drift
        size = p["amp"] * (1.0 + 0.18 * d["amp"](wall))
        p["theta"] = _clamp(p["theta"] + 0.35 * d["theta"](wall), 0.0, math.pi / 2)
        A = {key: value * size for key, value in GROOVE_DEG.items()}
        b = self._beats - voice["origin"]
        if voice["cycle"]:
            b %= voice["cycle"]
        out = GROOVE_MOVES[voice["move"]](b, self._accent(self._beats, voice), p, A)
        fill = None
        if voice["cycle"] and p["fill"]:
            fill = _fill(b - (voice["cycle"] - 1), A)
        elif voice["fill"]:
            fill = _fill(self._beats - voice["fill"][0], A, voice["fill"][1])
        for name, value in (fill or {}).items():
            out[name] = out.get(name, 0.0) + value
        out["rp"] = out.get("rp", 0.0) + p["lift"] + 5.0 * d["lift"](wall)
        out["lp"] = out.get("lp", 0.0) + p["lift"] + 5.0 * d["lift"](wall + 7.0)
        out["rr"] = out.get("rr", 0.0) + p["spread"] + 4.0 * (0.5 + 0.5 * d["spread"](wall))
        out["lr"] = out.get("lr", 0.0) + p["spread"] + 4.0 * (0.5 + 0.5 * d["spread"](wall + 5.0))
        return out

    def _dance(self, wall: float) -> dict[str, float]:
        pose = self._voice_pose(self._voice, wall)
        x = _clamp((self._beats - self._blend_from) / GROOVE_BLEND_BEATS, 0.0, 1.0)
        if self._previous is not None and x < 1.0:
            w = x * x * (3.0 - 2.0 * x)
            old = self._voice_pose(self._previous, wall)
            pose = {name: old.get(name, 0.0) + (pose.get(name, 0.0) - old.get(name, 0.0)) * w for name in POSE_KEYS}
        return pose


@dataclass
class UpperBodyInput:
    manual: dict[str, float] = field(default_factory=dict)  # authoring pose overrides [deg]
    auto: bool = True  # small gestures now and then, unasked
    groove: dict | None = None  # a music beat grid to sway to (Groove)
    wall: float = 0.0  # time.time(), the clock of the beat grid


class UpperBody:
    """Gestures, manual targets and the alive layer, mixed into one authoring pose per tick."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)
        self.alive = Alive(self.rng)
        # Its own generator: the gestures play out the same for a seed with or without music.
        self.groove = Groove(random.Random(None if seed is None else seed + 1))
        self._performance: Performance | None = None
        self._started = 0.0
        self._now = 0.0
        self._output: dict[str, float] | None = None
        self._offsets = {name: 0.0 for name in POSE_KEYS}
        self._life = 1.0
        self._next_auto: float | None = None
        self._manual: dict[str, list[float]] = {}

    @property
    def playing(self) -> str | None:
        return self._performance.gesture.name if self._performance else None

    @property
    def face(self) -> str | None:
        """The expression the playing gesture asks for now, if any (the API shows it)."""
        if self._performance is None:
            return None
        return self._performance.face(self._now - self._started + FACE_LEAD_S)

    def play(self, name: str, now: float, size: float = 1.0, keys: list[dict] | None = None, exact: bool = False) -> bool:
        """A gesture by name ("random" for any), or CUSTOM with its `keys` (custom_keys); `exact`
        plays CUSTOM keys as taught."""
        if name == "random":
            name = self.rng.choice([n for n in GESTURE_NAMES if n != self.playing])
        gesture = custom_gesture(keys or [], exact) if name == CUSTOM else GESTURES.get(name)
        if gesture is None or not gesture.steps:
            return False
        self._performance = Performance(gesture, self._settled(), self.rng, size)
        self._started = now
        return True

    def command(self, cmd: dict, now: float) -> None:
        """A gesture request from the API (motion.py's upper command): stop, or play."""
        name = cmd.get("gesture")
        if name == "stop":
            self.stop()
        elif name:
            self.play(name, now, _clamp(float(cmd.get("size") or 1.0), 0.3, 1.0), cmd.get("keys"), bool(cmd.get("exact")))

    def stop(self) -> None:
        if self._performance is not None and self._output is not None:
            # Settle from wherever the gesture got to.
            self._performance = Performance(G("_settle", "", back=0.6), self._settled(), self.rng)

    def _settled(self) -> dict[str, float]:
        """The last output without the alive layer, which is added again on top."""
        if self._output is None:
            return dict(REST)
        return {name: self._output[name] - self._offsets[name] for name in POSE_KEYS}

    def _follow_manual(self, dt: float, base: dict[str, float], manual: dict[str, float]) -> dict[str, float]:
        """Manual targets (and, once released, the base pose again) reached through two cascaded
        low-passes: slider steps arrive a few times a second and come out as one smooth move."""
        k = 1.0 - math.exp(-dt / MANUAL_TAU_S)
        held = self._settled()
        for name in manual:
            if name not in self._manual:
                self._manual[name] = [held[name], held[name]]
        for name in list(self._manual):
            target = manual.get(name, base[name])
            stage = self._manual[name]
            stage[0] += (target - stage[0]) * k
            stage[1] += (stage[0] - stage[1]) * k
            if name not in manual and abs(stage[1] - target) < 0.3:
                del self._manual[name]
        return {name: stage[1] for name, stage in self._manual.items()}

    def update(self, now: float, dt: float, base: dict[str, float], command: UpperBodyInput) -> dict[str, float]:
        self._now = now
        base = {**base, **self._follow_manual(dt, base, command.manual)}
        performance = self._performance
        if performance is not None and performance.done(now - self._started):
            self._performance = performance = None
        grooving = self.groove.weight > 0.3
        if command.auto and performance is None and not command.manual and not grooving:
            if self._next_auto is None:
                self._next_auto = now + self._auto_gap()
            elif now >= self._next_auto:
                self.play(self.rng.choice(IDLE_GESTURES), now, size=self.rng.uniform(0.65, 0.85))
                performance = self._performance
                self._next_auto = None
        elif performance is not None or command.manual:
            self._next_auto = None
        pose = performance.pose(now - self._started, base) if performance is not None else dict(base)
        busy = performance is not None or bool(command.manual)
        want_life = 0.35 if busy else 0.5 if grooving else 1.0
        self._life += (want_life - self._life) * (1.0 - math.exp(-dt / 0.4))
        self._offsets = self.alive.offsets(now, dt, self._life)
        # Gestures and held poses keep a little of the sway, under them.
        groove = command.groove and {**command.groove, "size": float(command.groove.get("size") or 0.0) * (0.3 if busy else 1.0)}
        for name, value in self.groove.offsets(command.wall, dt, groove).items():
            self._offsets[name] += value
        for name in POSE_KEYS:
            value = _clamp(pose[name] + self._offsets[name], *POSE_LIMITS[name])
            if self._output is not None:
                step = (PITCH_MAX_SPEED_DEG_S if name in PITCH_KEYS else MAX_SPEED_DEG_S) * dt
                value = _clamp(value, self._output[name] - step, self._output[name] + step)
            pose[name] = value
        self._output = pose
        return pose

    def _auto_gap(self) -> float:
        return _clamp(math.exp(self.rng.gauss(math.log(15.0), 0.45)), 6.0, 45.0)


# Head idle under the plain walk (moves/walk.py): authoring yaw / tilt [deg] it looks around in.
IDLE_YAW_DEG = 34.0
IDLE_TILT_DEG = (-14.0, 7.0)
IDLE_START_S = 0.8  # standing still this long before it starts looking around
IDLE_MAX_SPEED_DEG_S = 160.0
# HeadIdle aimed by a camera-guided source (the app's follow mode models this same move to know
# where the camera looks: FollowLogic.kt HEAD_OMEGA / HEAD_MAX_SPEED / HEAD_MAX).
AIM_YAW_DEG = 38.0
AIM_OMEGA = 7.0


class HeadIdle:
    """The head standing still, as a person's would: quick glances that hold, slow turns, a look
    down or up now and then, long stretches of staring ahead, back to the front often. Each look
    is a critically damped move (stiffness by kind), with a faint wander on top. Moving makes it
    look ahead again."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)
        self._pos = [0.0, 0.0]
        self._vel = [0.0, 0.0]
        self._target = [0.0, 0.0]
        self._omega = 3.0
        self._next = 0.0
        self._still_since: float | None = None
        self._wander = (_Noise(self.rng, 0.12), _Noise(self.rng, 0.1))

    def reset(self, yaw: float, tilt: float) -> None:
        """Take over from wherever the head is [deg]."""
        self._pos, self._vel = [yaw, tilt], [0.0, 0.0]
        self._target = [yaw, tilt]
        self._next = 0.0
        self._still_since = None

    def update(self, now: float, dt: float, still: bool, aim: float | None = None) -> dict[str, float]:
        """aim [deg, + left]: look there instead (a camera-guided source tracking someone), no wander."""
        limit = IDLE_YAW_DEG
        if aim is not None:
            limit = AIM_YAW_DEG
            self._still_since = None
            self._target, self._omega = [_clamp(aim, -limit, limit), 0.0], AIM_OMEGA
            still = False
        elif not still:
            self._still_since = None
            self._target, self._omega = [0.0, 0.0], 4.0
        else:
            if self._still_since is None:
                self._still_since = now
                self._next = now + IDLE_START_S + self.rng.uniform(0.0, 1.5)
            if now >= self._next:
                self._pick(now)
        for i in range(2):
            acc = self._omega * self._omega * (self._target[i] - self._pos[i]) - 2.0 * self._omega * self._vel[i]
            self._vel[i] = _clamp(self._vel[i] + acc * dt, -IDLE_MAX_SPEED_DEG_S, IDLE_MAX_SPEED_DEG_S)
            self._pos[i] += self._vel[i] * dt
        live = 0.0 if aim is not None else 1.0 if still else 0.3
        yaw = self._pos[0] + live * 1.2 * self._wander[0](now)
        tilt = self._pos[1] + live * 0.8 * self._wander[1](now)
        return {"yaw": _clamp(yaw, -limit, limit), "tilt": _clamp(tilt, *IDLE_TILT_DEG)}

    def _pick(self, now: float) -> None:
        rng = self.rng
        yaw, tilt = self._pos
        kind = rng.choices(("stare", "glance", "slow", "front", "down", "up"), (22, 30, 20, 16, 7, 5))[0]
        if kind == "stare":
            hold = rng.uniform(2.0, 6.0)
        elif kind == "glance":
            # A quick look that lands somewhere new, often to the other side.
            side = -1.0 if yaw > 8.0 else 1.0 if yaw < -8.0 else rng.choice((-1.0, 1.0))
            yaw = side * rng.uniform(10.0, IDLE_YAW_DEG) if rng.random() < 0.7 else rng.gauss(0.0, 12.0)
            tilt = rng.gauss(-2.0, 3.5)
            self._omega, hold = rng.uniform(7.0, 11.0), _clamp(math.exp(rng.gauss(math.log(1.6), 0.5)), 0.6, 5.0)
        elif kind == "slow":
            yaw = _clamp(yaw + rng.choice((-1.0, 1.0)) * rng.uniform(12.0, 40.0), -IDLE_YAW_DEG, IDLE_YAW_DEG)
            tilt = rng.gauss(-1.5, 3.0)
            self._omega, hold = rng.uniform(1.4, 2.6), rng.uniform(1.5, 4.0)
        elif kind == "front":
            yaw, tilt = rng.gauss(0.0, 2.5), rng.gauss(0.0, 1.5)
            self._omega, hold = rng.uniform(3.0, 7.0), rng.uniform(1.0, 4.0)
        elif kind == "down":
            yaw, tilt = rng.gauss(0.0, 10.0), rng.uniform(-14.0, -8.0)
            self._omega, hold = rng.uniform(3.0, 6.0), rng.uniform(0.8, 2.5)
        else:
            yaw, tilt = rng.gauss(0.0, 10.0), rng.uniform(3.0, 7.0)
            self._omega, hold = rng.uniform(3.0, 6.0), rng.uniform(0.8, 2.0)
        self._target = [_clamp(yaw, -IDLE_YAW_DEG, IDLE_YAW_DEG), _clamp(tilt, *IDLE_TILT_DEG)]
        self._next = now + hold


# Hugging a leg (the app's 跟屁虫 once its camera is covered), authoring space [deg]. The arm
# cannot cross inward and the elbow bends only toward where the shoulder points it (up, with the
# arm held forward), so the forearms close on the two sides of the leg, about 12 cm apart at
# shoulder roll 0 (a calf's width), and press it from there: a fixed squeeze inward of the roll
# the leg holds them at, so a thick leg does not strain the motors more than a thin one.
HUG_PITCH = (66.0, 80.0)       # rp / lp: arms held forward, a little under level
HUG_ELBOW = (26.0, 42.0)       # re / le: forearms curled up round the leg
HUG_OPEN_ROLL = (24.0, 32.0)   # rr / lr while reaching: wide, so the arms come round the leg, not into it
HUG_OPEN_S = (0.5, 0.8)        # at least this long open, and until the arms are forward
HUG_SQUEEZE_DEG = 6.0          # rr / lr: this far inward of where the leg holds the arm
HUG_TIGHTER_DEG = (6.0, 12.0)  # now and then a tighter squeeze, the elbows too
HUG_TIGHTER_GAP_S = (0.8, 2.5)
HUG_STROKE_DEG = (4.0, 9.0)    # rp / lp while nuzzling: the arms rub up and down the leg in turn
HUG_STROKE_HZ = (1.2, 2.0)
HUG_BOUNCE_DEG = (3.0, 6.0)    # rp / lp while looking up: both arms bounce together
HUG_BOUNCE_HZ = (2.5, 3.5)
HUG_YAW_DEG = (22.0, 40.0)     # nuzzling side to side
HUG_WIGGLE_DEG = (6.0, 12.0)   # quick little rubs on one side
# On the robot motor 20 looks up for positive angles, against HEAD_PITCH_UP (which the other
# moves keep): the hug's tilts are degrees up, turned into authoring tilt by this sign.
HUG_TILT_SIGN = -1.0
HUG_DOWN_TILT = (-10.0, -2.0)  # the head 10 deg down at most
HUG_UP_DEG = 30.0              # and all the way up, 30
HUG_ARM_SPEED_DEG_S = 220.0
HUG_HEAD_SPEED_DEG_S = 300.0


class Hug:
    """Holding on to a leg and not letting go, never twice the same: the arms open, come round
    and squeeze (tighter now and then), while the head nuzzles left and right a random number of
    times, at random angles, pace and pauses, with quick little rubs on a side, the arms rubbing
    up and down the leg in turn; then it looks up at the person for a while, the arms bouncing,
    and again. `phase` ("reach", "nuzzle", "look_up") and `cycles` let the API pick a face."""

    def __init__(self, start: dict[str, float], seed: int | None = None) -> None:
        rng = self.rng = random.Random(seed)
        self._pos = {key: float(start.get(key, REST[key])) for key in POSE_KEYS}
        self._vel = dict.fromkeys(POSE_KEYS, 0.0)
        self._omega = dict.fromkeys(POSE_KEYS, 5.0)
        # The two arms never quite alike.
        rp, re = rng.uniform(*HUG_PITCH), rng.uniform(*HUG_ELBOW)
        self._arms = {"rp": rp, "lp": rp + rng.gauss(0.0, 2.0), "re": re, "le": re + rng.gauss(0.0, 2.5)}
        self._open = rng.uniform(*HUG_OPEN_ROLL)
        self._target = {**self._pos, **self._arms, "rr": self._open, "lr": self._open + rng.gauss(0.0, 2.0),
                        "yaw": rng.gauss(0.0, 4.0), "tilt": HUG_TILT_SIGN * rng.uniform(-8.0, -3.0)}
        for key in ("rp", "lp", "re", "le", "rr", "lr"):
            self._omega[key] = rng.uniform(7.0, 10.0)
        self.phase = "reach"
        self.cycles = 0
        self._t0: float | None = None
        self._closing = 0.0
        self._looks: list[tuple[float, float, float, float]] = []
        self._next = 0.0
        self._tight = 0.0
        self._tight_until = 0.0
        self._next_tight = 0.0
        # The arms' rhythm on top of the hold: (amplitude [deg], rate [Hz], left arm's phase lag),
        # eased toward as the phase changes.
        self._rhythm = (0.0, 1.5, math.pi)
        self._amp, self._hz, self._lag, self._beat = 0.0, 1.5, math.pi, 0.0
        self._wander = (_Noise(rng, 0.15), _Noise(rng, 0.12), _Noise(rng, 0.25))

    def update(self, now: float, dt: float, roll: tuple[float, float] | None = None) -> dict[str, float]:
        """The pose this tick; roll: the measured (rr, lr), for the squeeze."""
        rng = self.rng
        if self._t0 is None:
            self._t0 = now
            self._closing = now + rng.uniform(*HUG_OPEN_S)
        forward = all(abs(self._pos[key] - self._arms[key]) < 8.0 for key in ("rp", "lp"))
        if self.phase == "reach" and now >= self._closing and (forward or now - self._t0 > 3.0):
            self._target["rr"] = self._target["lr"] = 0.0
            self._next_tight = now + rng.uniform(*HUG_TIGHTER_GAP_S)
            self._nuzzle(now)
        if self.phase != "reach":
            if now >= self._next:
                self._look(now)
            if now >= self._next_tight:
                self._tight = rng.uniform(*HUG_TIGHTER_DEG)
                self._tight_until = now + rng.uniform(0.3, 0.7)
                self._next_tight = now + rng.uniform(*HUG_TIGHTER_GAP_S)
            elif now >= self._tight_until:
                self._tight = 0.0
            for key in ("re", "le"):
                self._target[key] = self._arms[key] + self._tight
        for key in POSE_KEYS:
            omega = self._omega[key]
            acc = omega * omega * (self._target[key] - self._pos[key]) - 2.0 * omega * self._vel[key]
            top = (HUG_HEAD_SPEED_DEG_S if key in ("yaw", "tilt") else
                   min(HUG_ARM_SPEED_DEG_S, PITCH_MAX_SPEED_DEG_S) if key in PITCH_KEYS else HUG_ARM_SPEED_DEG_S)
            self._vel[key] = _clamp(self._vel[key] + acc * dt, -top, top)
            self._pos[key] = _clamp(self._pos[key] + self._vel[key] * dt, *POSE_LIMITS[key])
        pose = dict(self._pos)
        if self.phase != "reach" and roll is not None:
            squeeze = HUG_SQUEEZE_DEG + self._tight
            for key, measured in zip(("rr", "lr"), roll):
                held = max(POSE_LIMITS[key][0], measured - squeeze)
                if pose[key] < held:
                    pose[key] = self._pos[key] = held
                    self._vel[key] = 0.0
        # The rhythm, at most ~130 deg/s on the shoulders (HUG_BOUNCE at its fastest).
        amp, hz, lag = self._rhythm if self.phase != "reach" else (0.0, self._hz, self._lag)
        ease = min(1.0, dt / 0.3)
        self._amp += (amp - self._amp) * ease
        self._hz += (hz - self._hz) * ease
        self._lag += (lag - self._lag) * ease
        self._beat += 2.0 * math.pi * self._hz * dt
        for (pitch, elbow), at in ((("rp", "re"), self._beat), (("lp", "le"), self._beat + self._lag)):
            pose[pitch] += self._amp * math.sin(at)
            pose[elbow] += 0.5 * self._amp * math.cos(at)
        breath = 1.5 * self._wander[2](now)
        pose["rp"] += breath
        pose["lp"] += breath
        pose["yaw"] = pose["yaw"] + 1.2 * self._wander[0](now)
        pose["tilt"] = pose["tilt"] + 0.8 * self._wander[1](now)
        return {key: _clamp(value, *POSE_LIMITS[key]) for key, value in pose.items()}

    def _nuzzle(self, now: float) -> None:
        """A new round of looks to the sides: 2-5 times left and right, now and then a few quick
        little rubs on a side or a pause in between; the arms rub the leg in turn meanwhile."""
        rng = self.rng
        self.phase = "nuzzle"
        self._rhythm = (rng.uniform(*HUG_STROKE_DEG), rng.uniform(*HUG_STROKE_HZ), math.pi)
        back_and_forth = rng.choices((2, 3, 4, 5), (2, 4, 3, 2))[0]
        side = rng.choice((-1.0, 1.0))
        looks = []
        for _ in range(2 * back_and_forth):
            if looks and rng.random() < 0.1:
                looks.append((rng.gauss(0.0, 4.0), rng.uniform(*HUG_DOWN_TILT), rng.uniform(4.0, 6.0), rng.uniform(0.3, 0.8)))
            yaw, tilt = side * rng.uniform(*HUG_YAW_DEG), rng.uniform(*HUG_DOWN_TILT)
            hold = _clamp(math.exp(rng.gauss(math.log(0.3), 0.4)), 0.1, 0.8)
            looks.append((yaw, tilt, rng.uniform(8.0, 13.0), hold))
            if rng.random() < 0.35:
                for k in range(2 * rng.randint(1, 3)):
                    back = side * rng.uniform(*HUG_WIGGLE_DEG) if k % 2 == 0 else 0.0
                    looks.append((yaw - back, tilt + rng.gauss(0.0, 2.0), rng.uniform(14.0, 18.0), rng.uniform(0.02, 0.08)))
            side = -side
        self._looks = looks
        self._next = now

    def _look_up(self) -> None:
        """Up at the person, the arms bouncing, sometimes a happy little wag of the head."""
        rng = self.rng
        self.phase = "look_up"
        self._rhythm = (rng.uniform(*HUG_BOUNCE_DEG), rng.uniform(*HUG_BOUNCE_HZ), 0.0)
        yaw = rng.gauss(0.0, 5.0)
        looks = [(yaw, HUG_UP_DEG, rng.uniform(5.0, 8.0), rng.uniform(0.6, 1.2))]
        if rng.random() < 0.6:
            for k in range(2 * rng.randint(1, 2)):
                wag = rng.uniform(5.0, 9.0) * (1.0 if k % 2 == 0 else -1.0)
                looks.append((yaw + wag, HUG_UP_DEG, rng.uniform(10.0, 14.0), rng.uniform(0.05, 0.15)))
        looks.append((yaw, HUG_UP_DEG, rng.uniform(6.0, 9.0), rng.uniform(0.8, 1.8)))
        self._looks = looks

    def _look(self, now: float) -> None:
        rng = self.rng
        if not self._looks:
            if self.phase == "nuzzle":
                self._look_up()
            else:
                self.cycles += 1
                self._nuzzle(now)
        yaw, tilt, omega, hold = self._looks.pop(0)
        self._target["yaw"] = _clamp(yaw, *POSE_LIMITS["yaw"])
        self._target["tilt"] = _clamp(HUG_TILT_SIGN * tilt, *POSE_LIMITS["tilt"])
        self._omega["yaw"] = self._omega["tilt"] = omega
        # Long enough to get there (a critically damped move is ~95% there after 4.7 / omega), then hold.
        self._next = now + 4.7 / omega * rng.uniform(0.6, 1.0) + hold
