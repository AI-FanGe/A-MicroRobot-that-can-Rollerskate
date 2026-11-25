# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

"""The robot's 87 expressions for the 64x48 LED dot display, and the LED look of a frame.

A single-colour line-drawn face: single-stroke eyes (upright strokes, arcs, dashes), brows,
a mouth and small effects such as tears or floating hearts. Each expression is a `Look`
(face_engine.py) plus an optional animation of its own (laughing, nodding off, yawning...).
`face_engine.FaceAnimator` brings a look to life; the display loop (display_smile.py) asks it
for a frame 25 times a second.
"""

from __future__ import annotations

import base64
import json
import math
import random
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import numpy as np

from face_engine import (
    COLS,
    PALETTE,
    ROWS,
    BrowSpec,
    ExtraSpec,
    EyeSpec,
    FaceAnimator,
    Look,
    MotionSpec,
    MouthSpec,
    Pose,
    _hash,
    smooth,
)

MATRIX_COLS = COLS
MATRIX_ROWS = ROWS
EXPRESSION_FRAME_DT_S = 0.04
EXPRESSION_STATE_PATH = Path("/tmp/xiaoxiaobai_expression.json")
# "idle" is not a face of its own: the display wanders through IDLE_LOOKS (IdleCycle).
IDLE_EXPRESSION = "idle"
IDLE_LOOKS = ("calm", "smile", "smug", "curious", "thinking", "focused", "doubtful")
IDLE_HOLD_S = (5.0, 12.0)
DEFAULT_EXPRESSION = IDLE_EXPRESSION
# Web previews: a few seconds of a seeded animator (blinks and glances included).
PREVIEW_FRAMES = 40
PREVIEW_DT_S = 0.1
LIT_THRESHOLD = 12  # brightness (of 31) that counts as lit for the 1-bit pixel lists


# ---------------------------------------------------------------- building blocks


def eye(kind: str = "stroke", **kw) -> EyeSpec:
    """`stroke` (the default) is an upright single line; `ry` is its half-length."""
    if "r" in kw:
        kw["rx"] = kw["ry"] = kw.pop("r")
    if kind == "stroke":
        kw.setdefault("thick", 0.72)
    return EyeSpec(kind=kind, **kw)


def brows(**kw) -> tuple[BrowSpec, BrowSpec]:
    spec = BrowSpec(**kw)
    return spec, spec


def lips(**kw) -> MouthSpec:
    return MouthSpec(kind="lips", **kw)


def fx(*names: str, **kw) -> tuple[ExtraSpec, ...]:
    return tuple(ExtraSpec(name, **kw) for name in names)


def look(eyes: EyeSpec, mouth: MouthSpec, *, eye_r: EyeSpec | None = None,
         brow: tuple[BrowSpec, BrowSpec] | None = None, extras: tuple[ExtraSpec, ...] = (),
         motion: MotionSpec = MotionSpec(), anim=None) -> Look:
    return Look(eyes, eye_r or eyes, brow[0] if brow else None, brow[1] if brow else None, mouth, extras, motion, anim)


def _set_mouth(pose: Pose, **kw) -> None:
    if pose.mouth:
        pose.mouth[0].spec = replace(pose.mouth[0].spec, **kw)


def _set_eyes(pose: Pose, fn) -> None:
    for layer in pose.eye_l + pose.eye_r:
        layer.spec = fn(layer.spec)


# ---------------------------------------------------------------- animations of their own


def anim_laugh(pose: Pose, t: float, age: float, a: FaceAnimator) -> None:
    ha = abs(math.sin(2 * math.pi * 2.3 * t))  # "ha ha ha"
    _set_mouth(pose, open=4.8 + 2.6 * ha)
    pose.oy -= 0.35 * ha


def anim_surprise(pose: Pose, t: float, age: float, a: FaceAnimator) -> None:
    pop = math.exp(-age * 5.0)
    _set_eyes(pose, lambda s: replace(s, rx=s.rx * (1 + 0.28 * pop), ry=s.ry * (1 + 0.32 * pop)))
    if pose.mouth:
        _set_mouth(pose, open=pose.mouth[0].spec.open * (1 + 0.4 * pop))
    pose.brow_raise += 1.6 * pop


def anim_shocked(pose: Pose, t: float, age: float, a: FaceAnimator) -> None:
    jolt = math.exp(-age * 2.5)
    pose.ox += 0.7 * jolt * math.sin(t * 45.0)
    pose.oy -= 1.2 * math.exp(-age * 6.0)
    pose.brow_raise += 1.2 * jolt


def anim_crying(pose: Pose, t: float, age: float, a: FaceAnimator) -> None:
    sob = max(0.0, math.sin(2 * math.pi * 1.4 * t)) ** 4
    pose.oy += 0.55 * sob
    _set_mouth(pose, open=2.6 + 1.6 * sob + 0.3 * math.sin(t * 13.0))


def anim_tremble(pose: Pose, t: float, age: float, a: FaceAnimator) -> None:
    if not pose.mouth:
        return
    spec = pose.mouth[0].spec
    _set_mouth(pose, y=spec.y + 0.16 * math.sin(2 * math.pi * 8.5 * t), curve=spec.curve + 0.35 * math.sin(2 * math.pi * 3.7 * t))


def anim_sleepy(pose: Pose, t: float, age: float, a: FaceAnimator) -> None:
    cycle = int(t / 4.5)
    period = 3.5 + 2.5 * _hash(cycle, 3)
    phase = (t - cycle * 4.5) / period
    if phase < 0.82:  # lids sink, the head nods off...
        droop = smooth(phase / 0.82)
    elif phase < 1.0:  # ...and snaps back up
        droop = 1.0 - smooth((phase - 0.82) / 0.18)
    else:
        droop = 0.0
    _set_eyes(pose, lambda s: replace(s, lid_top=min(0.97, s.lid_top + 0.47 * droop)))
    pose.oy += 1.4 * droop


def anim_sleeping(pose: Pose, t: float, age: float, a: FaceAnimator) -> None:
    breath = math.sin(a._breath_phase)
    cycle = int(t / 15.0)
    local = t - cycle * 15.0
    yawn = 0.0
    if _hash(cycle, 9) < 0.55 and local < 2.6:
        p = local / 2.6
        yawn = smooth(p / 0.2) if p < 0.2 else (1.0 if p < 0.6 else 1.0 - smooth((p - 0.6) / 0.4))
    _set_mouth(pose, open=0.9 + 0.5 * breath + 6.5 * yawn, w=1.7 + 3.0 * yawn)
    _set_eyes(pose, lambda s: replace(s, ry=s.ry * (1 - 0.45 * yawn), dy=s.dy + 0.6 * yawn))


def anim_singing(pose: Pose, t: float, age: float, a: FaceAnimator) -> None:
    slot = int(t / 0.3)
    local = (t / 0.3) % 1.0
    note = 0.35 + 0.65 * _hash(slot, 5)
    _set_mouth(pose, open=1.2 + 5.0 * note * math.sin(math.pi * local) ** 0.6, w=4.5 + 2.0 * _hash(slot, 6))
    pose.ox += 1.3 * math.sin(2 * math.pi * 0.7 * t)
    pose.oy += 0.4 * abs(math.sin(2 * math.pi * 0.7 * t))


def anim_yummy(pose: Pose, t: float, age: float, a: FaceAnimator) -> None:
    _set_mouth(pose, tongue_x=2.6 * math.sin(2 * math.pi * 1.1 * t))


def anim_greedy(pose: Pose, t: float, age: float, a: FaceAnimator) -> None:
    lick = max(0.0, math.sin(2 * math.pi * t / 3.5)) ** 3
    _set_mouth(pose, tongue=0.8 * lick, tongue_x=1.5 * math.sin(t * 6.0))


def anim_annoyed(pose: Pose, t: float, age: float, a: FaceAnimator) -> None:
    phase = (t / 6.5) % 1.0
    if phase < 0.22:  # eye roll
        p = phase / 0.22
        angle = math.pi * (1.0 - smooth(p))
        weight = math.sin(math.pi * p)
        pose.gaze_override = (2.0 * math.cos(angle), -1.6 * math.sin(angle), weight)


def anim_bored(pose: Pose, t: float, age: float, a: FaceAnimator) -> None:
    phase = (t / 7.0) % 1.0
    sigh = math.sin(math.pi * phase / 0.3) if phase < 0.3 else 0.0
    pose.oy += 0.9 * sigh
    _set_mouth(pose, open=1.4 * sigh)


def anim_dizzy(pose: Pose, t: float, age: float, a: FaceAnimator) -> None:
    pose.ox += 1.4 * math.cos(2 * math.pi * 0.55 * t)
    pose.oy += 0.8 * math.sin(2 * math.pi * 0.55 * t)


def anim_relieved(pose: Pose, t: float, age: float, a: FaceAnimator) -> None:
    pose.oy += 0.9 * smooth(age / 1.4)  # the shoulders drop


def anim_thinking(pose: Pose, t: float, age: float, a: FaceAnimator) -> None:
    side = 1.0 if _hash(int(t / 5.0), 2) < 0.65 else -1.0
    pose.bias_x *= side


def anim_cheer(pose: Pose, t: float, age: float, a: FaceAnimator) -> None:
    _set_mouth(pose, open=3.4 + 2.2 * abs(math.sin(2 * math.pi * 1.6 * t)))


def anim_excited(pose: Pose, t: float, age: float, a: FaceAnimator) -> None:
    _set_mouth(pose, open=4.0 + 1.6 * abs(math.sin(2 * math.pi * 2.0 * t)))


def anim_sideeye(pose: Pose, t: float, age: float, a: FaceAnimator) -> None:
    cycle = int(t / 3.6)
    side = 1.0 if _hash(cycle, 4) < 0.5 else -1.0
    pose.gaze_override = (2.4 * side, 0.2, 0.85)


def _bump(x: float) -> float:
    """0 -> 1 -> 0 over x in [0, 1], 0 outside."""
    return math.sin(math.pi * x) if 0.0 < x < 1.0 else 0.0


def anim_huff(pose: Pose, t: float, age: float, a: FaceAnimator) -> None:
    hmph = math.exp(-(t % 3.2) * 6.0)
    pose.ox -= 2.4  # head turned away, chin up
    pose.oy -= 0.7 + 0.6 * hmph


def anim_clingy(pose: Pose, t: float, age: float, a: FaceAnimator) -> None:
    pose.ox += 1.3 * math.sin(2 * math.pi * 1.2 * t)  # wriggling from side to side
    pose.oy += 0.35 * abs(math.sin(2 * math.pi * 1.2 * t))


def anim_peek(pose: Pose, t: float, age: float, a: FaceAnimator) -> None:
    phase = (t / 5.0) % 1.0
    if phase < 0.15:  # slides out from behind the edge...
        out = smooth(phase / 0.15)
    elif phase < 0.75:
        out = 1.0
    elif phase < 0.85:  # ...and ducks back
        out = 1.0 - smooth((phase - 0.75) / 0.1)
    else:
        out = 0.0
    pose.ox += 18.0 - 10.0 * out
    pose.gaze_override = (-2.4, 0.3, 0.9)


def anim_observe(pose: Pose, t: float, age: float, a: FaceAnimator) -> None:
    pose.gaze_override = (2.3 * math.sin(2 * math.pi * t / 5.5), 0.2, 0.9)  # slowly scanning the room


def anim_ouch(pose: Pose, t: float, age: float, a: FaceAnimator) -> None:
    wince = math.exp(-(t % 2.6) * 4.0)
    pose.ox += 0.8 * wince * math.sin(t * 40.0)
    pose.oy += 0.6 * wince
    _set_eyes(pose, lambda s: replace(s, ry=s.ry * (1.0 - 0.3 * wince)))


def anim_busted(pose: Pose, t: float, age: float, a: FaceAnimator) -> None:
    anim_shocked(pose, t, age, a)
    pose.gaze_override = (-2.4, 0.4, 0.9 * smooth((age - 0.8) / 0.4))  # then looks anywhere but at you


def anim_cute(pose: Pose, t: float, age: float, a: FaceAnimator) -> None:
    tilt = math.sin(2 * math.pi * 0.3 * t)  # head tilted one way, then the other
    pose.ox += 1.6 * tilt
    pose.oy += 0.5 * abs(tilt)


def anim_dancing(pose: Pose, t: float, age: float, a: FaceAnimator) -> None:
    beat = abs(math.sin(2 * math.pi * t))  # 120 bpm
    pose.oy -= 1.3 * beat
    pose.ox += 2.0 * math.sin(math.pi * t)
    _set_mouth(pose, open=2.0 + 2.0 * beat)


def anim_working(pose: Pose, t: float, age: float, a: FaceAnimator) -> None:
    line = (t / 1.4) % 1.0  # reading along a line, then back to the start of the next one
    row = int(t / 1.4) % 3
    x = -2.0 + 4.0 * smooth(line / 0.85) if line < 0.85 else 2.0 - 4.0 * smooth((line - 0.85) / 0.15)
    pose.gaze_override = (x, 0.4 + 0.5 * row, 0.9)


def anim_done(pose: Pose, t: float, age: float, a: FaceAnimator) -> None:
    phase = t % 3.0  # two quick nods
    pose.oy += 1.0 * _bump(phase / 0.35) + 0.7 * _bump((phase - 0.4) / 0.35)


def anim_woken(pose: Pose, t: float, age: float, a: FaceAnimator) -> None:
    if t % 1.8 < 0.9:  # the alarm rings in bursts
        pose.ox += 0.35 * math.sin(t * 50.0)


def anim_waking(pose: Pose, t: float, age: float, a: FaceAnimator) -> None:
    lid = 0.97 * (1.0 - smooth((age - 0.5) / 1.2)) + 0.6 * _bump((age - 1.9) / 0.7)  # nearly nods off again
    _set_eyes(pose, lambda s: replace(s, lid_top=max(s.lid_top, min(0.97, lid))))
    pose.oy += 1.6 * (1.0 - smooth((age - 0.3) / 2.8))


def anim_drowsy(pose: Pose, t: float, age: float, a: FaceAnimator) -> None:
    opening = smooth((age - 0.3) / 0.7)  # the eyes only just came open, half way...
    phase = max(0.0, age - 1.2) / 2.4 % 1.0
    if phase < 0.42:  # ...the lids keep sinking, the head nodding with them...
        sink = smooth(phase / 0.42)
    elif phase < 0.52:  # ...and get dragged back up
        sink = 1.0 - smooth((phase - 0.42) / 0.1)
    else:
        sink = 0.0
    shut = max(1.0 - opening, 0.8 * sink)
    _set_eyes(pose, lambda s: replace(s, lid_top=s.lid_top + (0.95 - s.lid_top) * shut))
    pose.oy += 1.1 * sink
    pose.ox += 0.6 * math.sin(2 * math.pi * 0.2 * t)
    _set_mouth(pose, open=1.2 + 1.0 * sink)


def anim_dashing(pose: Pose, t: float, age: float, a: FaceAnimator) -> None:
    pose.brow_raise += 0.9 * math.exp(-(t % 3.5) * 3.0)  # an eyebrow flash


def anim_yawn(pose: Pose, t: float, age: float, a: FaceAnimator) -> None:
    phase = (t / 5.0) % 1.0
    if phase < 0.2:
        yawn = smooth(phase / 0.2)
    elif phase < 0.45:
        yawn = 1.0
    else:
        yawn = 1.0 - smooth((phase - 0.45) / 0.25)
    _set_mouth(pose, open=1.2 + 6.8 * yawn, w=2.2 + 3.0 * yawn)
    _set_eyes(pose, lambda s: replace(s, lid_top=min(0.95, s.lid_top + 0.45 * yawn)))
    pose.oy -= 0.9 * yawn


def anim_stretch(pose: Pose, t: float, age: float, a: FaceAnimator) -> None:
    reach = math.sin(math.pi * ((age / 4.0) % 1.0)) ** 2  # reaching up, then letting go
    pose.oy -= 1.6 * reach
    _set_mouth(pose, open=1.0 + 3.5 * reach)


def anim_shiver(pose: Pose, t: float, age: float, a: FaceAnimator) -> None:
    grip = smooth(age / 0.3)  # the whole face quivering at the top of a stretch, "嗷~~"
    pose.ox += grip * (0.55 * math.sin(t * 71.0) + 0.3 * math.sin(t * 43.0))
    pose.oy += grip * (0.4 * math.sin(t * 59.0) - 1.2)
    _set_mouth(pose, open=5.0 + 1.0 * grip * math.sin(t * 37.0), w=2.6 + 0.4 * grip * math.sin(t * 29.0))
    pose.brow_raise += 0.8 * grip


def anim_effort(pose: Pose, t: float, age: float, a: FaceAnimator) -> None:
    push = max(0.0, math.sin(2 * math.pi * t / 2.2)) ** 2  # straining, a breath, straining again
    pose.oy += 0.6 * push
    _set_eyes(pose, lambda s: replace(s, lid_top=min(0.9, s.lid_top + 0.25 * push)))


def anim_whisper(pose: Pose, t: float, age: float, a: FaceAnimator) -> None:
    pose.gaze_override = (2.2, 0.2, 0.85)
    _set_mouth(pose, open=1.4 + 1.2 * abs(math.sin(2 * math.pi * 2.6 * t)))


def anim_shout(pose: Pose, t: float, age: float, a: FaceAnimator) -> None:
    call = _bump((t % 2.4) / 1.4)  # "hey~", a pause, "hey~"
    _set_mouth(pose, open=1.5 + 5.0 * call)
    pose.oy -= 0.7 * call
    pose.brow_raise += 1.0 * call


def anim_fighting(pose: Pose, t: float, age: float, a: FaceAnimator) -> None:
    pump = math.exp(-(t % 0.9) * 6.0)  # on every pump of the fist
    pose.oy -= 1.1 * pump
    _set_mouth(pose, open=2.5 + 2.5 * pump)


def anim_sigh(pose: Pose, t: float, age: float, a: FaceAnimator) -> None:
    phase = (t / 5.5) % 1.0
    inhale = smooth(phase / 0.25) if phase < 0.25 else 1.0 - smooth((phase - 0.25) / 0.35)
    exhale = _bump((phase - 0.25) / 0.35)
    pose.oy += -0.7 * inhale + 1.4 * exhale
    _set_mouth(pose, open=2.2 * exhale)


def anim_proud(pose: Pose, t: float, age: float, a: FaceAnimator) -> None:
    pose.oy -= 0.9  # chin up
    pose.ox += 0.8 * math.sin(2 * math.pi * 0.25 * t)


def anim_joyride(pose: Pose, t: float, age: float, a: FaceAnimator) -> None:
    pose.ox += 0.3 * math.sin(t * 31.0) - 0.6  # rattling along, leaning into the wind
    pose.oy += 0.25 * math.sin(t * 23.0) - 0.4 * abs(math.sin(math.pi * t))


def anim_gazing(pose: Pose, t: float, age: float, a: FaceAnimator) -> None:
    pose.gaze_override = (1.2 * math.sin(2 * math.pi * t / 14.0), -1.3, 0.8)  # far away, slowly panning


def anim_giggle(pose: Pose, t: float, age: float, a: FaceAnimator) -> None:
    phase = t % 2.4
    pose.oy += 0.45 * abs(math.sin(2 * math.pi * 5.0 * t)) * _bump(phase)


def anim_exhausted(pose: Pose, t: float, age: float, a: FaceAnimator) -> None:
    pant = abs(math.sin(2 * math.pi * 1.5 * t))
    _set_mouth(pose, open=1.4 + 2.2 * pant)
    pose.oy += 0.6 + 0.4 * pant
    pose.ox += 1.0 * math.sin(2 * math.pi * 0.25 * t)


def anim_woozy(pose: Pose, t: float, age: float, a: FaceAnimator) -> None:
    lurch = math.sin(2 * math.pi * 0.4 * t) + 0.35 * math.sin(2 * math.pi * 1.1 * t)  # staggering
    pose.ox += 2.0 * lurch
    pose.oy += 0.8 * abs(lurch)


def anim_petted(pose: Pose, t: float, age: float, a: FaceAnimator) -> None:
    rub = math.sin(2 * math.pi * 0.7 * t)  # leaning into the hand
    pose.ox += 0.9 * rub
    pose.oy += 0.5 + 0.4 * abs(rub)


# ---------------------------------------------------------------- the catalogue

LOOKS: dict[str, tuple[str, Look]] = {
    "smile": ("微笑", look(eye(rx=3.2, ry=3.3, smile=0.35), lips(y=31.7, w=11.9, curve=5.6),
                         brow=brows(y=11.8, angle=-4, curve=0.4, length=6.5))),
    "happy": ("开心", look(eye("happy", rx=4.2, ry=3.0, thick=0.85, dy=0.6),
                         lips(y=31.3, w=13.5, curve=6.5, open=2.4), motion=MotionSpec(bounce=0.35, bounce_hz=0.7))),
    "laugh": ("大笑", look(eye("happy", rx=5.2, ry=3.4, thick=1.0, dy=1.2),
                         lips(y=29.5, w=11.5, curve=3.5, open=6.0, tongue=0.8),
                         motion=MotionSpec(bounce=0.75, bounce_hz=0.85, breath=0.1), anim=anim_laugh)),
    "surprise": ("惊讶", look(eye(rx=3.4, ry=4.2), lips(y=32.0, w=2.6, curve=0.0, open=5.5, thick=0.8),
                            brow=brows(y=10.6, curve=0.7, angle=-6, length=6), motion=MotionSpec(blinks=0.6),
                            anim=anim_surprise)),
    "content": ("满足", look(eye("happy", rx=4.4, ry=2.6, thick=0.85, dy=1.0), lips(y=30.5, w=9.5, curve=4.0),
                           extras=fx("blush"), motion=MotionSpec(breath=0.32))),
    "wink": ("眨眼", look(eye("happy", rx=4.6, ry=3.0, thick=0.9, dy=1.0), lips(x=33.0, y=30.8, w=10.5, curve=5.0, tilt=0.8),
                        eye_r=eye(rx=3.4, ry=3.6), extras=fx("sparkles", scale=0.8))),
    "curious": ("好奇", look(eye(rx=3.1, ry=3.0, dy=0.8), lips(x=32.5, y=32.5, w=2.4, curve=0.0, open=3.0),
                           eye_r=eye(rx=3.6, ry=4.0, dy=-0.8),
                           brow=(BrowSpec(y=12.0, angle=4, length=6), BrowSpec(y=10.4, angle=-10, curve=0.5, length=6.5)),
                           motion=MotionSpec(saccades=1.8, gaze=(2.2, 1.0), sway=0.45))),
    "thinking": ("思考", look(eye(rx=2.8, ry=3.0, lid_top=0.15), lips(x=33.5, y=33.0, w=4.5, curve=-0.6, tilt=0.8),
                            brow=(BrowSpec(y=12.6, angle=6, length=6.5), BrowSpec(y=10.2, angle=-12, curve=0.5)),
                            extras=fx("dots"), motion=MotionSpec(gaze_bias=(1.6, -1.3), gaze=(0.6, 0.4), saccades=0.6),
                            anim=anim_thinking)),
    "confused": ("困惑", look(eye(rx=2.8, ry=2.9), MouthSpec(kind="wave", y=33.5, w=5.5, amp=0.9, waves=1.6, thick=0.75),
                            brow=(BrowSpec(y=11.0, angle=-14, length=6.5), BrowSpec(y=12.6, angle=12, length=6.5)),
                            extras=fx("question"), motion=MotionSpec(sway=0.5))),
    "listening": ("聆听", look(eye(rx=1.7, ry=4.4), lips(y=33.0, w=2.6, curve=0.4),
                             motion=MotionSpec(saccades=0.7, gaze=(1.0, 0.5), blinks=0.9))),
    "speechless": ("无语", look(eye("line", rx=4.2, thick=0.8), lips(y=33.0, w=2.8, curve=0.0), extras=fx("sweat"),
                              motion=MotionSpec(blinks=0.4, saccades=0.4))),
    "shy": ("害羞", look(eye("happy", rx=4.8, ry=3.2, thick=0.9, dy=1.5), lips(y=31.8, w=7.0, curve=2.8), extras=fx("blush"),
                       motion=MotionSpec(gaze_bias=(0.0, 0.9), gaze=(1.8, 0.3), sway=0.3))),
    "wronged": ("委屈", look(eye(rx=3.4, ry=3.6, lid_top=0.12, lid_tilt=-1.4, dy=0.8),
                           lips(y=34.5, w=5.5, curve=-3.0, thick=0.8), brow=brows(y=11.2, angle=-18, length=6.5),
                           extras=fx("tear_well"), motion=MotionSpec(gaze_bias=(0.0, 0.5)), anim=anim_tremble)),
    "sad": ("难过", look(eye(rx=3.2, ry=3.3, lid_top=0.32, lid_tilt=-1.6, dy=0.5), lips(y=35.5, w=8.0, curve=-4.0),
                       brow=brows(y=11.6, angle=-20, length=6.5),
                       motion=MotionSpec(gaze_bias=(0.0, 1.0), gaze=(1.2, 0.3), saccades=0.5, blinks=0.7,
                                         slow_blinks=0.5, breath=0.3))),
    "crying": ("哭泣", look(eye("happy", rx=4.8, ry=2.8, thick=0.9, dy=1.2), lips(y=33.5, w=7.0, curve=-4.0, open=3.0),
                          brow=brows(y=11.2, angle=-22, length=6), extras=fx("tears"), anim=anim_crying)),
    "angry": ("生气", look(eye(rx=3.0, ry=3.0, lid_top=0.3, lid_tilt=1.7, dy=1.0), lips(y=34.0, w=8.5, curve=-2.5, thick=0.9),
                         brow=brows(y=12.6, angle=26, length=7.5, thick=0.9), extras=fx("anger"),
                         motion=MotionSpec(shake=0.25, saccades=0.5, blinks=0.6))),
    "nervous": ("紧张", look(eye(rx=3.3, ry=3.4), MouthSpec(kind="wave", y=33.5, w=5.0, amp=0.7, waves=2.5),
                           brow=brows(y=10.6, angle=-12, length=6), extras=fx("sweat"),
                           motion=MotionSpec(shake=0.18, saccades=2.5, blinks=2.0, gaze=(2.0, 0.8)))),
    "sleepy": ("困倦", look(eye(rx=3.4, ry=3.3, lid_top=0.5), lips(y=33.0, w=2.4, open=1.6, curve=0.0),
                          extras=fx("zzz", scale=0.7),
                          motion=MotionSpec(saccades=0.3, blinks=0.5, slow_blinks=1.0, breath=0.35), anim=anim_sleepy)),
    "sleeping": ("睡觉", look(eye("closed", rx=5.0, ry=2.4, thick=0.85, dy=1.2, blink=False),
                            lips(y=33.0, w=1.7, open=1.0, curve=0.0), extras=fx("zzz"),
                            motion=MotionSpec(saccades=0.05, gaze=(0.01, 0.01), breath=0.55, sway=0.1),
                            anim=anim_sleeping)),
    "love": ("喜欢", look(eye("heart", rx=3.4, blink=False), lips(y=31.0, w=12.0, curve=5.5),
                        extras=fx("hearts", "blush"), motion=MotionSpec(bounce=0.2))),
    "cool": ("酷", look(eye("glasses", rx=5.0, ry=2.4, blink=False), lips(x=33.5, y=32.5, w=7.0, curve=2.0, tilt=1.4),
                       motion=MotionSpec(saccades=0.4, sway=0.3))),
    "smug": ("得意", look(eye(rx=3.0, ry=3.1, lid_top=0.42, dy=0.6), lips(x=34.0, y=32.0, w=7.5, curve=3.0, tilt=2.0),
                        brow=(BrowSpec(y=12.4, angle=4), BrowSpec(y=10.2, angle=-8, curve=0.8)),
                        motion=MotionSpec(gaze_bias=(1.2, 0.0), saccades=0.5))),
    "deadpan": ("面瘫", look(eye(rx=1.45, ry=1.2), lips(y=33.5, w=7.5, curve=0.0, thick=0.72),
                           motion=MotionSpec(saccades=0.25, blinks=0.5, sway=0.1, breath=0.1))),
    "dizzy": ("晕", look(eye("spiral", rx=3.6, thick=0.65, blink=False),
                        MouthSpec(kind="wave", y=34.0, w=6.0, amp=0.9, waves=1.5), extras=fx("stars_spin"),
                        motion=MotionSpec(saccades=0.2), anim=anim_dizzy)),
    "sick": ("生病", look(eye(rx=3.0, ry=3.0, lid_top=0.45, lid_tilt=-1.0),
                        MouthSpec(kind="wave", y=33.5, w=6.0, amp=0.6, waves=2.0),
                        brow=brows(y=11.5, angle=-10), extras=fx("sweat"),
                        motion=MotionSpec(sway=0.5, blinks=0.6, slow_blinks=0.6))),
    "starstruck": ("星星眼", look(eye("star", rx=3.8, blink=False), lips(y=31.0, w=11.5, curve=5.0, open=3.0),
                                extras=fx("sparkles"), motion=MotionSpec(bounce=0.3))),
    "kiss": ("亲亲", look(eye("happy", rx=4.8, ry=3.1, thick=0.9, dy=1.0), MouthSpec(kind="kiss", y=33.4, w=3.2),
                        extras=fx("hearts", scale=0.8) + fx("blush"))),
    "tease": ("调皮", look(eye("happy", rx=4.6, ry=3.0, thick=0.9, dy=1.0),
                         lips(y=31.5, w=10.5, curve=4.5, tongue=1.0, tongue_x=2.5), eye_r=eye(rx=3.3, ry=3.4),
                         motion=MotionSpec(sway=0.35))),
    "scared": ("害怕", look(eye(rx=3.6, ry=4.2, lid_tilt=-0.8),
                          lips(y=33.5, w=6.0, curve=-1.5, open=2.5, teeth=1.0), brow=brows(y=10.2, angle=-16, length=6),
                          extras=fx("sweat"), motion=MotionSpec(shake=0.35, saccades=3.0, gaze=(2.2, 1.0), blinks=1.5))),
    "determined": ("坚定", look(eye(rx=2.9, ry=3.0, lid_top=0.25, lid_tilt=0.9, dy=0.8), lips(y=33.6, w=5.0, curve=-0.6, thick=0.9),
                              brow=brows(y=12.0, angle=16, length=7, thick=0.9),
                              motion=MotionSpec(saccades=0.4, blinks=0.6, gaze=(0.6, 0.3)))),
    "bored": ("无聊", look(eye(rx=3.2, ry=3.2, lid_top=0.55), lips(y=34.0, w=5.5, curve=-1.0, tilt=-0.8),
                         motion=MotionSpec(gaze_bias=(1.4, 0.3), saccades=0.35, blinks=0.6, slow_blinks=0.7, sway=0.2),
                         anim=anim_bored)),
    "greedy": ("馋", look(eye(rx=3.3, ry=3.6), lips(x=33.0, y=32.5, w=5.0, curve=2.0, open=2.2),
                        extras=fx("drool"), motion=MotionSpec(gaze_bias=(1.6, 0.4), gaze=(0.5, 0.3)), anim=anim_greedy)),
    "awkward": ("尴尬", look(eye(r=3.1), lips(y=32.8, w=7.0, curve=0.6, open=2.6, teeth=1.0),
                           brow=brows(y=10.8, angle=-8, length=6), extras=fx("sweat"),
                           motion=MotionSpec(gaze_bias=(-1.3, 0.2), saccades=1.5))),
    "sparkle": ("闪亮", look(eye(rx=3.5, ry=3.8, smile=0.2), lips(y=31.0, w=12.0, curve=5.5, open=1.6),
                           extras=fx("sparkles", scale=1.2), motion=MotionSpec(bounce=0.15))),
    "pout": ("嘟嘴", look(eye(rx=3.4, ry=3.5), lips(x=31.5, y=33.5, w=2.3, curve=-0.5, open=1.8, thick=0.85),
                        brow=brows(y=11.6, angle=10, length=6), motion=MotionSpec(gaze_bias=(-1.2, 0.1), sway=0.3))),
    "sideeye": ("斜眼", look(eye(rx=3.9, ry=3.0, lid_top=0.25), lips(y=33.5, w=6.0, curve=-0.4, tilt=-0.6),
                           motion=MotionSpec(saccades=0.3), anim=anim_sideeye)),
    "focused": ("专注", look(eye(rx=3.3, ry=3.0, lid_top=0.38, lid_bottom=0.2), lips(y=33.4, w=3.2, curve=0.0),
                           brow=brows(y=12.5, angle=8, length=7),
                           motion=MotionSpec(saccades=0.25, gaze=(0.4, 0.2), blinks=0.45, sway=0.1))),
    "cat": ("猫猫", look(eye("peak", rx=3.8, ry=4.0, thick=0.85), MouthSpec(kind="cat", y=31.8, w=4.0, curve=2.4),
                       extras=fx("whiskers", "blush"), motion=MotionSpec(bounce=0.15, bounce_hz=0.5))),
    "cheer": ("欢呼", look(eye("happy", rx=5.0, ry=3.4, thick=1.0, dy=0.6), lips(y=30.0, w=12.5, curve=5.0, open=4.5),
                         extras=fx("sparkles"), motion=MotionSpec(bounce=0.7, bounce_hz=0.9), anim=anim_cheer)),
    "mischief": ("坏笑", look(eye(rx=3.0, ry=3.0, lid_top=0.32, lid_tilt=1.1, dy=0.6), lips(x=32.5, y=31.0, w=11.0, curve=4.5, tilt=1.2),
                            brow=brows(y=11.8, angle=16, length=7),
                            motion=MotionSpec(sway=0.3, gaze_bias=(0.8, 0.0)))),
    "excited": ("兴奋", look(eye(rx=4.1, ry=4.4), lips(y=31.0, w=12.5, curve=6.0, open=5.0, tongue=0.6),
                           brow=brows(y=10.6, curve=0.6, angle=-6, length=6), extras=fx("sparkles"),
                           motion=MotionSpec(bounce=0.55, bounce_hz=1.5, saccades=2.0), anim=anim_excited)),
    "relieved": ("松了口气", look(eye("closed", rx=4.6, ry=2.4, thick=0.85, dy=0.8, blink=False), lips(y=31.5, w=8.5, curve=3.4),
                              brow=brows(y=11.4, angle=-10, curve=0.3, length=6), extras=fx("puff"),
                              motion=MotionSpec(breath=0.45), anim=anim_relieved)),
    "touched": ("感动", look(eye(rx=3.5, ry=3.8, smile=0.25), lips(y=32.0, w=7.5, curve=3.0),
                           brow=brows(y=11.0, angle=-12, length=6), extras=fx("tear_well"), anim=anim_tremble)),
    "disgusted": ("嫌弃", look(eye(rx=3.0, ry=3.0, lid_top=0.45, lid_bottom=0.22, lid_tilt=0.8),
                             MouthSpec(kind="wave", x=33.0, y=34.0, w=6.0, amp=0.7, waves=1.2, curve=-2.0),
                             eye_r=eye(rx=3.0, ry=3.0, lid_top=0.3, lid_bottom=0.15),
                             brow=(BrowSpec(y=12.8, angle=14, length=6.5), BrowSpec(y=11.2, angle=-4, length=6.5)),
                             motion=MotionSpec(gaze_bias=(-1.4, 0.3), saccades=0.5))),
    "doubtful": ("怀疑", look(eye(rx=3.0, ry=3.0, lid_top=0.38, lid_bottom=0.12), lips(x=33.0, y=33.5, w=5.5, curve=-0.8, tilt=-1.2),
                            eye_r=eye(rx=3.2, ry=3.4, lid_top=0.1),
                            brow=(BrowSpec(y=12.6, angle=10, length=6.5), BrowSpec(y=10.6, angle=-12, curve=0.5, length=6.5)),
                            motion=MotionSpec(saccades=0.5, gaze=(0.6, 0.3)))),
    "shocked": ("震惊", look(eye(rx=3.4, ry=4.6), lips(y=32.5, w=4.0, curve=0.0, open=7.5),
                           brow=brows(y=10.4, curve=0.7, angle=-8, length=6), extras=fx("exclaim"),
                           motion=MotionSpec(blinks=0.3, saccades=0.3), anim=anim_shocked)),
    "singing": ("唱歌", look(eye("closed", rx=4.6, ry=2.4, thick=0.85, dy=0.8, blink=False), lips(y=31.5, w=5.5, curve=1.5, open=3.0),
                           brow=brows(y=11.0, angle=-6, curve=0.5, length=6), extras=fx("notes"),
                           motion=MotionSpec(breath=0.2), anim=anim_singing)),
    "yummy": ("好吃", look(eye("happy", rx=4.8, ry=3.2, thick=0.9, dy=1.0), lips(y=31.4, w=9.0, curve=4.0, tongue=0.9),
                         extras=fx("blush") + fx("sparkles", scale=0.7), anim=anim_yummy)),
    "annoyed": ("不耐烦", look(eye(rx=3.2, ry=3.1, lid_top=0.5, lid_tilt=0.3), lips(y=34.0, w=6.0, curve=-1.6, tilt=-1.0),
                             brow=brows(y=12.2, angle=12, length=7), extras=fx("steam"),
                             motion=MotionSpec(saccades=0.3, blinks=0.7), anim=anim_annoyed)),
    "calm": ("平静", look(eye(rx=3.1, ry=3.2, lid_top=0.22, smile=0.2), lips(y=31.8, w=7.5, curve=2.2),
                        brow=brows(y=12.2, angle=-2, curve=0.3, length=6.5),
                        motion=MotionSpec(breath=0.3, saccades=0.6, blinks=0.8, slow_blinks=0.4, sway=0.2))),
    "huff": ("哼", look(eye("closed", rx=4.0, ry=2.2, thick=0.85, dy=0.4, blink=False),
                      lips(x=31.0, y=33.6, w=3.0, curve=-1.2, open=1.2, thick=0.85),
                      brow=brows(y=12.4, angle=18, length=6.5), extras=fx("steam"),
                      motion=MotionSpec(breath=0.15, sway=0.15), anim=anim_huff)),
    "lonely": ("失落", look(eye(rx=3.0, ry=3.0, lid_top=0.4, lid_tilt=-1.0, dy=0.8), lips(y=34.5, w=4.5, curve=-1.4),
                          brow=brows(y=12.0, angle=-12, length=5.5),
                          motion=MotionSpec(gaze_bias=(-0.6, 1.4), gaze=(0.6, 0.2), saccades=0.25, blinks=0.5,
                                            slow_blinks=0.8, breath=0.35, sway=0.1), anim=anim_bored)),
    "clingy": ("撒娇", look(eye("happy", rx=4.4, ry=3.0, thick=0.9, dy=1.2), MouthSpec(kind="cat", y=32.0, w=3.6, curve=2.0),
                          extras=fx("blush") + fx("hearts", scale=0.6), motion=MotionSpec(breath=0.3), anim=anim_clingy)),
    "peek": ("偷看", look(eye(rx=3.3, ry=3.8), lips(y=33.0, w=2.4, curve=0.0, open=1.4),
                        brow=brows(y=11.4, angle=-6, length=5.5), motion=MotionSpec(blinks=0.7, saccades=0.3),
                        anim=anim_peek)),
    "observe": ("暗中观察", look(eye(rx=3.6, ry=3.2, lid_top=0.35, lid_bottom=0.25), lips(y=33.8, w=3.0, curve=0.0),
                             brow=brows(y=12.4, angle=10, length=6.5),
                             motion=MotionSpec(saccades=0.1, blinks=0.4, sway=0.1), anim=anim_observe)),
    "ouch": ("好痛", look(eye("chevron", rx=3.0, ry=2.6, thick=0.85), lips(y=33.5, w=6.5, curve=-1.6, open=2.6, teeth=1.0),
                        brow=brows(y=11.4, angle=-16, length=6), extras=fx("sweat"), anim=anim_ouch)),
    "ko": ("摔懵", look(eye("x", rx=2.8, ry=2.8, thick=0.75, blink=False),
                      MouthSpec(kind="wave", y=34.0, w=5.5, amp=0.8, waves=1.5), extras=fx("stars_spin"),
                      motion=MotionSpec(saccades=0.1), anim=anim_dizzy)),
    "pleading": ("可怜巴巴", look(eye("solid", rx=3.6, ry=4.2, shine=0.42), lips(y=34.0, w=3.6, curve=-1.2, thick=0.8),
                              brow=brows(y=10.8, angle=-18, length=6), extras=fx("tear_well"),
                              motion=MotionSpec(saccades=0.3, blinks=0.6, gaze=(0.5, 0.3)), anim=anim_tremble)),
    "busted": ("被抓包", look(eye(rx=3.3, ry=4.3), lips(y=33.0, w=7.0, curve=0.5, open=2.6, teeth=1.0),
                           brow=brows(y=10.4, angle=-10, curve=0.5, length=6), extras=fx("sweat") + fx("exclaim", x=14.5),
                           motion=MotionSpec(blinks=0.8, saccades=1.5), anim=anim_busted)),
    "cute": ("卖萌", look(eye("solid", rx=3.2, ry=3.7, shine=0.4, smile=0.25),
                        MouthSpec(kind="cat", y=31.8, w=3.4, curve=2.0, open=1.2),
                        extras=fx("blush") + fx("sparkles", scale=0.7), motion=MotionSpec(bounce=0.2, bounce_hz=0.6),
                        anim=anim_cute)),
    "dancing": ("蹦迪", look(eye("happy", rx=4.6, ry=3.2, thick=0.95, dy=0.8), lips(y=30.8, w=11.0, curve=5.0, open=3.0),
                           extras=fx("notes"), motion=MotionSpec(breath=0.0, sway=0.0), anim=anim_dancing)),
    "idea": ("想起来了", look(eye(rx=3.3, ry=4.1), MouthSpec(kind="o", y=32.5, w=2.2, open=3.2),
                          brow=brows(y=10.4, curve=0.6, angle=-6, length=6), extras=fx("bulb"),
                          motion=MotionSpec(blinks=0.6), anim=anim_surprise)),
    "working": ("干活中", look(eye(rx=3.0, ry=3.0, lid_top=0.3), lips(y=33.6, w=3.4, curve=-0.2),
                            brow=brows(y=12.2, angle=8, length=6.5), extras=fx("loading"),
                            motion=MotionSpec(blinks=0.5, sway=0.05, breath=0.1), anim=anim_working)),
    "done": ("搞定", look(eye("happy", rx=4.4, ry=3.0, thick=0.9, dy=0.8), lips(x=33.0, y=31.0, w=10.5, curve=5.0, tilt=1.0),
                        brow=brows(y=11.0, angle=-6, curve=0.4, length=6), extras=fx("check") + fx("sparkles", scale=0.6),
                        anim=anim_done)),
    "woken": ("被吵醒", look(eye("line", rx=3.4, thick=0.8, dy=0.6),
                          MouthSpec(kind="wave", y=34.0, w=5.0, amp=0.7, waves=1.8, curve=-1.0),
                          eye_r=eye(rx=3.0, ry=3.0, lid_top=0.55, lid_tilt=0.6),
                          brow=(BrowSpec(y=12.6, angle=10, length=6), BrowSpec(y=11.0, angle=14, length=6)),
                          extras=fx("bell"), motion=MotionSpec(blinks=0.3, saccades=0.2), anim=anim_woken)),
    "waking": ("睡醒", look(eye(rx=3.2, ry=3.3), lips(y=32.6, w=3.0, curve=0.6, open=1.0),
                          brow=brows(y=11.8, angle=-4, curve=0.3, length=6),
                          motion=MotionSpec(blinks=0.8, saccades=0.6, breath=0.35), anim=anim_waking)),
    "dashing": ("帅气", look(eye(rx=3.0, ry=3.1, lid_top=0.3, lid_tilt=0.5), lips(x=34.0, y=32.0, w=8.0, curve=3.2, tilt=2.2),
                           eye_r=eye("happy", rx=4.2, ry=2.8, thick=0.9, dy=1.0),
                           brow=(BrowSpec(y=12.2, angle=8, length=6.5), BrowSpec(y=10.6, angle=-10, curve=0.6, length=6.5)),
                           extras=fx("sparkles", x=50.0, y=13.0, scale=1.1), motion=MotionSpec(saccades=0.3, sway=0.2),
                           anim=anim_dashing)),
    "yawn": ("打哈欠", look(eye(rx=3.2, ry=3.2, lid_top=0.5), lips(y=32.6, w=2.2, curve=0.0, open=1.2),
                         brow=brows(y=11.8, angle=-8, length=6), extras=fx("tear_well", scale=0.8),
                         motion=MotionSpec(blinks=0.4, slow_blinks=1.0, saccades=0.3, breath=0.3), anim=anim_yawn)),
    "stretch": ("伸懒腰", look(eye("chevron", rx=2.8, ry=2.4, thick=0.85), lips(y=32.0, w=4.5, curve=1.0, open=2.0),
                            brow=brows(y=11.2, angle=-8, curve=0.4, length=6),
                            motion=MotionSpec(breath=0.0, sway=0.1), anim=anim_stretch)),
    "effort": ("使劲", look(eye(rx=3.0, ry=3.0, lid_top=0.35, lid_tilt=1.4, dy=0.8),
                          lips(y=33.4, w=7.5, curve=-0.8, open=2.6, teeth=1.0),
                          brow=brows(y=12.2, angle=20, length=7, thick=0.9), extras=fx("sweat"),
                          motion=MotionSpec(shake=0.3, blinks=0.3, saccades=0.2), anim=anim_effort)),
    "whisper": ("悄悄话", look(eye(rx=3.0, ry=3.1, lid_top=0.2), MouthSpec(kind="o", x=36.0, y=32.8, w=1.7, open=2.2, thick=0.8),
                            brow=brows(y=11.4, angle=-6, curve=0.3, length=6), extras=fx("blush", scale=0.8),
                            motion=MotionSpec(saccades=0.4, sway=0.15), anim=anim_whisper)),
    "shout": ("喊话", look(eye(rx=3.3, ry=4.0), lips(y=31.5, w=5.0, curve=0.5, open=5.0),
                         brow=brows(y=10.6, angle=-6, curve=0.5, length=6), extras=fx("exclaim"),
                         motion=MotionSpec(blinks=0.6, saccades=0.4), anim=anim_shout)),
    "fighting": ("加油", look(eye(rx=3.0, ry=3.2, lid_top=0.2, lid_tilt=0.9, dy=0.6), lips(y=31.2, w=9.0, curve=3.5, open=4.0),
                            brow=brows(y=11.8, angle=16, length=7, thick=0.9), extras=fx("sparkles"),
                            motion=MotionSpec(blinks=0.5, saccades=0.3), anim=anim_fighting)),
    "sigh": ("叹气", look(eye(rx=3.0, ry=3.0, lid_top=0.5, lid_tilt=-0.6, dy=0.6), lips(y=33.8, w=4.0, curve=-0.8),
                        brow=brows(y=12.0, angle=-10, length=6), extras=fx("puff"),
                        motion=MotionSpec(blinks=0.5, slow_blinks=0.8, saccades=0.3, breath=0.0), anim=anim_sigh)),
    "proud": ("傲娇", look(eye("closed", rx=4.0, ry=2.2, thick=0.85, dy=0.4, blink=False),
                         lips(x=33.0, y=31.6, w=7.5, curve=3.0, tilt=1.6),
                         brow=brows(y=10.8, angle=-8, curve=0.5, length=6.5), extras=fx("sparkles", scale=0.8),
                         motion=MotionSpec(breath=0.25, sway=0.2), anim=anim_proud)),
    "joyride": ("兜风", look(eye("happy", rx=4.6, ry=3.0, thick=0.9, dy=0.6), lips(y=30.8, w=9.0, curve=5.0, open=3.6),
                           brow=brows(y=11.2, angle=-4, length=6), extras=fx("wind"),
                           motion=MotionSpec(breath=0.0, sway=0.0), anim=anim_joyride)),
    "chill": ("惬意", look(eye("closed", rx=4.4, ry=2.4, thick=0.85, dy=0.8, blink=False), lips(y=31.6, w=7.0, curve=2.8),
                         brow=brows(y=11.6, angle=-4, curve=0.3, length=6), extras=fx("notes", scale=0.8),
                         motion=MotionSpec(breath=0.5, sway=0.45))),
    "gazing": ("看风景", look(eye(rx=3.0, ry=3.3, smile=0.15), lips(y=32.6, w=3.5, curve=1.2),
                           brow=brows(y=11.8, angle=-2, length=6),
                           motion=MotionSpec(saccades=0.15, blinks=0.5, slow_blinks=0.6, breath=0.35, sway=0.3),
                           anim=anim_gazing)),
    "giggle": ("偷笑", look(eye("happy", rx=4.4, ry=3.0, thick=0.9, dy=1.0), lips(x=32.5, y=31.6, w=5.5, curve=3.0, tilt=0.6),
                          brow=brows(y=11.2, angle=-6, length=6), extras=fx("blush", scale=0.8), anim=anim_giggle)),
    "low_battery": ("电量低", look(eye(rx=3.0, ry=3.0, lid_top=0.5, lid_tilt=-0.8, dy=0.8),
                                MouthSpec(kind="wave", y=34.2, w=4.5, amp=0.5, waves=1.6, curve=-1.0),
                                extras=fx("battery_low"),
                                motion=MotionSpec(blinks=0.5, slow_blinks=1.0, saccades=0.25, breath=0.4, sway=0.3),
                                anim=anim_sleepy)),
    "exhausted": ("累坏了", look(eye(rx=3.0, ry=3.0, lid_top=0.55, lid_tilt=-1.2, dy=1.0),
                              lips(y=33.6, w=4.0, curve=-0.5, open=2.0),
                              brow=brows(y=12.0, angle=-14, length=6), extras=fx("sweat"),
                              motion=MotionSpec(blinks=0.5, slow_blinks=0.8, saccades=0.2, breath=0.0), anim=anim_exhausted)),
    "woozy": ("摇摇晃晃", look(eye(rx=3.0, ry=3.0, lid_top=0.6, dy=0.8),
                           MouthSpec(kind="wave", y=34.0, w=5.0, amp=0.8, waves=1.4),
                           eye_r=eye(rx=3.0, ry=3.0, lid_top=0.35, dy=0.4),
                           brow=(BrowSpec(y=12.2, angle=-10, length=6), BrowSpec(y=11.2, angle=-16, length=6)),
                           extras=fx("sweat"),
                           motion=MotionSpec(blinks=0.8, slow_blinks=0.9, saccades=0.3, breath=0.0, sway=0.0),
                           anim=anim_woozy)),
    "petted": ("被摸头", look(eye("happy", rx=4.6, ry=3.0, thick=0.9, dy=1.4), lips(y=31.6, w=6.5, curve=3.0),
                           brow=brows(y=11.4, angle=-8, curve=0.3, length=6), extras=fx("blush") + fx("hearts", scale=0.7),
                           motion=MotionSpec(breath=0.3, sway=0.0), anim=anim_petted)),
    "dreaming": ("美梦", look(eye("closed", rx=4.6, ry=2.4, thick=0.85, dy=1.2, blink=False), lips(y=32.0, w=4.5, curve=2.2),
                            extras=fx("zzz", scale=0.7) + fx("sparkles", scale=0.6),
                            motion=MotionSpec(saccades=0.05, gaze=(0.01, 0.01), breath=0.55, sway=0.15))),
    "charging": ("充电中", look(eye("closed", rx=4.4, ry=2.4, thick=0.85, dy=1.0, blink=False), lips(y=31.8, w=5.5, curve=2.4),
                             extras=fx("battery_charge"), motion=MotionSpec(breath=0.45, saccades=0.05, sway=0.1))),
    "stretch_shiver": ("伸懒腰发抖", look(eye("chevron", rx=3.0, ry=2.6, thick=0.95, blink=False),
                                     MouthSpec(kind="o", y=32.4, w=2.6, open=5.0),
                                     brow=brows(y=10.8, angle=-10, curve=0.5, length=6),
                                     extras=fx("tear_well"),
                                     motion=MotionSpec(breath=0.0, sway=0.0, saccades=0.1), anim=anim_shiver)),
    "drowsy": ("睡眼惺忪", look(eye("solid", rx=3.2, ry=3.8, lid_top=0.55, lid_tilt=-0.6, dy=0.6),
                             lips(y=33.2, w=2.8, curve=0.0, open=1.2),
                             eye_r=eye("solid", rx=3.2, ry=3.8, lid_top=0.65, lid_tilt=-0.6, dy=0.6),
                             brow=brows(y=12.6, angle=-6, curve=0.2, length=6),
                             motion=MotionSpec(blinks=0.2, saccades=0.15, gaze=(0.5, 0.2), gaze_bias=(0.0, 0.6),
                                               breath=0.3, sway=0.1), anim=anim_drowsy)),
}


@dataclass(frozen=True)
class Expression:
    name: str
    label: str
    index: int
    look: Look

    @property
    def code(self) -> str:
        return f"{self.index:02d}{self.label}"


EXPRESSIONS: tuple[Expression, ...] = (Expression(IDLE_EXPRESSION, "待机", 0, LOOKS[IDLE_LOOKS[0]][1]),) + tuple(
    Expression(name, label, index, item) for index, (name, (label, item)) in enumerate(LOOKS.items(), start=1)
)
EXPRESSION_BY_NAME = {item.name: item for item in EXPRESSIONS}


def resolve_expression(name: str | None) -> Expression:
    if name and name in EXPRESSION_BY_NAME:
        return EXPRESSION_BY_NAME[name]
    return EXPRESSION_BY_NAME[DEFAULT_EXPRESSION]


def look_of(name: str) -> Look:
    return resolve_expression(name).look


def new_animator(name: str, now: float = 0.0, seed: int | None = None) -> FaceAnimator:
    return FaceAnimator(look_of, resolve_expression(name).name, now, seed)


TALK_EXPRESSION = "calm"
# Words in the robot's reply (checked first) or in what the user said, and the face to talk with.
# Later entries win when several match, so put strong feelings last.
REPLY_MOODS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("thinking", ("让我想想", "我想想", "嗯……", "嗯...")),
    ("working", ("正在处理", "马上处理", "交给我", "我来办", "处理中")),
    ("smile", ("谢谢", "好的呀", "没问题", "当然")),
    ("done", ("搞定", "完成了", "办好了", "弄好了", "记下了", "已记录", "收到")),
    ("idea", ("想起来了", "我记得", "对了", "有了")),
    ("whisper", ("悄悄", "小声", "偷偷告诉")),
    ("fighting", ("加油", "你可以的", "冲呀", "冲鸭")),
    ("happy", ("开心", "高兴", "太好了", "太棒了", "真棒", "好耶", "有趣", "好玩")),
    ("dancing", ("跳舞", "蹦迪", "嗨起来")),
    ("chill", ("好舒服", "惬意", "放松")),
    ("proud", ("没我不行", "全靠我", "那当然", "本大人")),
    ("surprise", ("哇", "真的吗", "天哪", "天啊", "居然", "竟然", "没想到")),
    ("confused", ("不明白", "没听懂", "什么意思", "奇怪", "咦")),
    ("awkward", ("尴尬", "呃", "这个嘛", "抱歉", "对不起")),
    ("shy", ("害羞", "不好意思", "过奖", "夸我")),
    ("love", ("喜欢你", "爱你", "么么", "抱抱")),
    ("clingy", ("陪我", "想你了", "黏着你")),
    ("laugh", ("哈哈", "嘿嘿", "嘻嘻", "笑死", "好笑")),
    ("sleepy", ("好困", "困了", "想睡")),
    ("yawn", ("哈欠",)),
    ("sigh", ("唉", "叹气")),
    ("exhausted", ("好累", "累死", "累坏")),
    ("low_battery", ("没电", "电量低", "要充电")),
    ("sad", ("难过", "伤心", "遗憾", "可惜", "呜")),
    ("lonely", ("孤单", "没人理", "冷落")),
    ("wronged", ("委屈", "冤枉", "我没有")),
    ("pleading", ("求求", "拜托", "别走", "不要走")),
    ("ouch", ("好痛", "好疼", "哎哟", "摔疼")),
    ("scared", ("害怕", "好怕", "吓死", "吓到")),
    ("huff", ("哼",)),
    ("angry", ("生气", "气死", "讨厌", "不许", "过分", "别这样")),
)
USER_MOODS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("waking", ("起床", "早上好", "醒醒")),
    ("happy", ("真棒", "厉害", "好聪明", "真乖", "谢谢你")),
    ("shy", ("可爱", "好看", "漂亮", "帅")),
    ("cute", ("卖萌", "卖个萌")),
    ("petted", ("摸摸", "摸头", "乖乖")),
    ("love", ("喜欢你", "爱你")),
    ("dreaming", ("晚安", "睡吧")),
    ("sad", ("再见", "不要你", "不理你")),
    ("angry", ("笨蛋", "笨", "傻", "蠢", "讨厌你", "滚", "闭嘴", "垃圾")),
)


def _last_mood(text: str, table: tuple[tuple[str, tuple[str, ...]], ...]) -> str | None:
    best, best_at = None, -1
    for name, words in table:
        for word in words:
            at = text.rfind(word)
            if at > best_at or (at >= 0 and at == best_at):
                best, best_at = name, at
    return best


def speech_mood(reply: str, heard: str = "") -> str:
    """The expression to talk with: what the reply sounds like, else how the user spoke to the robot."""
    return _last_mood(reply, REPLY_MOODS) or _last_mood(heard, USER_MOODS) or TALK_EXPRESSION


class IdleCycle:
    """While the chosen expression is "idle", picks one of IDLE_LOOKS after another at random intervals."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self._name: str | None = None
        self._until = 0.0

    def resolve(self, name: str, now: float) -> str:
        if name != IDLE_EXPRESSION:
            self._name = None
            return name
        if self._name is None or now >= self._until:
            self._name = self._rng.choice([item for item in IDLE_LOOKS if item != self._name])
            self._until = now + self._rng.uniform(*IDLE_HOLD_S)
        return self._name


def read_expression_name(path: Path = EXPRESSION_STATE_PATH) -> str:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return DEFAULT_EXPRESSION
    if not isinstance(data, dict):
        return DEFAULT_EXPRESSION
    return resolve_expression(str(data.get("name") or DEFAULT_EXPRESSION)).name


def write_expression_name(name: str, path: Path = EXPRESSION_STATE_PATH) -> str:
    resolved = resolve_expression(name).name
    path.write_text(json.dumps({"name": resolved}, ensure_ascii=False) + "\n", encoding="utf-8")
    return resolved


# ---------------------------------------------------------------- previews (web UI, console, contact sheet)


@lru_cache(maxsize=8)
def preview_frames(name: str, count: int = PREVIEW_FRAMES, dt: float = PREVIEW_DT_S) -> tuple[bytes, ...]:
    """A few seconds of the expression from a seeded animator, so previews look the same every time."""
    animator = new_animator(name, 0.0, seed=sum(map(ord, name)))
    frames = []
    for i in range(count):
        if name == IDLE_EXPRESSION:  # a quick tour of the idle faces
            animator.set_expression(IDLE_LOOKS[int(i * dt / 1.4) % len(IDLE_LOOKS)], i * dt)
        frames.append(animator.frame(i * dt))
    return tuple(frames)


@lru_cache(maxsize=None)
def still_frame(name: str) -> bytes:
    return new_animator(name, 0.0, seed=1).frame(0.3)


def frame_pixels(frame: bytes) -> list[tuple[int, int]]:
    levels = np.frombuffer(frame, dtype=np.uint8).reshape(ROWS, COLS) & 31
    rows, cols = np.nonzero(levels >= LIT_THRESHOLD)
    return [(int(c), int(r)) for r, c in zip(rows, cols)]


def encode_levels(frame: bytes) -> str:
    from esp32_link import zero_rle

    return base64.b64encode(zero_rle(frame)).decode("ascii")


def expression_pixels(name: str, frame: int = 0) -> set[tuple[int, int]]:
    frames = preview_frames(name)
    return set(frame_pixels(frames[frame % len(frames)]))


def catalog(frame: int = 0) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for item in EXPRESSIONS:
        still = still_frame(item.name)
        items.append({
                "name": item.name,
                "label": item.label,
                "index": item.index,
                "code": item.code,
                "cols": MATRIX_COLS,
                "rows": MATRIX_ROWS,
            "frames": PREVIEW_FRAMES,
            "pixels": frame_pixels(still),
            "levels": encode_levels(still),
        })
    return items


def loop_frames(name: str, pixels: bool = False) -> dict[str, object]:
    item = resolve_expression(name)
    frames = preview_frames(item.name)
    out: dict[str, object] = {
        "name": item.name,
        "label": item.label,
        "index": item.index,
        "cols": MATRIX_COLS,
        "rows": MATRIX_ROWS,
        "frame_ms": round(PREVIEW_DT_S * 1000),
        "levels": [encode_levels(frame) for frame in frames],
    }
    if pixels:
        out["frames"] = [frame_pixels(frame) for frame in frames]
    return out


# ---------------------------------------------------------------- LED look (same as firmware/xiao_face/face_render.h)

PITCH = 5
LED_WIDTH, LED_HEIGHT = COLS * PITCH, ROWS * PITCH
DOT_RADIUS = 2.5
CORE_RADIUS = 1.0
UNLIT_RADIUS = 1.45
UNLIT = np.array((20, 21, 24), dtype=np.float32)
HOT = np.array((255, 250, 240), dtype=np.float32)
GAMMA = 1.0
HALO_GAIN = 0.8
HOT_GAIN = 0.4
_PALETTE = np.array(PALETTE, dtype=np.float32)


def _disc(radius: float) -> np.ndarray:
    """Coverage of a disc centred in a PITCH x PITCH cell, per pixel (8x8 supersampling)."""
    sub = (np.arange(PITCH * 8) + 0.5) / 8.0 - PITCH / 2.0
    inside = (sub[None, :] ** 2 + sub[:, None] ** 2) <= radius * radius
    return inside.reshape(PITCH, 8, PITCH, 8).mean(axis=(1, 3)).astype(np.float32)


_DOT = np.tile(_disc(DOT_RADIUS), (ROWS, COLS))
_CORE = np.tile(_disc(CORE_RADIUS), (ROWS, COLS))
_UNLIT_DOT = np.tile(_disc(UNLIT_RADIUS), (ROWS, COLS))


def _bilinear_axis(cells: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos = (np.arange(cells * PITCH) + 0.5 - PITCH / 2.0) / PITCH
    lo = np.floor(pos).astype(int)
    weight = (pos - lo).astype(np.float32)
    return np.clip(lo, 0, cells - 1), np.clip(lo + 1, 0, cells - 1), weight


_BX = _bilinear_axis(COLS)
_BY = _bilinear_axis(ROWS)


def _blur(image: np.ndarray) -> np.ndarray:
    kernel = (1.0, 4.0, 6.0, 4.0, 1.0)
    out = image
    for axis in (0, 1):
        padded = np.pad(out, [(2, 2) if a == axis else (0, 0) for a in range(3)])
        size = out.shape[axis]
        out = sum(k * np.take(padded, range(i, i + size), axis=axis) for i, k in enumerate(kernel)) / 16.0
    return out


def render_led_rgb(frame: bytes) -> np.ndarray:
    """One frame as the 320x240 RGB image of round LEDs with their glow."""
    data = np.frombuffer(frame, dtype=np.uint8).reshape(ROWS, COLS)
    bright = ((data & 31) / 31.0) ** GAMMA
    cells = _PALETTE[data >> 5] * bright[..., None]
    halo = _blur(cells)
    x0, x1, wx = _BX
    y0, y1, wy = _BY
    wx, wy = wx[None, :, None], wy[:, None, None]
    halo_up = ((1 - wy) * ((1 - wx) * halo[y0][:, x0] + wx * halo[y0][:, x1])
               + wy * ((1 - wx) * halo[y1][:, x0] + wx * halo[y1][:, x1]))
    cell_up = np.repeat(np.repeat(cells, PITCH, 0), PITCH, 1)
    bright_up = np.repeat(np.repeat(bright, PITCH, 0), PITCH, 1)
    dot = _DOT[..., None]
    out = (cell_up * dot
           + HOT * (_CORE * bright_up ** 2 * HOT_GAIN)[..., None]
           + halo_up * HALO_GAIN * (1.0 - 0.6 * dot)
           + UNLIT * (_UNLIT_DOT * (1.0 - bright_up))[..., None])
    return np.clip(out, 0, 255).astype(np.uint8)


def render_expression_rgb(name: str, frame: int = 0, width: int = LED_WIDTH, height: int = LED_HEIGHT) -> np.ndarray:
    frames = preview_frames(name)
    rgb = render_led_rgb(frames[frame % len(frames)])
    if (width, height) != (LED_WIDTH, LED_HEIGHT):
        from PIL import Image

        rgb = np.asarray(Image.fromarray(rgb, "RGB").resize((width, height), Image.LANCZOS))
    return rgb


def render_expression_frame(name: str, frame: int = 0, width: int = LED_WIDTH, height: int = LED_HEIGHT):
    from PIL import Image

    return Image.fromarray(render_expression_rgb(name, frame, width, height), "RGB")


def render_contact_sheet(cell_width: int = 320, cell_height: int = 240, frame: int = 3):
    from PIL import Image, ImageDraw, ImageFont

    cols = 8
    rows = math.ceil(len(EXPRESSIONS) / cols)
    pad, label_h = 14, 26
    sheet = Image.new("RGB", (cols * (cell_width + pad) + pad, rows * (cell_height + label_h + pad) + pad), (0, 0, 0))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("NotoSansCJK-Regular.ttc", 18)
    except OSError:
        font = ImageFont.load_default()
    for index, item in enumerate(EXPRESSIONS):
        x = pad + (index % cols) * (cell_width + pad)
        y = pad + (index // cols) * (cell_height + label_h + pad)
        sheet.paste(Image.fromarray(render_led_rgb(still_frame(item.name)), "RGB").resize((cell_width, cell_height)), (x, y))
        draw.text((x + 6, y + cell_height + 2), f"{item.index:02d} {item.label} {item.name}", fill=(210, 214, 220), font=font)
    return sheet


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render Xiaoxiaobai LED expressions")
    parser.add_argument("--sheet", type=Path, default=None, help="PNG with every expression")
    parser.add_argument("--name", default=DEFAULT_EXPRESSION)
    parser.add_argument("--out", type=Path, default=None, help="PNG of one frame of --name")
    parser.add_argument("--gif", type=Path, default=None, help="animated GIF of --name")
    args = parser.parse_args()
    if args.sheet is not None:
        args.sheet.parent.mkdir(parents=True, exist_ok=True)
        render_contact_sheet().save(args.sheet)
        print(f"wrote {args.sheet}")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        render_expression_frame(args.name, 3).save(args.out)
        print(f"wrote {args.out}")
    if args.gif is not None:
        from PIL import Image

        animator = new_animator(args.name, 0.0, seed=7)
        images = [Image.fromarray(render_led_rgb(animator.frame(i * EXPRESSION_FRAME_DT_S)), "RGB") for i in range(150)]
        images[0].save(args.gif, save_all=True, append_images=images[1:], duration=40, loop=0)
        print(f"wrote {args.gif}")
