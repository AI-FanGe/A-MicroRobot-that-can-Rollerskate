# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

"""Procedural face for the 64x48 LED dot display.

Features (eyes, brows, mouth, effects) are drawn as signed distances, so every LED gets a
brightness and a feature moving by a fraction of a dot still moves smoothly. On top of an
expression's static `Look`, `FaceAnimator` adds what makes a face feel alive: blinks and eye
saccades at random, human-like intervals, breathing, eased transitions between expressions
(the eyes blink into a new shape) and lips, brows and nods that follow the loudness of what the
robot is saying.

A frame is one byte per LED, row-major: colour index (3 bits) << 5 | brightness (5 bits), the
payload of kMsgFaceLevels in firmware/xiao_face/protocol.h.
"""

from __future__ import annotations

import math
import random
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, fields, replace

import numpy as np

COLS, ROWS = 64, 48
LEVEL_MAX = 31
WHITE, PINK, BLUE, RED, GOLD, GREEN, PURPLE, CYAN = range(8)
# kPalette in firmware/xiao_face/face_render.h
# One colour for the whole face (index 0); the rest stay for old frames and tools.
PALETTE = (
    (255, 236, 210),
    (255, 105, 155),
    (80, 170, 255),
    (255, 72, 48),
    (255, 196, 64),
    (120, 230, 110),
    (190, 120, 255),
    (150, 235, 255),
)

# Features are laid out in a 64x48 design space, enlarged around the centre onto the LEDs.
SCALE = 1.62
_OX, _OY = 32.0, 23.5
EYE_X, EYE_Y = 11.5, 18.8  # eyes at 32 -/+ EYE_X
AA = 0.9  # anti-aliasing ramp, in LEDs

_GRID_Y, _GRID_X = np.mgrid[0:ROWS, 0:COLS].astype(np.float32) + 0.5
# Design-space coordinates of every LED centre.
DX = (32.0 + (_GRID_X - _OX) / SCALE).astype(np.float32)
DY = (24.0 + (_GRID_Y - _OY) / SCALE).astype(np.float32)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if value < lo else hi if value > hi else value


def smooth(t: float) -> float:
    t = _clamp(t)
    return t * t * (3.0 - 2.0 * t)


def _ease_out(t: float) -> float:
    t = _clamp(t)
    return 1.0 - (1.0 - t) ** 3


def _hash(i: int, k: int = 0) -> float:
    """Deterministic pseudo-random number in [0, 1) for cycle `i` of effect `k`."""
    value = math.sin(i * 12.9898 + k * 78.233) * 43758.5453
    return value - math.floor(value)


# ---------------------------------------------------------------- canvas


def _seg_d(px: np.ndarray, py: np.ndarray, ax: float, ay: float, bx: float, by: float) -> np.ndarray:
    vx, vy = bx - ax, by - ay
    length2 = vx * vx + vy * vy
    if length2 < 1e-9:
        return np.hypot(px - ax, py - ay)
    t = np.clip(((px - ax) * vx + (py - ay) * vy) / length2, 0.0, 1.0)
    return np.hypot(px - ax - t * vx, py - ay - t * vy)


def _polyline_d(px: np.ndarray, py: np.ndarray, points: list[tuple[float, float]]) -> np.ndarray:
    dist = None
    for (ax, ay), (bx, by) in zip(points, points[1:]):
        seg = _seg_d(px, py, ax, ay, bx, by)
        dist = seg if dist is None else np.minimum(dist, seg)
    return dist if dist is not None else np.hypot(px - points[0][0], py - points[0][1])


def _polygon_d(px: np.ndarray, py: np.ndarray, points: list[tuple[float, float]]) -> np.ndarray:
    dist = _polyline_d(px, py, points + [points[0]])
    inside = np.zeros(px.shape, dtype=bool)
    xj, yj = points[-1]
    for xi, yi in points:
        span = (yj - yi) or 1e-9
        inside ^= ((yi > py) != (yj > py)) & (px < (xj - xi) * (py - yi) / span + xi)
        xj, yj = xi, yi
    return np.where(inside, -dist, dist)


def _ellipse_d(px: np.ndarray, py: np.ndarray, cx: float, cy: float, rx: float, ry: float) -> np.ndarray:
    rx, ry = max(rx, 0.05), max(ry, 0.05)
    nx, ny = (px - cx) / rx, (py - cy) / ry
    k0 = np.sqrt(nx * nx + ny * ny)
    k1 = np.sqrt((nx / rx) ** 2 + (ny / ry) ** 2)
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-6)


class Canvas:
    """LED brightness (0..1) and colour index; shapes are given in design units."""

    def __init__(self) -> None:
        self.level = np.zeros((ROWS, COLS), dtype=np.float32)
        self.color = np.zeros((ROWS, COLS), dtype=np.uint8)

    @staticmethod
    def window(x0: float, y0: float, x1: float, y1: float) -> tuple[slice, slice] | None:
        pad = 1.2 / SCALE
        c0 = max(0, int(math.floor(_OX + (x0 - pad - 32.0) * SCALE)))
        c1 = min(COLS, int(math.ceil(_OX + (x1 + pad - 32.0) * SCALE)) + 1)
        r0 = max(0, int(math.floor(_OY + (y0 - pad - 24.0) * SCALE)))
        r1 = min(ROWS, int(math.ceil(_OY + (y1 + pad - 24.0) * SCALE)) + 1)
        if c1 <= c0 or r1 <= r0:
            return None
        return slice(r0, r1), slice(c0, c1)

    def paint(self, win, dist: np.ndarray, color: int, alpha: float = 1.0, mode: str = "max") -> None:
        """`dist` in design units (negative inside). mode: max, over (recolour), erase."""
        cov = np.clip(0.5 - dist * (SCALE / AA), 0.0, 1.0)
        if alpha < 1.0:
            cov = cov * max(0.0, alpha)
        level = self.level[win]
        if mode == "erase":
            level *= 1.0 - cov
            return
        colors = self.color[win]
        if mode == "over":
            colors[cov > 0.45] = color
            np.maximum(level, cov, out=level)
            return
        better = cov > level
        level[better] = cov[better]
        colors[better] = color

    def sdf(self, bounds, fn: Callable[[np.ndarray, np.ndarray], np.ndarray], color: int = WHITE,
            alpha: float = 1.0, mode: str = "max") -> None:
        win = self.window(*bounds)
        if win is not None and alpha > 0.003:
            self.paint(win, fn(DX[win], DY[win]), color, alpha, mode)

    def disk(self, x: float, y: float, r: float, color: int = WHITE, alpha: float = 1.0, mode: str = "max") -> None:
        if r > 0:
            self.sdf((x - r, y - r, x + r, y + r), lambda px, py: np.hypot(px - x, py - y) - r, color, alpha, mode)

    def ellipse(self, x: float, y: float, rx: float, ry: float, color: int = WHITE, alpha: float = 1.0,
                mode: str = "max") -> None:
        self.sdf((x - rx, y - ry, x + rx, y + ry), lambda px, py: _ellipse_d(px, py, x, y, rx, ry), color, alpha, mode)

    def ring(self, x: float, y: float, rx: float, ry: float, thick: float, color: int = WHITE,
             alpha: float = 1.0) -> None:
        self.sdf((x - rx - thick, y - ry - thick, x + rx + thick, y + ry + thick),
                 lambda px, py: np.abs(_ellipse_d(px, py, x, y, rx, ry)) - thick, color, alpha)

    def polyline(self, points: list[tuple[float, float]], r: float, color: int = WHITE, alpha: float = 1.0,
                 mode: str = "max") -> None:
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        self.sdf((min(xs) - r, min(ys) - r, max(xs) + r, max(ys) + r),
                 lambda px, py: _polyline_d(px, py, points) - r, color, alpha, mode)

    def capsule(self, x0: float, y0: float, x1: float, y1: float, r: float, color: int = WHITE,
                alpha: float = 1.0, mode: str = "max") -> None:
        self.polyline([(x0, y0), (x1, y1)], r, color, alpha, mode)

    def arc(self, x: float, y: float, rx: float, ry: float, deg0: float, deg1: float, r: float,
            color: int = WHITE, alpha: float = 1.0, steps: int = 14) -> None:
        self.polyline(arc_points(x, y, rx, ry, deg0, deg1, steps), r, color, alpha)

    def polygon(self, points: list[tuple[float, float]], color: int = WHITE, alpha: float = 1.0,
                mode: str = "max", grow: float = 0.0) -> None:
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        self.sdf((min(xs) - grow, min(ys) - grow, max(xs) + grow, max(ys) + grow),
                 lambda px, py: _polygon_d(px, py, points) - grow, color, alpha, mode)

    def to_bytes(self) -> bytes:
        level = np.rint(np.clip(self.level, 0.0, 1.0) * LEVEL_MAX).astype(np.uint8)
        return np.where(level > 0, level | (self.color << 5), 0).astype(np.uint8).tobytes()


def arc_points(x: float, y: float, rx: float, ry: float, deg0: float, deg1: float, steps: int = 14):
    return [
        (x + rx * math.cos(math.radians(deg0 + (deg1 - deg0) * i / steps)),
         y + ry * math.sin(math.radians(deg0 + (deg1 - deg0) * i / steps)))
        for i in range(steps + 1)
    ]


def heart_points(x: float, y: float, s: float, squash: float = 1.0) -> list[tuple[float, float]]:
    out = []
    for i in range(24):
        t = 2.0 * math.pi * i / 24
        hx = 16 * math.sin(t) ** 3
        hy = -(13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t))
        out.append((x + hx / 16.0 * s, y + (hy / 16.0 * s + 0.1 * s) * squash))
    return out


def star_points(x: float, y: float, r: float, inner: float = 0.45, turn: float = 0.0, squash: float = 1.0):
    out = []
    for i in range(10):
        a = turn - math.pi / 2 + i * math.pi / 5
        rr = r if i % 2 == 0 else r * inner
        out.append((x + rr * math.cos(a), y + rr * math.sin(a) * squash))
    return out


def sparkle_points(x: float, y: float, r: float) -> list[tuple[float, float]]:
    """Four-pointed twinkle."""
    k = 0.28
    return [(x, y - r), (x + r * k, y - r * k), (x + r, y), (x + r * k, y + r * k),
            (x, y + r), (x - r * k, y + r * k), (x - r, y), (x - r * k, y - r * k)]


# ---------------------------------------------------------------- feature specs


@dataclass(frozen=True)
class EyeSpec:
    """One eye, described for the left one; the right eye mirrors inner / outer."""

    kind: str = "stroke"  # stroke pill solid ring line happy closed peak chevron heart star x spiral glasses three
    rx: float = 3.2
    ry: float = 3.2
    dx: float = 0.0  # outward
    dy: float = 0.0
    lid_top: float = 0.0  # 0 open .. 1 shut, from above
    lid_tilt: float = 0.0  # >0 lowers the upper lid's inner corner (angry), <0 its outer corner (sad)
    lid_bottom: float = 0.0
    smile: float = 0.0  # lower lid bowed up, as cheeks push it in a smile
    thick: float = 0.85  # stroke kinds
    tilt: float = 0.0  # stroke kinds: inner end lower
    pupil: float = 1.3  # ring kind
    shine: float = 0.0  # dark catch-light, fraction of the radius
    color: int = WHITE
    blink: bool = True
    round: float = 0.6  # pill: corner radius, fraction of the shorter half-size


@dataclass(frozen=True)
class BrowSpec:
    y: float = 11.6
    angle: float = 0.0  # degrees, >0 inner end lower (angry), <0 inner end higher (worried)
    curve: float = 0.0  # >0 arches up
    length: float = 7.0
    thick: float = 0.8
    dx: float = 0.0  # outward
    color: int = WHITE


@dataclass(frozen=True)
class MouthSpec:
    kind: str = "lips"  # lips pill o wave cat kiss
    x: float = 32.0
    y: float = 31.5
    w: float = 10.0  # half width
    curve: float = 4.0  # >0 corners up (smile), <0 down
    open: float = 0.0  # height of the opening
    thick: float = 0.85
    tilt: float = 0.0  # >0 raises the right corner
    tongue: float = 0.0
    tongue_x: float = 0.0
    teeth: float = 0.0
    amp: float = 1.0  # wave
    waves: float = 2.0
    color: int = WHITE


@dataclass(frozen=True)
class ExtraSpec:
    name: str
    x: float | None = None
    y: float | None = None
    scale: float = 1.0
    color: int | None = None


@dataclass(frozen=True)
class MotionSpec:
    breath: float = 0.22
    bounce: float = 0.0
    bounce_hz: float = 0.75
    shake: float = 0.0
    sway: float = 0.25
    gaze: tuple[float, float] = (1.6, 0.8)  # saccade range
    gaze_bias: tuple[float, float] = (0.0, 0.0)
    saccades: float = 1.0  # rate multiplier
    blinks: float = 1.0  # rate multiplier
    slow_blinks: float = 0.0  # share of slow blinks


Anim = Callable[["Pose", float, float, "FaceAnimator"], None]


@dataclass(frozen=True)
class Look:
    eye_l: EyeSpec
    eye_r: EyeSpec
    brow_l: BrowSpec | None
    brow_r: BrowSpec | None
    mouth: MouthSpec | None
    extras: tuple[ExtraSpec, ...] = ()
    motion: MotionSpec = MotionSpec()
    anim: Anim | None = None


@dataclass
class Layer:
    spec: object
    alpha: float = 1.0
    close: float = 0.0  # eyes: forced closing (blink-through on a shape change)


@dataclass
class Pose:
    eye_l: list[Layer]
    eye_r: list[Layer]
    brow_l: list[Layer]
    brow_r: list[Layer]
    mouth: list[Layer]
    extras: dict[str, Layer]
    ox: float = 0.0
    oy: float = 0.0
    bias_x: float = 0.0
    bias_y: float = 0.0
    gaze_x: float = 0.0
    gaze_y: float = 0.0
    blink: float = 0.0
    brow_raise: float = 0.0
    gaze_override: tuple[float, float, float] | None = None  # x, y, weight

    def copy(self) -> Pose:
        dup = replace(self)
        for name in ("eye_l", "eye_r", "brow_l", "brow_r", "mouth"):
            setattr(dup, name, [replace(layer) for layer in getattr(self, name)])
        dup.extras = {key: replace(layer) for key, layer in self.extras.items()}
        return dup


def pose_of(look: Look) -> Pose:
    extras = {}
    for index, spec in enumerate(look.extras):
        extras[f"{spec.name}{index}"] = Layer(spec)
    return Pose(
        eye_l=[Layer(look.eye_l)],
        eye_r=[Layer(look.eye_r)],
        brow_l=[Layer(look.brow_l)] if look.brow_l else [],
        brow_r=[Layer(look.brow_r)] if look.brow_r else [],
        mouth=[Layer(look.mouth)] if look.mouth else [],
        extras=extras,
        bias_x=look.motion.gaze_bias[0],
        bias_y=look.motion.gaze_bias[1],
    )


def lerp_spec(a, b, t: float):
    """Field-wise blend of two specs of the same type (numbers interpolate, the rest switches)."""
    values = {}
    for item in fields(a):
        va, vb = getattr(a, item.name), getattr(b, item.name)
        numeric = isinstance(va, (int, float)) and not isinstance(va, bool) and item.name != "color"
        if numeric and isinstance(vb, (int, float)) and not isinstance(vb, bool):
            values[item.name] = va + (vb - va) * t
        else:
            values[item.name] = vb if t >= 0.5 else va
    return type(a)(**values)


def _fade(layers: list[Layer], k: float) -> list[Layer]:
    return [replace(layer, alpha=layer.alpha * k) for layer in layers if layer.alpha * k > 0.003]


def blend(a: Pose, b: Pose, t: float) -> Pose:
    """`a` turning into `b`; t eased 0..1."""
    out = b.copy()
    for name in ("eye_l", "eye_r"):
        la, lb = getattr(a, name), getattr(b, name)
        if len(la) == 1 and len(lb) == 1 and la[0].spec.kind == lb[0].spec.kind:
            layer = Layer(lerp_spec(la[0].spec, lb[0].spec, t), 1.0, la[0].close + (lb[0].close - la[0].close) * t)
            setattr(out, name, [layer])
        elif t < 0.5:  # blink into the new shape
            k = smooth(t * 2.0)
            setattr(out, name, [replace(layer, close=max(layer.close, k)) for layer in la])
        else:
            k = 1.0 - smooth((t - 0.5) * 2.0)
            setattr(out, name, [replace(layer, close=max(layer.close, k)) for layer in lb])
    for name in ("brow_l", "brow_r"):
        la, lb = getattr(a, name), getattr(b, name)
        if len(la) == 1 and len(lb) == 1:
            setattr(out, name, [Layer(lerp_spec(la[0].spec, lb[0].spec, t), la[0].alpha + (lb[0].alpha - la[0].alpha) * t)])
        elif len(la) == 1 and not lb:  # brows relax and fade away
            spec = la[0].spec
            setattr(out, name, [Layer(replace(spec, angle=spec.angle * (1 - t), curve=spec.curve * (1 - t)), la[0].alpha * (1 - t))])
        elif len(lb) == 1 and not la:
            spec = lb[0].spec
            setattr(out, name, [Layer(replace(spec, angle=spec.angle * t, curve=spec.curve * t), lb[0].alpha * t)])
        else:
            setattr(out, name, _fade(la, 1 - t) + _fade(lb, t))
    ma, mb = a.mouth, b.mouth
    if len(ma) == 1 and len(mb) == 1 and ma[0].spec.kind == mb[0].spec.kind:
        out.mouth = [Layer(lerp_spec(ma[0].spec, mb[0].spec, t))]
    else:
        out.mouth = _fade(ma, 1 - t) + _fade(mb, t)
    extras: dict[str, Layer] = {}
    for key in a.extras.keys() | b.extras.keys():
        ea, eb = a.extras.get(key), b.extras.get(key)
        if ea and eb:
            extras[key] = replace(eb, alpha=ea.alpha + (eb.alpha - ea.alpha) * t)
        elif ea:
            extras[key] = replace(ea, alpha=ea.alpha * (1 - t))
        else:
            extras[key] = replace(eb, alpha=eb.alpha * t)
    out.extras = extras
    for name in ("ox", "oy", "bias_x", "bias_y", "brow_raise"):
        va, vb = getattr(a, name), getattr(b, name)
        setattr(out, name, va + (vb - va) * t)
    return out


# ---------------------------------------------------------------- drawing


def _eye_center(spec: EyeSpec, side: int, pose: Pose, follow: float) -> tuple[float, float]:
    gaze_x, gaze_y = pose.gaze_x + pose.bias_x, pose.gaze_y + pose.bias_y
    x = 32.0 + side * EYE_X + spec.dx * side + pose.ox + gaze_x * follow
    y = EYE_Y + spec.dy + pose.oy + gaze_y * follow * 0.8
    return x, y


def _lidded(canvas: Canvas, s: EyeSpec, side: int, x: float, y: float, rx: float, ry: float, close: float,
            gaze_y: float, alpha: float, ring: bool, pose: Pose) -> None:
    inner = -side
    room = max(0.0, 1.0 - s.lid_top - s.lid_bottom)
    top = s.lid_top + room * close * 0.84 + max(0.0, gaze_y) * 0.05
    bottom = s.lid_bottom + room * close * 0.16
    opening = 1.0 - top - bottom
    y_top = y - ry + 2.0 * ry * top
    y_bot = y + ry - 2.0 * ry * bottom
    tilt, smile = s.lid_tilt, s.smile * ry

    def lid_d(px: np.ndarray, py: np.ndarray) -> np.ndarray:
        u = (px - x) / max(rx, 0.1)
        d_top = (y_top + tilt * u * inner) - py
        d_bot = py - (y_bot - smile * np.clip(1.0 - u * u, 0.0, 1.0))
        return np.maximum(d_top, d_bot)

    bounds = (x - rx - 1, y - ry - 1, x + rx + 1, y + ry + 1)
    if opening > 0.02:
        if ring:
            limit = max(0.6, rx - s.pupil - 0.5)
            px_ = x + _clamp((pose.gaze_x + pose.bias_x) * 0.9, -limit, limit)
            py_ = y + _clamp((pose.gaze_y + pose.bias_y) * 0.8, -ry * 0.55, ry * 0.55)
            canvas.sdf(bounds, lambda px, py: np.maximum(np.abs(_ellipse_d(px, py, x, y, rx, ry)) - s.thick,
                                                         lid_d(px, py)), s.color, alpha)
            canvas.sdf(bounds, lambda px, py: np.maximum.reduce([np.hypot(px - px_, py - py_) - s.pupil,
                                                                 _ellipse_d(px, py, x, y, rx, ry), lid_d(px, py)]),
                       s.color, alpha)
        else:
            canvas.sdf(bounds, lambda px, py: np.maximum(_ellipse_d(px, py, x, y, rx, ry), lid_d(px, py)), s.color, alpha)
            if s.shine > 0 and opening > 0.55:
                r = s.shine * min(rx, ry)
                canvas.disk(x - 0.36 * rx, y - 0.4 * ry, r, s.color, alpha * smooth((opening - 0.55) / 0.3), "erase")
    shut = smooth((0.2 - opening) / 0.16)
    if shut > 0.0:
        meet = y_top + (y_bot - y_top) * 0.5 if opening > 0 else y_top - opening * ry
        w = rx * 1.05
        canvas.polyline([(x - w, meet - 0.4), (x - w * 0.5, meet + 0.25), (x, meet + 0.45),
                         (x + w * 0.5, meet + 0.25), (x + w, meet - 0.4)], 0.55, s.color, alpha * shut)


def draw_eye(canvas: Canvas, s: EyeSpec, side: int, alpha: float, close: float, pose: Pose, t: float) -> None:
    kind = s.kind
    blink = pose.blink if s.blink else 0.0
    shut = max(blink, close)
    follow = {"stroke": 1.0, "pill": 1.0, "solid": 1.0, "heart": 0.8, "star": 0.8, "ring": 0.25}.get(kind, 0.45)
    x, y = _eye_center(s, side, pose, follow)
    inner = -side
    if kind == "stroke":
        _stroke(canvas, s, side, x, y, shut, alpha)
        return
    if kind == "pill":
        _pill(canvas, s, side, x, y, shut, alpha, pose)
        return
    if kind in ("solid", "ring"):
        _lidded(canvas, s, side, x, y, s.rx, s.ry, shut, pose.gaze_y + pose.bias_y, alpha, kind == "ring", pose)
        return
    squash = 1.0 - shut
    if squash < 0.12:
        canvas.capsule(x - s.rx, y, x + s.rx, y, 0.55, s.color, alpha * smooth((0.2 - squash) / 0.12))
        if squash < 0.04:
            return
    ry = s.ry * squash
    tilt = s.tilt
    if kind == "line":
        canvas.capsule(x - s.rx, y - tilt * inner, x + s.rx, y + tilt * inner, s.thick * max(0.6, squash), s.color, alpha)
    elif kind == "happy":
        pts = arc_points(x, y + s.ry * 0.35, s.rx, ry, 200, 340, 12)
        canvas.polyline([(px, py + tilt * (px - x) / s.rx * inner) for px, py in pts], s.thick, s.color, alpha)
    elif kind == "closed":
        pts = arc_points(x, y - s.ry * 0.35, s.rx, ry, 20, 160, 12)
        canvas.polyline([(px, py + tilt * (px - x) / s.rx * inner) for px, py in pts], s.thick, s.color, alpha)
    elif kind == "peak":
        canvas.polyline([(x - s.rx, y + ry * 0.5), (x, y - ry * 0.5), (x + s.rx, y + ry * 0.5)], s.thick, s.color, alpha)
    elif kind == "chevron":
        tip = x + inner * s.rx * 0.55
        back = x - inner * s.rx * 0.55
        canvas.polyline([(back, y - ry), (tip, y), (back, y + ry)], s.thick, s.color, alpha)
    elif kind == "heart":
        beat = _heartbeat(t)
        canvas.polygon(heart_points(x, y, s.rx * (1.0 + 0.14 * beat), squash), s.color, alpha)
    elif kind == "star":
        twinkle = 1.0 + 0.1 * math.sin(t * 5.3 + side)
        canvas.polygon(star_points(x, y, s.rx * twinkle, 0.46, 0.25 * math.sin(t * 1.7 + side), squash), s.color, alpha)
    elif kind == "x":
        canvas.capsule(x - s.rx, y - ry, x + s.rx, y + ry, s.thick, s.color, alpha)
        canvas.capsule(x - s.rx, y + ry, x + s.rx, y - ry, s.thick, s.color, alpha)
    elif kind == "spiral":
        turn = -t * 5.0 * side
        pts = [(x + (0.3 + (s.rx - 0.3) * i / 28) * math.cos(turn + i * 0.45 * side),
                y + (0.3 + (s.rx - 0.3) * i / 28) * math.sin(turn + i * 0.45 * side) * squash) for i in range(29)]
        canvas.polyline(pts, s.thick, s.color, alpha)
    elif kind == "three":
        lobe = ry * 0.62
        a0, a1 = (80, 280) if side < 0 else (-80, 80)
        canvas.arc(x, y - lobe, s.rx * 0.7, lobe, a0, a1, s.thick, s.color, alpha, 10)
        canvas.arc(x, y + lobe, s.rx * 0.7, lobe, a0, a1, s.thick, s.color, alpha, 10)
    elif kind == "glasses" and side < 0:
        _glasses(canvas, s, pose, alpha, t, squash)


def _stroke(canvas: Canvas, s: EyeSpec, side: int, x: float, y: float, shut: float, alpha: float) -> None:
    """Single-line eye: a short upright stroke that lids shorten and slant; nearly shut it turns into a dash."""
    inner = -side
    ry = s.ry * 0.78  # leaves room for the brows
    top = y - ry + 2.0 * ry * s.lid_top
    bottom = y + ry - 2.0 * ry * s.lid_bottom - ry * s.smile * 0.5
    mid = (top + bottom) / 2.0
    half = max(0.0, (bottom - top) / 2.0) * (1.0 - shut)
    openness = half / max(ry, 0.1)
    upright = 1.0 - smooth((0.3 - openness) / 0.12)
    dash = smooth((0.3 - openness) / 0.15)
    if half > 0.05 and upright > 0.0:
        k = half / max(ry, 0.1)
        x_top = x - inner * s.lid_tilt * 0.4 * k  # angry \ /, sad / \
        x_bottom = x + inner * s.lid_tilt * 0.15 * k
        canvas.capsule(x_top, mid - half, x_bottom, mid + half, s.thick, s.color, alpha * upright)
    if dash > 0.0:
        w = s.rx * 0.8 * (0.3 + 0.7 * dash)  # the stroke folds over into a dash
        canvas.capsule(x - w, mid, x + w, mid, s.thick * 0.85, s.color, alpha * dash)


def _pill(canvas: Canvas, s: EyeSpec, side: int, x: float, y: float, shut: float, alpha: float, pose: Pose) -> None:
    """Rounded-rectangle eye: blinks squash it to a line, lids cut it at an angle, `smile` arches its bottom."""
    inner = -side
    toward = _clamp((pose.gaze_x + pose.bias_x) / 2.5, -1.0, 1.0) * side
    grow = 1.0 + 0.08 * toward  # the eye on the side being looked at comes a little closer
    h = max(0.5, s.ry * grow * (1.0 - shut))
    w = s.rx * grow * (1.0 + 0.12 * shut)
    rad = max(0.4, min(w, h) * s.round)
    top = y - h + 2.0 * h * s.lid_top
    bottom = y + h - 2.0 * h * s.lid_bottom
    cut_r = w * 1.6
    cut_y = y + h + cut_r - 2.0 * h * s.smile

    def dist(px: np.ndarray, py: np.ndarray) -> np.ndarray:
        qx = np.abs(px - x) - (w - rad)
        qy = np.abs(py - y) - (h - rad)
        d = np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0)) + np.minimum(np.maximum(qx, qy), 0.0) - rad
        u = (px - x) / w
        if s.lid_top > 0 or s.lid_tilt:
            d = np.maximum(d, top + s.lid_tilt * u * inner - py)
        if s.lid_bottom > 0:
            d = np.maximum(d, py - bottom)
        if s.smile > 0:
            d = np.maximum(d, cut_r - np.hypot(px - x, py - cut_y))
        return d

    canvas.sdf((x - w - 1, y - h - 1, x + w + 1, y + h + 1), dist, s.color, alpha)


def _glasses(canvas: Canvas, s: EyeSpec, pose: Pose, alpha: float, t: float, squash: float) -> None:
    y = EYE_Y + s.dy + pose.oy + (pose.gaze_y + pose.bias_y) * 0.15
    ox = pose.ox + (pose.gaze_x + pose.bias_x) * 0.15
    h = s.ry * squash
    lenses = []
    for cx in (32.0 - EYE_X + ox, 32.0 + EYE_X + ox):
        lens = [(cx - s.rx, y - h), (cx + s.rx, y - h), (cx + s.rx * 0.8, y + h * 0.7), (cx, y + h),
                (cx - s.rx * 0.8, y + h * 0.7)]
        lenses.append(lens)
        canvas.polygon(lens, s.color, alpha, grow=0.4)
    canvas.capsule(32.0 - EYE_X + s.rx + ox, y - h * 0.6, 32.0 + EYE_X - s.rx + ox, y - h * 0.6, 0.6, s.color, alpha)
    sweep = (t % 4.5) / 0.9  # a glint crosses the lenses every few seconds
    if sweep < 1.0:
        gx = 32.0 - EYE_X - s.rx - 2 + (2 * EYE_X + 2 * s.rx + 4) * sweep + ox
        canvas.capsule(gx - 1.5, y + h + 0.5, gx + 1.5, y - h - 0.5, 0.45, s.color, alpha, "erase")


def _heartbeat(t: float) -> float:
    phase = t * 1.15 % 1.0  # lub-dub
    return math.exp(-((phase - 0.08) / 0.05) ** 2) + 0.6 * math.exp(-((phase - 0.27) / 0.05) ** 2)


def draw_brow(canvas: Canvas, s: BrowSpec, side: int, alpha: float, pose: Pose) -> None:
    inner = -side
    bx = 32.0 + side * EYE_X + s.dx * side + pose.ox + (pose.gaze_x + pose.bias_x) * 0.25
    by = s.y + pose.oy - pose.brow_raise + min(0.0, pose.gaze_y + pose.bias_y) * 0.3
    by = max(by, 10.2 + 2.0 * max(0.0, s.curve))  # stay on screen
    half = s.length / 2.0
    rise = half * math.tan(math.radians(s.angle))
    ix, iy = bx + inner * half, by + rise
    ox, oy = bx - inner * half, by - rise
    mx, my = (ix + ox) / 2.0, (iy + oy) / 2.0 - s.curve * 2.0
    pts = []
    for i in range(9):
        u = i / 8
        pts.append(((1 - u) ** 2 * ox + 2 * (1 - u) * u * mx + u * u * ix, (1 - u) ** 2 * oy + 2 * (1 - u) * u * my + u * u * iy))
    canvas.polyline(pts, s.thick, s.color, alpha)


def _lip_curves(s: MouthSpec, x0: float, y0: float):
    open_k = _clamp(s.open / 6.0)

    def upper(u: np.ndarray | float):
        return y0 + s.curve * (0.5 - u * u) * (1.0 - 0.45 * open_k) - s.tilt * u * 0.5 - s.open * 0.2

    def lower(u: np.ndarray | float):
        bow = np.clip(1.0 - u * u, 0.0, 1.0) if isinstance(u, np.ndarray) else max(0.0, 1.0 - u * u)
        return upper(u) + s.open * bow ** 0.75 + max(0.0, s.curve) * 0.3 * bow * open_k

    return upper, lower


def draw_mouth(canvas: Canvas, s: MouthSpec, alpha: float, pose: Pose, t: float) -> None:
    x0, y0 = s.x + pose.ox, s.y + pose.oy
    w = max(0.6, s.w)
    if s.kind == "lips":
        upper, lower = _lip_curves(s, x0, y0)
        us = np.linspace(-1.0, 1.0, 15)
        top = [(x0 + w * u, float(upper(u))) for u in us]
        bounds_y0 = min(p[1] for p in top) - 1
        bounds_y1 = float(lower(0.0)) + 1
        if s.open > 0.25:
            def fill(px: np.ndarray, py: np.ndarray) -> np.ndarray:
                u = (px - x0) / w
                return np.maximum.reduce([upper(u) - py, py - lower(u), np.abs(px - x0) - w])
            canvas.sdf((x0 - w, bounds_y0, x0 + w, bounds_y1), fill, s.color, alpha)
            bottom = [(x0 + w * u, float(lower(u))) for u in us]
            canvas.polyline(bottom, s.thick * 0.9, s.color, alpha)
            if s.teeth > 0 and s.open > 1.2:
                mid = [(x0 + w * 0.85 * u, (float(upper(u)) + float(lower(u))) / 2.0) for u in np.linspace(-1, 1, 9)]
                canvas.polyline(mid, 0.32, s.color, alpha * s.teeth, "erase")
                for u in (-0.5, 0.0, 0.5):
                    canvas.capsule(x0 + w * u, float(upper(u)) + 0.5, x0 + w * u, float(lower(u)) - 0.5, 0.28,
                                   s.color, alpha * s.teeth, "erase")
        canvas.polyline(top, s.thick, s.color, alpha)
        if s.tongue > 0:
            tx = x0 + s.tongue_x
            if s.open > 2.0:
                ty = float(lower(0.0)) - s.open * 0.25
                canvas.ellipse(tx, ty, w * 0.42 * s.tongue, s.open * 0.32 * s.tongue, s.color, alpha * 0.55, "erase")
            else:
                ty = float(upper(s.tongue_x / w)) + 1.7 * s.tongue
                canvas.ellipse(tx, ty, 2.2 * s.tongue, 2.3 * s.tongue, s.color, alpha * 0.5)
    elif s.kind == "pill":
        h = max(s.thick * 0.8, s.open / 2.0)
        rad = min(w, h)
        y = y0 + s.open * 0.15

        def pill(px: np.ndarray, py: np.ndarray) -> np.ndarray:
            qx = np.abs(px - x0) - (w - rad)
            qy = np.abs(py - y) - (h - rad)
            return np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0)) + np.minimum(np.maximum(qx, qy), 0.0) - rad

        canvas.sdf((x0 - w, y - h, x0 + w, y + h), pill, s.color, alpha)
    elif s.kind == "o":
        ry = max(0.9, s.open / 2.0)
        canvas.ring(x0, y0 + ry * 0.3, w, ry, s.thick * 0.8, s.color, alpha)
    elif s.kind == "wave":
        phase = t * 2.2
        pts = [(x0 + w * u, y0 + s.amp * math.sin(math.pi * s.waves * u + phase) - s.curve * (u * u - 0.5) * 0.5)
               for u in np.linspace(-1.0, 1.0, 21)]
        canvas.polyline(pts, s.thick, s.color, alpha)
    elif s.kind == "cat":
        canvas.arc(x0 - w / 2, y0, w / 2, max(1.2, s.curve), 20, 160, s.thick, s.color, alpha, 10)
        canvas.arc(x0 + w / 2, y0, w / 2, max(1.2, s.curve), 20, 160, s.thick, s.color, alpha, 10)
        if s.open > 0.3:
            canvas.ellipse(x0, y0 + max(1.2, s.curve) + s.open * 0.5, w * 0.35, s.open * 0.5, s.color, alpha)
    elif s.kind == "kiss":
        canvas.arc(x0, y0 - 1.2, w, 2.2, 40, 140, s.thick, s.color, alpha, 10)
        canvas.arc(x0, y0 + 1.2, w, 2.2, 220, 320, s.thick, s.color, alpha, 10)


# ---------------------------------------------------------------- effects


def _drop(canvas: Canvas, x: float, y: float, r: float, color: int, alpha: float) -> None:
    canvas.polygon([(x, y - r * 2.1), (x + r * 0.75, y - r * 0.4), (x, y + r), (x - r * 0.75, y - r * 0.4)], color, alpha, grow=r * 0.25)
    canvas.disk(x, y, r, color, alpha)


def _z(canvas: Canvas, x: float, y: float, s: float, color: int, alpha: float) -> None:
    canvas.polyline([(x - s, y - s), (x + s, y - s), (x - s, y + s), (x + s, y + s)], 0.45 + 0.1 * s, color, alpha)


def draw_extra(canvas: Canvas, spec: ExtraSpec, alpha: float, pose: Pose, t: float, seed: int) -> None:
    name, k = spec.name, spec.scale
    ox, oy = pose.ox, pose.oy

    def at(x: float, y: float) -> tuple[float, float]:
        return (spec.x if spec.x is not None else x) + ox, (spec.y if spec.y is not None else y) + oy

    def col(default: int) -> int:
        return spec.color if spec.color is not None else default

    if name == "blush":
        flicker = 0.75 + 0.25 * math.sin(t * 1.3 + seed)
        for side in (-1, 1):
            bx, by = at(32 + side * (EYE_X + 4.0), 26.0)
            for i in (-1, 0, 1):
                cx = bx + i * 1.7
                canvas.capsule(cx + 0.6, by - 0.8, cx - 0.6, by + 0.8, 0.42 * k, col(WHITE), alpha * flicker)
    elif name == "tears":
        for side, offset in ((-1, 0.0), (1, 0.47)):
            phase = (t * 0.9 + offset + _hash(seed, side)) % 1.0
            x, y = at(32 + side * EYE_X + side * 2.0, EYE_Y + 4.5)
            drop_y = y + 9.0 * phase * phase
            _drop(canvas, x + side * phase * 0.8, drop_y, 1.0 * k, col(WHITE), alpha * (1.0 - smooth((phase - 0.75) / 0.25)))
            canvas.capsule(x, y, x + side * 0.3, min(drop_y - 1, y + 3.5), 0.45, col(WHITE), alpha * 0.7)
    elif name == "tear_well":
        shimmer = 0.75 + 0.25 * math.sin(t * 3.1 + seed)
        for side in (-1, 1):  # a tear held at the outer corner
            x, y = at(32 + side * (EYE_X + 2.6), EYE_Y + 3.4)
            _drop(canvas, x, y, 0.75 * k, col(WHITE), alpha * shimmer)
        cycle = int(t / 4.5)
        phase = (t / 4.5) % 1.0
        if _hash(cycle, seed) < 0.6 and phase < 0.7:
            side = 1 if _hash(cycle, seed + 1) < 0.5 else -1
            x, y = at(32 + side * (EYE_X + 2.0), EYE_Y + 4.5)
            _drop(canvas, x, y + 8 * (phase / 0.7) ** 2, 0.9, col(WHITE), alpha * (1 - smooth((phase - 0.5) / 0.2)))
    elif name == "sweat":
        phase = (t / 2.6 + _hash(seed)) % 1.0
        x, y = at(49.5, 13.5)
        _drop(canvas, x, y + 4.0 * smooth(phase), 1.25 * k, col(WHITE), alpha * (1.0 - smooth((phase - 0.8) / 0.2)) * smooth(phase / 0.12))
    elif name == "zzz":
        for i in range(3):
            phase = (t / 3.0 + i / 3.0) % 1.0
            x, y = at(44.0, 16.0)
            _z(canvas, x + 5 * phase, y - 5.5 * phase, (0.7 + 1.1 * phase) * k, col(WHITE),
               alpha * smooth(phase / 0.2) * (1 - smooth((phase - 0.7) / 0.3)))
    elif name == "hearts":
        for i in range(2):
            phase = (t / 2.4 + i / 2.0) % 1.0
            x, y = at(47.5, 23.0)
            hx = x + 1.2 * math.sin(phase * 6.0 + i) + i * 2.0
            canvas.polygon(heart_points(hx, y - 10 * phase, (1.1 + 0.7 * phase) * k), col(WHITE),
                           alpha * smooth(phase / 0.15) * (1 - smooth((phase - 0.7) / 0.3)))
    elif name == "sparkles":
        spots = ((13.5, 12.0), (50.5, 12.5), (32.0, 10.5), (13.5, 30.0), (50.5, 31.0), (45.0, 10.5), (19.0, 10.5))
        for i in range(3):
            period = 1.6 + 0.4 * i
            cycle = int(t / period + i * 0.37)
            phase = (t / period + i * 0.37) % 1.0
            sx, sy = spots[int(_hash(cycle, seed + i) * len(spots))]
            x, y = at(sx, sy)
            size = math.sin(math.pi * phase) * 1.9 * k
            if size > 0.2:
                canvas.polygon(sparkle_points(x, y, size), col(WHITE), alpha, grow=0.15)
    elif name == "anger":
        pulse = 1.0 + 0.18 * max(0.0, math.sin(t * 7.0))
        x, y = at(49.0, 15.5)
        r = 2.0 * k * pulse
        for qx, qy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            cx, cy = x + qx * r * 0.62, y + qy * r * 0.62
            start = {(-1, -1): 0, (1, -1): 90, (1, 1): 180, (-1, 1): 270}[(qx, qy)]
            canvas.arc(cx + qx * r * 0.4, cy + qy * r * 0.4, r * 0.5, r * 0.5, start, start + 90, 0.5, col(WHITE), alpha, 6)
    elif name in ("question", "exclaim"):
        bob = 0.6 * math.sin(t * 2.4)
        x, y = at(49.5, 15.5)
        y += bob * 0.6
        if name == "question":
            canvas.arc(x, y - 1.6, 1.8 * k, 1.8 * k, 180, 405, 0.55, col(WHITE), alpha, 12)
            canvas.capsule(x + 1.25 * k, y - 0.3, x, y + 1.0, 0.55, col(WHITE), alpha)
            canvas.disk(x, y + 3.0 * k, 0.65, col(WHITE), alpha)
        else:
            jolt = 1.0 + 0.25 * math.exp(-((t * 2.0) % 1.6) * 6)
            canvas.capsule(x, y - 3.2 * k * jolt, x, y + 0.8, 0.75, col(WHITE), alpha)
            canvas.disk(x, y + 3.0 * k, 0.75, col(WHITE), alpha)
    elif name == "notes":
        for i in range(2):
            phase = (t / 2.2 + i / 2.0) % 1.0
            x, y = at(45.5, 25.0)
            nx = x + 4 * phase + 1.0 * math.sin(phase * 7 + i)
            ny = y - 10 * phase
            a = alpha * smooth(phase / 0.15) * (1 - smooth((phase - 0.7) / 0.3))
            canvas.ellipse(nx, ny, 1.1 * k, 0.85 * k, col(WHITE), a)
            canvas.capsule(nx + 0.9 * k, ny, nx + 0.9 * k, ny - 3.6 * k, 0.35, col(WHITE), a)
            canvas.capsule(nx + 0.9 * k, ny - 3.6 * k, nx + 2.4 * k, ny - 2.6 * k, 0.35, col(WHITE), a)
    elif name == "drool":
        phase = (t / 3.0 + _hash(seed)) % 1.0
        x, y = at(39.5, 34.5)
        length = 1.0 + 4.0 * smooth(phase / 0.8)
        a = alpha * (1 - smooth((phase - 0.85) / 0.15))
        canvas.capsule(x, y, x, y + length, 0.45, col(WHITE), a)
        canvas.disk(x, y + length + 0.4, 0.9, col(WHITE), a)
    elif name == "dots":
        x, y = at(47.6, 19.0)
        phase = (t / 1.8) % 1.0
        for i in range(3):
            on = smooth((phase * 4 - i) / 0.5) * (1 - smooth((phase - 0.85) / 0.15))
            canvas.disk(x + i * 1.9, y - i * 0.9, 0.8 * k, col(WHITE), alpha * on)
    elif name == "whiskers":
        for side in (-1, 1):
            x, y = at(32 + side * 15.0, 30.5)
            for i in (-1, 0, 1):
                canvas.capsule(x, y + i * 1.6, x + side * 3.6, y + i * 2.4 - 0.3, 0.35, col(WHITE), alpha * 0.85)
    elif name == "horns":
        for side in (-1, 1):
            x, y = at(32 + side * 18.2, 11.6)
            canvas.polygon([(x - 1.4, y + 1.5), (x + side * 1.8, y - 2.2), (x + 1.4, y + 1.5)], col(WHITE), alpha, grow=0.2)
    elif name == "puff":
        phase = (t / 3.2) % 1.0
        x, y = at(43.0, 32.0)
        for i in range(3):
            p = phase - i * 0.08
            if 0 < p < 0.8:
                canvas.ring(x + 6.5 * p + i, y - 2 * p - i * 0.4, 0.8 + 1.2 * p, 0.7 + 1.0 * p, 0.35, col(WHITE),
                            alpha * (1 - smooth(p / 0.8)))
    elif name == "steam":
        for side in (-1, 1):
            phase = (t / 1.6 + (0.5 if side > 0 else 0.0)) % 1.0
            x, y = at(32 + side * 18.0, 13.5)
            canvas.disk(x + side * 1.5 * phase, y - 3 * phase, (0.7 + 1.1 * phase) * k, col(WHITE),
                        alpha * 0.8 * (1 - smooth(phase)))
    elif name == "bulb":
        x, y = at(49.2, 17.5)
        glow = 0.55 + 0.45 * abs(math.sin(t * 3.0))
        canvas.ring(x, y - 1.0, 1.8 * k, 1.9 * k, 0.42, col(WHITE), alpha)
        canvas.disk(x, y - 1.0, 0.75 * k * glow, col(WHITE), alpha * glow)
        canvas.capsule(x - 0.8 * k, y + 1.6 * k, x + 0.8 * k, y + 1.6 * k, 0.38, col(WHITE), alpha)
        canvas.capsule(x - 0.5 * k, y + 2.5 * k, x + 0.5 * k, y + 2.5 * k, 0.38, col(WHITE), alpha)
        for deg in (-135.0, -90.0, -45.0):
            c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
            canvas.capsule(x + 2.6 * k * c, y - 1.0 + 2.6 * k * s, x + 3.3 * k * c, y - 1.0 + 3.3 * k * s, 0.34,
                           col(WHITE), alpha * glow)
    elif name == "check":
        x, y = at(48.6, 15.0)
        s = 1.6 * k * (1.0 + 0.3 * math.exp(-(t % 3.0) * 5.0))
        canvas.polyline([(x - s, y), (x - 0.35 * s, y + 0.75 * s), (x + 1.1 * s, y - 0.95 * s)], 0.55, col(WHITE), alpha)
    elif name in ("battery_low", "battery_charge"):
        x, y = at(32.0, 12.6)
        w, h = 3.8 * k, 2.0 * k
        canvas.polyline([(x - w, y - h), (x + w, y - h), (x + w, y + h), (x - w, y + h), (x - w, y - h)], 0.35,
                        col(WHITE), alpha)
        canvas.capsule(x + w + 0.8, y - 0.5 * h, x + w + 0.8, y + 0.5 * h, 0.42, col(WHITE), alpha)
        if name == "battery_low":  # the last bar blinking
            bars, a = 1, alpha * (0.2 + 0.8 * (math.sin(t * 5.0) > 0))
        else:  # filling up, bar by bar
            bars, a = 1 + int(t / 0.6) % 4, alpha
        cell = (2 * w - 1.2) / 4
        for i in range(bars):
            cx = x - w + 0.6 + cell * (i + 0.5)
            canvas.capsule(cx, y - h + 1.1, cx, y + h - 1.1, 0.4, col(WHITE), a)
    elif name == "wind":
        for side in (-1, 1):
            for i, (row, length) in enumerate(((25.5, 3.0), (29.0, 2.2), (32.5, 2.6))):
                phase = (t * 1.8 + i * 0.37 + (0.5 if side > 0 else 0.0)) % 1.0
                x, y = at(32 + side * (13.5 + 3.0 * phase), row)
                canvas.capsule(x, y, x + side * length * k, y, 0.38, col(WHITE), alpha * math.sin(math.pi * phase))
    elif name == "bell":  # an alarm going off: a shaking bell between the brows
        ringing = (t % 1.8) < 0.9
        x, y = at(32.0, 13.6)
        x += 0.45 * math.sin(t * 45.0) * ringing
        canvas.arc(x, y, 2.0 * k, 2.4 * k, 180.0, 360.0, 0.42, col(WHITE), alpha, 10)
        canvas.capsule(x - 2.0 * k, y, x - 2.5 * k, y + 1.2 * k, 0.42, col(WHITE), alpha)
        canvas.capsule(x + 2.0 * k, y, x + 2.5 * k, y + 1.2 * k, 0.42, col(WHITE), alpha)
        canvas.capsule(x - 2.6 * k, y + 1.3 * k, x + 2.6 * k, y + 1.3 * k, 0.42, col(WHITE), alpha)
        canvas.disk(x + 0.5 * math.sin(t * 45.0) * ringing, y + 2.3 * k, 0.6 * k, col(WHITE), alpha)
        if ringing:
            for side in (-1, 1):
                canvas.arc(x + side * 1.0, y, 3.6 * k, 2.4 * k, 90 - side * 90 - 35, 90 - side * 90 + 35, 0.36,
                           col(WHITE), alpha * (0.4 + 0.6 * (math.sin(t * 25.0) > 0)), 6)
    elif name == "loading":
        x, y = at(47.8, 26.0)
        head = t * 10.0
        for i in range(8):
            ang = 2 * math.pi * i / 8 - math.pi / 2
            lag = (head - i) % 8
            canvas.disk(x + 2.3 * k * math.cos(ang), y + 2.3 * k * math.sin(ang), 0.72 * k, col(WHITE),
                        alpha * max(0.3, 1.0 - lag / 6.0))
    elif name == "stars_spin":
        for i in range(3):
            a = t * 2.4 + i * 2.094
            x, y = at(32.0, 11.0)
            canvas.polygon(star_points(x + 13 * math.cos(a), y + 1.2 * math.sin(a), 1.3 * k, 0.45, a), col(WHITE),
                           alpha * (0.55 + 0.45 * math.sin(a)))


def draw(pose: Pose, t: float) -> Canvas:
    canvas = Canvas()
    for key, layer in pose.extras.items():
        draw_extra(canvas, layer.spec, layer.alpha, pose, t, sum(map(ord, key)))
    for side, layers in ((-1, pose.eye_l), (1, pose.eye_r)):
        for layer in layers:
            draw_eye(canvas, layer.spec, side, layer.alpha, layer.close, pose, t)
    for side, layers in ((-1, pose.brow_l), (1, pose.brow_r)):
        for layer in layers:
            draw_brow(canvas, layer.spec, side, layer.alpha, pose)
    for layer in pose.mouth:
        draw_mouth(canvas, layer.spec, layer.alpha, pose, t)
    return canvas


# ---------------------------------------------------------------- life: noise, blinks, saccades, speech


class _Noise:
    """Smooth 1-D noise in about [-1, 1]: a few sines at unrelated frequencies."""

    def __init__(self, rng: random.Random, hz: float) -> None:
        self._parts = [(hz * m * rng.uniform(0.85, 1.15), rng.uniform(0, 2 * math.pi), a)
                       for m, a in ((1.0, 1.0), (2.37, 0.5), (4.11, 0.25))]

    def __call__(self, t: float) -> float:
        return sum(a * math.sin(2 * math.pi * f * t + p) for f, p, a in self._parts) / 1.75


class Blinker:
    """Human blinks: ~every 3 s at random (log-normal), sometimes double, a fast close and a slower open."""

    def __init__(self, rng: random.Random, now: float) -> None:
        self._rng = rng
        self._next = now + rng.uniform(0.8, 2.4)
        self._start: float | None = None
        self._slow = 1.0
        self._wake: tuple[float, float] | None = None

    def wake(self, now: float, closed: float = 0.45, opening: float = 0.7) -> None:
        """Eyes shut for `closed` seconds, then opening slowly, like waking up."""
        self._wake = (now + closed, opening)
        self._start = None

    def interval(self, rate: float) -> float:
        return _clamp(math.exp(self._rng.gauss(math.log(2.9), 0.55)) / max(0.05, rate), 0.5, 14.0)

    def blink_now(self, now: float, slow: bool = False) -> None:
        if self._start is None:
            self._start = now
            self._slow = 1.9 if slow else self._rng.uniform(0.9, 1.15)

    def value(self, now: float, rate: float, slow_share: float = 0.0) -> float:
        if self._wake is not None:
            opens, opening = self._wake
            if now < opens:
                return 1.0
            if now < opens + opening:
                return 1.0 - _ease_out((now - opens) / opening)
            self._wake = None
            self._next = now + self._rng.uniform(0.3, 0.7)  # a first blink soon after
            return 0.0
        if self._start is None and now >= self._next:
            self.blink_now(now, self._rng.random() < slow_share)
        if self._start is None:
            return 0.0
        close, hold, reopen = 0.075 * self._slow, 0.045 * self._slow, 0.15 * self._slow
        e = now - self._start
        if e < close:
            return smooth(e / close) ** 0.8
        if e < close + hold:
            return 1.0
        if e < close + hold + reopen:
            return 1.0 - _ease_out((e - close - hold) / reopen)
        self._start = None
        double = self._rng.random() < 0.13
        self._next = now + (self._rng.uniform(0.12, 0.25) if double else self.interval(rate))
        return 0.0


class Gaze:
    """Fixations of random length joined by fast saccades, with tiny drifts in between."""

    def __init__(self, rng: random.Random, now: float) -> None:
        self._rng = rng
        self.x = self.y = 0.0
        self._from = (0.0, 0.0)
        self._to = (0.0, 0.0)
        self._t0 = now
        self._dur = 0.05
        self._next = now + rng.uniform(0.4, 1.5)
        self._drift = (_Noise(rng, 0.35), _Noise(rng, 0.3))

    def kick(self, now: float, delay: float = 0.0) -> None:
        self._next = min(self._next, now + delay)

    def update(self, now: float, motion: MotionSpec, attentive: bool) -> float:
        """Advances the gaze; returns the size of a saccade started now (0 otherwise)."""
        started = 0.0
        if now >= self._next:
            rng = self._rng
            rx, ry = motion.gaze
            if rng.random() < 0.22:  # micro-saccade
                target = (self._to[0] + rng.gauss(0, 0.35), self._to[1] + rng.gauss(0, 0.25))
                hold = rng.uniform(0.25, 0.8)
            else:
                if rng.random() < (0.6 if attentive else 0.38):
                    target = (rng.gauss(0, 0.25), rng.gauss(0, 0.18))
                else:
                    a, r = rng.uniform(0, 2 * math.pi), math.sqrt(rng.random())
                    target = (rx * r * math.cos(a), ry * r * math.sin(a))
                hold = _clamp(math.exp(rng.gauss(math.log(1.3), 0.6)), 0.3, 5.0)
            target = (_clamp(target[0], -rx * 1.2, rx * 1.2), _clamp(target[1], -ry * 1.2, ry * 1.2))
            self._from = (self.x, self.y)
            self._to = target
            distance = math.hypot(target[0] - self.x, target[1] - self.y)
            self._dur = 0.035 + 0.02 * distance
            self._t0 = now
            self._next = now + self._dur + hold / max(0.05, motion.saccades)
            started = distance
        k = _ease_out((now - self._t0) / self._dur)
        self.x = self._from[0] + (self._to[0] - self._from[0]) * k
        self.y = self._from[1] + (self._to[1] - self._from[1]) * k
        return started

    def drift(self, now: float) -> tuple[float, float]:
        return self._drift[0](now) * 0.08, self._drift[1](now) * 0.06


@dataclass
class SpeechState:
    talk: float = 0.0  # 0..1, how much the face is in "talking" mode
    open: float = 0.0  # 0..1, mouth opening from loudness
    shape: float = 0.0  # -1 round (o, u) .. 1 wide (s, i)
    emphasis: float = 0.0  # 0..1 impulse on stressed syllables
    pause: bool = False  # a pause between phrases started now


class SpeechTracker:
    """Speaker loudness (dB) and zero-crossing rate from the ESP32, turned into mouth motion."""

    def __init__(self, rng: random.Random) -> None:
        self._rng = rng
        self._queue: deque[tuple[float, float, float]] = deque(maxlen=512)
        self._peak = 55.0
        self._target = 0.0
        self._zcr = 0.08
        self._last_level = -10.0
        self._last_voiced = -10.0
        self._last_onset = 0.0
        self._quiet_since: float | None = None
        self._paused = True
        self._t: float | None = None
        self.state = SpeechState()
        self._slow = 0.0

    def feed(self, db: float, zcr: float, now: float) -> None:
        """Thread-safe: called from the link's reader thread."""
        self._queue.append((now, db, zcr))

    def update(self, now: float) -> SpeechState:
        dt = 0.04 if self._t is None else _clamp(now - self._t, 0.001, 0.2)
        self._t = now
        s = self.state
        loudest = None
        while self._queue:
            t, db, zcr = self._queue.popleft()
            self._peak = max(db, self._peak - 5.0 * max(0.0, t - self._last_level), 45.0)
            self._last_level = t
            norm = 0.0 if db < 22.0 else _clamp((db - (self._peak - 26.0)) / 26.0)
            if norm > 0.05:
                self._last_voiced = t
            loudest = norm if loudest is None else max(loudest, norm)
            self._zcr += (zcr - self._zcr) * 0.3
        if loudest is not None:
            self._target = loudest ** 0.85
        elif now - self._last_level > 0.12:
            self._target = 0.0
        tau = 0.028 if self._target > s.open else 0.085
        s.open += (self._target - s.open) * (1.0 - math.exp(-dt / tau))
        self._slow += (s.open - self._slow) * (1.0 - math.exp(-dt / 0.32))
        if s.open - self._slow > 0.16 and s.open > 0.32 and now - self._last_onset > 0.15:
            s.emphasis = min(1.0, s.emphasis + (s.open - self._slow) * 2.0 * self._rng.uniform(0.55, 1.0))
            self._last_onset = now
        s.emphasis *= math.exp(-dt / 0.3)
        talking = now - self._last_voiced < 0.45
        s.talk += ((1.0 if talking else 0.0) - s.talk) * (1.0 - math.exp(-dt / (0.08 if talking else 0.3)))
        # At 24 kHz vowels cross zero ~2-6 % of samples, hissy consonants (s, x, f) far more.
        s.shape += (_clamp((self._zcr - 0.08) * 6.0, -1.0, 1.0) - s.shape) * (1.0 - math.exp(-dt / 0.06))
        s.pause = False
        if s.talk > 0.5 and s.open < 0.08:
            if self._quiet_since is None:
                self._quiet_since = now
            elif now - self._quiet_since > 0.28 and not self._paused:
                s.pause = self._paused = True
        elif s.open > 0.2:
            self._quiet_since = None
            self._paused = False
        return s


# ---------------------------------------------------------------- animator

TRANSITION_S = 0.4
TALK_MOUTH = MouthSpec(w=6.5, y=32.0, curve=1.5)


class FaceAnimator:
    """Turns the current expression name plus time (and speech) into LED frames."""

    def __init__(self, looks: Callable[[str], Look], name: str, now: float = 0.0, seed: int | None = None) -> None:
        self._looks = looks
        self.rng = random.Random(seed)
        self.name = name
        self.look = looks(name)
        self.since = now
        self._from: Pose | None = None
        self._t0 = now
        self._last: Pose | None = None
        self.blinker = Blinker(self.rng, now)
        self.gaze = Gaze(self.rng, now)
        self.speech = SpeechTracker(self.rng)
        self._breath_phase = self.rng.uniform(0, 2 * math.pi)
        self._breath_noise = _Noise(self.rng, 0.05)
        self._sway = (_Noise(self.rng, 0.11), _Noise(self.rng, 0.09))
        self._shake = (_Noise(self.rng, 7.0), _Noise(self.rng, 6.3))
        self._bounce_phase = 0.0
        self._t: float | None = None

    def set_expression(self, name: str, now: float) -> None:
        if name == self.name:
            return
        self.name = name
        self.look = self._looks(name)
        self.since = now
        if self._last is not None:
            self._from = self._last.copy()
            self._t0 = now
        self.gaze.kick(now, 0.05)

    def wake(self, now: float) -> None:
        self.blinker.wake(now)

    def feed_speech(self, db: float, zcr: float, now: float) -> None:
        self.speech.feed(db, zcr, now)

    def pose(self, now: float) -> Pose:
        dt = 0.04 if self._t is None else _clamp(now - self._t, 0.0, 0.25)
        self._t = now
        look, motion = self.look, self.look.motion
        pose = pose_of(look)
        age = now - self.since
        if look.anim is not None:
            look.anim(pose, now, age, self)
        # Breathing at a slowly wandering rate, sway, shake and bounce move the whole face.
        self._breath_phase += 2 * math.pi * 0.23 * (1.0 + 0.2 * self._breath_noise(now)) * dt
        self._bounce_phase += 2 * math.pi * motion.bounce_hz * dt
        pose.oy += motion.breath * math.sin(self._breath_phase) - motion.bounce * abs(math.sin(self._bounce_phase))
        pose.ox += motion.sway * self._sway[0](now) + motion.shake * self._shake[0](now)
        pose.oy += motion.sway * 0.4 * self._sway[1](now) + motion.shake * 0.6 * self._shake[1](now)
        if self._from is not None:
            p = (now - self._t0) / TRANSITION_S
            if p >= 1.0:
                self._from = None
            else:
                pose = blend(self._from, pose, smooth(p))
        speech = self.speech.update(now)
        saccade = self.gaze.update(now, motion, speech.talk > 0.3 or self.name == "listening")
        if saccade > 2.2 and self.rng.random() < 0.3:
            self.blinker.blink_now(now)
        if speech.pause:
            if self.rng.random() < 0.55:
                self.blinker.blink_now(now)
            if self.rng.random() < 0.35:
                self.gaze.kick(now, 0.1)
        drift_x, drift_y = self.gaze.drift(now)
        pose.gaze_x, pose.gaze_y = self.gaze.x + drift_x, self.gaze.y + drift_y
        if pose.gaze_override is not None:
            gx, gy, w = pose.gaze_override
            pose.gaze_x += (gx - pose.gaze_x) * w
            pose.gaze_y += (gy - pose.gaze_y) * w
        pose.blink = self.blinker.value(now, motion.blinks * (1.0 + 0.4 * speech.talk), motion.slow_blinks)
        if speech.talk > 0.01:
            self._apply_speech(pose, speech)
        self._last = pose
        return pose

    def _apply_speech(self, pose: Pose, s: SpeechState) -> None:
        base = pose.mouth[-1].spec if pose.mouth else TALK_MOUTH
        lips = base if base.kind == "lips" else TALK_MOUTH
        talking = replace(
            lips,
            open=lips.open + s.open * 6.5,
            w=_clamp(lips.w * (1.0 - 0.1 * s.open) * (1.0 + 0.16 * s.shape), 3.0, 13.0),
            curve=lips.curve * 0.6,
            y=lips.y + 0.4 * s.open,
            tongue=0.0,
            teeth=0.0,
        )
        if len(pose.mouth) == 1 and base.kind == "lips":
            pose.mouth = [Layer(lerp_spec(base, talking, s.talk))]
        else:
            pose.mouth = _fade(pose.mouth, 1.0 - s.talk) + [Layer(talking, s.talk)]
        pose.brow_raise += s.emphasis * 1.3 + s.open * 0.25
        pose.oy += -s.emphasis * 0.45 + s.open * 0.2
        for name in ("eye_l", "eye_r"):
            for layer in getattr(pose, name):
                if layer.spec.kind in ("stroke", "pill", "solid"):
                    layer.spec = replace(layer.spec, ry=layer.spec.ry * (1.0 + 0.07 * s.emphasis))

    def render(self, now: float) -> Canvas:
        return draw(self.pose(now), now)

    def frame(self, now: float) -> bytes:
        return self.render(now).to_bytes()
