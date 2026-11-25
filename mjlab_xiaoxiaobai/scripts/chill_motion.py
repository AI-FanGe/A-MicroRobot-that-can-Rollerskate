"""Idle loops for Xiaoxiaobai, upper and lower limbs and the head: sitting on the floor, lying on the
back and sitting on a chair, chilling. Procedural and seeded: smooth random signals instead of
fixed rhythms.

sit   Sitting on the floor leaning back on straight arms (the hip cannot flex enough to sit
      upright), legs out: the feet wag (ankles), wipe side to side (hip yaw) and now and then a
      knee comes up, the heel sliding back. The arms hold the body. The head looks around.
lie   Lying on the back: the legs and the arms each go from one lazy activity to another (ankles
      rocking, knees up swaying, a leg crossed up with the foot bobbing, lazy kicks, ankle
      circles; arms resting by the sides, a little apart, tapping, one hand up to look at, hands
      at the waist), bigger and smaller by turns. The head looks around, at a raised hand or
      foot, nods along.
chair Sitting on a stool (prop_boxes) with no backrest, leaning back on straight arms, the hands
      on the seat behind the hips, knees bent and the feet off the floor: the legs kick by turns,
      swing together, the toes tap or turn in and out, one leg stretches out; the weight shifts
      from hand to hand, a hand pats the seat.
wake  Played once, lying on the back (起床伸懒腰): still drowsy, a knee shuffles and the head rolls;
      then the arms go up over the head and the legs stretch long, toes pointed; held, the whole
      body shivers for WAKE_SHIVER_S; the arms flop back down by the sides, the legs let go, and it
      ends in the base pose. Not looped: export it with chill_export.py --once.

The floor bases are the prop-up reference's lying pose and its propped-up hold (propup_motion.py,
physically checked). Head pitch (motor 20, + down, -10..+30 deg) has no joint in the MJCF: it is
saved as head_pitch beside the joints.

usage: chill_motion.py {sit,lie,chair,wake} <out_raw.npz> [--seconds 40] [--seed 0]
(wake sets its own length.)
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

FPS = 50
BASE = Path(__file__).resolve().parents[1] / "data/motions/propup/propup_fast_raw.npz"
SIT_T, LIE_T = 4.4, 0.0  # base frames [s] in BASE
HEAD_YAW = (-38.0, 38.0)  # deg, + left (robot limits +-40)
HEAD_PITCH = (-9.0, 29.0)  # deg, + down (robot limits -10..+30)


# ------------------------------------------------------------------ signals

def noise(n, tau, rng):
    """Smooth unit-std noise, correlation time ~tau [s]."""
    x = gaussian_filter1d(rng.standard_normal(n + 400), tau * FPS, mode="wrap")[200:-200]
    return x / (x.std() + 1e-9)


def oscillator(n, f0, spread, tau, rng):
    """Phase of an oscillator around f0 [Hz] whose frequency wanders (x exp(spread*noise))."""
    f = f0 * np.exp(spread * noise(n, tau, rng))
    return np.cumsum(2 * np.pi * f / FPS) + rng.uniform(0, 2 * np.pi)


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * x * (10 - 15 * x + 6 * x * x)


def envelope(n, lo, hi, tau, rng):
    """Intensity wandering between lo and hi."""
    return lo + (hi - lo) * (0.5 + 0.5 * np.tanh(1.2 * noise(n, tau, rng)))


def gaze(n, rng, center, yaw_range, pitch_range, attention=None, nod=None):
    """Head yaw, pitch [deg]: fixations with quick minimum-jerk turns between them, sometimes a
    slow sweep, a little drift; attention (n, 3: weight, yaw, pitch) pulls the gaze to a target;
    nod (n,) adds bobbing."""
    yaw, pitch = np.zeros(n), np.zeros(n)
    cur = np.array(center, dtype=float)
    i = 0
    while i < n:
        if rng.random() < 0.65:  # glance somewhere nearby or come back toward the center
            tgt = np.array(center) + rng.normal(0, [14.0, 8.0])
        else:  # look well aside / up / down, or a combination
            tgt = np.array([rng.uniform(*yaw_range), rng.uniform(*pitch_range)])
        tgt = np.clip(tgt, [yaw_range[0], pitch_range[0]], [yaw_range[1], pitch_range[1]])
        amp = np.linalg.norm(tgt - cur)
        move = rng.uniform(1.8, 3.2) if rng.random() < 0.2 else 0.22 + 0.011 * amp + rng.uniform(0, 0.15)
        dwell = rng.uniform(0.7, 3.4)
        m, d = int(move * FPS), int(dwell * FPS)
        s = smoothstep(np.arange(1, m + 1) / m)
        seg = cur + s[:, None] * (tgt - cur)
        k = min(m, n - i)
        yaw[i:i + k], pitch[i:i + k] = seg[:k, 0], seg[:k, 1]
        i += k
        k = min(d, n - i)
        yaw[i:i + k], pitch[i:i + k] = tgt
        i += k
        cur = tgt
    yaw += 1.2 * noise(n, 0.8, rng)
    pitch += 0.9 * noise(n, 0.8, rng)
    if attention is not None:
        w = attention[:, 0]
        yaw = (1 - w) * yaw + w * attention[:, 1]
        pitch = (1 - w) * pitch + w * attention[:, 2]
    if nod is not None:
        pitch += nod
    return np.clip(yaw, *yaw_range), np.clip(pitch, *pitch_range)


# ------------------------------------------------------------------ joints

class Pose:
    """Per-frame joint targets [rad] in the raw motion's joint order, set in mirrored terms."""

    def __init__(self, names):
        self.names = names

    @staticmethod
    def leg(flex, knee, ankle, yaw=0.0, roll=0.0):
        """One leg [deg]: hip flex (thigh forward), knee bend, ankle (+ toes toward the shin),
        hip yaw and roll (+ outward). Returns the right side's joint angles; left = mirror."""
        return {"hip_pitch": -flex, "knee": knee, "ankle_pitch": -ankle, "hip_yaw": yaw, "hip_roll": roll}

    def set_leg(self, q, side, flex, knee, ankle, yaw=0.0, roll=0.0):
        r = self.leg(flex, knee, ankle, yaw, roll)
        s = 1.0 if side == "right" else -1.0
        for j, v in r.items():
            q[:, self.names.index(f"{side}_{j}")] = np.radians(s * v)

    def set_arm(self, q, side, p, r, e):
        """Gesture authoring terms [deg]: p lifts the arm forward, r out sideways, e bends the elbow."""
        s = 1.0 if side == "right" else -1.0
        q[:, self.names.index(f"{side}_shoulder_pitch")] = np.radians(s * p)
        q[:, self.names.index(f"{side}_shoulder_roll")] = np.radians(s * r)
        q[:, self.names.index(f"{side}_elbow")] = np.radians(-s * e)


def knees_up_ankle(flex, knee):
    """Ankle that keeps the sole flat on the floor, lying on the back."""
    return knee - flex - 90.0


# ------------------------------------------------------------------ sitting

def sit(n, rng, names, base):
    t = np.arange(n) / FPS
    pose = Pose(names)
    q = np.repeat(base[None], n, 0)
    sections = [(0.0, "坐着晃脚")]

    # Feet wag: one oscillator with a wandering tempo; the left foot runs at a phase lag that
    # drifts between together and alternating.
    ph = oscillator(n, 0.55, 0.35, 3.0, rng)
    lag = np.pi * (0.5 + 0.5 * np.tanh(1.3 * noise(n, 5.0, rng)))
    amp = envelope(n, 6.0, 26.0, 4.0, rng)
    ankle_r = -8.0 + amp * np.sin(ph)
    ankle_l = -8.0 + amp * np.sin(ph + lag) * envelope(n, 0.6, 1.0, 3.0, rng)
    # Windshield wipers: both feet tip the same way, slower, now and then.
    wph = oscillator(n, 0.28, 0.3, 4.0, rng)
    wipe = envelope(n, 0.0, 14.0, 5.0, rng) * np.sin(wph)
    # Knee comes up now and then (heel sliding back: the thigh lifts half the knee bend).
    knee = {"right": np.zeros(n), "left": np.zeros(n)}
    i = int(rng.uniform(2.0, 5.0) * FPS)
    while i < n:
        both = rng.random() < 0.15
        sides = ["right", "left"] if both else [rng.choice(["right", "left"])]
        # Both thighs up at once tips the body back onto the floor at KP 200.
        k = rng.uniform(14.0, 24.0) if both else rng.uniform(16.0, 38.0)
        up, hold, down = (int(rng.uniform(a, b) * FPS) for a, b in ((0.7, 1.3), (1.0, 4.0), (0.8, 1.5)))
        prof = np.concatenate([smoothstep(np.linspace(0, 1, up)), np.ones(hold), smoothstep(np.linspace(1, 0, down))])
        prof = prof[: n - i]
        for s in sides:
            knee[s][i:i + len(prof)] = np.maximum(knee[s][i:i + len(prof)], k * prof)
        sections.append((i / FPS, "抬膝" if len(sides) == 1 else "双膝抬起"))
        if i + len(prof) < n:
            sections.append(((i + len(prof)) / FPS, "坐着晃脚"))
        i += len(prof) + int(rng.uniform(2.5, 8.0) * FPS)
    flex0 = np.degrees(-base[names.index("right_hip_pitch")])
    for side, ank, yaw in (("right", ankle_r, wipe), ("left", ankle_l, -wipe)):
        kb = gaussian_filter1d(knee[side], 0.08 * FPS)
        pose.set_leg(q, side, flex0 + kb / 2, kb, ank - 0.3 * kb, yaw, 3.5)

    # Arms hold the body: straight, a little back; a slow breath in the shoulders.
    breath = 1.2 * np.sin(2 * np.pi * t / 4.6 + rng.uniform(0, 6))
    for side in ("right", "left"):
        pose.set_arm(q, side, -37.0 + breath, 4.0, 3.0)

    # Leaning back 24 deg: the horizon is ~18 deg of head pitch down.
    nod = np.where(envelope(n, 0, 1, 6.0, rng) > 0.75, 3.0 * np.sin(oscillator(n, 1.5, 0.1, 3.0, rng)), 0.0)
    nod = gaussian_filter1d(nod, 0.1 * FPS)
    yaw, pitch = gaze(n, rng, (0.0, 16.0), HEAD_YAW, HEAD_PITCH, nod=nod)
    q[:, names.index("head")] = np.radians(yaw)
    return q, np.radians(pitch), sections


# ------------------------------------------------------------------ lying

LEG_ACTS = ("脚踝摆动", "屈膝摇摆", "翘脚晃", "懒踢腿", "单膝画圈")
ARM_ACTS = ("手放两侧", "双手微张", "手指轻敲", "抬手看看", "手搭腰间")
REST_ARM = (-17.0, 8.0, 10.0)  # arms by the sides, the forearms just off the floor


def schedule(n, acts, rng, lo, hi, first=None):
    """Random activities back to back: list of (start, end, name), never the same twice in a row."""
    out, i, prev = [], 0, None
    while i < n:
        name = first if first and not out else rng.choice([a for a in acts if a != prev])
        d = int(rng.uniform(lo, hi) * FPS)
        out.append((i, min(n, i + d), name))
        i += d
        prev = name
    return out


def blend_schedule(n, sched, render, fade_s=1.3):
    """Each activity rendered over the whole time, crossfaded at the switches. render(name, i0)
    returns (n, k) arrays; i0 lets an activity start its own clock."""
    out = None
    for k, (i0, i1, name) in enumerate(sched):
        x = render(name, i0)
        w = np.zeros(n)
        f = int(fade_s * FPS)
        w[i0:] = 1.0  # until the next activity fades in over it
        if k > 0:
            w[i0:min(n, i0 + f)] = smoothstep(np.linspace(0, 1, min(n, i0 + f) - i0))
        if out is None:
            out = x.copy()
        else:
            out = out * (1 - w[:, None]) + x * w[:, None]
    return out


def lie(n, rng, names, base):
    t = np.arange(n) / FPS
    q = np.repeat(base[None], n, 0)
    pose = Pose(names)
    big = envelope(n, 0.35, 1.0, 5.0, rng)  # bigger / smaller by turns

    lift_side = {}

    def legs(name, i0):
        """(n, 10): right then left (flex, knee, ankle, yaw, roll)."""
        o = np.zeros((n, 10))
        R, L = o[:, :5], o[:, 5:]
        if name == "脚踝摆动":
            ph = oscillator(n, 0.7, 0.35, 2.5, rng)
            a = 8.0 + 22.0 * big
            R[:, 2] = -10 + a * np.sin(ph)
            L[:, 2] = -10 + a * np.sin(ph + np.pi)
            R[:, 3] = L[:, 3] = 4.0 * np.sin(0.5 * ph)
        elif name == "屈膝摇摆":
            sway = (6.0 + 8.0 * big) * np.sin(oscillator(n, 0.18, 0.3, 4.0, rng))
            tap = 7.0 * np.maximum(0, np.sin(oscillator(n, 1.1, 0.25, 3.0, rng))) * (big > 0.6)
            for X, s in ((R, 1), (L, -1)):
                X[:, 0], X[:, 1] = 45.0, 88.0
                X[:, 2] = knees_up_ankle(45.0, 88.0) + tap
                X[:, 4] = -s * sway  # both knees fall the same way
                X[:, 3] = 0.5 * sway * s
        elif name == "翘脚晃":
            side = lift_side.setdefault(i0, rng.choice(["right", "left"]))
            bob = np.sin(oscillator(n, 1.2, 0.3, 2.0, rng))
            for X, nm in ((R, "right"), (L, "left")):
                if nm == side:
                    X[:, 0] = 78.0 + 5.0 * np.sin(oscillator(n, 0.25, 0.2, 3.0, rng))
                    X[:, 1] = 45.0 + 8.0 * big * np.sin(oscillator(n, 0.4, 0.3, 3.0, rng))
                    X[:, 2] = -12.0 + (8.0 + 18.0 * big) * bob
                    X[:, 3] = 8.0
                else:
                    X[:, 0], X[:, 1] = 45.0, 88.0
                    X[:, 2] = knees_up_ankle(45.0, 88.0)
        elif name == "懒踢腿":
            for X, off in ((R, 0.0), (L, np.pi)):
                ph = oscillator(n, 0.22, 0.4, 3.0, rng) + off
                up = np.maximum(0, np.sin(ph)) ** 1.5
                X[:, 0] = (12.0 + 26.0 * big) * up
                X[:, 1] = 12.0 * up
                X[:, 2] = -25.0 * up + 6.0 * np.sin(2.3 * ph)
        elif name == "单膝画圈":
            side = lift_side.setdefault(i0, rng.choice(["right", "left"]))
            ph = oscillator(n, 0.45, 0.3, 3.0, rng)
            for X, nm in ((R, "right"), (L, "left")):
                if nm == side:
                    X[:, 0], X[:, 1] = 45.0, 88.0
                    X[:, 2] = knees_up_ankle(45.0, 88.0)
                else:
                    X[:, 2] = -8.0 + (10.0 + 14.0 * big) * np.sin(ph)
                    X[:, 3] = (6.0 + 8.0 * big) * np.cos(ph)
        R[:, 4] += 3.5
        L[:, 4] += 3.5
        return o

    hand_side = {}
    arm_big = envelope(n, 0.3, 1.0, 6.0, rng)

    def arms(name, i0):
        """(n, 6): right then left (p, r, e). Small and low: the shoulders sit outside the trunk and
        cannot bring the hands in over the belly, and above ~140 deg of pitch the hands go down
        behind the head into the floor and push the back up."""
        o = np.zeros((n, 6))
        drift = lambda s: s * noise(n, 1.5, rng)  # noqa: E731
        for k in (0, 3):
            o[:, k:k + 3] = REST_ARM
            o[:, k] += drift(1.5)
            o[:, k + 2] += drift(2.0)
        if name == "双手微张":
            for k in (0, 3):
                o[:, k + 1] = 20.0 + 12.0 * arm_big + drift(3.0)
                o[:, k + 2] = 18.0 + drift(4.0)
        elif name == "手指轻敲":
            tap = np.maximum(0, np.sin(oscillator(n, 1.5, 0.25, 3.0, rng))) ** 2
            both = rng.random() < 0.5
            side = hand_side.setdefault(i0, rng.choice(["right", "left"]))
            for k, nm in ((0, "right"), (3, "left")):
                if both or nm == side:
                    o[:, k + 2] = REST_ARM[2] + 4.0 + (6.0 + 8.0 * arm_big) * tap
        elif name == "抬手看看":
            side = hand_side.setdefault(i0, rng.choice(["right", "left"]))
            k = 0 if side == "right" else 3
            o[:, k] = 58.0 + 8.0 * np.sin(oscillator(n, 0.1, 0.3, 4.0, rng))
            o[:, k + 1] = 14.0 + 5.0 * arm_big * np.sin(oscillator(n, 0.25, 0.3, 3.0, rng))
            o[:, k + 2] = 35.0 + 10.0 * arm_big * np.sin(oscillator(n, 0.3, 0.3, 3.0, rng))
        elif name == "手搭腰间":
            for k in (0, 3):
                o[:, k] = -10.0 + drift(2.0)
                o[:, k + 1] = 6.0
                o[:, k + 2] = 40.0 + drift(4.0)
        return o

    leg_s = schedule(n, LEG_ACTS, rng, 5.0, 9.0, first="脚踝摆动")
    arm_s = schedule(n, ARM_ACTS, rng, 5.0, 10.0, first="手放两侧")
    lg = blend_schedule(n, leg_s, legs, 1.4)
    am = blend_schedule(n, arm_s, arms, 1.3)
    for X, side in ((lg[:, :5], "right"), (lg[:, 5:], "left")):
        pose.set_leg(q, side, *X.T)
    for X, side in ((am[:, :3], "right"), (am[:, 3:], "left")):
        pose.set_arm(q, side, *X.T)

    # Look at a raised hand, at a foot bobbing in the air.
    att = np.zeros((n, 3))
    for i0, i1, name in arm_s:
        if name == "抬手看看":
            side = hand_side[i0]
            w = np.zeros(n)
            a, b = i0 + int(0.6 * FPS), i1
            w[a:b] = 0.85 * (rng.random() < 0.8)
            att[:, 0] = np.maximum(att[:, 0], gaussian_filter1d(w, 0.4 * FPS))
            att[a:b, 1] = 30.0 if side == "left" else -30.0
            att[a:b, 2] = 4.0
    for i0, i1, name in leg_s:
        if name == "翘脚晃":
            side = lift_side[i0]
            w = np.zeros(n)
            a = i0 + int(1.0 * FPS)
            b = min(i1, a + int(rng.uniform(1.5, 3.0) * FPS))
            w[a:b] = 0.9
            att[:, 0] = np.maximum(att[:, 0], gaussian_filter1d(w, 0.3 * FPS))
            att[a:b, 1] = 10.0 if side == "left" else -10.0
            att[a:b, 2] = 28.0
    att[:, 1] = gaussian_filter1d(att[:, 1], 0.2 * FPS)
    att[:, 2] = gaussian_filter1d(att[:, 2], 0.2 * FPS)
    nod = np.where(envelope(n, 0, 1, 6.0, rng) > 0.7, 4.0 * np.sin(oscillator(n, 1.4, 0.1, 3.0, rng)), 0.0)
    nod = gaussian_filter1d(nod, 0.12 * FPS)
    yaw, pitch = gaze(n, rng, (0.0, 2.0), HEAD_YAW, HEAD_PITCH, attention=att, nod=nod)
    q[:, names.index("head")] = np.radians(yaw)

    sections = []
    for s in sorted({i0 for i0, _, _ in leg_s + arm_s}):
        leg = next(nm for i0, i1, nm in leg_s if i0 <= s < i1)
        arm = next(nm for i0, i1, nm in arm_s if i0 <= s < i1)
        sections.append((s / FPS, f"腿·{leg}  手·{arm}"))
    return q, np.radians(pitch), sections


# ------------------------------------------------------------------ waking up, stretching

WAKE_S = 9.0
WAKE_SHIVER = (3.1, 5.1)  # [s] arms over the head, legs long: the shiver
WAKE_SHIVER_S = WAKE_SHIVER[1] - WAKE_SHIVER[0]
WAKE_SHIVER_RAMP_S = 0.25
# Played at KP 400 (xiaoxiaobai CHILL_KP), the motors follow 3.5-5 Hz at ~40 % of the command in
# chill_sim.py (KP 200: ~10 %, 6.5-10 Hz: under 20 % even at KP 400): the amplitudes [deg]
# per channel at the shiver's peak (it swells from 0.8 to 1.2 of them) are sized for that.
WAKE_SHIVER_HZ = (3.5, 5.0)
WAKE_SHIVER_DEG = {"p": 5.5, "r": 6.0, "e": 6.0, "flex": 4.0, "knee": 6.0, "ankle": 7.0, "roll": 3.0, "yaw": 3.0,
                   "head": 2.5, "head_pitch": 2.5}
# The robot's face follows the sections (xiaoxiaobai api/motion.py CHILL_FACES); they start this
# early so that the face, a status period and a blend behind, lands with the move.
WAKE_FACE_LEAD_S = 0.2


def track(t, points):
    """A channel through (time [s], value) points, minimum-jerk from each to the next, held
    before the first and after the last."""
    ts = np.array([p[0] for p in points], dtype=float)
    vs = np.array([p[1] for p in points], dtype=float)
    k = np.clip(np.searchsorted(ts, t, side="right") - 1, 0, len(ts) - 2)
    return vs[k] + (vs[k + 1] - vs[k]) * smoothstep((t - ts[k]) / (ts[k + 1] - ts[k]))


def shiver(t, rng, amp):
    """Every channel's own tremble [deg]: two sines in WAKE_SHIVER_HZ, faded in and out over the hold."""
    a, b = WAKE_SHIVER
    into = np.clip(t - a, 0.0, b - a)
    env = smoothstep(np.minimum(into, b - a - into) / WAKE_SHIVER_RAMP_S) * (0.8 + 0.4 * into / (b - a))
    env[(t < a) | (t > b)] = 0.0
    wave = sum(np.sin(2 * np.pi * rng.uniform(*WAKE_SHIVER_HZ) * t + rng.uniform(0, 2 * np.pi)) for _ in range(2)) / 1.4
    return amp * env * wave


def wake(n, rng, names, base):
    t = np.arange(n) / FPS
    q = np.repeat(base[None], n, 0)
    pose = Pose(names)
    a, b = WAKE_SHIVER
    flex0 = np.degrees(-base[names.index("right_hip_pitch")])

    for side, lag in (("right", 0.0), ("left", rng.uniform(0.06, 0.14))):
        ts = t - lag  # one side a little behind the other
        sh = {k: shiver(t, rng, v) for k, v in WAKE_SHIVER_DEG.items()}
        # Up over the head (above ~140 deg the hands reach the floor behind it), the fists
        # passing the face on the way; down by the sides, the elbows giving a little.
        p = track(ts, [(1.6, 0), (3.0, 128), (b, 128), (6.3, 0)])
        r = track(ts, [(1.6, 5), (2.4, 14), (3.0, 22), (b, 24), (6.3, 12), (7.6, 5)])
        e = track(ts, [(1.6, 0), (2.3, 30), (3.0, 10), (b, 10), (5.7, 20), (6.4, 10), (7.6, 0)])
        pose.set_arm(q, side, p + sh["p"], r + sh["r"], e + sh["e"])
        # Still drowsy, the right knee shuffles up (the heel sliding back); then the legs
        # stretch long, toes pointed, a touch off the floor; let go, the toes flick back.
        shuffle = track(ts, [(0.4, 0), (1.0, 22), (1.6, 0)]) if side == "right" else 0.0
        flex = track(ts, [(2.0, flex0), (3.0, flex0 + 4), (b, flex0 + 4), (6.2, flex0)]) + shuffle / 2
        knee = track(ts, [(2.0, 0), (3.0, 8), (b, 8), (6.0, 0)]) + shuffle
        ankle = track(ts, [(2.0, 0), (3.0, -32), (b, -34), (5.7, 6), (6.4, 0)]) - 0.3 * shuffle
        roll = track(ts, [(2.0, 3.5), (3.0, 6.0), (b, 6.0), (6.2, 3.5)])
        pose.set_leg(q, side, flex + sh["flex"], knee + sh["knee"], ankle + sh["ankle"], sh["yaw"], roll + sh["roll"])

    # The head rolls side to side half asleep, tips back with the stretch, and looks down
    # along the body, content, once it let go.
    yaw = track(t, [(0.5, 0), (1.1, 14), (1.7, -6), (2.3, 0), (6.0, 0), (6.8, 10), (8.0, 0)])
    pitch = track(t, [(0.0, 0), (0.6, 5), (1.2, 1), (1.8, 0), (2.6, -8), (b, -8), (6.0, 2), (7.0, 4), (8.4, 0)])
    q[:, names.index("head")] = np.radians(yaw + shiver(t, rng, WAKE_SHIVER_DEG["head"]))
    pitch = np.clip(pitch + shiver(t, rng, WAKE_SHIVER_DEG["head_pitch"]), *HEAD_PITCH)

    lead = WAKE_FACE_LEAD_S
    sections = [(0.0, "迷糊醒来"), (1.6 - lead, "伸懒腰"), (a - lead, "全身发抖"), (b - lead, "放松")]
    return q, np.radians(pitch), sections


# ------------------------------------------------------------------ sitting on a chair

# Trunk leaning back and hip flex [deg], the hands propped on the seat behind the hips: no
# backrest, the feet off the floor. The hips alone cannot hold the leaning trunk and the thighs
# are too short to sit upright on (~1 cm of CoM error tips the robot off the seat edge), so the
# arms carry it as when sitting on the floor. The thighs slope 22 deg down so that the hip blocks
# and the thighs both lie on the seat.
CHAIR_LEAN = 28.0
CHAIR_FLEX = 40.0
CHAIR_SHIN = CHAIR_FLEX + CHAIR_LEAN  # knee bend with the shins hanging plumb
# The shins hang a little forward and never swing back past 10 deg short of plumb: behind that
# they reach the seat board (the hip blocks it carries end only ~45 mm behind the knees).
KNEE0, KNEE_MAX = CHAIR_SHIN - 16.0, CHAIR_SHIN - 10.0
PROP_ARM = (-35.0, 6.0, 0.0)  # straight arms back, the hands pressing ~2 mm into the seat
FEET_CLEAR = 0.02  # lowest the swinging feet get to the floor [m]
SEAT_BOARD = 0.02  # seat board thickness [m]; under it the heels swing back freely
CHAIR_ACTS = ("晃腿", "脚尖点点", "伸一条腿", "双腿一起荡", "脚尖内外摆")
CHAIR_ARMS = ("双手撑着", "重心换边", "轻拍椅面")


def chair_base(names):
    q = np.zeros(len(names))[None]
    pose = Pose(names)
    for side in ("right", "left"):
        pose.set_leg(q, side, CHAIR_FLEX, KNEE0, -12.0, 0.0, 3.5)
        pose.set_arm(q, side, *PROP_ARM)
    return q[0]


def chair(n, rng, names, base):
    t = np.arange(n) / FPS
    q = np.repeat(base[None], n, 0)
    pose = Pose(names)
    big = envelope(n, 0.35, 1.0, 5.0, rng)
    leg_side = {}
    swing_ph = oscillator(n, 0.7, 0.3, 3.0, rng)  # the legs' tempo, shared with the head bob

    def legs(name, i0):
        """(n, 6): right then left (knee bend, ankle, hip yaw); the hips stay at CHAIR_FLEX."""
        o = np.zeros((n, 6))
        o[:, [0, 3]] = KNEE0
        o[:, [1, 4]] = -12.0
        if name == "晃腿":  # kicking the legs by turns, like a kid on a high chair
            lag = np.pi * (0.75 + 0.25 * np.tanh(noise(n, 4.0, rng)))
            a = 6.0 + 10.0 * big
            for k, off in ((0, 0.0), (3, lag)):
                s = np.sin(swing_ph + off)
                o[:, k] = KNEE_MAX - a - a * s
                o[:, k + 1] = -14.0 + 8.0 * np.sin(swing_ph + off - 0.9)  # the foot trails the shin
        elif name == "脚尖点点":
            ph = oscillator(n, 1.1, 0.25, 2.5, rng)
            lag = np.pi * (0.5 + 0.5 * np.tanh(noise(n, 3.0, rng)))
            for k, off in ((0, 0.0), (3, lag)):
                o[:, k + 1] = -6.0 + (8.0 + 10.0 * big) * np.sin(ph + off)
        elif name == "伸一条腿":
            side = leg_side.setdefault(i0, rng.choice(["right", "left"]))
            k = 0 if side == "right" else 3
            m = int(rng.uniform(1.2, 2.0) * FPS)
            up = np.zeros(n)
            up[i0:] = 1.0
            up[i0:i0 + m] = smoothstep(np.linspace(0, 1, min(m, n - i0)))[: len(up[i0:i0 + m])]
            o[:, k] = KNEE0 - (30.0 + 12.0 * big) * up
            o[:, k + 1] = -12.0 + (14.0 * np.sin(oscillator(n, 0.8, 0.3, 2.0, rng)) + 6.0) * up  # wiggling toes
            o[:, 3 - k] = KNEE0 + 3.0 * np.sin(oscillator(n, 0.5, 0.3, 3.0, rng))
        elif name == "双腿一起荡":
            ph = oscillator(n, 0.45, 0.3, 3.0, rng)
            for k in (0, 3):
                a = 8.0 + 8.0 * big
                o[:, k] = KNEE_MAX - a - a * np.sin(ph)
                o[:, k + 1] = -14.0 + 6.0 * np.sin(ph - 0.9)
        elif name == "脚尖内外摆":
            ph = oscillator(n, 0.5, 0.3, 3.0, rng)
            for k in (0, 3):
                o[:, k + 2] = (6.0 + 8.0 * big) * np.sin(ph)  # + toes out on both sides
                o[:, k + 1] = -10.0 + 5.0 * np.sin(2 * ph)
        return o

    arm_side = {}

    def arms(name, i0):
        """(n, 6): right then left (p, r, e). The hands stay on the seat: one arm alone cannot hold
        the leaning trunk, so a hand only leaves it for a pat."""
        o = np.zeros((n, 6))
        for k in (0, 3):
            o[:, k:k + 3] = PROP_ARM
            o[:, k] += 1.0 * noise(n, 2.0, rng)
        if name == "重心换边":  # one elbow gives a little, the trunk sinks to that side, and back
            s = np.sin(oscillator(n, 0.15, 0.3, 4.0, rng))
            o[:, 2] += (4.0 + 6.0 * big) * np.maximum(0, s)
            o[:, 5] += (4.0 + 6.0 * big) * np.maximum(0, -s)
        elif name == "轻拍椅面":  # a hand pats the seat to the beat: up a little and straight back down
            side = arm_side.setdefault(i0, rng.choice(["right", "left"]))
            k = 0 if side == "right" else 3
            pat = np.maximum(0, np.sin(oscillator(n, 0.9, 0.2, 3.0, rng))) ** 4 * envelope(n, 0.4, 1.0, 3.0, rng)
            o[:, k] += 6.0 * pat
            o[:, k + 2] += 14.0 * pat
        return o

    leg_s = schedule(n, CHAIR_ACTS, rng, 5.0, 9.0, first="晃腿")
    arm_s = schedule(n, CHAIR_ARMS, rng, 6.0, 10.0, first="双手撑着")
    lg = np.clip(blend_schedule(n, leg_s, legs, 1.2), [-90.0, -40.0, -20.0] * 2, [KNEE_MAX, 30.0, 20.0] * 2)
    am = blend_schedule(n, arm_s, arms, 1.2)
    for X, side in ((lg[:, :3], "right"), (lg[:, 3:], "left")):
        pose.set_leg(q, side, np.full(n, CHAIR_FLEX), X[:, 0], X[:, 1], X[:, 2], np.full(n, 3.5))
    breath = 1.0 * np.sin(2 * np.pi * t / 4.4 + rng.uniform(0, 6))
    for X, side in ((am[:, :3], "right"), (am[:, 3:], "left")):
        pose.set_arm(q, side, X[:, 0] + breath, X[:, 1], X[:, 2])

    # Look down at a leg held out; bob the head to the legs' tempo now and then.
    att = np.zeros((n, 3))
    for i0, i1, name in leg_s:
        if name == "伸一条腿" and rng.random() < 0.8:
            w = np.zeros(n)
            a = i0 + int(1.5 * FPS)
            b = min(i1, a + int(rng.uniform(1.5, 3.0) * FPS))
            w[a:b] = 0.85
            att[:, 0] = np.maximum(att[:, 0], gaussian_filter1d(w, 0.3 * FPS))
            att[a:b, 1] = 10.0 if leg_side[i0] == "left" else -10.0
            att[a:b, 2] = 29.0
    att[:, 1] = gaussian_filter1d(att[:, 1], 0.2 * FPS)
    att[:, 2] = gaussian_filter1d(att[:, 2], 0.3 * FPS)
    nod = np.where(envelope(n, 0, 1, 6.0, rng) > 0.7, 3.0 * np.sin(swing_ph), 0.0)
    nod = gaussian_filter1d(nod, 0.12 * FPS)
    # Leaning back 28 deg: the horizon is ~22 deg of head pitch down.
    yaw, pitch = gaze(n, rng, (0.0, 18.0), HEAD_YAW, HEAD_PITCH, attention=att, nod=nod)
    q[:, names.index("head")] = np.radians(yaw)

    sections = []
    for s in sorted({i0 for i0, _, _ in leg_s + arm_s}):
        leg = next(nm for i0, i1, nm in leg_s if i0 <= s < i1)
        arm = next(nm for i0, i1, nm in arm_s if i0 <= s < i1)
        sections.append((s / FPS, f"腿·{leg}  手·{arm}"))
    return q, np.radians(pitch), sections


def chair_scene(names, q):
    """Root pose and the chair (a seat board, its base): the hip blocks on the seat, the feet
    FEET_CLEAR over the floor at their lowest, the board's front edge just behind the hanging
    shins over the whole loop. Measured on the shapes that touch the floor in the prop-up task
    (convex hulls of the shells, the foot soles). Prints the clearances."""
    import mujoco
    from scipy.spatial.transform import Rotation

    from mjlab_xiaoxiaobai.robot.xiaoxiaobai_constants import get_getup_spec

    model = get_getup_spec().compile()
    data = mujoco.MjData(model)
    qadr = [model.jnt_qposadr[model.joint(nm).id] for nm in names]
    quat = np.roll(Rotation.from_euler("y", -CHAIR_LEAN, degrees=True).as_quat(), 1)
    geoms = {model.geom(g).name: g for g in range(model.ngeom)
             if (model.geom(g).name or "").endswith("_ground_collision") or "foot_collision" in (model.geom(g).name or "")}
    grid = np.stack(np.meshgrid(*[np.linspace(-1, 1, 7)] * 3), -1).reshape(-1, 3)

    def points(frame):
        data.qpos[:] = 0
        data.qpos[3:7] = quat
        data.qpos[qadr] = frame
        mujoco.mj_forward(model, data)
        out = {}
        for nm, g in geoms.items():
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = model.geom_dataid[g]
                local = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
            else:
                local = grid * model.geom_size[g]
            out[nm] = data.geom_xpos[g] + local @ data.geom_xmat[g].reshape(3, 3).T
        return out

    frames = [points(frame) for frame in q[:: 5]]
    feet = [nm for nm in geoms if nm.startswith(("ankle_block", "foot"))]
    seat = [nm for nm in geoms if nm.startswith(("hip_block", "femur"))]
    z = FEET_CLEAR - min(p[nm][:, 2].min() for p in frames for nm in feet)
    p0 = points(q[0])
    seat_low = min(p0[nm][:, 2].min() for nm in seat)
    seat_h = z + seat_low - 0.0005
    hip_front = max(p0[nm][:, 0].max() for nm in seat if nm.startswith("hip_block"))
    hands = [nm for nm in geoms if nm.startswith("radius")]
    hand0 = np.concatenate([p0[nm] for nm in hands])
    hand_x = hand0[hand0[:, 2] < hand0[:, 2].min() + 0.003, 0].mean()
    shin = [nm for nm in geoms if nm.startswith(("tibia", "ankle_block", "foot"))]
    edge = 1.0
    com = []
    for k, frame in enumerate(q[:: 5]):
        p = points(frame)
        com.append(data.subtree_com[1][0])
        for nm in shin:  # the points of the shins and feet level with the seat board
            pz = p[nm][:, 2] + z
            # Vertices only: a band wider than the board catches the hull faces across it.
            inside = (pz > seat_h - SEAT_BOARD - 0.02) & (pz < seat_h + 0.02)
            if inside.any():
                edge = min(edge, p[nm][inside, 0].min())
    edge -= 0.004
    print(f"chair: seat {seat_h * 1000:.0f} mm, front edge {edge * 1000:.0f} mm (hip blocks reach "
          f"{hip_front * 1000:.0f}), hands {(hand0[:, 2].min() - seat_low) * 1000:+.1f} mm on the seat at "
          f"x {hand_x * 1000:.0f}, CoM x {min(com) * 1000:.0f}..{max(com) * 1000:.0f} mm")
    back, post = min(-0.12, hand_x - 0.04), -0.07
    wood = [0.55, 0.42, 0.3, 1.0]
    board = [(edge + back) / 2, 0.0, seat_h - SEAT_BOARD / 2, (edge - back) / 2, 0.14, SEAT_BOARD / 2, *wood, 1, 0, 0, 0]
    base = [(post + back) / 2, 0.0, (seat_h - SEAT_BOARD) / 2, (post - back) / 2, 0.12, (seat_h - SEAT_BOARD) / 2,
            *[c * 0.8 for c in wood[:3]], 1.0, 1, 0, 0, 0]
    return np.array([0.0, 0.0, z]), quat, np.array([board, base], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=("sit", "lie", "chair", "wake"))
    ap.add_argument("out", type=Path)
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src = np.load(BASE)
    names = [str(x) for x in src["joint_names"]]
    i = int(round((SIT_T if args.kind == "sit" else LIE_T) * float(src["fps"])))
    base = chair_base(names) if args.kind == "chair" else src["joint_pos"][i].copy()
    rng = np.random.default_rng(args.seed)
    n = int((WAKE_S if args.kind == "wake" else args.seconds) * FPS)
    q, head_pitch, sections = {"sit": sit, "lie": lie, "chair": chair, "wake": wake}[args.kind](n, rng, names, base)

    # Ease in from the base pose over the first second (the robot starts there).
    w = smoothstep(np.arange(n) / FPS / 1.0)[:, None]
    q = base + w * (q - base)
    pitch0 = {"sit": 16.0, "lie": 0.0, "chair": 18.0, "wake": 0.0}[args.kind]
    head_pitch = w[:, 0] * head_pitch + (1 - w[:, 0]) * np.radians(pitch0)
    extra = {}
    if args.kind == "chair":
        root_pos, root_quat, extra["prop_boxes"] = chair_scene(names, q)
    else:
        root_pos, root_quat = src["root_pos"][i], src["root_quat"][i]

    vel = np.abs(np.diff(q, axis=0)) * FPS
    j = int(vel.max(0).argmax())
    print(f"{args.kind}: {n / FPS:.0f} s, fastest joint {names[j]} {vel.max():.2f} rad/s, "
          f"head pitch {np.degrees(head_pitch.min()):.0f}..{np.degrees(head_pitch.max()):.0f} deg")
    for s, label in sections:
        print(f"  {s:5.1f}  {label}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, fps=np.float32(FPS), joint_names=np.array(names), joint_pos=q.astype(np.float32),
             root_pos=np.repeat(root_pos[None], n, 0), root_quat=np.repeat(root_quat[None], n, 0),
             head_pitch=head_pitch.astype(np.float32),
             section_names=np.array([s[1] for s in sections]), section_starts=np.array([s[0] for s in sections]),
             **extra)
    print("saved", args.out)


if __name__ == "__main__":
    main()
