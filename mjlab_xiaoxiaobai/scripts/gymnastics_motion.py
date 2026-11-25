"""Choreograph a radio-calisthenics routine (广播体操) for Xiaoxiaobai and write it in the retarget
layout of retarget_g1_to_xiaoxiaobai.py, for dance_motion_npz.py to turn into the motion file of
Mjlab-Gymnastics-Xiaoxiaobai.

The routine is keyed on the beat: arms and head in joint space, legs as a trunk pose plus foot
placements solved by inverse kinematics on Xiaoxiaobai's model. Between keys every channel follows
a monotone cubic (PCHIP): smooth, through every key, never past it, so planted feet stay on the
floor. The trunk slides horizontally until the centre of mass sits over the authored weight
shift (both soles, or the stance sole), so every frame balances quasi-statically.

Sections (8 counts each): 预备, 原地踏步, 伸展, 扩胸, 踢腿, 体侧, 体转, 全身, 跳跃 (knee bounces
with arm jacks: Xiaoxiaobai does not hop), 整理.

usage: gymnastics_motion.py <out.npz> [--beat 0.55]
then:  dance_motion_npz.py <out.npz> data/motions/xiaoxiaobai/gymnastics_50hz.npz --blend 0.3
"""
import argparse
from collections import defaultdict

import mujoco
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from mjlab_xiaoxiaobai.robot.xiaoxiaobai_constants import DANCE_ARM_RANGES_DEG, HOME_FRAME, get_dance_spec

FPS = 50
SIDES = ("left", "right")
LEG = ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch", "ankle_roll")
LEGS = [f"{s}_{j}" for s in SIDES for j in LEG]
ARMS = [f"{s}_{j}" for s in SIDES for j in ("shoulder_pitch", "shoulder_roll", "elbow")]
JOINTS = LEGS + ARMS + ["head"]
FOOT = {"left": "foot_2", "right": "foot"}
# Arms in mirrored anatomical terms [deg]: pitch forward/up, roll outward, elbow bend.
# Right = (+pitch, +roll, -elbow), left = (-pitch, -roll, +elbow).
ARM_SIGN = {"right": (1, 1, -1), "left": (-1, -1, 1)}
DOWN = (0.0, 6.0, 4.0)
# Stops of the reference inside the working ranges [deg]: the robot must reach the poses
# without riding the joint-limit penalty.
ELBOW_MAX, ROLL_MAX, PITCH_MAX = 75.0, 115.0, 140.0
# Legs solved inside this fraction of robot.xml's ranges (the soft limits are 0.9).
LEG_RANGE = 0.85
# IK residual weights: foot position [m], foot orientation [rad] down / lifted.
W_ROT_DOWN, W_ROT_AIR, DAMP = 0.05, 0.01, 1e-6
LIFTED = 0.001  # a foot this high is in the air [m]
# Soft knees through the routine (trunk lowered [m] on top of the keyed dz), straight at the
# ready pose it starts and ends in.
SOFT = -0.005


class Choreo:
    """Keys per channel, in beats. A later key at the same beat replaces the earlier one."""

    def __init__(self):
        self.keys: dict[str, dict[float, np.ndarray]] = defaultdict(dict)

    def at(self, t: float, **ch):
        for name, value in ch.items():
            self.keys[name][round(t, 4)] = np.atleast_1d(np.asarray(value, float))

    def sample(self, name: str, t: np.ndarray, default) -> np.ndarray:
        keys = dict(self.keys.get(name, {}))
        if not keys:
            return np.tile(np.atleast_1d(np.asarray(default, float)), (len(t), 1))
        ts = np.array(sorted(keys))
        vs = np.stack([keys[k] for k in ts])
        if len(ts) == 1:
            return np.tile(vs[0], (len(t), 1))
        return PchipInterpolator(ts, vs, axis=0, extrapolate=False)(np.clip(t, ts[0], ts[-1]))


NEUTRAL = dict(ra=DOWN, la=DOWN, head=0.0, dz=0.0, pitch=0.0, roll=0.0, yaw=0.0, shift=0.0,
               lf=(0.0, 0.0, 0.0, 0.0), rf=(0.0, 0.0, 0.0, 0.0))


def neutral(c: Choreo, t: float, **override):
    c.at(t, **{**NEUTRAL, **override})


def arms(c: Choreo, t: float, both=None, **ch):
    """Both arms in the same (mirrored) pose, or ra / la separately."""
    if both is not None:
        ch.setdefault("ra", both)
        ch.setdefault("la", both)
    c.at(t, **ch)


def step(c: Choreo, t0: float, foot: str, dx=0.0, dy=0.0, lift=0.012, beats=1.0, base=(0.0, 0.0)):
    """A foot from base to (dx, dy) over `beats`: lift-off, move in the air, set down."""
    key = "lf" if foot == "left" else "rf"
    bx, by = base
    c.at(t0, **{key: (bx, by, 0.0, 0.0)})
    c.at(t0 + 0.15 * beats, **{key: (bx, by, 0.0, 0.0)})
    c.at(t0 + 0.5 * beats, **{key: ((bx + dx) / 2, (by + dy) / 2, lift, 0.0)})
    c.at(t0 + 0.85 * beats, **{key: (dx, dy, 0.0, 0.0)})
    c.at(t0 + beats, **{key: (dx, dy, 0.0, 0.0)})


def march(c: Choreo, t0: float, counts: int, lift: float, swing: bool, first="left", base=None):
    """Steps in place, one per count, weight rocking onto the stance foot."""
    base = base or {"left": (0.0, 0.0), "right": (0.0, 0.0)}
    for k in range(counts):
        t = t0 + k
        foot = first if k % 2 == 0 else ("right" if first == "left" else "left")
        sign = -1.0 if foot == "left" else 1.0  # weight onto the other foot
        step(c, t, foot, *base[foot], lift=lift, base=base[foot])
        c.at(t + 0.5, shift=0.55 * sign, dz=-0.003)
        c.at(t + 1.0, shift=0.0, dz=0.0)
        if swing:
            # The arm opposite the lifted leg swings forward.
            fwd, back = (55.0, 8.0, 35.0), (-35.0, 8.0, 15.0)
            c.at(t + 0.5, ra=fwd if foot == "left" else back, la=back if foot == "left" else fwd)
    if swing:
        arms(c, t0 + counts, both=DOWN)


def choreograph() -> tuple[Choreo, list[tuple[float, str]]]:
    c = Choreo()
    sections = []
    t = 0.0
    neutral(c, t)

    # 预备: stand, settle into soft knees.
    sections.append((t, "预备"))
    c.at(t, soft=0.0)
    c.at(t + 1.6, soft=SOFT)
    neutral(c, t + 2.0)
    t += 2.0

    # 原地踏步: march with arm swings.
    sections.append((t, "原地踏步"))
    march(c, t, 8, lift=0.014, swing=True)
    t += 8.0
    neutral(c, t)

    # 伸展运动: forward, overhead (looking left, then right), side, down; knees soft on 1 and 3.
    sections.append((t, "伸展运动"))
    neutral(c, t)
    for rep, look in enumerate((25.0, -25.0)):
        b = t + 4 * rep
        arms(c, b + 1, both=(90.0, 6.0, 6.0), dz=-0.006, head=0.0, pitch=0.0)
        arms(c, b + 2, both=(PITCH_MAX, 15.0, 4.0), dz=0.0, head=look, pitch=-5.0)
        arms(c, b + 3, both=(0.0, 90.0, 6.0), dz=-0.009, head=0.0, pitch=0.0)
        arms(c, b + 4, both=DOWN, dz=0.0)
    t += 8.0

    # 扩胸运动: bent arms in front, elbows open back; straight arms in front, open back.
    sections.append((t, "扩胸运动"))
    neutral(c, t)
    for rep in range(2):
        b = t + 4 * rep
        arms(c, b + 1, both=(90.0, 12.0, ELBOW_MAX), dz=-0.006, pitch=0.0, head=0.0)
        arms(c, b + 2, both=(90.0, 100.0, ELBOW_MAX), dz=0.0, pitch=-5.0, head=8.0 * (1 - 2 * rep))
        arms(c, b + 3, both=(90.0, 8.0, 6.0), dz=-0.006, pitch=0.0, head=0.0)
        arms(c, b + 4, both=(88.0, ROLL_MAX, 6.0), dz=0.0, pitch=-5.0)
    arms(c, t + 9.0, both=DOWN, pitch=0.0)
    t += 9.0

    # 踢腿运动: weight onto one foot, arms out, kick forward, set down; then the other leg.
    sections.append((t, "踢腿运动"))
    neutral(c, t)
    for rep, kick in enumerate(SIDES):
        b = t + 4 * rep
        key = "lf" if kick == "left" else "rf"
        sign = -1.0 if kick == "left" else 1.0
        c.at(b, shift=0.0, dz=0.0, **{key: (0.0, 0.0, 0.0, 0.0)})
        arms(c, b, both=DOWN, head=0.0)
        # Leaning over the stance foot keeps its hip inside its adduction range.
        c.at(b + 1.0, shift=0.95 * sign, dz=-0.006, roll=6.0 * sign, **{key: (0.0, 0.0, 0.0, 0.0)})
        arms(c, b + 1.0, both=(0.0, 90.0, 6.0))
        c.at(b + 1.2, shift=0.95 * sign, **{key: (0.0, 0.0, 0.0, 0.0)})
        c.at(b + 2.0, dz=-0.004, pitch=4.0, **{key: (0.045, 0.012 * sign, 0.028, 0.0)})
        arms(c, b + 2.0, both=(10.0, 95.0, 4.0), head=-12.0 * sign)
        c.at(b + 2.8, pitch=0.0, **{key: (0.0, 0.0, 0.0, 0.0)})
        c.at(b + 3.0, shift=0.95 * sign, dz=-0.006, roll=6.0 * sign, **{key: (0.0, 0.0, 0.0, 0.0)})
        arms(c, b + 3.0, both=(0.0, 85.0, 6.0), head=0.0)
        c.at(b + 4.0, shift=0.0, dz=0.0, roll=0.0, **{key: (0.0, 0.0, 0.0, 0.0)})
        arms(c, b + 4.0, both=DOWN)
    t += 8.0

    # Step the left foot out: a wide stance for the side bends and twists.
    wide = 0.018
    sections.append((t, "开立"))
    neutral(c, t)
    c.at(t, shift=0.0)
    c.at(t + 0.7, shift=-0.8, dz=-0.012, roll=-5.0)
    step(c, t + 0.7, "left", dy=wide, lift=0.01, beats=1.0)
    c.at(t + 1.7, shift=-0.8, dz=-0.012, roll=-5.0)
    c.at(t + 2.4, shift=0.0, dz=0.0, roll=0.0)
    t += 2.4
    lf_wide = (0.0, wide, 0.0, 0.0)

    # 体侧运动: one arm over the head, the other hand on the hip, lean to the side twice.
    sections.append((t, "体侧运动"))
    neutral(c, t, lf=lf_wide)
    hip_hand = (-25.0, 35.0, ELBOW_MAX)
    for rep, bend in enumerate(("right", "left")):
        b = t + 4 * rep
        s = 1.0 if bend == "left" else -1.0
        up, hip = (PITCH_MAX - 10.0, 40.0, 15.0), hip_hand
        ra, la = (up, hip) if bend == "left" else (hip, up)
        c.at(b + 1, ra=ra, la=la, roll=7.0 * s, head=-12.0 * s, dz=-0.004, lf=lf_wide)
        c.at(b + 2, ra=ra, la=la, roll=10.0 * s, head=-15.0 * s, dz=0.0, lf=lf_wide)
        arms(c, b + 3, both=(0.0, 90.0, 6.0), roll=0.0, head=0.0, dz=-0.004, lf=lf_wide)
        arms(c, b + 4, both=DOWN, dz=0.0, roll=0.0, head=0.0, lf=lf_wide)
    t += 8.0

    # 体转运动: the hips turn the trunk, one arm bent across the front, the other out to the side,
    # the head looking further round; twice to each side.
    sections.append((t, "体转运动"))
    neutral(c, t, lf=lf_wide)
    for rep, turn in enumerate(("left", "right")):
        b = t + 4 * rep
        s = 1.0 if turn == "left" else -1.0
        front, side = (80.0, 8.0, 60.0), (8.0, 100.0, 8.0)
        ra, la = (front, side) if turn == "left" else (side, front)
        c.at(b + 1, ra=ra, la=la, yaw=16.0 * s, head=30.0 * s, dz=-0.004, lf=lf_wide)
        c.at(b + 2, ra=ra, la=la, yaw=22.0 * s, head=38.0 * s, dz=0.0, lf=lf_wide)
        arms(c, b + 3, both=(0.0, 90.0, 6.0), yaw=0.0, head=0.0, dz=-0.004, lf=lf_wide)
        arms(c, b + 4, both=DOWN, dz=0.0, yaw=0.0, head=0.0, lf=lf_wide)
    t += 8.0

    # Bring the left foot back in.
    sections.append((t, "并步"))
    neutral(c, t, lf=lf_wide)
    c.at(t, shift=0.0)
    c.at(t + 0.7, shift=-0.8, dz=-0.012, roll=-5.0)
    step(c, t + 0.7, "left", dy=0.0, lift=0.01, beats=1.0, base=(0.0, wide))
    c.at(t + 1.7, shift=-0.8, dz=-0.012, roll=-5.0)
    c.at(t + 2.4, shift=0.0, dz=0.0, roll=0.0)
    t += 2.4

    # 全身运动: reach up, fold forward with the arms down to the toes, squat with the arms
    # forward, stand.
    sections.append((t, "全身运动"))
    neutral(c, t)
    for rep in range(2):
        b = t + 4 * rep
        arms(c, b + 1, both=(PITCH_MAX, 15.0, 4.0), pitch=-4.0, dz=0.0, head=0.0)
        arms(c, b + 2, both=(45.0, 10.0, 4.0), pitch=28.0, dz=-0.017)
        arms(c, b + 3, both=(90.0, 6.0, 6.0), pitch=12.0, dz=-0.022)
        arms(c, b + 4, both=DOWN, pitch=0.0, dz=0.0)
    t += 8.0

    # 跳跃运动: a knee bounce on every count, the arms swinging out and up and down like jumping
    # jacks; the head turns with the second half.
    sections.append((t, "跳跃运动"))
    neutral(c, t)
    for k in range(8):
        b = t + k
        c.at(b + 0.5, dz=-0.011)
        arms(c, b + 1.0, both=(0.0, 105.0, 8.0) if k % 2 == 0 else (0.0, 30.0, 8.0), dz=0.0,
             head=(15.0 if k in (4, 5) else -15.0 if k in (6, 7) else 0.0))
    arms(c, t + 9.0, both=DOWN, head=0.0)
    t += 1.0
    t += 8.0

    # 整理运动: slow steps in place, the arms rising out to the sides and lowering with the breath.
    sections.append((t, "整理运动"))
    neutral(c, t)
    march(c, t, 8, lift=0.01, swing=False)
    for k in range(2):
        b = t + 4 * k
        arms(c, b + 2, both=(0.0, 85.0, 10.0), pitch=-3.0)
        arms(c, b + 4, both=DOWN, pitch=0.0)
    t += 8.0

    c.at(t, soft=SOFT)
    c.at(t + 1.0, soft=0.0)
    neutral(c, t + 1.0)
    sections.append((t + 1.0, "结束"))
    return c, sections


class Xiaoxiaobai:
    def __init__(self):
        self.model = get_dance_spec().compile()
        self.data = mujoco.MjData(self.model)
        m = self.model
        jid = {j: m.joint(j).id for j in JOINTS}
        self.qadr = {j: m.jnt_qposadr[i] for j, i in jid.items()}
        self.dadr = {j: m.jnt_dofadr[i] for j, i in jid.items()}
        self.lo, self.hi = {}, {}
        for j in JOINTS:
            r = m.jnt_range[jid[j]]
            mid, half = r.mean(), (r[1] - r[0]) / 2
            if j in LEGS:
                half *= LEG_RANGE
            elif j in DANCE_ARM_RANGES_DEG:
                lo, hi = np.radians(DANCE_ARM_RANGES_DEG[j])
                mid, half = (lo + hi) / 2, (hi - lo) / 2
            self.lo[j], self.hi[j] = mid - half, mid + half
        self.foot = {s: m.body(FOOT[s]).id for s in SIDES}
        self.soles = {s: m.geom(f"{s}_foot_collision").id for s in SIDES}
        self.home_q = {j: HOME_FRAME.joint_pos.get(j, 0.0) for j in JOINTS}
        self.qpos = np.zeros(m.nq)
        self.qpos[3] = 1.0
        for j, q in self.home_q.items():
            self.qpos[self.qadr[j]] = q
        self.fk()
        low = min(self.sole_bottom(s) for s in SIDES)
        self.qpos[2] = -low
        self.fk()
        self.home_z = self.qpos[2]
        self.home_foot_pos = {s: self.data.xpos[self.foot[s]].copy() for s in SIDES}
        self.home_foot_rot = {s: self.data.xmat[self.foot[s]].reshape(3, 3).copy() for s in SIDES}
        self.sole_offset = {s: self.data.geom_xpos[self.soles[s]] - self.home_foot_pos[s] for s in SIDES}
        sole_mid = np.mean([self.data.geom_xpos[self.soles[s]][:2] for s in SIDES], axis=0)
        self.com_offset = self.com()[:2] - sole_mid
        half = self.model.geom_size[self.soles["left"]][:2]
        self.sole_corners = np.array([[sx * half[0], sy * half[1]] for sx in (-1, 1) for sy in (-1, 1)])

    def fk(self):
        self.data.qpos[:] = self.qpos
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    def com(self) -> np.ndarray:
        return self.data.subtree_com[1].copy()

    def sole_bottom(self, s) -> float:
        g = self.soles[s]
        R = self.data.geom_xmat[g].reshape(3, 3)
        return self.data.geom_xpos[g, 2] - np.abs(R[2]) @ self.model.geom_size[g]

    def set(self, name, q):
        self.qpos[self.qadr[name]] = q

    def get(self, name) -> float:
        return self.qpos[self.qadr[name]]

    def solve_legs(self, targets: dict, iters=40) -> float:
        """Leg joints for the foot poses {side: (pos, rot, rot weight)}, root fixed."""
        m, d = self.model, self.data
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        err_max = 0.0
        for _ in range(iters):
            self.fk()
            err_max = 0.0
            for s in SIDES:
                pos, rot, w = targets[s]
                b = self.foot[s]
                e_p = pos - d.xpos[b]
                e_r = Rotation.from_matrix(rot @ d.xmat[b].reshape(3, 3).T).as_rotvec()
                err_max = max(err_max, np.abs(e_p).max())
                mujoco.mj_jacBody(m, d, jacp, jacr, b)
                names = [f"{s}_{j}" for j in LEG]
                cols = [self.dadr[n] for n in names]
                J = np.vstack([jacp[:, cols], w * jacr[:, cols]])
                e = np.r_[e_p, w * e_r]
                dq = np.linalg.solve(J.T @ J + DAMP * np.eye(6), J.T @ e)
                for n, v in zip(names, dq):
                    self.set(n, np.clip(self.get(n) + v, self.lo[n], self.hi[n]))
            if err_max < 2e-5:
                break
        self.fk()
        return err_max


def build(beat: float, choreograph=choreograph, channels=None, hook=None):
    """The routine `choreograph` keys, frame by frame. `channels` are extra keyed channels with
    their defaults; `hook(mb, ch, i)` runs after the legs are solved at frame i (the trunk and
    feet in place) and may set the arms, so the balance accounts for them."""
    c, sections = choreograph()
    end = max(max(k) for k in c.keys.values())
    t = np.arange(0.0, end * beat + 1e-9, 1.0 / FPS)
    tb = t / beat
    ch = {k: c.sample(k, tb, v) for k, v in {**NEUTRAL, **(channels or {})}.items()}
    ch["dz"] = ch["dz"] + c.sample("soft", tb, 0.0)
    mb = Xiaoxiaobai()

    n = len(t)
    joint_pos = np.empty((n, len(JOINTS)))
    root_pos = np.empty((n, 3))
    root_quat = np.empty((n, 4))
    ik_err = np.zeros(n)
    com_err = np.zeros(n)
    margin = np.zeros(n)
    stance = np.zeros((n, 2), bool)
    xy = np.zeros(2)
    for i in range(n):
        # Arms and head straight from the keys, clipped to the reference stops.
        for s, key in (("right", "ra"), ("left", "la")):
            pf, ro, eb = ch[key][i]
            pf, ro, eb = min(pf, PITCH_MAX), np.clip(ro, 0.0, ROLL_MAX), np.clip(eb, 0.0, ELBOW_MAX)
            for j, v, sign in zip(("shoulder_pitch", "shoulder_roll", "elbow"), (pf, ro, eb), ARM_SIGN[s]):
                name = f"{s}_{j}"
                mb.set(name, np.clip(np.radians(sign * v), mb.lo[name], mb.hi[name]))
        mb.set("head", np.radians(ch["head"][i, 0]))

        # Keyed roll leans the trunk to the robot's left when positive (about -x).
        rot = Rotation.from_euler("ZYX", np.radians([ch["yaw"][i, 0], ch["pitch"][i, 0], -ch["roll"][i, 0]]))
        mb.qpos[3:7] = rot.as_quat(scalar_first=True)
        mb.qpos[2] = mb.home_z + ch["dz"][i, 0]
        targets, soles = {}, {}
        for s, key in (("left", "lf"), ("right", "rf")):
            dx, dy, lift, yaw = ch[key][i]
            lift = max(lift, 0.0)
            R = Rotation.from_euler("z", np.radians(yaw)).as_matrix()
            pos = mb.home_foot_pos[s] + np.r_[dx, dy, lift]
            targets[s] = (pos, R @ mb.home_foot_rot[s], W_ROT_DOWN if lift < LIFTED else W_ROT_AIR)
            soles[s] = (pos + R @ mb.sole_offset[s])[:2]
            stance[i, SIDES.index(s)] = lift < LIFTED
        # Weight shift: +1 over the left sole, -1 over the right, between them in between.
        w = ch["shift"][i, 0]
        com_target = (soles["left"] + soles["right"]) / 2 + w * (soles["left"] - soles["right"]) / 2 + mb.com_offset
        for _ in range(12):
            mb.qpos[:2] = xy
            ik_err[i] = mb.solve_legs(targets)
            if hook:
                hook(mb, ch, i)
            e = com_target - mb.com()[:2]
            xy = xy + 1.3 * e
            if np.abs(e).max() < 1e-4:
                break
        mb.qpos[:2] = xy
        ik_err[i] = mb.solve_legs(targets)
        if hook:
            hook(mb, ch, i)
        com = mb.com()[:2]
        com_err[i] = np.linalg.norm(com_target - com)
        # Balance margin: distance of the centre of mass inside the hull of the planted soles.
        pts = np.concatenate([soles[s] + mb.sole_corners for k, s in enumerate(SIDES) if stance[i, k]])
        hull = ConvexHull(pts)
        margin[i] = -np.max(hull.equations[:, :2] @ com + hull.equations[:, 2])
        joint_pos[i] = [mb.get(j) for j in JOINTS]
        root_pos[i], root_quat[i] = mb.qpos[:3], mb.qpos[3:7]

    vel = np.gradient(joint_pos, 1.0 / FPS, axis=0)
    print(f"{n} frames at {FPS} Hz, {t[-1]:.1f} s, beat {beat:.2f} s ({60 / beat:.0f} BPM)")
    print("sections:", "  ".join(f"{name} {s * beat:.1f}s" for s, name in sections))
    print("joint                 range used [deg]     |vel| p99 / max [rad/s]   near limit")
    for j, name in enumerate(JOINTS):
        q = joint_pos[:, j]
        near = ((q < mb.lo[name] + 0.02) | (q > mb.hi[name] - 0.02)).mean()
        print(f"  {name:20s} {np.degrees(q.min()):7.1f} .. {np.degrees(q.max()):6.1f}"
              f"     {np.percentile(np.abs(vel[:, j]), 99):5.1f} / {np.abs(vel[:, j]).max():5.1f}      {near:5.1%}")
    single = stance.sum(1) == 1
    print(f"foot IK error max {ik_err.max() * 1000:.2f} mm, CoM off target max {com_err.max() * 1000:.2f} mm")
    leg_ids = [JOINTS.index(j) for j in LEGS]
    flags = {"foot IK > 2 mm": ik_err > 0.002, "CoM > 2 mm off": com_err > 0.002,
             "leg joint > 8 rad/s": np.abs(vel[:, leg_ids]).max(1) > 8.0,
             "knee straight": np.array([any(joint_pos[i, JOINTS.index(f"{s}_knee")] in (mb.lo[f"{s}_knee"], mb.hi[f"{s}_knee"])
                                            for s in SIDES) for i in range(n)])}
    for label, bad in flags.items():
        edges = np.flatnonzero(np.diff(np.r_[0, bad.astype(int), 0]))
        spans = [f"{t[a]:.1f}-{t[b - 1]:.1f}" for a, b in zip(edges[::2], edges[1::2])]
        if spans:
            print(f"  {label}: " + ", ".join(spans[:12]) + (" ..." if len(spans) > 12 else ""))
    print(f"single support {single.mean():.0%} of the time; CoM margin inside the support: "
          f"min {margin.min() * 1000:.1f} mm (double {margin[~single].min() * 1000:.1f},"
          f" single {margin[single].min() * 1000 if single.any() else float('nan'):.1f})")
    print(f"trunk travels {np.abs(root_pos[:, 1]).max() * 1000:.0f} mm sideways at most,"
          f" height {root_pos[:, 2].min():.3f}..{root_pos[:, 2].max():.3f} m")
    return t, joint_pos, root_pos, root_quat, sections, beat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--beat", type=float, default=0.55, help="seconds per count")
    args = ap.parse_args()
    t, joint_pos, root_pos, root_quat, sections, beat = build(args.beat)
    np.savez(args.out, fps=FPS, joint_names=np.array(JOINTS), joint_pos=joint_pos, root_pos=root_pos,
             root_quat=root_quat, beat=beat, section_names=np.array([s for _, s in sections]),
             section_starts=np.array([b * beat for b, _ in sections]))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
