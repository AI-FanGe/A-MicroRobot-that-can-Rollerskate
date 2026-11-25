"""Prop-up and lie-down reference for Xiaoxiaobai: lying on the back, the elbows bend and the
shoulders (pitch back, roll out) press the elbows into the floor, lifting the upper body onto
the elbows and flat forearms; then the hands stay planted and the elbows straighten, pushing
the upper body up to lean back on the hands. The lie-down is the same path played back.

The trunk pivots on the seat with the legs flat on the floor (hip flexion = trunk raise) and
the heels fixed. Each arm (the left mirrors the right) is solved by IK with the elbow planted
where it touched down and the forearm lowered flat by the top.

On the elbows the upper arm (5 cm) lifts the shoulders by ~2 cm: 14 deg of trunk raise, the
CoM 5-7 cm ahead of the elbows (toward the feet). Pushing on the planted hands the arms are
nearly straight by 54 deg; the hands walk back and in, two steps each, while the trunk keeps
rising to 66 deg (the arms' reach). Every frame is checked for the CoM inside the convex hull of
the floor contacts (legs, seat, the hands down).

--time-scale k plays the moves k times as long (lying still for --lie s first, the elbow bend
taking --bend-s s if given, holding the top for --hold s); --steps 1 walks each hand back in
one step of --step-s s, right then left. --smooth s low-passes the arm joints (a Gaussian of
that width): near their reach the arms straighten onto the elbow stop in a jump.

usage: propup_motion.py <out.npz> [--raise 66] [--time-scale 1] [--steps 2] [--step-s 0.6]
                        [--lie 0.8] [--bend-s] [--hold 2.0] [--smooth 0]
"""
import argparse

import mujoco
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares

from mjlab_xiaoxiaobai.robot.xiaoxiaobai_constants import get_getup_spec

FPS = 50
SIDES = ("left", "right")
LEG = ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch", "ankle_roll")
LEGS = [f"{s}_{j}" for s in SIDES for j in LEG]
ARMS = [f"{s}_{j}" for s in SIDES for j in ("shoulder_pitch", "shoulder_roll", "elbow")]
JOINTS = LEGS + ARMS + ["head"]
READY = {"left_shoulder_roll": -0.0873, "right_shoulder_roll": 0.0873, "left_hip_roll": -0.0611,
         "right_hip_roll": 0.0611, "left_ankle_roll": 0.0611, "right_ankle_roll": -0.0611}
RIGHT_ARM = ("right_shoulder_pitch", "right_shoulder_roll", "right_elbow")
ARM_BODIES = ("shoulder", "shoulder_2", "humerus__configuration_right", "humerus__configuration_left",
              "radius", "radius_2")
SOFT = 0.9  # soft limits: 0.9 of the measured stops
# Elbow touch-down [m]: toward the feet from the shoulder, out from the trunk centre.
ELBOW_DX, ELBOW_Y = 0.0, 0.11
# Trunk raise [deg] on the elbows (forearms flat), the most the upper arm reaches.
ELBOW_RAISE = 14.0
# Forearm angle to the floor [deg] when the elbow touches down; flat (0) at the top.
FOREARM_UP = 30.0
# Timeline of the up half [s]; the down half plays it back.
UP_S = 12.6
BEND = (0.8, 1.9)  # elbows bend out and press down
# Trunk raise (time [s], deg), scaled to --raise: flat until the elbows are down, through the
# elbow prop to 14 deg, then on the hands without a stop while they walk.
RAISE_KNOTS = ((1.8, 0.0), (4.0, 14.0), (6.2, 44.0), (7.2, 52.0), (8.4, 56.0), (9.7, 63.0), (10.6, 66.0))
# The hands walk back (toward the head) and in [m] in two steps each, right then left, lifted
# STEP_LIFT [m] for STEP_S [s], while the trunk keeps rising: start times (right, left) [s].
# The first steps wait for ~52 deg: lower, the CoM is outside the support on one hand.
WALK_BACK, WALK_IN = -0.02, 0.025
STEP_TIMES = ((7.2, 7.85), (8.6, 9.25))
STEP_LIFT, STEP_S = 0.012, 0.6
DOWN_NAMES = {"躺着": "躺好", "屈肘": "收手", "肘撑": "放下", "撑起": "躺下", "撑住": "撑住"}
# IK keeps this far [rad] inside the soft limits, so tracking errors stay off the stops.
IK_MARGIN = np.deg2rad(3.0)


def smooth(s):
    s = np.clip(s, 0.0, 1.0)
    return s * s * s * (s * (6 * s - 15) + 10)


class Xiaoxiaobai:
    def __init__(self):
        self.m = get_getup_spec().compile()
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.adr = {m.joint(j).name: m.jnt_qposadr[j] for j in range(m.njnt)}
        self.hulls = [g for g in range(m.ngeom) if m.geom(g).name.endswith("_ground_collision")]
        self.arm_hulls = {m.geom(f"{b}_ground_collision").id for b in ARM_BODIES
                          if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{b}_ground_collision") >= 0}
        ids = [m.joint(j).id for j in RIGHT_ARM]
        lo, hi = m.jnt_range[ids, 0], m.jnt_range[ids, 1]
        mid, half = (lo + hi) / 2, (hi - lo) / 2 * SOFT
        self.arm_lo, self.arm_hi = mid - half + IK_MARGIN, mid + half - IK_MARGIN
        named = {g: m.body(m.geom_bodyid[g]).name for g in self.hulls}
        self.seat_hulls = [g for g, b in named.items() if b in ("trunk", "hip", "hip_2") or b.startswith("hip_block")]
        self.leg_hulls = [g for g, b in named.items() if b.split("__")[0] in ("femur", "tibia", "ankle_block")
                          or b in ("foot", "foot_2")]
        self.fore_geom = m.geom("radius_ground_collision").id
        self.fore_body = m.body("radius").id
        self.rest_x = np.array([0.0, READY["right_shoulder_roll"], 0.0])
        self.place(0.0, self.rest_x)
        v = self._hull_world(self.fore_geom)
        b = self.fore_body
        local = (v - self.d.xpos[b]) @ self.d.xmat[b].reshape(3, 3)
        self.prox = np.linalg.norm(local, axis=1) < 0.025
        tip = v[v[:, 2] < v[:, 2].min() + 0.006].mean(axis=0)
        self.hand_local = self.d.xmat[b].reshape(3, 3).T @ (tip - self.d.xpos[b])
        self.shoulder = self.d.xpos[m.body("humerus__configuration_right").id].copy()

    def _hull_world(self, g):
        mid = self.m.geom_dataid[g]
        a, n = self.m.mesh_vertadr[mid], self.m.mesh_vertnum[mid]
        return self.m.mesh_vert[a:a + n] @ self.d.geom_xmat[g].reshape(3, 3).T + self.d.geom_xpos[g]

    def _set(self, raise_rad, q, root):
        d = self.d
        d.qpos[:] = 0
        d.qpos[:3] = root
        pitch = -np.pi / 2 + raise_rad
        d.qpos[3:7] = [np.cos(pitch / 2), 0, np.sin(pitch / 2), 0]
        for k, v in {**READY, **q}.items():
            d.qpos[self.adr[k]] = v
        mujoco.mj_kinematics(self.m, self.d)

    def _low(self, geoms):
        return min(self._hull_world(g)[:, 2].min() for g in geoms)

    def hip_flex(self, raise_rad):
        """Hip flexion that rests the legs on the floor with the seat: the pelvis rolls on the
        hip blocks, so flexing by the trunk raise lifts the legs off."""
        key = round(float(raise_rad), 6)
        cache = self.__dict__.setdefault("_hip_cache", {})
        if key not in cache:
            def gap(h):
                self._set(raise_rad, {"right_hip_pitch": -h, "left_hip_pitch": h}, (0.0, 0.0, 0.3))
                return self._low(self.leg_hulls) - self._low(self.seat_hulls)
            lo, hi = raise_rad - np.deg2rad(40), raise_rad + np.deg2rad(5)
            if gap(lo) * gap(hi) > 0:
                cache[key] = raise_rad
            else:
                for _ in range(40):
                    mid = 0.5 * (lo + hi)
                    if gap(mid) * gap(lo) > 0:
                        lo = mid
                    else:
                        hi = mid
                cache[key] = 0.5 * (lo + hi)
        return cache[key]

    def place(self, raise_rad, x, x_left=None):
        """On the back raised by ``raise_rad`` (head toward -x), right arm ``x`` (the left its
        mirror, or the mirror of ``x_left``), legs resting on the floor, the body (arms
        excluded) on the floor with the heels at x = 0."""
        p, r, e = x
        lp, lr, le = x if x_left is None else x_left
        h = self.hip_flex(raise_rad)
        q = {"right_hip_pitch": -h, "left_hip_pitch": h,
             "right_shoulder_pitch": p, "right_shoulder_roll": r, "right_elbow": e,
             "left_shoulder_pitch": -lp, "left_shoulder_roll": -lr, "left_elbow": -le}
        self._set(raise_rad, q, (0.0, 0.0, 0.3))
        dz = -min(self._hull_world(g)[:, 2].min() for g in self.hulls if g not in self.arm_hulls)
        foot = self.d.site_xpos[self.m.site("right_foot").id][0]
        self._set(raise_rad, q, (-foot, 0.0, 0.3 + dz))
        return q

    def elbow_low(self):
        return self._hull_world(self.fore_geom)[self.prox, 2].min()

    def hand_low(self):
        return self._hull_world(self.fore_geom)[~self.prox, 2].min()

    def hand_xy(self):
        b = self.fore_body
        return (self.d.xpos[b] + self.d.xmat[b].reshape(3, 3) @ self.hand_local)[:2]

    def elbow_xy(self):
        return self.d.xpos[self.fore_body][:2]

    def forearm_angle(self):
        u = self.d.xpos[self.fore_body] + self.d.xmat[self.fore_body].reshape(3, 3) @ self.hand_local
        u = u - self.d.xpos[self.fore_body]
        return np.arctan2(u[2], np.linalg.norm(u[:2]))

    def lowest_arm(self):
        return min(self._hull_world(g)[:, 2].min() for g in self.arm_hulls)

    def com(self):
        mujoco.mj_comPos(self.m, self.d)
        return self.d.subtree_com[1].copy()


def solve(mb, raise_rad, elbow_xy, alpha, x0, flat):
    """Arm with the elbow on the floor at ``elbow_xy`` and the forearm at ``alpha`` to the
    floor (or the hand end on the floor too if ``flat``)."""
    def res(x):
        mb.place(raise_rad, x)
        r = [mb.elbow_low(), mb.elbow_xy()[0] - elbow_xy[0], 0.3 * (mb.elbow_xy()[1] - elbow_xy[1])]
        r.append(mb.hand_low() if flat else 0.02 * (mb.forearm_angle() - alpha))
        return np.array(r + list(0.001 * (x - x0)))

    sol = least_squares(res, np.clip(x0, mb.arm_lo + 1e-6, mb.arm_hi - 1e-6), bounds=(mb.arm_lo, mb.arm_hi))
    r = res(sol.x)
    mb.place(raise_rad, sol.x)
    return sol.x, max(abs(r[0]), abs(r[1]), abs(r[3]) if flat else 0.0)


def solve_hand(mb, raise_rad, hand, x0, lift=0.0):
    """Right arm with the hand end ``lift`` [m] above the floor over ``hand`` (x, y). For the
    left arm pass its target mirrored (y -> -y): the result is its right-arm equivalent."""
    def res(x):
        mb.place(raise_rad, x)
        return np.array([*(mb.hand_xy() - hand), mb.hand_low() - lift, *(0.001 * (x - x0))])

    best = None
    for start in (x0, np.array([-0.3, 0.15, -0.2]), np.array([-0.5, 0.1, -0.6])):
        sol = least_squares(res, np.clip(start, mb.arm_lo + 1e-6, mb.arm_hi - 1e-6), bounds=(mb.arm_lo, mb.arm_hi))
        err = float(np.abs(res(sol.x)[:3]).max())
        # Another start only for a clearly better fit: near the reach the fits tie, and
        # switching between them jumps the arm from one frame to the next.
        if best is None or err < best[1] - 1e-3:
            best = (sol.x, err)
        if err < 2e-4:
            break
    mb.place(raise_rad, best[0])
    return best


def raise_curve(raise_max):
    """Trunk raise [rad] over time: one smooth rise through the elbow prop (14 deg at the
    hand-over to the hands) and the hand walk, no stops on the way."""
    knots = np.array(RAISE_KNOTS)
    t, deg = knots[:, 0], knots[:, 1] * np.rad2deg(raise_max) / knots[-1, 1]
    curve = PchipInterpolator(t, np.deg2rad(deg))
    return lambda time: float(curve(np.clip(time, t[0], t[-1])))


def up_half(mb, raise_max):
    """Frames (label, raise, right arm x, left arm x) of the prop-up, with the worst IK error [m].

    Lying, the elbows bend out and press down; the trunk then rises in one go: on the elbows
    to ELBOW_RAISE (the forearms lowering flat meanwhile), on the planted hands after, and the
    hands walk back and in, one at a time, while it keeps rising."""
    raise_at = raise_curve(raise_max)
    t_elbow = next(t / FPS for t in range(20 * FPS) if raise_at(t / FPS) >= np.deg2rad(ELBOW_RAISE) - 1e-9)
    elbow = np.array([mb.shoulder[0] + ELBOW_DX, -ELBOW_Y])
    touch, _ = solve(mb, 0.0, elbow, np.deg2rad(FOREARM_UP), np.array([-0.6, 0.5, -1.6]), flat=False)
    steps = [(side, k, t0) for k, (tr, tl) in enumerate(STEP_TIMES, start=1) for side, t0 in (("right", tr), ("left", tl))]
    frames, worst = [], 0.0
    x = xr = xl = touch.copy()
    hand = hr = hl = None
    for i in range(int(round(UP_S * FPS))):
        t = (i + 1) / FPS
        raise_rad = raise_at(t) if t >= RAISE_KNOTS[0][0] else 0.0
        if t < BEND[0]:
            label, xr = "躺着", mb.rest_x.copy()
            xl = xr
        elif t <= BEND[1] and raise_rad <= 1e-6:
            label = "屈肘"
            s = (t - BEND[0]) / (BEND[1] - BEND[0])
            # Roll out and bend first (forearm up), then the shoulder presses the elbow down.
            a, b = smooth(min(1.0, s / 0.6)), smooth(max(0.0, (s - 0.3) / 0.7))
            xr = mb.rest_x + np.array([b * (touch[0] - mb.rest_x[0]), a * (touch[1] - mb.rest_x[1]),
                                       a * (touch[2] - mb.rest_x[2])])
            xl = xr
        elif t <= t_elbow:
            label = "肘撑"
            s = smooth((t - RAISE_KNOTS[0][0]) / (t_elbow - RAISE_KNOTS[0][0]))
            x, err = solve(mb, raise_rad, elbow, np.deg2rad(FOREARM_UP) * (1.0 - s), xr, flat=t >= t_elbow - 0.5 / FPS)
            worst = max(worst, err)
            xr = xl = x
            hand = hr = hl = mb.hand_xy().copy()
        else:
            label = "撑起" if raise_rad < raise_max - 1e-4 else "撑住"
            target = hand + np.array([WALK_BACK, WALK_IN])
            arms = {}
            for side, h, xs in (("right", hr, xr), ("left", hl, xl)):
                lift = 0.0
                for sd, k, t0 in steps:
                    if sd != side or t < t0:
                        continue
                    a, b = hand + (target - hand) * (k - 1) / len(STEP_TIMES), hand + (target - hand) * k / len(STEP_TIMES)
                    s = (t - t0) / STEP_S
                    h = b if s >= 1.0 else a + (b - a) * smooth(s)
                    if s < 1.0:
                        lift = STEP_LIFT * np.sin(np.pi * s)
                xs, err = solve_hand(mb, raise_rad, h, xs, lift=lift)
                worst = max(worst, err)
                arms[side] = xs
            xr, xl = arms["right"], arms["left"]
        frames.append((label, raise_rad, xr.copy(), xl.copy()))
    return frames, worst


def support_margin(mb):
    """Distance [m] of the CoM inside the convex hull of the floor contacts (negative outside)."""
    from scipy.spatial import ConvexHull

    pts = []
    for g in mb.hulls:
        v = mb._hull_world(g)
        pts.append(v[v[:, 2] < 0.002, :2])
    pts = np.concatenate(pts)
    c = mb.com()[:2]
    return min(-(eq[:2] @ c + eq[2]) for eq in ConvexHull(pts).equations)


def retime(k, steps, step_s, lie, hold, bend_s=None):
    """Rescale the timeline: moves k times as long, after `lie` s lying still, `hold` s at the
    top; the elbow bend takes `bend_s` s if given. One step per hand starts each hand at the
    time of the first right step, in turn."""
    global BEND, RAISE_KNOTS, STEP_TIMES, STEP_S, UP_S
    t0, t1 = BEND
    bend_s = (t1 - t0) * k if bend_s is None else bend_s

    def at(t):
        return lie + (t - t0) * bend_s / (t1 - t0) if t <= t1 else lie + bend_s + (t - t1) * k

    BEND = (lie, at(BEND[1]))
    RAISE_KNOTS = tuple((at(t), deg) for t, deg in RAISE_KNOTS)
    STEP_S = step_s
    if steps == 1:
        tr = at(STEP_TIMES[0][0])
        STEP_TIMES = ((tr, tr + step_s + 0.05),)
    else:
        STEP_TIMES = tuple((at(a), at(b)) for a, b in STEP_TIMES)
    UP_S = max(RAISE_KNOTS[-1][0], STEP_TIMES[-1][1] + step_s) + hold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?")
    ap.add_argument("--raise", dest="raise_deg", type=float, default=66.0)
    ap.add_argument("--time-scale", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=2, choices=(1, 2))
    ap.add_argument("--step-s", type=float, default=STEP_S)
    ap.add_argument("--lie", type=float, default=BEND[0])
    ap.add_argument("--bend-s", type=float, default=None)
    ap.add_argument("--smooth", type=float, default=0.0)
    ap.add_argument("--hold", type=float, default=UP_S - RAISE_KNOTS[-1][0])
    args = ap.parse_args()
    retime(args.time_scale, args.steps, args.step_s, args.lie, args.hold, args.bend_s)
    mb = Xiaoxiaobai()
    up, worst = up_half(mb, np.deg2rad(args.raise_deg))
    frames = up + [(DOWN_NAMES[l], r, xr, xl) for l, r, xr, xl in reversed(up)]
    n = len(frames)
    joint_pos = np.zeros((n, len(JOINTS)))
    root_pos, root_quat = np.zeros((n, 3)), np.zeros((n, 4))
    floor, margin, head_z = 0.0, (np.inf, 0.0), []
    for i, (label, raise_rad, xr, xl) in enumerate(frames):
        q = {**READY, **mb.place(raise_rad, xr, xl)}
        joint_pos[i] = [q.get(j, 0.0) for j in JOINTS]
        root_pos[i], root_quat[i] = mb.d.qpos[:3], mb.d.qpos[3:7]
        floor = min(floor, mb.lowest_arm())
        if raise_rad > np.deg2rad(3):
            margin = min(margin, (support_margin(mb), i / FPS))
        head_z.append(mb.d.xpos[mb.m.body("head").id][2])
    if args.smooth > 0:
        from scipy.ndimage import gaussian_filter1d

        cols = [JOINTS.index(j) for j in ARMS]
        raw = joint_pos[:, cols].copy()
        joint_pos[:, cols] = gaussian_filter1d(raw, args.smooth * FPS, axis=0, mode="nearest")
        print(f"arm smoothing moved the joints by <= {np.rad2deg(np.abs(joint_pos[:, cols] - raw).max()):.1f} deg")
    vel = np.abs(np.gradient(joint_pos, 1.0 / FPS, axis=0))
    j = int(vel.max(0).argmax())
    print(f"fastest joint {JOINTS[j]} {vel[:, j].max():.1f} rad/s at {vel[:, j].argmax() / FPS:.2f} s")
    names, starts = [], []
    for i, (label, *_) in enumerate(frames):
        if not names or names[-1] != label:
            names.append(label)
            starts.append(i / FPS)
    arm = np.rad2deg(joint_pos[:, [JOINTS.index(j) for j in RIGHT_ARM]])
    print(f"{n} frames, {n / FPS:.1f} s, raise {args.raise_deg:.0f} deg; IK error <= {worst * 1000:.1f} mm; "
          f"arms into the floor <= {-floor * 1000:.1f} mm; CoM inside the support >= {margin[0] * 1000:.0f} mm "
          f"(at {margin[1]:.1f} s)")
    print(f"right arm [deg]: pitch {arm[:, 0].min():.0f}..{arm[:, 0].max():.0f}, roll {arm[:, 1].min():.0f}.."
          f"{arm[:, 1].max():.0f}, elbow {arm[:, 2].min():.0f}..{arm[:, 2].max():.0f}; "
          f"head {head_z[0] * 100:.1f} -> {max(head_z) * 100:.1f} cm")
    print("sections:", ", ".join(f"{a} {b:.1f}" for a, b in zip(names, starts)))
    if args.out:
        np.savez(args.out, fps=FPS, joint_names=np.array(JOINTS), joint_pos=joint_pos, root_pos=root_pos,
                 root_quat=root_quat, section_names=np.array(names), section_starts=np.array(starts))
        print(args.out)


if __name__ == "__main__":
    main()
