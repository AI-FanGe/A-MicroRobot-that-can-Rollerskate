"""Retarget a G1 motion CSV (LAFAN1 layout, 30 fps) to Xiaoxiaobai.

Each leg segment (thigh, shank, foot) of Xiaoxiaobai turns away from its rest pose by the rotation
the G1 segment makes from the G1 rest pose, relative to the pelvis: both stand on straight legs
at zero. The arms point along the G1 upper arm and forearm in the torso frame (the G1 rests
with its forearms forward). Xiaoxiaobai's joints are solved for those segment directions by damped
least squares inside Xiaoxiaobai's working ranges, together with the root position: the trunk takes the
pelvis orientation, and the soles the G1 has down are pinned to the floor where they set down.
One sole is always down (hops become steps in place).

usage: retarget_g1_to_xiaoxiaobai.py <csv> <out.npz> [--start 1:30] [--end 2:00]
"""
import argparse

import mujoco
import numpy as np
from scipy.ndimage import binary_closing, binary_opening, gaussian_filter1d
from scipy.spatial.transform import Rotation

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec as g1_spec
from mjlab_xiaoxiaobai.robot.xiaoxiaobai_constants import DANCE_ARM_RANGES_DEG, XIAOXIAOBAI_XML

FPS = 30
# Working ranges [deg] of the arms on the robot, as the dance task trains them. G1 bends its
# elbows deeper than Xiaoxiaobai's 85 deg; the reference stops short of that soft limit so the
# robot can follow it without riding the joint limit penalty.
ELBOW_REF_MAX_DEG = 78.0
ARM_RANGES_DEG = {
    j: (max(lo, -ELBOW_REF_MAX_DEG), min(hi, ELBOW_REF_MAX_DEG)) if j.endswith("elbow") else (lo, hi)
    for j, (lo, hi) in DANCE_ARM_RANGES_DEG.items()
}
LEGS = [f"{s}_{j}" for s in ("left", "right")
        for j in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch", "ankle_roll")]
ARMS = [f"{s}_{j}" for s in ("left", "right") for j in ("shoulder_pitch", "shoulder_roll", "elbow")]
JOINTS = LEGS + ARMS
# Xiaoxiaobai (point from, point to) per segment; "hand" is the forearm tip.
MB_BODIES = {
    "left": dict(hip="femur__configuration_left", knee="tibia__configuration_left",
                 ankle="ankle_block__configuration_left", foot="foot_2",
                 shoulder="humerus__configuration_left", elbow="radius_2"),
    "right": dict(hip="femur__configuration_right", knee="tibia__configuration_right",
                  ankle="ankle_block__configuration_right", foot="foot",
                  shoulder="humerus__configuration_right", elbow="radius"),
}
FOREARM = 0.063
# G1 bodies that carry each segment.
G1_SEG = {"thigh": "hip_yaw_link", "shank": "knee_link", "foot": "ankle_roll_link",
          "upper": "shoulder_yaw_link", "fore": "elbow_link"}
# Residual weights: segment directions, foot orientation [rad], planted soles [m], joint change
# from the last frame, root off its guess [m] (the scaled pelvis path, the last height).
W_DIR, W_FOOT, W_PIN, W_SMOOTH, W_ROOT, DAMP = 1.0, 0.5, 30.0, 0.05, 0.3, 1e-4
# A G1 sole within this of the floor and slower than this is down [m, m/s].
G1_DOWN_Z, G1_DOWN_SPEED = 0.025, 0.3
# A Xiaoxiaobai sole this close to the floor counts as down in the report [m].
STANCE_Z = 0.004
# Fastest the arm joints move [rad/s].
ARM_VMAX = 6.0


def rate_limit(q: np.ndarray, step: float) -> np.ndarray:
    """Mean of the forward and backward step-limited paths: no lag, no step above `step`."""
    def one_way(q):
        out = q.copy()
        for t in range(1, len(q)):
            out[t] = out[t - 1] + np.clip(q[t] - out[t - 1], -step, step)
        return out
    return (one_way(q) + one_way(q[::-1])[::-1]) / 2


def parse_time(s: str) -> float:
    m, _, sec = s.rpartition(":")
    return float(m or 0) * 60 + float(sec)


class G1:
    def __init__(self):
        self.model = g1_spec().compile()
        self.data = mujoco.MjData(self.model)
        self.body = {n: self.model.body(n).id for n in
                     ["pelvis", "torso_link"] + [f"{s}_{b}" for s in ("left", "right") for b in G1_SEG.values()]}
        self.rest = self.segment_rotations(np.r_[0, 0, 0.8, 1, 0, 0, 0, np.zeros(self.model.nq - 7)])

    def fk(self, qpos):
        self.data.qpos[:] = qpos
        mujoco.mj_kinematics(self.model, self.data)

    def segment_rotations(self, qpos) -> dict:
        """Each segment's rotation relative to the pelvis (legs) or torso (arms)."""
        self.fk(qpos)
        mat = lambda n: self.data.xmat[self.body[n]].reshape(3, 3)
        out = {}
        for s in ("left", "right"):
            for seg, body in G1_SEG.items():
                parent = mat("pelvis") if seg in ("thigh", "shank", "foot") else mat("torso_link")
                out[(s, seg)] = parent.T @ mat(f"{s}_{body}")
        return out

    def deltas(self, qpos) -> dict:
        rot = self.segment_rotations(qpos)
        return {k: rot[k] @ self.rest[k].T for k in rot}

    def contacts(self, qpos: np.ndarray) -> np.ndarray:
        """(frames, 2) left/right sole down: near the floor and not sliding, held >= 3 frames."""
        geoms = [[g for g in range(self.model.ngeom) if (self.model.geom(g).name or "").startswith(f"{s}_foot")]
                 for s in ("left", "right")]
        ankles = [self.model.body(f"{s}_ankle_roll_link").id for s in ("left", "right")]
        z, xy = np.empty((len(qpos), 2)), np.empty((len(qpos), 2, 2))
        for t, q in enumerate(qpos):
            self.fk(q)
            for i, gs in enumerate(geoms):
                z[t, i] = min(self.data.geom_xpos[g, 2] - abs(self.data.geom_xmat[g].reshape(3, 3)[2, 2])
                              * self.model.geom_size[g, 1] - self.model.geom_size[g, 0] for g in gs)
                xy[t, i] = self.data.xpos[ankles[i], :2]
        speed = np.linalg.norm(np.gradient(xy, 1 / FPS, axis=0), axis=2)
        down = (z < np.percentile(z, 2) + G1_DOWN_Z) & (speed < G1_DOWN_SPEED)
        for i in range(2):
            down[:, i] = binary_opening(binary_closing(down[:, i], np.ones(3)), np.ones(3))
        # Airborne frames keep the lower sole down: Xiaoxiaobai does not hop.
        none = ~down.any(1)
        down[none, np.argmin(z[none], axis=1)] = True
        return down

    def arm_dirs(self) -> dict:
        """Upper arm and forearm directions in the torso frame, after fk()."""
        torso = self.data.xmat[self.body["torso_link"]].reshape(3, 3)
        pos = lambda n: self.data.xpos[self.model.body(n).id]
        out = {}
        for s in ("left", "right"):
            for seg, (a, b) in dict(upper=("shoulder_pitch_link", "elbow_link"),
                                    fore=("elbow_link", "wrist_roll_link")).items():
                v = torso.T @ (pos(f"{s}_{b}") - pos(f"{s}_{a}"))
                out[(s, seg)] = v / np.linalg.norm(v)
        return out


class Xiaoxiaobai:
    def __init__(self):
        spec = mujoco.MjSpec.from_file(str(XIAOXIAOBAI_XML))
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.qadr = np.array([m.jnt_qposadr[m.joint(j).id] for j in JOINTS])
        lo, hi = [], []
        for j in JOINTS:
            if j in ARM_RANGES_DEG:
                a, b = np.radians(ARM_RANGES_DEG[j])
            else:
                r = m.jnt_range[m.joint(j).id]
                mid, half = r.mean(), 0.9 * (r[1] - r[0]) / 2
                a, b = mid - half, mid + half
            lo.append(a)
            hi.append(b)
        self.lo, self.hi = np.array(lo), np.array(hi)
        self.bid = {s: {k: m.body(v).id for k, v in d.items()} for s, d in MB_BODIES.items()}
        self.soles = [m.geom(f"{s}_foot_collision").id for s in ("left", "right")]
        self.qpos0 = np.zeros(m.nq)
        self.qpos0[3] = 1.0
        self.fk(np.zeros(len(JOINTS)))
        self.rest_dirs = self.segment_dirs()
        self.rest_foot = {s: self.data.xmat[self.bid[s]["foot"]].reshape(3, 3).copy() for s in ("left", "right")}
        hip_z = self.data.xpos[self.bid["left"]["hip"], 2]
        self.leg_length = hip_z - self.sole_z().min()

    def fk(self, q, root=None):
        qpos = self.qpos0.copy()
        if root is not None:
            qpos[:7] = root
        qpos[self.qadr] = q
        self.data.qpos[:] = qpos
        mujoco.mj_kinematics(self.model, self.data)

    def point(self, s, k):
        if k == "hand":
            e = self.bid[s]["elbow"]
            return self.data.xpos[e] + self.data.xmat[e].reshape(3, 3) @ np.array([0, 0, -FOREARM])
        return self.data.xpos[self.bid[s][k]].copy()

    def segment_dirs(self) -> dict:
        out = {}
        for s in ("left", "right"):
            for seg, (a, b) in dict(thigh=("hip", "knee"), shank=("knee", "ankle"),
                                    upper=("shoulder", "elbow"), fore=("elbow", "hand")).items():
                v = self.point(s, b) - self.point(s, a)
                out[(s, seg)] = v / np.linalg.norm(v)
        return out

    def sole_z(self) -> np.ndarray:
        out = []
        for g in self.soles:
            R = self.data.geom_xmat[g].reshape(3, 3)
            out.append(self.data.geom_xpos[g, 2] - np.abs(R[2]) @ self.model.geom_size[g])
        return np.array(out)

    def sole_xy(self) -> np.ndarray:
        return np.array([self.data.geom_xpos[g, :2] for g in self.soles])

    def residual(self, x, quat, targets) -> np.ndarray:
        """x: the joints then the root position; the root turns with the G1 pelvis."""
        q, root = x[:-3], x[-3:]
        self.fk(q, root=np.r_[root, quat])
        dirs = self.segment_dirs()
        r = [W_DIR * (dirs[k] - targets["dirs"][k]) for k in dirs]
        for s in ("left", "right"):
            R = self.data.xmat[self.bid[s]["foot"]].reshape(3, 3)
            r.append(W_FOOT * Rotation.from_matrix(R @ targets["foot"][s].T).as_rotvec())
        z, xy = self.sole_z(), self.sole_xy()
        for i, s in enumerate(("left", "right")):
            if s in targets["down"]:
                r.append([W_PIN * z[i]])
                pin = targets["pins"].get(s)
                r.append(W_PIN * (xy[i] - pin) if pin is not None else [0.0, 0.0])
            else:
                r.append([W_PIN * min(z[i], 0.0), 0.0, 0.0])
        r.append(W_SMOOTH * (q - targets["q_prev"]))
        r.append(W_ROOT * (root - targets["root_guess"]))
        return np.concatenate(r)

    def solve(self, x, quat, targets, iters=15) -> np.ndarray:
        eps = 1e-6
        for _ in range(iters):
            r0 = self.residual(x, quat, targets)
            J = np.empty((len(r0), len(x)))
            for i in range(len(x)):
                dx = x.copy()
                dx[i] += eps
                J[:, i] = (self.residual(dx, quat, targets) - r0) / eps
            step = np.linalg.solve(J.T @ J + DAMP * np.eye(len(x)), -J.T @ r0)
            x = x + step
            x[:-3] = np.clip(x[:-3], self.lo, self.hi)
            if np.abs(step).max() < 1e-6:
                break
        return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("out")
    ap.add_argument("--start", default="0:00")
    ap.add_argument("--end", default="")
    ap.add_argument("--smooth", type=float, default=1.0, help="Gaussian sigma [frames] on the G1 motion")
    args = ap.parse_args()

    raw = np.loadtxt(args.csv, delimiter=",")
    a = int(parse_time(args.start) * FPS)
    b = int(parse_time(args.end) * FPS) if args.end else len(raw)
    raw = raw[a:b]
    g1_q = np.concatenate([raw[:, :3], raw[:, [6, 3, 4, 5]], raw[:, 7:]], axis=1)

    g1, mb = G1(), Xiaoxiaobai()
    g1.fk(np.r_[0, 0, 0.8, 1, 0, 0, 0, np.zeros(g1.model.nq - 7)])
    g1_leg = g1.data.xpos[g1.model.body("left_hip_pitch_link").id, 2] - g1.data.xpos[g1.model.body("left_ankle_roll_link").id, 2] + 0.035
    scale = mb.leg_length / g1_leg
    print(f"{len(raw)} frames ({len(raw) / FPS:.1f} s), leg length G1 {g1_leg:.3f} m, Xiaoxiaobai {mb.leg_length:.3f} m, scale {scale:.3f}")

    if args.smooth > 0:
        g1_q[:, :3] = gaussian_filter1d(g1_q[:, :3], args.smooth, axis=0, mode="nearest")
        g1_q[:, 7:] = gaussian_filter1d(g1_q[:, 7:], args.smooth, axis=0, mode="nearest")
    g1_down = g1.contacts(g1_q)
    quat = g1_q[:, 3:7]

    # Soles that stay down hold the spot they set down on; the root goes where the legs put it,
    # pulled weakly along the scaled pelvis path.
    mb.fk(np.zeros(len(JOINTS)))
    x = np.r_[np.zeros(len(JOINTS)), 0.0, 0.0, -mb.sole_z().min()]
    joint_pos = np.empty((len(raw), len(JOINTS)))
    root_pos = np.empty((len(raw), 3))
    pins: dict = {}
    for t, qpos in enumerate(g1_q):
        delta = g1.deltas(qpos)
        dirs = {k: delta[k] @ v for k, v in mb.rest_dirs.items()}
        dirs.update(g1.arm_dirs())
        trunk = Rotation.from_quat(quat[t], scalar_first=True).as_matrix()
        dirs = {k: trunk @ v for k, v in dirs.items()}
        down = {s for i, s in enumerate(("left", "right")) if g1_down[t, i]}
        pins = {s: p for s, p in pins.items() if s in down}
        guess = x[-3:].copy()
        if t:
            guess[:2] += scale * (g1_q[t, :2] - g1_q[t - 1, :2])
        targets = {"dirs": dirs, "down": down, "pins": pins, "q_prev": x[:-3].copy(), "root_guess": guess,
                   "foot": {s: trunk @ delta[(s, "foot")] @ mb.rest_foot[s] for s in ("left", "right")}}
        x = mb.solve(np.r_[x[:-3], guess], quat[t], targets)
        mb.fk(x[:-3], root=np.r_[x[-3:], quat[t]])
        xy = mb.sole_xy()
        for i, s in enumerate(("left", "right")):
            if s in down and s not in pins:
                pins[s] = xy[i].copy()
        joint_pos[t], root_pos[t] = x[:-3], x[-3:]

    # The arm solution flips branch where an arm crosses its stops or points along the shoulder
    # pitch axis; cap the arms at what the servos can follow.
    arms = [JOINTS.index(j) for j in ARMS]
    joint_pos[:, arms] = rate_limit(joint_pos[:, arms], ARM_VMAX / FPS)

    def soles(t):
        mb.fk(joint_pos[t], root=np.r_[root_pos[t], quat[t]])
        return mb.sole_xy(), mb.sole_z() < STANCE_Z

    vel = np.gradient(joint_pos, 1 / FPS, axis=0)
    at_limit = (np.isclose(joint_pos, mb.lo, atol=1e-3) | np.isclose(joint_pos, mb.hi, atol=1e-3)).mean(0)
    print("joint          range used [deg]        |vel| p99, max [rad/s]  time at limit")
    for j, name in enumerate(JOINTS):
        print(f"  {name:20s} {np.degrees(joint_pos[:, j].min()):7.1f} .. {np.degrees(joint_pos[:, j].max()):6.1f}"
              f"   {np.percentile(np.abs(vel[:, j]), 99):5.1f} {np.abs(vel[:, j]).max():5.1f}   {at_limit[j] * 100:4.1f}%")
    xy, on_floor = map(np.array, zip(*(soles(t) for t in range(len(raw)))))
    slide = np.linalg.norm(np.gradient(xy, 1 / FPS, axis=0), axis=2)[g1_down]
    print(f"soles down: left {g1_down[:, 0].mean():.0%} right {g1_down[:, 1].mean():.0%} both {g1_down.all(1).mean():.0%}"
          f" (on the floor within {STANCE_Z * 1000:.0f} mm: {on_floor[g1_down].mean():.0%});"
          f" sliding while down: median {np.median(slide) * 1000:.0f} mm/s, p90 {np.percentile(slide, 90) * 1000:.0f} mm/s")
    print(f"root: travels {np.linalg.norm(np.diff(root_pos[:, :2], axis=0), axis=1).sum():.2f} m, "
          f"spread {np.linalg.norm(root_pos[:, :2] - root_pos[0, :2], axis=1).max():.2f} m, "
          f"height {root_pos[:, 2].min():.3f}..{root_pos[:, 2].max():.3f} m")
    np.savez(args.out, fps=FPS, joint_names=np.array(JOINTS), joint_pos=joint_pos,
             root_pos=root_pos, root_quat=quat, scale=scale)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
