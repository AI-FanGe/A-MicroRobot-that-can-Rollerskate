#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

"""Where does a real walk differ from sim? Compares a recording with a nominal sim replay
of the same commands.

    PYTHONPATH=src uv run --group sim src/debug/sim2real_diag.py [rec.npz]

1. Sensor noise: trunk rates (gyro, and from the orientation as an aliasing check) and joint
   velocities per frequency band, real vs sim.
2. Policy input attribution: the policy is re-run offline on the recorded observations
   (checked against the recorded targets); one input at a time is low-pass filtered, and
   the drop in action jitter shows how much of it that input's high-frequency content causes.
3. Each leg against sim separately (not left vs right), per joint.
4. Command smoothness: number of command changes and the shortest constant segment.
5. Standing: trunk tilt from the IMU vs from each leg's encoders (joint zero check).
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).parent))
from analyze_walk_rec import LEG, load, quat_to_mat  # noqa: E402
from sim_replay import run_replay  # noqa: E402

DT = 0.02
BANDS = [(0.0, 3.0), (3.0, 8.0), (8.0, 25.0)]


def moving_mask(c: dict) -> np.ndarray:
    ran = c["policy_ran"] > 0.5
    ran[:25] = False
    return ran & (np.abs(c["cmd"]).sum(1) > 1e-3)


def still_mask(c: dict, settle_ticks: int = 50) -> np.ndarray:
    """Zero command, at least `settle_ticks` after the last command change."""
    ran = c["policy_ran"] > 0.5
    ran[:25] = False
    change = np.r_[True, np.abs(np.diff(c["cmd"], axis=0)).sum(1) > 1e-6]
    since = np.zeros(len(change), int)
    for k in range(1, len(change)):
        since[k] = 0 if change[k] else since[k - 1] + 1
    return ran & (since > settle_ticks) & (np.abs(c["cmd"]).sum(1) < 1e-3)


def band_rms(x: np.ndarray, mask: np.ndarray) -> list[float]:
    """RMS of x (1-D) in each frequency band, over the longest masked run segments."""
    out = np.zeros(len(BANDS))
    total = 0
    for s, e in runs(mask, 64):
        seg = x[s:e] - x[s:e].mean()
        spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
        f = np.fft.rfftfreq(len(seg), DT)
        p = spec / spec.sum() * seg.var() * len(seg) if spec.sum() > 0 else spec
        for b, (lo, hi) in enumerate(BANDS):
            out[b] += p[(f >= lo) & (f < hi)].sum()
        total += len(seg)
    return list(np.sqrt(out / max(total, 1)))


def runs(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    out, start = [], None
    for k, m in enumerate(list(mask) + [False]):
        if m and start is None:
            start = k
        elif not m and start is not None:
            if k - start >= min_len:
                out.append((start, k))
            start = None
    return out


def smooth(x: np.ndarray, n: int = 5) -> np.ndarray:
    k = np.ones(n) / n
    pad = np.pad(x, ((n // 2, n // 2), (0, 0)), mode="edge")
    return np.stack([np.convolve(pad[:, i], k, mode="valid") for i in range(x.shape[1])], axis=1)


class Policy:
    def __init__(self, path: str) -> None:
        self.sess = ort.InferenceSession(path)
        self.inp = self.sess.get_inputs()[0].name

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        rows = obs.astype(np.float32)
        return np.vstack([self.sess.run(None, {self.inp: r[None]})[0] for r in rows])


def policy_obs(c: dict, meta: dict) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Observation rows rebuilt from a recording, the recorded actions, and the joint indices."""
    jx = {n: i for i, n in enumerate(meta["joints"])}
    pj = [jx[n] for n in meta["policy_joints"]]
    default = np.array([meta["default_pose"][n] for n in meta["policy_joints"]])
    action = c["target"][:, pj] - default
    last = np.vstack([np.zeros(len(pj)), action[:-1]])
    obs = np.hstack([c["gyro"], c["projected_gravity"], c["q"][:, pj] - default, c["dq"][:, pj], last, c["cmd"]])
    return obs, action, pj


def attribution(c: dict, meta: dict, policy: Policy, label: str) -> None:
    obs, action, _ = policy_obs(c, meta)
    m = moving_mask(c)
    m[0] = False
    ok = m & np.roll(m, 1)
    pred = policy(obs)
    err = np.degrees(np.abs(pred[ok] - action[ok])).max()
    n = len(meta["policy_joints"])

    def jitter(a: np.ndarray) -> float:
        return float(np.degrees(np.abs(np.diff(a, axis=0))[ok[1:]]).mean())

    groups = {"gyro": slice(0, 3), "proj_gravity": slice(3, 6), "joint_pos": slice(6, 6 + n),
              "joint_vel": slice(6 + n, 6 + 2 * n), "last_action": slice(6 + 2 * n, 6 + 3 * n)}
    base = jitter(pred)
    print(f"\n[{label}] offline policy vs recorded target: max |diff| {err:.3f} deg (moving ticks)")
    print(f"[{label}] action jitter mean|d action| = {base:.2f} deg/tick; with one input smoothed (5-tick mean):")
    for name, sl in groups.items():
        o = obs.copy()
        o[:, sl] = smooth(obs[:, sl])
        print(f"    {name:13s} -> {jitter(policy(o)):5.2f}  ({(jitter(policy(o)) / base - 1) * 100:+5.0f}%)")


def body_rates(c: dict, meta: dict) -> np.ndarray:
    """Gyro in the trunk frame (roll, pitch, yaw rates). The policy itself sees the IMU frame."""
    return c["gyro"] @ quat_to_mat(meta["imu_mount_quat"]).T


def quat_rates(c: dict, meta: dict) -> np.ndarray:
    """Trunk angular velocity from consecutive IMU orientations. The orientation is integrated at
    the IMU rate, so unlike the 50 Hz gyro samples it cannot alias vibration into this band."""
    mount = quat_to_mat(meta["imu_mount_quat"])
    rot = np.array([quat_to_mat(q) @ mount.T for q in c["quat"]])
    w = np.zeros((len(rot), 3))
    for k in range(1, len(rot)):
        d = rot[k - 1].T @ rot[k]
        ang = np.arccos(np.clip((np.trace(d) - 1) / 2, -1, 1))
        if ang > 1e-9:
            axis = np.array([d[2, 1] - d[1, 2], d[0, 2] - d[2, 0], d[1, 0] - d[0, 1]]) / (2 * np.sin(ang))
            w[k] = axis * ang / DT
    return w


def noise_table(real: dict, sim: dict, meta: dict, sim_meta: dict) -> None:
    jx = {n: i for i, n in enumerate(meta["joints"])}
    print("\nSignal RMS per band [0-3 | 3-8 | 8-25 Hz], moving ticks; real / sim")
    rows = [(f"trunk {a} rate", lambda c, m, i=i: body_rates(c, m)[:, i]) for i, a in enumerate(("roll", "pitch", "yaw"))]
    rows += [(f"trunk {a} rate (quat)", lambda c, m, i=i: quat_rates(c, m)[:, i]) for i, a in enumerate(("roll", "pitch", "yaw"))]
    rows += [(f"dq {s}_{j}", lambda c, m, k=jx[f"{s}_{j}"]: c["dq"][:, k]) for j in ("hip_roll", "knee", "ankle_roll") for s in ("left", "right")]
    for name, f in rows:
        r, s = band_rms(f(real, meta), moving_mask(real)), band_rms(f(sim, sim_meta), moving_mask(sim))
        print(f"  {name:24s}" + " | ".join(f"{a:7.3f} / {b:7.3f} (x{a / max(b, 1e-9):4.1f})" for a, b in zip(r, s)))
    for label, c, m in (("real", real, meta), ("sim", sim, sim_meta)):
        st = still_mask(c)
        if st.sum() > 10:
            print(f"  settled standing trunk-rate std {label}: " + " ".join(f"{v:.4f}" for v in body_rates(c, m)[st].std(0))
                  + " rad/s   (training gyro noise U(-0.03,0.03): std 0.0173)")


def leg_table(real: dict, sim: dict, meta: dict) -> None:
    jx = {n: i for i, n in enumerate(meta["joints"])}
    print("\nEach leg vs sim (moving ticks). amp = std(q), lead = rms(target-q), jit = mean|d target|; deg")
    print(f"  {'joint':12s} | {'amp L':>6s} {'R':>6s} {'sim':>6s} | {'lead L':>6s} {'R':>6s} {'sim':>6s} | {'jit L':>6s} {'R':>6s} {'sim':>6s}")
    mr, ms = moving_mask(real), moving_mask(sim)
    for j in LEG:
        vals = []
        for f in (lambda c, k, m: c["q"][m, k].std(),
                  lambda c, k, m: np.sqrt(((c["target"][m, k] - c["q"][m, k]) ** 2).mean()),
                  lambda c, k, m: np.abs(np.diff(c["target"][:, k]))[m[1:]].mean()):
            l, r = f(real, jx[f"left_{j}"], mr), f(real, jx[f"right_{j}"], mr)
            s = (f(sim, jx[f"left_{j}"], ms) + f(sim, jx[f"right_{j}"], ms)) / 2
            vals += [l, r, s]
        v = np.degrees(vals)
        print(f"  {j:12s} | {v[0]:6.1f} {v[1]:6.1f} {v[2]:6.1f} | {v[3]:6.1f} {v[4]:6.1f} {v[5]:6.1f} | {v[6]:6.2f} {v[7]:6.2f} {v[8]:6.2f}")


def tilt(g: np.ndarray) -> np.ndarray:
    """(roll, pitch) in degrees from gravity in the trunk frame; roll > 0 leans right."""
    return np.degrees(np.stack([np.arcsin(np.clip(-g[:, 1], -1, 1)), np.arcsin(np.clip(g[:, 0], -1, 1))], axis=1))


def imu_gravity(c: dict, meta: dict) -> np.ndarray:
    mount = quat_to_mat(meta["imu_mount_quat"])
    return np.array([(quat_to_mat(q) @ mount.T).T @ [0.0, 0.0, -1.0] for q in c["quat"]])


def encoder_gravity(q: np.ndarray, joints: list[str]) -> np.ndarray:
    """Gravity in the trunk frame implied by each sole lying flat: shape (N, 2 feet, 3)."""
    model = mujoco.MjModel.from_xml_path("src/model/mjcf/scene.xml")
    data = mujoco.MjData(model)
    adr = {k: model.jnt_qposadr[j] for k, name in enumerate(joints)
           if (j := mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)) >= 0}
    geoms = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{s}_foot_collision") for s in ("left", "right")]
    out = np.zeros((len(q), 2, 3))
    for i, row in enumerate(q):
        data.qpos[:] = 0.0
        data.qpos[3] = 1.0
        for k, a in adr.items():
            data.qpos[a] = row[k]
        mujoco.mj_kinematics(model, data)
        for f, g in enumerate(geoms):
            out[i, f] = -data.geom_xmat[g].reshape(3, 3)[:, 2]
    return out


def stance_table(real: dict, sim: dict, meta: dict, sim_meta: dict) -> None:
    """While standing both soles are flat, so the encoders alone fix the trunk tilt. A foot
    whose encoder tilt disagrees with the IMU has a joint zero (or horn) error in its leg."""
    print("\nStanding: trunk roll/pitch from the IMU vs from each leg's encoders (sole flat), deg")
    for label, c, m in (("real", real, meta), ("sim", sim, sim_meta)):
        st = still_mask(c)
        if st.sum() < 50:
            print(f"  {label}: too few settled standing ticks")
            continue
        imu = tilt(imu_gravity(c, m)[st]).mean(0)
        enc = encoder_gravity(c["q"][st], m["joints"])
        el, er = tilt(enc[:, 0]).mean(0), tilt(enc[:, 1]).mean(0)
        print(f"  {label:4s} IMU roll {imu[0]:+5.2f} pitch {imu[1]:+5.2f} | left leg roll {el[0]:+5.2f} pitch {el[1]:+5.2f} "
              f"| right leg roll {er[0]:+5.2f} pitch {er[1]:+5.2f}")


def command_table(c: dict) -> None:
    ran = c["policy_ran"] > 0.5
    cmd = c["cmd"][ran]
    change = np.nonzero(np.abs(np.diff(cmd, axis=0)).sum(1) > 1e-6)[0]
    seg = np.diff(np.concatenate([[0], change + 1, [len(cmd)]]))
    print(f"\nCommands: {len(change)} changes, shortest constant segment {seg.min() * DT:.2f} s, "
          f"segments < 0.3 s: {(seg < 15).sum()}")


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(sorted(glob.glob("logs/walk_rec/walk_*.npz"))[-1])
    real, meta = load(path)
    sim, sim_meta = run_replay(real["cmd"])
    print(f"recording {path.name}: {len(real['t'])} ticks")
    policy = Policy(meta.get("agent_path") or "src/agents/walk_low_battery_1.onnx")
    command_table(real)
    noise_table(real, sim, meta, sim_meta)
    attribution(real, meta, policy, "real")
    attribution(sim, sim_meta, policy, "sim")
    leg_table(real, sim, meta)
    stance_table(real, sim, meta, sim_meta)


if __name__ == "__main__":
    main()
