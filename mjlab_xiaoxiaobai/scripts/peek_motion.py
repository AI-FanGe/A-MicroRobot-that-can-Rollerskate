"""Choreograph a sneaky peek (偷看) for Xiaoxiaobai that stands on its own feet.

The earlier version planted both hands on a monitor and leaned on it. This one keeps the same
beats — step out, crouch, peek, duck, creep out, startle, two quick peeks, stand — but the hands
stay in front of the chest and never touch the monitor. The trunk leans further and the head
reaches further past the screen's left edge. Balance is the gymnastics one: feet by IK, centre of
mass over the planted soles. The monitor is a prop box for the preview only; training never sees it.

usage: peek_motion.py <out.npz>
then:  gymnastics_preview.py <out.npz> <out.mp4> "Xiaoxiaobai 偷看 参考动作" 195,315 0.2 0.95 0,<edge>,0.2
       dance_motion_npz.py <out.npz> data/motions/xiaoxiaobai/peek_v2_50hz.npz --blend 0.3
"""
import argparse

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from gymnastics_motion import DOWN, FPS, JOINTS, NEUTRAL, SIDES, Choreo, arms, build, step
from mjlab_xiaoxiaobai.robot.xiaoxiaobai_constants import get_dance_spec

WIDE = 0.022  # left foot steps out this far [m]: a little wider than the braced demo
# Hands in front of the chest, close to the body. Not braced on anything.
TUCK = (46.0, 18.0, 68.0)
# The peek-side arm opens so the head can come past it.
OPEN = (16.0, 52.0, 26.0)
# Trunk roll while peeked [deg], up from the braced demo's 15-19. The legs cannot
# hold the feet down past this: the peek-side hip is at its roll stop.
LEAN = 26.0
MONITOR_TOP, MONITOR_W, DEPTH = 0.38, 0.36, 0.03  # [m]
HAND_HALF = 0.013
L_OUT = (0.0, WIDE, 0.0, 0.0)
FOREARM = {"right": "radius", "left": "radius_2"}


def choreograph() -> tuple[Choreo, list[tuple[float, str]]]:
    c = Choreo()
    sections = []

    def hidden(t, dz=-0.024, head=0.0, pitch=8.0):
        c.at(t, roll=0.0, shift=0.25, yaw=0.0, dz=dz, pitch=pitch, head=head, lf=L_OUT)
        arms(c, t, both=TUCK)

    def out(t, roll, shift, head, dz=-0.036, pitch=3.0):
        c.at(t, roll=roll, shift=shift, dz=dz, pitch=pitch, head=head, lf=L_OUT)
        arms(c, t, ra=TUCK, la=OPEN)

    c.at(0.0, **NEUTRAL)
    arms(c, 0.0, both=DOWN)
    sections.append((0.0, "站着"))

    # Tiptoe the left foot out and crouch behind where the screen will be. Hands come up
    # in front of the chest; they do not reach a support.
    sections.append((1.0, "躲到屏幕后"))
    c.at(1.0, shift=0.0, roll=0.0, dz=0.0, pitch=0.0, head=0.0)
    arms(c, 1.0, both=DOWN)
    c.at(1.4, shift=-0.7, roll=-2.0, dz=-0.026)
    step(c, 1.4, "left", dy=WIDE, lift=0.008, beats=0.7)
    c.at(2.1, shift=-0.7, roll=-2.0, dz=-0.028, lf=L_OUT)
    arms(c, 1.7, both=TUCK)
    c.at(2.4, head=-12.0)
    hidden(2.6)
    hidden(3.0, dz=-0.026)

    # Quick peek, a look round, duck back. Weight stays over the left (peek-side) sole.
    sections.append((3.0, "探头"))
    out(3.5, LEAN - 2.0, 0.9, -14.0)
    c.at(3.95, head=-40.0)
    out(4.55, LEAN, 0.93, -8.0)
    hidden(4.95, dz=-0.028, pitch=9.0)
    sections.append((4.95, "缩回"))
    c.at(5.15, dz=-0.024)
    c.at(5.4, head=24.0)
    c.at(5.7, head=-22.0)
    hidden(6.1)

    # Creep out further, stare, double take, startle back.
    sections.append((6.1, "慢慢探出"))
    out(7.7, LEAN, 0.95, -10.0, dz=-0.038, pitch=2.0)
    c.at(8.2, head=-42.0)
    c.at(8.7, head=-42.0)
    c.at(8.9, head=-16.0)
    out(9.15, LEAN, 0.95, -42.0, dz=-0.038, pitch=2.0)
    sections.append((9.15, "被发现"))
    hidden(9.45, dz=-0.030, pitch=10.0)
    hidden(9.95)

    # Two quick peeks, the second longer.
    sections.append((10.3, "连探两下"))
    hidden(10.3)
    out(10.6, LEAN - 4.0, 0.8, -18.0)
    out(10.9, LEAN - 4.0, 0.8, -18.0)
    hidden(11.15)
    out(11.45, LEAN, 0.9, -28.0)
    out(11.9, LEAN, 0.9, -20.0)
    hidden(12.2)

    # Bring the foot back and stand. Nothing to let go of.
    sections.append((12.5, "站好"))
    hidden(12.5)
    c.at(13.05, shift=-0.7, roll=-2.0, dz=-0.026, pitch=2.0, head=0.0)
    arms(c, 13.05, both=TUCK)
    step(c, 13.15, "left", dy=0.0, lift=0.008, beats=0.7, base=(0.0, WIDE))
    c.at(13.85, shift=-0.7, roll=-2.0, dz=-0.024, lf=(0.0, 0.0, 0.0, 0.0))
    arms(c, 13.6, both=DOWN)
    c.at(14.5, **NEUTRAL)
    arms(c, 14.5, both=DOWN)
    sections.append((14.5, "结束"))
    return c, sections


def _hand_local(model, body_name: str) -> np.ndarray:
    b = model.body(body_name).id
    g = next(g for g in range(model.ngeom) if model.geom_bodyid[g] == b and model.geom(g).name.endswith("_collision"))
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, model.geom_quat[g])
    k = int(np.argmax(model.geom_size[g]))
    ends = [model.geom_pos[g] + sign * R.reshape(3, 3)[:, k] * model.geom_size[g, k] for sign in (1, -1)]
    return max(ends, key=np.linalg.norm)


def _point_aabb(p: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    d = np.maximum(lo - p, p - hi)
    outside = np.maximum(d, 0.0)
    inside = np.max(d)
    return float(np.linalg.norm(outside) if inside > 0 else inside)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    args = ap.parse_args()

    t, joint_pos, root_pos, root_quat, sections, beat = build(1.0, choreograph)

    m = get_dance_spec().compile()
    d = mujoco.MjData(m)
    qadr = [m.jnt_qposadr[m.joint(j).id] for j in JOINTS]
    head = m.geom("head_collision").id
    hand_local = {s: _hand_local(m, FOREARM[s]) for s in SIDES}
    hand_body = {s: m.body(FOREARM[s]).id for s in SIDES}
    head_out = np.empty(len(t))
    head_x = np.empty(len(t))
    hands = {s: np.empty((len(t), 3)) for s in SIDES}
    roll = -Rotation.from_quat(root_quat, scalar_first=True).as_euler("ZYX")[:, 2]
    for i in range(len(t)):
        d.qpos[:3], d.qpos[3:7] = root_pos[i], root_quat[i]
        d.qpos[qadr] = joint_pos[i]
        mujoco.mj_kinematics(m, d)
        R = d.geom_xmat[head].reshape(3, 3)
        head_out[i] = d.geom_xpos[head, 1] + np.abs(R[1]) @ m.geom_size[head]
        head_x[i] = d.geom_xpos[head, 0] + np.abs(R[0]) @ m.geom_size[head]
        for s in SIDES:
            hands[s][i] = d.xpos[hand_body[s]] + d.xmat[hand_body[s]].reshape(3, 3) @ hand_local[s]

    # The screen's left edge sits just past the head while the robot is crouched, so a deeper
    # lean is what carries the head out. Its back face stays clear of the hands and the head:
    # nothing rests on it.
    hidden = (np.abs(roll) < np.radians(6.0)) & (t > 2.4) & (t < 12.4)
    edge = float(np.percentile(head_out[hidden], 90) + 0.008)
    back = float(max(head_x.max(), max(hands[s][:, 0].max() for s in SIDES)) + 0.02)
    yc, zc, hz = edge - MONITOR_W / 2, MONITOR_TOP / 2 + 0.03, MONITOR_TOP / 2 - 0.03
    lo = np.array([back, edge - MONITOR_W, zc - hz])
    hi = np.array([back + DEPTH, edge, zc + hz])
    gap = min(_point_aabb(hands[s][i], lo, hi) for s in SIDES for i in range(len(t)))
    past = head_out - edge
    print(f"trunk roll max {np.degrees(roll.max()):.1f} deg (demo was 19)")
    print(f"head past the screen edge: max {past.max() * 1000:.0f} mm, hidden (< 0) {np.mean(past < 0):.0%}")
    print(f"head travels {((head_out - np.median(head_out[hidden])).max()) * 1000:.0f} mm out from the crouch")
    print(f"nearest hand to the monitor {gap * 1000:.1f} mm (negative would be inside it)")

    boxes = [
        (back + DEPTH / 2, yc, zc, DEPTH / 2, MONITOR_W / 2, hz, 0.12, 0.12, 0.14, 1),
        (back + DEPTH + 0.001, yc, zc, 0.001, MONITOR_W / 2 - 0.012, hz - 0.012, 0.15, 0.32, 0.55, 1),
        (back + 0.05, edge - MONITOR_W / 2, 0.015, 0.012, 0.012, 0.015, 0.2, 0.2, 0.22, 1),
        (back + 0.05, edge - MONITOR_W / 2, 0.002, 0.06, 0.09, 0.002, 0.2, 0.2, 0.22, 1),
    ]
    np.savez(args.out, fps=FPS, joint_names=np.array(JOINTS), joint_pos=joint_pos, root_pos=root_pos,
             root_quat=root_quat, beat=beat, section_names=np.array([s for _, s in sections]),
             section_starts=np.array([b * beat for b, _ in sections]), prop_boxes=np.array(boxes, float))
    print(f"saved {args.out}  screen edge y={edge:.3f}")


if __name__ == "__main__":
    main()
