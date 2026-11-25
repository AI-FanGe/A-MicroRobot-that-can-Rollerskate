"""Turn a retargeted Xiaoxiaobai motion (retarget_g1_to_xiaoxiaobai.py) into the motion file of the dance
task: turned to start facing +x at the origin, eased in from and out to the ready pose, resampled
to the 50 Hz policy rate, with every body's pose and velocity from Xiaoxiaobai's model.

With --no-ready the motion is kept as it is (it starts and ends lying, say), only moved to
start at the origin.

usage: dance_motion_npz.py <retarget.npz> <out.npz> [--blend 1.5] [--no-ready]
"""
import argparse

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

from mjlab.scene import Scene, SceneCfg
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab_xiaoxiaobai.robot.xiaoxiaobai_constants import HOME_FRAME, XIAOXIAOBAI_DANCE_ROBOT_CFG

FPS_OUT = 50


def ready_height(joint_names: list[str]) -> float:
    """Trunk height that sets the ready pose's soles on the floor."""
    model = XIAOXIAOBAI_DANCE_ROBOT_CFG.spec_fn().compile()
    data = mujoco.MjData(model)
    data.qpos[3] = 1.0
    for j, q in HOME_FRAME.joint_pos.items():
        data.qpos[model.jnt_qposadr[model.joint(j).id]] = q
    mujoco.mj_kinematics(model, data)
    low = min(data.geom_xpos[g, 2] - np.abs(data.geom_xmat[g].reshape(3, 3)[2]) @ model.geom_size[g]
              for g in (model.geom(f"{s}_foot_collision").id for s in ("left", "right")))
    return -low


def with_ready_pose(q, pos, rot, names, fps, blend_s):
    """Turned to start facing +x, eased in from and out to the ready pose."""
    yaw0 = Rotation.from_euler("z", -rot[0].as_euler("ZYX")[0])
    rot = yaw0 * rot
    pos = yaw0.apply(pos)
    home_q = np.array([HOME_FRAME.joint_pos.get(n, 0.0) for n in names])
    home_z = ready_height(names)
    nb = int(round(blend_s * fps))
    s = np.linspace(0.0, 1.0, nb, endpoint=False)
    ease = s * s * (3 - 2 * s)

    def blend(q_a, pos_a, rot_a, q_b, pos_b, rot_b):
        rots = Slerp([0, 1], Rotation.concatenate([rot_a, rot_b]))(ease)
        return (q_a + ease[:, None] * (q_b - q_a), pos_a + ease[:, None] * (pos_b - pos_a), rots)

    end_yaw = Rotation.from_euler("z", rot[-1].as_euler("ZYX")[0])
    q_in, p_in, r_in = blend(home_q, np.r_[0, 0, home_z], Rotation.identity(), q[0], pos[0], rot[0])
    q_out, p_out, r_out = blend(q[-1], pos[-1], rot[-1], home_q, np.r_[pos[-1, :2], home_z], end_yaw)
    q = np.concatenate([q_in, q, q_out, home_q[None]])
    pos = np.concatenate([p_in, pos, p_out, np.r_[pos[-1, :2], home_z][None]])
    rot = Rotation.concatenate([r_in, rot, r_out, end_yaw])

    return q, pos, rot, nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--blend", type=float, default=1.5, help="seconds from/to the ready pose")
    ap.add_argument("--no-ready", action="store_true", help="no ready pose before and after, no turn")
    args = ap.parse_args()

    src = np.load(args.src)
    fps = float(src["fps"])
    names = [str(n) for n in src["joint_names"]]
    q, pos = src["joint_pos"].copy(), src["root_pos"].copy()
    rot = Rotation.from_quat(src["root_quat"], scalar_first=True)

    pos[:, :2] -= pos[0, :2]
    nb = 0
    if not args.no_ready:
        q, pos, rot, nb = with_ready_pose(q, pos, rot, names, fps, args.blend)

    # Resample to the policy rate.
    t_in = np.arange(len(q)) / fps
    t_out = np.arange(0.0, t_in[-1], 1.0 / FPS_OUT)
    q = np.stack([np.interp(t_out, t_in, q[:, j]) for j in range(q.shape[1])], axis=1)
    pos = np.stack([np.interp(t_out, t_in, pos[:, k]) for k in range(3)], axis=1)
    rot = Slerp(t_in, rot)(t_out)
    dt = 1.0 / FPS_OUT
    qd = np.gradient(q, dt, axis=0)
    lin = np.gradient(pos, dt, axis=0)
    rv = (rot[2:] * rot[:-2].inv()).as_rotvec() / (2 * dt)
    ang = np.concatenate([rv[:1], rv, rv[-1:]])

    # Every body's pose and velocity from the dance robot.
    sim_cfg = SimulationCfg()
    sim_cfg.mujoco.timestep = dt
    scene = Scene(SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"), num_envs=1,
                           entities={"robot": XIAOXIAOBAI_DANCE_ROBOT_CFG}), device="cpu")
    model = scene.compile()
    sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device="cpu")
    scene.initialize(sim.mj_model, sim.model, sim.data)
    robot = scene["robot"]
    ids = robot.find_joints(names, preserve_order=True)[0]
    log = {k: [] for k in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")}
    quat = rot.as_quat(scalar_first=True)
    f32 = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)[None]
    for t in range(len(q)):
        root = robot.data.default_root_state.clone()
        root[:, :3], root[:, 3:7], root[:, 7:10], root[:, 10:] = f32(pos[t]), f32(quat[t]), f32(lin[t]), f32(ang[t])
        robot.write_root_state_to_sim(root)
        jp, jv = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
        jp[:, ids], jv[:, ids] = f32(q[t]), f32(qd[t])
        robot.write_joint_state_to_sim(jp, jv)
        sim.forward()
        scene.update(dt)
        log["joint_pos"].append(robot.data.joint_pos[0].numpy().copy())
        log["joint_vel"].append(robot.data.joint_vel[0].numpy().copy())
        log["body_pos_w"].append(robot.data.body_link_pos_w[0].numpy().copy())
        log["body_quat_w"].append(robot.data.body_link_quat_w[0].numpy().copy())
        log["body_lin_vel_w"].append(robot.data.body_link_lin_vel_w[0].numpy().copy())
        log["body_ang_vel_w"].append(robot.data.body_link_ang_vel_w[0].numpy().copy())
    # Section labels of a choreographed routine (gymnastics_motion.py), on the output's clock.
    extra = {}
    if "section_starts" in src:
        extra = dict(section_starts=src["section_starts"] + nb / fps, section_names=src["section_names"])
    np.savez(args.out, fps=[FPS_OUT], **{k: np.stack(v) for k, v in log.items()}, **extra)
    ready = "no ready pose" if args.no_ready else f"{args.blend} s in/out of the ready pose"
    print(f"{len(q)} frames at {FPS_OUT} Hz ({len(q) / FPS_OUT:.1f} s, {ready}),"
          f" {len(robot.joint_names)} joints, {len(robot.body_names)} bodies -> {args.out}")
    print("bodies:", robot.body_names)


if __name__ == "__main__":
    main()
