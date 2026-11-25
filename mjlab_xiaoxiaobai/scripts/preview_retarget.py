"""Side-by-side video of a G1 motion segment (left) and its Xiaoxiaobai retarget (right).

usage: preview_retarget.py <g1.csv> <xiaoxiaobai.npz> <out.mp4> [--start 1:30] [--title 名字]
"""
import argparse

import imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec as g1_spec
from mjlab_xiaoxiaobai.robot.xiaoxiaobai_constants import XIAOXIAOBAI_XML

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
W, H = 640, 540


def with_floor(spec: mujoco.MjSpec, size: float) -> mujoco.MjModel:
    spec.worldbody.add_light(pos=(0, 0, 4), dir=(0, 0, -1), type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                             diffuse=(0.7, 0.7, 0.7))
    tex = spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                           rgb1=(0.32, 0.36, 0.42), rgb2=(0.26, 0.29, 0.34), width=300, height=300)
    mat = spec.add_material(name="grid", texrepeat=(size, size), reflectance=0.1)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=(20, 20, 0.1), material="grid")
    model = spec.compile()
    model.vis.global_.offwidth, model.vis.global_.offheight = W, H
    model.vis.headlight.ambient[:] = (0.35, 0.35, 0.35)
    model.vis.headlight.diffuse[:] = (0.6, 0.6, 0.6)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("npz")
    ap.add_argument("out")
    ap.add_argument("--start", default="0:00")
    ap.add_argument("--title", default="", help="shown under the Xiaoxiaobai label")
    args = ap.parse_args()
    m, _, s = args.start.rpartition(":")
    start = float(m or 0) * 60 + float(s)

    mb_npz = np.load(args.npz)
    fps = int(mb_npz["fps"])
    n = len(mb_npz["joint_pos"])
    raw = np.loadtxt(args.csv, delimiter=",")[int(start * fps): int(start * fps) + n]
    g1_q = np.concatenate([raw[:, :3], raw[:, [6, 3, 4, 5]], raw[:, 7:]], axis=1)

    g1_m = with_floor(g1_spec(), 10)
    mb_m = with_floor(mujoco.MjSpec.from_file(str(XIAOXIAOBAI_XML)), 40)
    g1_d, mb_d = mujoco.MjData(g1_m), mujoco.MjData(mb_m)
    adr = [mb_m.jnt_qposadr[mb_m.joint(str(j)).id] for j in mb_npz["joint_names"]]
    rg, rm = mujoco.Renderer(g1_m, H, W), mujoco.Renderer(mb_m, H, W)
    cams = []
    for dist, z in ((3.0, 0.75), (0.75, 0.17)):
        c = mujoco.MjvCamera()
        c.type = mujoco.mjtCamera.mjCAMERA_FREE
        c.distance, c.elevation, c.azimuth = dist, -10.0, 150.0
        c.lookat[2] = z
        cams.append(c)
    font = ImageFont.truetype(FONT, 22)
    g1_origin = g1_q[0, :2].copy()
    with imageio.get_writer(args.out, fps=fps, codec="libx264", quality=7, macro_block_size=1) as w:
        for t in range(n):
            g1_d.qpos[:] = g1_q[t]
            mujoco.mj_kinematics(g1_m, g1_d)
            mb_d.qpos[:] = 0
            mb_d.qpos[:3] = mb_npz["root_pos"][t]
            mb_d.qpos[3:7] = mb_npz["root_quat"][t]
            mb_d.qpos[adr] = mb_npz["joint_pos"][t]
            mujoco.mj_kinematics(mb_m, mb_d)
            cams[0].lookat[:2] = g1_origin
            cams[1].lookat[:2] = 0.0
            rg.update_scene(g1_d, cams[0])
            rm.update_scene(mb_d, cams[1])
            img = Image.fromarray(np.concatenate([rg.render(), rm.render()], axis=1))
            d = ImageDraw.Draw(img)
            tt = start + t / fps
            d.text((12, 8), f"G1 原动作  {int(tt // 60)}:{tt % 60:04.1f}", font=font, fill=(255, 255, 255))
            d.text((W + 12, 8), "Xiaoxiaobai 映射结果", font=font, fill=(255, 255, 255))
            if args.title:
                d.text((W + 12, 38), args.title, font=font, fill=(255, 220, 120))
            w.append_data(np.asarray(img))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
