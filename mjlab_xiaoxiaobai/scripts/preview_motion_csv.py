"""Play retargeted G1 motion CSVs (LAFAN1 layout: root xyz, root quat xyzw, 29 joints in g1.xml
order, 30 fps) on the G1 model and write one preview video per clip, with a time stamp to pick
segments by, plus a line of stats: how far the dance travels, how low it crouches, how long both
feet are off the floor and how fast it turns.

usage: preview_motion_csv.py <out_dir> <csv> [<csv> ...]
env:   FPS (30)  SPEED (1.0)  WIDTH (854)  HEIGHT (480)
"""
import os
import sys
from pathlib import Path

import imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FPS = int(os.environ.get("FPS", 30))
SPEED = float(os.environ.get("SPEED", 1.0))
W, H = int(os.environ.get("WIDTH", 854)), int(os.environ.get("HEIGHT", 480))
# Both ankles this far above their standing height: the dancer is in the air.
AIR_CLEARANCE = 0.04


def build_model() -> mujoco.MjModel:
    spec = get_spec()
    spec.worldbody.add_light(pos=(0, 0, 4), dir=(0, 0, -1), type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                             diffuse=(0.8, 0.8, 0.8))
    tex = spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                           rgb1=(0.32, 0.36, 0.42), rgb2=(0.26, 0.29, 0.34), width=300, height=300)
    mat = spec.add_material(name="grid", texrepeat=(10, 10), reflectance=0.1)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=(20, 20, 0.1), material="grid")
    model = spec.compile()
    model.vis.global_.offwidth, model.vis.global_.offheight = W, H
    model.vis.headlight.ambient[:] = (0.35, 0.35, 0.35)
    model.vis.headlight.diffuse[:] = (0.6, 0.6, 0.6)
    return model


def stats(model: mujoco.MjModel, data: mujoco.MjData, qpos: np.ndarray) -> tuple[str, np.ndarray]:
    feet = [model.body(n).id for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
    ankle_z = np.empty((len(qpos), 2))
    for i, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        ankle_z[i] = data.xpos[feet, 2]
    stand_z = np.percentile(ankle_z, 5)
    air = (ankle_z.min(1) > stand_z + AIR_CLEARANCE)
    xy = qpos[:, :2]
    path = np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()
    spread = np.linalg.norm(xy - xy[0], axis=1).max()
    w, x, y, z = qpos[:, 3:7].T
    yaw = np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    yaw_rate = np.abs(np.gradient(yaw) * 30.0)
    pelvis = qpos[:, 2]
    text = (f"{len(qpos) / 30:.0f} s | 走动路程 {path:.1f} m, 离起点最远 {spread:.1f} m | 骨盆最低 {pelvis.min():.2f} m"
            f" (站立 {np.median(pelvis):.2f}) | 双脚离地 {air.mean() * 100:.1f}% 的时间 | 转身最快 {np.degrees(yaw_rate.max()):.0f} deg/s")
    return text, air


def render(csv: Path, out_dir: Path, model: mujoco.MjModel) -> str:
    raw = np.loadtxt(csv, delimiter=",")
    qpos = np.concatenate([raw[:, :3], raw[:, [6, 3, 4, 5]], raw[:, 7:]], axis=1)
    data = mujoco.MjData(model)
    text, air = stats(model, data, qpos)
    renderer = mujoco.Renderer(model, H, W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.elevation, cam.azimuth = 3.2, -12.0, 135.0
    font, small = ImageFont.truetype(FONT, 22), ImageFont.truetype(FONT, 15)
    lookat = qpos[0, :3].copy()
    step = 30.0 * SPEED / FPS
    path = out_dir / f"{csv.stem}.mp4"
    with imageio.get_writer(str(path), fps=FPS, codec="libx264", quality=7, macro_block_size=1) as writer:
        for f in np.arange(0, len(qpos), step):
            i = int(f)
            data.qpos[:] = qpos[i]
            mujoco.mj_kinematics(model, data)
            lookat += 0.08 * (np.array([qpos[i, 0], qpos[i, 1], 0.75]) - lookat)
            cam.lookat[:] = lookat
            renderer.update_scene(data, cam)
            img = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(img)
            t = i / 30.0
            draw.text((12, 8), f"{csv.stem}   {int(t // 60)}:{t % 60:04.1f}", font=font, fill=(255, 255, 255))
            if air[i]:
                draw.text((12, 38), "双脚离地", font=font, fill=(255, 120, 90))
            for k, line in enumerate(reversed(text.split(" | "))):
                draw.text((12, H - 24 * (k + 1)), line, font=small, fill=(230, 230, 230))
            writer.append_data(np.asarray(img))
    return f"{csv.stem}: {text}"


def main() -> None:
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    model = build_model()
    for csv in sys.argv[2:]:
        print(render(Path(csv), out_dir, model), flush=True)


if __name__ == "__main__":
    main()
