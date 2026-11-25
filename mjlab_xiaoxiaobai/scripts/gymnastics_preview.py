"""Video of a gymnastics_motion.py routine played kinematically (no physics), front and side
views with the section names, and the frames where two of Xiaoxiaobai's links interpenetrate.

usage: gymnastics_preview.py <raw.npz> <out.mp4> [title] [azimuth,azimuth] [lookat height] [distance] [x,y,z]
The optional point fixes the first camera's lookat (default: following the trunk). env ELEV: camera
elevation [deg] (-12).
Boxes in the npz's prop_boxes (rows of centre, half sizes, rgba[, quat wxyz]) are drawn as scenery. A head_pitch
array (motor 20, + down; the MJCF has no such joint) is shown on the head's wobble pitch joint.
"""
import os
import sys

import imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mjlab_xiaoxiaobai.robot.xiaoxiaobai_constants import get_dance_spec
from preview_retarget import FONT, with_floor

W, H = 640, 540


def main():
    src, out = sys.argv[1], sys.argv[2]
    title = sys.argv[3] if len(sys.argv) > 3 else "Xiaoxiaobai 广播体操 参考动作"
    azimuths = [float(a) for a in sys.argv[4].split(",")] if len(sys.argv) > 4 else (180.0, 120.0)
    look_z = float(sys.argv[5]) if len(sys.argv) > 5 else 0.11
    distance = float(sys.argv[6]) if len(sys.argv) > 6 else 0.75
    fixed = [float(v) for v in sys.argv[7].split(",")] if len(sys.argv) > 7 else None
    d = np.load(src)
    names = [str(n) for n in d["joint_names"]]
    spec = get_dance_spec()
    for k, box in enumerate(d["prop_boxes"] if "prop_boxes" in d else []):
        spec.worldbody.add_geom(name=f"prop_{k}", type=mujoco.mjtGeom.mjGEOM_BOX, pos=box[:3], size=box[3:6],
                                rgba=box[6:10], quat=box[10:14] if len(box) >= 14 else (1, 0, 0, 0),
                                contype=0, conaffinity=0)
    model = with_floor(spec, 4.0)
    data = mujoco.MjData(model)
    qadr = [model.jnt_qposadr[model.joint(n).id] for n in names]
    head_pitch = d["head_pitch"] if "head_pitch" in d else None
    pitch_adr = model.jnt_qposadr[model.joint("head_wobble_pitch").id]
    robot_geoms = {g for g in range(model.ngeom) if (model.geom(g).name or "").endswith("_collision")}
    starts, labels = d["section_starts"], [str(s) for s in d["section_names"]]
    renderer = mujoco.Renderer(model, H, W)
    cams = []
    for azimuth in azimuths:
        cam = mujoco.MjvCamera()
        cam.azimuth, cam.elevation, cam.distance = azimuth, float(os.environ.get("ELEV", -12.0)), distance
        cams.append(cam)
    font, small = ImageFont.truetype(FONT, 28), ImageFont.truetype(FONT, 20)
    fps = float(d["fps"])
    hits = []
    with imageio.get_writer(out, fps=fps, codec="libx264", quality=8, macro_block_size=1) as writer:
        for i in range(len(d["joint_pos"])):
            data.qpos[:] = 0
            data.qpos[:3], data.qpos[3:7] = d["root_pos"][i], d["root_quat"][i]
            data.qpos[qadr] = d["joint_pos"][i]
            if head_pitch is not None:
                data.qpos[pitch_adr] = head_pitch[i]
            mujoco.mj_forward(model, data)
            for c in data.contact[: data.ncon]:
                if c.geom1 in robot_geoms and c.geom2 in robot_geoms and c.dist < -0.002:
                    hits.append((i / fps, model.geom(c.geom1).name, model.geom(c.geom2).name, c.dist))
            frames = []
            for k, cam in enumerate(cams):
                cam.lookat[:] = fixed if fixed and k == 0 else (data.qpos[0], data.qpos[1], look_z)
                renderer.update_scene(data, cam)
                frames.append(renderer.render())
            image = Image.fromarray(np.concatenate(frames, axis=1))
            s = i / fps
            k = max(np.searchsorted(starts, s, side="right") - 1, 0)
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, image.width, 46), fill=(0, 0, 0))
            draw.text((14, 6), f"{title}   {labels[k]}", font=font, fill=(255, 255, 255))
            draw.text((image.width - 200, 12), f"{s:5.1f} s / {len(d['joint_pos']) / fps:.0f} s", font=small,
                      fill=(210, 210, 210))
            writer.append_data(np.asarray(image))
    print(out)
    if hits:
        pairs = {}
        for t, a, b, dist in hits:
            p = pairs.setdefault((a, b), [t, t, dist])
            p[1], p[2] = t, min(p[2], dist)
        for (a, b), (t0, t1, dist) in pairs.items():
            print(f"  self-contact {a} / {b}: {t0:.1f}-{t1:.1f} s, deepest {dist * 1000:.1f} mm")
    else:
        print("  no self-contact deeper than 2 mm")


if __name__ == "__main__":
    main()
