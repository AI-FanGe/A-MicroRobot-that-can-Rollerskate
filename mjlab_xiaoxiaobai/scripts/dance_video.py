"""Video of a dance checkpoint over the whole motion from the ready pose, with the reference as a
ghost and the tracking error in the caption.

usage: dance_video.py <ckpt> <out.mp4>
env:   TASK (Mjlab-Dance-Xiaoxiaobai)  GHOST=0 hide the reference  AZIMUTH (150)  DISTANCE (0.95)
"""
import os
import sys
from dataclasses import asdict

import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import mjlab_xiaoxiaobai.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils import spec_config as spec_cfg
from skate_demo_video import FONT, paint_robot

TASK = os.environ.get("TASK", "Mjlab-Dance-Xiaoxiaobai")


def main():
    ckpt, out = sys.argv[1], sys.argv[2]
    cfg = load_env_cfg(TASK, play=True)
    cfg.scene.num_envs = 1
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.textures = (
        spec_cfg.TextureCfg(name="groundplane", type="2d", builtin="checker", mark="edge",
                            rgb1=(0.93, 0.90, 0.84), rgb2=(0.80, 0.75, 0.66), markrgb=(0.45, 0.40, 0.35),
                            width=512, height=512),
    )
    cfg.scene.terrain.materials = (
        spec_cfg.MaterialCfg(name="groundplane", texuniform=True, texrepeat=(8.0, 8.0), reflectance=0.06,
                             texture="groundplane", geom_names_expr=("terrain$",)),
    )
    cfg.commands["motion"].debug_vis = os.environ.get("GHOST", "1") == "1"
    cfg.viewer.azimuth = float(os.environ.get("AZIMUTH", 150.0))
    cfg.viewer.elevation = -14.0
    cfg.viewer.distance = float(os.environ.get("DISTANCE", 0.95))
    cfg.viewer.width, cfg.viewer.height = 1280, 720
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode="rgb_array"))
    u = env.unwrapped
    paint_robot(u.sim.mj_model)
    runner = load_runner_cls(TASK)(env, asdict(load_rl_cfg(TASK)), device="cuda:0")
    runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
    policy = runner.get_inference_policy("cuda:0")

    cmd = u.command_manager.get_term("motion")
    T = cmd.motion.time_step_total - 1
    fps = 1.0 / u.step_dt
    font, small = ImageFont.truetype(FONT, 30), ImageFont.truetype(FONT, 22)
    motion = os.path.splitext(os.path.basename(cmd.cfg.motion_file))[0]
    ref = np.load(cmd.cfg.motion_file)
    sections = (ref["section_starts"], [str(n) for n in ref["section_names"]]) if "section_starts" in ref else None
    title = f"Xiaoxiaobai {'广播体操' if 'Gymnastics' in TASK else '跳舞'}  {motion}   {os.path.basename(os.path.dirname(ckpt))}/{os.path.basename(ckpt)}"
    obs, _ = env.reset()
    falls = 0
    with imageio.get_writer(out, fps=fps, codec="libx264", quality=8, macro_block_size=1) as writer:
        for t in range(T):
            with torch.no_grad():
                obs, _, dones, _ = env.step(policy(obs))
            falls += int(dones[0])
            frame = u.render()
            if frame.ndim == 4:
                frame = frame[0]
            err = float(torch.norm(cmd.body_pos_relative_w - cmd.robot_body_pos_w, dim=-1).mean()) * 1000
            s = t / fps
            if sections:
                phase = sections[1][max(int(np.searchsorted(sections[0], s, side="right")) - 1, 0)]
            else:
                phase = "准备姿态 → 起舞" if s < 1.5 else ("收舞 → 准备姿态" if s > T / fps - 1.5 else "跳舞")
            image = Image.fromarray(frame)
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, image.width, 84), fill=(0, 0, 0))
            draw.text((16, 6), title, font=font, fill=(255, 255, 255))
            draw.text((16, 48), f"{s:5.1f} s / {T / fps:.0f} s   {phase}   身体偏离参考 {err:4.1f} mm"
                      + ("   半透明 = 参考动作" if cfg.commands["motion"].debug_vis else "")
                      + (f"   中断 {falls} 次" if falls else ""), font=small, fill=(210, 210, 210))
            writer.append_data(np.asarray(image))
    print(out, f"{T / fps:.1f} s, interruptions {falls}")
    env.close()


if __name__ == "__main__":
    main()
