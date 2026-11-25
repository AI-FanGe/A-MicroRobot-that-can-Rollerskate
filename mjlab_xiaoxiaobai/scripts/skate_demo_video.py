"""Fixed-command demo clips of a skate checkpoint: forward, backward, turn left, turn right.

usage: skate_demo_video.py <ckpt> <out_dir>
The floor is flat tile (fixed grip and rolling resistance). Each clip holds one command.
"""
import os
import sys
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import mjlab_xiaoxiaobai.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils import spec_config as spec_cfg
from mjlab_xiaoxiaobai.robot.xiaoxiaobai_constants import SKATE_WHEEL_RADIUS
from mjlab_xiaoxiaobai.tasks.mdp import _twist_cfgs, set_command_velocity

TASK = os.environ.get("TASK", "Mjlab-Skate-Xiaoxiaobai")
DEVICE = os.environ.get("DEVICE", "cuda:0")
SECONDS = 8.0
# (name, filename, vx, wz, label). Backward is outside the trained command box.
CLIPS = (
    ("forward", "01_forward", 0.50, 0.0, "前进   指令 0.50 m/s"),
    ("backward", "02_backward", -0.25, 0.0, "后退   指令 -0.25 m/s"),
    ("turn_left", "03_turn_left", 0.30, 0.40, "左转   指令 0.30 m/s，+0.40 rad/s"),
    ("turn_right", "04_turn_right", 0.30, -0.40, "右转   指令 0.30 m/s，-0.40 rad/s"),
)
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def make_env(mu: float = 0.8, c_rr: float = 0.008, bearing: float = 1e-4, rgb1=(0.93, 0.90, 0.84),
             rgb2=(0.78, 0.72, 0.62), markrgb=(0.42, 0.38, 0.34), texrepeat=(5.0, 5.0)):
    cfg = load_env_cfg(TASK, play=True)
    cfg.scene.num_envs = 1
    cfg.episode_length_s = 1e3
    # CLEAN=1: no joint calibration error and a centred trunk mass, the policy's own gait.
    if os.environ.get("CLEAN") == "1":
        cfg.events.pop("encoder_bias", None)
        cfg.events.pop("base_com", None)
    # A checker plane shows the speed change; the generator's flat box is gray.
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.textures = (
        spec_cfg.TextureCfg(
            name="groundplane", type="2d", builtin="checker", mark="edge",
            rgb1=rgb1, rgb2=rgb2, markrgb=markrgb, width=512, height=512,
        ),
    )
    cfg.scene.terrain.materials = (
        spec_cfg.MaterialCfg(
            name="groundplane", texuniform=True, texrepeat=texrepeat, reflectance=0.08,
            texture="groundplane", geom_names_expr=("terrain$",),
        ),
    )
    rolling = c_rr * SKATE_WHEEL_RADIUS
    cfg.events["foot_friction"].params["ranges"] = (mu, mu)
    cfg.events["wheel_rolling_resistance"].params["ranges"] = (rolling, rolling)
    cfg.events["wheel_bearing_friction"].params["ranges"] = (bearing, bearing)
    cfg.events["wheel_bearing_damping"].params["ranges"] = (0.0, 0.0)
    cfg.commands["twist"].debug_vis = False
    cfg.viewer.origin_type = cfg.viewer.OriginType.ASSET_ROOT
    cfg.viewer.entity_name = "robot"
    cfg.viewer.azimuth = 135.0
    cfg.viewer.elevation = -16.0
    cfg.viewer.distance = 1.7
    cfg.viewer.width, cfg.viewer.height = 960, 540
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=DEVICE, render_mode="rgb_array"))
    paint_robot(env.unwrapped.sim.mj_model)
    return env


def paint_robot(model: mujoco.MjModel) -> None:
    """Shell, motors, deck and wheels get their own colors. The source meshes are one gray."""
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        if "visual" not in name:
            continue
        if "wheel" in name:
            color = (0.95, 0.78, 0.12, 1.0)
        elif "skate" in name:
            color = (0.10, 0.16, 0.32, 1.0)
        elif "motor" in name:
            color = (0.15, 0.16, 0.18, 1.0)
        else:
            color = (0.90, 0.91, 0.93, 1.0)
        model.geom_rgba[i] = color


def command_profile(t: float) -> tuple[float, str]:
    """Stand, accelerate twice, then drop the command to zero."""
    if t < 2.0:
        return 0.0, "静止"
    if t < 3.2:
        return 0.22 * (t - 2.0) / 1.2, "加速"
    if t < 5.2:
        return 0.22, "匀速"
    if t < 6.4:
        return 0.22 + 0.28 * (t - 5.2) / 1.2, "再加速"
    if t < 8.4:
        return 0.50, "最大"
    return 0.0, "急停"


def hold(env, vx, wz):
    for cfg in _twist_cfgs(env):
        cfg.rel_standing_envs = 0.0
    set_command_velocity(env, lin_vel_x=(vx, vx), lin_vel_y=(0.0, 0.0), ang_vel_z=(wz, wz))
    term = env.command_manager.get_term("twist")
    term.cfg.debug_vis = False
    term.is_standing_env[:] = False
    term.time_left[:] = 1.0e6


def stroke_rate(base) -> float:
    rate = base.__dict__.get("_stroke_rate")
    if rate is None:
        return 1.0
    return float(rate.reshape(-1)[0])


def caption(base, vx_now: float, wz_now: float) -> str:
    rate = stroke_rate(base)
    period = 1.2 / rate
    return (f"实际  vx {vx_now:+.2f} m/s    wz {wz_now:+.2f} rad/s"
            f"    步频 {rate:.2f}×（周期 {period:.2f} s）")


def label(frame, text, measured):
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(FONT, 28)
    small = ImageFont.truetype(FONT, 22)
    draw.rectangle((0, 0, image.width, 76), fill=(0, 0, 0))
    draw.text((16, 6), text, font=font, fill=(255, 255, 255))
    draw.text((16, 42), measured, font=small, fill=(210, 210, 210))
    return np.asarray(image)


def record_profile(env, base, policy, robot, out_dir: Path, fps: float, floor: str = "",
                   filename: str = "01_accel_stop.mp4") -> None:
    import imageio
    obs, _ = env.reset()
    hold(base, 0.0, 0.0)
    for _ in range(10):
        hold(base, 0.0, 0.0)
        with torch.no_grad():
            obs, _, _, _ = env.step(policy(obs))
    duration = 13.0
    steps = int(round(duration / base.step_dt))
    path = out_dir / filename
    by_phase: dict[str, list[float]] = {}
    with imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8, macro_block_size=1) as writer:
        for i in range(steps):
            t = i * base.step_dt
            vx, phase = command_profile(t)
            hold(base, vx, 0.0)
            with torch.no_grad():
                obs, _, _, _ = env.step(policy(obs))
            actual = float(robot.data.root_link_lin_vel_b[0, 0])
            by_phase.setdefault(phase, []).append(actual)
            frame = base.render()
            if frame is None:
                continue
            if frame.ndim == 4:
                frame = frame[0]
            writer.append_data(label(
                frame,
                f"{floor}  {phase}   指令 {vx:+.2f} m/s",
                caption(base, actual, float(robot.data.root_link_ang_vel_b[0, 2])),
            ))
    for phase, values in by_phase.items():
        print(f"{floor} {phase}: vx {np.mean(values):+.3f}")
    print(path)


def turn_profile(t: float) -> tuple[float, float, str]:
    """Stand, roll forward, turn left, then turn right. Forward speed stays 0.30 m/s."""
    if t < 1.5:
        return 0.0, 0.0, "静止"
    if t < 3.0:
        return 0.30 * (t - 1.5) / 1.5, 0.0, "起步"
    if t < 8.0:
        return 0.30, 0.40, "左转"
    return 0.30, -0.40, "右转"


def record_turns(env, base, policy, robot, out_dir: Path, fps: float, floor: str = "",
                 filename: str = "02_turns.mp4") -> None:
    import imageio
    obs, _ = env.reset()
    hold(base, 0.0, 0.0)
    for _ in range(10):
        hold(base, 0.0, 0.0)
        with torch.no_grad():
            obs, _, _, _ = env.step(policy(obs))
    duration = 13.0
    steps = int(round(duration / base.step_dt))
    path = out_dir / filename
    by_phase: dict[str, list[tuple[float, float]]] = {}
    with imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8, macro_block_size=1) as writer:
        for i in range(steps):
            t = i * base.step_dt
            vx, wz, phase = turn_profile(t)
            hold(base, vx, wz)
            with torch.no_grad():
                obs, _, _, _ = env.step(policy(obs))
            actual_vx = float(robot.data.root_link_lin_vel_b[0, 0])
            actual_wz = float(robot.data.root_link_ang_vel_b[0, 2])
            by_phase.setdefault(phase, []).append((actual_vx, actual_wz))
            frame = base.render()
            if frame is None:
                continue
            if frame.ndim == 4:
                frame = frame[0]
            writer.append_data(label(
                frame,
                f"{floor}  {phase}   指令 {vx:+.2f} m/s，{wz:+.2f} rad/s",
                caption(base, actual_vx, actual_wz),
            ))
    for phase, values in by_phase.items():
        vx_m = np.mean([v[0] for v in values])
        wz_m = np.mean([v[1] for v in values])
        print(f"{floor} {phase}: vx {vx_m:+.3f}  wz {wz_m:+.3f}")
    print(path)


FLOORS = (
    ("tile", "瓷砖", 0.60, 0.008, 1e-4,
     (0.93, 0.90, 0.84), (0.78, 0.72, 0.62), (0.42, 0.38, 0.34), (5.0, 5.0)),
    ("carpet", "地毯", 1.05, 0.08, 5e-4,
     (0.62, 0.36, 0.24), (0.40, 0.22, 0.14), (0.26, 0.14, 0.09), (14.0, 14.0)),
)


def record_forward(env, base, policy, robot, out_dir: Path, fps: float, floor: str) -> None:
    """Stand, ease up to 0.35 m/s, and hold it."""
    import imageio
    obs, _ = env.reset()
    hold(base, 0.0, 0.0)
    for _ in range(10):
        hold(base, 0.0, 0.0)
        with torch.no_grad():
            obs, _, _, _ = env.step(policy(obs))
    duration = 8.0
    steps = int(round(duration / base.step_dt))
    path = out_dir / "03_forward.mp4"
    measured = []
    with imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8, macro_block_size=1) as writer:
        for i in range(steps):
            t = i * base.step_dt
            vx = 0.0 if t < 1.5 else (0.35 if t >= 3.0 else 0.35 * (t - 1.5) / 1.5)
            phase = "静止" if t < 1.5 else ("起步" if t < 3.0 else "直行")
            hold(base, vx, 0.0)
            with torch.no_grad():
                obs, _, _, _ = env.step(policy(obs))
            actual = float(robot.data.root_link_lin_vel_b[0, 0])
            if phase == "直行":
                measured.append(actual)
            frame = base.render()
            if frame is None:
                continue
            if frame.ndim == 4:
                frame = frame[0]
            writer.append_data(label(
                frame, f"{floor}  {phase}   指令 {vx:+.2f} m/s",
                caption(base, actual, float(robot.data.root_link_ang_vel_b[0, 2])),
            ))
    print(f"{floor} 直行: vx {np.mean(measured):+.3f}")
    print(path)


SHOWCASE_SECONDS = 33.0


def showcase_profile(t: float) -> tuple[float, float, str]:
    """Stand; straight at three speeds with ramps up and down; a hard stop from 0.4 m/s;
    turns both ways; a hard stop out of the turn."""
    if t < 2.0:
        return 0.0, 0.0, "静止站立"
    if t < 3.5:
        return 0.30 * (t - 2.0) / 1.5, 0.0, "起步加速"
    if t < 6.5:
        return 0.30, 0.0, "直线 0.3"
    if t < 8.0:
        return 0.30 + 0.20 * (t - 6.5) / 1.5, 0.0, "加速"
    if t < 10.5:
        return 0.50, 0.0, "直线 0.5（快）"
    if t < 12.0:
        return 0.50 - 0.30 * (t - 10.5) / 1.5, 0.0, "减速"
    if t < 14.0:
        return 0.20, 0.0, "直线 0.2（慢）"
    if t < 15.0:
        return 0.20 + 0.20 * (t - 14.0), 0.0, "加速"
    if t < 17.0:
        return 0.40, 0.0, "直线 0.4"
    if t < 19.5:
        return 0.0, 0.0, "急停"
    if t < 21.0:
        return 0.30 * (t - 19.5) / 1.5, 0.0, "起步加速"
    if t < 25.5:
        return 0.30, 0.40, "左转"
    if t < 30.0:
        return 0.30, -0.40, "右转"
    return 0.0, 0.0, "急停"


def record_showcase(env, base, policy, robot, out_dir: Path, fps: float, floor: str, filename: str) -> Path:
    import imageio
    obs, _ = env.reset()
    hold(base, 0.0, 0.0)
    for _ in range(15):
        hold(base, 0.0, 0.0)
        with torch.no_grad():
            obs, _, _, _ = env.step(policy(obs))
    steps = int(round(SHOWCASE_SECONDS / base.step_dt))
    path = out_dir / filename
    by_phase: dict[str, list[tuple[float, float, float]]] = {}
    with imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8, macro_block_size=1) as writer:
        for i in range(steps):
            t = i * base.step_dt
            vx, wz, phase = showcase_profile(t)
            hold(base, vx, wz)
            with torch.no_grad():
                obs, _, _, _ = env.step(policy(obs))
            actual_vx = float(robot.data.root_link_lin_vel_b[0, 0])
            actual_wz = float(robot.data.root_link_ang_vel_b[0, 2])
            rate = stroke_rate(base)
            by_phase.setdefault(phase, []).append((actual_vx, actual_wz, rate))
            frame = base.render()
            if frame is None:
                continue
            if frame.ndim == 4:
                frame = frame[0]
            writer.append_data(label(
                frame,
                f"{floor}    {phase}",
                f"指令 {vx:+.2f} m/s  {wz:+.2f} rad/s     {caption(base, actual_vx, actual_wz)}",
            ))
    for phase, values in by_phase.items():
        print(f"{floor} {phase}: vx {np.mean([v[0] for v in values]):+.3f}"
              f"  wz {np.mean([v[1] for v in values]):+.3f}"
              f"  rate {np.mean([v[2] for v in values]):.2f}")
    print(path)
    return path


def record_stroke_demo(ckpt: str, out_dir: Path) -> None:
    import imageio
    import subprocess
    paths = []
    for name, label_cn, mu, c_rr, bearing, rgb1, rgb2, markrgb, repeat in FLOORS:
        env = make_env(mu=mu, c_rr=c_rr, bearing=bearing, rgb1=rgb1, rgb2=rgb2, markrgb=markrgb, texrepeat=repeat)
        base = env.unwrapped
        runner = load_runner_cls(TASK)(env, asdict(load_rl_cfg(TASK)), device=DEVICE)
        runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location=DEVICE)
        policy = runner.get_inference_policy(DEVICE)
        robot = base.scene["robot"]
        tag = f"{label_cn}  摩擦{mu:.2f}  滚阻{c_rr:.3f}  轴承{bearing:.0e} N·m"
        print(f"--- {tag}")
        floor = label_cn + ("（无标定误差）" if os.environ.get("CLEAN") == "1" else "（随机标定误差）")
        paths.append(record_showcase(env, base, policy, robot, out_dir, 1.0 / base.step_dt, floor, f"{name}.mp4"))
        env.close()
    list_file = out_dir / "concat.txt"
    list_file.write_text("".join(f"file '{p.resolve()}'\n" for p in paths))
    combined = out_dir / "stroke_tile_carpet_demo.mp4"
    ffmpeg = imageio.plugins.ffmpeg.get_exe()
    subprocess.run(
        [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(combined)],
        check=True, capture_output=True,
    )
    print("combined", combined)


def record_floors(ckpt: str, out_dir: Path) -> None:
    for name, label_cn, mu, c_rr, bearing, rgb1, rgb2, markrgb, repeat in FLOORS:
        floor_dir = out_dir / name
        floor_dir.mkdir(parents=True, exist_ok=True)
        env = make_env(mu=mu, c_rr=c_rr, bearing=bearing, rgb1=rgb1, rgb2=rgb2, markrgb=markrgb, texrepeat=repeat)
        base = env.unwrapped
        runner = load_runner_cls(TASK)(env, asdict(load_rl_cfg(TASK)), device=DEVICE)
        runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location=DEVICE)
        policy = runner.get_inference_policy(DEVICE)
        robot = base.scene["robot"]
        fps = 1.0 / base.step_dt
        tag = f"{label_cn}  μ{mu:.2f}  滚阻{c_rr:.3f}  轴承{bearing:.0e}"
        print(f"--- {tag}")
        record_profile(env, base, policy, robot, floor_dir, fps, floor=tag)
        record_turns(env, base, policy, robot, floor_dir, fps, floor=tag)
        record_forward(env, base, policy, robot, floor_dir, fps, floor=tag)
        env.close()


def main():
    ckpt, out_dir = sys.argv[1], Path(sys.argv[2])
    mode = sys.argv[3] if len(sys.argv) > 3 else ""
    out_dir.mkdir(parents=True, exist_ok=True)
    if mode == "floors":
        record_floors(ckpt, out_dir)
        return
    if mode == "showcase":
        record_stroke_demo(ckpt, out_dir)
        return

    env = make_env()
    base = env.unwrapped
    runner = load_runner_cls(TASK)(env, asdict(load_rl_cfg(TASK)), device=DEVICE)
    runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location=DEVICE)
    policy = runner.get_inference_policy(DEVICE)
    robot = base.scene["robot"]
    fps = 1.0 / base.step_dt
    if mode == "profile":
        record_profile(env, base, policy, robot, out_dir, fps)
        env.close()
        return
    if mode == "turns":
        record_turns(env, base, policy, robot, out_dir, fps)
        env.close()
        return

    steps = int(round(SECONDS / base.step_dt))

    import imageio
    paths = []
    for _name, filename, vx, wz, text in CLIPS:
        obs, _ = env.reset()
        hold(base, vx, wz)
        for _ in range(10):
            hold(base, vx, wz)
            with torch.no_grad():
                obs, _, _, _ = env.step(policy(obs))
        path = out_dir / f"{filename}.mp4"
        measured_vx, measured_wz = [], []
        with imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8, macro_block_size=1) as writer:
            for _ in range(steps):
                hold(base, vx, wz)
                with torch.no_grad():
                    obs, _, _, _ = env.step(policy(obs))
                vel = robot.data.root_link_lin_vel_b[0]
                wz_now = robot.data.root_link_ang_vel_b[0, 2]
                measured_vx.append(float(vel[0]))
                measured_wz.append(float(wz_now))
                frame = base.render()
                if frame is None:
                    continue
                if frame.ndim == 4:
                    frame = frame[0]
                caption = f"实际  vx {float(vel[0]):+.2f} m/s    wz {float(wz_now):+.2f} rad/s"
                writer.append_data(label(frame, text, caption))
        late = slice(steps // 3, None)
        print(f"{filename}: vx {np.mean(measured_vx[late]):+.3f} (cmd {vx:+.2f})"
              f"  wz {np.mean(measured_wz[late]):+.3f} (cmd {wz:+.2f})")
        paths.append(path)

    list_file = out_dir / "concat.txt"
    list_file.write_text("".join(f"file '{p.resolve()}'\n" for p in paths))
    combined = out_dir / "demo_forward_back_turns.mp4"
    ffmpeg = imageio.plugins.ffmpeg.get_exe()
    import subprocess
    subprocess.run(
        [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(combined)],
        check=True, capture_output=True,
    )
    print("combined", combined)
    env.close()


if __name__ == "__main__":
    main()
