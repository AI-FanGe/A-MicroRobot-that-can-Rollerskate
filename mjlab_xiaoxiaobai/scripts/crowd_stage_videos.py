"""Bird's-eye clips of many robots training a gait in parallel.

One fixed 16:9 camera looks slightly down on a grid. Each robot is reset
facing a random direction and commanded to go forward in its own body frame,
so the crowd heads off in random directions.

    python scripts/crowd_stage_videos.py probe
    python scripts/crowd_stage_videos.py render
"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
import torch

import mjlab_xiaoxiaobai.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils import spec_config as spec_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab_xiaoxiaobai.robot.xiaoxiaobai_constants import SKATE_WHEEL_RADIUS
from skate_demo_video import paint_robot

LOGS = Path(__file__).resolve().parents[1] / "logs" / "rsl_rl"
OUT_DIR = Path(os.environ.get("OUT_DIR", str(Path(__file__).resolve().parents[1] / "videos" / "parallel_gait_4k")))
DEVICE = "cuda:0"

# 16:9 4K. A short preview pass can override these.
WIDTH = int(os.environ.get("WIDTH", "3840"))
HEIGHT = int(os.environ.get("HEIGHT", "2160"))
NUM_ENVS = int(os.environ.get("NUM_ENVS", "22"))
SPACING = float(os.environ.get("SPACING", "1.2"))
SECONDS = float(os.environ.get("SECONDS", "6"))
FPS = int(os.environ.get("FPS", "30"))
# Low and close: elevation 0 is level with the robots, -90 is straight down.
ELEVATION = float(os.environ.get("ELEVATION", "-22"))
AZIMUTH = float(os.environ.get("AZIMUTH", "48"))
DISTANCE = float(os.environ.get("DISTANCE", "4.35"))
FOVY = float(os.environ.get("FOVY", "36"))


def _patch_renderer() -> None:
    """The stock renderer caps the scene at 10k geoms, short of a full crowd."""

    def initialize(self) -> None:
        if self._renderer is not None:
            raise RuntimeError("Renderer is already initialized.")
        self._renderer = mujoco.Renderer(
            model=self._model,
            height=self._cfg.height,
            width=self._cfg.width,
            max_geom=120_000,
        )

    OffscreenRenderer.initialize = initialize


def _wrap_noise(height: int, width: int, cells: int, rng: np.random.Generator) -> np.ndarray:
    """Bilinear value noise that matches on opposite edges, so the floor does not show a seam."""
    grid = rng.standard_normal((cells, cells)).astype(np.float32)
    ys = np.linspace(0, cells, height, endpoint=False)
    xs = np.linspace(0, cells, width, endpoint=False)
    y0 = np.floor(ys).astype(np.int32)
    x0 = np.floor(xs).astype(np.int32)
    fy = (ys - y0)[:, None]
    fx = (xs - x0)[None, :]
    y1 = (y0 + 1) % cells
    x1 = (x0 + 1) % cells
    g00 = grid[np.ix_(y0, x0)]
    g01 = grid[np.ix_(y0, x1)]
    g10 = grid[np.ix_(y1, x0)]
    g11 = grid[np.ix_(y1, x1)]
    return (
        g00 * (1 - fx) * (1 - fy)
        + g01 * fx * (1 - fy)
        + g10 * (1 - fx) * fy
        + g11 * fx * fy
    )


def _concrete_texture(height: int, width: int) -> np.ndarray:
    """Warm, tileable concrete. Variation stays small so the repeats do not read as panels."""
    rng = np.random.default_rng(19)
    coarse = _wrap_noise(height, width, 6, rng)
    mid = _wrap_noise(height, width, 18, rng)
    fine = _wrap_noise(height, width, 64, rng)
    grit = rng.random((height, width), dtype=np.float32)
    # Specks are sparse; wrapping a few of them across the edge is invisible.
    specks = np.where(grit > 0.985, rng.uniform(0.04, 0.12, (height, width)), 0).astype(np.float32)
    pits = np.where(grit < 0.012, 0.06, 0).astype(np.float32)
    tone = np.clip(0.58 + 0.045 * coarse + 0.03 * mid + 0.018 * fine + specks - pits, 0, 1)
    rgb = np.stack([
        tone * 0.95 + 0.03,
        tone * 0.92 + 0.028,
        tone * 0.86 + 0.025,
    ], axis=-1)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def _scatter_origins(base) -> None:
    """Drop the grid. Robots sit in a loose cluster with uneven gaps."""
    n = base.num_envs
    rng = np.random.default_rng(11)
    pts: list[np.ndarray] = []
    for min_dist in (0.58, 0.50, 0.44):
        pts = []
        for _ in range(40000):
            if len(pts) >= n:
                break
            # Stay inside the directional shadow volume. Robots near the frame
            # edge are outside it, so they render flat and cast nothing.
            p = rng.uniform(-1.15, 1.15, size=2)
            if pts and np.linalg.norm(np.asarray(pts) - p, axis=1).min() < min_dist:
                continue
            pts.append(p)
        if len(pts) >= n:
            break
    if len(pts) < n:
        raise RuntimeError(f"only placed {len(pts)} of {n} robots")
    placed = np.asarray(pts, dtype=np.float32)
    placed -= placed.mean(axis=0)
    origins = base.scene.env_origins
    origins[:, 0] = torch.as_tensor(placed[:, 0], device=origins.device, dtype=origins.dtype)
    origins[:, 1] = torch.as_tensor(placed[:, 1], device=origins.device, dtype=origins.dtype)


def _finish_look(model: mujoco.MjModel) -> None:
    """Concrete floor, softer light, and a little material response on the shell."""
    tex_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXTURE, "groundplane")
    if tex_id >= 0:
        h, w = int(model.tex_height[tex_id]), int(model.tex_width[tex_id])
        nch = int(model.tex_nchannel[tex_id])
        image = _concrete_texture(h, w)
        if nch == 1:
            image = image.mean(axis=-1).astype(np.uint8)
        adr = int(model.tex_adr[tex_id])
        model.tex_data[adr:adr + h * w * nch] = image.reshape(-1)
    mat_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "groundplane")
    if mat_id >= 0:
        model.mat_rgba[mat_id] = (0.92, 0.90, 0.86, 1.0)
        model.mat_roughness[mat_id] = 0.92
        model.mat_specular[mat_id] = 0.08
        model.mat_shininess[mat_id] = 0.08
        model.mat_metallic[mat_id] = 0.0
        model.mat_reflectance[mat_id] = 0.02
        model.mat_texrepeat[mat_id] = (1.4, 1.4)
    paint_robot(model)
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        mid = int(model.geom_matid[i])
        if "visual" not in name or mid < 0:
            continue
        if "wheel" in name:
            rough, spec, metal = 0.32, 0.42, 0.15
        elif "motor" in name:
            rough, spec, metal = 0.22, 0.55, 0.7
        elif "skate" in name:
            rough, spec, metal = 0.4, 0.25, 0.05
        else:
            rough, spec, metal = 0.58, 0.16, 0.0
        model.mat_roughness[mid] = rough
        model.mat_specular[mid] = spec
        model.mat_metallic[mid] = metal
        model.mat_shininess[mid] = 0.25
    # Key light from the side and above, so a low camera still models the bodies.
    model.light_castshadow[0] = 1
    direction = np.array([-0.35, 0.55, -0.76], dtype=np.float64)
    model.light_dir[0] = direction / np.linalg.norm(direction)
    model.light_diffuse[0] = (0.82, 0.78, 0.72)
    model.light_ambient[0] = (0.32, 0.32, 0.34)
    model.light_specular[0] = (0.22, 0.22, 0.22)
    model.vis.headlight.ambient[:] = (0.18, 0.18, 0.19)
    model.vis.headlight.diffuse[:] = (0.42, 0.42, 0.44)
    model.vis.headlight.specular[:] = (0.12, 0.12, 0.12)
    model.vis.quality.shadowsize = 8192


def make_env(task: str, drop_stroke: bool = False, fallen: bool = False):
    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs = NUM_ENVS
    cfg.scene.env_spacing = SPACING
    cfg.episode_length_s = 1e3
    cfg.events.pop("encoder_bias", None)
    cfg.events.pop("base_com", None)
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.scene.terrain.textures = (
        spec_cfg.TextureCfg(
            name="groundplane",
            type="2d",
            builtin="flat",
            mark="none",
            rgb1=(0.62, 0.60, 0.56),
            rgb2=(0.62, 0.60, 0.56),
            width=1024,
            height=1024,
        ),
    )
    cfg.scene.terrain.materials = (
        spec_cfg.MaterialCfg(
            name="groundplane",
            texuniform=True,
            texrepeat=(1.4, 1.4),
            reflectance=0.02,
            texture="groundplane",
            geom_names_expr=("terrain$",),
        ),
    )
    cfg.events.pop("push_robot", None)
    cfg.events.pop("push_standing", None)
    if drop_stroke and "stroke_rate" in cfg.actions:
        # v12 added the cadence action. Earlier checkpoints are the 18 joint targets only.
        cfg.actions = dict(cfg.actions)
        del cfg.actions["stroke_rate"]
        cfg.rewards.pop("stroke_rate", None)
    # Stage one stays on the ground. A tipped spawn reads as falling out of the sky.
    if "foot_friction" in cfg.events:
        cfg.events["foot_friction"].params["ranges"] = (0.9, 0.9)
    if "wheel_rolling_resistance" in cfg.events:
        rolling = 0.008 * SKATE_WHEEL_RADIUS
        cfg.events["wheel_rolling_resistance"].params["ranges"] = (rolling, rolling)
        cfg.events["wheel_bearing_friction"].params["ranges"] = (1e-4, 1e-4)
        cfg.events["wheel_bearing_damping"].params["ranges"] = (0.0, 0.0)
    cfg.events["reset_base"].params["pose_range"]["yaw"] = (-np.pi, np.pi)
    cfg.events["reset_base"].params["pose_range"]["roll"] = (0.0, 0.0)
    cfg.events["reset_base"].params["pose_range"]["pitch"] = (0.0, 0.0)
    cfg.events["reset_base"].params["pose_range"]["z"] = (0.0, 0.005)
    cfg.events["reset_base"].params["pose_range"]["x"] = (-0.08, 0.08)
    cfg.events["reset_base"].params["pose_range"]["y"] = (-0.08, 0.08)
    cfg.commands["twist"].debug_vis = False
    cfg.commands["twist"].rel_standing_envs = 0.0
    cfg.commands["twist"].rel_heading_envs = 0.0
    cfg.commands["twist"].rel_rotation_envs = 0.0
    cfg.commands["twist"].rel_forward_envs = 0.0
    cfg.viewer.origin_type = cfg.viewer.OriginType.WORLD
    # Aim a little toward the camera so the near robots are not cut off at the bottom.
    cfg.viewer.lookat = (0.30, 0.33, 0.22)
    cfg.viewer.azimuth = AZIMUTH
    cfg.viewer.elevation = ELEVATION
    cfg.viewer.distance = DISTANCE
    cfg.viewer.fovy = FOVY
    cfg.viewer.width = WIDTH
    cfg.viewer.height = HEIGHT
    cfg.viewer.max_extra_envs = NUM_ENVS - 1
    cfg.viewer.enable_reflections = False
    cfg.viewer.enable_shadows = True
    cfg.viewer.env_idx = 0
    base = ManagerBasedRlEnv(cfg=cfg, device=DEVICE, render_mode="rgb_array")
    _finish_look(base.sim.mj_model)
    _scatter_origins(base)
    # The renderer uploads textures when it is created, which is before the concrete is written.
    if base._offline_renderer is not None:
        base._offline_renderer.close()
        base._offline_renderer = OffscreenRenderer(
            model=base.sim.mj_model, cfg=cfg.viewer, scene=base.scene
        )
        base._offline_renderer.initialize()
    # OffscreenRenderer pins extent to the camera distance, and the directional
    # shadow box is that extent times shadowclip. Robots near the frame edge fall
    # outside it. Widen the box, and keep the near plane from scaling with it.
    model = base.sim.mj_model
    model.stat.extent = 18.0
    model.vis.map.znear = 0.002
    model.vis.map.shadowclip = 2.0
    env = RslRlVecEnvWrapper(base, clip_actions=load_rl_cfg(task).clip_actions)
    return env, base


def load_policy(task: str, env, checkpoint: str):
    agent_cfg = load_rl_cfg(task)
    runner = (load_runner_cls(task) or MjlabOnPolicyRunner)(
        env, asdict(agent_cfg), device=DEVICE
    )
    runner.load(checkpoint, load_cfg={"actor": True}, strict=True, map_location=DEVICE)
    return runner.get_inference_policy(device=DEVICE)


def assign_commands(base, vx_lo: float, vx_hi: float, wz: float) -> tuple[torch.Tensor, torch.Tensor]:
    n = base.num_envs
    g = torch.Generator(device="cpu")
    g.manual_seed(7)
    vx = (vx_lo + (vx_hi - vx_lo) * torch.rand(n, generator=g)).to(base.device)
    wz_t = ((torch.rand(n, generator=g) * 2 - 1) * wz).to(base.device)
    return vx, wz_t


def hold(base, vx: torch.Tensor, wz: torch.Tensor) -> None:
    term = base.command_manager.get_term("twist")
    term.cfg.rel_standing_envs = 0.0
    term.cfg.rel_heading_envs = 0.0
    term.cfg.rel_forward_envs = 0.0
    term.cfg.rel_world_envs = 0.0
    term.cfg.rel_rotation_envs = 0.0
    term.is_standing_env[:] = False
    term.is_heading_env[:] = False
    if hasattr(term, "is_world_env"):
        term.is_world_env[:] = False
    if hasattr(term, "is_rotation_env"):
        term.is_rotation_env[:] = False
    if hasattr(term, "is_forward_env"):
        term.is_forward_env[:] = False
    term.vel_command_b[:, 0] = vx
    term.vel_command_b[:, 1] = 0.0
    term.vel_command_b[:, 2] = wz
    term.time_left[:] = 1.0e6
    term.cfg.debug_vis = False


def act(policy, obs, stochastic: bool, wobble: float = 0.0):
    with torch.no_grad():
        if stochastic:
            return policy(obs, stochastic_output=True)
        if wobble > 0.0:
            mean = policy(obs)
            sample = policy(obs, stochastic_output=True)
            return mean + wobble * (sample - mean)
        return policy(obs)


def rollout_stats(env, base, policy, vx, wz, seconds: float, stochastic: bool = False) -> str:
    robot = base.scene["robot"]
    obs, _ = env.reset()
    hold(base, vx, wz)
    dt = base.step_dt
    steps = int(round(seconds / dt))
    height, speed = [], []
    for _ in range(steps):
        hold(base, vx, wz)
        obs, _, _, _ = env.step(act(policy, obs, stochastic))
        z = robot.data.root_link_pos_w[:, 2]
        v = torch.linalg.norm(robot.data.root_link_lin_vel_w[:, :2], dim=-1)
        height.append(z)
        speed.append(v)
    z = torch.cat(height)
    v = torch.cat(speed)
    z_sorted = z.sort().values
    def pct(p):
        return float(z_sorted[int((len(z_sorted) - 1) * p)])
    return (
        f"z p10 {pct(0.1):.3f} p50 {pct(0.5):.3f} p90 {pct(0.9):.3f}  "
        f"speed p50 {v.median():.3f} mean {v.mean():.3f}"
    )


def record(env, base, policy, vx, wz, path: Path, stochastic: bool = False, warmup_s: float = 0.0,
           wobble: float = 0.0) -> None:
    import imageio.v2 as imageio

    robot = base.scene["robot"]
    obs, _ = env.reset()
    hold(base, vx, wz)
    dt = base.step_dt
    for _ in range(int(round(warmup_s / dt))):
        hold(base, vx, wz)
        obs, _, _, _ = env.step(act(policy, obs, stochastic, wobble))
    frame_dt = 1.0 / FPS
    steps = int(round(SECONDS / dt))
    path.parent.mkdir(parents=True, exist_ok=True)
    next_frame = 0.0
    written = 0
    with imageio.get_writer(
        str(path), fps=FPS, codec="libx264", quality=8, macro_block_size=1,
        ffmpeg_params=["-pix_fmt", "yuv420p"],
    ) as writer:
        for i in range(steps):
            hold(base, vx, wz)
            obs, _, _, _ = env.step(act(policy, obs, stochastic, wobble))
            t = (i + 1) * dt
            if t + 1e-9 < next_frame:
                continue
            frame = base.render()
            if frame is None:
                continue
            if frame.ndim == 4:
                frame = frame[0]
            writer.append_data(frame)
            written += 1
            if os.environ.get("STILL") == "1":
                from PIL import Image
                Image.fromarray(frame).save(path.with_suffix(".jpg"), quality=90)
            next_frame += frame_dt
            if written % FPS == 0:
                z = robot.data.root_link_pos_w[:, 2]
                print(f"  {path.name} t={t:.1f}s frames={written} upright={(z > 0.18).float().mean():.0%}", flush=True)
    print(path, "frames", written, flush=True)


WALK = "Mjlab-Velocity-Xiaoxiaobai"
SKATE = "Mjlab-Skate-Stroke-Xiaoxiaobai"
WALK_DIR = LOGS / "mjlab_xiaoxiaobai_velocity" / "2026-11-02_23-26-32_v6_symmetry_aug_scratch"
# The walk everyone resumed as "lean27000". Free-arm policies splay the arms.
WALK_27000 = (
    LOGS / "mjlab_xiaoxiaobai_velocity" / "2026-11-04_16-54-20_v6_kp125_ft_lean_vx" / "model_27000.pt"
)
SKATE_V1 = LOGS / "mjlab_xiaoxiaobai_skate_stroke" / "2026-11-08_22-19-56_stroke_v1"
SKATE_V2 = LOGS / "mjlab_xiaoxiaobai_skate_stroke" / "2026-11-09_02-00-18_stroke_v2"

WALK_PROBE = [
    WALK_DIR / "model_0.pt",
    WALK_DIR / "model_1500.pt",
    WALK_DIR / "model_3000.pt",
    WALK_DIR / "model_5000.pt",
    WALK_DIR / "model_8000.pt",
    WALK_DIR / "model_14999.pt",
    LOGS / "mjlab_xiaoxiaobai_velocity" / "2026-11-20_11-28-31_natural_free_arms_lead_v1" / "model_39994.pt",
]
SKATE_PROBE = [
    SKATE_V1 / "model_0.pt",
    SKATE_V2 / "model_1500.pt",
    SKATE_V2 / "model_4000.pt",
    SKATE_V2 / "model_8000.pt",
    SKATE_V2 / "model_14999.pt",
    LOGS / "mjlab_xiaoxiaobai_skate_stroke" / "2026-11-09_16-23-32_stroke_v7_skater" / "model_32000.pt",
    LOGS / "mjlab_xiaoxiaobai_skate_stroke" / "2026-11-09_22-11-19_stroke_v11_turn" / "model_40500.pt",
    LOGS / "mjlab_xiaoxiaobai_skate_stroke" / "2026-11-11_23-15-45_stroke44999_backpack" / "model_48998.pt",
]


# Head yaw joined the skate action vector at v12. Earlier checkpoints are 18-D.
SKATE_ACT_18 = r"^(?!head|leg_flex|.*_wheel_).*$"


def _probe_group(task: str, ckpts: list[Path], vx_lo: float, vx_hi: float, wz: float,
                 action_filter: str | None, stochastic: bool = False):
    print(f"\n=== {task} actions={action_filter or 'default'} ===", flush=True)
    env, base = make_env(task, action_filter)
    term = env.unwrapped.action_manager.get_term("joint_pos")
    print(f"action_dim {term.action_dim} joints {term._target_names}", flush=True)
    vx, wz_t = assign_commands(base, vx_lo, vx_hi, wz)
    agent_cfg = load_rl_cfg(task)
    runner = (load_runner_cls(task) or MjlabOnPolicyRunner)(
        env, asdict(agent_cfg), device=DEVICE
    )
    print(
        f"actor in {runner.alg.actor.mlp[0].in_features} out {runner.alg.actor.mlp[-1].out_features}",
        flush=True,
    )
    for ckpt in ckpts:
        try:
            runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=DEVICE)
        except Exception as exc:
            msg = " | ".join(str(exc).splitlines()[:8])
            print(f"FAIL {ckpt.parent.name}/{ckpt.name}: {msg}", flush=True)
            continue
        policy = runner.get_inference_policy(device=DEVICE)
        stats = rollout_stats(env, base, policy, vx, wz_t, 3.0, stochastic)
        print(f"OK {ckpt.parent.name}/{ckpt.name}: {stats}", flush=True)
    env.close()


def probe() -> None:
    _probe_group(SKATE, [SKATE_V1 / "model_0.pt", SKATE_V2 / "model_14999.pt"], 0.30, 0.50, 0.25, SKATE_ACT_18)
    _probe_group(SKATE, [SKATE_PROBE[-1]], 0.30, 0.50, 0.25, None)


def render_all() -> None:
    # task, ckpt, filename, vx range, wz, drop the cadence action, sample actions, spawn fallen
    jobs = [
        (WALK, WALK_DIR / "model_500.pt", "walk_01_fall.mp4", 0.28, 0.42, 0.22, False, False, False, 0.8, 0.45),
        (WALK, WALK_DIR / "model_3000.pt", "walk_02_awkward.mp4", 0.32, 0.48, 0.22, False, False, False, 1.2, 0.0),
        (WALK, WALK_27000, "walk_03_natural.mp4", 0.32, 0.48, 0.22, False, False, False, 1.2, 0.0),
        (SKATE, SKATE_V2 / "model_1000.pt", "skate_01_fall.mp4", 0.22, 0.38, 0.18, True, False, False, 0.8, 0.4),
        (SKATE, SKATE_V2 / "model_4000.pt", "skate_02_awkward.mp4", 0.28, 0.42, 0.18, True, False, False, 1.2, 0.0),
        (SKATE, LOGS / "mjlab_xiaoxiaobai_skate_stroke" / "2026-11-09_22-11-19_stroke_v11_turn" / "model_40500.pt",
         "skate_03_natural.mp4", 0.28, 0.42, 0.18, True, False, False, 1.2, 0.0),
    ]
    only = os.environ.get("ONLY")
    if only:
        jobs = [job for job in jobs if only in job[2]]
    groups: dict[tuple, list] = {}
    for job in jobs:
        groups.setdefault((job[0], job[6], job[8]), []).append(job)
    for (task, drop_stroke, fallen), task_jobs in groups.items():
        env, base = make_env(task, drop_stroke=drop_stroke, fallen=fallen)
        print(
            f"geoms {base.sim.mj_model.ngeom} envs {base.num_envs} "
            f"actions {env.num_actions} fallen={fallen} drop_stroke={drop_stroke}",
            flush=True,
        )
        agent_cfg = load_rl_cfg(task)
        runner = (load_runner_cls(task) or MjlabOnPolicyRunner)(
            env, asdict(agent_cfg), device=DEVICE
        )
        for task, ckpt, name, vx_lo, vx_hi, wz, _drop, stochastic, _fallen, warmup, wobble in task_jobs:
            runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=DEVICE)
            policy = runner.get_inference_policy(device=DEVICE)
            vx, wz_t = assign_commands(base, vx_lo, vx_hi, wz)
            path = OUT_DIR / name
            print("recording", name, "stochastic" if stochastic else "mean", flush=True)
            record(env, base, policy, vx, wz_t, path, stochastic, warmup, wobble)
        env.close()


if __name__ == "__main__":
    configure_torch_backends()
    _patch_renderer()
    mode = sys.argv[1] if len(sys.argv) > 1 else "probe"
    os.chdir(Path(__file__).resolve().parent)
    if mode == "probe":
        probe()
    elif mode == "render":
        render_all()
    else:
        raise SystemExit(f"unknown mode {mode}")
