"""Left/right mirror augmentation for rsl_rl's ``symmetry_cfg``.

The mirror plane is the trunk's x-z plane. Every observation term is linear under
the mirror, so each observation group gets one ``[D, D]`` matrix built from the
term names and dimensions of the live observation manager.
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch
from tensordict import TensorDict

# Reflection across the trunk x-z plane for polar vectors (velocity, gravity, force).
_REFLECT = np.diag([1.0, -1.0, 1.0])
# Joint angles flip sign under the mirror: every paired left/right joint (and the
# single head yaw) has axes that are mirror images of each other in the MJCF.
_JOINT_SIGN = -1.0

_CACHE_ATTR = "_xiaoxiaobai_mirror_maps"


def _swap_side(name: str) -> str:
    if name.startswith("left_"):
        return "right_" + name[len("left_"):]
    if name.startswith("right_"):
        return "left_" + name[len("right_"):]
    return name


def _joint_block(names: list[str]) -> np.ndarray:
    n = len(names)
    block = np.zeros((n, n))
    for i, name in enumerate(names):
        block[i, names.index(_swap_side(name))] = _JOINT_SIGN
    return block


def _swap_block(num_per_foot: int, per_foot: np.ndarray | None = None) -> np.ndarray:
    k = num_per_foot
    per_foot = np.eye(k) if per_foot is None else per_foot
    block = np.zeros((2 * k, 2 * k))
    block[:k, k:] = per_foot
    block[k:, :k] = per_foot
    return block


def _imu_rotation(env) -> np.ndarray:
    model = env.sim.mj_model
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "robot/imu")
    if site_id < 0:
        raise KeyError("IMU site 'robot/imu' not found")
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, model.site_quat[site_id])
    return mat.reshape(3, 3)  # site frame -> trunk frame


def _term_joint_names(env, term_cfg) -> list[str]:
    robot = env.scene["robot"]
    asset_cfg = term_cfg.params.get("asset_cfg")
    ids = getattr(asset_cfg, "joint_ids", slice(None)) if asset_cfg is not None else slice(None)
    names = list(robot.joint_names)
    return names[ids] if isinstance(ids, slice) else [names[i] for i in ids]


def _build_maps(env) -> dict[str, np.ndarray]:
    rot = _imu_rotation(env)
    imu_vec = rot.T @ _REFLECT @ rot
    imu_axial = -imu_vec
    action_names = list(env.action_manager.get_term("joint_pos").target_names)

    om = env.observation_manager
    maps: dict[str, np.ndarray] = {}
    for group, term_names in om.active_terms.items():
        blocks = []
        for name, dims, term_cfg in zip(term_names, om.group_obs_term_dim[group], om._group_obs_term_cfgs[group]):
            dim = int(np.prod(dims))
            if name == "base_lin_vel":
                block = imu_vec
            elif name == "base_ang_vel":
                block = imu_axial
            elif name == "projected_gravity":
                block = _REFLECT
            elif name in ("joint_pos", "joint_vel"):
                block = _joint_block(_term_joint_names(env, term_cfg))
            elif name == "actions":
                block = _joint_block(action_names)
            elif name in ("root_height", "sit_command"):
                block = np.eye(1)
            elif name == "seat_size":
                block = np.eye(3)
            elif name == "seat_pos":
                block = _REFLECT
            elif name == "command":
                block = np.diag([1.0, -1.0, -1.0])
            elif name == "gait_phase":
                # The mirrored stroke is the other skate's, half a cycle on: (cos, sin) -> -(cos, sin).
                block = -np.eye(2)
            elif name in ("foot_height", "foot_air_time", "foot_contact"):
                block = _swap_block(1)
            elif name == "foot_contact_forces":
                block = _swap_block(3, _REFLECT)
            else:
                raise KeyError(f"No mirror rule for observation term '{group}/{name}'")
            if block.shape != (dim, dim):
                raise ValueError(f"Mirror block for '{group}/{name}' is {block.shape}, term dim is {dim}")
            blocks.append(block)
        maps[group] = _block_diag(blocks)
    # Joint targets mirror; other outputs (the stroke rate) are the same on both sides.
    am = env.action_manager
    maps["__actions__"] = _block_diag([
        _joint_block(action_names) if name == "joint_pos" else np.eye(am.get_term(name).action_dim)
        for name in am.active_terms
    ])
    return maps


def _block_diag(blocks: list[np.ndarray]) -> np.ndarray:
    size = sum(b.shape[0] for b in blocks)
    out = np.zeros((size, size))
    i = 0
    for b in blocks:
        n = b.shape[0]
        out[i:i + n, i:i + n] = b
        i += n
    return out


def mirror_maps(env, device) -> dict[str, torch.Tensor]:
    env = getattr(env, "unwrapped", env)
    maps = getattr(env, _CACHE_ATTR, None)
    if maps is None:
        maps = {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in _build_maps(env).items()}
        setattr(env, _CACHE_ATTR, maps)
    return maps


def mirror_augment(
    env,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """Return ``[original; mirrored]`` observations and actions (batch doubled)."""
    obs_aug = None
    actions_aug = None
    if obs is not None:
        maps = mirror_maps(env, obs.device)
        mirrored = obs.clone()
        for key in obs.keys():
            mirrored[key] = obs[key] @ maps[key].T
        obs_aug = torch.cat([obs, mirrored], dim=0)
    if actions is not None:
        maps = mirror_maps(env, actions.device)
        actions_aug = torch.cat([actions, actions @ maps["__actions__"].T], dim=0)
    return obs_aug, actions_aug
