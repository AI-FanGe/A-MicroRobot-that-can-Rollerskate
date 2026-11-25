# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import mujoco
import torch
from mjlab.entity import Entity
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.envs.mdp.events import push_by_setting_velocity
from mjlab.envs.mdp.dr._core import _get_entity_indices
from mjlab.envs.mdp.dr.geom import _recompute_geom_bounds
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, quat_from_euler_xyz, quat_mul, sample_uniform, yaw_quat
from mjlab.tasks.velocity.mdp.velocity_command import (
    UniformVelocityCommand,
    UniformVelocityCommandCfg,
)


############################ COMMANDS #############################

class UniformVelocityCommandWithRotation(UniformVelocityCommand):
    """Extends UniformVelocityCommand with a `rel_rotation_envs` fraction.

    Rotation-only environments receive zero linear velocity and a non-zero angular 
    velocity in [`cfg.rotation_env_ang_vel_range[0]`, `cfg.rotation_env_ang_vel_range[1]`], 
    with an absolute value of at least `cfg.rotation_min_ang_vel`.
    """

    cfg: "UniformVelocityCommandWithRotationCfg"

    def __init__(self, cfg: "UniformVelocityCommandWithRotationCfg", env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.is_rotation_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_switch_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.switching = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.switch_left = torch.zeros(self.num_envs, device=self.device)
        self.switch_target = torch.zeros(self.num_envs, 3, device=self.device)
        self.switch_rate = torch.zeros(self.num_envs, 3, device=self.device)

    def _update_command(self) -> None:
        self._update_switch()
        super()._update_command()

    def _update_switch(self) -> None:
        """Mid-command direction changes (``cfg.direction_switch``, see DirectionSwitchCfg)."""
        sw: DirectionSwitchCfg | None = getattr(self.cfg, "direction_switch", None)
        if sw is None or not self.is_switch_env.any():
            return
        dt = self._env.step_dt
        self.switch_left -= dt
        ids = (self.is_switch_env & (self.switch_left <= 0.0)).nonzero().flatten()
        if len(ids) > 0:
            n = len(ids)
            self.switch_left[ids] = torch.empty(n, device=self.device).uniform_(*sw.interval_s)
            base = torch.where(self.switching[ids, None], self.switch_target[ids], self.vel_command_b[ids])
            target = base.clone()
            kind = torch.randint(0, 3, (n,), device=self.device)  # 0 reverse, 1 turn, 2 both
            rev, yaw = kind != 1, kind != 0

            def away(x: torch.Tensor) -> torch.Tensor:
                coin = torch.where(torch.rand(n, device=self.device) < 0.5, 1.0, -1.0)
                return torch.where(x.abs() > 0.05, -torch.sign(x), coin)

            vx = torch.empty(n, device=self.device).uniform_(*sw.vx_range)
            wz = torch.empty(n, device=self.device).uniform_(*sw.wz_range)
            target[:, 0] = torch.where(rev, away(base[:, 0]) * vx, base[:, 0])
            target[:, 2] = torch.where(yaw, away(base[:, 2]) * wz, base[:, 2])

            ramp_s = torch.empty(n, 1, device=self.device).uniform_(*sw.ramp_s)
            self.switch_rate[ids] = torch.tensor(sw.full_speed, device=self.device) / ramp_s
            self.switch_target[ids] = target
            instant = torch.rand(n, device=self.device) < sw.instant_prob
            self.vel_command_b[ids[instant]] = target[instant]
            self.switching[ids] = ~instant

        a = self.switching.nonzero().flatten()
        if len(a) > 0:
            step = self.switch_rate[a] * dt
            cur = self.vel_command_b[a]
            cur += torch.clamp(self.switch_target[a] - cur, -step, step)
            self.vel_command_b[a] = cur
            self.switching[a] = (cur - self.switch_target[a]).abs().amax(dim=1) > 1e-6

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)

        # Straight walks forward or backward (vy = wz = 0), the joystick's most common command.
        rel_straight_envs = getattr(self.cfg, "rel_straight_envs", 0.0)
        if rel_straight_envs > 0.0:
            straight = env_ids[torch.rand(len(env_ids), device=self.device) < rel_straight_envs]
            self.vel_command_b[straight, 1:] = 0.0

        rel_rotation_envs = getattr(self.cfg, "rel_rotation_envs", 0.0)

        r = torch.empty(len(env_ids), device=self.device)
        self.is_rotation_env[env_ids] = r.uniform_(0.0, 1.0) <= rel_rotation_envs

        sw: DirectionSwitchCfg | None = getattr(self.cfg, "direction_switch", None)
        self.switching[env_ids] = False
        self.is_switch_env[env_ids] = False
        if sw is not None:
            self.is_switch_env[env_ids] = (
                (torch.rand(len(env_ids), device=self.device) < sw.rel_envs)
                & ~self.is_standing_env[env_ids]
                & ~self.is_rotation_env[env_ids]
            )
            self.switch_left[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(*sw.interval_s)

        rot_ids = env_ids[self.is_rotation_env[env_ids]]
        if len(rot_ids) == 0:
            return

        self.vel_command_b[rot_ids, 0] = 0.0
        self.vel_command_b[rot_ids, 1] = 0.0

        # Sample angular velocity from the rotation-specific range if provided,
        # otherwise reuse what the parent sampled from cfg.ranges.ang_vel_z.
        if self.cfg.rotation_env_ang_vel_range is not None:
            ang = torch.empty(len(rot_ids), device=self.device).uniform_(
                *self.cfg.rotation_env_ang_vel_range
            )
        else:
            ang = self.vel_command_b[rot_ids, 2]

        # Ensure non-zero angular velocity.
        min_abs_ang = self.cfg.rotation_min_ang_vel
        too_small = ang.abs() < min_abs_ang
        if too_small.any():
            signs = torch.where(
                torch.rand(too_small.sum(), device=self.device) > 0.5,
                torch.ones(too_small.sum(), device=self.device),
                -torch.ones(too_small.sum(), device=self.device),
            )
            ang[too_small] = signs * min_abs_ang
        self.vel_command_b[rot_ids, 2] = ang


@dataclass(kw_only=True)
class DirectionSwitchCfg:
    """Sudden direction changes in the middle of a command (``cfg.direction_switch``).

    At each resample a ``rel_envs`` fraction of the walking envs (not standing, not
    rotation-only) becomes a switch env until its next resample. Every ``interval_s`` it reverses
    vx (to ``vx_range``), turns (wz to ``wz_range``, against the current turn if any) or both, and
    ramps the command there at ``full_speed`` per ``ramp_s`` per axis, like the robot's
    command ramp (an instant step with ``instant_prob``).
    """

    rel_envs: float = 0.3
    interval_s: tuple[float, float] = (1.0, 2.5)
    vx_range: tuple[float, float] = (0.2, 0.5)
    wz_range: tuple[float, float] = (0.3, 0.75)
    full_speed: tuple[float, float, float] = (0.5, 0.3, 1.5)
    ramp_s: tuple[float, float] = (0.4, 1.0)
    instant_prob: float = 0.2


@dataclass(kw_only=True)
class UniformVelocityCommandWithRotationCfg(UniformVelocityCommandCfg):
    """Configuration for UniformVelocityCommandWithRotation."""

    rel_rotation_envs: float = 0.0
    """Fraction of environments that receive pure-rotation commands
    (zero linear velocity, non-zero angular velocity)."""

    rotation_min_ang_vel: float = 0.3
    """Minimum absolute angular velocity assigned to rotation-only environments."""

    rotation_env_ang_vel_range: tuple[float, float] | None = None
    """Angular velocity range for rotation-only environments.
    If None, uses cfg.ranges.ang_vel_z (same range as normal environments)."""

    def build(self, env: ManagerBasedRlEnv) -> UniformVelocityCommandWithRotation:
        return UniformVelocityCommandWithRotation(self, env)


class DragVelocityCommand(UniformVelocityCommandWithRotation):
    """Heading envs hold the heading the robot was reset to by reset_drag (``env._drag_yaw``),
    give or take ``cfg.heading_offset``, as the robot's heading hold does from where it starts."""

    cfg: "DragVelocityCommandCfg"

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        yaw = self._env.__dict__.get("_drag_yaw")
        if not self.cfg.heading_command or yaw is None:
            return
        offset = torch.empty(len(env_ids), device=self.device).uniform_(-self.cfg.heading_offset, self.cfg.heading_offset)
        self.heading_target[env_ids] = torch.remainder(yaw[env_ids] + offset + math.pi, 2 * math.pi) - math.pi


@dataclass(kw_only=True)
class DragVelocityCommandCfg(UniformVelocityCommandWithRotationCfg):
    heading_offset: float = 0.0

    def build(self, env: ManagerBasedRlEnv) -> DragVelocityCommand:
        return DragVelocityCommand(self, env)
    

############################ REWARDS ##############################

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _standing(env: ManagerBasedRlEnv, command_name: str, command_threshold: float) -> torch.Tensor:
    """Envs commanded to stand still (pose's standing regime)."""
    command = env.command_manager.get_command(command_name)
    return (torch.norm(command[:, :2], dim=1) + command[:, 2].abs()) < command_threshold


class upright:
    """Reward for keeping the base at a target pitch orientation.

    Penalizes deviation from a given pitch angle (in radians) rather than
    always rewarding a perfectly vertical posture.

    Args:
        std: Standard deviation of the Gaussian kernel (controls reward sharpness).
        pitch: Target pitch angle in radians. 0.0 = perfectly upright.
               Positive values mean leaning forward.
        pitch_per_vx: Added to the target per m/s of commanded forward velocity [rad s/m].
        pitch_per_vx_back: The same for backward commands (vx < 0); ``pitch_per_vx`` if None.
        std_standing: ``std`` while the command is below ``command_threshold``; ``std`` if None.
        pitch_back_band: Leaning back of the target by up to this much [rad] is not penalized.
        command_name: Velocity command read by ``pitch_per_vx`` and ``std_standing``.
        asset_cfg: Scene entity configuration for the robot body to track.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        pass

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        std: float,
        pitch: float = 0.0,
        pitch_per_vx: float = 0.0,
        pitch_per_vx_back: float | None = None,
        std_standing: float | None = None,
        command_threshold: float = 0.01,
        command_name: str | None = None,
        pitch_back_band: float = 0.0,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        asset: Entity = env.scene[asset_cfg.name]

        if asset_cfg.body_ids:
            body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
        else:
            body_quat_w = asset.data.root_link_quat_w

        gravity_w = asset.data.gravity_vec_w
        projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)

        # Normalize to unit vector so the error is scale-independent.
        gravity_norm = projected_gravity_b.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        projected_gravity_b_unit = projected_gravity_b / gravity_norm

        # At pitch angle θ, the normalised gravity unit vector in body frame has
        # x = sin(θ), y = 0 (assuming flat ground, no roll, no terrain slope).
        if (pitch_per_vx != 0.0 or pitch_per_vx_back) and command_name is not None:
            vx = env.command_manager.get_command(command_name)[:, 0]
            back = pitch_per_vx if pitch_per_vx_back is None else pitch_per_vx_back
            target_gx = torch.sin(pitch + torch.where(vx < 0.0, back, pitch_per_vx) * vx)
        else:
            target_gx = math.sin(pitch)
        if pitch_back_band > 0.0:
            gx = projected_gravity_b_unit[:, 0]
            target_pitch = torch.asin(torch.as_tensor(target_gx, device=gx.device)).expand_as(gx)
            target_gx = torch.clamp(gx, torch.sin(target_pitch - pitch_back_band), torch.sin(target_pitch))
        xy_error = (
            torch.square(projected_gravity_b_unit[:, 0] - target_gx)
            + torch.square(projected_gravity_b_unit[:, 1])
        )
        if std_standing is not None and command_name is not None:
            std = torch.where(_standing(env, command_name, command_threshold), std_standing, std)
        return torch.exp(-xy_error / std**2)

    def reset(self, env_ids: torch.Tensor) -> None:
        del env_ids  # Unused.


class gyro_with_touchdown_spikes:
    """IMU gyro plus a decaying, ringing disturbance that starts at foot touchdowns.

    On the robot each footfall puts a 20-80 ms burst on the gyro (median peak 0.64 rad/s
    on the roll axis, 3x sim) that does not reflect a matching trunk rotation, and a
    policy trained without it answers every touchdown with a sharp correction. Each
    touchdown (with ``probability``) restarts ``a * decay**k * cos(omega * k)`` per policy
    tick k, with ``a`` uniform in +-``amplitude`` per IMU axis.

    ``noise`` adds uniform +-``noise`` [rad/s] per axis, +-``noise_standing`` while the command
    is below ``command_threshold`` (the trunk vibration the gyro sees comes from walking). Use it
    instead of the term's ``noise`` cfg; it also applies with the observation corruption off.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        n, dev = env.num_envs, env.device
        self._amp = torch.zeros(n, 3, device=dev)
        self._decay = torch.zeros(n, device=dev)
        self._omega = torch.zeros(n, device=dev)
        self._age = torch.zeros(n, device=dev)
        self._spike = torch.zeros(n, 3, device=dev)
        self._last_step = -1

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        self._amp[slice(None) if env_ids is None else env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        sensor_name: str,
        contact_sensor_name: str,
        amplitude: tuple[float, float, float],
        decay_range: tuple[float, float],
        omega_range: tuple[float, float],
        probability: float,
        noise: float = 0.0,
        noise_standing: float | None = None,
        command_name: str = "twist",
        command_threshold: float = 0.01,
    ) -> torch.Tensor:
        gyro = env.scene[sensor_name].data
        if env.common_step_counter != self._last_step:
            self._last_step = env.common_step_counter
            touchdown = env.scene[contact_sensor_name].compute_first_contact(dt=env.step_dt).any(dim=1)
            start = touchdown & (torch.rand_like(self._decay) < probability)
            ids = start.nonzero(as_tuple=True)[0]
            if len(ids) > 0:
                amp = torch.tensor(amplitude, device=gyro.device)
                self._amp[ids] = (2.0 * torch.rand(len(ids), 3, device=gyro.device) - 1.0) * amp
                self._decay[ids] = torch.empty(len(ids), device=gyro.device).uniform_(*decay_range)
                self._omega[ids] = torch.empty(len(ids), device=gyro.device).uniform_(*omega_range)
                self._age[ids] = -1.0
            self._age += 1.0
            shape = self._decay ** self._age * torch.cos(self._omega * self._age)
            self._spike = self._amp * shape.unsqueeze(1)
        if noise == 0.0 and not noise_standing:
            return gyro + self._spike
        scale = torch.full_like(self._decay, noise)
        if noise_standing is not None:
            scale = torch.where(_standing(env, command_name, command_threshold), noise_standing, scale)
        return gyro + self._spike + (2.0 * torch.rand_like(gyro) - 1.0) * scale.unsqueeze(1)


class projected_gravity_with_mount_tilt:
    """Projected gravity as seen by an IMU mounted with a small roll/pitch error.

    Each episode draws a fixed tilt uniform in +-``max_tilt`` [rad] per axis, so the
    policy cannot treat the IMU's level as the trunk's level; rewards keep using the
    true trunk orientation.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        self._max_tilt = cfg.params["max_tilt"]
        self._rot = torch.eye(3, device=env.device).repeat(env.num_envs, 1, 1)
        self.reset(None)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        ids = torch.arange(self._rot.shape[0], device=self._rot.device) if env_ids is None else env_ids
        n = self._rot[ids].shape[0]
        if n == 0:
            return
        roll, pitch = (2.0 * torch.rand(2, n, device=self._rot.device) - 1.0) * self._max_tilt
        cr, sr, cp, sp = roll.cos(), roll.sin(), pitch.cos(), pitch.sin()
        zero, one = torch.zeros_like(roll), torch.ones_like(roll)
        rx = torch.stack([one, zero, zero, zero, cr, -sr, zero, sr, cr], dim=1).view(n, 3, 3)
        ry = torch.stack([cp, zero, sp, zero, one, zero, -sp, zero, cp], dim=1).view(n, 3, 3)
        self._rot[ids] = rx @ ry

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        max_tilt: float,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        del max_tilt
        g = env.scene[asset_cfg.name].data.projected_gravity_b
        return (self._rot @ g.unsqueeze(-1)).squeeze(-1)


def feet_distance_penalty(
    env: ManagerBasedRlEnv,
    min_dist: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize the feet getting too close to each other in the horizontal plane.

    Discourages the robot from stepping on its own feet by adding a smooth,
    anticipatory cost as soon as the horizontal distance between the two foot
    sites drops below ``min_dist``. Above the threshold the cost is zero.

    Returns ``clamp(min_dist - d, min=0)`` per env (use with a negative weight),
    where ``d`` is the horizontal (xy) distance between the two foot sites.

    Args:
        min_dist: Minimum desired horizontal distance between feet, in meters.
        asset_cfg: Scene entity configuration whose ``site_names`` select exactly
            the two foot sites.
    """
    asset: Entity = env.scene[asset_cfg.name]
    foot_pos_xy = asset.data.site_pos_w[:, asset_cfg.site_ids, :2]  # [B, 2, 2]
    dist = torch.norm(foot_pos_xy[:, 0] - foot_pos_xy[:, 1], dim=-1)  # [B]
    return torch.clamp(min_dist - dist, min=0.0)


def feet_too_wide_penalty(
    env: ManagerBasedRlEnv,
    max_dist: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize a stance wider than ``max_dist``.

    Uses the sideways distance between the two foot sites in the trunk's heading
    frame, so step length does not count. Returns ``clamp(d - max_dist, min=0)``
    per env (use with a negative weight).
    """
    asset: Entity = env.scene[asset_cfg.name]
    foot_pos = asset.data.site_pos_w[:, asset_cfg.site_ids, :]  # [B, 2, 3]
    rel_b = quat_apply_inverse(yaw_quat(asset.data.root_link_quat_w), foot_pos[:, 0] - foot_pos[:, 1])
    return torch.clamp(rel_b[:, 1].abs() - max_dist, min=0.0)


def foot_contact_forces_yaw(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Per-foot net contact force in the trunk's heading frame, log-compressed.

    The upstream term reports world-frame forces, which cannot be mirrored
    left/right without the heading; this frame makes the y component lateral.
    """
    sensor = env.scene[sensor_name]
    force_w = sensor.data.force  # [B, N, 3]
    asset: Entity = env.scene[asset_cfg.name]
    q = yaw_quat(asset.data.root_link_quat_w).unsqueeze(1).expand(-1, force_w.shape[1], -1)
    forces = quat_apply_inverse(q, force_w).flatten(start_dim=1)
    return torch.sign(forces) * torch.log1p(torch.abs(forces))


def command_lead_excess(
    env: ManagerBasedRlEnv,
    max_lead: float,
    action_name: str = "joint_pos",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Summed amount [rad] by which position targets lead their joints beyond ``max_lead``.

    Past ~160 deg of lead the servo model is at full PWM, so a larger target buys no
    torque and no other term reacts to it; targets there drifted to thousands of
    degrees, which the motor's goal-position register cannot represent.
    """
    term = env.action_manager.get_term(action_name)
    asset: Entity = env.scene[asset_cfg.name]
    lead = (term._processed_actions - asset.data.joint_pos[:, term.target_ids]).abs()
    return torch.clamp(lead - max_lead, min=0.0).sum(dim=1)


def standing_crouch(
    env: ManagerBasedRlEnv,
    deadband: float,
    command_name: str = "twist",
    command_threshold: float = 0.01,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Summed amount [rad] by which the joints leave the default pose beyond ``deadband``,
    while the command is below ``command_threshold`` (pose's standing regime).

    Linear, so it keeps pulling where pose's Gaussian has gone flat: fine-tuned down to 5.5 V,
    the walk sank a little further every time it stopped, after 18 s standing at 5.5 V knees
    1.4 rad and hips 1.0 rad off the default (trunk 18 -> 13 cm), and shook on the robot."""
    asset: Entity = env.scene[asset_cfg.name]
    offset = (asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]).abs()
    command = env.command_manager.get_command(command_name)
    standing = (torch.norm(command[:, :2], dim=1) + command[:, 2].abs()) < command_threshold
    return torch.clamp(offset - deadband, min=0.0).sum(dim=1) * standing.float()


def low_vin_envs(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    fraction: float,
    vin_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Startup: redraws the battery voltage of ``fraction`` of the envs from ``vin_range``.

    BamActuator draws it once, uniformly over its vin_range; the backpack fine-tune over
    5-8 V walked at 6-8 V but fell 22-31% at 5 V (scripts/walk_posture_eval.py)."""
    asset: Entity = env.scene[asset_cfg.name]
    low = torch.rand(env.num_envs, device=env.device) < fraction
    for act in asset.actuators:
        vin = getattr(act, "vin_tensor", None)
        if vin is not None:
            vin[low, 0] = torch.empty(int(low.sum()), device=env.device).uniform_(*vin_range)


def _vin_actuators(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> list:
    return [a for a in env.scene[asset_cfg.name].actuators if getattr(a, "vin_tensor", None) is not None]


def battery_discharge(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    fraction: float,
    vin_start_range: tuple[float, float],
    vin_end_min: float,
    min_drop: float,
    duration_range_s: tuple[float, float],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Reset: ``fraction`` of the episodes run their battery down from a voltage in
    ``vin_start_range`` by at least ``min_drop`` to no less than ``vin_end_min``, linearly over
    ``duration_range_s``, then hold it (``battery_discharge_step`` applies it). The others keep
    the env's startup voltage.

    BamActuator draws the voltage once per env for the whole run; on the robot it falls from
    7 through 6 to 5 V while walking."""
    acts = _vin_actuators(env, asset_cfg)
    state = getattr(env, "_xiaoxiaobai_battery", None)
    if state is None:
        zeros = torch.zeros(env.num_envs, device=env.device)
        state = env._xiaoxiaobai_battery = {
            "base": acts[0].vin_tensor[:, 0].clone(),
            "on": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
            "start": zeros.clone(), "end": zeros.clone(), "duration": torch.ones(env.num_envs, device=env.device),
        }
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    k = len(ids)
    start = torch.empty(k, device=env.device).uniform_(*vin_start_range)
    room = torch.clamp(start - min_drop - vin_end_min, min=0.0)
    state["on"][ids] = torch.rand(k, device=env.device) < fraction
    state["start"][ids] = start
    state["end"][ids] = vin_end_min + torch.rand(k, device=env.device) * room
    state["duration"][ids] = torch.empty(k, device=env.device).uniform_(*duration_range_s)
    vin = torch.where(state["on"][ids], start, state["base"][ids])
    for act in acts:
        act.vin_tensor[ids, 0] = vin


def battery_discharge_step(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Step: the supply voltage of the episodes ``battery_discharge`` runs down."""
    del env_ids
    state = getattr(env, "_xiaoxiaobai_battery", None)
    if state is None:
        return
    progress = torch.clamp(env.episode_length_buf * env.step_dt / state["duration"], max=1.0)
    vin = torch.where(state["on"], state["start"] + (state["end"] - state["start"]) * progress, state["base"])
    for act in _vin_actuators(env, asset_cfg):
        act.vin_tensor[:, 0] = vin


def arm_squeeze(
    env: ManagerBasedRlEnv,
    margin: float,
    action_name: str = "joint_pos",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Summed amount [rad] by which position targets come closer than ``margin`` to the joint
    limit on the trunk side (the limit nearer the default pose), or past it.

    dof_pos_limits sees only the achieved angle, and pose is happy with arms near the default:
    the crouch fine-tune stood with shoulder roll on the trunk stop and its target 0.13 rad
    inside the trunk half the time, and on the robot squeezed its arms against the body."""
    term = env.action_manager.get_term(action_name)
    asset: Entity = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    cols = [term.target_ids.tolist().index(i) for i in ids]
    target = term._processed_actions[:, cols]
    lower, upper = asset.data.joint_pos_limits[:, ids, 0], asset.data.joint_pos_limits[:, ids, 1]
    default = asset.data.default_joint_pos[:, ids]
    lower_is_trunk = (default - lower) < (upper - default)
    clearance = torch.where(lower_is_trunk, target - lower, upper - target)
    return torch.clamp(margin - clearance, min=0.0).sum(dim=1)


def arm_spread(
    env: ManagerBasedRlEnv,
    max_out: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Summed amount [rad] by which the joints open beyond ``max_out`` from zero, on the side
    of their default pose (shoulder roll: away from the trunk)."""
    asset: Entity = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    out = asset.data.joint_pos[:, ids] * torch.sign(asset.data.default_joint_pos[:, ids])
    return torch.clamp(out - max_out, min=0.0).sum(dim=1)


class sustained_roll:
    """|sin| of the trunk roll averaged over the last ``tau`` seconds (exponential average).

    Walking rocks the trunk a few degrees from side to side each step; the average keeps only
    a lasting lean, which a lopsided robot holds unless the policy rights it."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        self._alpha = min(1.0, env.step_dt / cfg.params["tau"])
        self._mean = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        self._mean[slice(None) if env_ids is None else env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        tau: float,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        del tau
        asset: Entity = env.scene[asset_cfg.name]
        quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1) if asset_cfg.body_ids else asset.data.root_link_quat_w
        g = quat_apply_inverse(quat, asset.data.gravity_vec_w)
        g_y = g[:, 1] / g.norm(dim=-1).clamp(min=1e-6)
        self._mean += self._alpha * (g_y - self._mean)
        return self._mean.abs()


def arm_reference_features(command: torch.Tensor, legs: torch.Tensor) -> torch.Tensor:
    """[1, vx, |vx|, vy, |vy|, wz, |wz|] x [1, leg joints], the products (scripts/arm_ref_fit.py)."""
    c = torch.stack([torch.ones_like(command[:, 0]), command[:, 0], command[:, 0].abs(), command[:, 1],
                     command[:, 1].abs(), command[:, 2], command[:, 2].abs()], dim=1)
    lf = torch.cat([torch.ones_like(legs[:, :1]), legs], dim=1)
    return (c[:, :, None] * lf[:, None, :]).flatten(1)


class arm_reference:
    """exp(-mean squared error / std^2) of the arm joints (from the default pose) against a
    reference policy's arms, as a linear function of the leg joints and the command fitted on
    its walk (``path``, scripts/arm_ref_fit.py).

    Cloning lean 27000's arm actions kept the arm network within ~1 deg of it on the same
    states, yet the fine-tune walking on straighter knees and leaning further back held
    shoulder roll 3-5 deg further out and swung shoulder pitch 2 deg less walking forward:
    the arms answer the trunk. This scores the achieved angles. On lean 27000's own walk the
    fit leaves ~3 deg rms on shoulder pitch and roll, ~1.7 deg on the elbow."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        import numpy as np

        data = np.load(cfg.params["path"])
        names = list(env.scene[cfg.params.get("asset_name", "robot")].joint_names)
        self._legs = torch.tensor([names.index(str(x)) for x in data["legs"]], device=env.device)
        self._arms = torch.tensor([names.index(str(x)) for x in data["arms"]], device=env.device)
        self._W = torch.tensor(data["W"], dtype=torch.float32, device=env.device)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        path: str,
        std: float,
        command_name: str = "twist",
        asset_name: str = "robot",
    ) -> torch.Tensor:
        del path
        asset: Entity = env.scene[asset_name]
        q = asset.data.joint_pos - asset.data.default_joint_pos
        ref = arm_reference_features(env.command_manager.get_command(command_name), q[:, self._legs]) @ self._W
        return torch.exp(-(q[:, self._arms] - ref).square().mean(dim=1) / std**2)


class walking_knee_floor:
    """Amount [rad] by which the knee flexion averaged over both knees and the last ``tau``
    seconds falls below ``min_flex`` while walking.

    Lean 27000 walks on 9-13 deg of knee flexion at 7.2 V; fine-tuned on 5-8 V with direction
    switches it straightened to 1-6 deg, stepped faster and lower, and walked less naturally."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        self._alpha = min(1.0, env.step_dt / cfg.params["tau"])
        self._init = cfg.params["min_flex"] + 0.1
        self._mean = torch.full((env.num_envs,), self._init, device=env.device)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        self._mean[slice(None) if env_ids is None else env_ids] = self._init

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        tau: float,
        min_flex: float,
        command_name: str = "twist",
        command_threshold: float = 0.01,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        del tau
        asset: Entity = env.scene[asset_cfg.name]
        q = asset.data.joint_pos[:, asset_cfg.joint_ids]
        flex = (q * torch.sign(asset.data.joint_pos_limits[:, asset_cfg.joint_ids].sum(dim=-1))).mean(dim=1)
        self._mean += self._alpha * (flex - self._mean)
        walking = ~_standing(env, command_name, command_threshold)
        return torch.clamp(min_flex - self._mean, min=0.0) * walking.float()


class lean_shortfall:
    """Amount [rad] by which the trunk, averaged over the last ``tau`` seconds, leans less
    against the commanded vx than ``per_vx`` (``per_vx_back`` backward) times |vx|.

    upright's Gaussian (std 0.15) barely tells 2 deg from the target, and a policy kept near
    lean 27000's legs settled on 27000's lean; tightening the std instead made walks jitter.
    Leaning further is free. With ``stand_pitch`` set, a standing command asks for at least that
    much forward lean."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        self._alpha = min(1.0, env.step_dt / cfg.params["tau"])
        self._mean = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        self._mean[slice(None) if env_ids is None else env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        tau: float,
        per_vx: float,
        per_vx_back: float,
        command_name: str = "twist",
        min_vx: float = 0.05,
        stand_pitch: float | None = None,
        command_threshold: float = 0.01,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        del tau
        asset: Entity = env.scene[asset_cfg.name]
        quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1) if asset_cfg.body_ids else asset.data.root_link_quat_w
        g = quat_apply_inverse(quat, asset.data.gravity_vec_w)
        pitch = torch.atan2(g[:, 0], -g[:, 2])
        vx = env.command_manager.get_command(command_name)[:, 0]
        need = torch.where(vx < 0.0, per_vx_back, per_vx) * vx.abs()
        shortfall = need + pitch * torch.sign(vx)
        active = vx.abs() >= min_vx
        if stand_pitch is not None:
            standing = _standing(env, command_name, command_threshold)
            shortfall = torch.where(standing, stand_pitch - pitch, shortfall)
            active |= standing
        self._mean += self._alpha * (shortfall - self._mean)
        return torch.clamp(self._mean, min=0.0) * active.float()


class walking_arm_spread:
    """Amount [rad] by which the joints, averaged over both sides and the last ``tau`` seconds,
    open beyond ``max_out`` on the side of their default pose while walking.

    Fine-tuned for 5-8 V and direction switches, lean 27000's arms held shoulder roll 3-5 deg
    further out walking backward (16-17 deg against 13) with the arm actions cloned from it.
    With ``standing_max_out`` set, standing is held under that instead."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        self._alpha = min(1.0, env.step_dt / cfg.params["tau"])
        self._mean = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        self._mean[slice(None) if env_ids is None else env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        tau: float,
        max_out: float,
        standing_max_out: float | None = None,
        command_name: str = "twist",
        command_threshold: float = 0.01,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        del tau
        asset: Entity = env.scene[asset_cfg.name]
        ids = asset_cfg.joint_ids
        out = (asset.data.joint_pos[:, ids] * torch.sign(asset.data.default_joint_pos[:, ids])).mean(dim=1)
        self._mean += self._alpha * (out - self._mean)
        standing = _standing(env, command_name, command_threshold)
        if standing_max_out is None:
            return torch.clamp(self._mean - max_out, min=0.0) * (~standing).float()
        return torch.clamp(self._mean - torch.where(standing, standing_max_out, max_out), min=0.0)


def action_rate_l2_standing(
    env: ManagerBasedRlEnv,
    standing_scale: float,
    command_name: str = "twist",
    command_threshold: float = 0.01,
) -> torch.Tensor:
    """action_rate_l2, times ``standing_scale`` while the command is below ``command_threshold``."""
    rate = torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
    return rate * torch.where(_standing(env, command_name, command_threshold), standing_scale, 1.0)


def _last_push_step(env: ManagerBasedRlEnv) -> torch.Tensor:
    if not hasattr(env, "_xiaoxiaobai_last_push_step"):
        env._xiaoxiaobai_last_push_step = torch.full((env.num_envs,), -(10**9), dtype=torch.long, device=env.device)
    return env._xiaoxiaobai_last_push_step


def push_and_mark(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    standing_only: bool = False,
    command_name: str = "twist",
    command_threshold: float = 0.01,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """push_by_setting_velocity that records the push step per env (no_stepping_penalty's
    ``push_grace_s``); with ``standing_only`` it pushes only envs commanded to stand."""
    if standing_only:
        env_ids = env_ids[_standing(env, command_name, command_threshold)[env_ids]]
    if len(env_ids) == 0:
        return
    push_by_setting_velocity(env, env_ids, velocity_range, asset_cfg)
    _last_push_step(env)[env_ids] = env.common_step_counter


def no_stepping_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str = "twist",
    command_threshold: float = 0.01,
    push_grace_s: float = 0.0,
) -> torch.Tensor:
    """
    Penalizes feet in the air when the commanded speed is below threshold.
    Discourages marching in place when the robot should stand still.
    Returns the count of airborne feet per environment (use with a negative weight).
    For ``push_grace_s`` after a push (push_and_mark) stepping is free: a step catches a
    push the ankles and hips cannot.
    """
    command = env.command_manager.get_command(command_name)  # (N, 3)
    cmd_speed = torch.norm(command[:, :2], dim=-1) + torch.abs(command[:, 2])
    below_threshold = cmd_speed < command_threshold

    sensor = env.scene.sensors[sensor_name]
    found = sensor.data.found  # (N, num_feet) or (N, num_feet, num_slots)
    if found.dim() == 3:
        found = found.any(dim=-1)  # (N, num_feet)
    in_air = ~found.bool()

    if push_grace_s > 0.0:
        since_push = (env.common_step_counter - _last_push_step(env)) * env.step_dt
        below_threshold = below_threshold & (since_push >= push_grace_s)
    return in_air.float().sum(dim=-1) * below_threshold.float()


########################## CURRICULUM #############################

class step_based_staged_curriculum:
    """
    Curriculum based on step count stages. Each stage is applied once when
    env.common_step_counter reaches the stage's step threshold.

    Stage definitions example:
    stages = [
        {
            "name": "stage 1",
            "step": 10_000 * 24,
            "apply": lambda env: env.reward_manager.get_term_cfg("term_name").weight = 1.0,
        },
        ...
    ]
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
        self.current_stage = 0

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor,
        stages: list[dict],
    ) -> dict[str, torch.Tensor]:
        del env_ids
        if (
            self.current_stage < len(stages)
            and env.common_step_counter >= stages[self.current_stage]["step"]
        ):
            stage = stages[self.current_stage]
            print(
                f"Curriculum stage {self.current_stage + 1}: {stage['name']} at step {env.common_step_counter}"
            )
            stage["apply"](env)
            self.current_stage += 1

        return {"stage": self.current_stage}

class reward_based_staged_curriculum:
    """
    Curriculum based on stages ending while a reward component gets its mean 
    episode reward accross all environments above a threshold.

    Stage definitions example:
    stages = [
        {
            "name": "stage 1",
            "reward_term_name": "term_name",
            "threshold": 0.5,
            "apply": lambda env: env.reward_manager.get_term_cfg("term_name").weight = 1.0,
        },
        ...
    ]
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
        self.rewards = torch.zeros(env.num_envs, device=env.device)
        self.current_stage = 0
        self.stage_first_step = 0
        
    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor,
        stages: list[dict],
    ) -> dict[str, torch.Tensor]:
        self.rewards[env_ids] = (
            env.reward_manager._episode_sums[stages[self.current_stage]["reward_term_name"]][env_ids]
            / env.max_episode_length_s
        )
        mean_reward = self.rewards.mean().item()

        if (
            self.current_stage < len(stages)
            and mean_reward >= stages[self.current_stage]["threshold"]
            and env.common_step_counter >= self.stage_first_step + 100 * 24
        ):
            stage = stages[self.current_stage]
            print(
                f"Curriculum stage {self.current_stage + 1}: {stage['name']} at step {env.common_step_counter} (mean episode reward: {mean_reward:.4f})"
            )
            stage["apply"](env)
            self.current_stage += 1
            self.stage_first_step = env.common_step_counter
            self.rewards.zero_()  # Reset rewards to avoid immediately triggering the next stage

        return {"stage": self.current_stage}

class reward_based_curriculum:
    """
    Curriculum based on the mean episode reward of a specific term accross all environments.
    Once the mean reward across envs exceeds a threshold, a new curriculum stage is applied.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
        self.rewards = torch.zeros(env.num_envs, device=env.device)
        self.current_stage = 0
        self.stage_first_step = 0
        
    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor,
        reward_term_name: str,
        stages: list[dict],
    ) -> dict[str, torch.Tensor]:
        self.rewards[env_ids] = (
            env.reward_manager._episode_sums[reward_term_name][env_ids]
            / env.max_episode_length_s
        )
        mean_reward = self.rewards.mean().item()

        if (
            self.current_stage < len(stages)
            and mean_reward >= stages[self.current_stage]["threshold"]
            and env.common_step_counter >= self.stage_first_step + 100 * 24
        ):
            stage = stages[self.current_stage]
            print(
                f"Curriculum stage {self.current_stage + 1}: {stage['name']} at step {env.common_step_counter} (mean episode reward: {mean_reward:.4f})"
            )
            stage["apply"](env)
            self.current_stage += 1
            self.stage_first_step = env.common_step_counter
        
        return {"stage": self.current_stage}

def set_command_velocity(
        env, 
        lin_vel_x=None, 
        lin_vel_y=None, 
        ang_vel_z=None, 
        rotation_env_ang_vel_z=None,
) -> None:
    """
    Helper function to set the command velocity parameters in the environment.
    """
    for cmd in _twist_cfgs(env):
        if lin_vel_x is not None:
            cmd.ranges.lin_vel_x = lin_vel_x
        if lin_vel_y is not None:
            cmd.ranges.lin_vel_y = lin_vel_y
        if ang_vel_z is not None:
            cmd.ranges.ang_vel_z = ang_vel_z
        if rotation_env_ang_vel_z is not None:
            cmd.rotation_env_ang_vel_range = rotation_env_ang_vel_z
    _resample_twist_command(env)

def set_stepping_parameters(
    env,
    air_time_weight: float | None = None,
    no_stepping_penalty_weight: float | None = None,
    rel_standing_envs: float | None = None,
    rel_rotation_envs: float | None = None,
) -> None:
    """
    Helper function to set stepping/standing curriculum parameters.
    """
    if air_time_weight is not None:
        env.reward_manager.get_term_cfg("air_time").weight = air_time_weight
    if no_stepping_penalty_weight is not None:
        env.reward_manager.get_term_cfg("no_stepping").weight = no_stepping_penalty_weight
    for cmd in _twist_cfgs(env):
        if rel_standing_envs is not None:
            cmd.rel_standing_envs = rel_standing_envs
        if rel_rotation_envs is not None:
            cmd.rel_rotation_envs = rel_rotation_envs
    _resample_twist_command(env)

def set_push_parameters(
    env,
    velocity_range: dict[str, tuple[float, float]] | None = None,
    interval_range: tuple[float, float] | None = None,
) -> None:
    """
    Helper function to set push event parameters.
    Returns a dict of the current (post-update) values for wandb logging.
    """
    push_event_cfg = env.event_manager.get_term_cfg("push_robot")
    if velocity_range is not None:
        push_event_cfg.params["velocity_range"] = velocity_range
    if interval_range is not None:
        push_event_cfg.interval_range_s = interval_range
        _resample_interval_time_left(env, "push_robot", interval_range)

def set_wheel_brake(env, damping: float, joint_pattern: str = r".*_wheel_.*") -> None:
    """Hold the skate wheels back with ``damping`` [N m s/rad] on top of their randomised
    bearing damping (taken on the first call), or release them with 0."""
    asset_cfg = SceneEntityCfg("robot", joint_names=(joint_pattern,))
    asset_cfg.resolve(env.scene)
    dofs = env.scene[asset_cfg.name].indexing.joint_v_adr[asset_cfg.joint_ids]
    base = getattr(env, "_wheel_base_damping", None)
    if base is None:
        base = env.sim.model.dof_damping[:, dofs].clone()
        env._wheel_base_damping = base
    env.sim.model.dof_damping[:, dofs] = base + damping


def set_reward_weights(env, weights: dict[str, float]) -> None:
    """Set reward weights from curriculum without changing reward definitions."""
    for name, weight in weights.items():
        env.reward_manager.get_term_cfg(name).weight = weight

def _twist_cfgs(env) -> list:
    """The command manager's stored cfg and the live term's cfg can be distinct
    copies; the term only reads its own, so both must be updated."""
    cfgs = [env.command_manager.get_term_cfg("twist")]
    term = env.command_manager.get_term("twist")
    if term.cfg is not cfgs[0]:
        cfgs.append(term.cfg)
    return cfgs

def _resample_twist_command(env) -> None:
    term = env.command_manager.get_term("twist")
    term._resample_command(torch.arange(env.num_envs, device=env.device))

def _resample_interval_time_left(env, term_name: str, interval_range: tuple[float, float]) -> None:
    """EventManager keeps a live countdown, so changing ``interval_range_s`` alone
    leaves the previously sampled delay in place."""
    manager = env.event_manager
    names = manager._mode_term_names.get("interval", [])
    if term_name not in names:
        return
    time_left = manager._interval_term_time_left[names.index(term_name)]
    lower, upper = float(interval_range[0]), float(interval_range[1])
    time_left.copy_(torch.rand(time_left.shape, device=time_left.device) * (upper - lower) + lower)

def penalize_stepping_while_standing(
    env: ManagerBasedRlEnv,
    air_time_weight: float,
    no_stepping_penalty_weight: float,
) -> torch.Tensor:
    """
    Updating the air_time and no_stepping reward weights to penalize stepping while standing.
    """
    env.reward_manager.get_term_cfg("air_time").weight = air_time_weight
    env.reward_manager.get_term_cfg("no_stepping").weight = no_stepping_penalty_weight

def stepping_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    air_time_weight: float,
    no_stepping_penalty_weight: float,
    rel_standing_envs: float = 0.0,
    rel_rotation_envs: float = 0.0,
    step: int = 10000 * 24,
) -> dict[str, torch.Tensor]:
    """
    Updating the air_time and no_stepping reward weights to penalize stepping while standing
    after a certain number of iterations.
    """
    del env_ids  # Unused.

    if env.common_step_counter >= step: 
        env.reward_manager.get_term_cfg("air_time").weight = air_time_weight
        env.reward_manager.get_term_cfg("no_stepping").weight = no_stepping_penalty_weight
        env.command_manager.get_term_cfg("twist").rel_standing_envs = rel_standing_envs
        env.command_manager.get_term_cfg("twist").rel_rotation_envs = rel_rotation_envs

    return {
        "air_time_weight": torch.tensor(env.reward_manager.get_term_cfg("air_time").weight),
        "no_stepping_penalty_weight": torch.tensor(env.reward_manager.get_term_cfg("no_stepping").weight),
        "rel_standing_envs": torch.tensor(env.command_manager.get_term_cfg("twist").rel_standing_envs),
        "rel_rotation_envs": torch.tensor(env.command_manager.get_term_cfg("twist").rel_rotation_envs),
    }


############################ GET-UP #############################

def reset_getup(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    standing_fraction: float,
    drop_height: float,
    joint_pos_range: float,
    flat_fraction: float = 0.0,
    supine_fraction: float = 0.5,
    flat_joint_pos_range: float = 0.2,
    flat_drop_height: float = 0.1,
    flat_tilt: float = 0.3,
    xy_range: float = 0.5,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Drop the robot from ``drop_height`` [m above the env origin] in a uniformly random
    orientation with the ``asset_cfg`` joints offset by up to ``joint_pos_range`` [rad]
    (clamped to the soft limits), so it lands in an arbitrary fallen pose. A
    ``standing_fraction`` of the envs start in the ready pose instead.

    A ``flat_fraction`` of the fallen envs lies flat on its back or front, as the robot does
    after a real fall: trunk face up or down tilted by up to ``flat_tilt`` [rad], joints
    within ``flat_joint_pos_range`` of the ready pose, dropped from ``flat_drop_height``.
    Random limbs alone prop the trunk up: only 10-15% of those landings lay flat.
    Other joints and all velocities reset to their defaults."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    asset: Entity = env.scene[asset_cfg.name]
    n = len(env_ids)
    device = env.device
    standing = torch.rand(n, device=device) < standing_fraction
    flat = ~standing & (torch.rand(n, device=device) < flat_fraction)

    root = asset.data.default_root_state[env_ids].clone()
    origins = env.scene.env_origins[env_ids]
    pos = root[:, :3] + origins
    pos[:, :2] += sample_uniform(-xy_range, xy_range, (n, 2), device=device)
    pos[:, 2] = torch.where(standing, pos[:, 2], origins[:, 2] + torch.where(flat, flat_drop_height, drop_height))
    yaw = sample_uniform(-math.pi, math.pi, (n,), device=device)
    zeros = torch.zeros(n, device=device)
    quat_standing = quat_mul(root[:, 3:7], quat_from_euler_xyz(zeros, zeros, yaw))
    quat_fallen = torch.nn.functional.normalize(torch.randn(n, 4, device=device), dim=-1)
    # Pitch -90 deg puts the trunk's front (x) up: on its back; +90 deg on its front.
    face = torch.where(torch.rand(n, device=device) < supine_fraction, -0.5 * math.pi, 0.5 * math.pi)
    tilt = quat_from_euler_xyz(
        sample_uniform(-flat_tilt, flat_tilt, (n,), device=device),
        face + sample_uniform(-flat_tilt, flat_tilt, (n,), device=device),
        zeros,
    )
    quat_flat = quat_mul(quat_from_euler_xyz(zeros, zeros, yaw), quat_mul(tilt, root[:, 3:7]))
    quat = torch.where(standing.unsqueeze(-1), quat_standing, torch.where(flat.unsqueeze(-1), quat_flat, quat_fallen))
    asset.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
    asset.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=device), env_ids=env_ids)

    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    offset = sample_uniform(-joint_pos_range, joint_pos_range, (n, len(asset_cfg.joint_ids)), device=device)
    offset[flat] *= flat_joint_pos_range / joint_pos_range
    offset[standing] = 0.0
    joint_pos[:, asset_cfg.joint_ids] += offset
    limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp(limits[..., 0], limits[..., 1])
    joint_vel = torch.zeros_like(joint_pos)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def _root_height(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]


def _height_fraction(env, min_height: float, stand_height: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return ((_root_height(env, asset_cfg) - min_height) / (stand_height - min_height)).clamp(0.0, 1.0)


def _stand_gate(env, min_height: float, stand_height: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """0 -> 1 as the trunk rises from 80% to 95% of the standing height and comes within
    ~25 -> ~14 deg of vertical."""
    g_z = env.scene[asset_cfg.name].data.projected_gravity_b[:, 2]
    height = ((_height_fraction(env, min_height, stand_height, asset_cfg) - 0.8) / 0.15).clamp(0.0, 1.0)
    tilt = ((-g_z - 0.9) / 0.07).clamp(0.0, 1.0)
    return height * tilt


def getup_assist_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice,
    initial: float,
    step: float,
    success_high: float,
    success_low: float,
    min_height: float,
    stand_height: float,
    start: float | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> dict[str, torch.Tensor]:
    """Assist level (fraction of body weight pulled up by ``getup_assist_force``) driven by
    the share of episodes that end standing: lowered by ``step`` per episode's worth of
    resets above ``success_high``, raised below ``success_low``, within [0, initial].
    Starts at ``start`` (default ``initial``; 0 when fine-tuning a policy that stands up).

    Without it the policy settled in a one-arm side plank: leaving it to shift the weight
    onto the feet mostly ended back on the ground."""
    state = getattr(env, "_getup_assist", None)
    if state is None:
        state = env._getup_assist = {"level": initial if start is None else start, "success": 0.0}
    elif env.common_step_counter > 0:
        stood = _stand_gate(env, min_height, stand_height, asset_cfg)[env_ids] > 0.5
        if stood.numel():
            rate = stood.numel() / env.num_envs
            state["success"] += rate * (stood.float().mean().item() - state["success"])
            if state["success"] > success_high:
                state["level"] -= step * rate
            elif state["success"] < success_low:
                state["level"] += step * rate
            state["level"] = min(max(state["level"], 0.0), initial)
    return {"assist": torch.tensor(state["level"]), "success": torch.tensor(state["success"])}


def getup_assist_force(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
) -> None:
    """Upward pull on the ``asset_cfg`` body for the coming episode, at the curriculum's assist
    level times the robot's weight (zero when no curriculum runs, as in play)."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    asset: Entity = env.scene[asset_cfg.name]
    level = getattr(env, "_getup_assist", {"level": 0.0})["level"]
    model = env.sim.mj_model
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{asset_cfg.name}/trunk")
    weight = float(model.body_subtreemass[root]) * 9.81
    forces = torch.zeros(len(env_ids), len(asset_cfg.body_ids), 3, device=env.device)
    forces[..., 2] = level * weight
    asset.write_external_wrench_to_sim(forces, torch.zeros_like(forces), env_ids=env_ids, body_ids=asset_cfg.body_ids)


def root_height(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Trunk height above the env origin [m] (critic only)."""
    return _root_height(env, asset_cfg).unsqueeze(-1)


def getup_speed_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice,
    limits: dict[str, tuple[float, float]],
    step: float,
    success_high: float,
    success_low: float,
) -> dict[str, torch.Tensor]:
    """Moves the ``max_speed`` of each speed penalty in ``limits`` (name -> (start, end)) from
    start to end with a level in [0, 1]: raised by ``step`` per episode's worth of resets while
    the get-up success tracked by ``getup_assist_curriculum`` (run before this term) is above
    ``success_high`` with no assist, lowered below ``success_low``. Ends at the slowest get-up
    that still works.

    The penalty weights must make the limits binding: scaling small weights up instead only
    slowed the get-up a little, and a small penalty on a low limit is simply paid."""
    state = getattr(env, "_getup_speed", None)
    if state is None:
        state = env._getup_speed = {"level": 0.0}
    elif env.common_step_counter > 0:
        assist = getattr(env, "_getup_assist", {"level": 0.0, "success": 1.0})
        n = len(range(env.num_envs)[env_ids]) if isinstance(env_ids, slice) else env_ids.numel()
        rate = n / env.num_envs
        if assist["success"] > success_high and assist["level"] <= 0.0:
            state["level"] += step * rate
        elif assist["success"] < success_low:
            state["level"] -= 2.0 * step * rate
        state["level"] = min(max(state["level"], 0.0), 1.0)
    for name, (start, end) in limits.items():
        env.reward_manager.get_term_cfg(name).params["max_speed"] = start + (end - start) * state["level"]
    return {"level": torch.tensor(state["level"])}


def _face_down(gx: torch.Tensor) -> torch.Tensor:
    """0 on the back or side (gx <= 0), 1 face down (gx >= 0.7)."""
    return ((gx - 0.0) / 0.7).clamp(0.0, 1.0)


def _upright_lock(gz: torch.Tensor) -> torch.Tensor:
    """0 until the trunk is within ~32 deg of vertical, 1 within ~26 deg. A sit-up from the
    back never reaches this, so it cannot turn the stand rewards back on."""
    return ((-gz - 0.85) / 0.10).clamp(0.0, 1.0)


def _feet_down(env: ManagerBasedRlEnv, feet_cfg: SceneEntityCfg) -> torch.Tensor:
    """1 when both feet are near the ground, 0 when either foot is clearly up."""
    asset: Entity = env.scene[feet_cfg.name]
    z = asset.data.site_pos_w[:, feet_cfg.site_ids, 2] - env.scene.env_origins[:, None, 2]
    return ((0.12 - z) / 0.06).clamp(0.0, 1.0).amin(dim=1)


def _pay_stand(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, feet_cfg: SceneEntityCfg) -> torch.Tensor:
    """1 when the learned front get-up applies: face down with both feet down, or already
    upright. 0 on the back, so sitting up from the back earns none of the stand rewards.

    Gating on both thighs lying flat instead cut off the front get-up, which tucks the
    legs right away (front success 86% -> 0%)."""
    g = env.scene[asset_cfg.name].data.projected_gravity_b
    ready = _face_down(g[:, 0]) * _feet_down(env, feet_cfg)
    return torch.maximum(ready, _upright_lock(g[:, 2]))


def getup_not_ready(
    env: ManagerBasedRlEnv,
    feet_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """1 until the face and both feet point down, or the robot is already upright.

    Penalised, and it does not fade when the chest lifts: the previous face-up penalty
    turned off above 30% of the stand height, so the policy sat up from the back, collected
    the upright and height rewards, and neither rolled nor stood."""
    return 1.0 - _pay_stand(env, asset_cfg, feet_cfg)


def getup_upright(
    env: ManagerBasedRlEnv,
    feet_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """0 upside down, 0.5 lying, 1 upright, paid only once face down with both feet down
    (or already upright), so the front get-up is reused instead of a sit-up from the back."""
    gz = env.scene[asset_cfg.name].data.projected_gravity_b[:, 2]
    return 0.5 * (1.0 - gz) * _pay_stand(env, asset_cfg, feet_cfg)


def getup_height(
    env: ManagerBasedRlEnv,
    min_height: float,
    stand_height: float,
    feet_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Trunk height from lying (0) to standing (1), on the same gate as ``getup_upright``."""
    return _height_fraction(env, min_height, stand_height, asset_cfg) * _pay_stand(env, asset_cfg, feet_cfg)


############################ ROLL-OVER #############################

def rollover_face_down(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """0 face up, 0.5 on a side, 1 face down."""
    return 0.5 * (1.0 + env.scene[asset_cfg.name].data.projected_gravity_b[:, 0])


def rollover_prone_pose(
    env: ManagerBasedRlEnv, std: float, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """Ready-pose match of the ``asset_cfg`` joints, paid only once face down: the get-up
    policy takes over from there and was trained from flat on the front with the joints
    near the ready pose."""
    asset: Entity = env.scene[asset_cfg.name]
    err = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    match = torch.exp(-torch.mean(torch.square(err), dim=1) / std**2)
    return _face_down(asset.data.projected_gravity_b[:, 0]) * match


def rollover_prone_still(
    env: ManagerBasedRlEnv, std: float, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """Low joint and trunk motion, paid only once face down. ``std`` in rad/s."""
    asset: Entity = env.scene[asset_cfg.name]
    joint_vel = torch.mean(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
    trunk_vel = torch.sum(torch.square(asset.data.root_link_ang_vel_b), dim=1)
    return _face_down(asset.data.projected_gravity_b[:, 0]) * torch.exp(-(joint_vel + trunk_vel) / std**2)


def rollover_done(
    env: ManagerBasedRlEnv,
    hold_s: float,
    min_gx: float,
    max_ang_vel: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """True once the robot has lain face down (gravity x above ``min_gx``) with the trunk
    turning slower than ``max_ang_vel`` [rad/s] for ``hold_s`` seconds: the condition on
    which the robot hands over to the get-up policy (see xiaoxiaobai GetupMove)."""
    asset: Entity = env.scene[asset_cfg.name]
    ready = (asset.data.projected_gravity_b[:, 0] > min_gx) & (
        asset.data.root_link_ang_vel_b.norm(dim=-1) < max_ang_vel
    )
    steps = getattr(env, "_rollover_steps", None)
    if steps is None:
        steps = env._rollover_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    steps[env.episode_length_buf <= 1] = 0
    steps[:] = torch.where(ready, steps + 1, 0)
    return steps * env.step_dt >= hold_s


def getup_stand_pose(
    env: ManagerBasedRlEnv,
    std: float,
    min_height: float,
    stand_height: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Ready-pose match of the ``asset_cfg`` joints, paid only once standing."""
    asset: Entity = env.scene[asset_cfg.name]
    err = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    match = torch.exp(-torch.mean(torch.square(err), dim=1) / std**2)
    return _stand_gate(env, min_height, stand_height, asset_cfg) * match


def getup_stand_still(
    env: ManagerBasedRlEnv,
    std: float,
    min_height: float,
    stand_height: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Low joint and trunk motion, paid only once standing. ``std`` in rad/s."""
    asset: Entity = env.scene[asset_cfg.name]
    joint_vel = torch.mean(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
    trunk_vel = torch.sum(torch.square(asset.data.root_link_ang_vel_b), dim=1)
    still = torch.exp(-(joint_vel + trunk_vel) / std**2)
    return _stand_gate(env, min_height, stand_height, asset_cfg) * still


def getup_stood_still(
    env: ManagerBasedRlEnv,
    hold_s: float,
    min_height: float,
    stand_height: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """True once the robot has stood (stand gate > 0.5) for ``hold_s`` seconds without a
    break. Use as a time-out so the value of standing on is bootstrapped.

    No speed threshold: under the exploration noise a standing robot's joints move at
    ~1.4 rad/s RMS and its trunk turns at ~1.2 rad/s (~0.03 for the mean action)."""
    still = _stand_gate(env, min_height, stand_height, asset_cfg) > 0.5
    steps = getattr(env, "_getup_still_steps", None)
    if steps is None:
        steps = env._getup_still_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    steps[env.episode_length_buf <= 1] = 0
    steps[:] = torch.where(still, steps + 1, 0)
    return steps * env.step_dt >= hold_s


def _motor_load(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Motor torque of every BAM actuator over its current-limit torque (max_current * kt):
    1 is the firmware current limit."""
    asset: Entity = env.scene[asset_cfg.name]
    loads = []
    for act in asset.actuators:
        torque = getattr(act, "_prev_motor_torque", None)
        if torque is not None and act.cfg.max_current is not None:
            loads.append(torque.abs() / (act.cfg.max_current * act._bam_model.kt.value))
    return torch.cat(loads, dim=1)


def motor_load_excess(
    env: ManagerBasedRlEnv, max_load: float, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """Sum over motors of the squared share of the current limit used above ``max_load``.

    Each motor is penalised on its own and the square makes one motor at its limit cost far
    more than several sharing the load: the get-up held single motors at the current limit
    (shoulders pushing up from the front, hips while rolling) for up to 13% of the time and
    stood up in sim but not on the robot."""
    return torch.sum(torch.square(torch.clamp(_motor_load(env, asset_cfg) - max_load, min=0.0)), dim=1)


def motor_load_l2(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Sum over motors of the squared share of the current limit."""
    return torch.sum(torch.square(_motor_load(env, asset_cfg)), dim=1)


############################ SIT ON A SEAT #############################

class SitCommand(CommandTerm):
    """1 = sit down on the seat behind and rest, 0 = stand up and stay up.

    Episodes start sitting with probability ``start_sit_prob``; every resample after that
    toggles, so an episode runs sit, stand, sit, ... from whatever state the last phase left."""

    cfg: "SitCommandCfg"

    def __init__(self, cfg: "SitCommandCfg", env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.sit = torch.zeros(self.num_envs, 1, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.sit

    def _update_metrics(self) -> None:
        pass

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        first = self.command_counter[env_ids] == 0
        start = (torch.rand(len(env_ids), device=self.device) < self.cfg.start_sit_prob).float()
        self.sit[env_ids, 0] = torch.where(first, start, 1.0 - self.sit[env_ids, 0])

    def _update_command(self) -> None:
        pass


@dataclass(kw_only=True)
class SitCommandCfg(CommandTermCfg):
    start_sit_prob: float = 0.8

    def build(self, env: ManagerBasedRlEnv) -> SitCommand:
        return SitCommand(self, env)


def _sit_command(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    return env.command_manager.get_command(command_name)[:, 0]


@requires_model_fields("geom_size", "geom_rbound", "geom_aabb")
def reset_on_seat_front(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    half_size_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    gap_range: tuple[float, float],
    lateral_range: tuple[float, float],
    yaw_range: tuple[float, float],
    xy_range: float = 0.5,
    robot_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    seat_cfg: SceneEntityCfg = SceneEntityCfg("seat"),
) -> None:
    """Robot standing in the ready pose (random yaw and position) with a box of random
    half sizes (x depth, y width, z height) [m] behind it: its front face ``gap_range`` [m]
    behind the trunk origin, offset sideways by ``lateral_range`` [m] and turned by
    ``yaw_range`` [rad]. The sizes are kept in ``env._seat_half_size`` for the critic."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    robot: Entity = env.scene[robot_cfg.name]
    seat: Entity = env.scene[seat_cfg.name]
    n = len(env_ids)
    device = env.device

    root = robot.data.default_root_state[env_ids].clone()
    origins = env.scene.env_origins[env_ids]
    pos = root[:, :3] + origins
    pos[:, :2] += sample_uniform(-xy_range, xy_range, (n, 2), device=device)
    yaw = sample_uniform(-math.pi, math.pi, (n,), device=device)
    zeros = torch.zeros(n, device=device)
    quat = quat_mul(quat_from_euler_xyz(zeros, zeros, yaw), root[:, 3:7])
    robot.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
    robot.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=device), env_ids=env_ids)
    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)

    half = torch.stack([sample_uniform(lo, hi, (n,), device=device) for lo, hi in half_size_range], dim=-1)
    geom_ids = torch.as_tensor(_get_entity_indices(seat.indexing, seat_cfg, "geom", False), device=device).long()
    env.sim.model.geom_size[env_ids.long()[:, None], geom_ids[None, :]] = half[:, None, :]
    _recompute_geom_bounds(env, env_ids.to(device, dtype=torch.int), seat_cfg)
    store = env.__dict__.setdefault("_seat_half_size", torch.zeros(env.num_envs, 3, device=device))
    store[env_ids] = half

    gap = sample_uniform(*gap_range, (n,), device=device)
    env.__dict__.setdefault("_seat_gap", torch.zeros(env.num_envs, device=device))[env_ids] = gap
    back = gap + half[:, 0]
    side = sample_uniform(*lateral_range, (n,), device=device)
    seat_pos = pos.clone()
    seat_pos[:, 0] += -back * torch.cos(yaw) - side * torch.sin(yaw)
    seat_pos[:, 1] += -back * torch.sin(yaw) + side * torch.cos(yaw)
    seat_pos[:, 2] = origins[:, 2] + half[:, 2] + 0.002
    seat_yaw = yaw + sample_uniform(*yaw_range, (n,), device=device)
    seat_quat = quat_from_euler_xyz(zeros, zeros, seat_yaw)
    seat.write_root_link_pose_to_sim(torch.cat([seat_pos, seat_quat], dim=-1), env_ids=env_ids)
    seat.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=device), env_ids=env_ids)


def seat_size(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Seat half sizes (depth, width, height) [m] (critic only)."""
    return env.__dict__.get("_seat_half_size", torch.zeros(env.num_envs, 3, device=env.device))


def seat_pos(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG, seat_cfg: SceneEntityCfg = SceneEntityCfg("seat")
) -> torch.Tensor:
    """Seat centre relative to the trunk, in the trunk's heading frame [m] (critic only)."""
    robot: Entity = env.scene[asset_cfg.name]
    rel = env.scene[seat_cfg.name].data.root_link_pos_w - robot.data.root_link_pos_w
    return quat_apply_inverse(yaw_quat(robot.data.root_link_quat_w), rel)


def _seat_support(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """Share of the robot's weight the seat carries through the bodies the sensor watches."""
    weight = env.__dict__.get("_robot_weight")
    if weight is None:
        model = env.sim.mj_model
        trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot/trunk")
        weight = env._robot_weight = float(model.body_subtreemass[trunk]) * 9.81
    force = env.scene[sensor_name].data.force
    return (force[..., 2].abs().sum(dim=1) / weight).clamp(0.0, 1.0)


def seat_support(env: ManagerBasedRlEnv, command_name: str, sensor_name: str) -> torch.Tensor:
    """Share of the weight on the seat (0..1), while sitting is commanded."""
    return _sit_command(env, command_name) * _seat_support(env, sensor_name)


def seated_rest(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    current_scale: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """exp(-summed motor current / ``current_scale`` [A]), weighted by the share of the weight
    on the seat, while sitting is commanded: the seat carries the robot, the motors relax."""
    current = total_current_excess(env, max_current=0.0, asset_cfg=asset_cfg)
    return _sit_command(env, command_name) * _seat_support(env, sensor_name) * torch.exp(-current / current_scale)


def seated_still(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    joint_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Low joint and trunk motion once on the seat, while sitting is commanded. ``std`` in rad/s."""
    asset: Entity = env.scene[asset_cfg.name]
    joint_vel = torch.mean(torch.square(asset.data.joint_vel[:, joint_cfg.joint_ids]), dim=1)
    trunk_vel = torch.sum(torch.square(asset.data.root_link_ang_vel_b), dim=1)
    still = torch.exp(-(joint_vel + trunk_vel) / std**2)
    return _sit_command(env, command_name) * _seat_support(env, sensor_name) * still


def trunk_upright(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """1 with the trunk vertical, 0 at 60 deg or more from it."""
    gz = env.scene[asset_cfg.name].data.projected_gravity_b[:, 2]
    return ((-gz - 0.5) / 0.5).clamp(0.0, 1.0)


def stand_height(
    env: ManagerBasedRlEnv,
    command_name: str,
    min_height: float,
    stand_height: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Trunk height from ``min_height`` (0) to ``stand_height`` (1), while standing is commanded."""
    return (1.0 - _sit_command(env, command_name)) * _height_fraction(env, min_height, stand_height, asset_cfg)


def commanded_stand_pose(env: ManagerBasedRlEnv, command_name: str, **params) -> torch.Tensor:
    """``getup_stand_pose`` while standing is commanded."""
    return (1.0 - _sit_command(env, command_name)) * getup_stand_pose(env, **params)


def commanded_stand_still(env: ManagerBasedRlEnv, command_name: str, **params) -> torch.Tensor:
    """``getup_stand_still`` while standing is commanded."""
    return (1.0 - _sit_command(env, command_name)) * getup_stand_still(env, **params)


def _seat_reached(env: ManagerBasedRlEnv, command_name: str, sensor_names: tuple[str, ...]) -> torch.Tensor:
    """True from the first touch of the seat (any of the ``sensor_names`` contact sensors) in
    the current sit phase; cleared while standing is commanded and at episode start. Updated
    once per step."""
    state = env.__dict__.setdefault("_seat_reached_state", {"step": -1, "reached": None})
    if state["step"] != env.common_step_counter or state["reached"] is None:
        reached = state["reached"]
        if reached is None:
            reached = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        touch = torch.zeros_like(reached)
        for name in sensor_names:
            touch |= (env.scene[name].data.found > 0).reshape(env.num_envs, -1).any(dim=1)
        sitting = _sit_command(env, command_name) > 0.5
        reached = (reached | touch) & sitting & (env.episode_length_buf > 1)
        state.update(step=env.common_step_counter, reached=reached)
    return state["reached"]


def back_up(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_names: tuple[str, ...],
    speed: float,
    std: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Backward walking at ``speed`` [m/s] (heading frame, std ``std``) while sitting is
    commanded and the legs have not yet touched the seat."""
    robot: Entity = env.scene[asset_cfg.name]
    vx = quat_apply_inverse(yaw_quat(robot.data.root_link_quat_w), robot.data.root_link_lin_vel_w)[:, 0]
    searching = (_sit_command(env, command_name) > 0.5) & ~_seat_reached(env, command_name, sensor_names)
    return searching * torch.exp(-torch.square(vx + speed) / std**2)


def backing_after_touch(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_names: tuple[str, ...],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Backward trunk speed [m/s] after the seat was touched: shoving the box away."""
    robot: Entity = env.scene[asset_cfg.name]
    vx = quat_apply_inverse(yaw_quat(robot.data.root_link_quat_w), robot.data.root_link_lin_vel_w)[:, 0]
    return _seat_reached(env, command_name, sensor_names) * torch.clamp(-vx, min=0.0)


def feet_slip(
    env: ManagerBasedRlEnv, sensor_name: str, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """Sum over the feet on the ground of their horizontal speed [m/s]: shuffling instead of
    stepping."""
    contact = env.scene[sensor_name].data.found > 0
    vel = env.scene[asset_cfg.name].data.site_lin_vel_w[:, asset_cfg.site_ids, :2].norm(dim=-1)
    return (vel * contact).sum(dim=1)


def seat_speed(env: ManagerBasedRlEnv, seat_cfg: SceneEntityCfg = SceneEntityCfg("seat")) -> torch.Tensor:
    """Horizontal speed of the seat [m/s]: pushed or kicked away."""
    return env.scene[seat_cfg.name].data.root_link_lin_vel_w[:, :2].norm(dim=-1)


def fell(
    env: ManagerBasedRlEnv, min_height: float, max_tilt: float, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """Trunk below ``min_height`` [m] or more than ``max_tilt`` [rad] from vertical: fell off
    or past the seat."""
    asset: Entity = env.scene[asset_cfg.name]
    tilted = asset.data.projected_gravity_b[:, 2] > -math.cos(max_tilt)
    return (_root_height(env, asset_cfg) < min_height) | tilted


############################ DRAG #############################
# Walking backward with a load taped to both hands (xiaoxiaobai_drag_env_cfg.py). Every env
# keeps a load level, a mass from DRAG_MASS_LEVELS: up after an episode that kept up with the
# command, down after a fall or one that fell well short.


# set_const: with the load's body_invweight0 left at another mass the tape welds blow up.
@requires_model_fields("body_mass", "body_inertia", "geom_friction", recompute=RecomputeLevel.set_const)
def reset_drag(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    load_offset: tuple[float, float, float],
    unit_inertia: tuple[float, float, float],
    mass_levels: tuple[float, ...],
    friction_range: tuple[float, float],
    initial_levels: int,
    light_fraction: float = 0.0,
    free_fraction: float = 0.0,
    xy_range: float = 0.5,
    robot_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    load_cfg: SceneEntityCfg = SceneEntityCfg("load", body_names=("load",), geom_names=("load_sled",)),
) -> None:
    """Robot in its default pose (random yaw and position) and the load in front of it at
    ``load_offset`` [m] from the root, in the robot's heading frame, where the hands meet its
    grips. The load weighs its level's mass (inertia ``unit_inertia`` per kg) and slides at a
    friction coefficient from ``friction_range``. A ``light_fraction`` of the episodes takes a
    level drawn from 0 to the env's own instead, and a ``free_fraction`` leaves the hands
    untaped (every equality inactive); both leave the level as it is."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    robot: Entity = env.scene[robot_cfg.name]
    load: Entity = env.scene[load_cfg.name]
    n = len(env_ids)
    device = env.device

    root = robot.data.default_root_state[env_ids].clone()
    pos = root[:, :3] + env.scene.env_origins[env_ids]
    pos[:, :2] += sample_uniform(-xy_range, xy_range, (n, 2), device=device)
    yaw = sample_uniform(-math.pi, math.pi, (n,), device=device)
    zeros = torch.zeros(n, device=device)
    heading = quat_from_euler_xyz(zeros, zeros, yaw)
    robot.write_root_link_pose_to_sim(torch.cat([pos, quat_mul(heading, root[:, 3:7])], dim=-1), env_ids=env_ids)
    robot.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=device), env_ids=env_ids)
    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)

    offset = torch.tensor(load_offset, device=device).expand(n, 3)
    load.write_root_link_pose_to_sim(torch.cat([pos + quat_apply(heading, offset), heading], dim=-1), env_ids=env_ids)
    load.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=device), env_ids=env_ids)

    levels = env.__dict__.get("_drag_level")
    if levels is None:
        levels = torch.randint(0, initial_levels, (env.num_envs,), device=device)
        env._drag_level = levels
    ids = env_ids.long()
    level = levels[ids]
    draw = torch.rand(n, device=device)
    light = draw < light_fraction
    free = (draw >= light_fraction) & (draw < light_fraction + free_fraction)
    drawn = (torch.rand(n, device=device) * (level + 1).float()).long()
    masses = torch.tensor(mass_levels, device=device)
    mass = masses[torch.where(light, drawn, level)]
    env.__dict__.setdefault("_drag_light", torch.zeros(env.num_envs, dtype=torch.bool, device=device))[ids] = light | free
    env.sim.data.eq_active[ids] = ~free[:, None]
    body = torch.as_tensor(_get_entity_indices(load.indexing, load_cfg, "body", False), device=device).long()
    geom = torch.as_tensor(_get_entity_indices(load.indexing, load_cfg, "geom", False), device=device).long()
    env.sim.model.body_mass[ids[:, None], body[None, :]] = mass[:, None]
    env.sim.model.body_inertia[ids[:, None], body[None, :]] = mass[:, None, None] * torch.tensor(unit_inertia, device=device)
    env.sim.model.geom_friction[ids[:, None], geom[None, :], 0] = sample_uniform(*friction_range, (n, 1), device=device)

    env.__dict__.setdefault("_drag_start", torch.zeros(env.num_envs, 2, device=device))[ids] = pos[:, :2]
    env.__dict__.setdefault("_drag_yaw", torch.zeros(env.num_envs, device=device))[ids] = yaw
    env.__dict__.setdefault("_drag_mass", torch.zeros(env.num_envs, device=device))[ids] = mass


def drag_arm_authority(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice,
    anneal_steps: int,
    action_name: str = "joint_pos",
    joint_pattern: str = r".*(shoulder|elbow).*",
) -> dict[str, torch.Tensor]:
    """Scales the arm actions from 1 down to 0 over ``anneal_steps`` env steps from the first
    step (a resumed run's step counter jumps to the checkpoint's after the first reset): a
    policy that pulled with its arms learns to do without them."""
    del env_ids  # Unused.
    term = env.action_manager.get_term(action_name)
    cols = [i for i, name in enumerate(term._target_names) if re.fullmatch(joint_pattern, name)]
    if env.common_step_counter == 0:
        scale = 1.0
    else:
        start = env.__dict__.setdefault("_drag_arm_anneal_start", env.common_step_counter)
        scale = max(0.0, 1.0 - (env.common_step_counter - start) / anneal_steps)
    term._scale[:, cols] = scale
    return {"arm_scale": torch.tensor(scale)}


def drag_load_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice,
    num_levels: int,
    min_distance: float = 0.3,
    up_ratio: float = 0.7,
    down_ratio: float = 0.4,
    start_s: float = 1.0,
    command_name: str = "twist",
    robot_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> dict[str, torch.Tensor]:
    """Moves each ending episode's load level: up if it covered ``up_ratio`` of the commanded
    backward distance, down if it fell or covered less than ``down_ratio``. The distance is
    the straight line from the start, signed by its direction against the starting heading
    (with the wz = 0 command the robot still turns ~20 deg in 10 s), and the command's is
    counted from ``start_s`` into the episode. Episodes commanded under ``min_distance`` [m]
    (standing) move down only on a fall."""
    levels = env.__dict__.get("_drag_level")
    if levels is None or "_drag_start" not in env.__dict__:
        return {}
    if isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)
    ids = env_ids.long()
    robot: Entity = env.scene[robot_cfg.name]
    moved = robot.data.root_link_pos_w[ids, :2] - env._drag_start[ids]
    yaw = env._drag_yaw[ids]
    along = -(moved[:, 0] * torch.cos(yaw) + moved[:, 1] * torch.sin(yaw))
    back = moved.norm(dim=-1) * torch.sign(along)
    vx = env.command_manager.get_command(command_name)[ids, 0]
    want = -vx * (env.episode_length_buf[ids] * env.step_dt - start_s).clamp(min=0.0)
    fell = env.termination_manager.terminated[ids]
    walking = want > min_distance
    up = walking & ~fell & (back >= up_ratio * want)
    down = fell | (walking & (back < down_ratio * want))
    light = env.__dict__.get("_drag_light")
    if light is not None:
        up, down = up & ~light[ids], down & ~light[ids]
    levels[ids] = (levels[ids] + up.long() - down.long()).clamp(0, num_levels - 1)
    mass = env._drag_mass
    return {
        "mean_kg": mass.mean(),
        "p90_kg": torch.quantile(mass, 0.9),
        "max_level": levels.max().float(),
    }


############################ SIT UP FROM LYING #############################
# Command 1 = sit up from lying on the back, propped on both hands; 0 = lie back down and
# rest. With the legs flat the hips (+-80 deg soft) cannot bring the trunk upright, so the
# seat leans back on the hands.


def _sit_up_level(env: ManagerBasedRlEnv, sit_tilt: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """0 lying on the back, 1 with the trunk within ``sit_tilt`` [rad] of vertical. Zero once
    the trunk leans forward past vertical: the seat is behind the hips, not over the feet."""
    g = env.scene[asset_cfg.name].data.projected_gravity_b
    return (-g[:, 2] / math.cos(sit_tilt)).clamp(0.0, 1.0) * (g[:, 0] < 0.1)


def _lie_level(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """1 flat on the back, 0 upright, linear in the cosine of the tilt; zero unless face up."""
    g = env.scene[asset_cfg.name].data.projected_gravity_b
    return (1.0 + g[:, 2]).clamp(0.0, 1.0) * (g[:, 0] < -0.5)


def _hands_down(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """Share of the hand bodies the sensor watches that touch the ground (0, 0.5, 1)."""
    return (env.scene[sensor_name].data.found > 0).float().reshape(env.num_envs, -1).mean(dim=1)


def _still(env: ManagerBasedRlEnv, std: float, joint_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Entity = env.scene[joint_cfg.name]
    joint_vel = torch.mean(torch.square(asset.data.joint_vel[:, joint_cfg.joint_ids]), dim=1)
    trunk_vel = torch.sum(torch.square(asset.data.root_link_ang_vel_b), dim=1)
    return torch.exp(-(joint_vel + trunk_vel) / std**2)


def _sat(env: ManagerBasedRlEnv, command_name: str, sit_tilt: float, sensor_name: str, asset_cfg: SceneEntityCfg):
    """1 while sitting is commanded with the trunk up and both hands on the ground."""
    up = _sit_up_level(env, sit_tilt, asset_cfg) > 0.95
    return _sit_command(env, command_name) * (up & (_hands_down(env, sensor_name) > 0.99))


def _lain(env: ManagerBasedRlEnv, command_name: str, asset_cfg: SceneEntityCfg):
    """1 while lying is commanded and the trunk is within ~25 deg of flat on the back."""
    return (1.0 - _sit_command(env, command_name)) * (_lie_level(env, asset_cfg) > 0.9)


def sit_up_level(
    env: ManagerBasedRlEnv, command_name: str, sit_tilt: float, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """Trunk raised from lying (0) to within ``sit_tilt`` of vertical (1), while sitting is commanded."""
    return _sit_command(env, command_name) * _sit_up_level(env, sit_tilt, asset_cfg)


def sit_up_hands(
    env: ManagerBasedRlEnv,
    command_name: str,
    sit_tilt: float,
    sensor_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Hands on the ground, times how far the trunk is up, while sitting is commanded: the
    trunk is pushed up on the hands rather than curled up on the abdomen."""
    return _sit_command(env, command_name) * _sit_up_level(env, sit_tilt, asset_cfg) * _hands_down(env, sensor_name)


def sat_rest(
    env: ManagerBasedRlEnv,
    command_name: str,
    sit_tilt: float,
    sensor_name: str,
    current_scale: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """exp(-summed motor current / ``current_scale`` [A]) once sitting on both hands."""
    current = total_current_excess(env, max_current=0.0, asset_cfg=asset_cfg)
    return _sat(env, command_name, sit_tilt, sensor_name, asset_cfg) * torch.exp(-current / current_scale)


def sat_still(
    env: ManagerBasedRlEnv,
    command_name: str,
    sit_tilt: float,
    sensor_name: str,
    std: float,
    joint_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Low joint and trunk motion [rad/s, ``std``] once sitting on both hands."""
    return _sat(env, command_name, sit_tilt, sensor_name, asset_cfg) * _still(env, std, joint_cfg)


def lie_level(env: ManagerBasedRlEnv, command_name: str, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Trunk lowered from upright (0) to flat on the back (1), while lying is commanded."""
    return (1.0 - _sit_command(env, command_name)) * _lie_level(env, asset_cfg)


def lain_pose(
    env: ManagerBasedRlEnv, command_name: str, std: float, joint_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Ready-pose match (arms along the sides, legs straight) once lying flat."""
    asset: Entity = env.scene[joint_cfg.name]
    err = asset.data.joint_pos[:, joint_cfg.joint_ids] - asset.data.default_joint_pos[:, joint_cfg.joint_ids]
    match = torch.exp(-torch.mean(torch.square(err), dim=1) / std**2)
    return _lain(env, command_name, asset_cfg) * match


def lain_rest(
    env: ManagerBasedRlEnv, command_name: str, current_scale: float, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """exp(-summed motor current / ``current_scale`` [A]) once lying flat."""
    current = total_current_excess(env, max_current=0.0, asset_cfg=asset_cfg)
    return _lain(env, command_name, asset_cfg) * torch.exp(-current / current_scale)


def lain_still(
    env: ManagerBasedRlEnv, command_name: str, std: float, joint_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Low joint and trunk motion [rad/s, ``std``] once lying flat."""
    return _lain(env, command_name, asset_cfg) * _still(env, std, joint_cfg)


def feet_lifted(env: ManagerBasedRlEnv, min_height: float, span: float, feet_cfg: SceneEntityCfg) -> torch.Tensor:
    """Sum over the feet of how far (0..1 over ``span`` [m]) the foot site is above ``min_height``
    [m]: lying and sitting keep the legs on the floor (sites at ~0.055 m)."""
    asset: Entity = env.scene[feet_cfg.name]
    z = asset.data.site_pos_w[:, feet_cfg.site_ids, 2] - env.scene.env_origins[:, None, 2]
    return ((z - min_height) / span).clamp(0.0, 1.0).sum(dim=1)


def joint_deviation_l2(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """Sum of squared offsets [rad^2] of the ``asset_cfg`` joints from the ready pose."""
    asset: Entity = env.scene[asset_cfg.name]
    err = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(err), dim=1)


def left_the_floor(
    env: ManagerBasedRlEnv,
    max_gx: float,
    max_gy: float,
    max_height: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Rolled face down (gravity x above ``max_gx``), onto a side (|gravity y| above ``max_gy``)
    or got up (trunk above ``max_height`` [m]): neither lying nor sitting on the floor."""
    asset: Entity = env.scene[asset_cfg.name]
    g = asset.data.projected_gravity_b
    return (g[:, 0] > max_gx) | (g[:, 1].abs() > max_gy) | (_root_height(env, asset_cfg) > max_height)


def skate_side_slip(
    env: ManagerBasedRlEnv, sensor_name: str, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """Squared sideways speed [m/s] of each skate on the ground, in the skate's own frame.

    The wheels roll only along the skate; a push has to come from a skate gripping sideways,
    not from scraping it across the floor."""
    asset: Entity = env.scene[asset_cfg.name]
    found = env.scene.sensors[sensor_name].data.found
    if found.dim() == 3:
        found = found.any(dim=-1)
    quat = asset.data.site_quat_w[:, asset_cfg.site_ids]
    vel = asset.data.site_lin_vel_w[:, asset_cfg.site_ids]
    side = quat_apply_inverse(quat.reshape(-1, 4), vel.reshape(-1, 3)).reshape(vel.shape)[..., 1]
    in_contact = (found > 0).float()
    env.extras["log"]["Metrics/skate_side_slip_mean"] = torch.sum(side.abs() * in_contact) / torch.clamp(
        torch.sum(in_contact), min=1
    )
    return torch.sum(torch.square(side) * in_contact, dim=1)


def single_skate_glide(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    max_air_time: float,
    body_weight: float,
    push_threshold: float,
    command_name: str = "twist",
    command_threshold: float = 0.01,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """1 while exactly one skate is on the ground under a moving command: the gliding skate
    carries the robot while the other has pushed off and hangs in the air. Both skates down
    count too while one pushes forward by more than ``push_threshold`` of the body weight:
    the push happens beside the gliding skate. Paid for single stance alone, a policy kept
    one skate up 97% of the time and pushed in only 5-9% of it.

    Only until the lifted skate has been up ``max_air_time``: a policy paid for any single
    stance glided on one leg 98% of the time and never swapped."""
    sensor = env.scene.sensors[sensor_name]
    found = sensor.data.found
    if found.dim() == 3:
        found = found.any(dim=-1)
    down = found > 0
    swapping = torch.all(down | (sensor.data.current_air_time < max_air_time), dim=-1)
    forward = -_in_heading_frame(env, sensor.data.force, asset_cfg)[..., 0] / body_weight
    pushing = torch.any(down & (forward > push_threshold), dim=-1) & (down.sum(dim=-1) == 2)
    stroke = (down.sum(dim=-1) == 1) | pushing
    return stroke.float() * swapping.float() * _moving(env, command_name, command_threshold)


def _moving(env: ManagerBasedRlEnv, command_name: str, command_threshold: float) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    return ((torch.norm(command[:, :2], dim=-1) + torch.abs(command[:, 2])) > command_threshold).float()


def _speed_deficit(env: ManagerBasedRlEnv, command_name: str, band: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """1 while the trunk's forward speed is ``band`` or more below the command, falling to 0
    at ``band`` above it: pushing pays only while the robot still needs speed."""
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    return torch.clamp((command[:, 0] - asset.data.root_link_lin_vel_b[:, 0] + band) / (2 * band), 0.0, 1.0)


def _in_heading_frame(env: ManagerBasedRlEnv, vec_w: torch.Tensor, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    q = yaw_quat(asset.data.root_link_quat_w).unsqueeze(1).expand(-1, vec_w.shape[1], -1)
    return quat_apply_inverse(q.reshape(-1, 4), vec_w.reshape(-1, 3)).reshape(vec_w.shape)


def joint_opening_excess(
    env: ManagerBasedRlEnv,
    max_opening: float,
    signs: tuple[float, ...],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Sum of squared angles [rad] by which joints open past ``max_opening``, each joint's
    opening direction given by its sign in ``signs`` (in ``asset_cfg`` joint order)."""
    asset: Entity = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    opening = q * torch.tensor(signs, device=q.device, dtype=q.dtype)
    return torch.sum(torch.square(torch.clamp(opening - max_opening, min=0.0)), dim=-1)


def joint_target_l2(
    env: ManagerBasedRlEnv,
    targets: tuple[float, ...],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Sum of squared deviations [rad] of joints from ``targets`` (in ``asset_cfg`` joint order)."""
    asset: Entity = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(q - torch.tensor(targets, device=q.device, dtype=q.dtype)), dim=-1)


def motion_joint_command(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reference joint positions (relative to the default pose) and velocities of the motion
    command, for the ``asset_cfg`` joints only (the passive joints have no reference)."""
    command = env.command_manager.get_term(command_name)
    asset: Entity = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    return torch.cat(
        [command.joint_pos[:, ids] - asset.data.default_joint_pos[:, ids], command.joint_vel[:, ids]], dim=-1
    )


# Dance feet: (motion body, foot site, contact sensor index, height scan index), right then left.
_DANCE_FEET = (("foot", "right_foot", 0, 1), ("foot_2", "left_foot", 1, 0))


def _dance_feet_lift(env: ManagerBasedRlEnv, command_name: str, height_sensor_name: str):
    """Lift [m] of each foot (right, left) over the ground: the robot's and the reference's."""
    command = env.command_manager.get_term(command_name)
    asset: Entity = env.scene[command.cfg.entity_name]
    bodies = [command.cfg.body_names.index(f[0]) for f in _DANCE_FEET]
    if not hasattr(command, "_feet_planted_z"):
        # Origin height of each foot standing on the floor in the reference.
        command._feet_planted_z = torch.quantile(command.motion.body_pos_w[:, bodies, 2], 0.05, dim=0)
    sites = [asset.site_names.index(f[1]) for f in _DANCE_FEET]
    scan = env.scene[height_sensor_name].data.heights[:, [f[3] for f in _DANCE_FEET]]
    ground = asset.data.site_pos_w[:, sites, 2] - scan
    robot = command.robot_body_pos_w[:, bodies, 2] - ground - command._feet_planted_z
    ref = command.body_pos_w[:, bodies, 2] - env.scene.env_origins[:, None, 2] - command._feet_planted_z
    return robot, ref


def motion_feet_height_error(
    env: ManagerBasedRlEnv, command_name: str, height_sensor_name: str, scale: float, max_error: float = 2.0
) -> torch.Tensor:
    """Sum over the feet of ((robot lift - reference lift) / scale)^2, each capped at max_error^2."""
    robot, ref = _dance_feet_lift(env, command_name, height_sensor_name)
    return torch.sum(torch.clamp(torch.abs(robot - ref) / scale, max=max_error) ** 2, dim=-1)


def motion_feet_contact_mismatch(
    env: ManagerBasedRlEnv, command_name: str, sensor_name: str, height_sensor_name: str, air_height: float
) -> torch.Tensor:
    """Feet on the ground while the reference lifts them above air_height, or off it while the
    reference plants them."""
    _, ref = _dance_feet_lift(env, command_name, height_sensor_name)
    found = env.scene[sensor_name].data.found
    down = (found.any(-1) if found.dim() == 3 else found > 0)[:, [f[2] for f in _DANCE_FEET]]
    return torch.sum(down != (ref < air_height), dim=-1).float()


def commanded_forward_speed(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    command_threshold: float = 0.01,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Forward speed clipped to the command [m/s].

    Each m/s short of the command pays the same, so the first 0.1 m/s of a max command is
    worth as much as the last. The narrow tracking kernel is flat there: standing and
    creeping both score nothing, and the policy stood still at 0.5 m/s."""
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    speed = asset.data.root_link_lin_vel_b[:, 0].clamp(min=0.0)
    return torch.minimum(speed, command[:, 0].clamp(min=0.0)) * _moving(env, command_name, command_threshold)


def skate_propulsion(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    body_weight: float,
    max_push: float,
    speed_band: float,
    brake_scale: float = 2.0,
    command_name: str = "twist",
    command_threshold: float = 0.01,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Forward push of the floor on each skate over the body weight (up to ``max_push``),
    paid while the robot is below its commanded speed, with backward (braking) force counted
    ``brake_scale`` times against it. Braking above the commanded speed is free: on tile it
    is the only way to slow down.

    A skate takes force only across its wheels, so pushing means setting the skate at an
    angle to the direction of travel; a skate pushed straight back just rolls. Uncapped and
    paid at any speed (weight 20), it was farmed by stomping hard enough to fall 4.8 times per
    episode."""
    # The sensor reports the robot's force on the floor.
    forward = -_in_heading_frame(env, env.scene.sensors[sensor_name].data.force, asset_cfg)[..., 0] / body_weight
    deficit = _speed_deficit(env, command_name, speed_band, asset_cfg)
    push = torch.clamp(forward, min=0.0, max=max_push).sum(dim=-1) * deficit
    brake = brake_scale * torch.clamp(-forward, min=0.0).sum(dim=-1) * deficit
    return (push - brake) * _moving(env, command_name, command_threshold)


def skate_push_power(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    body_weight: float,
    max_push: float,
    speed_band: float,
    command_name: str = "twist",
    command_threshold: float = 0.01,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Forward power the legs put into the trunk through the skates on the ground: the floor's
    forward push [body weights, up to ``max_push``] times how fast the skate moves back
    relative to the trunk [m/s], paid while the robot is below its commanded speed. A stroke
    sets the skate down near the trunk and drives it back and out under load; a skate that
    only carries the robot moves with the trunk and earns nothing. A policy without it set
    skates down 15 mm behind the trunk and lifted them 21 mm ahead.

    Only forward pushes count: a braking skate sliding forward also multiplies to a positive
    product."""
    asset: Entity = env.scene[asset_cfg.name]
    sensor = env.scene.sensors[sensor_name]
    found = sensor.data.found
    if found.dim() == 3:
        found = found.any(dim=-1)
    forward = -_in_heading_frame(env, sensor.data.force, asset_cfg)[..., 0] / body_weight
    back = asset.data.root_link_lin_vel_w.unsqueeze(1) - asset.data.site_lin_vel_w[:, asset_cfg.site_ids]
    back = _in_heading_frame(env, back, asset_cfg)[..., 0]
    power = (found > 0).float() * torch.clamp(forward, min=0.0, max=max_push) * back
    return (
        power.sum(dim=-1)
        * _speed_deficit(env, command_name, speed_band, asset_cfg)
        * _moving(env, command_name, command_threshold)
    )


def skate_liftoff_behind(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    max_dist: float,
    min_stance: float = 0.15,
    command_name: str = "twist",
    command_threshold: float = 0.01,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Distance [m] behind the trunk (heading frame, up to ``max_dist``) at which a skate
    lifts off: the end of a stroke that pushed it back. Only stances of at least
    ``min_stance`` s count; otherwise tapping a skate behind the trunk pays every tap."""
    asset: Entity = env.scene[asset_cfg.name]
    sensor = env.scene.sensors[sensor_name]
    first_air = sensor.compute_first_air(dt=env.step_dt).float()
    first_air = first_air * (sensor.data.last_contact_time >= min_stance).float()
    rel = asset.data.site_pos_w[:, asset_cfg.site_ids] - asset.data.root_link_pos_w.unsqueeze(1)
    behind = -_in_heading_frame(env, rel, asset_cfg)[..., 0]
    return torch.sum(first_air * torch.clamp(behind, min=0.0, max=max_dist), dim=-1) * _moving(
        env, command_name, command_threshold
    )


def skate_glide_misalignment(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    min_speed: float = 0.1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """sin^2 of the angle between a skate gliding alone and the trunk's horizontal velocity.

    A skate rolls only where it points, so a gliding skate turned off the line of travel
    curves the robot that way. A policy held every skate ~10 deg out through the whole
    stride and zig-zagged one curve per stroke. The pushing skate and the lifted one are free."""
    asset: Entity = env.scene[asset_cfg.name]
    sensor = env.scene.sensors[sensor_name]
    found = sensor.data.found
    if found.dim() == 3:
        found = found.any(dim=-1)
    down = found > 0
    alone = down & ~down.flip(-1)
    vel = asset.data.root_link_lin_vel_w[:, :2]
    speed = torch.norm(vel, dim=-1, keepdim=True)
    direction = vel / speed.clamp(min=1e-6)
    x_axis = torch.zeros_like(asset.data.site_pos_w[:, asset_cfg.site_ids])
    x_axis[..., 0] = 1.0
    nose = quat_apply(asset.data.site_quat_w[:, asset_cfg.site_ids].reshape(-1, 4), x_axis.reshape(-1, 3))
    nose = nose.reshape(x_axis.shape)[..., :2]
    nose = nose / torch.norm(nose, dim=-1, keepdim=True).clamp(min=1e-6)
    sin = nose[..., 0] * direction[:, None, 1] - nose[..., 1] * direction[:, None, 0]
    return torch.sum(alone.float() * torch.square(sin), dim=-1) * (speed[:, 0] > min_speed).float()


def skate_touchdown_ahead(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    margin: float = 0.0,
    command_name: str = "twist",
    command_threshold: float = 0.01,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Distance [m] by which a skate sets down ahead of the trunk (heading frame) beyond
    ``margin``. A walking step lands ahead and brakes; a skating push starts under or
    behind the body."""
    asset: Entity = env.scene[asset_cfg.name]
    sensor = env.scene.sensors[sensor_name]
    first = sensor.compute_first_contact(dt=env.step_dt).float()
    rel = asset.data.site_pos_w[:, asset_cfg.site_ids] - asset.data.root_link_pos_w.unsqueeze(1)
    ahead = _in_heading_frame(env, rel, asset_cfg)[..., 0]
    return torch.sum(first * torch.clamp(ahead - margin, min=0.0), dim=-1) * _moving(env, command_name, command_threshold)


def skate_lift_forward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    lift_time: float,
    command_name: str = "twist",
    command_threshold: float = 0.01,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Squared forward speed [m/s] of a skate relative to the trunk during the first
    ``lift_time`` of its swing: the lifted skate goes straight up, then back to push."""
    asset: Entity = env.scene[asset_cfg.name]
    sensor = env.scene.sensors[sensor_name]
    found = sensor.data.found
    if found.dim() == 3:
        found = found.any(dim=-1)
    lifting = ((found == 0) & (sensor.data.current_air_time < lift_time)).float()
    rel = asset.data.site_lin_vel_w[:, asset_cfg.site_ids] - asset.data.root_link_lin_vel_w.unsqueeze(1)
    forward = _in_heading_frame(env, rel, asset_cfg)[..., 0]
    return torch.sum(lifting * torch.square(torch.clamp(forward, min=0.0)), dim=-1) * _moving(
        env, command_name, command_threshold
    )


def gait_clock_steps(env: ManagerBasedRlEnv, command_name: str = "twist") -> torch.Tensor:
    """Policy ticks since the command last left standing, as the robot's WalkMove counts its
    reference phase: + the stroke rate (1 without a StrokeRateAction) each tick the mean
    |command| is over 0.01, back to 0 otherwise and on reset. Advanced once per env step,
    whichever term asks first."""
    clock = getattr(env, "_gait_clock", None)
    if clock is None:
        clock = {"steps": torch.zeros(env.num_envs, device=env.device), "counter": None}
        env._gait_clock = clock
    if clock["counter"] != env.common_step_counter:
        clock["counter"] = env.common_step_counter
        moving = env.command_manager.get_command(command_name).abs().mean(dim=-1) > 0.01
        rate = env.__dict__.get("_stroke_rate", 1.0)
        clock["steps"] = torch.where(moving, clock["steps"] + rate, 0.0)
    clock["steps"] = torch.where(env.episode_length_buf == 0, 0.0, clock["steps"])
    return clock["steps"]


class StrokeRateAction(ActionTerm):
    """One policy output setting how fast the stroke clock runs: rate = clip(1 + scale * a,
    *rate_range) clock ticks per policy tick, published as ``env._stroke_rate``. Drives no
    actuator; a zero output keeps the base period."""

    cfg: "StrokeRateActionCfg"

    def __init__(self, cfg: "StrokeRateActionCfg", env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._raw = torch.zeros(self.num_envs, 1, device=self.device)
        env._stroke_rate = torch.ones(self.num_envs, device=self.device)

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw[:] = actions
        lo, hi = self.cfg.rate_range
        self._env._stroke_rate = torch.clamp(1.0 + self.cfg.scale * actions[:, 0], lo, hi)

    def apply_actions(self) -> None:
        pass

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._raw[ids] = 0.0
        self._env._stroke_rate[ids] = 1.0


@dataclass(kw_only=True)
class StrokeRateActionCfg(ActionTermCfg):
    scale: float = 0.5
    rate_range: tuple[float, float] = (0.6, 1.2)

    def build(self, env: ManagerBasedRlEnv) -> StrokeRateAction:
        return StrokeRateAction(self, env)


def stroke_rate_cost(env: ManagerBasedRlEnv, command_name: str = "twist") -> torch.Tensor:
    """Stroke rate above the slowest allowed while the clock runs: strokes are paid for, so
    the policy pushes as seldom as the speed allows and glides longer where the floor lets it."""
    term = env.action_manager.get_term("stroke_rate")
    moving = (gait_clock_steps(env, command_name) > 0).float()
    env.extras["log"]["Metrics/stroke_rate_mean"] = env._stroke_rate.mean()
    return (env._stroke_rate - term.cfg.rate_range[0]) * moving


def _gait_phase(env: ManagerBasedRlEnv, period_steps: int, command_name: str) -> torch.Tensor:
    return (gait_clock_steps(env, command_name) % period_steps) / period_steps


def gait_phase(env: ManagerBasedRlEnv, period_steps: int, command_name: str = "twist") -> torch.Tensor:
    """(cos, sin) of the stroke clock, as appended by WalkMove.build_observation."""
    angle = 2 * math.pi * _gait_phase(env, period_steps, command_name)
    return torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)


def _stroke_schedule(
    env: ManagerBasedRlEnv, period_steps: int, push_fraction: float, margin: float, command_name: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scheduled contact of each skate (contact-sensor order: right, left) and whether the
    schedule is binding. Each skate's own phase starts as it begins pushing beside the
    gliding one (both down), lifts after ``push_fraction`` of the cycle and sets down at half
    the cycle to glide; the left runs half a cycle behind. Not binding within ``margin`` of
    the lift and touchdown instants. Standing schedules both skates down."""
    steps = gait_clock_steps(env, command_name)
    phase = (steps % period_steps).float() / period_steps
    local = torch.stack([phase, torch.remainder(phase + 0.5, 1.0)], dim=-1)
    down = (local < push_fraction) | (local >= 0.5)
    near_switch = ((local - push_fraction).abs() < margin) | ((local - 0.5).abs() < margin)
    moving = (steps > 0).unsqueeze(-1)
    down = down | ~moving
    binding = ~near_switch | ~moving
    return down, binding


def _stroke_contact(
    env: ManagerBasedRlEnv, sensor_name: str, period_steps: int, push_fraction: float, margin: float,
    command_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    down, binding = _stroke_schedule(env, period_steps, push_fraction, margin, command_name)
    found = env.scene.sensors[sensor_name].data.found
    if found.dim() == 3:
        found = found.any(dim=-1)
    return found > 0, down, binding


def stroke_contact_mismatch(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    period_steps: int,
    push_fraction: float,
    margin: float,
    command_name: str = "twist",
) -> torch.Tensor:
    """Number of skates (0-2) whose ground contact is off the stroke clock: down while it
    should hang, or up while it should push or glide.

    Paid as a match fraction instead, both skates parked on the floor kept 78% of it and a
    policy trained from scratch stood still in a braking V for 15000 iterations."""
    found, down, binding = _stroke_contact(env, sensor_name, period_steps, push_fraction, margin, command_name)
    off = (found != down) & binding
    env.extras["log"]["Metrics/stroke_schedule_match"] = 1.0 - off.float().mean()
    return off.float().sum(dim=-1)


def stroke_lift(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    period_steps: int,
    push_fraction: float,
    margin: float,
    command_name: str = "twist",
) -> torch.Tensor:
    """Number of skates off the ground while the stroke clock lifts them."""
    found, down, binding = _stroke_contact(env, sensor_name, period_steps, push_fraction, margin, command_name)
    return (~found & ~down & binding).float().sum(dim=-1)


def stroke_glide_follow(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    period_steps: int,
    margin: float,
    std: float,
    command_name: str = "twist",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Sum over the skates in their glide half of the stroke (own phase 0.5-1, outside
    ``margin``) and on the ground of exp(-(forward speed - trunk's)^2 / std^2), heading frame,
    times the trunk's forward speed over the command (0-1).

    On free wheels at a steady trunk speed the stance leg puts next to no horizontal force on
    its skate, so a skate set down at rest stays put while the trunk passes over it: the walk.
    A gliding skate is set down already moving with the trunk and rolls along with it."""
    asset: Entity = env.scene[asset_cfg.name]
    steps = gait_clock_steps(env, command_name)
    phase = (steps % period_steps).float() / period_steps
    local = torch.stack([phase, torch.remainder(phase + 0.5, 1.0)], dim=-1)  # right, left
    gliding = (local >= 0.5 + margin) & (local < 1.0 - margin) & (steps > 0).unsqueeze(-1)
    found = env.scene.sensors[sensor_name].data.found
    if found.dim() == 3:
        found = found.any(dim=-1)
    yaw = yaw_quat(asset.data.root_link_quat_w)
    vel = asset.data.site_lin_vel_w[:, asset_cfg.site_ids]
    foot_v = quat_apply_inverse(yaw.unsqueeze(1).expand(-1, vel.shape[1], -1).reshape(-1, 4),
                                vel.reshape(-1, 3)).reshape(vel.shape)[..., 0]
    trunk_v = quat_apply_inverse(yaw, asset.data.root_link_lin_vel_w)[:, 0:1]
    follow = torch.exp(-torch.square(foot_v - trunk_v) / std**2)
    # Scaled by the trunk's share of the commanded speed: standing still matches trivially.
    cmd = env.command_manager.get_command(command_name)[:, 0:1].clamp(min=0.1)
    follow = follow * torch.clamp(trunk_v / cmd, 0.0, 1.0)
    active = (gliding & (found > 0)).float()
    env.extras["log"]["Metrics/glide_speed_ratio"] = (foot_v * active).sum() / (trunk_v * active).sum().clamp(min=1e-6)
    return torch.sum(follow * active, dim=-1)


def foot_sagittal_excursion(
    env: ManagerBasedRlEnv,
    min_x: float,
    max_x: float,
    command_name: str = "twist",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Sum over the feet of the squared distance [m^2] by which each sits ahead of ``max_x``
    or behind ``min_x`` of the trunk (heading frame) while the stroke clock runs: a skater
    pushes out to the side and draws the skate back beside the other, it does not stride."""
    asset: Entity = env.scene[asset_cfg.name]
    rel = asset.data.site_pos_w[:, asset_cfg.site_ids] - asset.data.root_link_pos_w.unsqueeze(1)
    yaw = yaw_quat(asset.data.root_link_quat_w).unsqueeze(1).expand(-1, rel.shape[1], -1)
    x = quat_apply_inverse(yaw.reshape(-1, 4), rel.reshape(-1, 3)).reshape(rel.shape)[..., 0]
    excess = torch.clamp(min_x - x, min=0.0) + torch.clamp(x - max_x, min=0.0)
    moving = (gait_clock_steps(env, command_name) > 0).float()
    return torch.sum(torch.square(excess), dim=-1) * moving


def _stroke_reference(env: ManagerBasedRlEnv, keyframes: tuple, period_steps: int, command_name: str) -> torch.Tensor:
    """Reference joint angles [rad] off the default pose, right limb then mirrored left limb,
    linearly interpolated between ``keyframes`` ((phase, (deg per right-side joint)), ...)
    of the right leg's own phase; the left side runs half a cycle behind. The first
    keyframe sits at phase 0."""
    tables = env.__dict__.setdefault("_stroke_reference_tables", {})
    table = tables.get(keyframes)
    if table is None:
        phases = [k[0] for k in keyframes] + [1.0]
        values = [k[1] for k in keyframes] + [keyframes[0][1]]
        table = (torch.tensor(phases, device=env.device),
                 torch.deg2rad(torch.tensor(values, device=env.device, dtype=torch.float32)))
        tables[keyframes] = table
    knots, values = table
    phase = _gait_phase(env, period_steps, command_name)
    local = torch.stack([phase, torch.remainder(phase + 0.5, 1.0)], dim=-1)  # right, left
    seg = torch.clamp(torch.searchsorted(knots, local.contiguous(), right=True) - 1, 0, len(knots) - 2)
    t = ((local - knots[seg]) / (knots[seg + 1] - knots[seg])).unsqueeze(-1)
    ref = values[seg] * (1 - t) + values[seg + 1] * t  # [B, 2, J]
    return torch.cat([ref[:, 0], -ref[:, 1]], dim=-1)


def stroke_reference_pose(
    env: ManagerBasedRlEnv,
    keyframes: tuple,
    period_steps: int,
    std: float,
    command_name: str = "twist",
    metric: str = "stroke_reference_rms_deg",
    overspeed_band: float | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """exp(-mean squared error / std^2) of the selected joints against a stroke reference
    while the clock runs. ``asset_cfg`` lists the right side's joints, then the left's, in
    keyframe order; the rms error is logged as Metrics/``metric``. With ``overspeed_band``,
    nothing is paid while the trunk goes faster than the command by more than that [m/s].
    Paid only contact timing, the policy stepped like the walk."""
    asset: Entity = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    q = asset.data.joint_pos[:, ids] - asset.data.default_joint_pos[:, ids]
    ref = _stroke_reference(env, keyframes, period_steps, command_name)
    paid = (gait_clock_steps(env, command_name) > 0).float()
    if overspeed_band is not None:
        vx = quat_apply_inverse(yaw_quat(asset.data.root_link_quat_w), asset.data.root_link_lin_vel_w)[:, 0]
        cmd = env.command_manager.get_command(command_name)[:, 0]
        paid = paid * (vx <= cmd + overspeed_band).float()
    error = torch.mean(torch.square(q - ref), dim=-1)
    env.extras["log"][f"Metrics/{metric}"] = torch.rad2deg(torch.sqrt(error)).mean()
    return torch.exp(-error / std**2) * paid


def track_yaw_rate_filtered(
    env: ManagerBasedRlEnv,
    std: float,
    tau: float,
    command_name: str = "twist",
) -> torch.Tensor:
    """exp(-(yaw rate low-passed over ``tau`` [s] - command)^2 / std^2), world yaw rate.

    Each push twists the trunk +-1 rad/s, which swamps the instantaneous tracking kernel;
    the low-passed rate keeps the turn the robot makes over a stroke."""
    asset: Entity = env.scene["robot"]
    wz = asset.data.root_link_ang_vel_w[:, 2]
    state = env.__dict__.get("_yaw_rate_lowpass")
    if state is None:
        state = env._yaw_rate_lowpass = {"rate": torch.zeros_like(wz), "counter": -1}
    if state["counter"] != env.common_step_counter:
        state["counter"] = env.common_step_counter
        alpha = min(env.step_dt / tau, 1.0)
        state["rate"] = state["rate"] + alpha * (wz - state["rate"])
    state["rate"] = torch.where(env.episode_length_buf <= 1, wz, state["rate"])
    cmd = env.command_manager.get_command(command_name)[:, 2]
    error = state["rate"] - cmd
    env.extras["log"]["Metrics/yaw_rate_lowpass_error"] = error.abs().mean()
    return torch.exp(-torch.square(error) / std**2)


def standing_reference_pose(
    env: ManagerBasedRlEnv,
    targets: tuple[float, ...],
    std: float,
    command_name: str = "twist",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """exp(-mean squared error / std^2) of the selected joints against ``targets`` [deg off
    the default pose, right-side joints] while the stroke clock is stopped. ``asset_cfg``
    lists the right side's joints, then the left's, which mirror them."""
    asset: Entity = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    q = asset.data.joint_pos[:, ids] - asset.data.default_joint_pos[:, ids]
    right = torch.deg2rad(torch.tensor(targets, device=env.device, dtype=q.dtype))
    error = torch.mean(torch.square(q - torch.cat([right, -right])), dim=-1)
    standing = (gait_clock_steps(env, command_name) == 0).float()
    return torch.exp(-error / std**2) * standing


def overspeed_l2(
    env: ManagerBasedRlEnv,
    band: float,
    command_name: str = "twist",
    command_threshold: float = 0.01,
) -> torch.Tensor:
    """Squared forward speed [m^2/s^2] beyond ``band`` over the command while moving."""
    asset: Entity = env.scene["robot"]
    vx = quat_apply_inverse(yaw_quat(asset.data.root_link_quat_w), asset.data.root_link_lin_vel_w)[:, 0]
    cmd = env.command_manager.get_command(command_name)[:, 0]
    excess = torch.clamp(vx - cmd - band, min=0.0)
    return torch.square(excess) * _moving(env, command_name, command_threshold)


def yaw_rate_error_l2(env: ManagerBasedRlEnv, command_name: str = "twist") -> torch.Tensor:
    """Squared error [rad^2/s^2] of the trunk's world yaw rate against the command. The
    tracking kernel (std 0.35) pays nothing at the +-2.5 rad/s twist of each push, so the
    policy dropped turning altogether; the square still pulls there."""
    asset: Entity = env.scene["robot"]
    cmd = env.command_manager.get_command(command_name)[:, 2]
    return torch.square(asset.data.root_link_ang_vel_w[:, 2] - cmd)


def stance_joint_opening_excess(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    max_opening: float,
    signs: tuple[float, ...],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """joint_opening_excess for the legs whose skate is on the ground (one joint per skate,
    in contact-sensor order: right, left): the lifted leg folds freely to clear the floor."""
    asset: Entity = env.scene[asset_cfg.name]
    found = env.scene.sensors[sensor_name].data.found
    if found.dim() == 3:
        found = found.any(dim=-1)
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    opening = q * torch.tensor(signs, device=q.device, dtype=q.dtype)
    excess = torch.square(torch.clamp(opening - max_opening, min=0.0))
    return torch.sum(excess * (found > 0).float(), dim=-1)


def stroke_swing_clearance(
    env: ManagerBasedRlEnv,
    height_sensor_name: str,
    target_height: float,
    period_steps: int,
    push_fraction: float,
    margin: float,
    height_order: tuple[int, int] = (1, 0),
    command_name: str = "twist",
) -> torch.Tensor:
    """Height [m] over the floor of each skate scheduled in the air, over ``target_height``
    and capped at 1: a skate still on the ground while its schedule lifts it earns nothing
    but gets a gradient towards the lift. ``height_order`` maps the height scan's frames
    (left, right) to the contact sensor's (right, left)."""
    down, binding = _stroke_schedule(env, period_steps, push_fraction, margin, command_name)
    heights = env.scene[height_sensor_name].data.heights[:, list(height_order)]
    lift = torch.clamp(heights / target_height, min=0.0, max=1.0)
    return torch.sum(lift * (~down & binding).float(), dim=-1)


def stroke_ground_clearance_cost(
    env: ManagerBasedRlEnv,
    height_sensor_name: str,
    target_height: float,
    period_steps: int,
    push_fraction: float,
    margin: float,
    height_order: tuple[int, int] = (1, 0),
    command_name: str = "twist",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """feet_clearance for the skates scheduled in the air only: shortfall [m] below
    ``target_height`` times the skate's horizontal speed. The walk's version taxes every
    skate moving low, the gliding one included."""
    asset: Entity = env.scene[asset_cfg.name]
    down, _ = _stroke_schedule(env, period_steps, push_fraction, margin, command_name)
    heights = env.scene[height_sensor_name].data.heights[:, list(height_order)]
    speed = torch.norm(asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2], dim=-1)
    shortfall = torch.clamp(target_height - heights, min=0.0)
    return torch.sum(shortfall * speed * (~down).float(), dim=-1)


def skate_glide_rolling(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str = "twist",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Forward speed [m/s] along its own nose, capped at the commanded speed, of a skate
    carrying the robot alone: it rolls along with the trunk instead of being planted while
    the trunk vaults over it like a walking step."""
    asset: Entity = env.scene[asset_cfg.name]
    found = env.scene.sensors[sensor_name].data.found
    if found.dim() == 3:
        found = found.any(dim=-1)
    down = found > 0
    alone = down & ~down.flip(-1)
    quat = asset.data.site_quat_w[:, asset_cfg.site_ids]
    vel = asset.data.site_lin_vel_w[:, asset_cfg.site_ids]
    along = quat_apply_inverse(quat.reshape(-1, 4), vel.reshape(-1, 3)).reshape(vel.shape)[..., 0]
    cap = env.command_manager.get_command(command_name)[:, 0].clamp(min=0.0).unsqueeze(-1)
    return torch.sum(alone.float() * torch.minimum(along.clamp(min=0.0), cap), dim=-1)


def arm_swing(
    env: ManagerBasedRlEnv,
    max_speed: float,
    command_name: str = "twist",
    command_threshold: float = 0.01,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Mean arm joint speed [rad/s], capped at ``max_speed``, while moving: swinging arms
    carry momentum into the push."""
    asset: Entity = env.scene[asset_cfg.name]
    speed = torch.clamp(asset.data.joint_vel[:, asset_cfg.joint_ids].abs(), max=max_speed)
    return speed.mean(dim=-1) * _moving(env, command_name, command_threshold)


def mirror_pose_asymmetry(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Sum of squared left/right mismatch [rad^2] of the selected joints, given as
    (left, right) pairs in ``asset_cfg.joint_names`` order. Paired joints have mirrored
    axes, so a mirror-symmetric pose has left = -right."""
    asset: Entity = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(q[:, 0::2] + q[:, 1::2]), dim=-1)


def total_current_excess(
    env: ManagerBasedRlEnv, max_current: float, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """Amount [A] by which the motor currents of all BAM actuators, summed, exceed
    ``max_current``. The servos share one 2S pack: a pack-wide spike trips the BMS and
    reboots the Pi."""
    asset: Entity = env.scene[asset_cfg.name]
    current = torch.zeros(env.num_envs, device=env.device)
    for act in asset.actuators:
        torque = getattr(act, "_prev_motor_torque", None)
        if torque is not None:
            current = current + torque.abs().sum(dim=1) / act._bam_model.kt.value
    return torch.clamp(current - max_current, min=0.0)


def root_speed_xy_excess(
    env: ManagerBasedRlEnv, max_speed: float, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """Amount [m/s] by which the trunk's horizontal speed exceeds ``max_speed``."""
    speed = env.scene[asset_cfg.name].data.root_link_lin_vel_w[:, :2].norm(dim=-1)
    return torch.clamp(speed - max_speed, min=0.0)


def body_speed_excess(
    env: ManagerBasedRlEnv, max_speed: float, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """Amount [m/s] by which the fastest ``asset_cfg`` body exceeds ``max_speed``: limbs
    flung at the ground and the trunk slammed down are the impacts."""
    vel = env.scene[asset_cfg.name].data.body_link_lin_vel_w[:, asset_cfg.body_ids]
    return torch.clamp(vel.norm(dim=-1).max(dim=1).values - max_speed, min=0.0)


def joint_speed_excess(
    env: ManagerBasedRlEnv, max_speed: float, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """Sum over the ``asset_cfg`` joints of the amount [rad/s] by which |joint velocity|
    exceeds ``max_speed``: motors snapping at full speed to throw the body up."""
    vel = env.scene[asset_cfg.name].data.joint_vel[:, asset_cfg.joint_ids]
    return torch.clamp(vel.abs() - max_speed, min=0.0).sum(dim=1)


def trunk_ang_speed_excess(
    env: ManagerBasedRlEnv, max_speed: float, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """Amount [rad/s] by which the trunk's angular speed exceeds ``max_speed``: the body
    flipped or swung over instead of pushed up."""
    return torch.clamp(env.scene[asset_cfg.name].data.root_link_ang_vel_b.norm(dim=-1) - max_speed, min=0.0)


def ground_contact_slip(
    env: ManagerBasedRlEnv, sensor_name: str, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """Sum [m/s] over the bodies touching the ground of the horizontal speed of the contact
    point (body velocity plus angular velocity times the lever arm), so a limb rolling or
    pivoting on its contact costs nothing and one dragged along the floor does.

    Standing up by scraping the shell along the floor only works at the trained friction."""
    sensor = env.scene[sensor_name]
    asset: Entity = env.scene[asset_cfg.name]
    body_ids = getattr(sensor, "_slip_body_ids", None)
    if body_ids is None:
        names = list(dict.fromkeys(slot.primary_name for slot in sensor._slots))
        body_ids = sensor._slip_body_ids = asset.find_bodies(names, preserve_order=True)[0]
    data = sensor.data
    in_contact = data.found > 0
    lever = data.pos - asset.data.body_link_pos_w[:, body_ids]
    v = asset.data.body_link_lin_vel_w[:, body_ids] + torch.cross(
        asset.data.body_link_ang_vel_w[:, body_ids], lever, dim=-1
    )
    return (v[..., :2].norm(dim=-1) * in_contact).sum(dim=1)


def ground_friction_use(
    env: ManagerBasedRlEnv, sensor_name: str, max_ratio: float, min_normal_force: float = 0.2
) -> torch.Tensor:
    """Sum over the bodies pressing on the ground (normal force above ``min_normal_force`` [N])
    of the amount by which |horizontal force| / normal force exceeds ``max_ratio``.

    Below ``max_ratio`` every contact holds by static friction on any floor with a friction
    coefficient above it, so the get-up does not depend on how grippy the floor is.
    ``sensor_name`` is a netforce contact sensor against the flat terrain (world frame)."""
    force = env.scene[sensor_name].data.force
    normal = force[..., 2].abs()
    ratio = force[..., :2].norm(dim=-1) / normal.clamp(min=1e-6)
    excess = torch.clamp(ratio - max_ratio, min=0.0) * (normal > min_normal_force)
    return excess.sum(dim=1)


def _mark_nonfinite_envs(bad: torch.Tensor, value) -> None:
    """Flag environments whose batched tensor contains NaN or Inf."""
    if isinstance(value, dict):
        for item in value.values():
            _mark_nonfinite_envs(bad, item)
        return
    if not torch.is_tensor(value) or value.shape[:1] != bad.shape:
        return
    flat = value.reshape(bad.shape[0], -1)
    bad |= ~torch.isfinite(flat).all(dim=1)


def _scrub_nonfinite(value) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _scrub_nonfinite(item)
        return
    if torch.is_tensor(value) and value.is_floating_point():
        torch.nan_to_num_(value, nan=0.0, posinf=0.0, neginf=0.0)


def _install_diverged_env_reset() -> None:
    """Reset environments whose physics state diverged, instead of aborting the run.

    rsl_rl raises as soon as any observation is NaN. One of the 4096 parallel
    environments can drive the contact solver into NaN; the others are fine.
    Those environments are reset to the ready pose and that transition is
    treated as a fall (zero reward, terminated). Actions are not clipped.
    """
    if getattr(ManagerBasedRlEnv.step, "_xiaoxiaobai_nan_reset", False):
        return
    original_step = ManagerBasedRlEnv.step

    def step(self, action):
        obs, reward, terminated, timeouts, extras = original_step(self, action)
        bad = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        data = self.sim.data
        for name in ("qpos", "qvel", "qacc"):
            _mark_nonfinite_envs(bad, getattr(data, name, None))
        _mark_nonfinite_envs(bad, obs)
        _mark_nonfinite_envs(bad, reward)
        if not bool(bad.any()):
            return obs, reward, terminated, timeouts, extras

        env_ids = bad.nonzero(as_tuple=False).flatten()
        step._reset_events = getattr(step, "_reset_events", 0) + 1
        step._reset_count = getattr(step, "_reset_count", 0) + int(env_ids.numel())
        if step._reset_events <= 3 or step._reset_events % 200 == 0:
            print(
                f"[xiaoxiaobai] reset {int(env_ids.numel())} diverged envs "
                f"(events {step._reset_events}, cumulative {step._reset_count})",
                flush=True,
            )

        self._reset_idx(env_ids)
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.sim.sense()
        self.obs_buf = self.observation_manager.compute(update_history=True)
        _scrub_nonfinite(self.obs_buf)

        reward = reward.clone()
        reward[env_ids] = 0
        torch.nan_to_num_(reward, nan=0.0, posinf=0.0, neginf=0.0)
        self.reward_buf = reward

        terminated = terminated.clone()
        terminated[env_ids] = True
        self.reset_terminated = terminated
        timeouts = timeouts.clone()
        timeouts[env_ids] = False
        self.reset_time_outs = timeouts
        return self.obs_buf, reward, terminated, timeouts, extras

    step._xiaoxiaobai_nan_reset = True
    ManagerBasedRlEnv.step = step


_install_diverged_env_reset()
