"""Free-arms walk: the policy balances on legs 1-12 while the arms play gestures.

On the robot the arms (motors 13-18) belong to xiaoxiaobai's gesture engine under the walk. Here
they follow recorded engine output (scripts/build_arm_motion_bank.py) at KP_UPPER: the
policy's arm outputs are dropped and its arm observations (angles, speeds, last arm actions)
are zeroed, so it neither drives nor sees the arms and balances against whatever they do.
Network inputs and outputs keep the walk's shapes, so this fine-tunes a walk checkpoint; the
critic still sees the arms.

XIAOXIAOBAI_HAND_LEAD=<fraction>: that share of the episodes is led by the hand instead
(HandLeadSim), as xiaoxiaobai's 牵手走 (moves/lead.py, moves/upper_body.py).
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, fields
from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp

from mjlab_xiaoxiaobai.robot.xiaoxiaobai_constants import (
    ARM_JOINTS,
    FOREARMS,
    FREE_ARMS_KP,
    HOME_FRAME,
    XIAOXIAOBAI_FREE_ARMS_ROBOT_CFG,
    get_free_arms_spec,
)
from mjlab_xiaoxiaobai.tasks.xiaoxiaobai_velocity_env_cfg import make_xiaoxiaobai_velocity_env_cfg

ARM_MOTION_BANK = Path(__file__).parent / "data" / "arm_motion_bank.npz"
# Arms blend from the ready pose into their stream after a reset, as on the robot, where
# they start at rest.
BLEND_IN_S = 1.0

# -- Led by the hand (xiaoxiaobai moves/lead.py, moves/upper_body.py) --
# On the robot both arms go limp (KP 0); a hand takes one or both and walks, and the robot
# follows the shoulder angles: pitch forward 15-50 deg -> vx 0-0.3 m/s, back 15-40 -> 0-0.2,
# roll out 20-60 deg on one side -> turn to it at up to 0.6 rad/s. Let go, an arm still for
# 0.6 s eases back to rest at KP 100 and 25 deg/s, and goes limp again there.
# Here the hand is a spring-damper on the forearm tip toward a point that walks at the
# commanded velocity where the arm would be held for that command, and slowly settles back
# there when the robot drifts off it, with some jitter, its pull capped per episode.
LEAD_SIDES = ("right", "left")
LEAD_HOLD_S = (2.0, 12.0)
LEAD_RELEASE_S = (1.0, 5.0)
LEAD_BOTH_HANDS = 0.2           # share of the holds with both hands, the rest one side each
LEAD_STIFFNESS = (20.0, 80.0)   # N/m
LEAD_DAMPING = (0.3, 1.5)       # N s/m
LEAD_MAX_FORCE = (0.5, 2.5)     # N, against the 1.1 kg robot
LEAD_SLACK = (-0.1, 0.15)       # held point from the shoulder, in arm lengths beyond the arm
LEAD_ADAPT_S = 1.5              # the hand settles back to its place relative to the robot
LEAD_POSE_TAU_S = 0.3           # the held arm angle follows the command
LEAD_JITTER_M = 0.01
LEAD_JITTER_TAU_S = 0.5
LEAD_STILL_S = 0.6
LEAD_RETURN_KP_SCALE = 100.0 / FREE_ARMS_KP
LEAD_RETURN_RAD_S = math.radians(25.0)
LEAD_REST_RAD = math.radians(3.0)


def _arm_geometry() -> tuple[np.ndarray, float]:
    """Shoulder pivots (right, left) relative to the trunk at the ready pose, and the length from
    the pivot to the hand."""
    model = get_free_arms_spec().compile()
    data = mujoco.MjData(model)
    data.qpos[:3] = HOME_FRAME.pos
    data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    for name, value in HOME_FRAME.joint_pos.items():
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = value
    mujoco.mj_kinematics(model, data)
    root = data.xpos[model.body("trunk").id]
    pivots = np.stack([data.xanchor[model.joint(f"{side}_shoulder_roll").id] - root for side in LEAD_SIDES])
    hand = data.site_xpos[model.site("right_hand").id] - root
    return pivots, float(np.linalg.norm(hand - pivots[0]))


class HandLeadSim:
    """A hand leading the robot by one or both forearms, in the episodes ``fraction`` draws."""

    def __init__(self, env: ManagerBasedRlEnv, fraction: float, arm_cols: torch.Tensor, rest: torch.Tensor):
        self.env = env
        self.fraction = fraction
        self.device = env.device
        n = env.num_envs
        self.robot = env.scene["robot"]
        self.site_ids = self.robot.find_sites([f"{s}_hand" for s in LEAD_SIDES], preserve_order=True)[0]
        self.body_ids = self.robot.find_bodies([FOREARMS[s] for s in LEAD_SIDES], preserve_order=True)[0]
        self.arm_act = next(a for a in self.robot.actuators if any(re.match(ARM_JOINTS, t) for t in a.target_names))
        self.arm_cols = arm_cols
        self.rest = rest  # (n, 6) arm targets at rest
        self.arm_joint_ids = None  # set by the action term
        pivots, self.arm_len = _arm_geometry()
        self.pivots = torch.tensor(pivots, dtype=torch.float32, device=self.device)  # (2, 3)
        self.out_sign = torch.tensor([-1.0, 1.0], device=self.device)  # +y is the left side

        f = lambda *s: torch.zeros(n, *s, device=self.device)
        b = lambda *s: torch.zeros(n, *s, dtype=torch.bool, device=self.device)
        self.on, self.holding, self.held = b(), b(), b(2)
        self.timer, self.released_s = f(), f()
        self.returning = b()
        self.return_target = f(rest.shape[1])
        self.anchor, self.anchor_vel, self.jitter = f(2, 3), f(2, 3), f(2, 3)
        self.pitch, self.roll = f(2), f(2)
        self.k, self.c, self.fmax, self.slack = f(), f(), f(), f(2)
        self.force = f(2, 3)

    def _uniform(self, k: int, lo_hi: tuple[float, float]) -> torch.Tensor:
        return torch.empty(k, device=self.device).uniform_(*lo_hi)

    def reset(self, env_ids: torch.Tensor) -> None:
        k = len(env_ids)
        self.on[env_ids] = torch.rand(k, device=self.device) < self.fraction
        self.holding[env_ids] = False
        self.held[env_ids] = False
        self.timer[env_ids] = self._uniform(k, (0.0, 1.5))
        self.released_s[env_ids] = 0.0
        self.returning[env_ids] = False
        self.anchor[env_ids] = 0.0
        self.anchor_vel[env_ids] = 0.0
        self.jitter[env_ids] = 0.0
        self.pitch[env_ids] = 0.0
        self.roll[env_ids] = math.radians(8.0)
        self.k[env_ids] = self._uniform(k, LEAD_STIFFNESS)
        self.c[env_ids] = self._uniform(k, LEAD_DAMPING)
        self.fmax[env_ids] = self._uniform(k, LEAD_MAX_FORCE)
        self.force[env_ids] = 0.0
        self.arm_act.kp_scale[env_ids, 0] = torch.where(self.on[env_ids], 0.0, 1.0)

    def _held_place(self, command: torch.Tensor) -> torch.Tensor:
        """Where each hand is held for the command, in the heading frame from the trunk (n, 2, 3)."""
        vx, wz = command[:, 0], command[:, 2]
        fwd = torch.clamp(15.0 + vx / 0.3 * 35.0, max=60.0)
        back = -torch.clamp(15.0 + vx.abs() / 0.2 * 25.0, max=45.0)
        pitch = torch.deg2rad(torch.where(vx > 0.03, fwd, torch.where(vx < -0.03, back, torch.zeros_like(vx))))
        lift = torch.clamp(20.0 + wz.abs() / 0.6 * 40.0, max=60.0)
        roll_r = torch.where(wz < -0.05, lift, torch.zeros_like(wz)) + 8.0
        roll_l = torch.where(wz > 0.05, lift, torch.zeros_like(wz)) + 8.0
        a = min(1.0, self.env.step_dt / LEAD_POSE_TAU_S)
        self.pitch += a * (pitch[:, None] - self.pitch)
        self.roll += a * (torch.deg2rad(torch.stack([roll_r, roll_l], dim=1)) - self.roll)
        p, r = self.pitch, self.roll
        direction = torch.stack([torch.sin(p) * torch.cos(r), self.out_sign * torch.sin(r), -torch.cos(p) * torch.cos(r)], dim=-1)
        return self.pivots + self.arm_len * (1.0 + self.slack)[..., None] * direction

    def step(self, targets: torch.Tensor) -> None:
        """Once per env step: holds and releases, the held point, and the arms' targets and KP."""
        dt = self.env.step_dt
        on = self.on
        self.timer -= dt
        flip = on & (self.timer <= 0.0)
        grab, let_go = flip & ~self.holding, flip & self.holding
        self.holding ^= flip
        n_grab = int(grab.sum())
        if n_grab:
            self.timer[grab] = self._uniform(n_grab, LEAD_HOLD_S)
            r = torch.rand(n_grab, device=self.device)
            both = r < LEAD_BOTH_HANDS
            right = both | (r < LEAD_BOTH_HANDS + (1.0 - LEAD_BOTH_HANDS) / 2)
            self.held[grab] = torch.stack([right, both | ~right], dim=1)
            self.slack[grab] = torch.empty(n_grab, 2, device=self.device).uniform_(*LEAD_SLACK)
            self.anchor[grab] = self.robot.data.site_pos_w[grab][:, self.site_ids]
            self.jitter[grab] = 0.0
            self.returning[grab] = False
        if int(let_go.sum()):
            self.timer[let_go] = self._uniform(int(let_go.sum()), LEAD_RELEASE_S)
            self.held[let_go] = False
            self.released_s[let_go] = 0.0

        # The held point walks at the command, turning about the trunk, and settles back to
        # where the arm is held for it.
        robot = self.robot
        command = self.env.command_manager.get_command("twist")
        place = self._held_place(command)
        yaw = robot.data.heading_w
        cos, sin = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
        root = robot.data.root_link_pos_w[:, None, :]
        place_w = torch.stack([cos * place[..., 0] - sin * place[..., 1], sin * place[..., 0] + cos * place[..., 1],
                               place[..., 2]], dim=-1) + root
        vx, vy, wz = command[:, 0:1], command[:, 1:2], command[:, 2:3]
        rel = self.anchor - root
        self.anchor_vel = torch.stack([cos * vx - sin * vy - wz * rel[..., 1], sin * vx + cos * vy + wz * rel[..., 0],
                                       torch.zeros_like(rel[..., 0])], dim=-1)
        self.anchor += (place_w - self.anchor) * min(1.0, dt / LEAD_ADAPT_S)
        a = dt / LEAD_JITTER_TAU_S
        self.jitter += -a * self.jitter + LEAD_JITTER_M * math.sqrt(2.0 * a) * torch.randn_like(self.jitter)

        # Arms: limp while led, then after a release once still, back to rest at KP 100.
        q = robot.data.joint_pos[:, self.arm_joint_ids]
        free = on & ~self.holding
        self.released_s = torch.where(free, self.released_s + dt, torch.zeros_like(self.released_s))
        start = free & ~self.returning & (self.released_s >= LEAD_STILL_S) & ((q - self.rest).abs().amax(dim=1) > LEAD_REST_RAD)
        self.return_target[start] = q[start]
        self.returning |= start
        step = LEAD_RETURN_RAD_S * dt
        self.return_target += torch.clamp(self.rest - self.return_target, -step, step)
        home = self.returning & ((self.return_target - self.rest).abs().amax(dim=1) < 1e-4) & \
            ((q - self.rest).abs().amax(dim=1) < LEAD_REST_RAD)
        self.returning &= ~home & free
        lead_targets = torch.where(self.returning[:, None], self.return_target, q)
        targets[:, self.arm_cols] = torch.where(on[:, None], lead_targets, targets[:, self.arm_cols])
        kp = torch.where(self.returning, LEAD_RETURN_KP_SCALE, 0.0)
        self.arm_act.kp_scale[:, 0] = torch.where(on, kp, 1.0)

    def apply(self, physics_dt: float) -> None:
        """Every physics step: the hands' pull on the forearm tips."""
        robot = self.robot
        self.anchor += self.anchor_vel * physics_dt
        hand = robot.data.site_pos_w[:, self.site_ids]
        hand_vel = robot.data.site_lin_vel_w[:, self.site_ids]
        force = self.k[:, None, None] * (self.anchor + self.jitter - hand) + self.c[:, None, None] * (self.anchor_vel - hand_vel)
        norm = force.norm(dim=-1, keepdim=True)
        force = force * torch.clamp(self.fmax[:, None, None] / norm.clamp(min=1e-6), max=1.0)
        force = force * (self.held & self.holding[:, None] & self.on[:, None]).float()[..., None]
        self.force = force
        com = robot.data.body_com_pos_w[:, self.body_ids]
        robot.write_external_wrench_to_sim(force, torch.cross(hand - com, force, dim=-1), body_ids=self.body_ids)


class FreeArmsJointPositionAction(JointPositionAction):
    """Joint position targets with the arm columns taken from the motion bank."""

    cfg: FreeArmsJointPositionActionCfg

    def __init__(self, cfg: FreeArmsJointPositionActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg=cfg, env=env)
        data = np.load(cfg.bank_path)
        if abs(float(data["dt"]) - env.step_dt) > 1e-6:
            raise ValueError(f"motion bank dt {float(data['dt'])} != env step {env.step_dt}")
        names = [str(n) for n in data["joints"]]
        self._arm_cols = torch.tensor([self._target_names.index(n) for n in names], device=self.device)
        self._bank = torch.as_tensor(data["targets"].astype(np.float32), device=self.device)
        self._num_streams, self._num_frames = self._bank.shape[:2]
        self._stream = torch.randint(self._num_streams, (self.num_envs,), device=self.device)
        self._frame = torch.randint(self._num_frames, (self.num_envs,), device=self.device)
        self._blend = torch.ones(self.num_envs, 1, device=self.device)
        self._blend_step = env.step_dt / BLEND_IN_S
        self._physics_dt = env.cfg.sim.mujoco.timestep
        self._lead = None
        if cfg.lead_fraction > 0.0:
            self._lead = HandLeadSim(env, cfg.lead_fraction, self._arm_cols, self._offset[:, self._arm_cols])
            self._lead.arm_joint_ids = self._target_ids[self._arm_cols]
            self._lead.reset(torch.arange(self.num_envs, device=self.device))

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        rest = self._offset[:, self._arm_cols]
        arms = self._bank[self._stream, self._frame]
        self._processed_actions[:, self._arm_cols] = rest + (arms - rest) * self._blend
        self._frame = (self._frame + 1) % self._num_frames
        self._blend = torch.clamp(self._blend + self._blend_step, max=1.0)
        if self._lead is not None:
            self._lead.step(self._processed_actions)

    def apply_actions(self) -> None:
        if self._lead is not None:
            self._lead.apply(self._physics_dt)
        super().apply_actions()

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        n = self._stream[env_ids].shape[0]
        self._stream[env_ids] = torch.randint(self._num_streams, (n,), device=self.device)
        self._frame[env_ids] = torch.randint(self._num_frames, (n,), device=self.device)
        self._blend[env_ids] = 0.0
        if self._lead is not None:
            ids = torch.arange(self.num_envs, device=self.device)[env_ids]
            self._lead.reset(ids)


@dataclass(kw_only=True)
class FreeArmsJointPositionActionCfg(JointPositionActionCfg):
    bank_path: str = str(ARM_MOTION_BANK)
    lead_fraction: float = 0.0

    def build(self, env: ManagerBasedRlEnv) -> FreeArmsJointPositionAction:
        return FreeArmsJointPositionAction(self, env)


def _arm_mask(names: list[str], device) -> torch.Tensor:
    return torch.tensor([re.match(ARM_JOINTS, n) is None for n in names], dtype=torch.float32, device=device)


_MASKS: dict[tuple, torch.Tensor] = {}


def _joint_mask(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """1 for the leg (non-arm) joints of the term, 0 for the arms."""
    key = (id(env), asset_cfg.name, str(asset_cfg.joint_ids))
    if key not in _MASKS:
        names = list(env.scene[asset_cfg.name].joint_names)
        ids = asset_cfg.joint_ids
        names = names[ids] if isinstance(ids, slice) else [names[i] for i in ids]
        _MASKS[key] = _arm_mask(names, env.device)
    return _MASKS[key]


def joint_pos_rel_legs(env: ManagerBasedRlEnv, biased: bool = False,
                       asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return mdp.joint_pos_rel(env, biased=biased, asset_cfg=asset_cfg) * _joint_mask(env, asset_cfg)


def joint_vel_rel_legs(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return mdp.joint_vel_rel(env, asset_cfg=asset_cfg) * _joint_mask(env, asset_cfg)


def last_action_legs(env: ManagerBasedRlEnv, action_name: str = "joint_pos") -> torch.Tensor:
    term = env.action_manager.get_term(action_name)
    mask = getattr(term, "_legs_mask", None)
    if mask is None:
        mask = term._legs_mask = _arm_mask(list(term.target_names), env.device)
    return term.raw_action * mask


def make_xiaoxiaobai_free_arms_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_xiaoxiaobai_velocity_env_cfg(play=play)
    cfg.scene.entities = {"robot": XIAOXIAOBAI_FREE_ARMS_ROBOT_CFG}

    walk_action = cfg.actions["joint_pos"]
    cfg.actions["joint_pos"] = FreeArmsJointPositionActionCfg(
        **{f.name: getattr(walk_action, f.name) for f in fields(walk_action) if f.init},
        lead_fraction=float(os.environ.get("XIAOXIAOBAI_HAND_LEAD", 0.0)),
    )

    actor = cfg.observations["actor"].terms
    actor["joint_pos"].func = joint_pos_rel_legs
    actor["joint_vel"].func = joint_vel_rel_legs
    actor["actions"].func = last_action_legs
    actor["actions"].params = {"action_name": "joint_pos"}

    # Only the legs' pose, limits and command leads are the policy's to answer for.
    not_arms = r"^(?!leg_flex|left_shoulder|left_elbow|right_shoulder|right_elbow).*$"
    pose = cfg.rewards["pose"].params
    pose["asset_cfg"] = SceneEntityCfg("robot", joint_names=(not_arms,))
    for key in ("std_standing", "std_walking", "std_running"):
        pose[key] = {k: v for k, v in pose[key].items() if not re.search(r"shoulder|elbow", k)}
    cfg.rewards["dof_pos_limits"].params["asset_cfg"] = SceneEntityCfg("robot", joint_names=(not_arms,))
    # The arm targets come from the bank (or a hand), not the policy: the walk's arm terms
    # (XIAOXIAOBAI_VIN_LOW, XIAOXIAOBAI_POSTURE, XIAOXIAOBAI_STYLE) have nothing to act on.
    for name in ("arm_squeeze", "arm_spread", "arm_reference", "walking_arm_spread"):
        cfg.rewards.pop(name, None)

    # Iteration 27500 (upright weight 1, body_ang_vel -0.05) held -0.9 deg walking forward and
    # -0.2 deg backward against targets of about -/+3.5 deg, rocked 3.1 deg (std) under the
    # gestures, and on the robot leaned into the walking direction both ways.
    cfg.rewards["upright"].weight = 2.0
    cfg.rewards["body_ang_vel"].weight = -0.1
    # Still -1.4 / -0.3 deg after that. A linear penalty (-10 per rad) on pitch past the target
    # toward the walking direction left them at -1.8 / -0.3 deg and fell 26 % of walks at 7.5 V
    # instead of 12 %.
    return cfg
