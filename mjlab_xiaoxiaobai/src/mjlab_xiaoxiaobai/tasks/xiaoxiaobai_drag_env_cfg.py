# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

"""Dragging: the robot walks backward pulling a load taped to both hands, the arms held
straight out in front. The load is a sled on the floor with a light rigid handle up to the
hands; each hand is welded to its grip, as tape would hold it.

Built on the walk task, whose policy it starts from: same observations, so the walk's actor
and critic resume as they are. The arms' default pose is the drag pose and they hold it: the
policy's arm actions are dropped, it drives the legs. Every env keeps a load level that rises
while it keeps up with the command and falls when it falls behind or over, so the load grows
to whatever the robot can pull, on its own floor friction and supply voltage."""

import math
import os
from dataclasses import fields, replace

import mujoco
import numpy as np
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.entity import EntityCfg

from mjlab_xiaoxiaobai.robot.xiaoxiaobai_constants import DRAG_HOME_FRAME, XIAOXIAOBAI_DRAG_ROBOT_CFG, get_drag_spec
from mjlab_xiaoxiaobai.tasks import mdp as xiaoxiaobai_mdp
from mjlab_xiaoxiaobai.tasks.xiaoxiaobai_velocity_env_cfg import XiaoxiaobaiVelocityRlCfg, make_xiaoxiaobai_velocity_env_cfg

# Sled half sizes [m]: depth, width, height. The handle rises from its rear edge to the hands.
LOAD_SLED_HALF = (0.08, 0.09, 0.02)
# 50 g to 4 kg in steps of ~12%; at the 1.07 kg robot's ~0.5 body weights of pull, 4 kg is
# out of reach on all but the most slippery floors.
DRAG_MASS_LEVELS = tuple(float(m) for m in np.geomspace(0.05, 4.0, 40))
# Levels start uniform below this; XIAOXIAOBAI_DRAG_LEVELS0 when resuming a policy that drags.
DRAG_INITIAL_LEVELS = int(os.environ.get("XIAOXIAOBAI_DRAG_LEVELS0", 5))
# Drag v3 pushed nearly every env to 0.8-1 kg and then leaned back 18-26 deg whatever it
# pulled, falling backward with 0.25 kg in 4-67% (8-5 V). This share of the episodes draws a
# lighter load from the env's own range.
DRAG_LIGHT_FRACTION = 0.3
# Leaning on the welded load as on a crutch, drag v4 fell in all episodes with the hands
# untaped, standing or walking. This share of the episodes has no load.
DRAG_FREE_FRACTION = 0.15
# Sliding friction of the load on the floor: plastic on tile up to rubber or cloth on carpet.
DRAG_FRICTION = (0.15, 0.9)
DRAG_VX = (-0.3, -0.05)
# Heading hold (xiaoxiaobai DRAG_HEADING_KP / DRAG_VTHETA must match): vtheta [rad/s] per rad of
# heading error and its bound; heading envs start up to DRAG_HEADING_OFFSET off their target.
DRAG_HEADING_KP = 1.0
DRAG_VTHETA = 0.4
DRAG_HEADING_OFFSET = 0.3


def _hand_frames() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Hand site positions and orientations relative to the root at the drag pose."""
    model = get_drag_spec().compile()
    data = mujoco.MjData(model)
    data.qpos[:3] = DRAG_HOME_FRAME.pos
    data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    for name, value in DRAG_HOME_FRAME.joint_pos.items():
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = value
    mujoco.mj_kinematics(model, data)
    root = data.xpos[model.body("trunk").id]
    frames = {}
    for side in ("right", "left"):
        site = model.site(f"{side}_hand").id
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, data.site_xmat[site])
        frames[side] = (data.site_xpos[site] - root, quat)
    return frames


_HANDS = _hand_frames()
# The load frame sits midway between the hands, in the robot's heading frame.
LOAD_OFFSET = tuple(float(v) for v in (_HANDS["right"][0] + _HANDS["left"][0]) / 2)
_HAND_HEIGHT = DRAG_HOME_FRAME.pos[2] + LOAD_OFFSET[2]
_SLED_POS = (LOAD_SLED_HALF[0], 0.0, -_HAND_HEIGHT + LOAD_SLED_HALF[2] + 0.001)
_hx, _hy, _hz = LOAD_SLED_HALF
LOAD_UNIT_INERTIA = ((_hy**2 + _hz**2) / 3, (_hx**2 + _hz**2) / 3, (_hx**2 + _hy**2) / 3)


def get_load_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="load")
    body.add_freejoint(name="load_joint")
    # Above the terrain's priority 0, so its randomised friction is the contact's.
    body.add_geom(
        name="load_sled", type=mujoco.mjtGeom.mjGEOM_BOX, size=LOAD_SLED_HALF, pos=_SLED_POS,
        friction=(0.5, 0.005, 0.0001), condim=3, priority=1, group=1, rgba=(0.75, 0.55, 0.3, 1.0),
    )
    rise = (_HAND_HEIGHT - 2 * _hz) / 2
    half_width = abs(float(_HANDS["right"][0][1] - LOAD_OFFSET[1])) + 0.01
    body.add_geom(
        name="load_handle", type=mujoco.mjtGeom.mjGEOM_BOX, size=(0.004, half_width, rise),
        pos=(0.0, 0.0, -rise), contype=0, conaffinity=0, group=1, rgba=(0.6, 0.6, 0.65, 1.0),
    )
    for side, (pos, quat) in _HANDS.items():
        body.add_site(name=f"{side}_grip", pos=list(pos - np.asarray(LOAD_OFFSET)), quat=list(quat))
    # Mass and inertia are set per episode (reset_drag); the handle weighs nothing.
    body.mass = DRAG_MASS_LEVELS[0]
    body.ipos = list(_SLED_POS)
    body.inertia = [DRAG_MASS_LEVELS[0] * v for v in LOAD_UNIT_INERTIA]
    body.explicitinertial = True
    return spec


LOAD_CFG = EntityCfg(spec_fn=get_load_spec)


def _install_eq_data_compat() -> None:
    """MuJoCo 4.0 stores 12 numbers per equality, mujoco_warp reads 11. The 12th is new and
    last; a site weld reads only the 11th (torquescale), so it is dropped."""
    from mujoco_warp._src import io, types

    create = io._create_array
    if getattr(create, "_eq_data_compat", False):
        return

    def _create_array(data, spec, sizes):
        if data is not None and getattr(spec, "dtype", None) is types.vec11 and np.shape(data)[-1:] == (12,):
            data = np.asarray(data)[..., :11]
        return create(data, spec, sizes)

    _create_array._eq_data_compat = True
    io._create_array = _create_array


_install_eq_data_compat()


def _tape_hands(spec: mujoco.MjSpec) -> None:
    for side in ("right", "left"):
        spec.add_equality(
            name=f"{side}_tape", type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_SITE,
            name1=f"robot/{side}_hand", name2=f"load/{side}_grip",
        )


def make_xiaoxiaobai_drag_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_xiaoxiaobai_velocity_env_cfg(play)
    # A flat floor: on the walk's height field the sled overflows MuJoCo's contacts per pair.
    cfg.scene.terrain = TerrainEntityCfg(terrain_type="plane")
    cfg.scene.entities = {"robot": XIAOXIAOBAI_DRAG_ROBOT_CFG, "load": LOAD_CFG}
    cfg.scene.spec_fn = _tape_hands
    cfg.episode_length_s = 10.0

    #---------------------------- Actions ---------------------------
    # The arms hold the drag pose (DRAG_* in xiaoxiaobai_constants.py): their actions are dropped.
    cfg.actions["joint_pos"].scale = {r"^(?!.*(shoulder|elbow)).*$": 1.0, r".*(shoulder|elbow).*": 0.0}

    #---------------------------- Commands --------------------------
    # Backward only, one speed per episode (the curriculum compares the distance). Without a
    # yaw command drag v8 drifted 25-40 deg over 8 s (p90 80) and 115 deg in the deploy sim, so
    # most envs hold their start heading through vtheta = DRAG_HEADING_KP * error (clipped to
    # DRAG_VTHETA), as the robot's heading hold feeds it; the rest turn at a constant vtheta.
    command = cfg.commands["twist"]
    command = xiaoxiaobai_mdp.DragVelocityCommandCfg(
        **{f.name: getattr(command, f.name) for f in fields(command) if f.init},
        heading_offset=DRAG_HEADING_OFFSET,
    )
    cfg.commands["twist"] = command
    command.ranges.lin_vel_x = DRAG_VX
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (-DRAG_VTHETA, DRAG_VTHETA)
    command.ranges.heading = (-math.pi, math.pi)
    command.heading_command = True
    command.heading_control_stiffness = DRAG_HEADING_KP
    command.rel_rotation_envs = 0.0
    command.rel_heading_envs = 1.0 if play else 0.8
    command.rel_standing_envs = 0.0 if play else 0.1
    command.direction_switch = None
    command.resampling_time_range = (1e6, 1e6)

    #---------------------------- Events ----------------------------
    del cfg.events["reset_base"], cfg.events["reset_robot_joints"]
    cfg.events["reset_drag"] = EventTermCfg(
        func=xiaoxiaobai_mdp.reset_drag,
        mode="reset",
        params={
            "load_offset": LOAD_OFFSET,
            "unit_inertia": LOAD_UNIT_INERTIA,
            "mass_levels": DRAG_MASS_LEVELS,
            "friction_range": DRAG_FRICTION,
            "initial_levels": DRAG_INITIAL_LEVELS,
            "light_fraction": 0.0 if play else DRAG_LIGHT_FRACTION,
            "free_fraction": 0.0 if play else DRAG_FREE_FRACTION,
        },
    )

    #---------------------------- Rewards ---------------------------
    # The load pulls the hands forward; the trunk leans back against it as far as it must.
    # Held to 0 deg (std 0.25), drag v2 leaned 2-4 deg forward under loads it could not move
    # and stopped there, at ~1.5 N of pull.
    cfg.rewards["upright"].params["pitch_per_vx"] = 0.0
    cfg.rewards["upright"].params["pitch_per_vx_back"] = None
    cfg.rewards["upright"].params["pitch_back_band"] = float(np.deg2rad(25.0))
    cfg.rewards["upright"].params["std"] = 0.15
    # At the walk's sqrt(0.1) half the commanded -0.2 m/s still kept 90% of the reward.
    cfg.rewards["track_linear_velocity"].params["std"] = 0.1
    # Alone, std 0.1 left no reward to climb towards the faster commands: drag v7 tracked to
    # -0.2 m/s, stamped in place from -0.22 and crept forward at -0.25 to -0.3.
    cfg.rewards["track_linear_velocity_coarse"] = replace(
        cfg.rewards["track_linear_velocity"], params={"command_name": "twist", "std": 0.3}, weight=1.0
    )
    # At std sqrt(0.5) drag v7 yawed ~70 deg over 8 s backing up in the deploy sim.
    cfg.rewards["track_angular_velocity"].params["std"] = 0.3
    # Shoulder roll is held at 0, inside the margin; the policy has no say in it.
    cfg.rewards.pop("arm_squeeze", None)

    #---------------------------- Terminations ----------------------
    # Held up by the load, a robot fallen onto it can stay above the walk's 0.10 m.
    cfg.terminations["tilted"] = TerminationTermCfg(
        func=xiaoxiaobai_mdp.fell, params={"min_height": 0.10, "max_tilt": 1.0}
    )

    #---------------------------- Curriculum ------------------------
    if not play:
        cfg.curriculum["drag_load"] = CurriculumTermCfg(
            func=xiaoxiaobai_mdp.drag_load_curriculum, params={"num_levels": len(DRAG_MASS_LEVELS)}
        )
        # XIAOXIAOBAI_DRAG_ARM_ANNEAL=<iterations>: from a policy that drove the arms (drag v5
        # pulled with them: arms locked it moved 0.25 kg at 4% of the command, and dropped
        # at once its fine-tune settled crouched at 0.05 kg), hand them over gradually.
        if os.environ.get("XIAOXIAOBAI_DRAG_ARM_ANNEAL"):
            steps = int(os.environ["XIAOXIAOBAI_DRAG_ARM_ANNEAL"]) * XiaoxiaobaiVelocityRlCfg.num_steps_per_env
            cfg.curriculum["drag_arm_authority"] = CurriculumTermCfg(
                func=xiaoxiaobai_mdp.drag_arm_authority, params={"anneal_steps": steps}
            )
    return cfg


XiaoxiaobaiDragRlCfg = replace(
    XiaoxiaobaiVelocityRlCfg,
    wandb_project="mjlab_xiaoxiaobai_drag",
    experiment_name="mjlab_xiaoxiaobai_drag",
)
