# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

"""Xiaoxiaobai get-up environment: stand up from any fallen pose.

Built on the velocity environment, so observations, actions, actuators and domain
randomization match the walk policy and the ONNX runs on the robot like the walk with a
zero velocity command. The robot uses the measured joint stops and frictional contacts on
every box; rewards, resets and terminations are the get-up ones.
"""

from dataclasses import replace

import numpy as np
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from mjlab_xiaoxiaobai.robot.xiaoxiaobai_constants import GETUP_KP, XIAOXIAOBAI_GETUP_ROBOT_CFG
from mjlab_xiaoxiaobai.tasks import mdp as xiaoxiaobai_mdp
from mjlab_xiaoxiaobai.tasks.xiaoxiaobai_velocity_env_cfg import XiaoxiaobaiVelocityRlCfg, make_xiaoxiaobai_velocity_env_cfg

# Policy joints: the walk's action filter (no head yaw, no passive head_wobble/leg_flex).
ACTUATED = r"^(?!head|leg_flex).*$"
# Trunk height [m]: ~0.033 lying flat (median of random drops), 0.181 in the ready pose.
LYING_HEIGHT = 0.035
STAND_HEIGHT = 0.175
# Speed penalty limits (start, end of the getup_speed curriculum), already slow at the start:
# lowering them from the dynamic get-up's peaks (12 rad/s joints, 6.5 rad/s trunk) never
# got past about 8 and 4.4, the policy gave up getting up instead of slowing down. The
# assist carries the slow get-up until the policy pushes itself up. Joint and trunk angular
# speeds in rad/s, body and trunk horizontal speeds in m/s.
SPEED_LIMITS: dict[str, tuple[float, float]] = {
    "joint_speed": (4.0, 2.5),
    "trunk_ang_speed": (2.0, 1.2),
    "body_speed": (0.6, 0.35),
    "root_speed_xy": (0.2, 0.1),
}


def make_xiaoxiaobai_getup_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_xiaoxiaobai_velocity_env_cfg(play=play)
    cfg.scene.entities = {"robot": XIAOXIAOBAI_GETUP_ROBOT_CFG}
    # Lying on the ground with friction on every box: up to 28 contacts and 131 constraint
    # rows per robot in random drops on a plane.
    cfg.sim.nconmax = 96
    cfg.sim.njmax = 400
    cfg.episode_length_s = 10.0

    actuated = SceneEntityCfg("robot", joint_names=(ACTUATED,))
    feet = SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))
    height = {"min_height": LYING_HEIGHT, "stand_height": STAND_HEIGHT}
    # Stand rewards stay off until the face and both feet point down, then the front get-up
    # (already learned) is paid exactly as before.
    rolled = {"feet_cfg": feet}

    #---------------------------- Commands --------------------------
    # Always zero; kept so the observation layout is the walk's.
    twist = cfg.commands["twist"]
    twist.rel_standing_envs = 1.0
    twist.rel_rotation_envs = 0.0
    twist.ranges.lin_vel_x = (0.0, 0.0)
    twist.ranges.lin_vel_y = (0.0, 0.0)
    twist.ranges.ang_vel_z = (0.0, 0.0)
    twist.rotation_env_ang_vel_range = (0.0, 0.0)

    #---------------------------- Observations ----------------------
    cfg.observations["critic"].terms["root_height"] = ObservationTermCfg(func=xiaoxiaobai_mdp.root_height)

    #---------------------------- Events ----------------------------
    del cfg.events["reset_base"], cfg.events["reset_robot_joints"]
    # Stands up from the front only: on the back or a side the roll-over policy runs first
    # and hands over face down (Mjlab-Rollover-Xiaoxiaobai). Flat on the front with the limbs
    # around the ready pose, as real falls and the roll-over end; a few random drops and
    # standing starts so it also copes with odd poses and stays up.
    cfg.events["reset_getup"] = EventTermCfg(
        func=xiaoxiaobai_mdp.reset_getup,
        mode="reset",
        params={
            "standing_fraction": 0.05,
            "drop_height": 0.25,
            "joint_pos_range": 1.0,
            "flat_fraction": 0.9,
            "supine_fraction": 0.0,
            "flat_joint_pos_range": 0.4,
            "asset_cfg": actuated,
        },
    )
    if not play:
        cfg.events["push_robot"].params["velocity_range"] = {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}
    # Plastic shells on floors: the arms, knees and trunk slide or grip. Up to 2.0 because the
    # policy trained at a fixed 1.0 got stuck on the robot's grippier floor (sim at 2.0: 61%
    # instead of 99%). The ground hulls outrank the terrain, so this is the contact friction.
    cfg.events["body_friction"] = EventTermCfg(
        mode="startup",
        func=dr.geom_friction,
        params={
            "asset_cfg": SceneEntityCfg("robot", geom_names=(r".*_ground_collision$",)),
            "operation": "abs",
            "ranges": (0.3, 2.0),
            "shared_random": True,
        },
    )
    cfg.events["foot_friction"].params["ranges"] = (0.5, 1.5)
    cfg.scene.sensors = (
        *cfg.scene.sensors,
        ContactSensorCfg(
            name="ground_contact",
            primary=ContactMatch(mode="body", pattern=r".*", entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "pos"),
            reduce="maxforce",
            num_slots=1,
        ),
        ContactSensorCfg(
            name="ground_force",
            primary=ContactMatch(mode="body", pattern=r".*", entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"),
            reduce="netforce",
            num_slots=1,
        ),
    )

    #---------------------------- Rewards ---------------------------
    # Dense progress (upright, height) plus the ready pose and stillness once standing;
    # joint limits, self contact and command lead are penalised, never clipped.
    cfg.rewards = {
        "upright": RewardTermCfg(func=xiaoxiaobai_mdp.getup_upright, weight=1.0, params=rolled),
        "height": RewardTermCfg(func=xiaoxiaobai_mdp.getup_height, weight=1.0, params={**height, **rolled}),
        # On the back this stays on even if the chest lifts, until the face and both feet
        # point down. Lying face down pays nothing extra, so it is not worth staying there.
        "not_ready": RewardTermCfg(func=xiaoxiaobai_mdp.getup_not_ready, weight=-1.5, params=rolled),
        "stand_pose": RewardTermCfg(
            func=xiaoxiaobai_mdp.getup_stand_pose, weight=2.0, params={"std": 0.3, "asset_cfg": actuated, **height}
        ),
        "stand_still": RewardTermCfg(
            func=xiaoxiaobai_mdp.getup_stand_still, weight=1.0, params={"std": 1.0, "asset_cfg": actuated, **height}
        ),
        # The soft limits are 90% of the measured stops.
        "dof_pos_limits": RewardTermCfg(func=envs_mdp.joint_pos_limits, weight=-10.0, params={"asset_cfg": actuated}),
        "action_rate_l2": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.05),
        "joint_vel_l2": RewardTermCfg(func=envs_mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": actuated}),
        "self_collisions": replace(cfg.rewards["self_collisions"], weight=-0.3),
        # The walk's 150 deg at kp 125, scaled to where kp GETUP_KP reaches full PWM.
        "command_lead_excess": replace(
            cfg.rewards["command_lead_excess"], params={"max_lead": np.deg2rad(150.0 * 125 / GETUP_KP)}
        ),
        # Quasi-static get-up: pushed up on the legs, not thrown up. v11 peaked at 12 rad/s on
        # the joints (the motors' top speed) and 6.5 rad/s on the trunk, which the robot's
        # motors do not deliver. The getup_speed curriculum lowers these limits from about
        # those peaks for as long as the robot still stands up; the weights keep them binding.
        # Current is kept clear of the deploy safety cut at 20 A (constants.OVERCURRENT_CUTOFF_A).
        "total_current": RewardTermCfg(
            func=xiaoxiaobai_mdp.total_current_excess, weight=-0.5, params={"max_current": 15.0}
        ),
        # Every motor well below its current limit, the load shared between motors: the
        # robot could not get up with motors held at the limit as in sim.
        "motor_load": RewardTermCfg(func=xiaoxiaobai_mdp.motor_load_excess, weight=-10.0, params={"max_load": 0.5}),
        "motor_load_l2": RewardTermCfg(func=xiaoxiaobai_mdp.motor_load_l2, weight=-0.1),
        "root_speed_xy": RewardTermCfg(
            func=xiaoxiaobai_mdp.root_speed_xy_excess, weight=-20.0, params={"max_speed": SPEED_LIMITS["root_speed_xy"][0]}
        ),
        "body_speed": RewardTermCfg(
            func=xiaoxiaobai_mdp.body_speed_excess, weight=-20.0, params={"max_speed": SPEED_LIMITS["body_speed"][0]}
        ),
        "trunk_ang_speed": RewardTermCfg(
            func=xiaoxiaobai_mdp.trunk_ang_speed_excess, weight=-5.0, params={"max_speed": SPEED_LIMITS["trunk_ang_speed"][0]}
        ),
        "joint_speed": RewardTermCfg(
            func=xiaoxiaobai_mdp.joint_speed_excess,
            weight=-2.0,
            params={"max_speed": SPEED_LIMITS["joint_speed"][0], "asset_cfg": actuated},
        ),
        # Push up on the joints, not by scraping the shell along the floor: contacts should not
        # slide, and should hold by static friction on any floor above 0.3 (v9 used over half
        # of a 1.0 friction in a third of its loaded contacts).
        "friction_use": RewardTermCfg(
            func=xiaoxiaobai_mdp.ground_friction_use, weight=-1.0, params={"sensor_name": "ground_force", "max_ratio": 0.3}
        ),
        "contact_slip": RewardTermCfg(
            func=xiaoxiaobai_mdp.ground_contact_slip, weight=-2.0, params={"sensor_name": "ground_contact"}
        ),
    }

    #---------------------------- Terminations ----------------------
    del cfg.terminations["fell_over"]
    # Standing for a second ends the episode, so the samples are get-ups from new poses
    # rather than standing; the robot hands over to the walk policy there anyway.
    if not play:
        cfg.terminations["stood_still"] = TerminationTermCfg(
            func=xiaoxiaobai_mdp.getup_stood_still, time_out=True, params={"hold_s": 1.0, **height}
        )

    #---------------------------- Curriculum ------------------------
    # Assist: the trunk is pulled up by a share of the body weight that fades to zero as more
    # episodes end standing (play runs without it). It starts at half the body weight so the
    # slow get-up is learnt before the policy has to push itself up.
    cfg.events["getup_assist"] = EventTermCfg(
        func=xiaoxiaobai_mdp.getup_assist_force,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk",))},
    )
    cfg.curriculum = {}
    if not play:
        cfg.curriculum["getup_assist"] = CurriculumTermCfg(
            func=xiaoxiaobai_mdp.getup_assist_curriculum,
            # The slow get-up held at 0.57 success with 0.145 assist for 1000 iterations,
            # between the old 0.6 / 0.35 thresholds, so the assist never faded further.
            # From scratch the assist starts at "initial"; set "start" (e.g. 0.15) to fine-tune.
            params={"initial": 0.6, "step": 0.05, "success_high": 0.5, "success_low": 0.3, "start": None, **height},
        )
        # As slow as it can go while still standing up: the speed limits come down while
        # nearly every episode ends standing and go back up when that drops.
        cfg.curriculum["getup_speed"] = CurriculumTermCfg(
            func=xiaoxiaobai_mdp.getup_speed_curriculum,
            params={"limits": SPEED_LIMITS, "step": 0.01, "success_high": 0.95, "success_low": 0.85},
        )

    return cfg


XiaoxiaobaiGetupRlCfg = replace(
    XiaoxiaobaiVelocityRlCfg,
    experiment_name="mjlab_xiaoxiaobai_getup",
    wandb_project="mjlab_xiaoxiaobai_getup",
    max_iterations=10_000,
)
