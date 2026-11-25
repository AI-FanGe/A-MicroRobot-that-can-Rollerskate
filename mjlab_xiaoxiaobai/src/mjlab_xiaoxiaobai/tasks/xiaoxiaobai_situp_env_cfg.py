# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

"""Xiaoxiaobai sit-up environment: wake up and go back to sleep on the floor.

Starting flat on the back, the sit command (1) has the robot push itself up slowly on both
hands and sit leaning back on them; the lie command (0) has it lower itself back down on
the hands and lie still with the arms along the sides. The command toggles every few
seconds, so one policy learns both and the robot switches them with a button.

Built on the get-up environment (robot, kp, contacts, speed / current / slip penalties).
The actor sees the walk's observations plus the sit command, as the sit-on-a-box task.
"""

from dataclasses import replace

import numpy as np
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.terrains import TerrainEntityCfg

from mjlab_xiaoxiaobai.tasks import mdp as xiaoxiaobai_mdp
from mjlab_xiaoxiaobai.tasks.xiaoxiaobai_getup_env_cfg import ACTUATED, XiaoxiaobaiGetupRlCfg, make_xiaoxiaobai_getup_env_cfg

# Sitting: trunk within this of vertical. The hips reach ~80 deg, so with the legs flat
# the trunk leans back at least ~15 deg; the hands reach the floor from 20 to 60 deg.
SIT_TILT = np.deg2rad(35.0)
HANDS = r"^radius(_2)?$"
# Slow: the trunk turns through the ~60 deg between lying and sitting in about 1.5 s or more.
SPEED_LIMITS: dict[str, float] = {
    "joint_speed": 2.5,
    "trunk_ang_speed": 0.8,
    "body_speed": 0.3,
    "root_speed_xy": 0.1,
}
LEG_SPLAY = r"^(left|right)_(hip_yaw|hip_roll|ankle_roll)$"


def make_xiaoxiaobai_situp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_xiaoxiaobai_getup_env_cfg(play=play)
    cfg.scene.terrain = TerrainEntityCfg(terrain_type="plane")
    cfg.episode_length_s = 20.0

    actuated = SceneEntityCfg("robot", joint_names=(ACTUATED,))
    feet = SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))
    sit = {"command_name": "sit"}
    hands = {**sit, "sit_tilt": SIT_TILT, "sensor_name": "hand_ground"}

    #---------------------------- Commands --------------------------
    # Every resample toggles: sit, lie, sit, ... Mostly starts with the sit command, as
    # lying still from lying is easy.
    cfg.commands["sit"] = xiaoxiaobai_mdp.SitCommandCfg(
        resampling_time_range=(4.0, 7.0), start_sit_prob=1.0 if play else 0.85
    )

    #---------------------------- Observations ----------------------
    for group in ("actor", "critic"):
        cfg.observations[group].terms["sit_command"] = ObservationTermCfg(
            func=envs_mdp.generated_commands, params=sit
        )

    #---------------------------- Events ----------------------------
    del cfg.events["getup_assist"]
    cfg.curriculum = {}
    # Flat on the back with the limbs near the ready pose, as the robot is put down to sleep.
    cfg.events["reset_getup"].params.update(
        {
            "standing_fraction": 0.0,
            "flat_fraction": 1.0,
            "supine_fraction": 1.0,
            "flat_tilt": 0.15,
            "joint_pos_range": 0.3,
            "flat_joint_pos_range": 0.3,
            "flat_drop_height": 0.08,
        }
    )
    if not play:
        cfg.events["push_robot"].params["velocity_range"] = {"x": (-0.2, 0.2), "y": (-0.2, 0.2)}
    cfg.scene.sensors = (
        *cfg.scene.sensors,
        ContactSensorCfg(
            name="hand_ground",
            primary=ContactMatch(mode="body", pattern=HANDS, entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found",),
            reduce="maxforce",
            num_slots=1,
        ),
    )

    #---------------------------- Rewards ---------------------------
    kept = (
        "dof_pos_limits", "action_rate_l2", "joint_vel_l2", "self_collisions", "command_lead_excess",
        "total_current", "motor_load", "motor_load_l2", "root_speed_xy", "body_speed", "trunk_ang_speed", "joint_speed",
        "friction_use", "contact_slip",
    )
    rewards = {name: replace(cfg.rewards[name]) for name in kept}
    for name, limit in SPEED_LIMITS.items():
        rewards[name].params = {**rewards[name].params, "max_speed": limit}
    cfg.rewards = {
        "sit_up": RewardTermCfg(func=xiaoxiaobai_mdp.sit_up_level, weight=3.0, params={**sit, "sit_tilt": SIT_TILT}),
        "sit_hands": RewardTermCfg(func=xiaoxiaobai_mdp.sit_up_hands, weight=1.5, params=hands),
        "sat_rest": RewardTermCfg(func=xiaoxiaobai_mdp.sat_rest, weight=1.0, params={**hands, "current_scale": 3.0}),
        "sat_still": RewardTermCfg(
            func=xiaoxiaobai_mdp.sat_still, weight=1.0, params={**hands, "std": 1.0, "joint_cfg": actuated}
        ),
        "lie": RewardTermCfg(func=xiaoxiaobai_mdp.lie_level, weight=3.0, params=sit),
        "lain_pose": RewardTermCfg(
            func=xiaoxiaobai_mdp.lain_pose, weight=1.5, params={**sit, "std": 0.3, "joint_cfg": actuated}
        ),
        "lain_rest": RewardTermCfg(func=xiaoxiaobai_mdp.lain_rest, weight=1.0, params={**sit, "current_scale": 1.0}),
        "lain_still": RewardTermCfg(
            func=xiaoxiaobai_mdp.lain_still, weight=1.0, params={**sit, "std": 1.0, "joint_cfg": actuated}
        ),
        "feet_lifted": RewardTermCfg(
            func=xiaoxiaobai_mdp.feet_lifted, weight=-1.0, params={"min_height": 0.07, "span": 0.04, "feet_cfg": feet}
        ),
        # Legs together and pointing ahead, as people sit and lie.
        "leg_splay": RewardTermCfg(
            func=xiaoxiaobai_mdp.joint_deviation_l2,
            weight=-1.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(LEG_SPLAY,))},
        ),
        "fell": RewardTermCfg(func=envs_mdp.is_terminated, weight=-20.0),
        **rewards,
    }

    #---------------------------- Terminations ----------------------
    cfg.terminations.pop("stood_still", None)
    cfg.terminations["left_the_floor"] = TerminationTermCfg(
        func=xiaoxiaobai_mdp.left_the_floor, params={"max_gx": 0.6, "max_gy": 0.85, "max_height": 0.13}
    )
    return cfg


XiaoxiaobaiSitupRlCfg = replace(
    XiaoxiaobaiGetupRlCfg,
    experiment_name="mjlab_xiaoxiaobai_situp",
    wandb_project="mjlab_xiaoxiaobai_situp",
    max_iterations=8_000,
)
