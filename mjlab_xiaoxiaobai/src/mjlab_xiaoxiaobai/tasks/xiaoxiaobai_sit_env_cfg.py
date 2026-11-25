# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

"""Xiaoxiaobai sit environment: sit down on a box behind the robot, rest, and stand back up.

On the floor the only seat that holds is leaning back on the hands (the hips flex to ~80 deg
only), and standing up from there means rolling onto the front first. A box of about tissue
box or book-stack height instead takes the weight with the trunk upright and the feet under
the knees. The box is random in size, mass and placement and the policy does not see it: it
sits back until the seat takes the weight, so any similar object will do.

Built on the get-up environment (robot, kp, contacts, penalties). The actor sees the walk's
observations plus the sit command (1 sit, 0 stand); the critic also sees the box.
"""

from dataclasses import replace

import mujoco
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.terrains import TerrainEntityCfg

from mjlab_xiaoxiaobai.tasks import mdp as xiaoxiaobai_mdp
from mjlab_xiaoxiaobai.tasks.xiaoxiaobai_getup_env_cfg import (
    ACTUATED,
    LYING_HEIGHT,
    SPEED_LIMITS,
    STAND_HEIGHT,
    XiaoxiaobaiGetupRlCfg,
    make_xiaoxiaobai_getup_env_cfg,
)

# Box half sizes [m]: depth, width, height. Seats 8-14 cm high: a scripted sit and stand
# worked at 11 cm, sat but did not stand up at 9 cm and could not sit at 7 cm (the knees
# bend to ~85 deg). A tissue box is ~9-12 cm high.
SEAT_HALF_SIZE = ((0.04, 0.10), (0.05, 0.125), (0.04, 0.07))
# Front face of the box behind the trunk origin [m]; the heels are ~2 cm behind it.
SEAT_GAP = (0.03, 0.09)
SEAT_LATERAL = (-0.02, 0.02)
SEAT_YAW = (-0.3, 0.3)
# The bodies that sit on the seat.
SEAT_BODIES = r"^(trunk|hip|hip_2|hip_block.*|femur.*)$"
# Backing up (Mjlab-Sit-Xiaoxiaobai-Backup): the box further back, the robot steps back until
# a leg touches it, then sits.
BACKUP_SEAT_GAP = (0.03, 0.30)
BACKUP_SPEED = 0.08
LEG_BODIES = r"^(tibia.*|ankle_block.*|foot|foot_2)$"


def get_seat_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="seat")
    body.add_freejoint(name="seat_joint")
    # Sizes are set per episode (reset_on_seat_front).
    body.add_geom(
        name="seat_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.06, 0.09, 0.05),
        mass=0.3,
        friction=(0.6, 0.005, 0.0001),
        rgba=(0.75, 0.55, 0.3, 1.0),
    )
    return spec


SEAT_CFG = EntityCfg(spec_fn=get_seat_spec)


def make_xiaoxiaobai_sit_env_cfg(play: bool = False, backup: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_xiaoxiaobai_getup_env_cfg(play=play)
    # A flat floor: on the walk's bumpy height field the box's base overflows MuJoCo's 50
    # height-field contacts per geom pair every step.
    cfg.scene.terrain = TerrainEntityCfg(terrain_type="plane")
    cfg.scene.entities = {**cfg.scene.entities, "seat": SEAT_CFG}
    cfg.episode_length_s = 15.0

    actuated = SceneEntityCfg("robot", joint_names=(ACTUATED,))
    sit = {"command_name": "sit"}
    seated = {**sit, "sensor_name": "seat_contact"}
    height = {"min_height": LYING_HEIGHT, "stand_height": STAND_HEIGHT}

    #---------------------------- Commands --------------------------
    cfg.commands["sit"] = xiaoxiaobai_mdp.SitCommandCfg(
        resampling_time_range=(2.5, 6.0), start_sit_prob=1.0 if play else 0.8
    )

    #---------------------------- Observations ----------------------
    for group in ("actor", "critic"):
        cfg.observations[group].terms["sit_command"] = ObservationTermCfg(
            func=envs_mdp.generated_commands, params=sit
        )
    cfg.observations["critic"].terms["seat_size"] = ObservationTermCfg(func=xiaoxiaobai_mdp.seat_size)
    cfg.observations["critic"].terms["seat_pos"] = ObservationTermCfg(func=xiaoxiaobai_mdp.seat_pos)

    #---------------------------- Events ----------------------------
    del cfg.events["reset_getup"], cfg.events["getup_assist"]
    cfg.curriculum = {}
    cfg.events["reset_on_seat_front"] = EventTermCfg(
        func=xiaoxiaobai_mdp.reset_on_seat_front,
        mode="reset",
        params={
            "half_size_range": SEAT_HALF_SIZE,
            "gap_range": SEAT_GAP,
            "lateral_range": SEAT_LATERAL,
            "yaw_range": SEAT_YAW,
            "seat_cfg": SceneEntityCfg("seat", geom_names=("seat_geom",)),
        },
    )
    # From a 0.1 kg tissue box, which the robot can shove away, to a 1 kg stack of books.
    cfg.events["seat_mass"] = EventTermCfg(
        mode="startup",
        func=dr.pseudo_inertia,
        params={"asset_cfg": SceneEntityCfg("seat", body_names=("seat",)), "alpha_range": (-0.55, 0.6)},
    )
    cfg.events["seat_friction"] = EventTermCfg(
        mode="startup",
        func=dr.geom_friction,
        params={"asset_cfg": SceneEntityCfg("seat", geom_names=("seat_geom",)), "operation": "abs", "ranges": (0.3, 1.0)},
    )
    cfg.scene.sensors = (
        *cfg.scene.sensors,
        ContactSensorCfg(
            name="seat_contact",
            primary=ContactMatch(mode="body", pattern=SEAT_BODIES, entity="robot"),
            secondary=ContactMatch(mode="body", pattern="seat", entity="seat"),
            fields=("found", "force"),
            reduce="netforce",
            num_slots=1,
        ),
    )

    #---------------------------- Rewards ---------------------------
    # The get-up's penalties (joint limits, current, motor load, speeds, contact slip) keep
    # the motion slow; its stand rewards apply while standing is commanded.
    for name in ("upright", "height", "not_ready", "stand_pose", "stand_still"):
        del cfg.rewards[name]
    for name, (start, _) in SPEED_LIMITS.items():
        cfg.rewards[name].params["max_speed"] = start
    cfg.rewards["trunk_upright"] = RewardTermCfg(func=xiaoxiaobai_mdp.trunk_upright, weight=1.0)
    cfg.rewards["seat_support"] = RewardTermCfg(func=xiaoxiaobai_mdp.seat_support, weight=2.0, params=seated)
    cfg.rewards["seated_rest"] = RewardTermCfg(
        func=xiaoxiaobai_mdp.seated_rest, weight=2.0, params={**seated, "current_scale": 1.0}
    )
    cfg.rewards["seated_still"] = RewardTermCfg(
        func=xiaoxiaobai_mdp.seated_still, weight=1.0, params={**seated, "std": 1.0, "joint_cfg": actuated}
    )
    cfg.rewards["stand_height"] = RewardTermCfg(
        func=xiaoxiaobai_mdp.stand_height, weight=2.0, params={**sit, "min_height": 0.10, "stand_height": STAND_HEIGHT}
    )
    cfg.rewards["stand_pose"] = RewardTermCfg(
        func=xiaoxiaobai_mdp.commanded_stand_pose,
        weight=2.0,
        params={**sit, "std": 0.3, "asset_cfg": actuated, **height},
    )
    cfg.rewards["stand_still"] = RewardTermCfg(
        func=xiaoxiaobai_mdp.commanded_stand_still,
        weight=1.0,
        params={**sit, "std": 1.0, "asset_cfg": actuated, **height},
    )
    cfg.rewards["seat_speed"] = RewardTermCfg(func=xiaoxiaobai_mdp.seat_speed, weight=-2.0)
    cfg.rewards["fell"] = RewardTermCfg(func=envs_mdp.is_terminated, weight=-20.0)

    #---------------------------- Terminations ----------------------
    cfg.terminations.pop("stood_still", None)
    cfg.terminations["fell"] = TerminationTermCfg(
        func=xiaoxiaobai_mdp.fell, params={"min_height": 0.07, "max_tilt": 1.05}
    )

    if backup:
        _add_backup(cfg, play)
    return cfg


def _add_backup(cfg: ManagerBasedRlEnvCfg, play: bool) -> None:
    # 30 cm at 8 cm/s plus sitting down.
    cfg.episode_length_s = 20.0
    cfg.commands["sit"].resampling_time_range = (4.0, 8.0)
    cfg.events["reset_on_seat_front"].params["gap_range"] = BACKUP_SEAT_GAP
    cfg.scene.sensors = (
        *cfg.scene.sensors,
        ContactSensorCfg(
            name="leg_seat_contact",
            primary=ContactMatch(mode="body", pattern=LEG_BODIES, entity="robot"),
            secondary=ContactMatch(mode="body", pattern="seat", entity="seat"),
            fields=("found",),
            reduce="none",
            num_slots=1,
        ),
    )
    touch = {"command_name": "sit", "sensor_names": ("leg_seat_contact", "seat_contact")}
    cfg.rewards["back_up"] = RewardTermCfg(
        func=xiaoxiaobai_mdp.back_up, weight=2.0, params={**touch, "speed": BACKUP_SPEED, "std": 0.05}
    )
    cfg.rewards["backing_after_touch"] = RewardTermCfg(
        func=xiaoxiaobai_mdp.backing_after_touch, weight=-10.0, params=touch
    )
    cfg.rewards["feet_slip"] = RewardTermCfg(
        func=xiaoxiaobai_mdp.feet_slip,
        weight=-2.0,
        params={
            "sensor_name": "feet_ground_contact",
            "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
        },
    )


XiaoxiaobaiSitRlCfg = replace(
    XiaoxiaobaiGetupRlCfg,
    experiment_name="mjlab_xiaoxiaobai_sit",
    wandb_project="mjlab_xiaoxiaobai_sit",
    max_iterations=5_000,
)
