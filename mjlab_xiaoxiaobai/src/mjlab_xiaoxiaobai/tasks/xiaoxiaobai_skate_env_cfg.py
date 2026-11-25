# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

"""Roller skating on the skate shoes: one skate pushes off sideways and lifts while the other
glides, then they swap. Built on the walk task, whose policy it starts from; tile and carpet
differ in the wheels' rolling resistance and grip, randomised per robot."""

import math
import os
import re
from copy import deepcopy
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.rewards import action_acc_l2, joint_vel_l2
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.terrains import BoxFlatTerrainCfg, HfRandomUniformTerrainCfg

from mjlab_xiaoxiaobai.robot.xiaoxiaobai_constants import BACKPACK_MASS, XIAOXIAOBAI_SKATE_ROBOT_CFG
from mjlab_xiaoxiaobai.tasks import mdp as xiaoxiaobai_mdp
from mjlab_xiaoxiaobai.tasks.xiaoxiaobai_velocity_env_cfg import XiaoxiaobaiVelocityRlCfg, make_xiaoxiaobai_velocity_env_cfg

FEET = ("left_foot", "right_foot")
# 1.133 kg robot and two 22.6 g skates (and XIAOXIAOBAI_BACKPACK's 130 g).
SKATE_ROBOT_WEIGHT = (1.178 + BACKPACK_MASS) * 9.81


def make_xiaoxiaobai_skate_env_cfg(play: bool = False, skate_base: bool = True) -> ManagerBasedRlEnvCfg:
    cfg = make_xiaoxiaobai_velocity_env_cfg(play)

    cfg.scene.entities = {"robot": XIAOXIAOBAI_SKATE_ROBOT_CFG}
    # Floors are flat; carpet pile and tile joints leave up to ~2 mm under a 20 mm wheel.
    generator = deepcopy(cfg.scene.terrain.terrain_generator)
    generator.sub_terrains = {
        "flat": BoxFlatTerrainCfg(proportion=0.5),
        "bumpy": replace(generator.sub_terrains["bumpy"], proportion=0.5, noise_range=(0.0, 0.002), noise_step=0.001),
    }
    cfg.scene.terrain.terrain_generator = generator

    # The wheels are passive and carry no encoder: out of observations and joint randomisation.
    cfg.observations["actor"].terms["joint_pos"].params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=(r"^(?!head|leg_flex|.*_wheel_).*$",)
    )
    cfg.observations["actor"].terms["joint_vel"].params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=(r"^(?!head|leg_flex|.*_wheel_).*$",)
    )
    for name in ("joint_pos", "joint_vel"):
        cfg.observations["critic"].terms[name].params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=(r"^(?!head_wobble|leg_flex|.*_wheel_).*$",)
        )
    not_wheel = SceneEntityCfg("robot", joint_names=(r"^(?!.*_wheel_).*$",))
    cfg.events["reset_robot_joints"].params["asset_cfg"] = not_wheel
    cfg.events["encoder_bias"].params["asset_cfg"] = not_wheel
    cfg.events["dof_armature_randomization"].params["asset_cfg"] = not_wheel
    cfg.events["dof_friction_randomization"].params["asset_cfg"] = not_wheel

    #---------------------------- Floors ----------------------------
    # Grip of polyurethane wheels: ~0.25 on dusty tile up to ~1.6 on carpet.
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = (r".*_wheel_.*_collision",)
    cfg.events["foot_friction"].params["ranges"] = (0.25, 1.6)
    # Rolling resistance c_rr ~0.005 on tile to ~0.35 on thick carpet (the 20 mm wheels sink
    # into the pile), times the 10 mm wheel radius. MuJoCo's soft rolling constraint delivers
    # ~70-85% of the set torque, hence the top end.
    cfg.events["wheel_rolling_resistance"] = EventTermCfg(
        mode="startup",
        func=dr.geom_friction,
        params={
            "asset_cfg": SceneEntityCfg("robot", geom_names=(r".*_wheel_.*_collision",)),
            "operation": "abs",
            "distribution": "log_uniform",
            "axes": [2],
            "ranges": (5e-5, 5.0e-3),
            "shared_random": True,
        },
    )
    # Bearings: Coulomb 3e-4-4e-3 N m per wheel (c_rr 0.02-0.28 more at the ~1.4 N a wheel
    # carries in double support) and viscous up to 2e-5 N m s/rad (1e-3 N m at 0.5 m/s). The
    # robot barely moved on a wooden desk, which sat at the top of an earlier 2e-3 range, and
    # model_12500 still laboured on carpet.
    cfg.events["wheel_bearing_friction"] = EventTermCfg(
        mode="startup",
        func=dr.joint_friction,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*_wheel_.*",)),
            "operation": "abs",
            "distribution": "log_uniform",
            "ranges": (3e-4, 4e-3),
            "shared_random": True,
        },
    )
    cfg.events["wheel_bearing_damping"] = EventTermCfg(
        mode="startup",
        func=dr.joint_damping,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*_wheel_.*",)),
            "operation": "abs",
            "ranges": (0.0, 2e-5),
            "shared_random": True,
        },
    )

    #---------------------------- Commands --------------------------
    # Skating forward and turning; skates neither step sideways nor spin in place.
    command = cfg.commands["twist"]
    command.ranges.lin_vel_x = (0.0, 0.5)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (-0.5, 0.5)
    command.rel_rotation_envs = 0.0
    command.rel_standing_envs = 0.1

    #---------------------------- Rewards ---------------------------
    # At the walk's sqrt(0.5) a skater turning 0.23 of a 0.4 rad/s command kept 94% of the
    # reward; at 0.35 it keeps ~79%.
    cfg.rewards["track_angular_velocity"].params["std"] = 0.35
    cfg.rewards["track_angular_velocity"].weight = 3.0
    # At the walk's 0.2 a swizzle doing 0.455 m/s for 0.3 on tile kept 55% of the reward.
    cfg.rewards["track_linear_velocity"].params["std"] = 0.15
    cfg.rewards["track_linear_velocity"].weight = 3.0
    # At 0.15 skating 0.1 m/s for 0.5 earns as little as standing (<1e-3), so on heavy floors
    # the policy stood still: at c_rr 0.25 it swung the hip yaw 1.7 deg at 34% motor load.
    # At 0.5 it earns 0.53 of the weight against 0.37 standing.
    cfg.rewards["track_linear_velocity_coarse"] = deepcopy(cfg.rewards["track_linear_velocity"])
    cfg.rewards["track_linear_velocity_coarse"].params["std"] = 0.5
    cfg.rewards["track_linear_velocity_coarse"].weight = 1.5
    # At weight 10, 0.15 m/s of a 0.5 command earns 1.5 where both tracking kernels earn ~0.
    cfg.rewards["commanded_forward_speed"] = RewardTermCfg(
        func=xiaoxiaobai_mdp.commanded_forward_speed, weight=10.0,
    )

    # The walk leans back with speed against a foot that stops; a gliding skate does not.
    cfg.rewards["upright"].params["pitch_per_vx"] = 0.0

    std_walking = cfg.rewards["pose"].params["std_walking"]
    # Pushing off turns the skate out (hip yaw), spreads the leg (hip roll) and sets the skate
    # on its inner edge (ankle roll). At 0.4/0.3/0.3 the pushing skate turned out only ~10 deg,
    # so the push went 0.26 body weights sideways for 0.06 forward.
    std_walking.update({r".*hip_yaw.*": 0.8, r".*hip_roll.*": 0.35, r".*ankle_roll.*": 0.5,
                        r".*knee.*": 0.3})
    cfg.rewards["pose"].params["std_running"] = std_walking
    # The knees bend in a crouch to push: model_12500 skated at 53 deg, model_14999 at 36.
    # Bent past 30 deg they cost; at weight -40 a knee at 40 deg costs 1.2 a step.
    cfg.rewards["knee_bend"] = RewardTermCfg(
        func=xiaoxiaobai_mdp.joint_opening_excess,
        weight=-40.0,
        params={
            "max_opening": math.radians(30.0),
            "signs": (1.0, -1.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=("right_knee", "left_knee"), preserve_order=True),
        },
    )
    # The hips roll the legs outward (left negative, right positive) to push; on the robot
    # the model_12500 spread (mean 19 deg, p90 30 deg, max 34 deg) was already too wide.
    # At weight -50 a hip at 32 deg costs 0.37 a step.
    cfg.rewards["hip_roll_opening"] = RewardTermCfg(
        func=xiaoxiaobai_mdp.joint_opening_excess,
        weight=-50.0,
        params={
            "max_opening": math.radians(25.0),
            "signs": (-1.0, 1.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=("left_hip_roll", "right_hip_roll"),
                                        preserve_order=True),
        },
    )

    # A gliding skate moves on the ground by design; only sideways scraping is slip.
    del cfg.rewards["foot_slip"]
    cfg.rewards["skate_side_slip"] = RewardTermCfg(
        func=xiaoxiaobai_mdp.skate_side_slip,
        weight=-5.0,
        params={"sensor_name": "feet_ground_contact", "asset_cfg": SceneEntityCfg("robot", site_names=FEET)},
    )

    # Push, lift and hang one skate for 0.15-0.6 s while the other glides; past 0.6 s the
    # lifted skate earns nothing until it is set down and the other one lifts. Air time pays
    # every step a skate is up, so a large weight squeezes out the two-skate push phase.
    cfg.rewards["air_time"].params["threshold_min"] = 0.15
    cfg.rewards["air_time"].params["threshold_max"] = 0.6
    cfg.rewards["air_time"].weight = 1.0
    cfg.rewards["no_stepping"].weight = -1.0
    cfg.rewards["single_skate_glide"] = RewardTermCfg(
        func=xiaoxiaobai_mdp.single_skate_glide,
        weight=0.5,
        params={"sensor_name": "feet_ground_contact", "max_air_time": 0.6,
                "body_weight": SKATE_ROBOT_WEIGHT, "push_threshold": 0.01},
    )

    # Pushing: the floor's forward push on the skates while short of the commanded speed,
    # braking counted double. At weight 3 it was worth ~0.01 per step and changed nothing; at
    # 20, uncapped and paid at any speed, the policy stomped and fell 4.8 times per episode.
    stroke_params = {"sensor_name": "feet_ground_contact", "body_weight": SKATE_ROBOT_WEIGHT,
                     "max_push": 0.5, "speed_band": 0.05}
    cfg.rewards["skate_propulsion"] = RewardTermCfg(
        func=xiaoxiaobai_mdp.skate_propulsion, weight=5.0, params=dict(stroke_params),
    )
    # The skate gliding alone points along the line of travel; the pushing one turns out freely.
    cfg.rewards["skate_glide_misalignment"] = RewardTermCfg(
        func=xiaoxiaobai_mdp.skate_glide_misalignment,
        weight=-5.0,
        params={"sensor_name": "feet_ground_contact", "asset_cfg": SceneEntityCfg("robot", site_names=FEET)},
    )
    # The stroke itself: driven back under load, lifted off behind the trunk.
    cfg.rewards["skate_push_power"] = RewardTermCfg(
        func=xiaoxiaobai_mdp.skate_push_power,
        weight=10.0,
        params={**stroke_params, "asset_cfg": SceneEntityCfg("robot", site_names=FEET)},
    )
    cfg.rewards["skate_liftoff_behind"] = RewardTermCfg(
        func=xiaoxiaobai_mdp.skate_liftoff_behind,
        weight=200.0,
        params={"sensor_name": "feet_ground_contact", "max_dist": 0.06,
                "asset_cfg": SceneEntityCfg("robot", site_names=FEET)},
    )
    # The lifted skate rises straight up and sets down under or behind the trunk to push,
    # instead of stepping ahead like the walk it started from.
    cfg.rewards["skate_touchdown_ahead"] = RewardTermCfg(
        func=xiaoxiaobai_mdp.skate_touchdown_ahead,
        weight=-100.0,
        params={"sensor_name": "feet_ground_contact", "asset_cfg": SceneEntityCfg("robot", site_names=FEET)},
    )
    cfg.rewards["skate_lift_forward"] = RewardTermCfg(
        func=xiaoxiaobai_mdp.skate_lift_forward,
        weight=-2.0,
        params={"sensor_name": "feet_ground_contact", "lift_time": 0.2,
                "asset_cfg": SceneEntityCfg("robot", site_names=FEET)},
    )
    # The arms hold something in front of the chest: shoulders pitched 20 deg forward, elbows
    # bent 60 deg, shoulder roll at rest. Left free (pose std 0.8) they swung 35 deg behind the
    # trunk to push. Measured from that pose instead of the default, which hangs them down.
    pose = cfg.rewards["pose"].params
    pose["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=(r"^(?!leg_flex|.*_wheel_|left_shoulder|left_elbow|right_shoulder|right_elbow).*$",)
    )
    for key in ("std_standing", "std_walking", "std_running"):
        pose[key] = {k: v for k, v in pose[key].items() if not re.search(r"shoulder|elbow", k)}
    hold = {"right_shoulder_pitch": 20.0, "left_shoulder_pitch": -20.0,
            "right_shoulder_roll": 5.0, "left_shoulder_roll": -5.0,
            "right_elbow": -60.0, "left_elbow": 60.0}
    # At weight -3, 10 deg off on every arm joint costs 0.55 a step.
    cfg.rewards["arm_hold_pose"] = RewardTermCfg(
        func=xiaoxiaobai_mdp.joint_target_l2,
        weight=-3.0,
        params={"targets": tuple(math.radians(v) for v in hold.values()),
                "asset_cfg": SceneEntityCfg("robot", joint_names=tuple(hold), preserve_order=True)},
    )
    # The skates swizzle in step, so the arms move as mirror images. Paying arm speed (weight
    # 0.2) was farmed by shaking them.
    arm_pairs = tuple(f"{side}_{joint}" for joint in ("shoulder_pitch", "shoulder_roll", "elbow")
                      for side in ("left", "right"))
    cfg.rewards["arm_symmetry"] = RewardTermCfg(
        func=xiaoxiaobai_mdp.mirror_pose_asymmetry,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=arm_pairs, preserve_order=True)},
    )
    # Held still: at -0.05 an arm swinging 10 deg at 2 Hz costs ~0.1 a step per joint.
    cfg.rewards["arm_joint_vel"] = RewardTermCfg(
        func=joint_vel_l2,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(r".*shoulder_.*", r".*elbow"))},
    )

    # On the robot this policy shook at a high frequency once set on the floor; in sim it
    # changed its leg targets ~40% more per tick than the deployed walk while standing.
    # On the mean action the walk's -0.3 cost ~0.5% of the tracking reward; together these
    # cost ~4%. The second difference singles out the alternating, high-frequency part.
    cfg.rewards["action_rate_l2"].weight = -1.0
    cfg.rewards["action_acc_l2"] = RewardTermCfg(func=action_acc_l2, weight=-0.5)

    # Room for the sideways push; the skates' inner wheels still clear each other at 0.070.
    cfg.rewards["feet_too_wide"].params["max_dist"] = 0.15

    # Each motor on its own, as on the robot. Below 0.9 of the current limit is free; one
    # motor held at the limit costs 0.06 a step. At max_load 0.7 the policy pushed at half
    # the limit and stopped on heavy floors.
    cfg.rewards["motor_load"] = RewardTermCfg(
        func=xiaoxiaobai_mdp.motor_load_excess, weight=-6.0, params={"max_load": 0.9}
    )

    # The walk's low-voltage terms hold the walk's ready pose, not a skating stance.
    for name in ("standing_crouch", "arm_squeeze"):
        cfg.rewards.pop(name, None)

    # XIAOXIAOBAI_SKATE_BASE=s5: the rewards and bearings the deployed skate.onnx (skate_s5_maxpush
    # model_12500) was trained under, for fine-tunes of it; tasks built on this one pass
    # skate_base=False.
    if skate_base and os.environ.get("XIAOXIAOBAI_SKATE_BASE") == "s5":
        for name in ("knee_bend", "hip_roll_opening", "arm_hold_pose"):
            del cfg.rewards[name]
        pose["asset_cfg"] = SceneEntityCfg("robot", joint_names=(r"^(?!leg_flex|.*_wheel_).*$",))
        pose["std_standing"].update({r".*shoulder_pitch.*": 0.1, r".*shoulder_roll.*": 0.1, r".*elbow.*": 0.1})
        pose["std_walking"].update({r".*shoulder_pitch.*": 0.8, r".*shoulder_roll.*": 0.5, r".*elbow.*": 0.4,
                                    r".*hip_roll.*": 0.5, r".*knee.*": 0.4})
        pose["std_running"] = pose["std_walking"]
        cfg.rewards["arm_joint_vel"].weight = -0.02
        cfg.rewards["commanded_forward_speed"].weight = 8.0
        cfg.events["wheel_bearing_friction"].params["ranges"] = (1e-4, 4e-3)

    #---------------------------- Pushes ----------------------------
    # Shoves only once the stroke is learnt: the walk's +-0.5 m/s stopped or reversed a
    # 1.2 kg robot on free wheels before it could push. The step counter restarts on resume.
    cfg.events["push_robot"].params["velocity_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0)}

    def set_push(env, speed: float) -> None:
        env.event_manager.get_term_cfg("push_robot").params["velocity_range"] = {
            "x": (-speed, speed), "y": (-speed, speed)
        }

    # From scratch, as for the walk, stepping while standing is penalised only after the robot
    # has learnt to lift its feet. The step counter restarts on resume; a fine-tune of a
    # policy that already skates sets XIAOXIAOBAI_FINETUNE=1 to keep the penalty and shove sooner.
    finetune = os.environ.get("XIAOXIAOBAI_FINETUNE") == "1"
    cfg.rewards["no_stepping"].weight = 0.0
    cfg.curriculum = {
        "stepping_curriculum": CurriculumTermCfg(
            func=xiaoxiaobai_mdp.step_based_staged_curriculum,
            params={
                "stages": [
                    {"name": "penalize stepping", "step": 0 if finetune else 3000 * 24,
                     "apply": lambda env: xiaoxiaobai_mdp.set_reward_weights(env, {"no_stepping": -1.0})},
                ],
            },
        ),
        "push_curriculum": CurriculumTermCfg(
            func=xiaoxiaobai_mdp.step_based_staged_curriculum,
            params={
                "stages": [
                    {"name": "light pushes", "step": (300 if finetune else 8000) * 24,
                     "apply": lambda env: set_push(env, 0.2)},
                    {"name": "full pushes", "step": (1500 if finetune else 12000) * 24,
                     "apply": lambda env: set_push(env, 0.3)},
                ],
            },
        ),
    }

    if play:
        cfg.commands["twist"].rel_standing_envs = 0.0
        cfg.curriculum = {}

    return cfg


XiaoxiaobaiSkateRlCfg = replace(
    XiaoxiaobaiVelocityRlCfg,
    wandb_project="mjlab_xiaoxiaobai_skate",
    experiment_name="mjlab_xiaoxiaobai_skate",
)
