# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

"""Xiaoxiaobai velocity environment"""

import os
import numpy as np
from copy import deepcopy
from dataclasses import dataclass, field

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from mjlab_xiaoxiaobai.robot.xiaoxiaobai_constants import XIAOXIAOBAI_ROBOT_CFG
from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.envs.mdp import dr

from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from mjlab.envs.mdp import dr
from mjlab.envs.mdp.terminations import root_height_below_minimum

from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.scene import SceneCfg
from mjlab.terrains import (
    BoxFlatTerrainCfg,
    HfRandomUniformTerrainCfg,
    TerrainEntityCfg,
    TerrainGeneratorCfg,
)
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, ObjRef, TerrainHeightSensorCfg, RingPatternCfg

from mjlab_xiaoxiaobai.tasks.mdp import (
    reward_based_staged_curriculum,
    reward_based_curriculum,
    step_based_staged_curriculum,
    set_command_velocity,
    set_stepping_parameters,
    set_push_parameters,
    set_reward_weights,
    no_stepping_penalty,
    feet_distance_penalty,
    feet_too_wide_penalty,
    foot_contact_forces_yaw,
    command_lead_excess,
    standing_crouch,
    low_vin_envs,
    battery_discharge,
    battery_discharge_step,
    arm_squeeze,
    arm_spread,
    arm_reference,
    walking_knee_floor,
    lean_shortfall,
    walking_arm_spread,
    sustained_roll,
    action_rate_l2_standing,
    push_and_mark,
    penalize_stepping_while_standing,
    stepping_curriculum,
    UniformVelocityCommandWithRotation,
    DirectionSwitchCfg,
    upright as local_upright,
    gyro_with_touchdown_spikes,
    projected_gravity_with_mount_tilt,
)

# Mostly flat floor with a few millimetres of smooth unevenness, so foot placement and
# landing timing are not tuned to a perfectly flat plane.
TERRAIN_GENERATOR_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=10.0,
    num_rows=5,
    num_cols=5,
    sub_terrains={
        "flat": BoxFlatTerrainCfg(proportion=0.4),
        "bumpy": HfRandomUniformTerrainCfg(
            proportion=0.6,
            noise_range=(0.0, 0.008),
            noise_step=0.002,
            vertical_scale=0.001,
            horizontal_scale=0.05,
            downsampled_scale=0.1,
            border_width=0.25,
        ),
    },
)

SCENE_CFG = SceneCfg(
    terrain=TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=TERRAIN_GENERATOR_CFG,
    ),
    num_envs=1,
    extent=2.0,
    entities={"robot": XIAOXIAOBAI_ROBOT_CFG},
)

VIEWER_CONFIG = ViewerConfig(
    origin_type=ViewerConfig.OriginType.ASSET_BODY,
    entity_name="robot",
    body_name="trunk",
    distance=1.15,
    elevation=-12.0,
    azimuth=90.0,
    height=720,
    width=1280,
)

SIM_CFG = SimulationCfg(
    mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
        ccd_iterations=100,
    ),
    # nconmax=256,
    # njmax=1024,
)

def make_xiaoxiaobai_velocity_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_velocity_env_cfg()

    cfg.viewer = deepcopy(VIEWER_CONFIG)
    cfg.sim = deepcopy(SIM_CFG)
    cfg.scene = deepcopy(SCENE_CFG)

    foot_site_names = ["left_foot", "right_foot"]

    #---------------------------- Sensors ---------------------------
    feet_ground_sensor_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(foot|foot_2)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    foot_height_scan_cfg = TerrainHeightSensorCfg(
        name="foot_height_scan",
        frame=tuple(ObjRef(type="site", name=s, entity="robot") for s in foot_site_names),
        pattern=RingPatternCfg.single_ring(radius=0.04, num_samples=2),
        ray_alignment="yaw",
        max_distance=1.0,
        exclude_parent_body=True,
        include_geom_groups=(0,),
        debug_vis=False,
    )

    self_collision_sensor_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    
    cfg.scene.sensors = (
        feet_ground_sensor_cfg,
        foot_height_scan_cfg,
        self_collision_sensor_cfg,
    )

    #---------------------------- Actions ---------------------------
    # Excludes the head yaw motor and the passive head_wobble_* and leg_flex_* joints.
    dofs_filter = r"^(?!head|leg_flex).*$"

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0
    cfg.actions["joint_pos"].actuator_names = (dofs_filter,)

    #---------------------------- Observations ----------------------
    del cfg.observations["actor"].terms["base_lin_vel"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    # The passive head_wobble_* and leg_flex_* joints stay out of the critic too (unmeasurable
    # on the robot, and it keeps the critic input of earlier checkpoints).
    for name in ("joint_pos", "joint_vel"):
        cfg.observations["critic"].terms[name] = deepcopy(cfg.observations["critic"].terms[name])
        cfg.observations["critic"].terms[name].params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=(r"^(?!head_wobble|leg_flex).*$",)
        )

    cfg.observations["critic"].terms["foot_contact_forces"] = ObservationTermCfg(
        func=foot_contact_forces_yaw,
        params={"sensor_name": "feet_ground_contact"},
    )

    # Biased like a real encoder with a zero error: the action term already subtracts the
    # same encoder_bias from the target, so reading and command share one calibration error.
    cfg.observations["actor"].terms["joint_pos"] = ObservationTermCfg(
        func=mdp.joint_pos_rel,
        params={"biased": True, "asset_cfg": SceneEntityCfg("robot", joint_names=(dofs_filter,))},
        noise=Unoise(n_min=-0.001, n_max=0.001),    
        delay_min_lag=0,
        delay_max_lag=1,
    )

    cfg.observations["actor"].terms["joint_vel"] = ObservationTermCfg(
        func=mdp.joint_vel_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(dofs_filter,))},
        noise=Unoise(n_min=-0.25, n_max=0.25),
        delay_min_lag=0,
        delay_max_lag=2,
    )

    # Observation delays/noises to simulate IMU sensor readings (only for actor, not critic)
    cfg.observations["actor"].terms["projected_gravity"] = deepcopy(cfg.observations["actor"].terms["projected_gravity"])
    cfg.observations["actor"].terms["projected_gravity"].noise = Unoise(n_min=-0.01, n_max=0.01)
    # Hand-assembled robots carry a few degrees of IMU mounting error that calibration misses.
    cfg.observations["actor"].terms["projected_gravity"].func = projected_gravity_with_mount_tilt
    cfg.observations["actor"].terms["projected_gravity"].params = {"max_tilt": np.deg2rad(2.0)}

    cfg.observations["actor"].terms["base_ang_vel"] = deepcopy(cfg.observations["actor"].terms["base_ang_vel"])
    # Walking on the robot puts ~0.3 rad/s of 3-8 Hz trunk vibration on the gyro that sim lacks;
    # a policy trained on +-0.03 noise used about twice the gyro gain of the upstream policy.
    cfg.observations["actor"].terms["base_ang_vel"].noise = Unoise(n_min=-0.2, n_max=0.2)
    # Amplitudes are ~1.3x the p90 touchdown peaks measured on the robot per IMU axis
    # (x = pitch, y = -yaw, z = -roll); at 1x the robot still twitched on touchdown.
    # omega spans 0-25 Hz ringing at 50 Hz.
    cfg.observations["actor"].terms["base_ang_vel"].func = gyro_with_touchdown_spikes
    cfg.observations["actor"].terms["base_ang_vel"].params = {
        "sensor_name": "robot/imu_ang_vel",
        "contact_sensor_name": "feet_ground_contact",
        "amplitude": (0.5, 0.65, 1.2),
        "decay_range": (0.3, 0.8),
        "omega_range": (0.0, np.pi),
        "probability": 0.9,
    }

    # The deploy pipeline sees the gyro 3 and the orientation 4 ticks late (sim2real check);
    # the range covers that with margin so the policy keeps phase margin on the real robot.
    cfg.observations["actor"].terms["base_ang_vel"].delay_min_lag = 0
    cfg.observations["actor"].terms["base_ang_vel"].delay_max_lag = 5
    cfg.observations["actor"].terms["base_ang_vel"].delay_update_period = 64
    cfg.observations["actor"].terms["projected_gravity"].delay_min_lag = 0
    cfg.observations["actor"].terms["projected_gravity"].delay_max_lag = 5
    cfg.observations["actor"].terms["projected_gravity"].delay_update_period = 64

    #---------------------------- Rewards ---------------------------
    # At sqrt(0.1), ignoring a 0.15 m/s command kept 80% of the reward, and under heavy
    # randomization the policy stood still until commanded 0.3 m/s. At 0.2 it keeps ~57%.
    cfg.rewards["track_linear_velocity"].params["std"] = 0.2
    cfg.rewards["track_linear_velocity"].weight = 2.0

    cfg.rewards["track_angular_velocity"].params["std"] = np.sqrt(0.5)
    cfg.rewards["track_angular_velocity"].weight = 2.0

    # Knee 0.1: a policy that held 0.15 sank into a ~50 deg crouch on the robot while standing.
    std_standing = {
        r".*head.*": 0.3,
        r".*shoulder_pitch.*": 0.1,
        r".*shoulder_roll.*": 0.1,
        r".*elbow.*": 0.1,
        r".*hip_roll.*": 0.1,
        r".*hip_pitch.*": 0.15,
        r".*hip_yaw.*": 0.1,
        r".*knee.*": 0.1,
        r".*ankle_pitch.*": 0.1,
        r".*ankle_roll.*": 0.1,
    }

    std_walking = {
        r".*head.*": 0.3,
        r".*shoulder_pitch.*": 0.4,
        r".*shoulder_roll.*": 0.2,
        r".*elbow.*": 0.2,
        r".*hip_roll.*": 0.2,
        r".*hip_pitch.*": 0.4,
        r".*hip_yaw.*": 0.2,
        r".*knee.*": 0.4,
        r".*ankle_pitch.*": 0.3,
        r".*ankle_roll.*": 0.2,
    }

    walking_threshold = 0.01

    cfg.rewards["pose"].params["asset_cfg"] = SceneEntityCfg("robot", joint_names=(r"^(?!leg_flex).*$",))
    cfg.rewards["pose"].params["std_standing"] = std_standing
    cfg.rewards["pose"].params["std_walking"] = std_walking
    cfg.rewards["pose"].params["std_running"] = std_walking
    cfg.rewards["pose"].params["walking_threshold"] = walking_threshold
    cfg.rewards["pose"].weight = 1.0

    cfg.rewards["upright"].func = local_upright
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk",)
    # Upstream's 10 deg matched its forward-leaning ready pose. This ready pose stands the
    # trunk vertical; a forward target here is only reachable by folding hips and knees.
    cfg.rewards["upright"].params["pitch"] = 0.0
    # With std sqrt(0.1) a 6 deg lean kept 90% of the reward, and a policy trained with
    # +-15 mm CoM randomization walked backward leaning 6 deg back. At 0.15 a 3 deg lean still
    # kept 89%, and with a lateral CoM error the policy walked tilted rather than shifting its
    # stance; at 0.1 it keeps ~76%.
    cfg.rewards["upright"].params["std"] = 0.1
    # Lean against the walking direction in proportion to the commanded speed, 5 deg at
    # 0.5 m/s: back when walking forward, forward when walking backward. On the robot fast
    # forward walks fell forward and backward walks fell backward.
    cfg.rewards["upright"].params["pitch_per_vx"] = -np.deg2rad(10.0)
    cfg.rewards["upright"].params["command_name"] = "twist"
    cfg.rewards["upright"].weight = 1.0
    
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk",)
    cfg.rewards["body_ang_vel"].weight = -0.05

    cfg.rewards["angular_momentum"].weight = -0.02

    for reward_name in ["foot_clearance", "foot_slip"]:
        cfg.rewards[reward_name].params["asset_cfg"].site_names = foot_site_names

    cfg.rewards["foot_clearance"].params["command_threshold"] = walking_threshold
    cfg.rewards["foot_clearance"].params["target_height"] = 0.02

    cfg.rewards["foot_swing_height"].params["command_threshold"] = walking_threshold
    cfg.rewards["foot_swing_height"].params["target_height"] = 0.02

    cfg.rewards["air_time"].params["command_threshold"] = walking_threshold
    cfg.rewards["air_time"].params["threshold_min"] = 0.125
    cfg.rewards["air_time"].params["threshold_max"] = 0.300
    cfg.rewards["air_time"].weight = 3.0

    cfg.rewards["no_stepping"] = RewardTermCfg(
        func=no_stepping_penalty,
        weight=0.0,
        params={
            "sensor_name": feet_ground_sensor_cfg.name,
            "command_name": "twist",
            "command_threshold": walking_threshold,
        },
    )

    # Upstream weight (-1e-5) is sized for ~35 kg robots. Here a 20 N touchdown costs
    # 0.1 and a 60 N stomp 0.3, comparable to one step's air_time reward.
    cfg.rewards["soft_landing"].params["command_threshold"] = walking_threshold
    cfg.rewards["soft_landing"].weight = -0.005

    cfg.rewards["foot_slip"].params["command_threshold"] = walking_threshold
    cfg.rewards["foot_slip"].weight = -1.0

    # On the robot the targets jittered twice as much as in sim (~9 vs ~5 deg/tick) and the
    # trunk shook at ~5 Hz; a smoother policy leaves more margin for unmodelled dynamics.
    cfg.rewards["action_rate_l2"].weight = -0.3

    # As upstream, joint limits are enforced only through dof_pos_limits on the achieved
    # angle; commands are not limited. With these soft motors the command leads the angle
    # by tens of degrees to produce torque, and bounding it to the working range left the
    # policy unable to support itself on one leg.

    # Walking leads stay under ~110 deg (p99 ~100); torque saturates near 160 deg. Beyond
    # that only this term sees the command (standing arms pressed on the stop drifted to
    # ~5000 deg targets).
    cfg.rewards["command_lead_excess"] = RewardTermCfg(
        func=command_lead_excess,
        weight=-0.1,
        params={"max_lead": np.deg2rad(150.0)},
    )

    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_sensor_cfg.name},
    )

    # Foot sites are 0.060 m apart with straight legs and 0.076 m in the ready pose; the
    # foot boxes still clear each other by 11 mm at 0.062 m.
    cfg.rewards["feet_distance"] = RewardTermCfg(
        func=feet_distance_penalty,
        weight=-1000.0,
        params={
            "min_dist": 0.070,
            "asset_cfg": SceneEntityCfg("robot", site_names=foot_site_names),
        },
    )

    # Without it the policy settled on a 107 mm standing stance. The robot stood at ~74 mm and
    # walked at ~78 mm under an 80 mm cap; the cap is raised ~1 cm for lateral margin.
    cfg.rewards["feet_too_wide"] = RewardTermCfg(
        func=feet_too_wide_penalty,
        weight=-50.0,
        params={
            "max_dist": 0.092,
            "asset_cfg": SceneEntityCfg("robot", site_names=foot_site_names),
        },
    )

    #---------------------------- Commands --------------------------
    command = cfg.commands["twist"]
    command.build = lambda env, _cmd=command: UniformVelocityCommandWithRotation(_cmd, env)
    command.viz.z_offset = 0.5

    command.rel_standing_envs = 0.1
    command.rel_heading_envs = 0.0
    command.rel_rotation_envs = 0.1

    command.ranges.lin_vel_x = (-0.5, 0.5)
    command.ranges.lin_vel_y = (-0.3, 0.3)
    command.ranges.ang_vel_z = (-0.75, 0.75)

    command.rotation_env_ang_vel_range = (-1.5, 1.5)
    command.rotation_min_ang_vel = 0.5

    #---------------------------- Events ----------------------------
    cfg.events["reset_base"].params["pose_range"]["z"] = (0.0, 0.01)

    cfg.events["push_robot"].params["velocity_range"] = {
        "x": (-0.6, 0.6),
        "y": (-0.6, 0.6),
    }

    cfg.events["foot_friction"].params["asset_cfg"].geom_names = (
        r".*left_foot_collision.*",
        r".*right_foot_collision.*",
    )

    # A 10 mm fore/aft CoM error toppled the +-5 mm policy while standing; walking backward the
    # robot leaned back several degrees more than sim. +-15 mm shifted the learned walking
    # posture ~5 deg backward under the looser upright std. A robot leaning right on every
    # walk, whichever leg carried which motors, matched a ~5 mm lateral CoM error in sim.
    cfg.events["base_com"].params["ranges"] = {
        0: (-0.015, 0.015),
        1: (-0.012, 0.012),
        2: (-0.005, 0.005),
    }
    cfg.events["base_com"].params["asset_cfg"].body_names = ("trunk",)

    cfg.events["dof_armature_randomization"] = EventTermCfg(
        mode="startup",
        func=dr.joint_armature,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
            "operation": "scale",
            "ranges": (0.9, 1.1),
        },
    )

    cfg.events["dof_friction_randomization"] = EventTermCfg(
        mode="startup",
        func=dr.joint_friction,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
            "operation": "scale",
            "ranges": (0.9, 1.1),
        },
    )

    # Measured zero errors on the robot reached 2-4.5 deg; +-2.6 deg per joint keeps the
    # policy tolerant of what remains after calibration.
    # Larger errors (one leg's roll chain read ~4.5 deg off on the robot) are calibrated out
    # with neutral offsets: a policy trained on +-4.6 deg roll errors stopped trusting its
    # joint readings and let the legs sag into a crouch.
    cfg.events["encoder_bias"].params["bias_range"] = (-0.045, 0.045)

    # Mass scale e^(2*alpha): legs 0.95-1.25 covers the 26 g shells (+12% per leg).
    cfg.events["leg_mass_randomization"] = EventTermCfg(
        mode="startup",
        func=dr.pseudo_inertia,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=(r"hip.*", r"femur.*", r"tibia.*", r"ankle_block.*", r"foot.*")
            ),
            "alpha_range": (-0.025, 0.11),
        },
    )
    # Head rocking on the yaw motor horn: ~3-15 Hz with the head's ~3.6e-4 kg m^2 about the
    # horn (gravity removes ~0.06 N m/rad), damping ratio ~0.02-0.5. Footfalls excite it and
    # it shakes the trunk and IMU.
    cfg.events["head_wobble_stiffness"] = EventTermCfg(
        mode="startup",
        func=dr.joint_stiffness,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(r"head_wobble_.*",)),
            "operation": "abs",
            "ranges": (0.2, 3.2),
        },
    )
    cfg.events["head_wobble_damping"] = EventTermCfg(
        mode="startup",
        func=dr.joint_damping,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(r"head_wobble_.*",)),
            "operation": "abs",
            "ranges": (0.001, 0.008),
        },
    )

    # Series roll compliance at each hip (plastic frame, horn and gear play). In single stance
    # the stance hip carries ~0.25 N m (p50), and on the robot the soles tilted ~4.5 deg more
    # against the IMU than in sim on either leg; 2.5-10 N m/rad gives ~5.7-1.4 deg of extra
    # flex, damping ratio ~0.06-0.3.
    cfg.events["leg_flex_stiffness"] = EventTermCfg(
        mode="startup",
        func=dr.joint_stiffness,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(r"leg_flex_.*",)),
            "operation": "abs",
            "distribution": "log_uniform",
            "ranges": (2.5, 10.0),
        },
    )
    cfg.events["leg_flex_damping"] = EventTermCfg(
        mode="startup",
        func=dr.joint_damping,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(r"leg_flex_.*",)),
            "operation": "abs",
            "ranges": (0.01, 0.05),
        },
    )

    cfg.events["body_mass_randomization"] = EventTermCfg(
        mode="startup",
        func=dr.pseudo_inertia,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk", "head")),
            "alpha_range": (-0.05, 0.05),
        },
    )

    #---------------------------- Curriculum ------------------------
    cfg.curriculum = {}

    cfg.curriculum["staged_curriculum"] = CurriculumTermCfg(
        func=step_based_staged_curriculum,
        params={
            "stages": [
                {
                    # Upstream also widened the command ranges here, but its helpers edited a
                    # cfg copy the live command term never read, so its policy was effectively
                    # trained on the initial ranges. Kept that way; set_command_velocity now works.
                    "name": "penalize stepping",
                    # The step counter restarts on resume; a fine-tune of a policy that
                    # already passed this stage sets XIAOXIAOBAI_FINETUNE=1 to keep it.
                    "step": 0 if os.environ.get("XIAOXIAOBAI_FINETUNE") == "1" else 3000 * 24,
                    "apply": lambda env: set_reward_weights(
                        env,
                        {
                            "air_time": 3.0,
                            "no_stepping": -1.0,
                        },
                    ),
                },
            ],
        },
    )

    #---------------------------- Terminations ----------------------
    cfg.terminations["fell_over"] = TerminationTermCfg(
        func=root_height_below_minimum,
        params={"minimum_height": 0.10},
    )

    #---------------------------- Deployed-policy base ----------------
    # XIAOXIAOBAI_BASE=24996 trains under the randomization of the deployed model_24996 (run
    # 2026-11-04_10-50-58); fine-tunes under the heavier settings above came out soft and slow
    # on the robot. robot/xiaoxiaobai_constants.py drops the leg_flex joints and the low vin too.
    if os.environ.get("XIAOXIAOBAI_BASE") == "24996":
        cfg.events["base_com"].params["ranges"] = {0: (-0.01, 0.01), 1: (-0.005, 0.005), 2: (-0.005, 0.005)}
        cfg.events["encoder_bias"].params["bias_range"] = (-0.035, 0.035)
        cfg.events["push_robot"].params["velocity_range"] = {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}
        del cfg.events["leg_flex_stiffness"], cfg.events["leg_flex_damping"]
        cfg.observations["actor"].terms["base_ang_vel"].params.update(
            amplitude=(0.4, 0.5, 0.9), decay_range=(0.3, 0.75), probability=0.8
        )
        cfg.observations["actor"].terms["projected_gravity"].func = mdp.projected_gravity
        cfg.observations["actor"].terms["projected_gravity"].params = {}
        cfg.rewards["track_linear_velocity"].params["std"] = np.sqrt(0.1)
        cfg.rewards["pose"].params["std_standing"][r".*knee.*"] = 0.15
        cfg.rewards["upright"].params["std"] = 0.15
    # Fine-tuned down to XIAOXIAOBAI_VIN_LOW the walk learned to crouch when stopped; lean 27000
    # stands within ~0.1 rad of the default on these joints. Skate and dance share this base.
    if os.environ.get("XIAOXIAOBAI_VIN_LOW"):
        cfg.rewards["standing_crouch"] = RewardTermCfg(
            func=standing_crouch,
            weight=-1.0,
            params={
                "deadband": 0.1,
                "command_threshold": walking_threshold,
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*hip_pitch.*", r".*knee.*", r".*ankle_pitch.*")),
            },
        )
        # The default pose holds shoulder roll 0.087 rad off the trunk stop. Under training
        # randomization lean 27000 averaged 0.21 rad under it, the crouch fine-tune 0.34.
        cfg.rewards["arm_squeeze"] = RewardTermCfg(
            func=arm_squeeze,
            weight=-1.0,
            params={
                "margin": 0.087,
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*shoulder_roll.*",)),
            },
        )
        # XIAOXIAOBAI_VIN_LOW_FRAC=<f>: a further fraction f of the envs within 1 V of the low end.
        if os.environ.get("XIAOXIAOBAI_VIN_LOW_FRAC"):
            vin_low = float(os.environ["XIAOXIAOBAI_VIN_LOW"])
            cfg.events["low_vin_envs"] = EventTermCfg(
                mode="startup",
                func=low_vin_envs,
                params={"fraction": float(os.environ["XIAOXIAOBAI_VIN_LOW_FRAC"]), "vin_range": (vin_low, vin_low + 1.0)},
            )
        # XIAOXIAOBAI_DISCHARGE=1: half the episodes run the battery down mid-walk, as on the robot.
        # 自然 36995 (XIAOXIAOBAI_STYLE=3), trained on fixed voltages, already fell no more often over
        # a 7.6 -> 5.0 V run (scripts/walk_discharge_eval.py) than at the voltages it passed.
        if os.environ.get("XIAOXIAOBAI_DISCHARGE") == "1":
            vin_low = float(os.environ["XIAOXIAOBAI_VIN_LOW"])
            cfg.events["battery_discharge"] = EventTermCfg(
                mode="reset",
                func=battery_discharge,
                params={
                    "fraction": 0.5,
                    "vin_start_range": (vin_low + 1.0, 8.0),
                    "vin_end_min": vin_low,
                    "min_drop": 0.5,
                    "duration_range_s": (5.0, 20.0),
                },
            )
            cfg.events["battery_discharge_step"] = EventTermCfg(mode="step", func=battery_discharge_step)

    # XIAOXIAOBAI_POSTURE=1: lean 27000's posture fine-tune (scripts/walk_posture_eval.py, 7.2 V,
    # vx +-0.4). Walking with the trunk CoM 8 mm to one side it leaned 3-5 deg (standing 1 deg),
    # on the robot it needed the IMU roll correction to walk upright; it leaned back 3.9 deg
    # walking forward but only 1.4 deg forward walking backward; shoulder roll sat 11-12 deg out.
    # A first try (roll -8, spread capped at 8 deg, back lean 22 deg per m/s, no swing term)
    # pulled shoulder roll to 5 deg and walking backward swung the arms 3.4 deg instead of 8.9.
    # XIAOXIAOBAI_POSTURE=2 keeps 27000's walking IMU response (the roll term and the tighter
    # upright of =1 jittered on the robot) and makes only standing more responsive.
    posture = os.environ.get("XIAOXIAOBAI_POSTURE")
    if posture == "1":
        cfg.events["base_com"].params["ranges"][1] = (-0.012, 0.012)
        cfg.rewards["sustained_roll"] = RewardTermCfg(
            func=sustained_roll,
            weight=-15.0,
            params={"tau": 1.0, "asset_cfg": SceneEntityCfg("robot", body_names=("trunk",))},
        )
        cfg.rewards["upright"].params["std"] = 0.1
        cfg.rewards["upright"].params["pitch_per_vx"] = -np.deg2rad(16.0)
        cfg.rewards["upright"].params["pitch_per_vx_back"] = -np.deg2rad(18.0)
    if posture == "2":
        # Under upright std 0.15 lean 27000 reached ~1/3 of the backward target (4 deg asked,
        # 1.4 deg at -0.4 m/s), so backward asks for more.
        cfg.rewards["upright"].params["pitch_per_vx"] = -np.deg2rad(16.0)
        cfg.rewards["upright"].params["pitch_per_vx_back"] = -np.deg2rad(26.0)
        # Standing, as the original: gyro noise +-0.03 (the +-0.2 is walking vibration, and it
        # buried the start of a fall), action rate -0.1, upright std sqrt(0.1) -> 0.1 here.
        gyro = cfg.observations["actor"].terms["base_ang_vel"]
        gyro.noise = None
        gyro.params.update(noise=0.2, noise_standing=0.03, command_name="twist", command_threshold=walking_threshold)
        cfg.rewards["action_rate_l2"].func = action_rate_l2_standing
        cfg.rewards["action_rate_l2"].params = {"standing_scale": 1.0 / 3.0, "command_threshold": walking_threshold}
        cfg.rewards["upright"].params.update(std_standing=0.1, command_threshold=walking_threshold)
        # 27000 met a push mostly while walking (10% standing commands, one push per 1-3 s).
        # More standing time, extra pushes there, and a free recovery step for 1 s after a push
        # (no_stepping otherwise charges for it).
        cfg.commands["twist"].rel_standing_envs = 0.2
        cfg.events["push_robot"].func = push_and_mark
        cfg.events["push_standing"] = EventTermCfg(
            func=push_and_mark,
            mode="interval",
            interval_range_s=(2.0, 4.0),
            params={
                "velocity_range": {"x": (-0.6, 0.6), "y": (-0.6, 0.6)},
                "standing_only": True,
                "command_threshold": walking_threshold,
            },
        )
        cfg.rewards["no_stepping"].params["push_grace_s"] = 1.0
    if posture in ("1", "2"):
        cfg.rewards["arm_spread"] = RewardTermCfg(
            func=arm_spread,
            weight=-2.0,
            params={
                "max_out": np.deg2rad(10.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*shoulder_roll.*",)),
            },
        )
        # lean 27000 swung shoulder pitch with a std of 4.3 / 6.2 deg at vx +0.2 / -0.2 and
        # 7.4 / 8.9 deg at +0.4 / -0.4. A penalty on the swing (std over 1 s) falling short of
        # 15 deg per m/s did not hold it: pushes and command changes count as swing, and the
        # fine-tunes swung 3-5 deg walking straight (vy = wz = 0, rare in training), falling there
        # in 14% of 15 s backward walks at 7.2 V instead of 2.9%. The arms clone lean 27000's
        # shoulder pitch and elbow actions instead (XIAOXIAOBAI_TEACHER, teacher_ppo.py); shoulder
        # roll stays free for arm_spread.
        cfg.commands["twist"].rel_straight_envs = 0.2
    # XIAOXIAOBAI_POSTURE=3: lean 27000 with only =2's lean targets (stand_v1 at 7.2 V, vx +-0.4:
    # 4.6 deg back walking forward, 4.9 deg forward walking backward) and direction changes while
    # walking. 27000 changed its command every 3-8 s, so a reversal or a sudden turn mid-walk
    # (the robot ramps the command through it in ~1.5 s) was rare; on the robot it fell there.
    if posture == "3":
        cfg.rewards["upright"].params["pitch_per_vx"] = -np.deg2rad(16.0)
        cfg.rewards["upright"].params["pitch_per_vx_back"] = -np.deg2rad(26.0)
        cfg.commands["twist"].direction_switch = DirectionSwitchCfg()
    # XIAOXIAOBAI_STYLE=1: keep lean 27000's walk. Fine-tuned under =3 (变向 29999) it straightened
    # its knees walking (9-13 -> 1-6 deg at 7.2 V), stepped faster and lower, and with the arm
    # network cloned to within ~1 deg it still held shoulder roll 3-5 deg further out and swung
    # shoulder pitch 2 deg less forward / 1.5 deg more backward (scripts/walk_style_compare.py).
    # The arms are scored on their achieved angles against 27000's (arm_reference); pair it with
    # XIAOXIAOBAI_TEACHER_LEG_LR, which keeps the legs' actions near 27000's.
    style = os.environ.get("XIAOXIAOBAI_STYLE")
    if style in ("1", "2", "3"):
        cfg.rewards["arm_reference"] = RewardTermCfg(
            func=arm_reference,
            weight=2.0,
            params={"path": os.path.join(os.path.dirname(__file__), "arm_ref_lean27000.npz"), "std": 0.1},
        )
        cfg.rewards["walking_knee_floor"] = RewardTermCfg(
            func=walking_knee_floor,
            weight=-5.0,
            params={
                "tau": 1.0,
                "min_flex": np.deg2rad(7.0),
                "command_threshold": walking_threshold,
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*knee.*",)),
            },
        )
    # XIAOXIAOBAI_STYLE=2: =1 settled at 27000's lean (7.2 V, vx +-0.4: 4.0 deg back walking
    # forward, 3.2 forward walking backward; 变向 29999 5.5 and 4.3) and walking backward held
    # shoulder roll 16-17 deg out (27000 13). On the robot walks fell toward the walking direction.
    # XIAOXIAOBAI_STYLE=3: =2 stood with shoulder roll 14.6 deg out and the trunk 1.5 deg back at
    # 7.2 V (27000 10.8 deg out, 2.3 deg forward).
    if style in ("2", "3"):
        cfg.rewards["lean_shortfall"] = RewardTermCfg(
            func=lean_shortfall,
            weight=-10.0,
            params={
                "tau": 0.5,
                "per_vx": np.deg2rad(16.0),
                "per_vx_back": np.deg2rad(14.0),
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk",)),
            },
        )
        cfg.rewards["walking_arm_spread"] = RewardTermCfg(
            func=walking_arm_spread,
            weight=-10.0,
            params={
                "tau": 1.0,
                "max_out": np.deg2rad(14.0),
                "command_threshold": walking_threshold,
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*shoulder_roll.*",)),
            },
        )
    if style == "3":
        cfg.rewards["lean_shortfall"].params.update(stand_pitch=np.deg2rad(1.5), command_threshold=walking_threshold)
        cfg.rewards["walking_arm_spread"].params["standing_max_out"] = np.deg2rad(11.5)

    #---------------------------- Play mode -------------------------
    if play:
        cfg.curriculum = {}
        
        cfg.commands["twist"].rel_standing_envs = 0.0
        cfg.commands["twist"].rel_rotation_envs = 0.0
        cfg.commands["twist"].direction_switch = None

        cfg.events["push_robot"].params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
        }
        cfg.events.pop("push_standing", None)
        cfg.observations["actor"].terms["base_ang_vel"].params.pop("noise", None)
        cfg.observations["actor"].terms["base_ang_vel"].params.pop("noise_standing", None)

        # cfg.commands["twist"].ranges.lin_vel_x = (0.5, 0.5)
        # cfg.commands["twist"].ranges.lin_vel_y = (0.0, 0.0)
        # cfg.commands["twist"].ranges.ang_vel_z = (0.0, 0.0)
        # cfg.commands["twist"].rotation_env_ang_vel_range = (1.0, 1.0)

        # cfg.commands["twist"].ranges.lin_vel_x = (-0.7, 0.7)
        # cfg.commands["twist"].ranges.lin_vel_y = (-0.3, 0.3)
        # cfg.commands["twist"].ranges.ang_vel_z = (-1.5, 1.5)
        # cfg.commands["twist"].rotation_env_ang_vel_range = (-3.0, 3.0)

        # Can be used to edit neutral pose with a zero agent
        # cfg.events["reset_base"].params["pose_range"]["x"] = (0.0, 0.0)
        # cfg.events["reset_base"].params["pose_range"]["y"] = (0.0, 0.0)
        # cfg.events["reset_base"].params["pose_range"]["z"] = (0.3, 0.3)
        # cfg.events["reset_base"].params["pose_range"]["yaw"] = (0.0, 0.0)
        # cfg.events["reset_base"].interval_range_s = (0.0, 0.0)
        # cfg.events["reset_base"].mode = "interval"

        # del cfg.rewards["track_linear_velocity"]
        # del cfg.rewards["track_angular_velocity"]
        # del cfg.rewards["pose"]
        # del cfg.rewards["upright"]
        # cfg.terminations = {}

        cfg.observations["actor"].enable_corruption = False

        # Can be used to print something every step
        def debug(env: ManagerBasedRlEnv, _):
            env.observation_manager.compute_group("actor", update_history=True)

            terms = dict(env.observation_manager.get_active_iterable_terms(env_idx=0))

            base_ang_vel = terms["actor-base_ang_vel"]
            projected_gravity = terms["actor-projected_gravity"]

            print("\n")
            print(f"base_ang_vel: ")
            print(f"x: {base_ang_vel[0]:.3f}")
            print(f"y: {base_ang_vel[1]:.3f}")
            print(f"z: {base_ang_vel[2]:.3f}")
            print("\n")
            print(f"projected_gravity: ")
            print(f"x: {projected_gravity[0]:.3f}")
            print(f"y: {projected_gravity[1]:.3f}")
            print(f"z: {projected_gravity[2]:.3f}")

        # cfg.events["debug"] = EventTermCfg(
        #     func=debug, mode="interval", interval_range_s=(0.0, 0.0)
        # )

    return cfg


@dataclass
class MirrorSymmetryCfg:
    """rsl_rl ``symmetry_cfg``: train on every sample and its left/right mirror."""

    use_data_augmentation: bool = True
    use_mirror_loss: bool = False
    mirror_loss_coeff: float = 0.0
    data_augmentation_func: str = "mjlab_xiaoxiaobai.tasks.symmetry:mirror_augment"


@dataclass
class XiaoxiaobaiPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    symmetry_cfg: MirrorSymmetryCfg | None = field(default_factory=MirrorSymmetryCfg)


XiaoxiaobaiVelocityRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=XiaoxiaobaiPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    ),
    wandb_project="mjlab_xiaoxiaobai_velocity",
    experiment_name="mjlab_xiaoxiaobai_velocity",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=15_000,
)
if os.environ.get("XIAOXIAOBAI_TEACHER"):
    XiaoxiaobaiVelocityRlCfg.algorithm.class_name = "mjlab_xiaoxiaobai.tasks.teacher_ppo:TeacherArmsPPO"
