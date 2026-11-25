"""Dance: track a retargeted human dance (BeyondMimic-style motion tracking, mjlab's tracking task).

The robot, motors, sensing noise/delays and randomisation are the walk's (make_xiaoxiaobai_velocity_env_cfg,
train with the same XIAOXIAOBAI_BASE); only the arm joints get their working ranges (get_dance_spec),
the twist command becomes the motion, and the walk rewards become tracking rewards. Lengths and
linear speeds of mjlab's G1 tracking terms are scaled to Xiaoxiaobai (leg 0.157 m vs 0.689 m).

Motion: scripts/retarget_g1_to_xiaoxiaobai.py then scripts/dance_motion_npz.py; XIAOXIAOBAI_DANCE_MOTION
overrides the default file.
"""
import os
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.tracking import mdp as tracking
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_xiaoxiaobai.robot.xiaoxiaobai_constants import get_dance_spec
from mjlab_xiaoxiaobai.tasks import mdp
from mjlab_xiaoxiaobai.tasks.xiaoxiaobai_velocity_env_cfg import XiaoxiaobaiVelocityRlCfg, make_xiaoxiaobai_velocity_env_cfg

MOTION_FILE = os.environ.get(
    "XIAOXIAOBAI_DANCE_MOTION",
    str(Path(__file__).resolve().parents[3] / "data/motions/xiaoxiaobai/dance2_s3_130_200_50hz.npz"),
)
# Xiaoxiaobai / G1 length scale for the tracking stds and thresholds.
SCALE = 0.227
ANCHOR = "trunk"
# Anchor first: the command resets the root from the first body.
BODIES = (
    "trunk",
    "femur__configuration_left", "tibia__configuration_left", "ankle_block__configuration_left", "foot_2",
    "femur__configuration_right", "tibia__configuration_right", "ankle_block__configuration_right", "foot",
    "humerus__configuration_left", "radius_2",
    "humerus__configuration_right", "radius",
)
FEET = ("foot", "foot_2")
HANDS = ("radius", "radius_2")
ACTUATED = SceneEntityCfg("robot", joint_names=(r"^(?!head|leg_flex).*$",))


class DanceMotionCommand(tracking.MotionCommand):
    """Adaptive sampling, plus a share of the episodes from the first frame: the robot always
    starts the dance from the ready pose, a moment adaptive sampling seldom picks."""

    def _adaptive_sampling(self, env_ids: torch.Tensor):
        super()._adaptive_sampling(env_ids)
        from_start = torch.rand(len(env_ids), device=self.device) < self.cfg.start_ratio
        self.time_steps[env_ids[from_start]] = 0


@dataclass(kw_only=True)
class DanceMotionCommandCfg(tracking.MotionCommandCfg):
    start_ratio: float = 0.15

    def build(self, env) -> DanceMotionCommand:
        return DanceMotionCommand(self, env)


def make_xiaoxiaobai_dance_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_xiaoxiaobai_velocity_env_cfg()
    cfg.scene.entities["robot"] = replace(cfg.scene.entities["robot"], spec_fn=get_dance_spec)
    cfg.episode_length_s = 10.0
    cfg.curriculum = {}

    cfg.commands = {
        "motion": DanceMotionCommandCfg(
            entity_name="robot",
            motion_file=MOTION_FILE,
            anchor_body_name=ANCHOR,
            body_names=BODIES,
            resampling_time_range=(1.0e9, 1.0e9),
            debug_vis=play,
            pose_range={"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.003, 0.003),
                        "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-0.2, 0.2)},
            velocity_range={"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.05, 0.05),
                            "roll": (-0.52, 0.52), "pitch": (-0.52, 0.52), "yaw": (-0.78, 0.78)},
            joint_position_range=(-0.1, 0.1),
            sampling_mode="start" if play else "adaptive",
        )
    }

    # The motion command places the robot on its reference at every reset.
    for name in ("reset_base", "reset_robot_joints"):
        del cfg.events[name]

    # -- Observations: the walk's proprioception, the reference instead of the twist --
    actor = cfg.observations["actor"].terms
    del actor["command"]
    actor["motion_command"] = ObservationTermCfg(
        func=mdp.motion_joint_command, params={"command_name": "motion", "asset_cfg": ACTUATED}
    )
    # Torso orientation relative to the reference's: the IMU quaternion, yaw aligned to the
    # reference when the dance starts. Sensed like projected gravity.
    actor["motion_anchor_ori_b"] = replace(
        actor["projected_gravity"], func=tracking.motion_anchor_ori_b, params={"command_name": "motion"},
        noise=Unoise(n_min=-0.05, n_max=0.05),
    )
    critic = cfg.observations["critic"].terms
    for name in ("command", "foot_height"):
        del critic[name]
    critic["motion_command"] = ObservationTermCfg(func=tracking.generated_commands, params={"command_name": "motion"})
    for name in ("motion_anchor_pos_b", "motion_anchor_ori_b", "robot_body_pos_b", "robot_body_ori_b"):
        critic[name] = ObservationTermCfg(func=getattr(tracking, name), params={"command_name": "motion"})

    # -- Rewards: mjlab's tracking terms, lengths scaled; the walk's motor regularisers --
    def track(func, weight, std):
        return RewardTermCfg(func=func, weight=weight, params={"command_name": "motion", "std": std})

    walk = cfg.rewards
    cfg.rewards = {
        "motion_global_root_pos": track(tracking.motion_global_anchor_position_error_exp, 0.5, 0.3 * SCALE),
        "motion_global_root_ori": track(tracking.motion_global_anchor_orientation_error_exp, 0.5, 0.4),
        "motion_body_pos": track(tracking.motion_relative_body_position_error_exp, 1.0, 0.3 * SCALE),
        "motion_body_ori": track(tracking.motion_relative_body_orientation_error_exp, 1.0, 0.4),
        "motion_body_lin_vel": track(tracking.motion_global_body_linear_velocity_error_exp, 1.0, 1.0 * SCALE),
        "motion_body_ang_vel": track(tracking.motion_global_body_angular_velocity_error_exp, 1.0, 3.14),
        # The steps are 1.5-4 cm high on Xiaoxiaobai, too little of the body error to be worth the
        # single support: penalise the feet off the reference's lift and contacts directly.
        "motion_feet_height": RewardTermCfg(
            func=mdp.motion_feet_height_error, weight=-0.6,
            params={"command_name": "motion", "height_sensor_name": "foot_height_scan", "scale": 0.01},
        ),
        "motion_feet_contact": RewardTermCfg(
            func=mdp.motion_feet_contact_mismatch, weight=-0.75,
            params={"command_name": "motion", "sensor_name": "feet_ground_contact",
                    "height_sensor_name": "foot_height_scan", "air_height": 0.008},
        ),
        # With the feet penalties, quitting early must never pay: at -1/-1 the policy learned to
        # trip the end-effector termination.
        "termination": RewardTermCfg(func=tracking.is_terminated, weight=-100.0),
        "action_rate_l2": replace(walk["action_rate_l2"], weight=-0.1),
        "dof_pos_limits": replace(walk["dof_pos_limits"], weight=-10.0),
        "command_lead_excess": walk["command_lead_excess"],
        "self_collisions": replace(walk["self_collisions"], weight=-10.0),
    }

    # -- Terminations: off the reference (heights scaled), or down --
    cfg.terminations = {
        "time_out": cfg.terminations["time_out"],
        "fell_over": cfg.terminations["fell_over"],
        "anchor_pos": TerminationTermCfg(
            func=tracking.bad_anchor_pos_z_only, params={"command_name": "motion", "threshold": 0.25 * SCALE}
        ),
        "anchor_ori": TerminationTermCfg(
            func=tracking.bad_anchor_ori,
            params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 0.8},
        ),
        "ee_body_pos": TerminationTermCfg(
            func=tracking.bad_motion_body_pos_z_only,
            params={"command_name": "motion", "threshold": 0.25 * SCALE, "body_names": FEET},
        ),
        # The arms help balance on one foot; the hands' tracking reward keeps them close enough.
        "ee_hands_pos": TerminationTermCfg(
            func=tracking.bad_motion_body_pos_z_only,
            params={"command_name": "motion", "threshold": 0.08, "body_names": HANDS},
        ),
    }

    if play:
        cfg.episode_length_s = 1e9
        cfg.events["push_robot"].params["velocity_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0)}
        cfg.observations["actor"].enable_corruption = False
    return cfg


XiaoxiaobaiDanceRlCfg = replace(
    XiaoxiaobaiVelocityRlCfg,
    algorithm=replace(XiaoxiaobaiVelocityRlCfg.algorithm, symmetry_cfg=None),
    wandb_project="mjlab_xiaoxiaobai_dance",
    experiment_name="mjlab_xiaoxiaobai_dance",
    max_iterations=30_000,
)
