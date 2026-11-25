# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

"""Skating in strokes: one skate pushes off beside the gliding one, lifts and hangs while the
other glides alone, then sets down to glide while the other pushes. Same floors, skates and
randomisation as the skate task; a stroke clock in the observation sets the rhythm and a
reference leg pose per clock phase sets the stroke's shape.

Without the clock the skate task, trained from scratch, swizzled with both skates on the
floor (out-toe, in-toe); started from the walk, it stepped on its wheels instead of gliding."""

import math
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_xiaoxiaobai.tasks import mdp as xiaoxiaobai_mdp
from mjlab_xiaoxiaobai.tasks.xiaoxiaobai_skate_env_cfg import FEET, XiaoxiaobaiSkateRlCfg, make_xiaoxiaobai_skate_env_cfg

# Policy ticks per stroke cycle (both skates push once) at 50 Hz: 1.2 s. Exported to the ONNX
# metadata as phase_period_steps for WalkMove's reference phase. At 0.8 s each skate glided
# alone for 0.24 s and the cadence read as brisk walking.
STROKE_PERIOD_STEPS = 60
# Each skate pushes beside the gliding one for 0.24 s, then hangs 0.36 s while the other
# glides alone.
STROKE_PUSH_FRACTION = 0.2
# Lift-off and touchdown may land within ~2.4 ticks of the clock.
STROKE_MARGIN = 0.04

STROKE_LEG_JOINTS = ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch", "ankle_roll")
# Right leg [deg off the default pose] through its own stroke phase. Push end solved by
# inverse kinematics for the skate's site (heading frame); the crouched poses keep the sole
# level (hip pitch + knee + ankle pitch = 0) with the foot under the hip:
#   glide: skate straight under the hip, crouched on a 32 deg knee. On 40 the motors of
#          79-87% of the robots hit their current limit on every floor.
#   push: from beside the gliding skate, turned out 15 deg, extending to 30 mm back, 35 mm
#         out, turned out 30 deg on a ~12 deg inner edge, still on the floor.
#   swing: folded over the floor and back under the trunk, turned out 10 deg; the ankle
#          stops at its -28.6 deg limit.
STROKE_REFERENCE = (
    (0.00, (15.0, 3.0, -17.0, 30.0, -13.0, 2.0)),
    (0.20, (30.0, 13.7, -3.4, 11.1, -7.7, -1.7)),
    (0.35, (10.0, 7.9, -38.0, 70.0, -28.6, 0.0)),
    (0.50, (0.0, 0.3, -18.2, 32.0, -13.8, 0.0)),
    (0.90, (0.0, 0.3, -18.2, 32.0, -13.8, 0.0)),
)

STROKE_ARM_JOINTS = ("shoulder_pitch", "shoulder_roll", "elbow")
# Right arm [deg] through the right leg's phase: it swings forward (shoulder pitch positive)
# as the right skate pushes back, back as the left one does, elbow bent 45-70 deg, the left
# arm in mirror half a cycle later. The arms' yaw momentum then opposes the push's.
STROKE_ARM_REFERENCE = (
    (0.00, (11.0, 6.4, -60.0)),
    (0.20, (35.0, 8.0, -70.0)),
    (0.45, (5.0, 6.0, -57.5)),
    (0.70, (-25.0, 8.0, -45.0)),
    (0.95, (5.0, 6.0, -57.5)),
)
# Standing [deg, right leg then right arm, in STROKE_LEG_JOINTS + STROKE_ARM_JOINTS order]:
# skates turned out 10 deg, the glide crouch's sole-level 26 deg knee, arms hanging.
STROKE_RATE_RANGE = (0.6, 1.2)
STROKE_STAND_POSE = (10.0, 3.0, -14.8, 25.9, -11.1, 0.0, 5.0, 6.0, -30.0)


def make_xiaoxiaobai_skate_stroke_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_xiaoxiaobai_skate_env_cfg(play, skate_base=False)
    clock = {"period_steps": STROKE_PERIOD_STEPS}
    schedule = {**clock, "push_fraction": STROKE_PUSH_FRACTION, "margin": STROKE_MARGIN}

    for group in ("actor", "critic"):
        cfg.observations[group].terms["gait_phase"] = ObservationTermCfg(
            func=xiaoxiaobai_mdp.gait_phase, params=dict(clock)
        )
        # The robot feeds back the joint targets only; the clock's phase carries the rate.
        cfg.observations[group].terms["actions"].params["action_name"] = "joint_pos"

    # At one cadence the push that holds 0.3 m/s on carpet overshoots on tile with free
    # bearings. The policy sets the clock's rate, 0.6-1.2 of the base (strokes of 2.0-1.0 s).
    cfg.actions["stroke_rate"] = xiaoxiaobai_mdp.StrokeRateActionCfg(
        entity_name="robot", scale=0.5, rate_range=STROKE_RATE_RANGE,
    )
    # At -1 the base rate costs 0.4 a second over the slowest; the slower stroke glides on.
    # At -0.5 the rate stayed at 0.97-1.0 for 800 iterations, 0.5 m/s on tile for 0.3.
    cfg.rewards["stroke_rate"] = RewardTermCfg(func=xiaoxiaobai_mdp.stroke_rate_cost, weight=-1.0)

    # The clock decides when a skate is up; these paid any single stance or air time.
    del cfg.rewards["air_time"]
    del cfg.rewards["single_skate_glide"]

    # A skate is due up 44% of the time (either side, outside the switch margins). Both
    # skates parked on the floor, standing or swizzling, pay -0.9 a second on the mismatch and
    # forgo 1.3 from the lift and 1.3 from the clearance.
    cfg.rewards["stroke_contact_mismatch"] = RewardTermCfg(
        func=xiaoxiaobai_mdp.stroke_contact_mismatch,
        weight=-2.0,
        params={"sensor_name": "feet_ground_contact", **schedule},
    )
    cfg.rewards["stroke_lift"] = RewardTermCfg(
        func=xiaoxiaobai_mdp.stroke_lift,
        weight=3.0,
        params={"sensor_name": "feet_ground_contact", **schedule},
    )
    cfg.rewards["stroke_swing_clearance"] = RewardTermCfg(
        func=xiaoxiaobai_mdp.stroke_swing_clearance,
        weight=3.0,
        params={"height_sensor_name": "foot_height_scan", "target_height": 0.02, **schedule},
    )
    # The walk's clearance cost taxed the gliding skate for moving low.
    cfg.rewards["foot_clearance"] = RewardTermCfg(
        func=xiaoxiaobai_mdp.stroke_ground_clearance_cost,
        weight=-2.0,
        params={"height_sensor_name": "foot_height_scan", "target_height": 0.02, **schedule,
                "asset_cfg": SceneEntityCfg("robot", site_names=FEET)},
    )
    # The gliding skate rolls along with the trunk. Paid instead as the lone skate's speed
    # along its nose (weight 3), it earned 0.05 of ~140 an episode and the stance skate stood
    # at 2% of the trunk's speed. At std 0.2 a planted skate under a trunk at 0.3 m/s keeps
    # 11% of the weight; one skate is gliding ~84% of the time.
    cfg.rewards["stroke_glide_follow"] = RewardTermCfg(
        func=xiaoxiaobai_mdp.stroke_glide_follow,
        weight=8.0,
        params={"sensor_name": "feet_ground_contact", **clock, "margin": STROKE_MARGIN, "std": 0.2,
                "asset_cfg": SceneEntityCfg("robot", site_names=FEET)},
    )

    # With contact timing alone, on wheels braked like shoes or from the walk, the policy
    # stepped: skates planted, set down ahead, the trunk vaulting over them. At std 0.25 rad
    # a leg tracking the stroke 14 deg off on every joint keeps 37% of the weight.
    legs = tuple(f"{side}_{j}" for side in ("right", "left") for j in STROKE_LEG_JOINTS)
    cfg.rewards["stroke_reference_pose"] = RewardTermCfg(
        func=xiaoxiaobai_mdp.stroke_reference_pose,
        weight=4.0,
        # The full push accelerates on tile with free bearings (0.41 m/s for 0.3 despite the
        # overspeed cost); past 0.1 m/s over the command the leg may push softly or not at all.
        params={"keyframes": STROKE_REFERENCE, **clock, "std": 0.25, "overspeed_band": 0.1,
                "asset_cfg": SceneEntityCfg("robot", joint_names=legs, preserve_order=True)},
    )
    # Held in front of the chest, the arms looked stiff, and the trunk twisted +-2.5 rad/s
    # at every push. They swing with the stroke instead, in mirror, not in step.
    arms = tuple(f"{side}_{j}" for side in ("right", "left") for j in STROKE_ARM_JOINTS)
    del cfg.rewards["arm_hold_pose"]
    del cfg.rewards["arm_symmetry"]
    cfg.rewards["stroke_arm_swing"] = RewardTermCfg(
        func=xiaoxiaobai_mdp.stroke_reference_pose,
        weight=3.0,
        params={"keyframes": STROKE_ARM_REFERENCE, **clock, "std": 0.3, "metric": "stroke_arm_rms_deg",
                "asset_cfg": SceneEntityCfg("robot", joint_names=arms, preserve_order=True)},
    )
    # At -0.05 the swing (30 deg each way every 1.2 s) cost ~0.4 a step.
    cfg.rewards["arm_joint_vel"].weight = -0.01

    # Stopped, with the stroke references off, the policy locked both skates crosswise at
    # the 65 deg hip yaw limit, 146 mm apart and 118 mm staggered, until it moved again. It
    # stands instead with the skates turned out 10 deg in a slight crouch, arms hanging.
    # At std 0.25 that lock (55 deg off on both hip yaws) keeps 8% of the weight.
    stand_joints = STROKE_LEG_JOINTS + STROKE_ARM_JOINTS
    cfg.rewards["stroke_stand_pose"] = RewardTermCfg(
        func=xiaoxiaobai_mdp.standing_reference_pose,
        weight=4.0,
        params={"targets": STROKE_STAND_POSE, "std": 0.25,
                "asset_cfg": SceneEntityCfg("robot", joint_names=tuple(
                    f"{side}_{j}" for side in ("right", "left") for j in stand_joints), preserve_order=True)},
    )

    # Turning was given up (yaw error 1.1 in the walk start, 2.5 by v7) under the twist.
    cfg.rewards["track_angular_velocity"].weight = 5.0
    cfg.rewards["yaw_rate_error"] = RewardTermCfg(func=xiaoxiaobai_mdp.yaw_rate_error_l2, weight=-0.3)
    # With the twist down to +-1 rad/s the robot still turned a quarter of a 0.4 rad/s command
    # either way: the instantaneous kernels see the twist, not the turn. Low-passed over 1 s
    # (the twist alternates every 0.6 s) the turn itself is paid. At std 0.15 turning 0.1 of
    # 0.4 keeps 2% of the weight.
    cfg.rewards["track_yaw_rate_lowpass"] = RewardTermCfg(
        func=xiaoxiaobai_mdp.track_yaw_rate_filtered, weight=6.0, params={"std": 0.15, "tau": 1.0},
    )

    # On one skate every robot drove some motor past 0.9 of its current limit within 10 s;
    # the swizzle (skate.onnx) did in 13%. At 0.8 and -15, 79-87% still did on every floor;
    # at -30 a motor at the limit costs 1.2.
    cfg.rewards["motor_load"].params["max_load"] = 0.8
    cfg.rewards["motor_load"].weight = -30.0
    # The grounded skates slide sideways at ~0.03 m/s.
    cfg.rewards["skate_side_slip"].weight = -10.0
    # Pushing every stroke, on tile with free bearings the policy went 0.46 m/s for 0.3 and
    # ran off the terrain or fell (14%); the swizzle held 0.34. At -100, 0.16 m/s over the
    # command costs 1.7 a step. Somewhat faster on a free floor is fine; 0.08 over is free.
    cfg.rewards["overspeed"] = RewardTermCfg(
        func=xiaoxiaobai_mdp.overspeed_l2, weight=-100.0, params={"band": 0.08},
    )

    # The walking pose term pulls the legs to the straight default; the reference sets them.
    for key in ("std_walking", "std_running"):
        cfg.rewards["pose"].params[key].update({rf".*{j}.*": 1.0 for j in STROKE_LEG_JOINTS})
    # The skate task's knee cap (30 deg), on either leg, kept the skater upright; the lifted
    # leg folds 70 deg to clear the floor and the gliding one crouches on 32.
    cfg.rewards["knee_bend"].func = xiaoxiaobai_mdp.stance_joint_opening_excess
    cfg.rewards["knee_bend"].params["sensor_name"] = "feet_ground_contact"
    cfg.rewards["knee_bend"].params["max_opening"] = math.radians(50.0)

    # Following the reference to 10 deg rms, the pushing skate still went 70 mm behind the
    # trunk and only 15 mm out, then swung 80 mm forward at twice the trunk's speed: a long
    # walking stride. Skates sit ~10 mm ahead of the trunk when gliding; at weight -1500 a
    # skate 35 mm past the band costs 1.8 a step.
    cfg.rewards["foot_sagittal_excursion"] = RewardTermCfg(
        func=xiaoxiaobai_mdp.foot_sagittal_excursion,
        weight=-1500.0,
        params={"min_x": -0.035, "max_x": 0.03, "asset_cfg": SceneEntityCfg("robot", site_names=FEET)},
    )

    # The skate task's smoothing (action rate -1, second difference -0.5) was sized to calm a
    # policy that already skated. Here, from the start, it shrank the action noise from 1.0
    # to 0.19 in 200 iterations, before any skate had left the floor. The walk's -0.3 holds
    # while the stroke is learnt; the skate task's weights return once it is.
    cfg.rewards["action_rate_l2"].weight = -0.3
    cfg.rewards["action_acc_l2"].weight = 0.0
    if not play:
        cfg.curriculum["smoothing_curriculum"] = CurriculumTermCfg(
            func=xiaoxiaobai_mdp.step_based_staged_curriculum,
            params={
                "stages": [
                    {"name": "skate smoothing", "step": 5500 * 24,
                     "apply": lambda env: xiaoxiaobai_mdp.set_reward_weights(
                         env, {"action_rate_l2": -1.0, "action_acc_l2": -0.5})},
                ],
            },
        )

    return cfg


XiaoxiaobaiSkateStrokeRlCfg = replace(
    XiaoxiaobaiSkateRlCfg,
    wandb_project="mjlab_xiaoxiaobai_skate_stroke",
    experiment_name="mjlab_xiaoxiaobai_skate_stroke",
)
