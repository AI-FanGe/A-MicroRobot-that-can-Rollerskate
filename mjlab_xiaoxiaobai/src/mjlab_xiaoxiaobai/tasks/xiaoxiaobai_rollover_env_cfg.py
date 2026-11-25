# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

"""Xiaoxiaobai roll-over environment: from lying on the back (or any non face-down pose),
roll onto the front with the limbs near the ready pose and lie still.

The get-up policy stands up from the front (86% from flat on the front) but never learned
to from the back, and gating its rewards on the pose left it unable to do either. So the
robot runs this policy until it lies face down, then hands over to the get-up policy
(xiaoxiaobai GetupMove). Built on the get-up environment: same robot, observations, actions,
actuators, domain randomization and speed / friction penalties.
"""

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from mjlab_xiaoxiaobai.tasks import mdp as xiaoxiaobai_mdp
from mjlab_xiaoxiaobai.tasks.xiaoxiaobai_getup_env_cfg import ACTUATED, XiaoxiaobaiGetupRlCfg, make_xiaoxiaobai_getup_env_cfg

# Hand-over to the get-up policy (mirrored in xiaoxiaobai constants): face down past this
# gravity x, trunk turning slower than this, for this long.
HANDOVER_MIN_GX = 0.8
HANDOVER_MAX_ANG_VEL = 0.5
HANDOVER_HOLD_S = 0.3


def make_xiaoxiaobai_rollover_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_xiaoxiaobai_getup_env_cfg(play=play)
    cfg.episode_length_s = 6.0
    actuated = SceneEntityCfg("robot", joint_names=(ACTUATED,))

    #---------------------------- Events ----------------------------
    # Mostly flat on the back, as real falls end; the rest random drops (sides, head, odd
    # limbs) that also have to end face down. Never face down already: the get-up runs then.
    cfg.events["reset_getup"].params.update(
        {"standing_fraction": 0.0, "flat_fraction": 0.8, "supine_fraction": 1.0, "flat_tilt": 0.5}
    )
    del cfg.events["getup_assist"]

    #---------------------------- Rewards ---------------------------
    # Face down, then the ready pose and stillness there. The get-up's joint limit, self
    # contact, current, speed and friction penalties are kept as they are.
    kept = (
        "dof_pos_limits", "action_rate_l2", "joint_vel_l2", "self_collisions", "command_lead_excess",
        "total_current", "motor_load", "motor_load_l2", "root_speed_xy", "body_speed", "trunk_ang_speed", "joint_speed",
        "friction_use", "contact_slip",
    )
    cfg.rewards = {
        "face_down": RewardTermCfg(func=xiaoxiaobai_mdp.rollover_face_down, weight=2.0),
        "prone_pose": RewardTermCfg(
            func=xiaoxiaobai_mdp.rollover_prone_pose, weight=1.0, params={"std": 0.3, "asset_cfg": actuated}
        ),
        "prone_still": RewardTermCfg(
            func=xiaoxiaobai_mdp.rollover_prone_still, weight=1.0, params={"std": 1.0, "asset_cfg": actuated}
        ),
        **{name: replace(cfg.rewards[name]) for name in kept},
    }

    #---------------------------- Terminations ----------------------
    # Ends where the robot hands over to the get-up, as a time-out so lying there stays valued.
    cfg.terminations.pop("stood_still", None)
    if not play:
        cfg.terminations["rolled"] = TerminationTermCfg(
            func=xiaoxiaobai_mdp.rollover_done,
            time_out=True,
            params={"hold_s": HANDOVER_HOLD_S, "min_gx": HANDOVER_MIN_GX, "max_ang_vel": HANDOVER_MAX_ANG_VEL},
        )

    #---------------------------- Curriculum ------------------------
    cfg.curriculum = {}
    return cfg


XiaoxiaobaiRolloverRlCfg = replace(
    XiaoxiaobaiGetupRlCfg,
    experiment_name="mjlab_xiaoxiaobai_rollover",
    wandb_project="mjlab_xiaoxiaobai_rollover",
)
