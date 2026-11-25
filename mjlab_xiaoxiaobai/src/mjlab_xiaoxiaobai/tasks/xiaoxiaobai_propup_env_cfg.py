"""Prop up and lie back down (起身/躺下): the dance task (xiaoxiaobai_dance_env_cfg.py) tracking a
reference choreographed on Xiaoxiaobai (scripts/propup_motion.py --time-scale 0.27 --bend-s 0.6
--steps 1 --step-s 0.3 --lie 0.5 --hold 1.0 --smooth 0.04, then scripts/dance_motion_npz.py
--no-ready): from lying on the back, up on the elbows, then on the hands while they walk back,
to lean back at ~66 deg in 2.9 s; after 2 s there it plays back down in 3 s.

The robot is the get-up one (frictional hulls on every body, the measured joint stops, the
weighed mass, kp GETUP_KP) on a flat floor. Lying is the motion here, so the walk's fall
termination is off and the feet terms of the dance (steps) are left out; the get-up's current
and motor load limits keep the push within what the motors give. XIAOXIAOBAI_PROPUP_MOTION
overrides the default motion file.
"""
import os
from dataclasses import replace
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.terrains import TerrainEntityCfg

from mjlab_xiaoxiaobai.robot.xiaoxiaobai_constants import XIAOXIAOBAI_GETUP_ROBOT_CFG
from mjlab_xiaoxiaobai.tasks import mdp
from mjlab_xiaoxiaobai.tasks.xiaoxiaobai_dance_env_cfg import HANDS, XiaoxiaobaiDanceRlCfg, make_xiaoxiaobai_dance_env_cfg

MOTION_FILE = os.environ.get(
    "XIAOXIAOBAI_PROPUP_MOTION",
    str(Path(__file__).resolve().parents[3] / "data/motions/xiaoxiaobai/propup_fast_50hz.npz"),
)


def make_xiaoxiaobai_propup_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_xiaoxiaobai_dance_env_cfg(play=play)
    cfg.scene.entities["robot"] = XIAOXIAOBAI_GETUP_ROBOT_CFG
    cfg.scene.terrain = TerrainEntityCfg(terrain_type="plane")
    # Lying with friction on every box: up to 28 contacts per robot (get-up task).
    cfg.sim.nconmax = 96
    cfg.sim.njmax = 400
    cfg.episode_length_s = 1e9 if play else 8.0

    motion = cfg.commands["motion"]
    motion.motion_file = MOTION_FILE
    # Small offsets only: tilting a lying robot pushes its shell into the floor.
    motion.pose_range = {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (0.0, 0.004),
                         "roll": (-0.03, 0.03), "pitch": (-0.03, 0.03), "yaw": (-0.2, 0.2)}
    motion.velocity_range = {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.02, 0.02),
                             "roll": (-0.2, 0.2), "pitch": (-0.2, 0.2), "yaw": (-0.2, 0.2)}

    # Plastic shells on floors (get-up task); the hulls outrank the terrain.
    cfg.events["body_friction"] = EventTermCfg(
        mode="startup",
        func=dr.geom_friction,
        params={
            "asset_cfg": SceneEntityCfg("robot", geom_names=(r".*_ground_collision$",)),
            "operation": "abs",
            "ranges": (0.4, 1.5),
            "shared_random": True,
        },
    )
    if not play:
        cfg.events["push_robot"].params["velocity_range"] = {"x": (-0.1, 0.1), "y": (-0.1, 0.1)}

    for name in ("motion_feet_height", "motion_feet_contact"):
        del cfg.rewards[name]
    cfg.rewards["total_current"] = RewardTermCfg(
        func=mdp.total_current_excess, weight=-0.5, params={"max_current": 15.0}
    )
    cfg.rewards["motor_load"] = RewardTermCfg(func=mdp.motor_load_excess, weight=-5.0, params={"max_load": 0.6})

    del cfg.terminations["fell_over"]
    cfg.terminations["ee_hands_pos"].params["threshold"] = 0.05
    assert cfg.terminations["ee_hands_pos"].params["body_names"] == HANDS
    return cfg


XiaoxiaobaiPropupRlCfg = replace(
    XiaoxiaobaiDanceRlCfg,
    wandb_project="mjlab_xiaoxiaobai_propup",
    experiment_name="mjlab_xiaoxiaobai_propup",
    max_iterations=20_000,
)
