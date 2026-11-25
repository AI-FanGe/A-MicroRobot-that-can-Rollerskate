"""Radio calisthenics (广播体操): the dance task (xiaoxiaobai_dance_env_cfg.py) tracking a routine
choreographed on Xiaoxiaobai (scripts/gymnastics_motion.py, then scripts/dance_motion_npz.py).

The head yaw motor stays outside the policy's actions and observations, as in every task; here
it plays the routine's head turns from the reference. XIAOXIAOBAI_GYMNASTICS_MOTION overrides the
default motion file.
"""
import os
from dataclasses import dataclass, fields, replace
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnvCfg

from mjlab_xiaoxiaobai.tasks.xiaoxiaobai_dance_env_cfg import (
    DanceMotionCommand,
    DanceMotionCommandCfg,
    XiaoxiaobaiDanceRlCfg,
    make_xiaoxiaobai_dance_env_cfg,
)

MOTION_FILE = os.environ.get(
    "XIAOXIAOBAI_GYMNASTICS_MOTION",
    str(Path(__file__).resolve().parents[3] / "data/motions/xiaoxiaobai/gymnastics_v1_50hz.npz"),
)


# Frames the head target runs ahead of the reference: the motor's delay and response, which
# left it ~9 deg behind 20 deg turns when played on time.
HEAD_LEAD = 4


class GymnasticsMotionCommand(DanceMotionCommand):
    """The dance's motion command, plus the head motor's target on the reference."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._head = torch.tensor(self.robot.find_joints(("head",))[0], device=self.device)

    def _update_command(self):
        super()._update_command()
        ahead = torch.clamp(self.time_steps + HEAD_LEAD, max=self.motion.time_step_total - 1)
        self.robot.set_joint_position_target(self.motion.joint_pos[ahead][:, self._head], joint_ids=self._head)


@dataclass(kw_only=True)
class GymnasticsMotionCommandCfg(DanceMotionCommandCfg):
    def build(self, env) -> GymnasticsMotionCommand:
        return GymnasticsMotionCommand(self, env)


def make_xiaoxiaobai_gymnastics_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_xiaoxiaobai_dance_env_cfg(play=play)
    dance = cfg.commands["motion"]
    cfg.commands["motion"] = GymnasticsMotionCommandCfg(
        **{f.name: getattr(dance, f.name) for f in fields(dance) if f.init}
    )
    cfg.commands["motion"].motion_file = MOTION_FILE
    return cfg


XiaoxiaobaiGymnasticsRlCfg = replace(
    XiaoxiaobaiDanceRlCfg,
    wandb_project="mjlab_xiaoxiaobai_gymnastics",
    experiment_name="mjlab_xiaoxiaobai_gymnastics",
)
