"""Freestanding peek (偷看): the gymnastics tracker on a routine that balances on its feet.

The monitor in the reference video is scenery. It is not in the training scene, and the motion
never puts a hand on it. XIAOXIAOBAI_PEEK_MOTION overrides the default motion file.
"""
import os
from dataclasses import replace
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg

from mjlab_xiaoxiaobai.tasks.xiaoxiaobai_gymnastics_env_cfg import (
    XiaoxiaobaiGymnasticsRlCfg,
    make_xiaoxiaobai_gymnastics_env_cfg,
)

MOTION_FILE = os.environ.get(
    "XIAOXIAOBAI_PEEK_MOTION",
    str(Path(__file__).resolve().parents[3] / "data/motions/xiaoxiaobai/peek_v2_50hz.npz"),
)


def make_xiaoxiaobai_peek_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_xiaoxiaobai_gymnastics_env_cfg(play=play)
    cfg.commands["motion"].motion_file = MOTION_FILE
    # The lean is the move. At the dance's 0.4 rad std, half of a 25 deg lean cost under a
    # quarter of a 0.5 term, and the policy kept the trunk near upright.
    ori = cfg.rewards["motion_global_root_ori"]
    cfg.rewards["motion_global_root_ori"] = replace(ori, weight=1.5, params={**ori.params, "std": 0.2})
    return cfg


XiaoxiaobaiPeekRlCfg = replace(
    XiaoxiaobaiGymnasticsRlCfg,
    wandb_project="mjlab_xiaoxiaobai_peek",
    experiment_name="mjlab_xiaoxiaobai_peek",
    max_iterations=8_000,
)
