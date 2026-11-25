# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from mjlab_xiaoxiaobai.tasks.xiaoxiaobai_velocity_env_cfg import (
    make_xiaoxiaobai_velocity_env_cfg,
    XiaoxiaobaiVelocityRlCfg,
)
from mjlab_xiaoxiaobai.tasks.xiaoxiaobai_free_arms_env_cfg import make_xiaoxiaobai_free_arms_env_cfg
from mjlab_xiaoxiaobai.tasks.xiaoxiaobai_getup_env_cfg import (
    make_xiaoxiaobai_getup_env_cfg,
    XiaoxiaobaiGetupRlCfg,
)
from mjlab_xiaoxiaobai.tasks.xiaoxiaobai_rollover_env_cfg import (
    make_xiaoxiaobai_rollover_env_cfg,
    XiaoxiaobaiRolloverRlCfg,
)
from mjlab_xiaoxiaobai.tasks.xiaoxiaobai_skate_env_cfg import (
    make_xiaoxiaobai_skate_env_cfg,
    XiaoxiaobaiSkateRlCfg,
)
from mjlab_xiaoxiaobai.tasks.xiaoxiaobai_skate_stroke_env_cfg import (
    make_xiaoxiaobai_skate_stroke_env_cfg,
    XiaoxiaobaiSkateStrokeRlCfg,
)
from mjlab_xiaoxiaobai.tasks.xiaoxiaobai_dance_env_cfg import (
    make_xiaoxiaobai_dance_env_cfg,
    XiaoxiaobaiDanceRlCfg,
)
from mjlab_xiaoxiaobai.tasks.xiaoxiaobai_gymnastics_env_cfg import (
    make_xiaoxiaobai_gymnastics_env_cfg,
    XiaoxiaobaiGymnasticsRlCfg,
)
from mjlab_xiaoxiaobai.tasks.xiaoxiaobai_peek_env_cfg import (
    make_xiaoxiaobai_peek_env_cfg,
    XiaoxiaobaiPeekRlCfg,
)
from mjlab_xiaoxiaobai.tasks.xiaoxiaobai_propup_env_cfg import XiaoxiaobaiPropupRlCfg, make_xiaoxiaobai_propup_env_cfg
from mjlab_xiaoxiaobai.tasks.xiaoxiaobai_situp_env_cfg import XiaoxiaobaiSitupRlCfg, make_xiaoxiaobai_situp_env_cfg
from mjlab_xiaoxiaobai.tasks.xiaoxiaobai_sit_env_cfg import (
    make_xiaoxiaobai_sit_env_cfg,
    XiaoxiaobaiSitRlCfg,
)
from mjlab_xiaoxiaobai.tasks.xiaoxiaobai_drag_env_cfg import XiaoxiaobaiDragRlCfg, make_xiaoxiaobai_drag_env_cfg

register_mjlab_task(
    task_id="Mjlab-Velocity-Xiaoxiaobai",
    env_cfg=make_xiaoxiaobai_velocity_env_cfg(),
    play_env_cfg=make_xiaoxiaobai_velocity_env_cfg(play=True),
    rl_cfg=XiaoxiaobaiVelocityRlCfg,
    runner_cls=VelocityOnPolicyRunner,
)

# Walk backward dragging a load taped to both hands (xiaoxiaobai_drag_env_cfg.py).
register_mjlab_task(
    task_id="Mjlab-Drag-Xiaoxiaobai",
    env_cfg=make_xiaoxiaobai_drag_env_cfg(),
    play_env_cfg=make_xiaoxiaobai_drag_env_cfg(play=True),
    rl_cfg=XiaoxiaobaiDragRlCfg,
    runner_cls=VelocityOnPolicyRunner,
)

# Legs-only walk with the arms playing gestures (xiaoxiaobai_free_arms_env_cfg.py).
register_mjlab_task(
    task_id="Mjlab-Velocity-Xiaoxiaobai-FreeArms",
    env_cfg=make_xiaoxiaobai_free_arms_env_cfg(),
    play_env_cfg=make_xiaoxiaobai_free_arms_env_cfg(play=True),
    rl_cfg=XiaoxiaobaiVelocityRlCfg,
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Getup-Xiaoxiaobai",
    env_cfg=make_xiaoxiaobai_getup_env_cfg(),
    play_env_cfg=make_xiaoxiaobai_getup_env_cfg(play=True),
    rl_cfg=XiaoxiaobaiGetupRlCfg,
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Rollover-Xiaoxiaobai",
    env_cfg=make_xiaoxiaobai_rollover_env_cfg(),
    play_env_cfg=make_xiaoxiaobai_rollover_env_cfg(play=True),
    rl_cfg=XiaoxiaobaiRolloverRlCfg,
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Skate-Xiaoxiaobai",
    env_cfg=make_xiaoxiaobai_skate_env_cfg(),
    play_env_cfg=make_xiaoxiaobai_skate_env_cfg(play=True),
    rl_cfg=XiaoxiaobaiSkateRlCfg,
    runner_cls=VelocityOnPolicyRunner,
)

# Skating in alternating one-skate strokes, timed by a stroke clock (xiaoxiaobai_skate_stroke_env_cfg.py).
register_mjlab_task(
    task_id="Mjlab-Skate-Stroke-Xiaoxiaobai",
    env_cfg=make_xiaoxiaobai_skate_stroke_env_cfg(),
    play_env_cfg=make_xiaoxiaobai_skate_stroke_env_cfg(play=True),
    rl_cfg=XiaoxiaobaiSkateStrokeRlCfg,
    runner_cls=VelocityOnPolicyRunner,
)

# Sit down on a box behind the robot and stand back up (xiaoxiaobai_sit_env_cfg.py).
register_mjlab_task(
    task_id="Mjlab-Sit-Xiaoxiaobai",
    env_cfg=make_xiaoxiaobai_sit_env_cfg(),
    play_env_cfg=make_xiaoxiaobai_sit_env_cfg(play=True),
    rl_cfg=XiaoxiaobaiSitRlCfg,
    runner_cls=VelocityOnPolicyRunner,
)

# Prop up from lying and lie back down: tracking a choreographed reference (xiaoxiaobai_propup_env_cfg.py).
register_mjlab_task(
    task_id="Mjlab-PropUp-Xiaoxiaobai",
    env_cfg=make_xiaoxiaobai_propup_env_cfg(),
    play_env_cfg=make_xiaoxiaobai_propup_env_cfg(play=True),
    rl_cfg=XiaoxiaobaiPropupRlCfg,
    runner_cls=MotionTrackingOnPolicyRunner,
)

# Sit up from lying on the back on both hands and lie back down, on a command (xiaoxiaobai_situp_env_cfg.py).
register_mjlab_task(
    task_id="Mjlab-SitUp-Xiaoxiaobai",
    env_cfg=make_xiaoxiaobai_situp_env_cfg(),
    play_env_cfg=make_xiaoxiaobai_situp_env_cfg(play=True),
    rl_cfg=XiaoxiaobaiSitupRlCfg,
    runner_cls=VelocityOnPolicyRunner,
)

# The same with the box further back: steps back until a leg touches it, then sits.
register_mjlab_task(
    task_id="Mjlab-Sit-Xiaoxiaobai-Backup",
    env_cfg=make_xiaoxiaobai_sit_env_cfg(backup=True),
    play_env_cfg=make_xiaoxiaobai_sit_env_cfg(play=True, backup=True),
    rl_cfg=XiaoxiaobaiSitRlCfg,
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Dance-Xiaoxiaobai",
    env_cfg=make_xiaoxiaobai_dance_env_cfg(),
    play_env_cfg=make_xiaoxiaobai_dance_env_cfg(play=True),
    rl_cfg=XiaoxiaobaiDanceRlCfg,
    runner_cls=MotionTrackingOnPolicyRunner,
)

# Radio calisthenics: the dance task on a routine choreographed for Xiaoxiaobai (xiaoxiaobai_gymnastics_env_cfg.py).
register_mjlab_task(
    task_id="Mjlab-Gymnastics-Xiaoxiaobai",
    env_cfg=make_xiaoxiaobai_gymnastics_env_cfg(),
    play_env_cfg=make_xiaoxiaobai_gymnastics_env_cfg(play=True),
    rl_cfg=XiaoxiaobaiGymnasticsRlCfg,
    runner_cls=MotionTrackingOnPolicyRunner,
)

# Freestanding peek: same tracker, a lean that does not brace on the monitor.
register_mjlab_task(
    task_id="Mjlab-Peek-Xiaoxiaobai",
    env_cfg=make_xiaoxiaobai_peek_env_cfg(),
    play_env_cfg=make_xiaoxiaobai_peek_env_cfg(play=True),
    rl_cfg=XiaoxiaobaiPeekRlCfg,
    runner_cls=MotionTrackingOnPolicyRunner,
)
