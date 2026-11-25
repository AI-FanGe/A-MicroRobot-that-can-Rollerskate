#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

"""Replay the command sequence of a real walk recording through the deploy pipeline
(Scheduler + WalkMove + MuJoCo BAM controller), headless and faster than real time.

The result is a recording in the same format as the robot's, so analyze_walk_rec.py and
compare_walk_recs.py can compare real and sim tick by tick. Optional hardware faults:
per-motor KP scale (weak motor), joint zero offsets, trunk CoM offset.
"""

from __future__ import annotations

import contextlib
import io
import math
import tempfile
import threading
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


class _NoViewer:
    def __init__(self) -> None:
        self.opt = mujoco.MjvOption()

    def is_running(self) -> bool:
        return True

    def sync(self) -> None:
        pass

    def lock(self):
        return threading.Lock()


def run_replay(
    cmds: np.ndarray,
    kp_scale: dict[str, float] | None = None,
    offsets_deg: dict[str, float] | None = None,
    com: tuple[float, float, float] = (0.0, 0.0, 0.0),
    agent_path: str | None = None,
    delays: dict[str, int] | None = None,
    edit_model=None,
):
    """Return (columns, meta) of a sim recording that follows `cmds` (N x 3, physical units).

    `delays` overrides MuJoCoController delay arguments by name, e.g. {"delay_act_steps": 6}.
    `edit_model(model)` may modify the MjModel (contact, friction, masses) before the run.
    """
    mujoco.viewer.launch_passive = lambda *a, **k: _NoViewer()
    import scheduler as sched_mod
    from analyze_walk_rec import load
    from constants import MOTOR_TO_ID
    from input.input_source import InputSource, UserInput
    from moves.rotate_head import RotateHeadMove
    from moves.walk import WalkMove
    from sim.mujoco_controller import MuJoCoController

    stop_flag = Path("/tmp/xiaoxiaobai_scheduler.stop")

    class _Replay(InputSource):
        def __init__(self) -> None:
            self.k = 0

        def read(self) -> UserInput:
            k = self.k
            self.k += 1
            if k >= len(cmds):
                stop_flag.write_text("stop\n")
                return UserInput()
            vx, vy, wz = cmds[k]
            return UserInput(active_moves={"walk"}, velocity={"vx": vx, "vy": vy, "vtheta": wz})

    scale = {MOTOR_TO_ID[j]: s for j, s in (kp_scale or {}).items()}
    offsets = {MOTOR_TO_ID[j]: math.radians(e) for j, e in (offsets_deg or {}).items()}
    delay_args = {"delay_act_steps": 2, "delay_vel_ticks": 1, "delay_gyro_ticks": 3, "delay_quat_ticks": 4}
    delay_args.update(delays or {})
    ctrl = MuJoCoController(mjcf_path="src/model/mjcf/scene.xml", root_z=0.181, use_bam=True,
                            trunk_com_offset=com, **delay_args)
    if edit_model is not None:
        edit_model(ctrl._model)
    write_kp, write_goal, read_pos = ctrl.sync_write_kp, ctrl.sync_write_goal_position, ctrl.sync_read_present_position
    ctrl.sync_write_kp = lambda ids, gains: write_kp(ids, [g * scale.get(i, 1.0) for i, g in zip(ids, gains)])
    ctrl.sync_write_goal_position = lambda ids, pos: write_goal(ids, [p + offsets.get(i, 0.0) for i, p in zip(ids, pos)])
    ctrl.sync_read_present_position = lambda ids: [q - offsets.get(i, 0.0) for i, q in zip(ids, read_pos(ids))]

    scale_velocity, sleep = sched_mod.scale_velocity, time.sleep
    sched_mod.scale_velocity = lambda v: dict(v)
    time.sleep = lambda s: None
    try:
        s = sched_mod.Scheduler(frequency_hz=50.0, controller=ctrl, input_source=_Replay(),
                                moves={"head": RotateHeadMove(), "walk": WalkMove(controller=ctrl, agent_path=agent_path)})
        out = Path(tempfile.mkdtemp(prefix="sim_replay_"))
        s.recorder.out_dir = out
        with contextlib.redirect_stdout(io.StringIO()):
            s.run()
    finally:
        sched_mod.scale_velocity, time.sleep = scale_velocity, sleep
    return load(next(out.glob("walk_*.npz")))
