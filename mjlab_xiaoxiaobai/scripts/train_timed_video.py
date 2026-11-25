"""``mjlab train`` with preview videos triggered by wall-clock time.

Same CLI as ``uv run train``; with ``--video True`` a clip of ``--video-length``
frames starts every ``XIAOXIAOBAI_VIDEO_INTERVAL_S`` seconds (default 120).

    uv run python scripts/train_timed_video.py Mjlab-Velocity-Xiaoxiaobai \
        --env.scene.num-envs 4096 --video True --video-length 200
"""

from __future__ import annotations

import os
import time

from mjlab.scripts import train
from mjlab.utils.wrappers import VideoRecorder

VIDEO_INTERVAL_S = float(os.environ.get("XIAOXIAOBAI_VIDEO_INTERVAL_S", "120"))


class TimedVideoRecorder(VideoRecorder):
    def __init__(self, env, *args, step_trigger=None, **kwargs):
        del step_trigger
        super().__init__(env, *args, step_trigger=self._timed_trigger, **kwargs)
        self._next_video_s = time.monotonic()

    def _timed_trigger(self, step: int) -> bool:
        del step
        now = time.monotonic()
        if self.is_recording or now < self._next_video_s:
            return False
        self._next_video_s = now + VIDEO_INTERVAL_S
        return True


train.VideoRecorder = TimedVideoRecorder

if __name__ == "__main__":
    train.main()
