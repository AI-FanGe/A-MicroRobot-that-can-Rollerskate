# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

"""In-process event bus feeding the web UI's Server-Sent Events stream."""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from typing import Any

SUBSCRIBER_QUEUE = 256


class EventBus:
    def __init__(self, history: int = 200) -> None:
        self._lock = threading.Lock()
        self._subscribers: list[queue.Queue] = []
        self._history: deque[dict[str, Any]] = deque(maxlen=history)

    def publish(self, kind: str, **data: Any) -> None:
        event = {"type": kind, "ts": time.time(), **data}
        with self._lock:
            if not kind.endswith("_delta"):
                self._history.append(event)
            subscribers = list(self._subscribers)
        for sub in subscribers:
            try:
                sub.put_nowait(event)
            except queue.Full:
                pass  # a stalled browser tab only loses events, never blocks the robot

    def alert(self, level: str, message: str, **data: Any) -> None:
        self.publish("alert", level=level, message=message, **data)

    def subscribe(self) -> queue.Queue:
        sub: queue.Queue = queue.Queue(maxsize=SUBSCRIBER_QUEUE)
        with self._lock:
            self._subscribers.append(sub)
        return sub

    def unsubscribe(self, sub: queue.Queue) -> None:
        with self._lock:
            if sub in self._subscribers:
                self._subscribers.remove(sub)

    def recent(self, kinds: set[str] | None = None, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            items = [e for e in self._history if kinds is None or e["type"] in kinds]
        return items[-limit:]
