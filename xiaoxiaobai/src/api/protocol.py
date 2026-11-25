# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

"""Response model shared by every endpoint, after the PrimeBot SDK CommonResponse.

Clients check header.code == 0 (the request was accepted) first, then state.value == 1
(SUCCESS) or a running/pending state; on failure state.reason says why and message
explains it in words.
"""

from __future__ import annotations

import itertools
import threading
import time
from enum import IntEnum
from typing import Any


class ResponseCode(IntEnum):
    SUCCESS = 0
    FAIL = 1


class CommonState(IntEnum):
    UNKNOWN = 0
    SUCCESS = 1
    FAILURE = 2
    ABORTED = 3
    TIMEOUT = 4
    INVALID = 5
    IN_MANUAL = 6
    NOT_READY = 100
    PENDING = 200
    CREATED = 300
    RUNNING = 400


class Reason(IntEnum):
    SUCCESS = 0
    STARTUP_CHECK = 2
    SHUTDOWN = 3
    LOW_BATTERY = 5
    ACTION = 9
    TASK_RUNNING = 15
    TARGET_STATE = 16
    NOT_FOUND = 17
    NO_TRANSFER_PATH = 22
    NOT_SUPPORTED = 26
    NOT_INTERRUPTIBLE = 28
    # Xiaoxiaobai additions, kept above the SDK range.
    INPUT_SOURCE_BUSY = 100
    NOT_CONFIGURED = 101
    UPSTREAM = 102
    HARDWARE = 103


REASON_TEXT = {
    Reason.SUCCESS: "成功",
    Reason.STARTUP_CHECK: "启动自检中",
    Reason.SHUTDOWN: "关机中",
    Reason.LOW_BATTERY: "电量过低",
    Reason.ACTION: "当前模式不支持",
    Reason.TASK_RUNNING: "其他任务正在运行",
    Reason.TARGET_STATE: "已经是目标状态",
    Reason.NOT_FOUND: "未找到",
    Reason.NO_TRANSFER_PATH: "当前状态无法直接切换到目标状态",
    Reason.NOT_SUPPORTED: "当前机型或环境不支持",
    Reason.NOT_INTERRUPTIBLE: "当前动作不可打断",
    Reason.INPUT_SOURCE_BUSY: "被更高优先级的输入源占用",
    Reason.NOT_CONFIGURED: "尚未配置",
    Reason.UPSTREAM: "上游服务出错",
    Reason.HARDWARE: "硬件或进程异常",
}


class ApiError(Exception):
    """Raised by handlers; turned into a CommonResponse with header.code = FAIL."""

    def __init__(
        self,
        message: str,
        state: CommonState = CommonState.FAILURE,
        reason: Reason = Reason.SUCCESS,
        http_status: int = 400,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.state = state
        self.reason = reason
        self.http_status = http_status


def response(
    data: Any = None,
    *,
    state: CommonState = CommonState.SUCCESS,
    reason: Reason = Reason.SUCCESS,
    message: str = "",
    task_id: int | None = None,
    code: ResponseCode = ResponseCode.SUCCESS,
) -> dict[str, Any]:
    return {
        "header": {"stamp": time.time(), "code": int(code)},
        "state": {
            "value": int(state),
            "name": state.name,
            "reason": int(reason),
            "reason_name": f"REASON_{reason.name}",
            "reason_text": "" if code == ResponseCode.FAIL and reason == Reason.SUCCESS else REASON_TEXT.get(reason, ""),
        },
        "message": message,
        "task_id": task_id,
        "data": data,
    }


def error_response(exc: ApiError) -> dict[str, Any]:
    return response(state=exc.state, reason=exc.reason, message=exc.message, code=ResponseCode.FAIL)


class TaskRegistry:
    """Asynchronous tasks (mode switches, timed walks); polled with GET /api/v1/tasks/<id>."""

    def __init__(self, keep: int = 200) -> None:
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self._tasks: dict[int, dict[str, Any]] = {}
        self._keep = keep

    def create(self, kind: str, **info: Any) -> int:
        task_id = next(self._ids)
        with self._lock:
            self._tasks[task_id] = {
                "task_id": task_id,
                "kind": kind,
                "state": CommonState.RUNNING.name,
                "created": time.time(),
                "updated": time.time(),
                "message": "",
                **info,
            }
            for old in sorted(self._tasks)[: max(0, len(self._tasks) - self._keep)]:
                self._tasks.pop(old, None)
        return task_id

    def finish(self, task_id: int, state: CommonState, message: str = "") -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task["state"] != CommonState.RUNNING.name:
                return
            task.update(state=state.name, message=message, updated=time.time())

    def finish_kind(self, kind: str, state: CommonState, message: str = "") -> None:
        with self._lock:
            ids = [tid for tid, t in self._tasks.items() if t["kind"] == kind and t["state"] == CommonState.RUNNING.name]
        for task_id in ids:
            self.finish(task_id, state, message)

    def get(self, task_id: int) -> dict[str, Any] | None:
        with self._lock:
            task = self._tasks.get(task_id)
            return dict(task) if task else None

    def recent(self, limit: int = 30) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(self._tasks[k]) for k in sorted(self._tasks, reverse=True)[:limit]]
