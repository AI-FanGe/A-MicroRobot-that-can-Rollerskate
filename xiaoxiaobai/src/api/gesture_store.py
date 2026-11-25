# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

"""Upper-body moves taught in the web UI's editor: named keyframe lists, played back as recorded.

Each move has an id (`my_...`, what the assistant and the API name it by) and the name the user
gave it, which is also what they say to have it done ("做一下螃蟹舞").
"""

from __future__ import annotations

import re
import threading
import time
from pathlib import Path
from typing import Any

import gestures
from api.memory import new_id
from api.protocol import ApiError, CommonState, Reason
from api.store import DATA_DIR, read_json, write_json_atomic

GESTURES_PATH = DATA_DIR / "gestures.json"
MAX_MOVES = 100
NAME_MAX = 16
ID_PREFIX = "my"
# Said around a move's name, left out when matching what was heard against the names.
_FILLER = re.compile(r"[\s,，。.!！?？、~～'\"“”‘’]+")


def _plain(text: str) -> str:
    return _FILLER.sub("", text or "").lower()


class GestureStore:
    def __init__(self, path: Path = GESTURES_PATH) -> None:
        self._path = path
        self._lock = threading.Lock()
        data = read_json(path, {"moves": []})
        self._moves: list[dict[str, Any]] = [m for m in data.get("moves", []) if isinstance(m, dict) and m.get("id")]

    def _save(self) -> None:
        write_json_atomic(self._path, {"moves": self._moves})

    @staticmethod
    def summary(move: dict[str, Any]) -> dict[str, Any]:
        keys = move.get("keys") or []
        seconds = sum(k["dur"] + k["hold"] for k in keys)
        return {**move, "frames": len(keys), "seconds": round(seconds, 2)}

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self.summary(m) for m in self._moves]

    def get(self, move_id: str) -> dict[str, Any]:
        with self._lock:
            for move in self._moves:
                if move["id"] == move_id:
                    return dict(move)
        raise ApiError(f"没有自定义动作 {move_id}", CommonState.INVALID, Reason.NOT_FOUND, 404)

    def find(self, name_or_id: str) -> dict[str, Any] | None:
        wanted = _plain(name_or_id)
        with self._lock:
            for move in self._moves:
                if move["id"] == name_or_id or _plain(move["name"]) == wanted:
                    return dict(move)
        return None

    def heard(self, text: str) -> dict[str, Any] | None:
        """The move whose name is in what was said (the longest one, if several are)."""
        said = _plain(text)
        with self._lock:
            hits = [m for m in self._moves if _plain(m["name"]) and _plain(m["name"]) in said]
        return dict(max(hits, key=lambda m: len(_plain(m["name"])))) if hits else None

    def names(self) -> list[str]:
        with self._lock:
            return [m["name"] for m in self._moves]

    def save(self, name: Any, keys: Any, move_id: str | None = None) -> dict[str, Any]:
        """Create a move, or replace the one with `move_id` (rename and/or new keyframes)."""
        name = str(name or "").strip()[:NAME_MAX]
        if not name:
            raise ApiError("给动作起个名字", CommonState.INVALID)
        builtin = {g["label"] for g in gestures.catalog()} | set(gestures.GESTURE_NAMES)
        if name in builtin:
            raise ApiError(f"「{name}」和预设动作重名了，换个名字", CommonState.INVALID)
        clean = gestures.custom_keys(keys, gestures.TAUGHT_MAX_KEYS, gestures.TAUGHT_MAX_S)
        if not clean:
            raise ApiError("至少记录一个关键帧", CommonState.INVALID)
        now = time.time()
        with self._lock:
            for move in self._moves:
                if move["id"] != move_id and _plain(move["name"]) == _plain(name):
                    raise ApiError(f"已经有叫「{name}」的动作了", CommonState.INVALID)
            for move in self._moves:
                if move["id"] == move_id:
                    move.update(name=name, keys=clean, updated=now)
                    self._save()
                    return self.summary(move)
            if move_id:
                raise ApiError(f"没有自定义动作 {move_id}", CommonState.INVALID, Reason.NOT_FOUND, 404)
            if len(self._moves) >= MAX_MOVES:
                raise ApiError(f"最多保存 {MAX_MOVES} 个自定义动作", CommonState.INVALID)
            move = {"id": new_id(ID_PREFIX), "name": name, "keys": clean, "created": now, "updated": now}
            self._moves.append(move)
            self._save()
            return self.summary(move)

    def delete(self, move_id: str) -> dict[str, Any]:
        with self._lock:
            for i, move in enumerate(self._moves):
                if move["id"] == move_id:
                    del self._moves[i]
                    self._save()
                    return move
        raise ApiError(f"没有自定义动作 {move_id}", CommonState.INVALID, Reason.NOT_FOUND, 404)
