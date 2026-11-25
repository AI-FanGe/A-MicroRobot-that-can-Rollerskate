# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

"""Where the robot API keeps its settings (secrets, 0600) and its data (knowledge, memory, chats)."""

from __future__ import annotations

import copy
import json
import os
import threading
from pathlib import Path
from typing import Any

CONFIG_DIR = Path(os.environ.get("XIAOXIAOBAI_CONFIG_DIR") or Path.home() / ".config" / "xiaoxiaobai")
DATA_DIR = Path(os.environ.get("XIAOXIAOBAI_DATA_DIR") or Path.home() / ".local" / "share" / "xiaoxiaobai")
SETTINGS_PATH = CONFIG_DIR / "robot_api.json"
DASHSCOPE_KEY_PATH = CONFIG_DIR / "dashscope_api_key"


def write_json_atomic(path: Path, payload: Any, private: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if private:
        os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def qwen_api_key(voice_settings: dict[str, Any]) -> tuple[str | None, str]:
    """The Qwen key and where it came from: voice settings, $DASHSCOPE_API_KEY or `make voice-key`'s file."""
    configured = (voice_settings.get("api_key") or "").strip()
    if configured:
        return configured, "settings"
    env = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if env:
        return env, "env"
    try:
        key = DASHSCOPE_KEY_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        key = ""
    return (key or None), ("file" if key else "none")


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return copy.deepcopy(default)


ROBOT_NAME = "小小白"
_PERSONA_BODY = (
    "，一个约 30 厘米高、会走路的开源桌面人形机器人，有 19 个舵机、一块点阵表情屏、"
    "头上的摄像头、麦克风和扬声器。你性格友好、好奇、有点俏皮。"
    "用简短自然的中文回答，通常一到三句话；需要列步骤时再分点。"
)
DEFAULT_PERSONA = f"你是{ROBOT_NAME}{_PERSONA_BODY}"
# Defaults saved before the rename; settings still holding them follow the new name.
_OLD_DEFAULTS = {"name": "Xiaoxiaobai", "persona": f"你是 Xiaoxiaobai{_PERSONA_BODY}"}

DEFAULT_SETTINGS: dict[str, Any] = {
    "robot": {"name": ROBOT_NAME, "persona": DEFAULT_PERSONA},
    "llm": {
        "active": "",
        "providers": [],
        "temperature": 0.7,
        "max_tokens": 1024,
        "history_turns": 8,
        "allow_motion_tools": True,
        "voice_motion": True,
        "allow_expression_tools": True,
        "auto_memory": True,
        "memory_rev": 2,
        "memory_recall": True,
        "kb_top_k": 4,
        "web_search": True,
    },
    "voice": {
        "route": "esp32",
        "voice": "Mochi",
        "volume": 100,
        "volume_rev": 2,
        "mic_gain": 2.0,
        "camera": True,
        "full_duplex": False,
        "search": True,
        "search_rev": 2,
        "show_user_text": True,
        "share_persona": True,
        "url": "",
        "api_key": "",
        "speech_rate": 1.0,
        "volc_api_key": "",
        "volc_resource": "",
    },
    "safety": {"low_battery_v": 6.4, "warn_battery_v": 6.8},
    "display": {"brightness": 100},
    "memo": {
        "enabled": True,
        # The robot's clock may be set to another zone; "today" follows this one.
        "timezone": "Asia/Shanghai",
        "show_on_screen": True,
        "board_seconds": 30,
        "voice_capture": True,
        "remind": True,
    },
    "workbuddy": {
        "enabled": False,
        "url": "",
        "password": "",
        "voice_trigger": True,
        "triggers": ["让电脑", "叫电脑", "电脑帮我", "在电脑上", "用电脑", "WorkBuddy", "work buddy"],
        "chat_tool": True,
        "announce": True,
        "timeout_s": 600,
    },
    "music": {
        "enabled": True,
        "volume": 60,
        "quality": "exhigh",
        "voice_control": True,
        "resume_voice": True,
        "show_on_screen": True,
        "groove": True,
        "groove_size": 70,
        "groove_offset_ms": 0,
        "dance_music": True,
        "dance_volume": 80,
    },
}


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


class Settings:
    """settings.json with defaults merged in; secrets never leave through public()."""

    def __init__(self, path: Path = SETTINGS_PATH) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._data = _merge(DEFAULT_SETTINGS, read_json(path, {}))
        robot = self._data["robot"]
        for key, old in _OLD_DEFAULTS.items():
            if robot.get(key) == old:
                robot[key] = DEFAULT_SETTINGS["robot"][key]
        # Speech was too quiet at the old default (60): settings saved before go to full volume
        # once; volume_rev marks them so a volume set later is kept.
        voice = self._data["voice"]
        saved = read_json(path, {}).get("voice", {})
        if saved.get("volume_rev", 1) < 2:
            voice["volume"], voice["volume_rev"] = 100, 2
        # Web search used to be off by default: on once, the same way.
        if saved.get("search_rev", 1) < 2:
            voice["search"], voice["search_rev"] = True, 2
        # Automatic memory used to be off by default: on once, the same way.
        llm = self._data["llm"]
        if read_json(path, {}).get("llm", {}).get("memory_rev", 1) < 2:
            llm["auto_memory"], llm["memory_rev"] = True, 2

    def get(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._data)

    def section(self, name: str) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._data.get(name, {}))

    def update(self, name: str, values: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            current = self._data.get(name, {})
            self._data[name] = _merge(current, values) if isinstance(current, dict) else values
            self._save()
            return copy.deepcopy(self._data[name])

    def replace(self, name: str, value: Any) -> None:
        with self._lock:
            self._data[name] = copy.deepcopy(value)
            self._save()

    def _save(self) -> None:
        write_json_atomic(self._path, self._data, private=True)
        try:
            os.chmod(self._path.parent, 0o700)
        except OSError:
            pass


def mask_secret(value: str) -> str:
    value = value or ""
    if not value:
        return ""
    if len(value) <= 8:
        return "•" * len(value)
    return f"{value[:3]}••••{value[-4:]}"


def is_masked(value: str | None) -> bool:
    return bool(value) and "••" in str(value)
