# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

import codecs
import os
import signal
import subprocess
import sys
import select
import shlex
import termios
import tty
import threading
from pathlib import Path

from constants import WALK_AGENT_DEFAULT, WALK_AGENTS
from input.input_source import InputSource, UserInput

VELOCITY_STEP = 0.1
VELOCITY_MAX = 1.0
IMU_CORR_STEP_DEG = 0.5
IMU_CORR_MAX_DEG = 6.0
AGENT_KEYS = "cnmlpoj"  # selects WALK_AGENTS[0], [1], [2], [3], [4], [5], [6]

_ESCAPE = "\x1b"
_ARROW_UP = "\x1b[A"
_ARROW_DOWN = "\x1b[B"
_ARROW_RIGHT = "\x1b[C"
_ARROW_LEFT = "\x1b[D"


class KeyboardInputSource(InputSource):
    """Read keyboard input from a raw terminal in a background daemon thread.

    Args:
        move_keys: mapping from key character to move name, e.g. {"h": "head", "w": "walk"}.
        exclusive_moves: moves of which at most one is enabled; enabling one disables the others.
        stop_flag_path: path to the stop flag file polled by the scheduler.
    """

    def __init__(
        self,
        move_keys: dict[str, str],
        exclusive_moves: frozenset[str] = frozenset(),
        stop_flag_path: str = "/tmp/xiaoxiaobai_scheduler.stop",
    ) -> None:
        self._move_keys = move_keys
        self._exclusive_moves = exclusive_moves
        self._stop_flag_path = Path(stop_flag_path)
        self._state = UserInput(walk_agent=WALK_AGENT_DEFAULT)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False
        self._old_settings: list | None = None
        self._voice: subprocess.Popen | None = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        self._print_help()

    def stop(self) -> None:
        self._running = False
        self._stop_voice()
        self._restore_terminal()

    def read(self) -> UserInput:
        with self._lock:
            return UserInput(
                active_moves=set(self._state.active_moves),
                velocity=dict(self._state.velocity),
                show_imu=self._state.show_imu,
                imu_roll_corr_deg=self._state.imu_roll_corr_deg,
                imu_pitch_corr_deg=self._state.imu_pitch_corr_deg,
                walk_agent=self._state.walk_agent,
            )

    # ------------------------------------------------------------------
    # Internal

    def _read_loop(self) -> None:
        fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while self._running:
                # Use select with a timeout so the loop can notice _running=False.
                if not select.select([sys.stdin], [], [], 0.1)[0]:
                    continue
                key = self._read_key(fd)
                if key:
                    self._handle_key(key)
        finally:
            self._restore_terminal()

    def _read_key(self, fd: int) -> str:
        ch = os.read(fd, 1).decode("utf-8", errors="replace")
        if ch == _ESCAPE:
            # Try to read the CSI sequence that follows (e.g., "[A" for arrow up).
            if select.select([sys.stdin], [], [], 0.05)[0]:
                rest = os.read(fd, 2).decode("utf-8", errors="replace")
                return ch + rest
        return ch

    def _handle_key(self, key: str) -> None:
        if key in self._move_keys:
            move_name = self._move_keys[key]
            with self._lock:
                if move_name in self._state.active_moves:
                    self._state.active_moves.discard(move_name)
                    print(f"Move '{move_name}' disabled", end="\r\n", flush=True)
                else:
                    if move_name in self._exclusive_moves:
                        for other in self._exclusive_moves & self._state.active_moves:
                            self._state.active_moves.discard(other)
                            print(f"Move '{other}' disabled", end="\r\n", flush=True)
                    self._state.active_moves.add(move_name)
                    print(f"Move '{move_name}' enabled", end="\r\n", flush=True)
        elif key == "x":
            with self._lock:
                self._state.velocity = {"vx": 0.0, "vy": 0.0, "vtheta": 0.0}
            print("Velocity reset to zero", end="\r\n", flush=True)
        elif key in ("q", "\x03"):  # q or Ctrl+C
            self._stop_flag_path.write_text("stop\n", encoding="ascii")
            print("Stop requested", end="\r\n", flush=True)
        elif key == "i":
            with self._lock:
                self._state.show_imu = not self._state.show_imu
            status = "enabled" if self._state.show_imu else "disabled"
            print(f"IMU display {status}", end="\r\n", flush=True)
        elif key == "b":
            from esp32_link import request_sound

            request_sound()
            print("Speaker test requested", end="\r\n", flush=True)
        elif key == "t":
            self._toggle_voice()
        elif key == "a":
            self._adjust_imu_corr("roll", +IMU_CORR_STEP_DEG)
        elif key == "d":
            self._adjust_imu_corr("roll", -IMU_CORR_STEP_DEG)
        elif key == "w":
            self._adjust_imu_corr("pitch", +IMU_CORR_STEP_DEG)
        elif key == "s":
            self._adjust_imu_corr("pitch", -IMU_CORR_STEP_DEG)
        elif key == "z":
            self._adjust_imu_corr("roll", None)
            self._adjust_imu_corr("pitch", None)
        elif key in AGENT_KEYS[:len(WALK_AGENTS)]:
            index = AGENT_KEYS.index(key)
            with self._lock:
                self._state.walk_agent = index
            label, name = WALK_AGENTS[index]
            print(f"Walk policy -> [{key}] {label} ({name})", end="\r\n", flush=True)
        elif key == _ARROW_UP:
            self._adjust_velocity("vx", +VELOCITY_STEP)
        elif key == _ARROW_DOWN:
            self._adjust_velocity("vx", -VELOCITY_STEP)
        elif key == _ARROW_RIGHT:
            self._adjust_velocity("vtheta", -VELOCITY_STEP)
        elif key == _ARROW_LEFT:
            self._adjust_velocity("vtheta", +VELOCITY_STEP)

    def _adjust_velocity(self, axis: str, delta: float) -> None:
        with self._lock:
            v = self._state.velocity
            v[axis] = max(-VELOCITY_MAX, min(VELOCITY_MAX, v[axis] + delta))
        print(f"{axis}={self._state.velocity[axis]:.1f}", end="\r\n", flush=True)

    def _adjust_imu_corr(self, axis: str, delta: float | None) -> None:
        """delta=None resets to the default from constants. Positive roll rights the robot
        toward the left, positive pitch rights it forward."""
        field = f"imu_{axis}_corr_deg"
        with self._lock:
            corr = getattr(UserInput(), field) if delta is None else getattr(self._state, field) + delta
            corr = max(-IMU_CORR_MAX_DEG, min(IMU_CORR_MAX_DEG, corr))
            setattr(self._state, field, corr)
        pos, neg = ("left", "right") if axis == "roll" else ("forward", "backward")
        side = pos if corr > 0 else neg if corr < 0 else "none"
        print(f"IMU {axis} correction {corr:+.1f} deg (pushes robot {side})", end="\r\n", flush=True)

    def _toggle_voice(self) -> None:
        proc = self._voice
        if proc is not None and proc.poll() is None:
            self._stop_voice()
            print("Voice chat stopped", end="\r\n", flush=True)
            return
        repo = Path(__file__).resolve().parents[2]
        env = dict(os.environ, PYTHONPATH=str(repo / "src"), PYTHONUNBUFFERED="1")
        self._voice = subprocess.Popen(
            [sys.executable, "-u", "src/voice_chat.py", "--verbose", "--show-user-text",
             *shlex.split(os.environ.get("XIAOXIAOBAI_VOICE_ARGS", ""))],
            cwd=repo,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        threading.Thread(target=self._echo_voice, args=(self._voice,), daemon=True).start()
        print("Voice chat started (ESP32 direct). Press t again to stop.", end="\r\n", flush=True)

    @staticmethod
    def _echo_voice(proc: subprocess.Popen) -> None:
        """The terminal is in raw mode: every line needs its carriage return."""
        assert proc.stdout is not None
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        while chunk := os.read(proc.stdout.fileno(), 4096):
            text = decoder.decode(chunk).replace("\r\n", "\n").replace("\n", "\r\n")
            print(text, end="", flush=True)

    def _stop_voice(self) -> None:
        proc, self._voice = self._voice, None
        if proc is None or proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, signal.SIGINT)
        except OSError:
            proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()

    def _restore_terminal(self) -> None:
        if self._old_settings is not None:
            try:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_settings)
            except Exception:
                pass
            self._old_settings = None

    def _print_help(self) -> None:
        lines = [
            "Keyboard controls:",
            *(f"  [{key}]      toggle move '{name}'" for key, name in self._move_keys.items()),
            "  [i]      toggle IMU/gyro display",
            "  [b]      speaker test (ESP32)",
            "  [t]      voice chat (ESP32 talks to Qwen; press again to stop)",
            "  [arrows] vx (up/down), vtheta (left/right)",
            "  [x]      reset velocity to zero",
            f"  [a]/[d]  IMU roll correction +/-{IMU_CORR_STEP_DEG} deg (a pushes the robot left)",
            f"  [w]/[s]  IMU pitch correction +/-{IMU_CORR_STEP_DEG} deg (w pushes the robot forward)",
            f"  [z]      reset IMU corrections to roll {UserInput().imu_roll_corr_deg:+.1f}, pitch {UserInput().imu_pitch_corr_deg:+.1f}",
            *(f"  [{key}]      walk policy {label} ({name}){' - default' if i == WALK_AGENT_DEFAULT else ''}"
              for i, (key, (label, name)) in enumerate(zip(AGENT_KEYS, WALK_AGENTS))),
            "  [q]      stop scheduler",
        ]
        for line in lines:
            print(line, end="\r\n", flush=True)