# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

import sys
import time
import numpy as np
import os
from pathlib import Path

from constants import CHILL_CLIPS, GYMNASTICS_AGENT, MOTOR_TO_ID, NEUTRAL_POSE, KP_DEFAULT, PEEK_AGENT
from display_smile import start_smile_animation
from robot_controller import RobotController
from scheduler import Scheduler
from input.input_source import CompositeInputSource, InputSource
from input.keyboard_input import KeyboardInputSource
from input.remote_input import RemoteInputSource
from moves.chill import ChillMove
from moves.dance import DanceMove
from moves.propup import PropUpMove
from moves.rotate_head import RotateHeadMove
from moves.skate import SkateMove
from moves.upper_body import UpperBodyMove
from moves.walk import WalkMove

PID_FILE = Path("/tmp/xiaoxiaobai_scheduler.pid")


def _another_session_running() -> bool:
    """True if a live control loop already owns the PID file (e.g. launched by the
    gamepad daemon). Avoids two instances fighting over the motor bus."""
    if not PID_FILE.exists():
        return False
    try:
        pid = int(PID_FILE.read_text(encoding="ascii").strip())
        os.kill(pid, 0)
    except (ValueError, OSError):
        return False
    return True

# Which moves can be toggled from the gamepad / keyboard.
MOVE_KEYS = {"h": "head", "v": "walk", "k": "skate", "u": "upper"}
# Each drives the legs and arms with its own KP.
EXCLUSIVE_MOVES = frozenset({"walk", "skate", "gymnastics", "peek", "propup", *CHILL_CLIPS})
GAMEPAD_BUTTON_MOVES = {"A": "walk", "Y": "skate", "X": "upper"}


def build_input_source() -> InputSource:
    """Merge the console remote channel with gamepad or keyboard.

    Override with XIAOXIAOBAI_INPUT=keyboard|gamepad|remote|auto.
    The remote file channel is always attached so the web console can drive
    a robot that was started from SSH without a TTY.
    """
    requested = os.environ.get("XIAOXIAOBAI_INPUT", "auto").lower()
    sources: list[InputSource] = [RemoteInputSource()]

    if requested in ("auto", "gamepad"):
        from input.gamepad_input import GamepadInputSource, find_gamepad_path

        if find_gamepad_path() is not None:
            sources.append(GamepadInputSource(button_moves=GAMEPAD_BUTTON_MOVES, exclusive_moves=EXCLUSIVE_MOVES))
            return CompositeInputSource(sources)
        if requested == "gamepad":
            raise RuntimeError("XIAOXIAOBAI_INPUT=gamepad but no gamepad was found.")
        print("No gamepad detected; using keyboard / remote input.")

    if requested == "keyboard" or (requested != "remote" and sys.stdin.isatty()):
        sources.append(KeyboardInputSource(move_keys=MOVE_KEYS, exclusive_moves=EXCLUSIVE_MOVES))

    return sources[0] if len(sources) == 1 else CompositeInputSource(sources)


def ramp_to_neutral(controller: RobotController, duration_s: float = 2.0) -> None:
    """Ramp all motors smoothly to neutral position before starting the control loop."""
    motor_ids = list(MOTOR_TO_ID.values())
    initial_positions = np.array(controller.sync_read_present_position(motor_ids))
    target_neutral = np.array([NEUTRAL_POSE[name] for name in MOTOR_TO_ID])

    print("Ramping all motors to neutral position...")
    start_time = time.perf_counter()
    elapsed_time = 0.0

    while elapsed_time < duration_s:
        elapsed_time = time.perf_counter() - start_time
        progress = min(elapsed_time / duration_s, 1.0)
        target_positions = initial_positions + progress * (target_neutral - initial_positions)
        controller.sync_write_goal_position(motor_ids, target_positions.flatten().tolist())

    controller.sync_write_goal_position(motor_ids, target_neutral.tolist())
    print("All motors reached neutral position.")


def link_upper_body(moves: dict) -> None:
    """The upper body goes back to the walks' KP when it stops."""
    upper = moves.get("upper")
    if upper is None:
        return
    for name, move in moves.items():
        if name in ("walk", "skate"):
            upper.peers.append(move)


def main() -> None:
    if _another_session_running():
        print("A control loop is already running (see the PID file); aborting.")
        return

    PID_FILE.write_text(f"{os.getpid()}\n", encoding="ascii")
    display_animation = start_smile_animation()

    controller = RobotController()
    motor_ids = list(MOTOR_TO_ID.values())
    controller.sync_write_torque_enable(motor_ids, [True] * len(motor_ids))
    controller.sync_write_status_return_level(motor_ids, [1] * len(motor_ids))
    controller.sync_write_kp(motor_ids, [KP_DEFAULT] * len(motor_ids))

    try:
        ramp_to_neutral(controller)

        scheduler = Scheduler(
            frequency_hz=50.0,
            controller=controller,
            input_source=build_input_source(),
            moves={
                "head": RotateHeadMove(),
                "walk": WalkMove(controller=controller),
                "skate": SkateMove(controller=controller),
                "gymnastics": DanceMove(f"src/agents/{GYMNASTICS_AGENT}", controller=controller),
                "peek": DanceMove(f"src/agents/{PEEK_AGENT}", controller=controller),
                "propup": PropUpMove(controller=controller),
                **{name: ChillMove(name, controller=controller) for name in CHILL_CLIPS},
                # After the walks: it overwrites their arm targets.
                "upper": UpperBodyMove(controller=controller),
            },
        )
        link_upper_body(scheduler.registered_moves)

        for move in scheduler.registered_moves.values():
            move.preload()

        # Flush stale UART bytes accumulated during preload
        for _ in range(10):
            try:
                controller.sync_read_present_position(motor_ids)
            except RuntimeError:
                pass

        scheduler.run()

    finally:
        for attempt in range(3):
            try:
                controller.sync_write_torque_enable(motor_ids, [False] * len(motor_ids))
                break
            except Exception as exc:  # noqa: BLE001 - best-effort emergency cleanup
                print(f"Warning: torque disable failed ({attempt + 1}/3): {exc}", end="\r\n", flush=True)
                time.sleep(0.05)
        try:
            controller.shutdown()
        except Exception:
            pass
        if display_animation is not None:
            display_animation.stop()
        if PID_FILE.exists():
            PID_FILE.unlink()


if __name__ == "__main__":
    main()
