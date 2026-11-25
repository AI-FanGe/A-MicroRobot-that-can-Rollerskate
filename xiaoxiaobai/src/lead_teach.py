# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

"""Lead / follow teleop for matching servo IDs.

Lead (microduck): torque off on the teach joints, stream present positions.
Follow (xiaoxiaobai): torque on those IDs, track the streamed pose.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from rustypot import Xl330PyController

from esp32_link import esp32_device_paths

# Keep this file free of constants.py / numpy so the teach robot only needs rustypot.
DEFAULT_LEAD_IDS = [13, 14, 15, 16, 17, 18]
LEG_IDS = list(range(1, 13))
REVERSE_FOLLOW_IDS = {15, 18}  # elbows: teach and follower turn opposite
KP_DEFAULT = 400
STOP_FLAG = Path("/tmp/xiaoxiaobai_scheduler.stop")
PID_FILE = Path("/tmp/xiaoxiaobai_scheduler.pid")

ID_TO_MOTOR = {
    1: "right_hip_yaw",
    2: "right_hip_roll",
    3: "right_hip_pitch",
    4: "right_knee",
    5: "right_ankle_pitch",
    6: "right_ankle_roll",
    7: "left_hip_yaw",
    8: "left_hip_roll",
    9: "left_hip_pitch",
    10: "left_knee",
    11: "left_ankle_pitch",
    12: "left_ankle_roll",
    13: "right_shoulder_pitch",
    14: "right_shoulder_roll",
    15: "right_elbow",
    16: "left_shoulder_pitch",
    17: "left_shoulder_roll",
    18: "left_elbow",
    19: "head",
    20: "head_pitch",
}

MOTOR_SIGN = {
    "left_hip_yaw": 1.0,
    "left_hip_roll": 1.0,
    "left_hip_pitch": -1.0,
    "left_knee": 1.0,
    "left_ankle_pitch": 1.0,  # motor 11
    "left_ankle_roll": -1.0,  # motor 12
    "right_hip_yaw": 1.0,
    "right_hip_roll": 1.0,
    "right_hip_pitch": -1.0,
    "right_knee": 1.0,
    "right_ankle_pitch": 1.0,  # motor 5
    "right_ankle_roll": -1.0,  # motor 6
    "left_shoulder_pitch": 1.0,
    "left_shoulder_roll": 1.0,
    "left_elbow": -1.0,
    "right_shoulder_pitch": 1.0,
    "right_shoulder_roll": 1.0,
    "right_elbow": -1.0,
    "head": 1.0,
    "head_pitch": 1.0,
}


def list_serial_ports() -> list[str]:
    ports: list[str] = []
    override = os.environ.get("XIAOXIAOBAI_SERIAL_PORT")
    if override:
        ports.append(override)
    by_id = Path("/dev/serial/by-id")
    if by_id.is_dir():
        ports.extend(str(path) for path in sorted(by_id.glob("*OpenRB*")))
        ports.extend(str(path) for path in sorted(by_id.glob("*ROBOTIS*")))
        ports.extend(str(path) for path in sorted(by_id.iterdir()) if path.is_symlink() or path.exists())
    for name in (
        "/dev/ttyACM0",
        "/dev/ttyACM1",
        "/dev/ttyACM2",
        "/dev/ttyUSB0",
        "/dev/ttyUSB1",
        "/dev/ttyAMA0",
    ):
        if Path(name).exists():
            ports.append(name)
    unique: list[str] = []
    seen: set[str] = esp32_device_paths()
    for port in ports:
        try:
            resolved = str(Path(port).resolve())
        except OSError:
            resolved = port
        if resolved in seen or not Path(port).exists():
            continue
        seen.add(resolved)
        unique.append(port)
    return unique


def resolve_serial_port() -> str:
    ports = list_serial_ports()
    if ports:
        return ports[0]
    raise FileNotFoundError("未找到舵机控制板 / 电机串口")


def motor_sign(motor_id: int) -> float:
    name = ID_TO_MOTOR.get(motor_id)
    if name is None:
        return 1.0
    return float(MOTOR_SIGN.get(name, 1.0))


def motor_name(motor_id: int) -> str:
    return ID_TO_MOTOR.get(motor_id, f"id{motor_id}")


def parse_ids(text: str) -> list[int]:
    ids: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            ids.extend(range(int(start), int(end) + 1))
        else:
            ids.append(int(part))
    return list(dict.fromkeys(ids))


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _wait_dead(pid: int, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _pid_alive(pid):
            return True
        time.sleep(0.05)
    return not _pid_alive(pid)


def request_scheduler_stop() -> None:
    try:
        STOP_FLAG.write_text("stop\n", encoding="ascii")
    except OSError:
        pass
    if not PID_FILE.exists():
        return
    try:
        pid = int(PID_FILE.read_text(encoding="ascii").strip())
    except (ValueError, OSError):
        return
    if not _pid_alive(pid):
        return
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            os.kill(pid, sig)
        except OSError:
            return
        if _wait_dead(pid, 2.0):
            return


def _kill_matching(pattern: str, mine: int) -> None:
    try:
        raw = subprocess.check_output(["pgrep", "-f", pattern], text=True)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return
    for token in raw.split():
        try:
            pid = int(token)
        except ValueError:
            continue
        if pid == mine:
            continue
        try:
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", "replace")
        except OSError:
            cmdline = ""
        if "python" not in cmdline:
            continue
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            continue
        _wait_dead(pid, 1.5)


def reclaim_serial_bus() -> None:
    """Stop leftover Python jobs that keep the servo-board port."""
    mine = os.getpid()
    _kill_matching("src/lead_teach.py", mine)
    _kill_matching("src/upper_body_slider.py", mine)


def stop_bus_users() -> None:
    request_scheduler_stop()
    reclaim_serial_bus()
    _kill_matching("src/main.py", os.getpid())
    time.sleep(0.6)


def probe_ids(controller: Xl330PyController, ids: list[int]) -> list[int]:
    alive: list[int] = []
    for motor_id in ids:
        try:
            controller.sync_read_present_position([motor_id])
            alive.append(motor_id)
            continue
        except Exception:
            pass
        try:
            controller.sync_write_status_return_level([motor_id], [1])
            controller.sync_read_present_position([motor_id])
            alive.append(motor_id)
        except Exception:
            continue
    return alive


def _open_controller(port: str) -> Xl330PyController:
    return Xl330PyController(serial_port=port, baudrate=1_000_000, timeout=0.25)


def find_lead_motors(ids: list[int]) -> tuple[Xl330PyController, str, list[int], list[int], list[str]]:
    tried: list[str] = []
    last_found: list[int] = []
    last_port = ""
    last_controller: Xl330PyController | None = None
    for port in list_serial_ports():
        tried.append(port)
        try:
            controller = _open_controller(port)
        except Exception:
            continue
        wanted = probe_ids(controller, ids)
        scanned = wanted if wanted else probe_ids(controller, list(range(1, 21)))
        if wanted or scanned:
            return controller, port, wanted, scanned, tried
        last_found = scanned
        last_port = port
        last_controller = controller
    if last_controller is None:
        raise FileNotFoundError("未找到可用串口: " + (", ".join(tried) or "(none)"))
    return last_controller, last_port, [], last_found, tried


def read_positions(controller: Xl330PyController, ids: list[int]) -> list[float]:
    raw = controller.sync_read_present_position(ids)
    return [float(value) * motor_sign(motor_id) for motor_id, value in zip(ids, raw)]


def write_positions(controller: Xl330PyController, ids: list[int], positions: list[float]) -> None:
    hardware = [pos * motor_sign(motor_id) for motor_id, pos in zip(ids, positions)]
    controller.sync_write_goal_position(ids, hardware)


def run_lead(ids: list[int], hz: float) -> None:
    stop_bus_users()
    try:
        controller, port, alive, found, tried = find_lead_motors(ids)
    except Exception as exc:
        print(json.dumps({"error": f"lead 打不开串口: {exc}"}), flush=True)
        raise SystemExit(2) from exc
    if not alive:
        looks_like_legs = any(motor_id in found for motor_id in range(7, 13))
        if not looks_like_legs:
            for group in (
                [1, 2, 3, 4, 5, 6],
                [31, 32, 33, 41, 42, 43],
            ):
                if all(motor_id in found for motor_id in group):
                    alive = group
                    break
    if not alive:
        print(
            json.dumps(
                {
                    "error": "lead 上没有可用电机（示教机串口被占用、没上电，或手臂 ID 不是 13-18）",
                    "port": port,
                    "wanted": ids,
                    "found_ids": found,
                    "tried_ports": tried,
                }
            ),
            flush=True,
        )
        raise SystemExit(2)
    controller.sync_write_torque_enable(alive, [False] * len(alive))
    print(
        json.dumps(
            {
                "ready": "lead",
                "ids": alive,
                "names": [motor_name(i) for i in alive],
                "port": port,
            }
        ),
        flush=True,
    )
    period = 1.0 / max(hz, 1.0)
    try:
        while True:
            started = time.perf_counter()
            positions = read_positions(controller, alive)
            print(json.dumps({"ids": alive, "pos": positions, "t": time.time()}), flush=True)
            leftover = period - (time.perf_counter() - started)
            if leftover > 0:
                time.sleep(leftover)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            controller.sync_write_torque_enable(alive, [False] * len(alive))
        except Exception:
            pass


def run_follow(ids: list[int], ramp_s: float) -> None:
    request_scheduler_stop()
    reclaim_serial_bus()
    port = resolve_serial_port()
    controller = Xl330PyController(serial_port=port, baudrate=1_000_000, timeout=0.08)
    alive = probe_ids(controller, ids)
    if not alive:
        print(f"follow 上没有可用电机 port={port}", flush=True)
        raise SystemExit(2)
    controller.sync_write_status_return_level(alive, [1] * len(alive))
    controller.sync_write_position_p_gain(alive, [KP_DEFAULT] * len(alive))
    start_pose = read_positions(controller, alive)
    write_positions(controller, alive, start_pose)
    controller.sync_write_torque_enable(alive, [True] * len(alive))

    held_legs = probe_ids(controller, LEG_IDS)
    if held_legs:
        controller.sync_write_status_return_level(held_legs, [1] * len(held_legs))
        controller.sync_write_position_p_gain(held_legs, [KP_DEFAULT] * len(held_legs))
        write_positions(controller, held_legs, read_positions(controller, held_legs))
        controller.sync_write_torque_enable(held_legs, [True] * len(held_legs))
    print(f"follow ready ids={alive} hold_legs={held_legs} port={port}", flush=True)

    ramp_begin: float | None = None
    start_map = dict(zip(alive, start_pose))
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("error"):
                print(payload["error"], flush=True)
                break
            incoming_ids = [int(v) for v in payload.get("ids") or []]
            incoming_pos = [float(v) for v in payload.get("pos") or []]
            if len(incoming_ids) != len(incoming_pos):
                continue
            lead = {motor_id: pos for motor_id, pos in zip(incoming_ids, incoming_pos) if motor_id in alive}
            if not lead:
                continue
            ordered_ids = [motor_id for motor_id in alive if motor_id in lead]
            if ramp_begin is None:
                ramp_begin = time.perf_counter()
            alpha = 1.0 if ramp_s <= 0 else min(1.0, (time.perf_counter() - ramp_begin) / ramp_s)
            goals: list[float] = []
            for motor_id in ordered_ids:
                # Same ID, same angle. Ramp from the follower's current pose first.
                target = -lead[motor_id] if motor_id in REVERSE_FOLLOW_IDS else lead[motor_id]
                if alpha < 1.0:
                    target = start_map[motor_id] + (target - start_map[motor_id]) * alpha
                goals.append(target)
            write_positions(controller, ordered_ids, goals)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            controller.sync_write_torque_enable(alive, [False] * len(alive))
        except Exception:
            pass
        print("follow stopped, torque off", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lead/follow 示教：按电机 ID 同步手臂")
    parser.add_argument("role", choices=("lead", "follow"))
    parser.add_argument("--ids", default="13-18", help="电机 ID，例如 13-18 或 13,14,15,16,17,18")
    parser.add_argument("--hz", type=float, default=50.0)
    parser.add_argument("--ramp", type=float, default=1.2)
    args = parser.parse_args()
    ids = parse_ids(args.ids)
    allowed = set(DEFAULT_LEAD_IDS)
    invalid = [motor_id for motor_id in ids if motor_id not in allowed]
    if invalid:
        parser.error(f"只支持 13-18 电机，收到：{','.join(str(motor_id) for motor_id in invalid)}")
    if not ids:
        parser.error("至少指定一个 13-18 的电机 ID")
    if args.role == "lead":
        run_lead(ids, args.hz)
    else:
        run_follow(ids, args.ramp)


if __name__ == "__main__":
    main()
