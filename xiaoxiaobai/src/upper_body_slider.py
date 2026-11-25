#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

"""File-backed upper-body motor slider controller for the web console."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

from rustypot import Xl330PyController

from lead_teach import (
    DEFAULT_LEAD_IDS,
    KP_DEFAULT,
    motor_name,
    parse_ids,
    probe_ids,
    read_positions,
    reclaim_serial_bus,
    request_scheduler_stop,
    resolve_serial_port,
    write_positions,
)

TARGET_PATH = Path("/tmp/xiaoxiaobai_upper_body_slider.json")
STATE_PATH = Path("/tmp/xiaoxiaobai_upper_body_slider_state.json")
PID_FILE = Path("/tmp/xiaoxiaobai_upper_body_slider.pid")
STOP_FLAG = Path("/tmp/xiaoxiaobai_upper_body_slider.stop")
DEFAULT_UDP_PORT = 8765
# Travel per motor, motor angle [deg]: gestures.JOINT_LIMITS_DEG times MOTOR_SIGN (the left
# roll and both elbows are mirrored by their sign).
MOTOR_LIMITS_DEG = {
    13: (-150.0, 150.0), 14: (0.0, 160.0), 15: (0.0, 85.0),
    16: (-150.0, 150.0), 17: (-160.0, 0.0), 18: (-85.0, 0.0),
    19: (-40.0, 40.0), 20: (-10.0, 30.0),
}
MOTOR_LIMITS_RAD = {i: (math.radians(lo), math.radians(hi)) for i, (lo, hi) in MOTOR_LIMITS_DEG.items()}
# Slider targets arrive in steps (one per UI message): the motors follow them through a
# critically damped filter, so a drag comes out as one continuous move.
SMOOTH_TIME_S = 0.12
MAX_SPEED_RAD_S = math.radians(300.0)


class Follower:
    """Critically damped approach to a moving target, speed capped (SmoothDamp)."""

    def __init__(self, positions: dict[int, float]) -> None:
        self.pos = dict(positions)
        self.vel = {motor_id: 0.0 for motor_id in positions}

    def step(self, targets: dict[int, float], dt: float) -> dict[int, float]:
        omega = 2.0 / SMOOTH_TIME_S
        x = omega * dt
        decay = 1.0 / (1.0 + x + 0.48 * x * x + 0.235 * x * x * x)
        for motor_id, target in targets.items():
            pos, vel = self.pos.get(motor_id, target), self.vel.get(motor_id, 0.0)
            change = pos - target
            reach = MAX_SPEED_RAD_S * SMOOTH_TIME_S
            change = max(-reach, min(reach, change))
            goal = pos - change
            temp = (vel + omega * change) * dt
            vel = (vel - omega * temp) * decay
            new = goal + (change + temp) * decay
            if (target - pos > 0.0) == (new > target):
                new, vel = target, 0.0
            self.pos[motor_id], self.vel[motor_id] = new, vel
        return dict(self.pos)


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


def stop_existing() -> None:
    mine = os.getpid()
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text(encoding="ascii").strip())
        except (ValueError, OSError):
            pid = 0
        if pid and pid != mine and _pid_alive(pid):
            STOP_FLAG.write_text("stop\n", encoding="ascii")
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
            _wait_dead(pid, 2.0)

    try:
        raw = subprocess.check_output(["pgrep", "-f", "src/upper_body_slider.py"], text=True)
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
            os.kill(pid, signal.SIGTERM)
        except OSError:
            continue
        _wait_dead(pid, 1.5)


def _targets_from_payload(
    data: object,
    alive: list[int],
    fallback: dict[int, float],
    limit_rad: float,
    *,
    reject_stale: bool,
) -> dict[int, float]:
    raw_targets = data.get("targets") if isinstance(data, dict) else None
    if not isinstance(raw_targets, dict):
        return fallback
    if reject_stale:
        try:
            age = time.time() - float(data.get("ts", 0.0))
        except (TypeError, ValueError):
            return fallback
        if age > 2.0:
            return fallback
    targets = dict(fallback)
    for motor_id in alive:
        raw = raw_targets.get(str(motor_id))
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            lo, hi = MOTOR_LIMITS_RAD.get(motor_id, (-limit_rad, limit_rad))
            targets[motor_id] = max(max(lo, -limit_rad), min(min(hi, limit_rad), value))
    return targets


def read_file_targets(alive: list[int], fallback: dict[int, float], limit_rad: float) -> dict[int, float]:
    try:
        data = json.loads(TARGET_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback
    return _targets_from_payload(data, alive, fallback, limit_rad, reject_stale=True)


def read_udp_targets(
    sock: socket.socket,
    alive: list[int],
    fallback: dict[int, float],
    limit_rad: float,
) -> tuple[dict[int, float], set[int] | None]:
    """Targets from the queued packets, and the motors to leave limp if a packet said so
    ({"limp": [ids]}: torque off on those, to be moved by hand; [] holds them all again)."""
    targets = fallback
    limp: set[int] | None = None
    while True:
        try:
            packet, _addr = sock.recvfrom(4096)
        except BlockingIOError:
            return targets, limp
        except OSError:
            return targets, limp
        try:
            data = json.loads(packet.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and isinstance(data.get("limp"), list):
            limp = {int(i) for i in data["limp"] if str(i).isdigit() and int(i) in alive}
        targets = _targets_from_payload(data, alive, targets, limit_rad, reject_stale=False)


def write_state(alive: list[int], positions: dict[int, float], running: bool = True, error: str | None = None,
                limp: set[int] | None = None) -> None:
    payload = {
        "running": running,
        "ids": alive,
        "names": {str(motor_id): motor_name(motor_id) for motor_id in alive},
        "positions": {str(motor_id): positions[motor_id] for motor_id in alive if motor_id in positions},
        "limp": sorted(limp or ()),
        "error": error,
        "ts": time.time(),
    }
    STATE_PATH.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


def open_udp_socket(port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", port))
    sock.setblocking(False)
    return sock


def run(ids: list[int], hz: float, limit_deg: float, udp_port: int) -> None:
    request_scheduler_stop()
    reclaim_serial_bus()
    stop_existing()
    STOP_FLAG.unlink(missing_ok=True)
    PID_FILE.write_text(f"{os.getpid()}\n", encoding="ascii")

    limit_rad = math.radians(abs(limit_deg))
    udp_sock = open_udp_socket(udp_port)
    port = resolve_serial_port()
    controller = Xl330PyController(serial_port=port, baudrate=1_000_000, timeout=0.08)
    alive = probe_ids(controller, ids)
    if not alive:
        write_state([], {}, running=False, error=f"没有可用的上半身电机 port={port}")
        raise SystemExit(2)

    controller.sync_write_status_return_level(alive, [1] * len(alive))
    controller.sync_write_position_p_gain(alive, [KP_DEFAULT] * len(alive))
    current = dict(zip(alive, read_positions(controller, alive)))
    targets = read_file_targets(alive, current, limit_rad)
    write_positions(controller, alive, [targets[motor_id] for motor_id in alive])
    controller.sync_write_torque_enable(alive, [True] * len(alive))
    write_state(alive, targets)

    period = 1.0 / max(1.0, hz)
    follower = Follower(targets)
    limp: set[int] = set()
    stopping = False

    def clamp(motor_id: int, value: float) -> float:
        lo, hi = MOTOR_LIMITS_RAD.get(motor_id, (-limit_rad, limit_rad))
        return max(lo, min(hi, value))

    def hold_where_they_are(ids: list[int], positions: list[float] | None = None) -> None:
        for motor_id, pos in zip(ids, positions if positions is not None else read_positions(controller, ids)):
            targets[motor_id] = follower.pos[motor_id] = clamp(motor_id, pos)
            follower.vel[motor_id] = 0.0

    def request_stop(*_args: object) -> None:
        nonlocal stopping
        stopping = True
        try:
            STOP_FLAG.write_text("stop\n", encoding="ascii")
        except OSError:
            pass

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    try:
        while not stopping and not STOP_FLAG.exists():
            started = time.perf_counter()
            targets, want_limp = read_udp_targets(udp_sock, alive, targets, limit_rad)
            if want_limp is not None and want_limp != limp:
                released, held = sorted(want_limp - limp), sorted(limp - want_limp)
                if released:
                    controller.sync_write_torque_enable(released, [False] * len(released))
                if held:
                    hold_where_they_are(held)
                    write_positions(controller, held, [targets[motor_id] for motor_id in held])
                    controller.sync_write_torque_enable(held, [True] * len(held))
                limp = want_limp
            measured: dict[int, float] = {}
            if limp:
                ids = sorted(limp)
                positions = read_positions(controller, ids)
                measured = dict(zip(ids, positions))
                hold_where_they_are(ids, positions)
            commanded = follower.step(targets, period)
            write_positions(controller, alive, [commanded[motor_id] for motor_id in alive])
            write_state(alive, {**commanded, **measured}, limp=limp)
            leftover = period - (time.perf_counter() - started)
            if leftover > 0:
                time.sleep(leftover)
    finally:
        try:
            controller.sync_write_torque_enable(alive, [False] * len(alive))
        except Exception:
            pass
        write_state(alive, targets, running=False)
        udp_sock.close()
        STOP_FLAG.unlink(missing_ok=True)
        if PID_FILE.exists():
            PID_FILE.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Xiaoxiaobai 上半身 slider 控制")
    parser.add_argument("--ids", default="13-20", help="电机 ID，例如 13-20")
    parser.add_argument("--hz", type=float, default=50.0)
    parser.add_argument("--limit-deg", type=float, default=160.0)
    parser.add_argument("--udp-port", type=int, default=DEFAULT_UDP_PORT)
    args = parser.parse_args()
    ids = parse_ids(args.ids) or list(DEFAULT_LEAD_IDS)
    run(ids, args.hz, args.limit_deg, args.udp_port)


if __name__ == "__main__":
    main()
