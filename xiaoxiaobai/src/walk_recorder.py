# SPDX-License-Identifier: GPL-3.0-or-later

"""Per-tick recording of every walk and dance session, saved as one .npz file per session.

The scheduler starts a recording when the walk (or dance) move becomes active and saves it
as <move>_<time>.npz when the move stops, the overcurrent safety trips or the scheduler exits.
Meanwhile the rows are appended every PART_FLUSH_TICKS to <move>_<start time>.part (one JSON
line of metadata, then raw float64 rows; load_recording reads both), deleted once the .npz is
saved: a session cut by a power loss or a crash is still there.
Analyse on the PC with src/debug/analyze_walk_rec.py after `make fetch-walk-rec`, or download them
from the web UI / app (运动控制 → 走路数据记录, api/server.py walk_rec_*).
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import socket
import threading
import time
from pathlib import Path

import numpy as np

from constants import IMU_MOUNT_QUAT, KP_RL, MOTOR_SIGN, MOTOR_TO_ID, NEUTRAL_POSE, OBSERVATION_DOF_ORDER

RECORD_DIR = Path("logs/walk_rec")
JOINTS = list(MOTOR_TO_ID)
PART_FLUSH_TICKS = 250  # 5 s at 50 Hz


class WalkRecorder:
    def __init__(self, out_dir: Path = RECORD_DIR) -> None:
        self.out_dir = out_dir
        self.enabled = os.environ.get("XIAOXIAOBAI_RECORD", "1") != "0"
        self.active = False
        self.move = "walk"
        self._rows: list[np.ndarray] = []
        self._meta: dict = {}
        self._writers: list[threading.Thread] = []
        self._part: Path | None = None
        self._flushed = 0  # rows handed to the .part writer
        self._part_queue: queue.Queue = queue.Queue()
        self._part_writer: threading.Thread | None = None
        n = len(JOINTS)
        # Column layout of one row (float64).
        self.columns = {
            "t": 1, "loop_ms": 1, "policy_ran": 1, "current_proxy_a": 1,
            "cmd": 3, "gyro": 3, "acc": 3, "quat": 4, "projected_gravity": 3,
            "q": n, "dq": n, "target": n, "imu_roll_corr_deg": 1, "imu_pitch_corr_deg": 1,
            "walk_agent": 1, "frame": 1, "mirror_policy": 1,
            "supply_v": 1, "kp_written": 1, "imu_trim_deg": 2,
            "imu_mount_deg": 2, "kp_vin_comp": 1, "imu_auto_trim": 1,
        }
        self._width = sum(self.columns.values())

    def start(self, controller, walk_move, move: str = "walk") -> None:
        if not self.enabled or self.active:
            return
        self.active = True
        self.move = move
        self._rows = []
        agent_path = getattr(walk_move, "agent_path", None)
        self._meta = {
            "move": move,
            "start_wallclock": time.strftime("%Y-%m-%d %H:%M:%S"),
            "host": socket.gethostname(),
            "joints": JOINTS,
            "policy_joints": list(OBSERVATION_DOF_ORDER),
            "columns": self.columns,
            "default_pose": dict(getattr(walk_move, "_default_pose", {})),
            "neutral_pose": dict(NEUTRAL_POSE),
            "neutral_offsets": dict(getattr(controller, "_offsets", {}) or {}),
            "motor_sign": dict(MOTOR_SIGN),
            "motor_to_id": dict(MOTOR_TO_ID),
            "imu_mount_quat": list(IMU_MOUNT_QUAT),
            "kp_rl": KP_RL,
            "agent_path": agent_path,
            "agent_md5": _md5(agent_path),
            "walk_agents": [{"path": p, "md5": _md5(p)} for p in getattr(walk_move, "_agent_paths", [])],
            "mirrorable": list(getattr(walk_move, "_mirrorable", [])),
        }
        self._part = self.out_dir / f"{move}_{time.strftime('%Y%m%d_%H%M%S')}.part"
        self._flushed = 0
        if self._part_writer is None:
            self._part_writer = threading.Thread(target=self._write_parts, name="walk-rec-part", daemon=True)
            self._part_writer.start()
        self._part_queue.put((self._part, json.dumps({**self._meta, "stop_reason": "cut (power loss or crash)"})))
        print(f"{move.capitalize()} recording started", end="\r\n", flush=True)

    def append(self, robot_state, user_input, target_angles: dict[str, float], policy_ran: bool,
               current_proxy_a: float, loop_ms: float, frame: float = np.nan, kp_written: float = np.nan,
               imu_trim_deg: tuple[float, float] = (0.0, 0.0)) -> None:
        if not self.active:
            return
        row = np.full(self._width, np.nan)
        v = user_input.velocity
        parts = [
            [robot_state.time_s], [loop_ms], [float(policy_ran)], [current_proxy_a],
            [v["vx"], v["vy"], v["vtheta"]],
            robot_state.gyro or [np.nan] * 3,
            robot_state.acc or [np.nan] * 3,
            robot_state.quat or [np.nan] * 4,
            robot_state.projected_gravity or [np.nan] * 3,
            [robot_state.motor_positions.get(j, np.nan) for j in JOINTS],
            [robot_state.motor_velocities.get(j, np.nan) for j in JOINTS],
            [target_angles.get(j, np.nan) for j in JOINTS],
            [getattr(user_input, "imu_roll_corr_deg", 0.0)],
            [getattr(user_input, "imu_pitch_corr_deg", 0.0)],
            [getattr(user_input, "walk_agent", 0)],
            [frame],
            [float(getattr(user_input, "mirror_policy", False))],
            [robot_state.supply_voltage if robot_state.supply_voltage is not None else np.nan],
            [kp_written],
            list(imu_trim_deg),
            [getattr(user_input, "imu_mount_roll_deg", 0.0), getattr(user_input, "imu_mount_pitch_deg", 0.0)],
            [float(getattr(user_input, "kp_vin_comp", False))],
            [float(getattr(user_input, "imu_auto_trim", False))],
        ]
        row[:] = np.concatenate([np.asarray(p, dtype=np.float64) for p in parts])
        self._rows.append(row)
        if len(self._rows) - self._flushed >= PART_FLUSH_TICKS:
            self._part_queue.put((self._part, self._rows[self._flushed:]))
            self._flushed = len(self._rows)

    def stop(self, reason: str, background: bool = True) -> None:
        if not self.active:
            return
        self.active = False
        rows, meta = self._rows, {**self._meta, "stop_reason": reason}
        self._rows = []
        part, self._part = self._part, None
        if not rows:
            self._part_queue.put((part, None))
            return
        path = self.out_dir / f"{self.move}_{time.strftime('%Y%m%d_%H%M%S')}.npz"
        if background:
            writer = threading.Thread(target=self._save_then_drop, args=(path, rows, meta, part), daemon=False)
            writer.start()
            self._writers.append(writer)
        else:
            self._save_then_drop(path, rows, meta, part)

    def close(self) -> None:
        self.stop("scheduler exit", background=False)
        for writer in self._writers:
            writer.join()
        for _ in range(500):  # 5 s at most for the writer to drop the saved session's .part
            if not self._part_queue.unfinished_tasks:
                break
            time.sleep(0.01)

    def _save_then_drop(self, path: Path, rows: list[np.ndarray], meta: dict, part: Path | None) -> None:
        _save(path, rows, meta)
        self._part_queue.put((part, None))

    def _write_parts(self) -> None:
        """Writer thread of the .part files: (path, metadata JSON) opens one, (path, rows)
        appends to it, (path, None) deletes it (its .npz is saved). Each write is fsynced:
        on a power loss the page cache would otherwise lose the last ~30 s, the fall."""
        try:
            os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 10)
        except (AttributeError, OSError):
            pass
        while True:
            path, item = self._part_queue.get()
            try:
                if path is None:
                    pass
                elif item is None:
                    path.unlink(missing_ok=True)
                elif isinstance(item, str):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with open(path, "wb") as f:
                        f.write(item.encode("utf-8") + b"\n")
                        f.flush()
                        os.fsync(f.fileno())
                    _fsync_dir(path.parent)
                else:
                    with open(path, "ab") as f:
                        f.write(np.stack(item).tobytes())
                        f.flush()
                        os.fsync(f.fileno())
            except Exception as exc:  # noqa: BLE001 - the recording must never stop the walk
                print(f"Warning: walk recording part {path.name}: {exc}", end="\r\n", flush=True)
            finally:
                self._part_queue.task_done()


def load_recording(path: Path) -> tuple[np.ndarray, dict]:
    """(rows, metadata) of a .npz recording or of a .part left by a cut session."""
    path = Path(path)
    if path.suffix == ".npz":
        npz = np.load(path, allow_pickle=False)
        return npz["data"], json.loads(str(npz["meta"]))
    raw = path.read_bytes()
    head, _, body = raw.partition(b"\n")
    meta = json.loads(head.decode("utf-8"))
    width = sum(meta["columns"].values())
    rows = len(body) // (8 * width)
    return np.frombuffer(body[:rows * 8 * width], dtype=np.float64).reshape(rows, width), meta


def _md5(path: str | None) -> str | None:
    if not path or not Path(path).exists():
        return None
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def _fsync_dir(path: Path) -> None:
    dir_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _save(path: Path, rows: list[np.ndarray], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.stack(rows)
    # fsync before reporting: the robot is often powered off right after a walk, and
    # unsynced page cache then leaves a 0-byte file on the SD card.
    tmp = path.with_suffix(".npz.tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, data=data, meta=json.dumps(meta))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)
    duration = data[-1, 0] - data[0, 0]
    print(f"Recording saved: {path} ({len(rows)} ticks, {duration:.1f} s, {meta['stop_reason']})",
          end="\r\n", flush=True)
