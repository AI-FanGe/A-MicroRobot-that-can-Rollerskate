# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

import json
import os
import time
import math
from collections import deque
from pathlib import Path
from typing import Optional


from constants import (
    MOTOR_TO_ID,
    OBSERVATION_DOF_ORDER,
    BAM_MAX_CURRENT,
    OVERCURRENT_CUTOFF_A,
    OVERCURRENT_DEBOUNCE_TICKS,
    OVERCURRENT_PROXY_DELAY_TICKS,
    PROXY_KT,
    PROXY_R,
    PROXY_VIN,
    PROXY_ERROR_GAIN,
    PROXY_MAX_PWM,
    PROXY_KP,
    HEAD_LIMITS_DEG,
)
from controller import ControllerProtocol
from imu_reader import apply_mount_correction, imu_quat_to_body
from observer import Observer, Observation
from input.input_source import InputSource, UserInput, scale_velocity
from moves.move import MotorCommand, Move, MoveState
from moves.rotate_head import RotateHeadMove
from moves.walk import WalkMove
from imu_calib import ImuCalib
from tilt_trim import TiltTrim, load_posture_ref
from walk_recorder import WalkRecorder

# Read by the robot API (src/api): what the loop is doing, a few times per second.
STATUS_PATH = Path("/tmp/xiaoxiaobai_status.json")
STATUS_PERIOD_S = 0.2
RECORDED_MOVES = ("walk", "gymnastics", "peek", "propup", "chill_lie", "chill_sit", "chill_chair",
                  "wake_stretch")
# One motor's input voltage is read this often (a single bus transaction) for the battery gauge
# and the walk's KP compensation, which follows it smoothed (one read sags with the step load).
VOLTAGE_PERIOD_S = 5.0
VOLTAGE_MOTOR_ID = MOTOR_TO_ID["head"]
VOLTAGE_SMOOTHING = 0.3  # weight of a new read: ~15 s time constant
VOLTAGE_VALID_V = (4.0, 10.0)
HEAD_LIMITS_RAD = {name: (math.radians(lo), math.radians(hi)) for name, (lo, hi) in HEAD_LIMITS_DEG.items()}


class Scheduler:
    def __init__(
        self,
        frequency_hz: float = 50.0,
        controller: ControllerProtocol = None,
        stop_flag_path: str = "/tmp/xiaoxiaobai_scheduler.stop",
        input_source: Optional[InputSource] = None,
        moves: Optional[dict[str, Move]] = None,
    ):
        self.dt = 1.0 / frequency_hz
        self.controller = controller
        self.stop_flag_path = Path(stop_flag_path)
        self._cleanup_done = False
        self.input_source = input_source

        self.observer = Observer(self.controller)

        # All moves are registered here. They only run when activated via user_input.active_moves.
        self.registered_moves: dict[str, Move] = moves if moves is not None else {
            "head": RotateHeadMove(),
            "walk": WalkMove(),
        }

        self.loop_start_time = time.perf_counter()
        self._serial_errors = 0
        self._last_imu_print_s: float = 0.0
        self._last_imu_stale_warn_s: float = 0.0
        self._overcurrent_ticks = 0
        # History of sent target_angles, to align the current proxy with the delayed feedback:
        # the oldest entry is the command issued OVERCURRENT_PROXY_DELAY_TICKS ticks ago.
        self._cmd_history: deque[dict[str, float]] = deque(maxlen=OVERCURRENT_PROXY_DELAY_TICKS + 1)
        self._last_total_current = 0.0
        self.recorder = WalkRecorder()
        self._stop_reason = "stopped"
        self._next_status_s = 0.0
        self._status_error_logged = False
        self._next_voltage_s = 0.0
        self._voltage_v: float | None = None
        self._supply_v: float | None = None
        self._posture_ref = load_posture_ref()
        self.tilt_trim: TiltTrim | None = TiltTrim(self._posture_ref)
        self.imu_calib = ImuCalib(self._posture_ref)
        self._trim_deg = (0.0, 0.0)
        self._overruns = 0
        self._last_loop_ms = 0.0

    def run(self):
        print(f"Starting control loop at {1 / self.dt:.1f} Hz", end="\r\n", flush=True)
        if self.stop_flag_path.exists():
            self.stop_flag_path.unlink()

        if self.input_source:
            self.input_source.start()

        try:
            while True:
                if self.stop_flag_path.exists():
                    print("Stop requested through stop flag", end="\r\n", flush=True)
                    self._stop_reason = "stop_flag"
                    break

                start_time = time.perf_counter()

                # Read robot observations and user input
                try:
                    robot_state = self.observer.read_state(self.dt)
                except RuntimeError as e:
                    self._serial_errors += 1
                    if self._serial_errors >= 3:
                        print(f"Serial communication error: {e}", end="\r\n", flush=True)
                        self._stop_reason = "serial_error"
                        break
                    print(f"Warning: serial read error (attempt {self._serial_errors}/3): {e}", end="\r\n", flush=True)
                    continue
                self._serial_errors = 0

                robot_state.time_s = start_time - self.loop_start_time
                robot_state.supply_voltage = self._supply_v
                user_input = self.input_source.read() if self.input_source else UserInput()
                user_input.velocity = scale_velocity(user_input.velocity)
                if robot_state.projected_gravity:
                    raw = robot_state.projected_gravity
                    standing = all(move.state == MoveState.INACTIVE
                                   for name, move in self.registered_moves.items() if name != "head")
                    self.imu_calib.update(robot_state.time_s, user_input.imu_calib_seq, raw, robot_state.gyro,
                                          robot_state.motor_positions, standing)
                    mount = (user_input.imu_mount_roll_deg, user_input.imu_mount_pitch_deg)
                    self._trim_deg = self._update_tilt_trim(apply_mount_correction(raw, *mount), robot_state.time_s,
                                                            user_input)
                    robot_state.projected_gravity = apply_mount_correction(
                        raw, mount[0] + user_input.imu_roll_corr_deg + self._trim_deg[0],
                        mount[1] + user_input.imu_pitch_corr_deg + self._trim_deg[1])
                obs = Observation(robot_state=robot_state, user_input=user_input)

                imu_status_getter = getattr(self.controller, "get_imu_status", None)
                if callable(imu_status_getter):
                    imu_status = imu_status_getter()
                    age_s = float(imu_status.get("age_s", 0.0))
                    if age_s > self.dt and (start_time - self._last_imu_stale_warn_s) >= 1.0:
                        print(f"Warning: stale IMU data ({age_s * 1000.0:.1f} ms old)", end="\r\n", flush=True)
                        self._last_imu_stale_warn_s = start_time

                # Update move states and dispatch one call per move per tick. Stopping moves go
                # first, so a move handing over to another (walk <-> skate) cannot overwrite the
                # KP the starting move sets.
                command = MotorCommand()
                for name, move in self.registered_moves.items():
                    in_active = name in obs.user_input.active_moves

                    if in_active and move.state == MoveState.INACTIVE:
                        move.state = MoveState.STARTING
                    elif not in_active and move.state in (MoveState.STARTING, MoveState.ACTIVE):
                        move.state = MoveState.STOPPING

                moves = sorted(self.registered_moves.values(), key=lambda m: m.state != MoveState.STOPPING)
                for move in moves:
                    if move.state == MoveState.STARTING:
                        move.on_start(obs, command)
                    elif move.state == MoveState.ACTIVE:
                        move.step(obs, command)
                    elif move.state == MoveState.STOPPING:
                        move.on_stop(obs, command)

                # Overcurrent safety: estimate the current from the command that was actually
                # active when the (delayed) position/velocity feedback was sampled — the oldest
                # buffered command (== DELAY_TICKS old once full). This reconstructs the real
                # (delayed) current instead of pairing a fresh target with stale feedback, which
                # would inflate the error term and false-trigger at gait start.
                aligned_targets = self._cmd_history[0] if self._cmd_history else command.target_angles
                overcurrent = self._check_overcurrent(robot_state, aligned_targets)
                self._record_tick(obs, command, start_time)
                if overcurrent:
                    self.recorder.stop("overcurrent")
                    self._stop_reason = "overcurrent"
                    break

                # Send command to motors
                self._cmd_history.append(dict(command.target_angles))
                self._send_to_motors(command)

                if start_time >= self._next_voltage_s:
                    self._next_voltage_s = start_time + VOLTAGE_PERIOD_S
                    self._read_supply_voltage()
                if start_time >= self._next_status_s:
                    self._next_status_s = start_time + STATUS_PERIOD_S
                    try:
                        self._publish_status(obs, start_time, command)
                    except Exception as exc:  # noqa: BLE001 - status is for the web UI only
                        if not self._status_error_logged:
                            self._status_error_logged = True
                            print(f"Warning: status publish failed: {exc!r}", end="\r\n", flush=True)

                # IMU / gyro terminal display
                if obs.user_input.show_imu and (start_time - self._last_imu_print_s) >= 0.5:
                    acc = obs.robot_state.acc
                    gyro = obs.robot_state.gyro
                    quat = obs.robot_state.quat
                    print("--------------------------------------------", end="\r\n", flush=True)
                    if gyro:
                        gx, gy, gz = gyro
                        print(f"Gyro: gx={gx:+.3f}  gy={gy:+.3f}  gz={gz:+.3f} rad/s", end="\r\n", flush=True)
                    if acc:
                        ax, ay, az = acc
                        print(f"Acc:  ax={ax:+.3f}  ay={ay:+.3f}  az={az:+.3f} g", end="\r\n", flush=True)
                    if quat:
                        w, x, y, z = quat
                        roll = math.degrees(math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
                        pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))))
                        yaw = math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
                        print(f"IMU:  roll={roll:+.1f}°  pitch={pitch:+.1f}°  yaw={yaw:+.1f}°", end="\r\n", flush=True)
                        bw, bx, by, bz = imu_quat_to_body((w, x, y, z))
                        b_roll = math.degrees(math.atan2(2 * (bw * bx + by * bz), 1 - 2 * (bx * bx + by * by)))
                        b_pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2 * (bw * by - bz * bx)))))
                        b_yaw = math.degrees(math.atan2(2 * (bw * bz + bx * by), 1 - 2 * (by * by + bz * bz)))
                        print(f"Body: roll={b_roll:+.1f}°  pitch={b_pitch:+.1f}°  yaw={b_yaw:+.1f}°", end="\r\n", flush=True)
                    pg = obs.robot_state.projected_gravity
                    if pg:
                        lean_r = math.degrees(math.asin(max(-1.0, min(1.0, -pg[1]))))
                        lean_f = math.degrees(math.asin(max(-1.0, min(1.0, pg[0]))))
                        u = obs.user_input
                        print(f"Policy sees lean_right={lean_r:+.1f}° lean_fwd={lean_f:+.1f}° (IMU correction roll "
                              f"{u.imu_mount_roll_deg:+.1f}{u.imu_roll_corr_deg:+.1f}{self._trim_deg[0]:+.1f}°, pitch "
                              f"{u.imu_mount_pitch_deg:+.1f}{u.imu_pitch_corr_deg:+.1f}{self._trim_deg[1]:+.1f}° "
                              "mount+manual+auto)", end="\r\n", flush=True)
                    self._last_imu_print_s = start_time

                # Sleep to keep a fixed control frequency
                elapsed_time = time.perf_counter() - start_time
                sleep_time = self.dt - elapsed_time
                self._last_loop_ms = elapsed_time * 1000.0

                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    self._overruns += 1
                    ms_late = -sleep_time * 1000
                    print(f"Warning: control loop overrun by {ms_late:.2f} ms", end="\r\n", flush=True)

        except KeyboardInterrupt:
            print("Control loop interrupted by user", end="\r\n", flush=True)
            self._stop_reason = "interrupted"
        except Exception:
            self._stop_reason = "error"
            raise
        finally:
            self._cleanup()

    def _cleanup(self) -> None:
        """Disable torque, stop input source, and clear stop artifacts."""
        if self._cleanup_done:
            return

        self._cleanup_done = True

        try:
            self.recorder.close()
        except Exception as exc:  # noqa: BLE001 - cleanup should continue
            print(f"Warning: saving walk recording failed: {exc}", end="\r\n", flush=True)

        if self.input_source:
            self.input_source.stop()

        motor_ids = list(MOTOR_TO_ID.values())
        for attempt in range(3):
            try:
                self.controller.sync_write_torque_enable(motor_ids, [False] * len(motor_ids))
                print("Torque disabled on all motors", end="\r\n", flush=True)
                break
            except Exception as exc:  # noqa: BLE001 - cleanup should continue
                print(f"Warning: torque disable failed ({attempt + 1}/3): {exc}", end="\r\n", flush=True)
                time.sleep(0.05)

        shutdown = getattr(self.controller, "shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception:
                pass

        if self.stop_flag_path.exists():
            self.stop_flag_path.unlink()

        self._write_status({"ts": time.time(), "pid": os.getpid(), "running": False, "stop_reason": self._stop_reason})

    def _publish_status(self, obs: Observation, now_s: float, command: MotorCommand) -> None:
        """Snapshot for the robot API; never allowed to stop the loop."""
        state = obs.robot_state
        targets = command.target_angles
        # name: [position rad, target rad, velocity rad/s, current A, voltage V]; None = not read.
        joints = {}
        for name, pos in state.motor_positions.items():
            vel = state.motor_velocities.get(name)
            cur = state.motor_currents.get(name)
            vol = state.motor_voltages.get(name)
            # Values may be numpy scalars (policy outputs), which json cannot encode.
            joints[name] = [
                round(float(pos), 4),
                round(float(targets[name]), 4) if name in targets else None,
                round(float(vel), 3) if vel is not None else None,
                round(float(cur), 3) if cur is not None else None,
                round(float(vol), 2) if vol is not None else None,
            ]
        walk = self.registered_moves.get("walk")
        roll_deg = pitch_deg = None
        pg = obs.robot_state.projected_gravity
        if pg:
            roll_deg = round(math.degrees(math.asin(max(-1.0, min(1.0, -pg[1])))), 1)
            pitch_deg = round(math.degrees(math.asin(max(-1.0, min(1.0, pg[0])))), 1)
        self._write_status(
            {
                "ts": time.time(),
                "pid": os.getpid(),
                "running": True,
                "moves": {name: move.state.name.lower() for name, move in self.registered_moves.items()},
                "active_moves": sorted(obs.user_input.active_moves),
                "velocity": {k: round(v, 3) for k, v in obs.user_input.velocity.items()},
                "current_a": round(self._last_total_current, 2),
                "voltage_v": self._voltage_v,
                "supply_v": round(self._supply_v, 2) if self._supply_v is not None else None,
                "kp_written": getattr(walk, "kp_written", None),
                "imu_trim_deg": [round(self._trim_deg[0], 2), round(self._trim_deg[1], 2)],
                "imu_trim_given_up": self.tilt_trim.given_up if self.tilt_trim is not None else [],
                "imu_mount_deg": [round(obs.user_input.imu_mount_roll_deg, 2),
                                  round(obs.user_input.imu_mount_pitch_deg, 2)],
                "imu_calib": self.imu_calib.result,
                "imu_calib_running": self.imu_calib.running,
                "roll_deg": roll_deg,
                "pitch_deg": pitch_deg,
                "loop_ms": round(self._last_loop_ms, 2),
                "overruns": self._overruns,
                "joints": joints,
                "gyro": [round(float(g), 4) for g in state.gyro] if state.gyro else None,
                "quat": [round(float(q), 5) for q in state.quat] if state.quat else None,
                **{
                    name: move.status()
                    for name, move in self.registered_moves.items()
                    if move.state != MoveState.INACTIVE and callable(getattr(move, "status", None))
                },
            }
        )

    def _read_supply_voltage(self) -> None:
        reader = getattr(self.controller, "read_present_input_voltage", None)
        if not callable(reader):
            return
        try:
            volts = float(reader(VOLTAGE_MOTOR_ID)) * 0.1
        except Exception:  # noqa: BLE001 - the battery gauge is optional
            return
        if not VOLTAGE_VALID_V[0] <= volts <= VOLTAGE_VALID_V[1]:
            return
        self._voltage_v = round(volts, 2)
        prev = self._supply_v
        self._supply_v = volts if prev is None else prev + VOLTAGE_SMOOTHING * (volts - prev)

    def _update_tilt_trim(self, gravity: list[float], now_s: float, user_input: UserInput) -> tuple[float, float]:
        """(roll, pitch) [deg] the automatic trim adds to the IMU corrections: the walk's only.
        gravity: the projected gravity after the mount calibration only."""
        walk = self.registered_moves.get("walk")
        if self.tilt_trim is None or not isinstance(walk, WalkMove):
            return 0.0, 0.0
        if not user_input.imu_auto_trim:
            if self.tilt_trim.learned:
                self.tilt_trim = TiltTrim(self._posture_ref)
                print("IMU auto trim off: trim cleared", end="\r\n", flush=True)
            return 0.0, 0.0
        return self.tilt_trim.update(now_s, gravity, walk.policy_velocity, walk.state == MoveState.ACTIVE,
                                     os.path.basename(walk.agent_path), walk.mirrored)

    @staticmethod
    def _write_status(payload: dict) -> None:
        try:
            text = json.dumps(payload, default=float)
            tmp = STATUS_PATH.with_suffix(".tmp")
            tmp.write_text(text + "\n", encoding="utf-8")
            os.replace(tmp, STATUS_PATH)
        except (OSError, TypeError, ValueError):
            pass

    def _record_tick(self, obs: Observation, command: MotorCommand, start_time: float) -> None:
        """Record every tick while the walk or the dance is running (one file per session)."""
        rec = self.recorder
        if rec.active and self.registered_moves[rec.move].state == MoveState.INACTIVE:
            rec.stop(f"{rec.move} stopped")
        if not rec.active:
            for name in RECORDED_MOVES:
                move = self.registered_moves.get(name)
                if move is not None and move.state == MoveState.ACTIVE:
                    rec.start(self.controller, move, name)
                    break
        if rec.active:
            move = self.registered_moves[rec.move]
            rec.append(
                obs.robot_state,
                obs.user_input,
                command.target_angles,
                policy_ran=move.state == MoveState.ACTIVE and OBSERVATION_DOF_ORDER[0] in command.target_angles,
                current_proxy_a=self._last_total_current,
                loop_ms=(time.perf_counter() - start_time) * 1000.0,
                frame=getattr(move, "_frame", float("nan")),
                kp_written=getattr(move, "kp_written", float("nan")),
                imu_trim_deg=self._trim_deg,
            )

    def _estimate_total_current(self, robot_state, target_angles: dict[str, float]) -> float:
        """Estimate the total pack current [A] from whichever signal is available.

        - If present_current was read (Observer.observe_current = True), use the measured
          sum of |current| over all motors.
        - Otherwise, fall back to the bam servo current proxy (no extra bus read), from
          the (delay-aligned) command target, present_position and present_velocity:
            duty = clip(PROXY_KP * PROXY_ERROR_GAIN * (target - q), ±PROXY_MAX_PWM)
            I    = (PROXY_VIN * duty - PROXY_KT * dq) / PROXY_R, |I| capped at BAM_MAX_CURRENT
          The back-EMF term (PROXY_KT * dq) keeps the estimate low when the motor is moving
          toward an overshot RL target, while still flagging stalled motors (low dq, high error).
        """
        currents = robot_state.motor_currents
        if currents:
            return sum(abs(c) for c in currents.values())

        # The firmware P gain is the one the active RL move set (walk KP_RL, prop up KP_GROUND);
        # a move may set its own on some joints (joint_kp: the upper body under the walk).
        active = [m for m in self.registered_moves.values() if m.state == MoveState.ACTIVE]
        kp_all = next((m.kp for m in active if hasattr(m, "kp")), PROXY_KP)
        joint_kp = {name: kp for m in active for name, kp in getattr(m, "joint_kp", {}).items()}
        positions = robot_state.motor_positions
        velocities = robot_state.motor_velocities
        total = 0.0
        for name, target in target_angles.items():
            pos = positions.get(name)
            if pos is None:
                continue
            dq = velocities.get(name, 0.0)
            duty = joint_kp.get(name, kp_all) * PROXY_ERROR_GAIN * (target - pos)
            duty = max(-PROXY_MAX_PWM, min(PROXY_MAX_PWM, duty))
            current = (PROXY_VIN * duty - PROXY_KT * dq) / PROXY_R
            total += min(abs(current), BAM_MAX_CURRENT)
        return total

    def _check_overcurrent(self, robot_state, target_angles: dict[str, float]) -> bool:
        """Report when the estimated total pack current stays above OVERCURRENT_CUTOFF_A
        for OVERCURRENT_DEBOUNCE_TICKS consecutive ticks.

        When True, the run loop breaks and _cleanup() disables torque on every motor,
        leaving the robot compliant so the BMS does not trip on the current spike.
        """
        total_current = self._estimate_total_current(robot_state, target_angles)
        self._last_total_current = total_current
        if total_current >= OVERCURRENT_CUTOFF_A:
            self._overcurrent_ticks += 1
        else:
            self._overcurrent_ticks = 0

        if self._overcurrent_ticks >= OVERCURRENT_DEBOUNCE_TICKS:
            print(
                f"Overcurrent safety triggered: {total_current:.2f} A (threshold "
                f"{OVERCURRENT_CUTOFF_A:.2f} A) — disabling torque",
                end="\r\n",
                flush=True,
            )
            return True
        return False

    def _send_to_motors(self, command: MotorCommand):
        """Send one batched goal position command from the composed command dict."""
        if not command.target_angles:
            return

        targets = command.target_angles
        for name, (lo, hi) in HEAD_LIMITS_RAD.items():
            if name in targets:
                targets[name] = min(hi, max(lo, float(targets[name])))
        motor_ids = [MOTOR_TO_ID[name] for name in targets]
        target_positions = list(targets.values())

        self.controller.sync_write_goal_position(motor_ids, target_positions)