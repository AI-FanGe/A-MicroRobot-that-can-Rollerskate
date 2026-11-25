# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

import math
import os

import onnxruntime as ort
import numpy as np

from constants import (FREE_ARMS_AGENTS, IMU_MOUNT_QUAT, MOTOR_TO_ID, KP_DEFAULT, KP_RL, KP_UPPER,
                       KP_VIN_MIN_STEP, KP_VIN_REF, KP_VIN_SCALE_RANGE, KP_VIN_STILL_S, OBSERVATION_DOF_ORDER,
                       SOFT_RAMP_DOWN_S, SOFT_RAMP_JERK_S, SOFT_RAMP_K, SOFT_RAMP_UP_S,
                       VELOCITY_RAMP_DOWN_S, VELOCITY_RAMP_UP_S, VTHETA_MAX_STATIONARY, VX_MAX, VY_MAX,
                       WALK_AGENT_DEFAULT, WALK_AGENTS, WALK_UPPER_AGENT)
from controller import ControllerProtocol
from gestures import REST, HeadIdle, from_joints, to_joints
from imu_reader import quat_apply_inverse
from observer import Observation
from moves.lead import LEAD_JOINTS, HandLead
from moves.move import MotorCommand, Move, MoveState


ARMS = [name for name in OBSERVATION_DOF_ORDER if "shoulder" in name or "elbow" in name]
HEAD = ("head", "head_pitch")
# Moves that drive the head themselves: the walk leaves it to them.
HEAD_MOVES = ("head", "upper")
# Set by the robot API (api/motion.py HEAD_FRONT) while a camera-guided source drives: the
# head keeps looking ahead even when the robot stands still, or where user_input.head_yaw says.
HEAD_FRONT = "head_front"
# Set by the robot API while the app's camera follow drives (constants.py FOLLOW_SOURCE): the
# command is ramped the soft way (SOFT_RAMP_*).
SOFT_RAMP = "soft_ramp"
STILL_SPEED = 0.1      # normalized speed under which the head looks around
STILL_VTHETA = 0.15    # rad/s
STILL_COMMAND = 0.01   # m/s, rad/s: the policy is fed no command (KP_VIN_STILL_S)
FULL_SPEED = np.array([VX_MAX, VY_MAX, VTHETA_MAX_STATIONARY])  # per axis, for the ramp
# A policy started with an arm this far from its reference pose [rad] answers with targets
# hundreds of degrees off: the arms are eased to the pose over ARM_BLEND_S first, hidden from
# the policy, which balances on the legs meanwhile.
ARM_BLEND_MIN = 0.5
ARM_BLEND_S = 1.5

# Set to True to log motor positions and voltages during the walk move
# Note: requires to set observe_voltage = True in the Observer to log voltages
LOGGING = False


def _mirror_matrices() -> tuple[np.ndarray, np.ndarray]:
    """(observation, action) mirror matrices across the trunk x-z plane, for the walk's
    observation layout gyro(3) + proj_grav(3) + pos(N) + vel(N) + action(N) + cmd(3)."""
    n = len(OBSERVATION_DOF_ORDER)
    joints = np.zeros((n, n))
    for i, name in enumerate(OBSERVATION_DOF_ORDER):
        side, rest = name.split("_", 1)
        joints[i, OBSERVATION_DOF_ORDER.index(("right_" if side == "left" else "left_") + rest)] = -1.0
    # The gyro is read in the IMU frame and is an axial vector: -(reflection of the trunk y axis).
    y_imu = np.asarray(quat_apply_inverse(list(IMU_MOUNT_QUAT), [0.0, 1.0, 0.0]))
    gyro = 2.0 * np.outer(y_imu, y_imu) - np.eye(3)
    blocks = [gyro, np.diag([1.0, -1.0, 1.0]), joints, joints, joints, np.diag([1.0, -1.0, -1.0])]
    obs = np.zeros((6 + 3 * n + 3, 6 + 3 * n + 3))
    i = 0
    for b in blocks:
        obs[i:i + len(b), i:i + len(b)] = b
        i += len(b)
    return obs, joints


def kp_voltage_scale(supply_voltage: float | None, enabled: bool = True) -> float:
    """Factor on KP_RL that keeps KP * supply voltage at KP_RL * KP_VIN_REF (constants.py)."""
    if not enabled or not supply_voltage or supply_voltage <= 0.0:
        return 1.0
    lo, hi = KP_VIN_SCALE_RANGE
    return min(max(KP_VIN_REF / supply_voltage, lo), hi)


class WalkMove(Move):
    """Walk using a RL policy trained in simulation."""

    kp = KP_RL
    label = "Walk"
    head_idle = True  # motors 19-20 look around while it stands still (gestures.HeadIdle)
    agent_field = "walk_agent"  # UserInput field holding the index into the agents
    # The walk's observation layout: user_input.mirror_policy applies (_mirror_matrices).
    mirrorable = True

    def __init__(
        self,
        controller: ControllerProtocol | None = None,
        agent_path: str | None = None,
        agents: list[tuple[str, str]] | None = None,
    ) -> None:
        super().__init__()
        self._controller = controller
        self._last_action = [0.0] * len(OBSERVATION_DOF_ORDER)

        # Load ONNX policies: all of agents (WALK_AGENTS by default; agent_path replaces the
        # first), switched live by user_input.walk_agent, and with WALK_AGENTS the
        # WALK_UPPER_AGENT, run instead while the upper body is free. They must share the
        # joints and observation layout.
        selectable = agents is None
        agents = WALK_AGENTS if agents is None else agents
        self._agent_paths = [f"src/agents/{name}" for _, name in agents]
        if agent_path:
            self._agent_paths[0] = agent_path
        self._upper_index: int | None = None
        if selectable:
            self._upper_index = len(self._agent_paths)
            self._agent_paths.append(f"src/agents/{WALK_UPPER_AGENT}")
        self._sessions = [ort.InferenceSession(path) for path in self._agent_paths]
        metas = [s.get_modelmeta().custom_metadata_map for s in self._sessions]
        # Inputs: base_obs = gyro(3) + proj_grav(3) + pos(N) + vel(N) + action(N) + cmd(3), and
        # for a reference-phase policy (cos, sin) of its phase after that. The policies in the
        # list share the first one's inputs, except that each may have the phase or not, with its
        # own period (phase_period_steps metadata). Subclasses (dance) build their own inputs.
        # Each keeps its own reference pose.
        base_obs_size = 3 + 3 + 3 * len(OBSERVATION_DOF_ORDER) + 3
        walk_inputs = {base_obs_size, base_obs_size + 2}
        first_size = self._sessions[0].get_inputs()[0].shape[1]
        for path, m, s in zip(self._agent_paths[1:], metas[1:], self._sessions[1:]):
            size = s.get_inputs()[0].shape[1]
            if m["joint_names"] != metas[0]["joint_names"] \
                    or (size != first_size and not {size, first_size} <= walk_inputs):
                raise ValueError(f"{path} does not match the joints / inputs of {self._agent_paths[0]}")
        self._phase_flags = [s.get_inputs()[0].shape[1] > base_obs_size for s in self._sessions]
        self._phase_periods = [int(m.get("phase_period_steps", 20)) for m in metas]
        # An explicit agent_path replaces slot 0 and is what sim replay runs; otherwise start
        # on the keyboard default so the first tick does not switch models.
        self._agent_index = WALK_AGENT_DEFAULT if selectable and not agent_path else 0
        self._ort_session = self._sessions[self._agent_index]
        self.agent_path = self._agent_paths[self._agent_index]
        print(f"{self.label} policy: {self.agent_path}", end="\r\n", flush=True)

        self.action_scale = 1.0
        self._head = HeadIdle()
        self._head_owned = False
        self._last_t: float | None = None
        self._velocity = np.zeros(3)  # ramped command fed to the policy
        self._accel = np.zeros(3)  # its rate of change under the soft ramp
        self.kp_scale = 1.0  # supply-voltage factor on self.kp (kp_voltage_scale)
        self._still_since: float | None = None  # since when the policy is fed no command
        self._lead = HandLead()
        self._leading = False

        # Reference pose of each policy: read from ONNX metadata
        self._default_poses: list[dict[str, float]] = [
            dict(zip(m["joint_names"].split(","), (float(v) for v in m["default_joint_pos"].split(","))))
            for m in metas
        ]
        self._default_pose: dict[str, float] = self._default_poses[self._agent_index]
        self._blend: tuple[float, dict[str, float]] | None = None  # start time, measured pose

        self._phase_step = 0.0
        # Stroke policies (stroke_rate_* metadata) output one more value after the joints, the
        # rate the phase advances at: clip(1 + scale * a, lo, hi) per tick. Others advance 1.
        self._rate_specs = [
            (float(m["stroke_rate_scale"]), *(float(v) for v in m["stroke_rate_range"].split(",")))
            if "stroke_rate_scale" in m else None
            for m in metas
        ]
        self._stroke_rate = 1.0

        # user_input.mirror_policy runs a policy mirrored left/right: it sees the mirrored
        # observation and its action is mirrored back, a policy with the opposite left/right bias.
        # A lean or veer that stays on the same side comes from the robot, one that changes side
        # from the policy. Not for policies with a reference phase or extra outputs (skate
        # stroke), nor around an asymmetric reference pose ((q - pose) would not mirror).
        self._mirror_obs, self._mirror_action = _mirror_matrices()
        self._mirrorable: list[bool] = []
        for pose, phase, rate in zip(self._default_poses, self._phase_flags, self._rate_specs):
            q = np.array([pose[n] for n in OBSERVATION_DOF_ORDER])
            self._mirrorable.append(self.mirrorable and not phase and rate is None
                                    and np.allclose(self._mirror_action @ q, q, atol=1e-6))
        self._mirrored = False

        # Safety parameters
        self._projected_gravity_z_threshold = -0.5  # Threshold for detecting a fall based on projected gravity

        # Logging
        self.position: dict[str, list[float]] = {name: [] for name in MOTOR_TO_ID}
        self.voltage: dict[str, list[float]] = {name: [] for name in MOTOR_TO_ID}

    @property
    def _use_reference_phase(self) -> bool:
        return self._phase_flags[self._agent_index]

    @property
    def _phase_total_steps(self) -> int:
        return self._phase_periods[self._agent_index]

    @property
    def free_arms(self) -> bool:
        """The running policy drives the legs only (FREE_ARMS_AGENTS): it neither sees nor moves
        the arms, which rest at KP_UPPER or belong to the upper body move."""
        return os.path.basename(self.agent_path) in FREE_ARMS_AGENTS

    @property
    def joint_kp(self) -> dict[str, int]:
        return {name: KP_UPPER for name in ARMS} if self.free_arms and self.state != MoveState.INACTIVE else {}

    @property
    def kp_written(self) -> int:
        """The P gain written to the motors: self.kp (what the scheduler's current proxy assumes,
        at PROXY_VIN) scaled for the supply voltage."""
        return round(self.kp * self.kp_scale)

    @property
    def policy_velocity(self) -> tuple[float, float, float]:
        """(vx, vy, vtheta) the policy is fed: the command after the ramp or the lead."""
        vx, vy, vtheta = self._velocity
        return float(vx), float(vy), float(vtheta)

    def _write_kp(self, upper_body: bool = False) -> None:
        """upper_body: the upper body move holds the arms and head at its own KP."""
        if self._controller is None:
            return
        kp = {name: self.kp_written for name in MOTOR_TO_ID}
        kp.update({name: KP_UPPER for name in ARMS} if self.free_arms else {})
        if upper_body:
            kp = {name: gain for name, gain in kp.items() if name not in ARMS and name not in HEAD}
        self._controller.sync_write_kp([MOTOR_TO_ID[n] for n in kp], list(kp.values()))

    @property
    def mirrored(self) -> bool:
        return self._mirrored

    def _kp_scale_for(self, obs: Observation) -> float:
        """kp_voltage_scale while walking; standing still (no command fed to the policy for
        KP_VIN_STILL_S) never stiffer than self.kp: the stiffer legs tremble in place."""
        scale = kp_voltage_scale(obs.robot_state.supply_voltage, obs.user_input.kp_vin_comp)
        now = obs.robot_state.time_s
        if np.any(np.abs(self._velocity) > STILL_COMMAND):
            self._still_since = None
        elif self._still_since is None:
            self._still_since = now
        if self._still_since is not None and now - self._still_since >= KP_VIN_STILL_S:
            scale = min(scale, 1.0)
        return scale

    def _follow_supply_voltage(self, obs: Observation) -> None:
        enabled = obs.user_input.kp_vin_comp
        scale = self._kp_scale_for(obs)
        if abs(round(self.kp * scale) - self.kp_written) < KP_VIN_MIN_STEP:
            return
        before = self.kp_written
        self.kp_scale = scale
        try:
            self._write_kp(upper_body="upper" in obs.user_input.active_moves)
        except Exception as exc:  # noqa: BLE001 - retried on the next tick
            self.kp_scale = before / self.kp
            print(f"Warning: {self.label} KP write failed: {exc}", end="\r\n", flush=True)
            return
        volts = obs.robot_state.supply_voltage
        reason = f"for a {volts:.2f} V supply" if enabled and volts else "(voltage compensation off)"
        if self.kp_scale < kp_voltage_scale(volts, enabled):
            reason += ", standing: no stiffer than KP " + str(self.kp)
        print(f"{self.label} KP {before} -> {self.kp_written} {reason}", end="\r\n", flush=True)

    def on_start(self, obs: Observation, command: MotorCommand) -> None:
        self._velocity = np.zeros(3)
        self._still_since = obs.robot_state.time_s - KP_VIN_STILL_S
        self.kp_scale = self._kp_scale_for(obs)
        self._write_kp()
        self._head_owned = False
        self._accel = np.zeros(3)
        self._lead.reset()
        self._leading = False
        self._start_blend(obs)
        self.state = MoveState.ACTIVE

    def _start_blend(self, obs: Observation) -> None:
        measured = obs.robot_state.motor_positions
        arms = [] if self.free_arms else [n for n in ARMS if n in measured]
        far = any(abs(measured[n] - self._default_pose[n]) > ARM_BLEND_MIN for n in arms)
        self._blend = (obs.robot_state.time_s, {n: measured[n] for n in arms}) if far else None

    def _blend_arms(self, obs: Observation, command: MotorCommand) -> None:
        """Arm targets on the way from the measured pose to the reference pose."""
        if self._blend is None:
            return
        start_s, start = self._blend
        s = min(max((obs.robot_state.time_s - start_s) / ARM_BLEND_S, 0.0), 1.0)
        s = s * s * (3.0 - 2.0 * s)
        for name, q in start.items():
            command.target_angles[name] = q + (self._default_pose[name] - q) * s
        if s >= 1.0:
            self._blend = None

    def step(self, obs: Observation, command: MotorCommand) -> None:
        index = getattr(obs.user_input, self.agent_field, 0)
        if self._upper_index is not None:
            if "upper" in obs.user_input.active_moves:
                index = self._upper_index
            elif index == self._upper_index:
                index = self._agent_index
        if index != self._agent_index and 0 <= index < len(self._sessions):
            self._agent_index = index
            self._ort_session = self._sessions[index]
            self._stroke_rate = 1.0
            self._phase_step = 0.0
            free_before = self.free_arms
            self.agent_path = self._agent_paths[index]
            print(f"{self.label} policy switched to {self.agent_path}", end="\r\n", flush=True)
            if self.free_arms != free_before:
                self._write_kp()
            if self._default_poses[index] != self._default_pose:
                # The last action is relative to the previous policy's pose.
                self._default_pose = self._default_poses[index]
                self._last_action = [0.0] * len(OBSERVATION_DOF_ORDER)
                self._start_blend(obs)
        self._follow_supply_voltage(obs)

        now = obs.robot_state.time_s
        dt = 0.02 if self._last_t is None else min(max(now - self._last_t, 0.0), 0.1)
        self._last_t = now
        self._update_lead(obs, dt)
        self._ramp_velocity(obs, dt)
        if self._blend is not None:
            self._velocity[:] = 0.0
            self._accel[:] = 0.0

        # Update reference phase
        if self._use_reference_phase:
            commanded_vel = np.mean(np.abs(self._velocity))
            if commanded_vel > 0.01:
                self._phase_step += self._stroke_rate
            else:
                self._phase_step = 0.0

        # Safety check: if the robot is fallen, stop the policy
        if obs.robot_state.projected_gravity[2] > self._projected_gravity_z_threshold:
            return

        # Run policy
        input_obs = self.build_observation(obs)
        mirror = obs.user_input.mirror_policy and self._mirrorable[self._agent_index]
        if mirror != self._mirrored:
            self._mirrored = mirror
            print(f"{self.label} policy {'mirrored left/right' if mirror else 'unmirrored'}: {self.agent_path}",
                  end="\r\n", flush=True)
        if mirror:
            input_obs = (self._mirror_obs @ np.asarray(input_obs)).tolist()
        ort_inputs = {self._ort_session.get_inputs()[0].name: [input_obs]}
        ort_outs = self._ort_session.run(None, ort_inputs)
        action = ort_outs[0][0]
        rate_spec = self._rate_specs[self._agent_index]
        if rate_spec is not None:
            scale, lo, hi = rate_spec
            self._stroke_rate = float(np.clip(1.0 + scale * action[len(OBSERVATION_DOF_ORDER)], lo, hi))
            action = action[:len(OBSERVATION_DOF_ORDER)]
        if mirror:
            action = self._mirror_action @ action
        self._last_action = action.tolist()

        # Update command
        for i, name in enumerate(OBSERVATION_DOF_ORDER):
            if self.free_arms and name in ARMS:
                command.target_angles[name] = self._default_pose[name]
            else:
                command.target_angles[name] = self._default_pose[name] + action[i] * self.action_scale
        self._blend_arms(obs, command)
        self._look_around(obs, command, dt)

        # Log positions and voltages
        if LOGGING:
            for name in MOTOR_TO_ID.keys():
                self.position[name].append(obs.robot_state.motor_positions[name])
                self.voltage[name].append(obs.robot_state.motor_voltages[name])

    def _ramp_velocity(self, obs: Observation, dt: float) -> None:
        """Move the policy's velocity command towards command_velocity() at a bounded rate:
        faster while slowing down (or reversing) than while speeding up. Under SOFT_RAMP the
        rate itself is ramped (an S-shaped start and stop) and eases off onto the target."""
        target = np.array(self.command_velocity(obs))
        slowing = (np.abs(target) < np.abs(self._velocity)) | (target * self._velocity < 0)
        if SOFT_RAMP not in obs.user_input.active_moves:
            rate = FULL_SPEED * np.where(slowing, 1.0 / VELOCITY_RAMP_DOWN_S, 1.0 / VELOCITY_RAMP_UP_S)
            self._velocity += np.clip(target - self._velocity, -rate * dt, rate * dt)
            self._accel[:] = 0.0
            return
        top = FULL_SPEED / np.where(slowing, SOFT_RAMP_DOWN_S, SOFT_RAMP_UP_S)
        want = np.clip(SOFT_RAMP_K * (target - self._velocity), -top, top)
        jerk = top / SOFT_RAMP_JERK_S * dt
        self._accel += np.clip(want - self._accel, -jerk, jerk)
        before = self._velocity.copy()
        self._velocity += self._accel * dt
        landed = (target - before) * (target - self._velocity) <= 0.0
        self._velocity[landed] = target[landed]
        self._accel[landed] = 0.0

    def _look_around(self, obs: Observation, command: MotorCommand, dt: float) -> None:
        """Head (19-20) idle while standing still, back to the front while moving; left to the
        head and upper body moves when they run."""
        now = obs.robot_state.time_s
        if not self.head_idle or any(m in obs.user_input.active_moves for m in HEAD_MOVES):
            self._head_owned = False
            return
        if not self._head_owned:
            measured = {n: obs.robot_state.motor_positions.get(n, 0.0) for n in HEAD}
            pose = from_joints(measured)
            self._head.reset(pose["yaw"], pose["tilt"])
            self._head_owned = True
        vx, vy, vtheta = self._velocity
        still = math.hypot(vx / VX_MAX, vy / VY_MAX) < STILL_SPEED and abs(vtheta) < STILL_VTHETA
        front = HEAD_FRONT in obs.user_input.active_moves
        aim = obs.user_input.head_yaw if front else None
        look = self._head.update(now, dt, still and not front, aim)
        joints = to_joints({**REST, **look})
        for name in HEAD:
            command.target_angles[name] = joints[name]

    def command_velocity(self, obs: Observation) -> tuple[float, float, float]:
        """(vx, vy, vtheta) the policy is ramped towards (_ramp_velocity): led by the hand
        (moves/lead.py) unless a stick or the app is driving."""
        v = obs.user_input.velocity
        if self._leading and all(abs(value) < 1e-3 for value in v.values()):
            return self._lead.vx, 0.0, self._lead.vtheta
        return v["vx"], v["vy"], v["vtheta"]

    def _update_lead(self, obs: Observation, dt: float) -> None:
        """Lead by the hand: asked for by the API (upper input "lead") under the free-arms walk,
        whose arms the upper body move then leaves limp."""
        upper = obs.user_input.upper
        leading = (isinstance(upper, dict) and bool(upper.get("lead")) and "upper" in obs.user_input.active_moves
                   and self.free_arms)
        if not leading:
            if self._leading:
                self._lead.reset()
            self._leading = False
            return
        joints = {name: obs.robot_state.motor_positions[name] for name in LEAD_JOINTS
                  if name in obs.robot_state.motor_positions}
        if len(joints) == len(LEAD_JOINTS):
            self._lead.update(joints, dt)
            self._leading = True

    def status(self) -> dict:
        if not self._leading:
            return {"lead": None}
        return {"lead": {"vx": round(self._lead.vx, 3), "vtheta": round(self._lead.vtheta, 3)}}

    def build_observation(self, obs: Observation) -> list[float]:
        """Build policy observation from robot state."""
        input_obs = []
        
        # IMU data: gyroscope and projected gravity in body frame
        input_obs.extend(obs.robot_state.gyro)
        input_obs.extend(obs.robot_state.projected_gravity)
        
        # Motor positions, velocities, last action; a free-arms policy was trained with its
        # arm entries zeroed, and they are zeroed while the arms are eased to the pose
        hidden = set(ARMS) if self.free_arms or self._blend is not None else set()
        for name in OBSERVATION_DOF_ORDER:
            input_obs.append(0.0 if name in hidden else obs.robot_state.motor_positions[name] - self._default_pose[name])
        for name in OBSERVATION_DOF_ORDER:
            input_obs.append(0.0 if name in hidden else obs.robot_state.motor_velocities[name])
        input_obs.extend(0.0 if name in hidden else a for name, a in zip(OBSERVATION_DOF_ORDER, self._last_action))

        # Command
        input_obs.extend(self._velocity.tolist())

        # Reference phase
        if self._use_reference_phase:
            reference_phase = (self._phase_step % self._phase_total_steps) / self._phase_total_steps * 2 * np.pi
            input_obs.append(np.cos(reference_phase))
            input_obs.append(np.sin(reference_phase))

        return input_obs

    def on_stop(self, obs: Observation, command: MotorCommand) -> None:
        # Hold the measured pose before raising KP: the last policy targets can lead q by
        # more than 100 deg, and KP_DEFAULT on those leads draws a pack-wide current spike
        # (BMS brownout / Pi reboot).
        hold = {name: obs.robot_state.motor_positions[name] for name in OBSERVATION_DOF_ORDER}
        command.target_angles.update(hold)
        if self._controller is not None:
            self._controller.sync_write_goal_position([MOTOR_TO_ID[n] for n in hold], list(hold.values()))
            ids = list(MOTOR_TO_ID.values())
            self._controller.sync_write_kp(ids, [KP_DEFAULT] * len(ids))
        self.state = MoveState.INACTIVE

        # Save json logs
        if LOGGING:
            import json
            with open("walk_log.json", "w") as f:
                json.dump({
                    "position": self.position,
                    "voltage": self.voltage,
                }, f, indent=4)