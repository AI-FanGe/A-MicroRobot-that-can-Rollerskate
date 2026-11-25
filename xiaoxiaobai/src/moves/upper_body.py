# SPDX-License-Identifier: GPL-3.0-or-later

"""Arms and head (motors 13-20) free under the walk: gestures, manual targets, idle life.

Under a free-arms walk policy (FREE_ARMS_AGENTS, trained on legs 1-12 while its arms played
these gestures) the arms are entirely the upper body's: their targets are the gesture pose.

While music plays, the API hands over the song's beat grid ("groove") and the upper body sways
to it (gestures.Groove).

Under a free-arms policy the upper body can also go limp (input "limp", the web UI's hand
teaching): KP 0 on 13-20 while the legs keep balancing, the targets following the measured
angles so that stiffening again holds the pose the hands left, before fading back to the
gesture engine (which the API points at that pose).

Older walk policies keep the legs and still balance with their arms: the arm targets are
their own, plus what gestures, manual targets and the alive layer of gestures.UpperBody add
to a low-passed copy of them, scaled down per policy and with speed. The policy observes the
real arms. In simulation (MuJoCo + BAM, random walk commands, a gesture every 2.5-4 s) that
kept it on its feet, where feeding it virtual arms that follow its own actions, or the real
arms minus the gesture, made it fall.
"""

import math
import os
import time

from constants import (KP_DEFAULT, KP_UPPER, MOTOR_TO_ID, NEUTRAL_POSE, UPPER_BODY_SCALE, UPPER_BODY_SCALE_UNKNOWN,
                       VX_MAX, VX_MAX_BACKWARD, VY_MAX)
from controller import ControllerProtocol
from gestures import (JOINT_LIMITS_DEG, POSE_KEYS, POSE_LIMITS, REST, UPPER_JOINTS, Hug, UpperBody, UpperBodyInput,
                      from_joints, to_joints)
from moves.move import MotorCommand, Move, MoveState
from observer import Observation

FADE_S = 0.5            # blend in / out against the policy arms
BASE_TAU_S = 0.25       # low-pass of the policy arm targets used as the resting arms
KP_REFRESH_S = 1.0      # a walk (re)start writes KP_RL to every motor: write ours again
COMMAND_MAX_AGE_S = 2.0  # a gesture request older than this (e.g. found on start) is ignored
FALLEN_GRAVITY_Z = -0.5  # same threshold as the walk
# Big arm moves shift the centre of mass: smaller ones the faster the walk, most of all
# backwards (arms up and back while stepping back tipped the robot over in simulation).
SPEED_SHRINK = 0.25     # at full speed
BACKWARD_SHRINK = 0.35  # extra, at full backward speed
SHRINK_TAU_S = 0.4
STILL_SPEED = 0.1       # normalized speed under which idle gestures may start on their own
LIMP_FADE_S = 0.8       # from the pose left by hand back to the gesture engine
GROOVE_WALK_SHRINK = 0.6  # the sway to music, at full speed
LEAD_HEAD_SPEED_DEG_S = 60.0  # led by the hand: the head turns back ahead this fast
# Led by the hand, each arm is limp while a hand moves it; let go (still for LEAD_STILL_S, away
# from rest), it eases back to rest at LEAD_RETURN_DEG_S at LEAD_KP, its target at most
# LEAD_LEASH_DEG ahead, and the walk slows down with it (moves/lead.py). Held again (the target
# LEAD_HELD_DEG ahead for LEAD_HELD_S), it goes limp again.
LEAD_ARMS = {"right": ("right_shoulder_pitch", "right_shoulder_roll", "right_elbow"),
             "left": ("left_shoulder_pitch", "left_shoulder_roll", "left_elbow")}
LEAD_KP = 100
LEAD_LEASH_DEG = 10.0
LEAD_RETURN_DEG_S = 25.0
LEAD_STILL_DEG_S = 6.0
LEAD_STILL_S = 0.6
LEAD_HELD_STILL_S = 2.0  # after a hand was found holding it: wait longer before trying again
LEAD_SPEED_TAU_S = 0.15
LEAD_AT_REST_DEG = 3.0   # back at rest: limp again, so a hand can take it up freely
LEAD_HELD_DEG = 7.0
LEAD_HELD_S = 0.25
HUG_IN_S = 0.6          # into a hug (input "hug", gestures.Hug) and out of it
HUG_OUT_S = 0.9
ARMS = [name for name in UPPER_JOINTS if name not in ("head", "head_pitch")]
LIMITS_RAD = {name: (math.radians(lo), math.radians(hi)) for name, (lo, hi) in JOINT_LIMITS_DEG.items()}


class UpperBodyMove(Move):
    label = "Upper body"
    kp_upper = KP_UPPER

    def __init__(self, controller: ControllerProtocol | None = None, seed: int | None = None) -> None:
        super().__init__()
        self._controller = controller
        self.engine = UpperBody(seed)
        # Moves whose KP the motors go back to (walk, skate), and moves that own the arms
        # while they run: set by the caller.
        self.peers: list[Move] = []
        self.yield_to: list[Move] = []
        self._weight = 0.0
        self._base: dict[str, float] | None = None
        self._last_t: float | None = None
        self._next_kp_s = 0.0
        self._seq: int | None = None
        self._input = UpperBodyInput()
        self._scale = 1.0
        self._want_limp = False
        self._limp = False
        # Led by the hand (input "lead", moves/lead.py): the arms soft and easing back to rest
        # (_spring_back), the head held ahead.
        self._want_lead = False
        self._lead = False
        self._lead_target: dict[str, float] = {}
        self._lead_arms: dict[str, dict] = {}
        # Joint targets left by hand, blended out over LIMP_FADE_S once stiff again.
        self._held: dict[str, float] = {}
        self._held_w = 0.0
        # Hugging a leg: the hug's seed while asked for, its generator and how far it is blended in.
        self._want_hug: int | None = None
        self._hug: Hug | None = None
        self._hug_seed: int | None = None
        self._hug_w = 0.0
        self._last_pose: dict[str, float] | None = None

    @property
    def joint_kp(self) -> dict[str, int]:
        if self.state == MoveState.INACTIVE:
            return {}
        if not self._limp:
            return {name: self.kp_upper for name in UPPER_JOINTS}
        if self._lead:
            stiff = {name for side, joints in LEAD_ARMS.items() if self._lead_arms.get(side, {}).get("returning")
                     for name in joints}
            return {name: (LEAD_KP if name in stiff else 0) if name in ARMS else self.kp_upper for name in UPPER_JOINTS}
        return {name: 0 for name in UPPER_JOINTS}

    def on_start(self, obs: Observation, command: MotorCommand) -> None:
        self._base = None
        self._last_t = None
        self._next_kp_s = 0.0
        self._seq = None
        self._want_limp = self._limp = False
        self._want_lead = self._lead = False
        self._lead_target, self._lead_arms = {}, {}
        self._held, self._held_w = {}, 0.0
        self._want_hug = self._hug = self._hug_seed = None
        self._hug_w = 0.0
        self._last_pose = None
        self.state = MoveState.ACTIVE
        self.step(obs, command)

    def step(self, obs: Observation, command: MotorCommand) -> None:
        self._tick(obs, command, stopping=False)

    def on_stop(self, obs: Observation, command: MotorCommand) -> None:
        self._tick(obs, command, stopping=True)
        if self._weight <= 0.0:
            self._limp = False
            peer = self._peer()
            self._write_kp(getattr(peer, "kp_written", getattr(peer, "kp", KP_DEFAULT)), getattr(peer, "joint_kp", {}))
            self.engine.stop()
            self.state = MoveState.INACTIVE

    def status(self) -> dict:
        return {
            "gesture": self.engine.playing,
            "face": self.engine.face,
            "auto": self._input.auto,
            "manual": sorted(self._input.manual),
            "weight": round(self._weight, 2),
            "scale": round(self._scale, 2),
            "seq": self._seq,
            "limp": self._limp,
            "lead": self._limp and self._lead,
            "hug": self._hug.phase if self._hug is not None and self._hug_w > 0.0 else None,
            "hug_cycles": self._hug.cycles if self._hug is not None else 0,
            "groove": round(self.engine.groove.weight, 2),
            "groove_move": self.engine.groove.move,
            "groove_section": self.engine.groove.section,
        }

    def _tick(self, obs: Observation, command: MotorCommand, stopping: bool) -> None:
        now = obs.robot_state.time_s
        dt = 0.02 if self._last_t is None else min(max(now - self._last_t, 0.0), 0.1)
        self._last_t = now

        fallen = bool(obs.robot_state.projected_gravity) and obs.robot_state.projected_gravity[2] > FALLEN_GRAVITY_Z
        yielding = any(m.state != MoveState.INACTIVE for m in self.yield_to)
        want = 0.0 if stopping or fallen or yielding else 1.0
        step = dt / FADE_S
        self._weight = min(want, self._weight + step) if want > self._weight else max(want, self._weight - step)

        peer = self._peer()
        free_arms = bool(getattr(peer, "free_arms", False))
        self._read_input(obs, now)
        # Only a legs-only policy can take limp arms: the others balance with them.
        limp = (self._want_limp or self._want_lead) and free_arms and not (stopping or fallen or yielding)
        lead = self._want_lead and not self._want_limp
        released = False
        if limp != self._limp or (limp and lead != self._lead):
            self._limp = limp
            self._lead = lead
            self._lead_target, self._lead_arms = {}, {}
            self._next_kp_s = 0.0
            if limp:
                self.engine.stop()
            else:
                released = True
                self._held_w = 1.0 if self._held else 0.0
        if not yielding and (not stopping or released) and now >= self._next_kp_s:
            self._next_kp_s = now + KP_REFRESH_S
            self._write_kp(self.kp_upper, self.joint_kp if self._limp else None)
        if self._limp:
            limp_joints = ARMS if self._lead else UPPER_JOINTS
            for name in limp_joints:
                measured = obs.robot_state.motor_positions.get(name)
                if measured is not None:
                    self._held[name] = float(measured)
                    command.target_angles[name] = float(measured)
            if self._lead:
                self._spring_back(obs, dt, command)
                ahead = to_joints(REST)
                step = math.radians(LEAD_HEAD_SPEED_DEG_S) * dt
                for name in ("head", "head_pitch"):
                    current = self._held.get(name, obs.robot_state.motor_positions.get(name, ahead[name]))
                    self._held[name] = current + min(step, max(-step, ahead[name] - current))
                    command.target_angles[name] = self._held[name]
            return

        v = obs.user_input.velocity
        speed = min(1.0, math.hypot(v["vx"] / VX_MAX, v["vy"] / VY_MAX))
        backward = min(1.0, max(0.0, -v["vx"]) / VX_MAX_BACKWARD)
        want_scale = 1.0 if free_arms else \
            self._policy_scale() * (1.0 - SPEED_SHRINK * speed - BACKWARD_SHRINK * backward)
        self._scale += (want_scale - self._scale) * (1.0 - math.exp(-dt / SHRINK_TAU_S))

        if fallen and self.engine.playing:
            self.engine.stop()
        still = speed < STILL_SPEED and abs(v["vtheta"]) < 0.3
        groove = self._input.groove
        if groove is not None:
            groove = {**groove, "size": float(groove.get("size") or 0.0) * (1.0 - GROOVE_WALK_SHRINK * speed)}
        hugging = self._want_hug is not None and not (stopping or fallen or yielding)
        command_input = UpperBodyInput(manual=self._input.manual, auto=self._input.auto and still and not hugging,
                                       groove=groove, wall=time.time())

        # Resting arms: the policy's own arm targets this tick (or the neutral pose), smoothed;
        # under a free-arms policy the gestures' rest pose.
        policy = {name: float(command.target_angles.get(name, NEUTRAL_POSE.get(name, 0.0))) for name in UPPER_JOINTS}
        if free_arms:
            base = dict(REST)
        else:
            if self._base is None:
                self._base = dict(policy)
            k = 1.0 - math.exp(-dt / BASE_TAU_S)
            for name in ARMS:
                self._base[name] += (policy[name] - self._base[name]) * k
            base = from_joints({name: self._base[name] for name in ARMS}, REST)
            base = {key: min(max(value, POSE_LIMITS[key][0]), POSE_LIMITS[key][1]) for key, value in base.items()}

        pose = self._hug_pose(obs, now, dt, hugging, self.engine.update(now, dt, base, command_input))
        w = self._weight
        if free_arms:
            moved = to_joints(pose)
            for name in UPPER_JOINTS:
                command.target_angles[name] = policy[name] + (moved[name] - policy[name]) * w
        else:
            # Arms: the policy target plus what gestures, manual targets and life add to the
            # resting pose, so the fast part of the policy's arm targets (its balancing) stays.
            moved, rest = to_joints(pose, clamp=False), to_joints(base, clamp=False)
            for name in ARMS:
                offset = (moved[name] - rest[name]) * w * self._scale
                lo, hi = LIMITS_RAD[name]
                command.target_angles[name] = min(max(policy[name] + offset, lo), hi)
            head = to_joints(pose)
            for name in ("head", "head_pitch"):
                command.target_angles[name] = policy[name] + (head[name] - policy[name]) * w
        if self._held_w > 0.0:
            for name, held in self._held.items():
                target = command.target_angles.get(name, held)
                command.target_angles[name] = target + (held - target) * self._held_w
            self._held_w = max(0.0, self._held_w - dt / LIMP_FADE_S)

    def _spring_back(self, obs: Observation, dt: float, command: MotorCommand) -> None:
        """Each arm while led: limp in the hand; once it has stood still for LEAD_STILL_S (let
        go), stiff enough to ease back to rest, its target never more than the leash ahead of the
        measured angle nor pushing it away from rest; limp again as soon as it stops following
        (a hand took it)."""
        rest = to_joints(REST)
        step = math.radians(LEAD_RETURN_DEG_S) * dt
        leash = math.radians(LEAD_LEASH_DEG)
        now = obs.robot_state.time_s
        k = 1.0 - math.exp(-dt / LEAD_SPEED_TAU_S)
        for side, joints in LEAD_ARMS.items():
            positions = obs.robot_state.motor_positions
            if any(positions.get(name) is None for name in joints):
                continue
            measured = {name: float(positions[name]) for name in joints}
            arm = self._lead_arms.setdefault(side, {"returning": False, "still": None, "behind": None,
                                                    "last": None, "speed": 0.0, "wait": LEAD_STILL_S})
            if arm["last"] is not None and dt > 0.0:
                moved = max(abs(measured[name] - arm["last"][name]) for name in joints)
                arm["speed"] += (math.degrees(moved) / dt - arm["speed"]) * k
            arm["last"] = measured
            if not arm["returning"]:
                away = max(abs(measured[name] - rest[name]) for name in joints)
                if arm["speed"] < LEAD_STILL_DEG_S and away > math.radians(LEAD_AT_REST_DEG):
                    arm["still"] = arm["still"] if arm["still"] is not None else now
                    if now - arm["still"] >= arm["wait"]:
                        arm.update(returning=True, behind=None)
                        self._lead_target.update(measured)
                        self._next_kp_s = 0.0
                else:
                    arm["still"] = None
            if not arm["returning"]:
                continue
            behind = 0.0
            for name in joints:
                toward = min(leash, max(-leash, rest[name] - measured[name]))
                lo, hi = sorted((measured[name], measured[name] + toward))
                target = self._lead_target.get(name, measured[name])
                target += min(step, max(-step, rest[name] - target))
                self._lead_target[name] = target = min(hi, max(lo, target))
                command.target_angles[name] = target
                behind = max(behind, abs(target - measured[name]))
            if max(abs(measured[name] - rest[name]) for name in joints) < math.radians(LEAD_AT_REST_DEG):
                arm.update(returning=False, still=None, behind=None, wait=LEAD_STILL_S)
                self._next_kp_s = 0.0
            elif behind >= math.radians(LEAD_HELD_DEG):
                arm["behind"] = arm["behind"] if arm["behind"] is not None else now
                if now - arm["behind"] >= LEAD_HELD_S:
                    arm.update(returning=False, still=None, behind=None, wait=LEAD_HELD_STILL_S)
                    self._next_kp_s = 0.0
            else:
                arm["behind"] = None

    def _hug_pose(self, obs: Observation, now: float, dt: float, hugging: bool, pose: dict[str, float]) -> dict[str, float]:
        """The gesture engine's pose, or the hug's blended over it (eased in and out)."""
        if hugging and (self._hug is None or self._hug_seed != self._want_hug):
            self._hug = Hug(self._last_pose or pose, self._want_hug)
            self._hug_seed = self._want_hug
            self._hug_w = 0.0
        if hugging:
            self._hug_w = min(1.0, self._hug_w + dt / HUG_IN_S)
        else:
            self._hug_w = max(0.0, self._hug_w - dt / HUG_OUT_S)
        if self._hug is not None and self._hug_w > 0.0:
            measured = obs.robot_state.motor_positions
            roll = None
            if measured.get("right_shoulder_roll") is not None and measured.get("left_shoulder_roll") is not None:
                arms = from_joints({name: float(measured[name]) for name in ("right_shoulder_roll", "left_shoulder_roll")})
                roll = (arms["rr"], arms["lr"])
            hug = self._hug.update(now, dt, roll)
            k = self._hug_w * self._hug_w * (3.0 - 2.0 * self._hug_w)
            pose = {key: pose[key] + (hug[key] - pose[key]) * k for key in POSE_KEYS}
        elif not hugging:
            self._hug = self._hug_seed = None
        self._last_pose = pose
        return pose

    def _peer(self) -> Move | None:
        """The walk (or skate) running under the upper body, if any."""
        return next((m for m in self.peers if m.state != MoveState.INACTIVE), None)

    def _policy_scale(self) -> float:
        """How big gestures may get under the walk policy now running (1 with none)."""
        peer = self._peer()
        if peer is None:
            return 1.0
        name = os.path.basename(getattr(peer, "agent_path", "") or "")
        return UPPER_BODY_SCALE.get(name, UPPER_BODY_SCALE_UNKNOWN)

    def _read_input(self, obs: Observation, now: float) -> None:
        upper = getattr(obs.user_input, "upper", None)
        if not isinstance(upper, dict):
            return
        manual = upper.get("manual") or {}
        self._want_limp = bool(upper.get("limp"))
        self._want_lead = bool(upper.get("lead"))
        hug = upper.get("hug")
        self._want_hug = int(hug.get("seed") or 0) if isinstance(hug, dict) else None
        groove = upper.get("groove")
        self._input = UpperBodyInput(
            manual={k: float(v) for k, v in manual.items() if k in REST},
            auto=bool(upper.get("auto", True)),
            groove=groove if isinstance(groove, dict) else None,
        )
        seq = upper.get("seq")
        if seq == self._seq:
            return
        self._seq = seq
        age = time.time() - float(upper.get("gesture_ts") or 0.0)
        if age <= COMMAND_MAX_AGE_S:
            self.engine.command(upper, now)

    def _write_kp(self, kp: int, joint_kp: dict[str, int] | None = None) -> None:
        if self._controller is None:
            return
        ids = [MOTOR_TO_ID[name] for name in UPPER_JOINTS]
        try:
            self._controller.sync_write_kp(ids, [(joint_kp or {}).get(name, kp) for name in UPPER_JOINTS])
        except Exception as exc:  # noqa: BLE001 - a missed KP write is retried next second
            print(f"Warning: upper body KP write failed: {exc}", end="\r\n", flush=True)
