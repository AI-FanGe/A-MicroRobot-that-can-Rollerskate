# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

"""Motion control after the PrimeBot SDK: McAction modes with a transition table, and
prioritised velocity input sources with a watchdog.

The control loop (main.py / scheduler.py) stays unchanged in spirit: this module only
writes the remote-input file it already reads (active moves + velocity) at 10 Hz and reads
the status file the scheduler publishes. The actual mode is always derived from what the
loop reports, never assumed from what was asked.
"""

from __future__ import annotations

import math
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gestures
from api.backend import JOINT_IDS, REPO, Backend, read_constants
from api.events import EventBus
from api.gesture_store import GestureStore
from api.protocol import ApiError, CommonState, Reason, TaskRegistry, response
from api.store import Settings

WRITE_PERIOD_S = 0.1
STATUS_STALE_S = 2.0
START_TIMEOUT_S = 60.0
SWITCH_TIMEOUT_S = 20.0
SEQUENCE_MAX_STEPS = 8
GESTURE_WAIT_S = 20.0  # longest a sequence waits for one gesture to play out
GESTURE_START_S = 1.5  # a gesture the loop took but is not playing after this never started
STREAM_PERIOD_S = 0.02  # taught moves played from here under 「上肢控制」 (no control loop there)
TEACH_IDS = list(range(13, 21))


@dataclass(frozen=True)
class ActionInfo:
    name: str
    value: int
    label: str
    description: str
    settable: bool = True


ACTIONS: dict[str, ActionInfo] = {
    a.name: a
    for a in (
        ActionInfo("PASSIVE_DEFAULT", 1, "零力矩", "控制循环未运行，全部电机断开力矩。急停后也回到这里。"),
        ActionInfo("STOPPING", 2, "停止中", "正在停止控制循环并断开力矩。", settable=False),
        ActionInfo("STARTING", 100, "启动中", "控制循环启动中：电机缓慢回到中立位并加载策略模型。", settable=False),
        ActionInfo("BIPED_STAND_DEFAULT", 201, "位控站立", "保持中立站姿，不主动平衡；需要平衡请切到走跑。"),
        ActionInfo("BIPED_WALK_RUN", 1002, "走跑", "强化学习走路策略，原地踏步并接收速度指令。"),
        ActionInfo("BIPED_CUSTOM_UPPER", 1014, "上肢控制", "单独控制 13–20 号电机，腿部无力矩；先把机器人放平或悬挂。"),
        ActionInfo("BIPED_WALK_UPPER", 1015, "走跑·上肢自由",
                   "走路策略管双腿，13–20 号电机自由：拟人动作、空闲时的小动作和手动调角度，同样接收速度指令。"),
        ActionInfo("BIPED_SKATE", 1016, "滑板", "穿滑板鞋时的强化学习策略：只能向前滑（0–0.5 m/s）和转弯，后退即停。"),
        ActionInfo("BIPED_GYMNASTICS", 1019, "广播体操",
                   "强化学习做操策略：从站姿做一套约 46 秒的广播体操（踏步、伸展、扩胸、踢腿、体侧、体转、全身、跳跃、整理），"
                   "做完自动回到走跑原地平衡。需要平整空旷的地面，不接收速度指令。"),
        ActionInfo("BIPED_PEEK", 1024, "偷看",
                   "强化学习偷看策略：从站姿左脚迈开、蹲低，身体向左斜出去探头张望，缩回、慢慢探出、被发现缩回、连探两下，"
                   "约 15 秒，全程只靠双脚站稳、不扶任何东西；做完自动回到走跑原地平衡。需要平整的地面，不接收速度指令。"),
        ActionInfo("BIPED_PROPUP", 1020, "撑起/躺下",
                   "强化学习撑起策略：机器人仰面躺着时，用手肘和手撑起上身并保持（约 3 秒）；再按一次躺回去（约 3 秒）。"
                   "一键切换：再次设置本模式即在撑起和躺下之间切换。只能从仰躺开始，不接收速度指令。"),
        ActionInfo("BIPED_CHILL_LIE", 1021, "躺着休闲",
                   "仰面躺着循环做放松的小动作：腿时大时小地抬、晃脚踝，手臂轻动，头四处看（约 37 秒一轮，无缝循环）。"
                   "只能从仰躺开始（如「撑起/躺下」躺下后）；离开时先回到平躺姿势再切换。"),
        ActionInfo("BIPED_CHILL_SIT", 1022, "坐地休闲",
                   "坐在地上、双手向后撑着，双脚自然晃动、头左右上下张望。只能从后仰坐姿开始：「撑起/躺下」撑起后切过来，"
                   "或在位控站立时把机器人扶成坐姿；离开时先回到撑起姿势，可再切「撑起/躺下」躺回去。"),
        ActionInfo("BIPED_CHILL_CHAIR", 1023, "坐凳休闲",
                   "坐在约 11 cm 高、没有靠背的凳子边上，双手向后撑住凳面，双脚悬空晃动，头四处张望。"
                   "在位控站立时把机器人放到凳子上扶稳再切换；结束时请先扶住，再切回位控站立。"),
        ActionInfo("BIPED_WAKE_STRETCH", 1025, "起床伸懒腰",
                   "仰面躺着做一次约 9 秒的起床伸懒腰：迷迷糊糊蹭蹭腿、晃晃头，双臂举过头顶、双腿伸直绷脚尖，"
                   "保持住全身发抖 2 秒，再把手放回身侧、腿放松，表情跟着变。只能从仰躺开始（位控站立时把机器人平放在地上），"
                   "做完自动回到位控站立。"),
    )
}
ACTION_BY_VALUE = {a.value: a for a in ACTIONS.values()}

PASSIVE = "PASSIVE_DEFAULT"
STOPPING = "STOPPING"
STARTING = "STARTING"
STAND = "BIPED_STAND_DEFAULT"
WALK = "BIPED_WALK_RUN"
CUSTOM_UPPER = "BIPED_CUSTOM_UPPER"
WALK_UPPER = "BIPED_WALK_UPPER"
SKATE = "BIPED_SKATE"
GYMNASTICS = "BIPED_GYMNASTICS"
PEEK = "BIPED_PEEK"
PROPUP = "BIPED_PROPUP"
CHILL_LIE = "BIPED_CHILL_LIE"
CHILL_SIT = "BIPED_CHILL_SIT"
CHILL_CHAIR = "BIPED_CHILL_CHAIR"
WAKE_STRETCH = "BIPED_WAKE_STRETCH"
DANCES = {GYMNASTICS: "gymnastics", PEEK: "peek"}  # action -> its DanceMove in main.py
CHILLS = {CHILL_LIE: "chill_lie", CHILL_SIT: "chill_sit", CHILL_CHAIR: "chill_chair",
          WAKE_STRETCH: "wake_stretch"}  # action -> its ChillMove
LYING_CHILLS = {CHILL_LIE, WAKE_STRETCH}
# 撑起/躺下 must have reached this phase before switching to the chill mode.
CHILL_FROM_PROPUP = {CHILL_LIE: ("lying", "先在「撑起/躺下」里躺下，再切到躺着休闲"),
                     CHILL_SIT: ("up", "先在「撑起/躺下」里撑起，再切到坐地休闲"),
                     WAKE_STRETCH: ("lying", "先在「撑起/躺下」里躺下，再切到起床伸懒腰")}
# Faces (expressions.py) for the sections of a chill clip (mjlab_xiaoxiaobai scripts/chill_motion.py).
CHILL_FACES = {"迷糊醒来": "drowsy", "伸懒腰": "stretch", "全身发抖": "stretch_shiver", "放松": "relieved"}

# Legal switches from the actual mode (PASSIVE is always reachable: emergency stop).
TRANSITIONS: dict[str, set[str]] = {
    PASSIVE: {STAND, CUSTOM_UPPER, PROPUP, *LYING_CHILLS},
    STARTING: {STAND, PROPUP, *LYING_CHILLS},
    STAND: {WALK, WALK_UPPER, SKATE, *DANCES, PROPUP, *CHILLS},
    WALK: {STAND, WALK_UPPER, SKATE, *DANCES},
    WALK_UPPER: {STAND, WALK, SKATE, *DANCES},
    SKATE: {STAND, WALK},
    **{dance: {STAND, WALK} for dance in DANCES},
    PROPUP: {STAND, CHILL_LIE, CHILL_SIT, WAKE_STRETCH},
    CHILL_LIE: {STAND, PROPUP, WAKE_STRETCH},
    CHILL_SIT: {STAND, PROPUP},
    CHILL_CHAIR: {STAND},
    WAKE_STRETCH: {STAND, PROPUP, CHILL_LIE},
    CUSTOM_UPPER: set(),
    STOPPING: set(),
}
MOVES_FOR_ACTION = {STAND: set(), WALK: {"walk"}, WALK_UPPER: {"walk", "upper"}, SKATE: {"skate"},
                    **{dance: {move} for dance, move in DANCES.items()}, PROPUP: {"propup"},
                    **{chill: {move} for chill, move in CHILLS.items()}}
LOOP_ACTIONS = {STAND, WALK, WALK_UPPER, SKATE, *DANCES, PROPUP, *CHILLS}
WALKING = {WALK, WALK_UPPER, SKATE}  # take velocity commands
# Faces (expressions.py) for each phase of a hug (gestures.Hug), one picked at random each time.
HUG_FACES = {"reach": ("shy", "content", "cat", "clingy"), "nuzzle": ("content", "cat", "shy", "kiss", "clingy", "petted"),
             "look_up": ("shy", "giggle", "cute", "clingy")}

# Walk policy and IMU mount correction, as the keyboard of `main.py` sets them ([c]/[n]/[m], [a]/[d]/[w]/[s]/[z]).
_WALK = read_constants("WALK_AGENTS", "WALK_AGENT_DEFAULT", "SKATE_AGENTS", "SKATE_AGENT_DEFAULT",
                       "IMU_ROLL_CORRECTION_DEG", "IMU_PITCH_CORRECTION_DEG", "KP_VIN_COMP", "IMU_AUTO_TRIM",
                       "IMU_MOUNT_ROLL_DEG", "IMU_MOUNT_PITCH_DEG", "IMU_CALIB_S", "IMU_CALIB_MAX_DEG")
WALK_AGENTS: list[tuple[str, str]] = [tuple(a) for a in _WALK["WALK_AGENTS"]]
SKATE_AGENTS: list[tuple[str, str]] = [tuple(a) for a in _WALK["SKATE_AGENTS"]]
AGENT_LISTS = {"walk_agent": WALK_AGENTS, "skate_agent": SKATE_AGENTS}
TUNING_DEFAULTS = {
    "walk_agent": int(_WALK["WALK_AGENT_DEFAULT"]),
    "skate_agent": int(_WALK["SKATE_AGENT_DEFAULT"]),
    "imu_roll_corr_deg": float(_WALK["IMU_ROLL_CORRECTION_DEG"]),
    "imu_pitch_corr_deg": float(_WALK["IMU_PITCH_CORRECTION_DEG"]),
    # Diagnostic (moves/walk.py): not saved, a restart walks unmirrored again.
    "mirror_policy": False,
    # Supply-voltage compensation of the walk's KP (moves/walk.py), automatic IMU trim (tilt_trim.py).
    "kp_vin_comp": bool(_WALK["KP_VIN_COMP"]),
    "imu_auto_trim": bool(_WALK["IMU_AUTO_TRIM"]),
    # IMU mount calibration (imu_calib.py, imu_calibrate): a property of the robot, kept by a reset.
    "imu_mount_roll_deg": float(_WALK["IMU_MOUNT_ROLL_DEG"]),
    "imu_mount_pitch_deg": float(_WALK["IMU_MOUNT_PITCH_DEG"]),
    "imu_calib_ts": 0.0,  # when they were calibrated (0: never)
}
UNSAVED_TUNING = {"mirror_policy"}
TUNING_SWITCHES = ("mirror_policy", "kp_vin_comp", "imu_auto_trim")
IMU_CORR_MAX_DEG = 6.0
MOUNT_KEYS = ("imu_mount_roll_deg", "imu_mount_pitch_deg", "imu_calib_ts")
IMU_MOUNT_MAX_DEG = float(_WALK["IMU_CALIB_MAX_DEG"])
IMU_CALIB_WAIT_S = float(_WALK["IMU_CALIB_S"]) + 5.0
IMU_CALIB_ERRORS = {
    "not_standing": "机器人没在站立模式",
    "moving": "机器人在晃动，放平地上别碰它再试",
    "pose": "关节没在站立姿态（{joints}），先切到「位控站立」并等它站稳",
    "range": "算出的安装偏差 横滚 {roll_deg:+.1f}° 俯仰 {pitch_deg:+.1f}° 超过 ±{max:.0f}°，检查地面是否水平、IMU 是否松动",
    "no_ref": "缺少 src/agents/posture_ref.json（运行 src/debug/posture_ref.py 生成）",
}
_FOLLOW = read_constants("FOLLOW_SOURCE", "FOLLOW_WALK_AGENT", "FOLLOW_LINGER_S")
FOLLOW_SOURCE: str = _FOLLOW["FOLLOW_SOURCE"]
FOLLOW_AGENT_INDEX = next(i for i, (_, file) in enumerate(WALK_AGENTS) if file == _FOLLOW["FOLLOW_WALK_AGENT"])
FOLLOW_LINGER_S = float(_FOLLOW["FOLLOW_LINGER_S"])
SOFT_RAMP = "soft_ramp"  # in active_moves: moves/walk.py ramps the command softly
PROPUP_DOWN = read_constants("PROPUP_DOWN")["PROPUP_DOWN"]
CHILL_SETTLE = read_constants("CHILL_SETTLE")["CHILL_SETTLE"]
_DANCE = read_constants("GYMNASTICS_MUSIC", "DANCE_MUSIC_PAD_S", "DANCE_START_LATENCY_S")
DANCE_MUSIC_PATHS = {GYMNASTICS: REPO / "src" / "agents" / _DANCE["GYMNASTICS_MUSIC"]}
DANCE_MUSIC_WAIT_S = 10.0  # the soundtrack may wait for a voice reply to end; then the dance goes without it
# Upper body under the walk: motor ID -> (authoring key, joint name), in gestures.UPPER_JOINTS order.
UPPER_KEYS = {JOINT_IDS[name]: (key, name) for key, name in zip(gestures.POSE_KEYS, gestures.UPPER_JOINTS)}
_SIGN = read_constants("MOTOR_SIGN")["MOTOR_SIGN"]
# Raw motor travel [deg] for BIPED_CUSTOM_UPPER: the joint ranges times the motor sign.
UPPER_MOTOR_LIMITS_DEG = {
    motor_id: tuple(sorted(v * _SIGN[name] for v in gestures.JOINT_LIMITS_DEG[name]))
    for motor_id, (_key, name) in UPPER_KEYS.items()
}

STOP_REASONS = {
    "stop_flag": "已按请求停止",
    "overcurrent": "过流保护触发，已断开力矩",
    "serial_error": "电机串口通信失败",
    "interrupted": "控制循环被中断",
    "error": "控制循环异常退出",
}


@dataclass
class InputSource:
    name: str
    priority: int
    timeout_ms: int
    description: str
    builtin: bool = False
    head_front: bool = False  # while it drives, the head looks ahead instead of idling (camera-guided sources)


# In active_moves: no move of that name, moves/walk.py reads it to keep the head ahead (or
# where the source aims it, VelocityLock.head_yaw).
HEAD_FRONT = "head_front"
HEAD_AIM_DEG = 38.0  # inside HEAD_LIMITS_DEG["head"]

BUILTIN_SOURCES = [
    InputSource("rc", 80, 1000, "实体手柄（在机器人上本地读取，摇杆一动即优先于网络输入）", True),
    InputSource("interaction", 80, 1000, "语音、大模型等交互指令", True),
    InputSource("app_proxy", 60, 1000, "网页 / App 摇杆", True),
]


@dataclass
class VelocityLock:
    source: str
    priority: int
    timeout_s: float
    ts: float
    velocity: dict[str, float] = field(default_factory=dict)
    head_yaw: float | None = None  # [deg] a head_front source aiming the head

    def fresh(self, now: float) -> bool:
        return now - self.ts <= self.timeout_s


def clamp_unit(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return max(-1.0, min(1.0, number))


def _number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def battery_percent(voltage: float | None) -> int | None:
    """2S Li-ion (2x 18650) open-circuit-ish curve; under load it reads a little low."""
    if voltage is None:
        return None
    cell = voltage / 2.0
    curve = [(3.30, 0), (3.50, 5), (3.65, 20), (3.75, 40), (3.85, 60), (3.97, 80), (4.15, 100)]
    if cell <= curve[0][0]:
        return 0
    for (v0, p0), (v1, p1) in zip(curve, curve[1:]):
        if cell <= v1:
            return int(round(p0 + (p1 - p0) * (cell - v0) / (v1 - v0)))
    return 100


class MotionController:
    def __init__(self, backend: Backend, events: EventBus, tasks: TaskRegistry, settings: Settings) -> None:
        self.backend = backend
        self.events = events
        self.tasks = tasks
        self.settings = settings
        self._lock = threading.RLock()
        self._target: str = PASSIVE
        # Only write the remote-input file once the API itself picked the mode, so a loop
        # started and driven from the gamepad is never overridden.
        self._commanding = False
        self._head = False
        self._actual: str = PASSIVE
        self._actual_since = time.time()
        self._start_requested = 0.0
        self._stop_requested = 0.0
        self._velocity_lock: VelocityLock | None = None
        self._follow_until = 0.0  # the follow's walk agent and soft ramp hold until then
        self._sources: dict[str, InputSource] = {s.name: s for s in BUILTIN_SOURCES}
        self._status: dict[str, Any] = {}
        self._battery: dict[str, Any] = {"voltage_v": None, "ts": None}
        self._last_stop: dict[str, Any] | None = None
        self._low_battery_warned = False
        self._walk_task: tuple[int, threading.Event] | None = None
        self._sequence: tuple[int, threading.Event] | None = None
        self._upper_targets: dict[int, float] = {}
        # Written to the control loop with WALK_UPPER (moves/upper_body.py reads it).
        self._upper_cmd: dict[str, Any] = {"seq": 0, "gesture": None, "gesture_ts": 0.0, "size": 1.0,
                                           "manual": {}, "auto": True, "limp": False, "lead": False, "hug": None}
        # Hugging a leg (set_hug): the mode to go back to, the face before it, the hug phase shown.
        self._hug_from: str | None = None
        self._hug_face: str | None = None
        self._hug_shown: tuple[Any, Any] | None = None
        # A gesture's own faces (gestures.Key face): the one shown, and the face before them.
        self._gesture_shown: str | None = None
        self._gesture_face: str | None = None
        self.custom = GestureStore()
        # 13-20 limp and moved by hand: in 「上肢控制」 (legs limp too) or 「走跑·上肢自由」 (legs balancing).
        self._teach = False
        self._streaming: tuple[str, threading.Event] | None = None  # taught move played under 「上肢控制」
        self._tuning = self._load_tuning()
        self._calib_seq = 0.0  # time of the last IMU mount calibration request (imu_calibrate)
        self.music: Any = None  # set by the app: plays the dance soundtrack (play_dance)
        self._dance_pending = 0.0  # when a dance was asked for, waiting on its soundtrack
        self._dance_next = GYMNASTICS  # the dance waiting on its soundtrack
        self._dance_timer: threading.Timer | None = None
        self._dance_music = False  # the soundtrack plays with the running dance
        self._propup_down = False  # 撑起/躺下: lie down (else prop up)
        self._chill_exit: str | None = None  # leaving a chill mode for this one once it settled
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="xiaoxiaobai-api-motion", daemon=True)

    def start(self) -> None:
        self._actual = self._derive(time.time())
        if self._actual in LOOP_ACTIONS or self._actual == CUSTOM_UPPER:
            self._target = self._actual  # adopt a loop started by the gamepad daemon or the console
        self._thread.start()

    def close(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------ state

    def _derive(self, now: float) -> str:
        if self.backend.upper_pid():
            return CUSTOM_UPPER
        pid = self.backend.loop_pid()
        if pid is None:
            return PASSIVE
        status = self.backend.read_status() or {}
        self._status = status
        if status.get("pid") == pid and not status.get("running", True):
            return STOPPING
        if self._stop_requested and now - self._stop_requested < 8.0:
            return STOPPING
        fresh = status.get("pid") == pid and status.get("running") and now - float(status.get("ts", 0)) < STATUS_STALE_S
        if not fresh:
            return STARTING
        voltage = status.get("voltage_v")
        if voltage:
            self._battery = {"voltage_v": voltage, "ts": now}
        moves = status.get("moves") or {}
        if moves.get("skate") in ("starting", "active"):
            return SKATE
        if moves.get("propup") in ("starting", "active"):
            return PROPUP
        for chill, move in CHILLS.items():
            if moves.get(move) in ("starting", "active"):
                return chill
        for dance, move in DANCES.items():
            if moves.get(move) in ("starting", "active"):
                return dance
        if moves.get("walk") in ("starting", "active"):
            return WALK_UPPER if moves.get("upper") in ("starting", "active") else WALK
        return STAND

    def _run(self) -> None:
        while not self._stop.is_set():
            now = time.time()
            try:
                self._tick(now)
            except Exception as exc:  # noqa: BLE001 - the motion thread must survive
                self.events.alert("error", f"运动控制线程异常：{exc}")
            self._stop.wait(WRITE_PERIOD_S)

    def _tick(self, now: float) -> None:
        with self._lock:
            previous = self._actual
            actual = self._derive(now)
            if actual != previous:
                self._actual, self._actual_since = actual, now
                self._on_change(previous, actual, now)
            self._end_dance(actual)
            self._check_propup(actual)
            self._check_chill(actual)
            if self._dance_pending and self._dance_timer is None and now - self._dance_pending > DANCE_MUSIC_WAIT_S:
                self.events.alert("warning", "配乐没能及时播放，直接开始动作")
                self._dance_pending, self._target = 0.0, self._dance_next
                if self.music is not None:
                    self.music.stop()
            target = self._target
            if self._commanding and actual in (STARTING, *LOOP_ACTIONS) and target in LOOP_ACTIONS:
                from input.remote_input import write_remote_input

                moves = set(MOVES_FOR_ACTION[target])
                if target == PROPUP and self._propup_down:
                    moves.add(PROPUP_DOWN)
                if target in CHILLS and self._chill_exit:
                    moves.add(CHILL_SETTLE)
                calibrating = now - self._calib_seq < IMU_CALIB_WAIT_S
                if self._head and not calibrating:
                    moves.add("head")
                lock = self._velocity_lock
                velocity = None
                head_yaw = None
                tuning = dict(self._tuning)
                tuning["imu_calib_seq"] = self._calib_seq
                if lock is not None and lock.source == FOLLOW_SOURCE:
                    self._follow_until = max(self._follow_until, lock.ts + lock.timeout_s + FOLLOW_LINGER_S)
                elif lock is not None and lock.fresh(now):
                    self._follow_until = 0.0
                if now < self._follow_until:
                    moves.add(SOFT_RAMP)
                    tuning["walk_agent"] = FOLLOW_AGENT_INDEX
                if lock is not None and lock.fresh(now):
                    velocity = lock.velocity if actual in WALKING else {"vx": 0.0, "vy": 0.0, "vtheta": 0.0}
                    source = self._sources.get(lock.source)
                    if source is not None and source.head_front:
                        moves.add(HEAD_FRONT)
                        head_yaw = lock.head_yaw
                elif lock is not None:
                    self._velocity_lock = None
                    self.events.publish("input_source", owner=None, released=lock.source)
                upper = dict(self._upper_cmd) if target == WALK_UPPER else None
                write_remote_input(active_moves=moves, velocity=velocity, upper=upper, tuning=tuning,
                                   head_yaw=head_yaw, path=self.backend.paths.remote_input)
            if actual == STARTING and self._start_requested and now - self._start_requested > START_TIMEOUT_S:
                self._start_requested = 0.0
                self.events.alert("error", "控制循环启动超时，请查看「开发者 → 日志 → run」")
                self.tasks.finish_kind("mc_action", CommonState.TIMEOUT, "启动超时")
            self._check_battery()
            self._finish_tasks(actual, now)
            self._hug_expression(actual)
            self._gesture_expression(actual)

    def _on_change(self, previous: str, actual: str, now: float) -> None:
        self.events.publish("mode", action=actual, previous=previous, label=ACTIONS[actual].label)
        if actual == PASSIVE:
            self._velocity_lock = None
            self._head = False
            expected = self._stop_requested and now - self._stop_requested < 15.0
            reason = (self._status or {}).get("stop_reason") or ("stop_flag" if expected else "error")
            self._last_stop = {"ts": now, "reason": reason, "text": STOP_REASONS.get(reason, reason), "expected": bool(expected)}
            if not expected and previous != CUSTOM_UPPER:
                self.events.alert("error", f"控制循环已退出：{STOP_REASONS.get(reason, reason)}", reason=reason)
            self._target = PASSIVE
            self._commanding = False
            self._stop_requested = 0.0
            self._start_requested = 0.0
        if previous == STARTING and actual in LOOP_ACTIONS:
            self._start_requested = 0.0
        if actual == CUSTOM_UPPER:
            self._target = CUSTOM_UPPER
        if previous == CUSTOM_UPPER:
            self._upper_targets.clear()
            self._teach = False
            self._cancel_stream()
        if previous in DANCES and self._dance_music:
            self._cancel_dance_music()
        if previous in CHILLS:
            self._chill_exit = None
        if previous == WALK_UPPER:
            self._upper_cmd["manual"] = {}
            self._upper_cmd["limp"] = False
            self._upper_cmd["lead"] = False
            self._teach = False
            self._end_hug()
        if actual in LOOP_ACTIONS and actual != self._target and not self._action_pending():
            # Switched from the gamepad: follow it and stop writing remote moves.
            self._target = actual
            self._commanding = False

    def _end_dance(self, actual: str) -> None:
        """The dance holds its last frame once played: hand over to the walk's balance, or to
        the position stand if the robot fell."""
        if actual not in DANCES or self._target != actual or not self._commanding:
            return
        dance = self._status.get(DANCES[actual]) or {}
        if not dance.get("done"):
            return
        if dance.get("fallen"):
            self._target = STAND
            self.events.alert("warning", "动作中检测到摔倒，已停止并切到位控站立")
        else:
            self._target = WALK
            self._dance_music = False  # the soundtrack rings out on its own
            self.events.publish("dance_done", duration_s=dance.get("duration_s"))

    def _check_propup(self, actual: str) -> None:
        """撑起 only starts lying on the back: otherwise the move holds still, go to the position stand."""
        if actual != PROPUP or self._target != PROPUP or not self._commanding:
            return
        if (self._status.get("propup") or {}).get("refused"):
            self._target = STAND
            self.events.alert("warning", "撑起需要机器人仰面躺着，已切到位控站立")

    def _check_chill(self, actual: str) -> None:
        """A chill mode that did not start (wrong posture) goes to the position stand, as does one
        played once when it is done; one being left hands over once it settled back in its first pose."""
        if actual not in CHILLS or self._target != actual or not self._commanding:
            return
        chill = self._status.get(CHILLS[actual]) or {}
        if chill.get("refused"):
            self._target, self._chill_exit = STAND, None
            posture = "仰面躺着" if actual in LYING_CHILLS else "坐着并向后仰"
            self.events.alert("warning", f"{ACTIONS[actual].label}需要机器人{posture}，已切到位控站立")
        elif self._chill_exit and chill.get("phase") == "settled":
            self._target, self._chill_exit = self._chill_exit, None
        elif chill.get("phase") == "done":
            self._target, self._chill_exit = STAND, None
            self.events.publish("chill_done", action=actual, label=ACTIONS[actual].label)

    def _dance_music_file(self, dance: str) -> Path | None:
        path = DANCE_MUSIC_PATHS.get(dance)
        if self.music is None or path is None or not path.exists():
            return None
        cfg = self.settings.section("music")
        return path if cfg.get("enabled", True) and cfg.get("dance_music", True) else None

    def _cancel_dance_music(self) -> None:
        """Drops a dance start waiting on its soundtrack, and the soundtrack of a dance cut short."""
        playing = self._dance_pending or self._dance_music
        if self._dance_pending:
            self.tasks.finish_kind("mc_action", CommonState.FAILURE, "动作已取消")
        self._dance_pending, self._dance_music = 0.0, False
        if self._dance_timer is not None:
            self._dance_timer.cancel()
            self._dance_timer = None
        if playing and self.music is not None:
            self.music.stop()

    def _dance_music_started(self, heard_at: float | None) -> None:
        """From the music thread: the soundtrack's first sample is heard at heard_at (None: it
        will not play). The dance starts so that its frame 0 falls on the soundtrack's t = 0."""
        with self._lock:
            if not self._dance_pending or self._dance_timer is not None:
                return
            if heard_at is None:
                self.events.alert("warning", "配乐没能播放，直接开始动作")
                delay = 0.0
            else:
                # 动作时机 (music.groove_offset_ms): positive moves the dance earlier.
                early = max(-300, min(300, int(self.settings.section("music").get("groove_offset_ms", 0)))) / 1000.0
                delay = heard_at + _DANCE["DANCE_MUSIC_PAD_S"] - _DANCE["DANCE_START_LATENCY_S"] - early - time.time()
            self._dance_timer = threading.Timer(max(0.0, delay), self._begin_dance, args=(heard_at is not None,))
            self._dance_timer.daemon = True
            self._dance_timer.start()

    def _begin_dance(self, with_music: bool) -> None:
        with self._lock:
            if not self._dance_pending:
                return
            self._dance_pending, self._dance_timer = 0.0, None
            self._target, self._dance_music = self._dance_next, with_music
        self._tick(time.time())  # the remote input now, not on the next 10 Hz tick

    def _action_pending(self) -> bool:
        return any(
            t["kind"] == "mc_action" and t["state"] == CommonState.RUNNING.name for t in self.tasks.recent(20)
        )

    def _finish_tasks(self, actual: str, now: float) -> None:
        for task in self.tasks.recent(20):
            if task["kind"] != "mc_action" or task["state"] != CommonState.RUNNING.name:
                continue
            if actual == task["target"]:
                self.tasks.finish(task["task_id"], CommonState.SUCCESS, f"已切换到{ACTIONS[actual].label}")
            elif task["target"] != PASSIVE and actual == PASSIVE and now - task["created"] > 1.0:
                self.tasks.finish(task["task_id"], CommonState.FAILURE, "控制循环已退出")
            elif now - task["created"] > (START_TIMEOUT_S if task.get("needs_start") else SWITCH_TIMEOUT_S):
                self.tasks.finish(task["task_id"], CommonState.TIMEOUT, "切换超时")

    def _check_battery(self) -> None:
        voltage = self._battery.get("voltage_v")
        warn = float(self.settings.section("safety").get("warn_battery_v", 6.8))
        if voltage and voltage < warn and not self._low_battery_warned:
            self._low_battery_warned = True
            self.events.alert("warning", f"电池电压 {voltage:.2f} V，电量偏低，请尽快充电")
        elif voltage and voltage > warn + 0.2:
            self._low_battery_warned = False

    # ------------------------------------------------------------------ queries

    def get_action(self) -> dict[str, Any]:
        with self._lock:
            actual = self._actual
            info = ACTIONS[actual]
            allowed = sorted(TRANSITIONS.get(actual, set()) | ({PASSIVE} if actual != PASSIVE else set()))
            return {
                "current_action": {"name": info.name, "value": info.value, "label": info.label},
                "target_action": self._chill_exit or self._target,
                "since": self._actual_since,
                "allowed": [{"name": a, "value": ACTIONS[a].value, "label": ACTIONS[a].label} for a in allowed],
                "head_motion": self._head,
                "loop": {
                    "pid": self.backend.loop_pid(),
                    "active_moves": self._status.get("active_moves", []) if actual in LOOP_ACTIONS else [],
                    "moves": self._status.get("moves", {}) if actual in LOOP_ACTIONS else {},
                    "velocity": self._status.get("velocity") if actual in WALKING else None,
                    "current_a": self._status.get("current_a") if actual in LOOP_ACTIONS else None,
                    "roll_deg": self._status.get("roll_deg") if actual in LOOP_ACTIONS else None,
                    "pitch_deg": self._status.get("pitch_deg") if actual in LOOP_ACTIONS else None,
                    "supply_v": self._status.get("supply_v") if actual in LOOP_ACTIONS else None,
                    "kp_written": self._status.get("kp_written") if actual in LOOP_ACTIONS else None,
                    "imu_trim_deg": self._status.get("imu_trim_deg") if actual in LOOP_ACTIONS else None,
                    "imu_trim_given_up": self._status.get("imu_trim_given_up", []) if actual in LOOP_ACTIONS else [],
                    "imu_mount_deg": self._status.get("imu_mount_deg") if actual in LOOP_ACTIONS else None,
                    "imu_calib_running": bool(self._status.get("imu_calib_running")) if actual in LOOP_ACTIONS else False,
                    "loop_ms": self._status.get("loop_ms") if actual in LOOP_ACTIONS else None,
                    "overruns": self._status.get("overruns") if actual in LOOP_ACTIONS else None,
                },
                "last_stop": self._last_stop,
                "upper": self._upper_state() if actual == WALK_UPPER else None,
                "dance": self._status.get(DANCES[actual]) if actual in DANCES else None,
                "propup": self._propup_state() if actual == PROPUP else None,
                "chill": {**(self._status.get(CHILLS[actual]) or {}), "exit_to": self._chill_exit}
                if actual in CHILLS else None,
                "dance_pending": self._dance_next if self._dance_pending else None,
                "editor": {"teach": self._teach, "playing": self._streaming[0] if self._streaming else None},
                "tuning": dict(self._tuning),
            }

    def _propup_state(self) -> dict[str, Any]:
        """phase (the loop's): rising / up / lowering / lying / refused; down: what was asked for."""
        return {**(self._status.get("propup") or {}), "down": self._propup_down}

    def catalog(self) -> list[dict[str, Any]]:
        return [
            {
                "name": a.name,
                "value": a.value,
                "label": a.label,
                "description": a.description,
                "settable": a.settable,
                "from": ["*"] if a.name == PASSIVE else sorted(src for src, dst in TRANSITIONS.items() if a.name in dst),
            }
            for a in ACTIONS.values()
        ]

    def battery(self) -> dict[str, Any]:
        voltage = self._battery.get("voltage_v")
        safety = self.settings.section("safety")
        level = "unknown"
        if voltage:
            level = "critical" if voltage < safety["low_battery_v"] else "low" if voltage < safety["warn_battery_v"] else "ok"
        return {
            "voltage_v": voltage,
            "percent": battery_percent(voltage),
            "level": level,
            "measured_at": self._battery.get("ts"),
            "note": None if voltage else "控制循环运行时每 5 秒读一次电机输入电压",
        }

    def joint_state(self) -> dict[str, Any]:
        """Per-joint feedback in the SDK's JointState layout; fields the robot does not read are null."""
        with self._lock:
            actual = self._actual
            live = (self._status.get("joints") or {}) if actual in LOOP_ACTIONS else {}
            upper = dict(self._upper_targets) if actual == CUSTOM_UPPER else {}
            stamp = self._status.get("ts") if live else None
        joints = []
        for name, motor_id in sorted(JOINT_IDS.items(), key=lambda item: item[1]):
            row = live.get(name) or [None] * 5
            target = row[1] if live else upper.get(motor_id)
            joints.append({
                "name": name,
                "motor_id": motor_id,
                "position": row[0],
                "target": target,
                "velocity": row[2],
                "effort": None,
                "current": row[3],
                "motor_vol": row[4],
                "motor_temp": None,
                "coil_temp": None,
            })
        if live:
            source, note = "loop", None
        elif actual == CUSTOM_UPPER:
            source, note = "upper_body", "上肢控制模式只下发目标角，不回读位置"
        else:
            source, note = None, "控制循环运行时（站立 / 走跑 / 滑板）以 5 Hz 回传关节位置"
        return {"source": source, "stamp": stamp, "joints": joints, "note": note}

    def imu_state(self) -> dict[str, Any]:
        with self._lock:
            live = self._actual in LOOP_ACTIONS and bool(self._status)
            status = self._status if live else {}
        return {
            "chip": "IMU",
            "bus": "I2C-1",
            "live": live,
            "stamp": status.get("ts"),
            "roll_deg": status.get("roll_deg"),
            "pitch_deg": status.get("pitch_deg"),
            "gyro": status.get("gyro"),
            "quat": status.get("quat"),
            "note": None if live else "IMU 由控制循环读取，站立后才有数据",
        }

    def actual(self) -> str:
        return self._actual

    # ------------------------------------------------------------------ commands

    def set_action(self, requested: Any, source: str = "app_proxy") -> dict[str, Any]:
        name = self._resolve(requested)
        with self._lock:
            actual = self._actual
            if name == PASSIVE:
                return self._estop(source)
            if not ACTIONS[name].settable:
                raise ApiError(f"{ACTIONS[name].label} 是过渡状态，不能直接设置", CommonState.INVALID, Reason.ACTION)
            if name == PROPUP and actual == PROPUP and self._target == PROPUP:
                return self._toggle_propup(source)
            if name == actual and self._target == actual:
                if self._dance_pending:
                    self._cancel_dance_music()
                self._chill_exit = None
                self._commanding = self._commanding or name in LOOP_ACTIONS
                return response(
                    self.get_action(), state=CommonState.SUCCESS, reason=Reason.TARGET_STATE, message="已经是目标状态"
                )
            if name not in TRANSITIONS.get(actual, set()):
                raise ApiError(
                    f"不能从「{ACTIONS[actual].label}」直接切到「{ACTIONS[name].label}」",
                    CommonState.INVALID,
                    Reason.NO_TRANSFER_PATH,
                )
            voltage = self._battery.get("voltage_v")
            low = float(self.settings.section("safety").get("low_battery_v", 6.4))
            if name in (WALK, WALK_UPPER, SKATE, *DANCES, PROPUP, *CHILLS) and voltage and voltage < low:
                raise ApiError(f"电池电压 {voltage:.2f} V 低于 {low:.1f} V，禁止走路和做动作", CommonState.FAILURE, Reason.LOW_BATTERY)
            if actual == PROPUP and name in CHILL_FROM_PROPUP:
                phase, hint = CHILL_FROM_PROPUP[name]
                if (self._status.get("propup") or {}).get("phase") != phase:
                    raise ApiError(hint, CommonState.INVALID, Reason.NO_TRANSFER_PATH)
            if name == PROPUP:
                self._propup_down = False
            # Leaving a chill mode: it first eases back to its first pose (_check_chill hands over).
            leaving_chill = (actual in CHILLS and self._target == actual and self._commanding
                             and not (self._status.get(CHILLS[actual]) or {}).get("refused"))

            needs_start = actual == PASSIVE and name in LOOP_ACTIONS
            if name == CUSTOM_UPPER:
                self.backend.start_upper()
            elif needs_start:
                self.backend.start_loop()
                self._start_requested = time.time()
            self._cancel_dance_music()
            music = self._dance_music_file(name) if name in DANCES else None
            if music is not None:
                # The current mode holds until the soundtrack is heard (_dance_music_started).
                self._dance_pending, self._dance_next = time.time(), name
                self.music.play_dance(music, self._dance_music_started, ACTIONS[name].label)
            elif leaving_chill:
                self._chill_exit = name
            else:
                self._target = name
            self._commanding = name in LOOP_ACTIONS
            if name not in WALKING:
                self._velocity_lock = None
            task_id = self.tasks.create("mc_action", target=name, source=source, needs_start=needs_start)
            self.events.publish("action_request", target=name, source=source, task_id=task_id)
        message = "正在启动控制循环（约 5–15 秒）" if needs_start else f"正在切换到{ACTIONS[name].label}"
        return response(self.get_action(), state=CommonState.RUNNING, message=message, task_id=task_id)

    def _toggle_propup(self, source: str) -> dict[str, Any]:
        """Already in 撑起/躺下: the same request switches between propping up and lying down."""
        self._propup_down = not self._propup_down
        self._commanding = True
        down = self._propup_down
        self.events.publish("propup", down=down, source=source)
        self._tick(time.time())  # the remote input now, not on the next 10 Hz tick
        return response(self.get_action(), state=CommonState.SUCCESS, message="躺下" if down else "撑起")

    def set_propup(self, down: bool, source: str = "app_proxy") -> dict[str, Any]:
        """Prop up (down False) or lie down (True); from another mode it switches to 撑起/躺下 first."""
        with self._lock:
            if self._actual == PROPUP and self._target == PROPUP:
                if self._propup_down == bool(down):
                    return response(self.get_action(), reason=Reason.TARGET_STATE, message="已经是目标状态")
                return self._toggle_propup(source)
        if down:
            raise ApiError("先进入「撑起/躺下」并撑起后才能躺下", CommonState.INVALID, Reason.ACTION)
        return self.set_action(PROPUP, source=source)

    def _estop(self, source: str) -> dict[str, Any]:
        actual = self._actual
        self._cancel_dance_music()
        self._target = PASSIVE
        self._chill_exit = None
        self._velocity_lock = None
        self._head = False
        if actual == PASSIVE:
            return response(self.get_action(), reason=Reason.TARGET_STATE, message="已经是零力矩状态")
        self._stop_requested = time.time()
        if self.backend.upper_pid():
            self.backend.stop_upper()
        if self.backend.loop_pid():
            self.backend.stop_loop()
        self._cancel_walk_task()
        self._cancel_sequence()
        self._cancel_stream()
        task_id = self.tasks.create("mc_action", target=PASSIVE, source=source)
        self.events.publish("action_request", target=PASSIVE, source=source, task_id=task_id)
        return response(self.get_action(), state=CommonState.RUNNING, message="正在停止并断开力矩", task_id=task_id)

    def emergency_stop(self, source: str = "estop") -> dict[str, Any]:
        with self._lock:
            result = self._estop(source)
        self.events.alert("warning", "急停：控制循环停止，电机断开力矩", source=source)
        return result

    @staticmethod
    def _resolve(requested: Any) -> str:
        if isinstance(requested, dict):
            requested = requested.get("value") or requested.get("name")
        if isinstance(requested, (int, float)) or (isinstance(requested, str) and requested.isdigit()):
            info = ACTION_BY_VALUE.get(int(requested))
            if info:
                return info.name
        elif isinstance(requested, str) and requested.upper() in ACTIONS:
            return requested.upper()
        raise ApiError(f"未知的运动模式：{requested}", CommonState.INVALID, Reason.NOT_FOUND)

    def set_head(self, enabled: bool) -> dict[str, Any]:
        with self._lock:
            if enabled and self._actual not in LOOP_ACTIONS and self._actual != STARTING:
                raise ApiError("控制循环未运行，先站立", CommonState.INVALID, Reason.ACTION)
            self._head = bool(enabled)
        return response({"head_motion": self._head}, message="头部摆动已开启" if enabled else "头部摆动已关闭")

    # ------------------------------------------------------------------ walk policy / IMU correction

    def _load_tuning(self) -> dict[str, Any]:
        saved = self.settings.section("walk")
        tuning = dict(TUNING_DEFAULTS)
        try:
            for key, agents in AGENT_LISTS.items():
                if 0 <= int(saved.get(key, -1)) < len(agents):
                    tuning[key] = int(saved[key])
            for key in ("imu_roll_corr_deg", "imu_pitch_corr_deg"):
                if key in saved:
                    tuning[key] = max(-IMU_CORR_MAX_DEG, min(IMU_CORR_MAX_DEG, float(saved[key])))
            for key in MOUNT_KEYS[:2]:
                if key in saved:
                    tuning[key] = max(-IMU_MOUNT_MAX_DEG, min(IMU_MOUNT_MAX_DEG, float(saved[key])))
            tuning["imu_calib_ts"] = float(saved.get("imu_calib_ts", 0.0))
        except (TypeError, ValueError):
            pass
        for key in TUNING_SWITCHES:
            if key not in UNSAVED_TUNING and isinstance(saved.get(key), bool):
                tuning[key] = saved[key]
        return tuning

    def walk_tuning(self) -> dict[str, Any]:
        return {
            **self._tuning,
            "agents": [{"index": i, "label": label, "file": file} for i, (label, file) in enumerate(WALK_AGENTS)],
            "skate_agents": [{"index": i, "label": label, "file": file} for i, (label, file) in enumerate(SKATE_AGENTS)],
            "defaults": dict(TUNING_DEFAULTS),
            "imu_corr_max_deg": IMU_CORR_MAX_DEG,
            "imu_mount_max_deg": IMU_MOUNT_MAX_DEG,
            "imu_calib_s": float(_WALK["IMU_CALIB_S"]),
        }

    def set_walk_tuning(self, body: dict[str, Any]) -> dict[str, Any]:
        """walk_agent / skate_agent: index into WALK_AGENTS / SKATE_AGENTS; imu_*_corr_deg: projected-gravity
        correction; mirror_policy: run the policies mirrored left/right; kp_vin_comp: supply-voltage compensation
        of the walk's KP; imu_auto_trim: automatic IMU trim; reset: back to the defaults. Applies right away (also
        while walking) and is kept across restarts, except mirror_policy."""
        with self._lock:
            tuning = dict(self._tuning)
            if body.get("reset"):
                tuning = {**TUNING_DEFAULTS, **{key: tuning[key] for key in MOUNT_KEYS}}
            for key, agents in AGENT_LISTS.items():
                if body.get(key) is None:
                    continue
                try:
                    index = int(body[key])
                except (TypeError, ValueError):
                    index = -1
                if not 0 <= index < len(agents):
                    raise ApiError(f"{key} 范围 0–{len(agents) - 1}", CommonState.INVALID)
                tuning[key] = index
            for key in ("imu_roll_corr_deg", "imu_pitch_corr_deg"):
                if body.get(key) is None:
                    continue
                try:
                    value = float(body[key])
                except (TypeError, ValueError):
                    value = math.nan
                if not math.isfinite(value) or abs(value) > IMU_CORR_MAX_DEG:
                    raise ApiError(f"{key} 范围 ±{IMU_CORR_MAX_DEG:.0f}°", CommonState.INVALID)
                tuning[key] = round(value, 2)
            for key in MOUNT_KEYS[:2]:
                if body.get(key) is None:
                    continue
                try:
                    value = float(body[key])
                except (TypeError, ValueError):
                    value = math.nan
                if not math.isfinite(value) or abs(value) > IMU_MOUNT_MAX_DEG:
                    raise ApiError(f"{key} 范围 ±{IMU_MOUNT_MAX_DEG:.0f}°", CommonState.INVALID)
                tuning[key] = round(value, 2)
            for key in TUNING_SWITCHES:
                if body.get(key) is None:
                    continue
                if not isinstance(body[key], bool):
                    raise ApiError(f"{key} 应为 true / false", CommonState.INVALID)
                tuning[key] = body[key]
            self._tuning = tuning
            self.settings.replace("walk", {k: v for k, v in tuning.items()
                                           if k not in UNSAVED_TUNING and v != TUNING_DEFAULTS[k]})
        label = WALK_AGENTS[tuning["walk_agent"]][0]
        skate_label = SKATE_AGENTS[tuning["skate_agent"]][0]
        self.events.publish("walk_tuning", **tuning)
        return response(self.walk_tuning(), message=f"走路策略 {label}，滑板策略 {skate_label}，"
                                                     f"IMU 校正 横滚 {tuning['imu_roll_corr_deg']:+.1f}° "
                                                     f"俯仰 {tuning['imu_pitch_corr_deg']:+.1f}°"
                                                     f"，电压补偿 KP {'开' if tuning['kp_vin_comp'] else '关'}"
                                                     f"，IMU 自动修正 {'开' if tuning['imu_auto_trim'] else '关'}"
                                                     + ("，策略左右镜像中" if tuning["mirror_policy"] else ""))

    def imu_calibrate(self) -> dict[str, Any]:
        """IMU mount calibration (imu_calib.py): the robot stands still in STAND on flat ground for
        IMU_CALIB_S; the mount correction found is saved and the manual corrections go back to 0."""
        with self._lock:
            if self._actual != STAND or self._target != STAND:
                raise ApiError("先切到「位控站立」并等机器人站稳，放在平地上别碰它", CommonState.INVALID, Reason.ACTION)
            if time.time() - self._calib_seq < IMU_CALIB_WAIT_S:
                raise ApiError("IMU 标定进行中", CommonState.INVALID, Reason.ACTION)
            self._commanding = True
            seq = self._calib_seq = time.time()
        deadline = seq + IMU_CALIB_WAIT_S
        result = None
        while time.time() < deadline:
            time.sleep(0.1)
            with self._lock:
                found = self._status.get("imu_calib") or {}
                if found.get("seq") == seq:
                    result = found
                    break
        with self._lock:
            self._calib_seq = 0.0
            if result is None:
                raise ApiError("控制循环没有回应标定请求（是否在运行最新代码？）", CommonState.TIMEOUT)
            if not result.get("ok"):
                text = IMU_CALIB_ERRORS.get(result.get("error"), result.get("error") or "未知错误")
                text = text.format(joints="、".join(result.get("joints", [])), max=IMU_MOUNT_MAX_DEG,
                                   roll_deg=result.get("roll_deg", 0.0), pitch_deg=result.get("pitch_deg", 0.0))
                raise ApiError(f"IMU 标定失败：{text}", CommonState.INVALID)
            tuning = dict(self._tuning)
            tuning.update(imu_mount_roll_deg=result["roll_deg"], imu_mount_pitch_deg=result["pitch_deg"],
                          imu_calib_ts=round(seq), imu_roll_corr_deg=0.0, imu_pitch_corr_deg=0.0)
            self._tuning = tuning
            self.settings.replace("walk", {k: v for k, v in tuning.items()
                                           if k not in UNSAVED_TUNING and v != TUNING_DEFAULTS[k]})
        self.events.publish("walk_tuning", **tuning)
        return response({**self.walk_tuning(), "calibration": result},
                        message=f"IMU 标定完成：安装偏差 横滚 {result['roll_deg']:+.2f}° 俯仰 {result['pitch_deg']:+.2f}°"
                                f"（站姿实测 右倾 {result['lean_right']:+.1f}° 前倾 {result['lean_fwd']:+.1f}°，"
                                f"应为 {result['ref_right']:+.1f}° {result['ref_fwd']:+.1f}°），手动校正已归零")

    # ------------------------------------------------------------------ velocity / input sources

    def sources(self) -> dict[str, Any]:
        with self._lock:
            lock = self._velocity_lock
            now = time.time()
            owner = lock.source if lock and lock.fresh(now) else None
            return {
                "current": owner,
                "sources": [
                    {**s.__dict__, "active": s.name == owner} for s in sorted(self._sources.values(), key=lambda s: -s.priority)
                ],
            }

    def register_source(
        self, name: str, priority: int, timeout_ms: int, description: str = "", head_front: bool = False
    ) -> dict[str, Any]:
        name = (name or "").strip()
        if not name or len(name) > 32:
            raise ApiError("输入源名称为 1–32 个字符", CommonState.INVALID)
        existing = self._sources.get(name)
        if existing and existing.builtin:
            raise ApiError(f"{name} 是系统内置输入源，不能修改", CommonState.INVALID, Reason.NOT_SUPPORTED)
        if not 0 <= int(priority) <= 59:
            raise ApiError("自定义输入源优先级范围 0–59（内置：rc/interaction 80，app_proxy 60）", CommonState.INVALID)
        if not 100 <= int(timeout_ms) <= 5000:
            raise ApiError("超时范围 100–5000 ms", CommonState.INVALID)
        with self._lock:
            self._sources[name] = InputSource(
                name, int(priority), int(timeout_ms), description or "SDK 自定义输入源", head_front=bool(head_front)
            )
        return response(self.sources(), message=f"已注册输入源 {name}")

    def delete_source(self, name: str) -> dict[str, Any]:
        with self._lock:
            source = self._sources.get(name)
            if source is None:
                raise ApiError(f"没有输入源 {name}", CommonState.INVALID, Reason.NOT_FOUND)
            if source.builtin:
                raise ApiError("内置输入源不能删除", CommonState.INVALID, Reason.NOT_SUPPORTED)
            del self._sources[name]
        return response(self.sources())

    def velocity(self, source: str, vx: Any, vy: Any, vtheta: Any, head_yaw: Any = None) -> dict[str, Any]:
        """head_yaw [deg, + left]: for a head_front source, where the walk points the head. The
        reply's head_yaw is the angle applied, null when the head is not this source's to turn
        (another source, the head or upper body moves run, or not walking)."""
        now = time.time()
        with self._lock:
            src = self._sources.get(source)
            if src is None or src.name == "rc":
                raise ApiError(f"输入源 {source} 未注册（先调用 SetMcInputSource）", CommonState.INVALID, Reason.NOT_FOUND)
            if self._actual not in WALKING:
                raise ApiError(
                    f"当前是「{ACTIONS[self._actual].label}」，速度指令只在走跑模式生效", CommonState.INVALID, Reason.ACTION
                )
            lock = self._velocity_lock
            if lock and lock.fresh(now) and lock.source != source and src.priority <= lock.priority:
                raise ApiError(
                    f"输入源 {lock.source}（优先级 {lock.priority}）正在控制", CommonState.FAILURE, Reason.INPUT_SOURCE_BUSY
                )
            changed = lock is None or lock.source != source or not lock.fresh(now)
            velocity = {"vx": clamp_unit(vx), "vy": clamp_unit(vy), "vtheta": clamp_unit(vtheta)}
            aim = None
            if _number(head_yaw) and src.head_front and self._actual == WALK and not self._head:
                aim = round(max(-HEAD_AIM_DEG, min(HEAD_AIM_DEG, float(head_yaw))), 2)
            self._velocity_lock = VelocityLock(source, src.priority, src.timeout_ms / 1000.0, now, velocity, aim)
            # The follow walks on FOLLOW_AGENT_INDEX, not the free-arms walk: out of 「走跑·上肢自由」
            # as it takes over, unless the arms are busy (a hug, a gesture, led by the hand...).
            to_walk = (changed and source == FOLLOW_SOURCE and self._actual == WALK_UPPER
                       and self._target == WALK_UPPER and self._upper_idle())
        if changed:
            self.events.publish("input_source", owner=source)
        if to_walk:
            try:
                self.set_action(WALK, source=source)
            except ApiError as exc:
                self.events.alert("warning", f"跟随没能切回「走跑」：{exc.message}")
        return response({"source": source, "velocity": velocity, "head_yaw": aim})

    def _upper_idle(self) -> bool:
        cmd = self._upper_cmd
        live = self._status.get("upper") or {}
        return (cmd.get("hug") is None and not cmd.get("lead") and not cmd.get("limp") and not cmd.get("manual")
                and not self._teach and self._sequence is None and not live.get("gesture"))

    # ------------------------------------------------------------------ timed walk (assistant tools)

    def _cancel_walk_task(self) -> None:
        if self._walk_task is not None:
            task_id, cancel = self._walk_task
            cancel.set()
            self.tasks.finish(task_id, CommonState.ABORTED, "被新指令取消")
            self._walk_task = None

    def walk_for(self, vx: float, vy: float, vtheta: float, duration_s: float, source: str = "interaction") -> dict[str, Any]:
        duration_s = max(0.3, min(10.0, float(duration_s)))
        if self._actual == STAND:
            self.set_action(WALK, source=source)
            deadline = time.time() + 4.0
            while time.time() < deadline and self._actual != WALK:
                time.sleep(0.1)
        if self._actual not in WALKING:
            raise ApiError(
                f"机器人在「{ACTIONS[self._actual].label}」，需要先站立并进入走跑", CommonState.INVALID, Reason.ACTION
            )
        self.velocity(source, vx, vy, vtheta)
        with self._lock:
            self._cancel_walk_task()
            cancel = threading.Event()
            task_id = self.tasks.create("walk", vx=vx, vy=vy, vtheta=vtheta, duration_s=duration_s, source=source)
            self._walk_task = (task_id, cancel)

        def run() -> None:
            end = time.time() + duration_s
            try:
                while time.time() < end and not cancel.is_set():
                    self.velocity(source, vx, vy, vtheta)
                    cancel.wait(0.1)
                self.velocity(source, 0.0, 0.0, 0.0)
                self.tasks.finish(task_id, CommonState.SUCCESS if not cancel.is_set() else CommonState.ABORTED)
            except ApiError as exc:
                self.tasks.finish(task_id, CommonState.ABORTED, exc.message)

        threading.Thread(target=run, name="xiaoxiaobai-api-walk", daemon=True).start()
        return response({"duration_s": duration_s}, state=CommonState.RUNNING, task_id=task_id, message=f"走 {duration_s:.1f} 秒")

    # ------------------------------------------------------------------ sequences (voice, assistant)

    def _cancel_sequence(self) -> None:
        if self._sequence is not None:
            self._sequence[1].set()
            self._sequence = None

    def run_sequence(self, steps: Any, source: str = "interaction") -> dict[str, Any]:
        """Walks and gestures one after the other: [{"walk": {vx, vy, vtheta, seconds}},
        {"gesture": name or taught move (id or name)}, {"keys": [keyframes], "exact": bool}].
        Gestures play under 「走跑·上肢自由」: from 站立 or 走跑 the robot goes there for them and
        back to where it was at the end."""
        plan: list[tuple[str, Any]] = []
        for step in steps[:SEQUENCE_MAX_STEPS] if isinstance(steps, list) else []:
            if not isinstance(step, dict):
                continue
            taught = self.custom.find(str(step["gesture"])) if isinstance(step.get("gesture"), str) else None
            if isinstance(step.get("walk"), dict):
                walk = step["walk"]
                try:
                    seconds = max(0.3, min(10.0, float(walk.get("seconds", 2.0))))
                except (TypeError, ValueError):
                    continue
                plan.append(("walk", (clamp_unit(walk.get("vx")), clamp_unit(walk.get("vy")),
                                      clamp_unit(walk.get("vtheta")), seconds)))
            elif step.get("gesture") in (*gestures.GESTURE_NAMES, "random"):
                plan.append(("gesture", step["gesture"]))
            elif taught is not None:
                plan.append(("keys", (taught["keys"], True, taught["name"])))
            elif step.get("exact") and gestures.custom_keys(step.get("keys"), gestures.TAUGHT_MAX_KEYS, gestures.TAUGHT_MAX_S):
                plan.append(("keys", (step["keys"], True, str(step.get("label") or "") or None)))
            elif gestures.custom_keys(step.get("keys")):
                plan.append(("keys", (gestures.custom_keys(step["keys"]), False, None)))
        if not plan:
            raise ApiError("没有可执行的动作", CommonState.INVALID)
        previous = self._actual
        if previous not in (STAND, WALK, WALK_UPPER):
            raise ApiError(f"机器人在「{ACTIONS[previous].label}」，先启动并站稳再做动作", CommonState.INVALID, Reason.ACTION)
        with self._lock:
            self._cancel_sequence()
            cancel = threading.Event()
            task_id = self.tasks.create("motion_sequence", steps=[kind for kind, _ in plan], source=source)
            self._sequence = (task_id, cancel)
        threading.Thread(target=self._run_sequence, args=(plan, previous, task_id, cancel, source),
                         name="xiaoxiaobai-api-sequence", daemon=True).start()
        return response({"steps": [kind for kind, _ in plan]}, state=CommonState.RUNNING, task_id=task_id,
                        message=f"开始执行 {len(plan)} 个动作")

    def _run_sequence(self, plan: list[tuple[str, Any]], previous: str, task_id: int, cancel: threading.Event,
                      source: str) -> None:
        state, message, seq = CommonState.SUCCESS, "", None
        try:
            for kind, arg in plan:
                if cancel.is_set():
                    break
                if kind == "walk":
                    self.walk_for(*arg, source=source)
                    cancel.wait(arg[3] + 0.4)
                    continue
                if kind == "keys":
                    keys, exact, label = arg
                    seq = self.play_gesture(gestures.CUSTOM, 1.0, source, keys=keys, exact=exact, label=label)["data"]["seq"]
                else:
                    seq = self.play_gesture(arg, 1.0, source)["data"]["seq"]
                self._wait_gesture(seq, cancel)
        except ApiError as exc:
            state, message = CommonState.FAILURE, exc.message
        if cancel.is_set():
            state, message = CommonState.ABORTED, "被新指令取消"
        # Back to the mode it started in, unless something else took over in the meantime.
        elif (previous in (STAND, WALK) and self._actual in (WALK, WALK_UPPER) and self._actual != previous
              and (seq is None or self._upper_cmd["seq"] == seq)):
            try:
                self.set_action(previous, source=source)
            except ApiError as exc:
                message = exc.message
        self.tasks.finish(task_id, state, message)
        with self._lock:
            if self._sequence is not None and self._sequence[0] == task_id:
                self._sequence = None

    def _wait_gesture(self, seq: int, cancel: threading.Event) -> None:
        """Until the control loop has played gesture `seq` out (its status names what plays)."""
        start = time.time()
        seen = False
        keys = self._upper_cmd.get("keys") or []
        longest = GESTURE_WAIT_S + sum(k["dur"] + k["hold"] for k in keys)
        while time.time() - start < longest and not cancel.is_set():
            if self._actual != WALK_UPPER:
                raise ApiError(f"动作被打断：机器人在「{ACTIONS[self._actual].label}」", CommonState.ABORTED, Reason.ACTION)
            live = self._status.get("upper") or {}
            if live.get("seq") == seq:
                if live.get("gesture"):
                    seen = True
                elif seen or time.time() - start > GESTURE_START_S:
                    return
            cancel.wait(0.1)

    # ------------------------------------------------------------------ upper body

    UPPER_IDS = range(13, 21)

    def upper_body(self, targets_deg: dict[str, Any]) -> dict[str, Any]:
        if self._actual != CUSTOM_UPPER:
            raise ApiError("先切换到「上肢控制」模式", CommonState.INVALID, Reason.ACTION)
        targets: dict[int, float] = {}
        for key, raw in (targets_deg or {}).items():
            try:
                motor_id, value = int(key), float(raw)
            except (TypeError, ValueError):
                continue
            if motor_id not in self.UPPER_IDS or not math.isfinite(value):
                continue
            lo, hi = UPPER_MOTOR_LIMITS_DEG[motor_id]
            if not lo - 0.5 <= value <= hi + 0.5:
                raise ApiError(f"ID {motor_id} 超出 {lo:.0f}°..{hi:.0f}°：{value:.1f}°", CommonState.INVALID)
            targets[motor_id] = math.radians(min(max(value, lo), hi))
        if not targets:
            raise ApiError("没有可下发的 13–20 号目标角", CommonState.INVALID)
        self._cancel_stream()
        self.backend.send_upper(targets)
        with self._lock:
            self._upper_targets.update(targets)
        return response({"targets_rad": targets})

    # ------------------------------------------------------------------ gestures (walk + free upper body)

    def gesture_catalog(self) -> dict[str, Any]:
        joints = []
        for motor_id, (key, name) in sorted(UPPER_KEYS.items()):
            lo, hi = gestures.JOINT_LIMITS_DEG[name]
            rest = math.degrees(gestures.to_joints(gestures.REST)[name])
            joints.append({"motor_id": motor_id, "name": name, "min_deg": lo, "max_deg": hi, "rest_deg": round(rest, 1)})
        return {"gestures": gestures.catalog(), "joints": joints, "mode": WALK_UPPER}

    def _upper_state(self) -> dict[str, Any]:
        cmd = self._upper_cmd
        manual = {}
        for motor_id, (key, name) in UPPER_KEYS.items():
            if key in cmd["manual"]:
                pose = {**gestures.REST, key: cmd["manual"][key]}
                manual[str(motor_id)] = round(math.degrees(gestures.to_joints(pose)[name]), 1)
        live = (self._status.get("upper") or {}) if self._actual == WALK_UPPER else {}
        lead = ((self._status.get("walk") or {}).get("lead") or {}) if self._actual == WALK_UPPER else {}
        return {"auto": cmd["auto"], "manual_deg": manual, "playing": live.get("gesture"), "scale": live.get("scale"),
                "weight": live.get("weight"), "limp": bool(live.get("limp")), "groove": live.get("groove") or 0.0,
                "lead": bool(cmd.get("lead")), "lead_active": bool(live.get("lead")),
                "lead_vx": lead.get("vx"), "lead_vtheta": lead.get("vtheta"),
                "hug": cmd.get("hug") is not None, "hug_phase": live.get("hug"),
                "groove_move": live.get("groove_move"), "groove_section": live.get("groove_section")}

    def set_groove(self, grid: dict[str, Any] | None) -> None:
        """The beat grid of the music playing (music.py), for the upper body to sway to in
        「走跑·上肢自由」 (gestures.Groove); None stops the sway."""
        with self._lock:
            if grid:
                self._upper_cmd["groove"] = {**grid, "ts": time.time()}
            else:
                self._upper_cmd.pop("groove", None)

    def _enter_walk_upper(self, source: str) -> None:
        if self._actual == WALK_UPPER:
            return
        if self._actual not in (STAND, WALK):
            raise ApiError(
                f"机器人在「{ACTIONS[self._actual].label}」，先站立后再做动作", CommonState.INVALID, Reason.ACTION
            )
        self.set_action(WALK_UPPER, source=source)
        deadline = time.time() + 6.0
        while time.time() < deadline and self._actual != WALK_UPPER:
            time.sleep(0.1)
        if self._actual != WALK_UPPER:
            raise ApiError("切换到「走跑·上肢自由」超时", CommonState.TIMEOUT, Reason.ACTION)

    def play_gesture(self, name: str, size: Any = 1.0, source: str = "app_proxy", keys: Any = None,
                     exact: bool = False, label: str | None = None) -> dict[str, Any]:
        """A gesture by name, "random", or gestures.CUSTOM made of `keys` (gestures.custom_keys);
        `exact` plays the keys as taught (longer moves allowed, no human variation)."""
        name = str(name or "").strip()
        if name == gestures.CUSTOM:
            keys = gestures.custom_keys(keys, gestures.TAUGHT_MAX_KEYS, gestures.TAUGHT_MAX_S) if exact else gestures.custom_keys(keys)
            if not keys:
                raise ApiError("自编动作没有可用的关键帧", CommonState.INVALID)
        elif name not in gestures.GESTURES and name != "random":
            raise ApiError(f"没有动作 {name}", CommonState.INVALID, Reason.NOT_FOUND)
        else:
            keys = None
        try:
            size = max(0.3, min(1.0, float(size)))
        except (TypeError, ValueError):
            size = 1.0
        self._enter_walk_upper(source)
        with self._lock:
            cmd = self._upper_cmd
            if cmd.get("limp"):
                self._end_walk_teach()
            cmd["lead"] = False
            self._end_hug()
            cmd.update(seq=cmd["seq"] + 1, gesture=name, gesture_ts=time.time(), size=size, keys=keys, exact=bool(exact))
            seq = cmd["seq"]
        label = label or {"random": "随机动作", gestures.CUSTOM: "自编动作"}.get(name) or gestures.GESTURES[name].label
        self.events.publish("gesture", name=name, label=label, source=source)
        return response({"gesture": name, "label": label, "size": size, "seq": seq}, message=f"做动作：{label}")

    def stop_gesture(self) -> dict[str, Any]:
        with self._lock:
            self._cancel_sequence()
            self._cancel_stream()
            cmd = self._upper_cmd
            cmd.update(seq=cmd["seq"] + 1, gesture="stop", gesture_ts=time.time())
        return response(None, message="动作已停止")

    def upper_settings(self, auto: Any = None, manual_deg: Any = None, release: Any = None) -> dict[str, Any]:
        """auto: idle gestures on/off; manual_deg: {motor_id: joint deg} to hold; release: motor ids or "all"."""
        with self._lock:
            if self._actual != WALK_UPPER and manual_deg:
                raise ApiError("先切换到「走跑·上肢自由」模式", CommonState.INVALID, Reason.ACTION)
            cmd = self._upper_cmd
            if auto is not None:
                cmd["auto"] = bool(auto)
            manual = dict(cmd["manual"])
            if release == "all":
                manual = {}
            elif isinstance(release, list):
                for motor_id in release:
                    key = UPPER_KEYS.get(int(motor_id), (None, None))[0]
                    manual.pop(key, None)
            for raw_id, raw in (manual_deg or {}).items():
                try:
                    motor_id, value = int(raw_id), float(raw)
                except (TypeError, ValueError):
                    continue
                if motor_id not in UPPER_KEYS or not math.isfinite(value):
                    continue
                key, name = UPPER_KEYS[motor_id]
                lo, hi = gestures.JOINT_LIMITS_DEG[name]
                if not lo <= value <= hi:
                    raise ApiError(f"ID {motor_id} 超出范围 {lo:.0f}…{hi:.0f}°：{value:.1f}°", CommonState.INVALID)
                manual[key] = gestures.from_joints({name: math.radians(value)})[key]
            cmd["manual"] = manual
            return response(self._upper_state())

    # ------------------------------------------------------------------ move editor (taught moves)
    # Poses are in the gestures' authoring space (gestures.POSE_KEYS, degrees, the same for both arms).

    def editor_info(self) -> dict[str, Any]:
        labels = {"rp": "右肩前举", "rr": "右肩侧抬", "re": "右肘弯曲", "lp": "左肩前举", "lr": "左肩侧抬",
                  "le": "左肘弯曲", "yaw": "头左右（+左）", "tilt": "头上下（+抬头）"}
        joints = [{"key": key, "label": labels[key], "min": round(gestures.POSE_LIMITS[key][0], 1) + 0.0,
                   "max": round(gestures.POSE_LIMITS[key][1], 1) + 0.0, "rest": gestures.REST[key]} for key in gestures.POSE_KEYS]
        return {
            "joints": joints,
            "moves": self.custom.list(),
            "limits": {"max_keys": gestures.TAUGHT_MAX_KEYS, "max_s": gestures.TAUGHT_MAX_S,
                       "dur_s": gestures.CUSTOM_DUR_S, "hold_s": gestures.CUSTOM_HOLD_S},
            "eases": [["io", "平滑"], ["out", "快起慢停"], ["in", "慢起快停"], ["back", "冲过头再回"], ["lin", "匀速"]],
        }

    @staticmethod
    def _motor_to_pose(motor_rad: dict[int, float]) -> dict[str, float]:
        joints = {name: motor_rad[motor_id] * _SIGN[name] for motor_id, (_key, name) in UPPER_KEYS.items() if motor_id in motor_rad}
        return gestures.from_joints(joints)

    @staticmethod
    def _pose_to_motor(pose: dict[str, float]) -> dict[int, float]:
        joints = gestures.to_joints({**gestures.REST, **pose})
        return {motor_id: joints[name] * _SIGN[name] for motor_id, (_key, name) in UPPER_KEYS.items()}

    def upper_pose(self) -> dict[str, Any]:
        """The pose the upper body is in: measured when the motors are limp (teach) or the loop
        reads them back, otherwise what was last commanded."""
        actual = self._actual
        source = None
        pose = dict(gestures.REST)
        if actual == CUSTOM_UPPER:
            state = self.backend.upper_state() or {}
            motor = dict(self._upper_targets)
            if state.get("running", True):
                for raw_id, value in (state.get("positions") or {}).items():
                    try:
                        motor[int(raw_id)] = float(value)
                    except (TypeError, ValueError):
                        continue
            if motor:
                pose = self._motor_to_pose(motor)
                source = "measured" if self._teach else "commanded"
        elif actual in LOOP_ACTIONS:
            live = self._status.get("joints") or {}
            joints = {name: float(live[name][0]) for name in gestures.UPPER_JOINTS if live.get(name) and live[name][0] is not None}
            if joints:
                pose = gestures.from_joints(joints)
                source = "measured"
        return {
            "pose": {key: round(value, 1) + 0.0 for key, value in pose.items()},
            "source": source,
            "mode": actual,
            "teach": self._teach,
        }

    def set_upper_pose(self, pose: Any) -> dict[str, Any]:
        """Hold the upper body in `pose` (any subset of gestures.POSE_KEYS): 「上肢控制」 sends the
        motor targets, 「走跑·上肢自由」 holds them as manual targets over the walk."""
        clean: dict[str, float] = {}
        for key, value in (pose or {}).items() if isinstance(pose, dict) else ():
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if key in gestures.POSE_LIMITS and math.isfinite(number):
                clean[key] = min(max(number, gestures.POSE_LIMITS[key][0]), gestures.POSE_LIMITS[key][1])
        if not clean:
            raise ApiError("没有可用的关节角", CommonState.INVALID)
        with self._lock:
            actual = self._actual
            if actual == CUSTOM_UPPER:
                self._cancel_stream()
                current = self.upper_pose()["pose"] if len(clean) < len(gestures.POSE_KEYS) else {}
                targets = self._pose_to_motor({**current, **clean})
                self.backend.send_upper(targets)
                self._upper_targets.update(targets)
            elif actual == WALK_UPPER:
                self._upper_cmd["manual"] = {**self._upper_cmd["manual"], **clean}
            else:
                raise ApiError("先进入「上肢控制」（零力矩下，腿不出力）或「走跑·上肢自由」", CommonState.INVALID, Reason.ACTION)
        return response({"pose": clean, "mode": actual})

    def release_pose(self) -> dict[str, Any]:
        """Hand the joints held by the editor back (「走跑·上肢自由」: gestures and idle life again)."""
        with self._lock:
            self._upper_cmd["manual"] = {}
        return response(None, message="已松开")

    def set_teach(self, enabled: bool) -> dict[str, Any]:
        """13-20 go limp so the arms and head can be moved by hand (their angles are read back for
        the editor); off holds them where they were left. 「走跑·上肢自由」: the legs keep
        balancing (the free-arms walk policy does not use the arms). 「上肢控制」: torque off,
        the legs are limp too."""
        enabled = bool(enabled)
        with self._lock:
            actual = self._actual
            if actual == WALK_UPPER:
                self._cancel_sequence()
                if enabled:
                    self._upper_cmd["lead"] = False
                    self._end_hug()
                    self._upper_cmd["limp"] = True
                    self._teach = True
                else:
                    self._end_walk_teach()
            elif actual == CUSTOM_UPPER:
                self._cancel_stream()
                self.backend.send_upper({}, limp=TEACH_IDS if enabled else [])
                self._teach = enabled
            else:
                raise ApiError("手动示教要在「走跑·上肢自由」（腿自己保持平衡）或「上肢控制」模式下用",
                               CommonState.INVALID, Reason.ACTION)
        message = "手臂和头已放松，可以用手掰到想要的姿势" if enabled else "已锁定在当前姿势"
        return response({"teach": self._teach}, message=message)

    def set_lead(self, enabled: bool, source: str = "app_proxy") -> dict[str, Any]:
        """Lead by the hand (moves/lead.py): in 「走跑·上肢自由」 the arms go soft and the walk
        follows them, arm pulled forward / back: walk forward / back, lifted out sideways: turn
        that way. Off: the arms stiffen and go back to rest. From 站立 / 走跑 it switches over."""
        if enabled:
            self._enter_walk_upper(source)
        with self._lock:
            if self._actual != WALK_UPPER:
                if enabled:
                    raise ApiError("牵手走要在「走跑·上肢自由」模式下用", CommonState.INVALID, Reason.ACTION)
                self._upper_cmd["lead"] = False
                return response({"lead": False})
            cmd = self._upper_cmd
            if enabled:
                self._cancel_sequence()
                cmd.update(seq=cmd["seq"] + 1, gesture="stop", gesture_ts=time.time())
                cmd["limp"] = False
                self._teach = False
                cmd["manual"] = {}
                self._end_hug()
            cmd["lead"] = bool(enabled)
        self.events.publish("lead", enabled=bool(enabled), source=source)
        message = ("手臂已放软：拉着手往前走它就跟着走，往后推就后退，把手臂往侧上方抬就原地往那边转；"
                   "松手后手臂慢慢回到原位，它也慢慢停下"
                   if enabled else "牵手走已关闭，手臂回到自然姿态")
        return response({"lead": bool(enabled)}, message=message)

    def set_hug(self, enabled: bool, source: str = "app_proxy") -> dict[str, Any]:
        """Hug the leg in front (the app's 跟屁虫 once its camera is covered): in 「走跑·上肢自由」
        the arms close round it and hold on (gestures.Hug), the head nuzzling side to side and
        looking up, with faces to match. From 站立 / 走跑 it switches over; off lets go and goes
        back to the mode it came from."""
        before = self._actual
        if enabled:
            self._enter_walk_upper(source)
        with self._lock:
            cmd = self._upper_cmd
            if enabled:
                if self._actual != WALK_UPPER:
                    raise ApiError("拥抱要在「走跑·上肢自由」模式下用", CommonState.INVALID, Reason.ACTION)
                if cmd.get("hug") is None:
                    self._hug_from = before if before in (STAND, WALK) else None
                    self._hug_face = self._gesture_face or self._expression()
                    self._hug_shown = None
                    self._cancel_sequence()
                    cmd.update(seq=cmd["seq"] + 1, gesture="stop", gesture_ts=time.time())
                    cmd["limp"] = cmd["lead"] = False
                    self._teach = False
                    cmd["manual"] = {}
                    cmd["hug"] = {"seed": random.randrange(1, 1 << 30)}
            else:
                back = self._hug_from if cmd.get("hug") is not None else None
                self._end_hug()
                if back is not None and self._actual == WALK_UPPER and self._target == WALK_UPPER:
                    self.set_action(back, source=source)
        self.events.publish("hug", enabled=bool(enabled), source=source)
        return response({"hug": bool(enabled)}, message="抱住啦" if enabled else "松开了")

    def _end_hug(self) -> None:
        if self._upper_cmd.get("hug") is None:
            return
        self._upper_cmd["hug"] = None
        if self._hug_face:
            self._show_face(self._hug_face)
        self._hug_from = self._hug_face = self._hug_shown = None

    def _hug_expression(self, actual: str) -> None:
        """A new face with each phase of the hug (moves/upper_body.py status), picked at random."""
        if self._upper_cmd.get("hug") is None or actual != WALK_UPPER:
            return
        live = self._status.get("upper") or {}
        shown = (live.get("hug"), live.get("hug_cycles"))
        if shown[0] not in HUG_FACES or shown == self._hug_shown:
            return
        self._hug_shown = shown
        faces = [face for face in HUG_FACES[shown[0]] if face != self._expression()]
        self._show_face(random.choice(faces or HUG_FACES[shown[0]]))

    def _move_face(self, actual: str) -> str | None:
        """The face the move playing asks for: a gesture's (moves/upper_body.py status "face"),
        or that of the section a chill clip is in (CHILL_FACES)."""
        if actual == WALK_UPPER:
            return (self._status.get("upper") or {}).get("face")
        if actual in CHILLS:
            chill = self._status.get(CHILLS[actual]) or {}
            if chill.get("phase") in ("easing", "playing", "done"):
                return CHILL_FACES.get(chill.get("section"))
        return None

    def _gesture_expression(self, actual: str) -> None:
        """The faces a gesture or a chill clip names as it plays (_move_face); once it is over,
        the face from before, unless someone picked another one meanwhile."""
        face = self._move_face(actual)
        shown = self._gesture_shown
        if face == shown:
            return
        self._gesture_shown = face
        if face is not None:
            if shown is None:
                self._gesture_face = self._expression()
            self._show_face(face, "gesture")
            return
        if self._gesture_face and self._expression() == shown:
            self._show_face(self._gesture_face, "gesture")
        self._gesture_face = None

    def _expression(self) -> str | None:
        try:
            from expressions import read_expression_name

            return read_expression_name(self.backend.paths.expression)
        except Exception:  # noqa: BLE001 - no face to come back to
            return None

    def _show_face(self, name: str, source: str = "hug") -> None:
        try:
            from expressions import EXPRESSION_BY_NAME, write_expression_name

            write_expression_name(name, self.backend.paths.expression)
            self.events.publish("expression", name=name, label=EXPRESSION_BY_NAME[name].label, source=source)
        except Exception as exc:  # noqa: BLE001 - the move goes on without its face
            self.events.alert("warning", f"{'拥抱' if source == 'hug' else '动作'}表情没能切换：{exc}")

    def _end_walk_teach(self) -> None:
        """Stiffen 13-20 again in 「走跑·上肢自由」, holding the pose the hands left (as manual
        targets, until released or a move plays)."""
        held = self.upper_pose()["pose"] if self._actual == WALK_UPPER else {}
        self._upper_cmd["manual"] = held
        self._upper_cmd["limp"] = False
        self._teach = False

    def save_custom(self, body: dict[str, Any]) -> dict[str, Any]:
        move = self.custom.save(body.get("name"), body.get("keys"), body.get("id") or None)
        self.events.publish("custom_gestures", action="save", id=move["id"], name=move["name"])
        return response(move, message=f"已保存「{move['name']}」")

    def delete_custom(self, move_id: str) -> dict[str, Any]:
        move = self.custom.delete(move_id)
        self.events.publish("custom_gestures", action="delete", id=move_id, name=move["name"])
        return response(None, message=f"已删除「{move['name']}」")

    def play_custom(self, name_or_id: str | None = None, keys: Any = None, source: str = "app_proxy") -> dict[str, Any]:
        """A taught move (by id or name), or unsaved `keys` (the editor's preview), as recorded.
        「上肢控制」 plays it from here; 站立 / 走跑 / 走跑·上肢自由 through the control loop."""
        if name_or_id:
            move = self.custom.find(name_or_id)
            if move is None:
                raise ApiError(f"没有叫「{name_or_id}」的自定义动作", CommonState.INVALID, Reason.NOT_FOUND)
            keys, label = move["keys"], move["name"]
        else:
            label = "动作预览"
        keys = gestures.custom_keys(keys, gestures.TAUGHT_MAX_KEYS, gestures.TAUGHT_MAX_S)
        if not keys:
            raise ApiError("没有可播放的关键帧", CommonState.INVALID)
        actual = self._actual
        if actual == CUSTOM_UPPER:
            return self._stream_custom(keys, label, source)
        if actual in (STAND, WALK, WALK_UPPER):
            result = self.run_sequence([{"keys": keys, "exact": True, "label": label}], source=source)
            return response(result["data"], state=CommonState.RUNNING, task_id=result["task_id"], message=f"做动作：{label}")
        raise ApiError(f"机器人在「{ACTIONS[actual].label}」：先站立，或进入「上肢控制」", CommonState.INVALID, Reason.ACTION)

    def _cancel_stream(self) -> None:
        if self._streaming is not None:
            self._streaming[1].set()
            self._streaming = None

    def _stream_custom(self, keys: list[dict], label: str, source: str) -> dict[str, Any]:
        with self._lock:
            if self._teach:
                self.backend.send_upper({}, limp=[])
                self._teach = False
            self._cancel_stream()
            cancel = threading.Event()
            self._streaming = (label, cancel)
            start = self.upper_pose()["pose"]
            task_id = self.tasks.create("custom_gesture", name=label, source=source)
        performance = gestures.Performance(gestures.custom_gesture(keys, exact=True), start, random.Random(), 1.0)

        def run() -> None:
            t0 = time.monotonic()
            state = CommonState.SUCCESS
            while not cancel.is_set():
                t = time.monotonic() - t0
                if self._actual != CUSTOM_UPPER:
                    state = CommonState.ABORTED
                    break
                targets = self._pose_to_motor(performance.pose(t, start))
                self.backend.send_upper(targets)
                self._upper_targets.update(targets)
                if performance.done(t):
                    break
                cancel.wait(STREAM_PERIOD_S)
            if cancel.is_set():
                state = CommonState.ABORTED
            self.tasks.finish(task_id, state, "" if state == CommonState.SUCCESS else "被打断")
            with self._lock:
                if self._streaming is not None and self._streaming[1] is cancel:
                    self._streaming = None

        threading.Thread(target=run, name="xiaoxiaobai-api-custom-gesture", daemon=True).start()
        self.events.publish("gesture", name=gestures.CUSTOM, label=label, source=source)
        return response({"name": label, "seconds": round(performance.duration, 2)}, state=CommonState.RUNNING,
                        task_id=task_id, message=f"做动作：{label}")
