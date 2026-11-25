# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

import os
from pathlib import Path

import numpy as np


def _find_motor_serial_port() -> str:
    override = os.environ.get("XIAOXIAOBAI_SERIAL_PORT")
    if override:
        return override
    boards = sorted(Path("/dev/serial/by-id").glob("usb-ROBOTIS_OpenRB-150_*-if00"))
    return str(boards[0]) if boards else "/dev/ttyACM0"


# Any servo board, so swapping the board needs no code change; XIAOXIAOBAI_SERIAL_PORT overrides.
MOTOR_SERIAL_PORT = _find_motor_serial_port()

# Left/right swap test: motors 2-6 now sit in the left leg and 8-12 in the right leg.
# Original mapping: right 1-6, left 7-12.
MOTOR_TO_ID = {
    "right_hip_yaw": 1,
    "right_hip_roll": 8,
    "right_hip_pitch": 9,
    "right_knee": 10,
    "right_ankle_pitch": 11,
    "right_ankle_roll": 12,
    "left_hip_yaw": 7,
    "left_hip_roll": 2,
    "left_hip_pitch": 3,
    "left_knee": 4,
    "left_ankle_pitch": 5,
    "left_ankle_roll": 6,
    "right_shoulder_pitch": 13,
    "right_shoulder_roll": 14,
    "right_elbow": 15,
    "left_shoulder_pitch": 16,
    "left_shoulder_roll": 17,
    "left_elbow": 18,
    "head": 19,
    "head_pitch": 20,
}

ID_TO_MOTOR = {v: k for k, v in MOTOR_TO_ID.items()}

# Same as HOME_FRAME in mjlab_xiaoxiaobai (robot/xiaoxiaobai_constants.py).
NEUTRAL_POSE = {
    "left_hip_yaw": float(np.deg2rad(0.0)),
    "left_hip_roll": float(np.deg2rad(-3.5)),
    "left_hip_pitch": float(np.deg2rad(0.0)),
    "left_knee": float(np.deg2rad(0.0)),
    "left_ankle_pitch": float(np.deg2rad(0.0)),
    "left_ankle_roll": float(np.deg2rad(3.5)),
    "right_hip_yaw": float(np.deg2rad(0.0)),
    "right_hip_roll": float(np.deg2rad(3.5)),
    "right_hip_pitch": float(np.deg2rad(0.0)),
    "right_knee": float(np.deg2rad(0.0)),
    "right_ankle_pitch": float(np.deg2rad(0.0)),
    "right_ankle_roll": float(np.deg2rad(-3.5)),
    "left_shoulder_pitch": float(np.deg2rad(0.0)),
    "left_shoulder_roll": float(np.deg2rad(-5.0)),
    "left_elbow": float(np.deg2rad(0.0)),
    "right_shoulder_pitch": float(np.deg2rad(0.0)),
    "right_shoulder_roll": float(np.deg2rad(5.0)),
    "right_elbow": float(np.deg2rad(0.0)),
    "head": float(np.deg2rad(0.0)),
    "head_pitch": float(np.deg2rad(0.0)),
}

# Swap test: the pitch motors (3-5, 9-11) turn the other way in the opposite leg, so their
# signs are flipped. Original: hip_pitch -1, knee +1, ankle_pitch +1 on both legs.
MOTOR_SIGN = {
    "left_hip_yaw": 1.0,
    "left_hip_roll": 1.0,
    "left_hip_pitch": 1.0,
    "left_knee": -1.0,
    "left_ankle_pitch": -1.0,
    "left_ankle_roll": -1.0,
    "right_hip_yaw": 1.0,
    "right_hip_roll": 1.0,
    "right_hip_pitch": 1.0,
    "right_knee": -1.0,
    "right_ankle_pitch": -1.0,
    "right_ankle_roll": -1.0,
    "left_shoulder_pitch": 1.0,
    "left_shoulder_roll": 1.0,
    "left_elbow": -1.0,
    "right_shoulder_pitch": 1.0,
    "right_shoulder_roll": 1.0,
    "right_elbow": -1.0,
    "head": 1.0,
    "head_pitch": 1.0,
}

# Position P Gain (servo register value)
KP_DEFAULT: int = 400        # ~0.886 Nm/rad in MuJoCo
KP_RL: int = 125             # ~0.277 Nm/rad in MuJoCo
KP_GROUND: int = 200         # prop up and chill clips on the ground; GETUP_KP in mjlab_xiaoxiaobai (robot/xiaoxiaobai_constants.py)
KP_UPPER: int = 300          # arms and head while the upper body plays gestures under the walk
KP_GAIN_PRM: float = 0.0022  # Nm/rad per register unit (for the servo)

# BAM motor model (bam package)
BAM_VIN: float = 7.5
BAM_VIN_MIN: float = 6.0
BAM_VOLTAGE_DROP_GAIN: float = 0.2
BAM_MAX_CURRENT: float = 1.75 # servo firmware current limit [A]: clips motor torque to ±BAM_MAX_CURRENT * kt

# Overcurrent safety: emergency torque-off when the summed |present_current| of all
# motors stays above OVERCURRENT_CUTOFF_A for OVERCURRENT_DEBOUNCE_TICKS consecutive ticks.
# Goal: cut the robot before a current spike (e.g. all motors snapping during a fall) trips the BMS.
#
# 2S pack, so pack current is the cell current. Datasheet: 15 A continuous without
# a cell-temperature sensor, 30 A only with an 80 °C surface cut (this robot has neither),
# and the pack's discharge protection must operate below 30 A. 19 servos clipped at 1.75 A
# could ask for 33 A, which is already past that. The get-up sim peaks at about 14 A (p90),
# and walking was tuned to stay under the old 15 A cutoff. 20 A sits above that get-up peak
# with margin for the current proxy reading high, and 10 A under the 30 A cell limit.
# It is a 40 ms spike detector, not a licence to hold 20 A: there is no 80 °C cut on the pack.
PRESENT_CURRENT_UNIT_A: float = 0.001   # servo present_current register unit (1.0 mA/LSB)
OVERCURRENT_CUTOFF_A: float = 20.0
OVERCURRENT_DEBOUNCE_TICKS: int = 2     # consecutive over-threshold ticks before cutting

# Current proxy used when present_current is NOT read (Observer.observe_current = False), so the
# safety needs no extra bus transaction. Reproduces the bam servo voltage-controlled model from
# data already read (present_position, present_velocity) and the command target:
#   duty = clip(PROXY_KP * PROXY_ERROR_GAIN * (target - q), ±PROXY_MAX_PWM)
#   I    = (PROXY_VIN * duty - PROXY_KT * dq) / PROXY_R      then |I| capped at BAM_MAX_CURRENT
PROXY_KT: float = 0.366                  # servo torque constant [Nm/A]
PROXY_R: float = 2.811                   # servo motor resistance [Ohm]
PROXY_VIN: float = BAM_VIN               # supply voltage [V]
PROXY_ERROR_GAIN: float = 0.0028773775   # duty cycle per (kp * rad), servo encoder/gain scaling
PROXY_MAX_PWM: float = 1.0               # max duty cycle magnitude
PROXY_KP: int = KP_RL                    # firmware P gain assumed by the proxy when no move sets one
OVERCURRENT_PROXY_DELAY_TICKS: int = 3   # number of ticks to delay the proxy current estimate

# How big arm gestures may get under each walk policy (moves/upper_body.py), by its file.
# MuJoCo + BAM, 60 s runs with random commands and a gesture every 2.5-4 s: the walks fell on
# arms overhead above 0.6 (cheer, stretch). Skate is untested.
UPPER_BODY_SCALE: dict[str, float] = {
    "walk_low_battery_1.onnx": 0.6,
    "walk_low_battery_2.onnx": 0.6,
    "walk_free_arms.onnx": 1.0,
    "skate_glide.onnx": 0.5,
    "skate_single_leg.onnx": 0.5,
}
UPPER_BODY_SCALE_UNKNOWN: float = 0.5

# Head travel [deg] (head = motor 19, yaw: + turns left; head_pitch = motor 20). The MJCF has
# no head_pitch joint: the walk policy neither sees nor drives it.
HEAD_LIMITS_DEG: dict[str, tuple[float, float]] = {"head": (-40.0, 40.0), "head_pitch": (-10.0, 30.0)}

# Velocity command limits [m/s, m/s, rad/s], applied centrally to every input source.
# Input sources emit normalized commands in [-1, 1]; scale_velocity() maps them to these.
# Rotation gets a wider range when turning in place (vx = vy = 0) than while translating.
# Equal to the ranges the walk policy was trained on (xiaoxiaobai_velocity_env_cfg.py): commands
# beyond them are outside the training distribution.
VX_MAX: float = 0.5
VX_MAX_BACKWARD: float = 0.5
VY_MAX: float = 0.3
VTHETA_MAX_STATIONARY: float = 1.5  # rotation-only envs: rotation_env_ang_vel_range
VTHETA_MAX_MOVING: float = 0.75     # ranges.ang_vel_z
# The walk / skate policies get the velocity command ramped: from 0 to an axis' full speed
# in VELOCITY_RAMP_UP_S, from full speed back to 0 in VELOCITY_RAMP_DOWN_S (a step would
# throw the robot's weight around).
VELOCITY_RAMP_UP_S: float = 1.0
VELOCITY_RAMP_DOWN_S: float = 0.8

# The app's camera follow (跟随 / 跟屁虫, input source FOLLOW_SOURCE) walks on FOLLOW_WALK_AGENT
# whichever of WALK_AGENTS is selected, kept FOLLOW_LINGER_S after its last command so it is not
# swapped mid-stop, with the softer S-shaped ramp below. The app stops with a plain 0 when it
# hugs, loses the robot or is stopped: this is what keeps those stops gentle.
FOLLOW_SOURCE: str = "follow"
FOLLOW_WALK_AGENT: str = "walk_low_battery_1.onnx"
FOLLOW_LINGER_S: float = 3.0
# Per axis (vx, vy, vtheta): 0 to full speed / full speed to 0 [s]. vtheta keeps the plain ramp's
# times, which the app's heading model assumes (FollowLogic.kt ROBOT_RAMP_*).
SOFT_RAMP_UP_S: tuple[float, float, float] = (2.0, 2.0, VELOCITY_RAMP_UP_S)
SOFT_RAMP_DOWN_S: tuple[float, float, float] = (1.2, 1.2, VELOCITY_RAMP_DOWN_S)
SOFT_RAMP_JERK_S: float = 0.25  # the acceleration builds up (and lets go) over this
SOFT_RAMP_K: float = 8.0        # 1/s: how the acceleration eases off onto the target speed
# Against the plain ramp (sim of the command alone): 0 -> 0.3 m/s in 1.4 s at 0.25 m/s^2 at most
# (0.6 s, 0.5), 0.3 m/s -> 0 in 0.9 s over 14 cm (0.5 s, 7 cm), 0.1 m/s -> 0 over 2.3 cm (0.7 cm,
# the creep into a hug), turning 0 -> 0.7 rad/s in 0.66 s (0.48 s).

# IMU I2C bus number on the controller
IMU_I2C_BUS: int = 1

# Rotation from trunk frame (body) to IMU sensor frame
IMU_MOUNT_QUAT: tuple[float, float, float, float] = (0.5, -0.5, -0.5, 0.5)

# Fine correction of the IMU mount [deg], applied to the projected gravity fed to the policy.
# Positive roll makes the policy see the trunk leaning further right, so it rights the robot
# toward the left; positive pitch makes it see the trunk leaning further back, so it rights the
# robot forward. Adjustable live with the keyboard ([a]/[d] roll, [w]/[s] pitch, [z] resets).
IMU_ROLL_CORRECTION_DEG: float = 0.0
IMU_PITCH_CORRECTION_DEG: float = 0.0

# IMU mount calibration [deg], same convention, applied under the corrections above: what makes
# the IMU read the true trunk attitude. Measured by imu_calib.py (web UI 走路调校 IMU 安装标定:
# the robot standing still in 「位控站立」 at NEUTRAL_POSE on flat ground, its IMU against the
# trunk attitude of both soles flat at the measured leg joints, from the kinematics,
# posture_ref.json "neutral") and saved on the robot.
IMU_MOUNT_ROLL_DEG: float = 0.0
IMU_MOUNT_PITCH_DEG: float = 0.0
IMU_CALIB_S: float = 3.0
IMU_CALIB_MAX_GYRO: float = 0.15       # rad/s: someone is holding or pushing the robot
IMU_CALIB_MAX_JOINT_DEG: float = 4.0   # leg joints this far from NEUTRAL_POSE: not standing in the pose
IMU_CALIB_MAX_DEG: float = 10.0        # a larger mount error is a fall or a slope, not the mount

# Supply-voltage compensation of the walk's KP. The servo sets its PWM duty from KP times the
# position error, so its torque scales with KP * supply voltage (the BAM model the policies were
# trained on), and the policies cannot see the voltage: as the battery runs down the legs sag and
# the trunk drifts into a lean (sim: trunk pitch -1.8 deg at 7.5 V, +4.2 at 5.5 V).
# While walking, the walk writes KP_RL * KP_VIN_REF / V instead, V the motors' supply voltage
# (scheduler, read every VOLTAGE_PERIOD_S and smoothed), so the legs stay as stiff as at
# KP_VIN_REF over the whole battery. Standing still (KP_VIN_STILL_S after the command stops) it
# never raises KP above KP_RL: stiffer standing legs shook on the robot. The scale is clipped to
# KP_VIN_SCALE_RANGE and rewritten only when KP_RL changes by KP_VIN_MIN_STEP register units.
# KP_VIN_COMP is the default of the switch (web UI 走路调校 电压补偿 KP, saved on the robot).
KP_VIN_COMP: bool = True
KP_VIN_REF: float = 7.0
KP_VIN_SCALE_RANGE: tuple[float, float] = (0.85, 1.25)
KP_VIN_MIN_STEP: int = 2
KP_VIN_STILL_S: float = 0.5

# Automatic trim on top of the IMU corrections while the walk runs (tilt_trim.py). The trunk
# attitude (IMU with the mount calibration) averaged over IMU_TRIM_WINDOW_S of steady walking:
# - roll is held at the attitude the same policy keeps in sim at that speed (src/agents/
#   posture_ref.json, src/debug/posture_ref.py; about upright): the side lean the policy leaves;
# - pitch at the one the robot walked with at that speed over its first IMU_TRIM_BASELINE_WINDOWS
#   windows since the loop started: the drift only (the robot's pitch differs from the sim's by
#   several degrees, which forcing on it tipped it over).
# Each window moves the correction by IMU_TRIM_GAIN of the error divided by how much the policy
# turns a correction into lean (posture_ref.json gain_*), at most IMU_TRIM_MAX_STEP_DEG, within
# IMU_TRIM_MAX_*_DEG. Windows swaying more than IMU_TRIM_MAX_STD_DEG (stumbles, pushes) are
# dropped; an axis pinned at its limit for IMU_TRIM_GIVE_UP_WINDOWS windows is reset and left
# alone (the robot does not follow it); a walk start fades the trim in over IMU_TRIM_FADE_S.
# IMU_AUTO_TRIM is the default of the switch (web UI 走路调校 IMU 自动修正, saved on the robot);
# switching it off clears the trim.
IMU_AUTO_TRIM: bool = True
IMU_TRIM_WINDOW_S: float = 2.0
IMU_TRIM_STEADY_S: float = 1.0  # the command must not have changed for this long
IMU_TRIM_GAIN: float = 0.25
IMU_TRIM_MAX_STEP_DEG: float = 0.3
IMU_TRIM_MAX_ROLL_DEG: float = 3.0
IMU_TRIM_MAX_PITCH_DEG: float = 2.0
IMU_TRIM_BASELINE_WINDOWS: int = 2
IMU_TRIM_MAX_STD_DEG: float = 6.0
IMU_TRIM_GIVE_UP_WINDOWS: int = 3
IMU_TRIM_FADE_S: float = 2.0

# Walk policies in src/agents, selectable live in the web UI (运动 → 走路调校 → 走路策略) and with
# the keyboard ([c]/[n]). Both drive legs and arms, lean back walking forward and forward walking
# backward, and were trained on supply voltages 5-8 V.
# 低电量步态1 (walk_low_battery_1.onnx, Mjlab-Velocity-Xiaoxiaobai iteration 36995, XIAOXIAOBAI_STYLE=3):
# a natural gait, knees kept bent above 7 deg walking, shoulder roll capped (14 deg walking, 11.5
# standing), with 30% of the walking commands reversing vx and/or turning every 1-2.5 s. It stands
# with the knees bent 13 deg. Sim at 7.2 V: arm swing (std) 8.4 / 9.4 deg at vx +0.4 / -0.4, foot
# lift 20-27 mm, trunk pitch -6.5 / +5.3 deg; falls at 5.0 V: 6 % forward, 15 % backward, 1 % standing.
# 低电量步态2 (walk_low_battery_2.onnx, iteration 38494, XIAOXIAOBAI_DISCHARGE=1): 低电量步态1 fine-tuned
# 1500 iterations more with half the episodes running the battery down mid-walk, from 6-8 V by
# 0.5 V or more to 5.0-5.5 V over 5-20 s. Sim, 7.6 -> 5.0 V over 90 s with random commands: falls
# 10.0 -> 8.4 %. It stands straighter (knees 7.5 deg) and at 5.0 V creeps backward less.
WALK_AGENTS: list[tuple[str, str]] = [
    ("低电量步态1", "walk_low_battery_1.onnx"),
    ("低电量步态2", "walk_low_battery_2.onnx"),
]
WALK_AGENT_DEFAULT: int = 0

# The walk policy while the upper body is free (move "upper", web UI 走跑·上肢自由), whichever
# of WALK_AGENTS is selected: it drives the legs (1-12) only.
WALK_UPPER_AGENT: str = "walk_free_arms.onnx"

# Walk policies run on the legs (1-12) only: their arm observations are zeroed and their arm
# actions dropped, the arms rest at KP_UPPER or play gestures at full size (moves/upper_body.py).
# walk_free_arms.onnx is 低电量步态1 fine-tuned on Mjlab-Velocity-Xiaoxiaobai-FreeArms with the same
# settings (5-8 V, POSTURE=3, STYLE=3 without its arm terms), 30% of the episodes led by the hand
# (XIAOXIAOBAI_HAND_LEAD=0.3: arms limp, one or both forearms pulled by a spring toward a point
# walking at the command, up to 0.5-2.5 N, let go and taken again, the arms easing back to rest at
# KP 100), iteration 39000. Sim, 1024 envs over 20 s, falls at 7.5 / 6.0 / 5.0 V: 7.5 / 7.0 / 8.2 %
# under gestures (walk_voltage_eval.py), 10.2 / 7.1 / 14.7 % led by the hand (hand_lead_eval.py).
FREE_ARMS_AGENTS: tuple[str, ...] = (WALK_UPPER_AGENT,)

# Skating on the roller-skate shoes (Mjlab-Skate-Xiaoxiaobai in mjlab_xiaoxiaobai). Same reference
# pose, observation layout and KP as the walk. Its commands were trained forward only and
# without side steps: vx in [0, SKATE_VX_MAX], vy = 0, |vtheta| <= SKATE_VTHETA_MAX.
# Selectable live in the web UI (运动 → 走路调校 → 滑板策略), like WALK_AGENTS.
# 双脚滑行 (skate_glide.onnx, Mjlab-Skate-Xiaoxiaobai iteration 14250): both skates on the floor,
# trained on supply voltages down to 5.5 V. Sim, vx 0.3, 512 envs: 0.25 m/s at 7.5 V and 5.5 V,
# 0 % falls, standing at 5.5 V 0 % falls.
# 单脚蹬滑 (skate_single_leg.onnx, Mjlab-Skate-Stroke-Xiaoxiaobai iteration 44999): one skate pushes
# while the other glides alone, the arms swing in mirror; it sets its own stroke rate (one stroke
# 1.0-2.0 s, slower on free floors) and was trained on supply voltages 5-8 V. Sim, vx 0.3,
# 512 envs, 0 % falls: 0.26 m/s at 7.5 V, 0.25 m/s at 5.0 V; +-0.4 rad/s turns +-0.27-0.29.
# On a floor freer than trained (tile, bearings 1e-4 N m) it overshoots: 0.5 m/s for 0.3.
SKATE_AGENTS: list[tuple[str, str]] = [
    ("双脚滑行", "skate_glide.onnx"),
    ("单脚蹬滑", "skate_single_leg.onnx"),
]
SKATE_AGENT_DEFAULT: int = 0
SKATE_VX_MAX: float = 0.5
SKATE_VTHETA_MAX: float = 0.5
# Skate heading hold: the policy drifts off the commanded yaw rate (veers when sent straight),
# so while skating the IMU yaw tracks a heading reference integrated from the user's vtheta
# (frozen when going straight) and the error is fed back into the vtheta given to the policy.
SKATE_HEADING_HOLD: bool = True
SKATE_HEADING_KP: float = 1.5          # vtheta [rad/s] per rad of heading error
SKATE_HEADING_KI: float = 0.6          # vtheta trim [rad/s] per rad.s: learns the policy's steady veer
SKATE_HEADING_MAX_TRIM: float = 0.3    # rad/s
SKATE_HEADING_MAX_ERROR: float = 0.35  # rad; the reference is dragged along past it (anti-windup)
SKATE_HEADING_MIN_VX: float = 0.05     # m/s of ramped vx under which the hold is released

# Motion tracking moves (moves/dance.py): a policy with the walk's reference pose and KP, fed its
# reference frame by frame from <agent>_ref.npz (mjlab_xiaoxiaobai scripts/export_dance.py writes
# both). The reference starts and ends in the reference pose.
# A soundtrack on the move's own beats: after DANCE_MUSIC_PAD_S of silence its t = 0 is the
# reference's frame 0, so the API starts the move that long after the first sample is heard, less
# the control loop's latency to the first step.
DANCE_MUSIC_PAD_S: float = 0.5
DANCE_START_LATENCY_S: float = 0.03
# Radio calisthenics (广播体操, Mjlab-Gymnastics-Xiaoxiaobai: a routine choreographed on Xiaoxiaobai,
# mjlab_xiaoxiaobai scripts/gymnastics_motion.py): the same move and API with gymnastics.onnx, its
# reference gymnastics_ref.npz and its soundtrack on the routine's counts (109 BPM). The reference
# also turns the head, which no policy drives: DanceMove plays its head_pos on the head motor,
# DANCE_HEAD_LEAD_FRAMES ahead like the training. Run 2026-11-10_18-27-03_gym_v1 iteration 59000.
GYMNASTICS_AGENT: str = "gymnastics.onnx"
GYMNASTICS_MUSIC: str = "gymnastics_music.wav"
DANCE_HEAD_LEAD_FRAMES: int = 4
# Sneaky peek (偷看, Mjlab-Peek-Xiaoxiaobai, mjlab_xiaoxiaobai scripts/peek_motion.py): the same move
# and API with peek.onnx and its reference peek_ref.npz, about 15 s from the stand. It leans out
# to the left on its own feet (nothing to hold on to) and turns the head like the gymnastics.
# No soundtrack. Run 2026-11-12_10-24-57_peek_v3_lean iteration 79498.
PEEK_AGENT: str = "peek.onnx"
# Prop up / lie down (撑起/躺下, Mjlab-PropUp-Xiaoxiaobai, mjlab_xiaoxiaobai scripts/propup_motion.py):
# from lying on the back, up onto the elbows and hands and back down, tracked by propup.onnx
# (reference propup_ref.npz) at KP_GROUND. The reference lies still until 0.5 s, is up from 3.5 s,
# holds still until 5.35 s, lies down and is still again from 8.5 s to its end (= frame 0). Up
# plays it to PROPUP_HOLD_S and holds that frame, down plays on to the end (moves/propup.py).
# Run 2026-11-11_00-07-32_propup_fast iteration 5000.
PROPUP_AGENT: str = "propup.onnx"
PROPUP_HOLD_S: float = 5.2
PROPUP_UP_S: float = 3.5  # status reports up from there (to PROPUP_HOLD_S)
PROPUP_DOWN: str = "propup_down"  # in active_moves: lie down (no move of that name)
PROPUP_MIN_SUPINE_GX: float = 0.8  # starts only lying on the back: projected gravity x below -this
# Chill (躺着/坐地/坐凳休闲, mjlab_xiaoxiaobai scripts/chill_motion.py, chill_export.py): joint
# loops played open loop (no policy) at KP_GROUND (moves/chill.py). Move name -> (clip in
# src/agents, projected gravity x range it starts in): lying on the back, or sitting leaned back
# 9-53 deg (the clips lean 24 and 28 deg). Prop up also starts from that lean, at its hold frame.
CHILL_CLIPS: dict[str, tuple[str, float, float]] = {
    "chill_lie": ("chill_lie.npz", -1.1, -0.8),
    "chill_sit": ("chill_sit.npz", -0.8, -0.15),
    "chill_chair": ("chill_chair.npz", -0.8, -0.15),
    # 起床伸懒腰 (chill_motion.py wake, chill_export.py --once): played once lying on the back,
    # holding its last frame (the first one again) at the end.
    "wake_stretch": ("wake_stretch.npz", -1.1, -0.8),
}
# Clips played stiffer than KP_GROUND: the shiver of wake_stretch (3.5-5 Hz) is mostly filtered
# out at KP 200 (chill_sim.py: ~10 % of the command, ~40 % at 400; trunk still within 5 deg).
CHILL_KP: dict[str, int] = {"wake_stretch": KP_DEFAULT}
CHILL_MAX_GY: float = 0.35  # |projected gravity y| it starts in: not tipped onto a side
CHILL_SETTLE: str = "chill_settle"  # in active_moves: ease back to the clip's first pose, then hold
CHILL_EASE_DEG_S: float = 45.0  # mean speed of the ease from the measured pose (and of the settle)
CHILL_EASE_MIN_S: float = 1.0

# Observation DoF ordering
OBSERVATION_DOF_ORDER = [
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_elbow",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_elbow",
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll"
]