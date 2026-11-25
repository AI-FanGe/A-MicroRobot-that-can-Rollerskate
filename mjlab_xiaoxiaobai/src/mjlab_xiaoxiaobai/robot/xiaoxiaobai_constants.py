# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

import json
import os
import re
import numpy as np
from dataclasses import replace
from pathlib import Path

import mujoco
from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

XIAOXIAOBAI_XML: Path = Path(os.path.dirname(__file__)) / "xiaoxiaobai" / "robot.xml"
assert XIAOXIAOBAI_XML.exists(), f"XML not found: {XIAOXIAOBAI_XML}"

# XIAOXIAOBAI_BASE=24996: the model and supply range the deployed model_24996 was trained on.
_BASE_24996 = os.environ.get("XIAOXIAOBAI_BASE") == "24996"
# XIAOXIAOBAI_VIN_LOW=<V>: supply voltages down to <V> (and load sag 0.5 V below). Trained down to
# 7.0 V with a 6.0 V floor, the walk leaned ~5 deg forward at 5.5 V and fell in 13% of walks
# in sim (41% at 5.0 V); on the robot it leaned and fell as the pack ran low.
_VIN_LOW = float(os.environ["XIAOXIAOBAI_VIN_LOW"]) if os.environ.get("XIAOXIAOBAI_VIN_LOW") else None


# XIAOXIAOBAI_BACKPACK=1: the charging backpack of modelforRLV7.3dm (a 100 g battery in a 30 g bag)
# on the trunk's back, from robot/xiaoxiaobai/backpack.json (scripts/rhino3dm_backpack.py).
_BACKPACK = os.environ.get("XIAOXIAOBAI_BACKPACK") == "1"
_BACKPACK_PARTS = json.loads((XIAOXIAOBAI_XML.parent / "backpack.json").read_text(encoding="utf-8"))["parts"]
BACKPACK_MASS = sum(p["mass"] for p in _BACKPACK_PARTS) if _BACKPACK else 0.0


def _add_backpack(spec: mujoco.MjSpec) -> None:
    pack = json.loads((XIAOXIAOBAI_XML.parent / "backpack.json").read_text(encoding="utf-8"))
    trunk = next(b for b in spec.bodies if b.name == "trunk")
    i = trunk.fullinertia
    bodies = [(trunk.mass, np.array(trunk.ipos), np.array([[i[0], i[3], i[4]], [i[3], i[1], i[5]], [i[4], i[5], i[2]]]))]
    bodies += [(p["mass"], np.array(p["com"]), np.array(p["inertia"])) for p in pack["parts"]]
    mass = sum(m for m, _, _ in bodies)
    com = sum(m * c for m, c, _ in bodies) / mass
    inertia = sum(j + m * (np.dot(c - com, c - com) * np.eye(3) - np.outer(c - com, c - com)) for m, c, j in bodies)
    trunk.mass, trunk.ipos = mass, com
    trunk.fullinertia = [inertia[0, 0], inertia[1, 1], inertia[2, 2], inertia[0, 1], inertia[0, 2], inertia[1, 2]]
    for p in pack["parts"]:
        spec.add_mesh(name=f"{p['mesh']}_mesh", file=f"{p['mesh']}.obj")
        trunk.add_geom(name=f"{p['mesh']}_visual", default=spec.find_default("visual"), type=mujoco.mjtGeom.mjGEOM_MESH,
                       meshname=f"{p['mesh']}_mesh", rgba=[0.2, 0.35, 0.6, 1.0])
    trunk.add_geom(name="backpack_collision", default=spec.find_default("collision"), type=mujoco.mjtGeom.mjGEOM_BOX,
                   pos=pack["collision"]["pos"], size=pack["collision"]["size"])


def get_spec(backpack: bool = _BACKPACK) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(XIAOXIAOBAI_XML))
    if _BASE_24996:
        for joint in [j for j in spec.joints if j.name.startswith("leg_flex")]:
            spec.delete(joint)
    if backpack:
        _add_backpack(spec)
    return spec

# Ready pose: every motor at its neutral position except hip/ankle roll, which spread
# the legs outward by the same angle so the feet stay flat, and shoulder roll.
_ROLL_SPREAD_DEG = 3.5
# Shoulder roll neutral is the inner end of its 0-45 deg range, 2.5 deg from the hard
# stop. Resting there, the policy learned to press the arms into the stop with ever
# larger commands (thousands of degrees), which the motor register cannot represent.
_SHOULDER_ROLL_OUT_DEG = 5.0

HOME_FRAME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.181),
    joint_pos={
        "head": float(np.deg2rad(0.0)),
        "left_shoulder_pitch": float(np.deg2rad(0.0)),
        "left_shoulder_roll": float(np.deg2rad(-_SHOULDER_ROLL_OUT_DEG)),
        "left_elbow": float(np.deg2rad(0.0)),
        "right_shoulder_pitch": float(np.deg2rad(0.0)),
        "right_shoulder_roll": float(np.deg2rad(_SHOULDER_ROLL_OUT_DEG)),
        "right_elbow": float(np.deg2rad(0.0)),
        "left_hip_yaw": float(np.deg2rad(0.0)),
        "left_hip_roll": float(np.deg2rad(-_ROLL_SPREAD_DEG)),
        "left_hip_pitch": float(np.deg2rad(0.0)),
        "left_knee": float(np.deg2rad(0.0)),
        "left_ankle_pitch": float(np.deg2rad(0.0)),
        "left_ankle_roll": float(np.deg2rad(_ROLL_SPREAD_DEG)),
        "right_hip_yaw": float(np.deg2rad(0.0)),
        "right_hip_roll": float(np.deg2rad(_ROLL_SPREAD_DEG)),
        "right_hip_pitch": float(np.deg2rad(0.0)),
        "right_knee": float(np.deg2rad(0.0)),
        "right_ankle_pitch": float(np.deg2rad(0.0)),
        "right_ankle_roll": float(np.deg2rad(-_ROLL_SPREAD_DEG)),
    },
    joint_vel={r".*": 0.0},
)

FULL_COLLISION = CollisionCfg(
    geom_names_expr=(r".*_collision",),
    condim={r"^(left|right)_foot_collision$": 3, r".*_collision": 1},
    priority={r"^(left|right)_foot_collision$": 1},
    friction={r"^(left|right)_foot_collision$": (1.0,)},
)

from bam.mjlab import BamActuatorCfg

actuators = BamActuatorCfg(
    motor_name="xl330",
    model="m6",
    target_names_expr=(r"^(?!head_wobble|leg_flex).*$",),
    kp_fw=125,
    # Down to a nearly flat 2S pack: on the robot the legs sagged further under load than in sim.
    vin_range=(_VIN_LOW, 8.0) if _VIN_LOW else (7.0, 8.0) if _BASE_24996 else (6.4, 8.0),
    vin_drop_gain_range=(0.0, 0.2),
    vin_min=_VIN_LOW - 0.5 if _VIN_LOW else 6.0,
    max_current=1.75,
    delay_min_lag=2,
    delay_max_lag=8,
)

# -- Old actuator (XML position, MuJoCo default) --
# actuators = XmlActuatorCfg(
#     target_names_expr=(r".*",),
#     delay_min_lag=0,
#     delay_max_lag=3,
# )

XIAOXIAOBAI_ROBOT_CFG = EntityCfg(
    spec_fn=get_spec,
    init_state=HOME_FRAME,
    collisions=(FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(actuators,),
        # The soft limits are the working ranges (dof_pos_limits penalises leaving them);
        # robot.xml widens its ranges by 1/0.9 so they land exactly there.
        soft_joint_pos_limit_factor=0.9,
    ),
)

# -- Free-arms walk (Mjlab-Velocity-Xiaoxiaobai-FreeArms) --
# The policy drives the legs only; the arms play gestures at KP_UPPER over the ranges set on
# the robot (xiaoxiaobai gestures.JOINT_LIMITS_DEG): shoulder pitch +-150, roll 0..160 outward,
# elbow 0..85 bent. The joints get a few degrees beyond them, so a target at the end of a
# range does not sit on the stop.
FREE_ARM_RANGES_DEG: dict[str, tuple[float, float]] = {
    "right_shoulder_pitch": (-150.0, 150.0),
    "right_shoulder_roll": (0.0, 160.0),
    "right_elbow": (-85.0, 0.0),
    "left_shoulder_pitch": (-150.0, 150.0),
    "left_shoulder_roll": (-160.0, 0.0),
    "left_elbow": (0.0, 85.0),
}
FREE_ARM_MARGIN_DEG = 3.0
ARM_JOINTS = r"^(left|right)_(shoulder_pitch|shoulder_roll|elbow)$"
FREE_ARMS_KP = 300  # xiaoxiaobai KP_UPPER
# Forearm bodies and the forearm tip in their frames, where a hand leads the robot.
FOREARMS = {"right": "radius", "left": "radius_2"}
HAND_SITE = (0.0, 0.0, -0.069)


def get_free_arms_spec() -> mujoco.MjSpec:
    spec = get_spec()
    for joint in spec.joints:
        if joint.name in FREE_ARM_RANGES_DEG:
            lo, hi = FREE_ARM_RANGES_DEG[joint.name]
            joint.range = np.deg2rad((lo - FREE_ARM_MARGIN_DEG, hi + FREE_ARM_MARGIN_DEG))
    for side, body_name in FOREARMS.items():
        body = next(b for b in spec.bodies if b.name == body_name)
        body.add_site(name=f"{side}_hand", pos=list(HAND_SITE))
    return spec


XIAOXIAOBAI_FREE_ARMS_ROBOT_CFG = replace(
    XIAOXIAOBAI_ROBOT_CFG,
    spec_fn=get_free_arms_spec,
    articulation=replace(
        XIAOXIAOBAI_ROBOT_CFG.articulation,
        actuators=(
            replace(actuators, target_names_expr=(r"^(?!head_wobble|leg_flex|left_shoulder|left_elbow|right_shoulder|right_elbow).*$",)),
            replace(actuators, target_names_expr=(ARM_JOINTS,), kp_fw=FREE_ARMS_KP),
        ),
    ),
)

# -- Get-up task only (the walk keeps robot.xml) --
# Mechanical stops measured on the robot with xiaoxiaobai's joint-limit slider (real angles,
# calibration/getup_joint_limits.json), mirrored to the other side; shoulder roll keeps
# robot.xml's inner stop. They replace the robot.xml ranges, so the 0.9 soft limits sit
# inside the stops.
GETUP_JOINT_RANGES_DEG: dict[str, tuple[float, float]] = {
    "left_ankle_pitch": (-59.4, 37.6),
    "right_ankle_pitch": (-37.6, 59.4),
    "left_shoulder_pitch": (-159.3, 135.4),
    "right_shoulder_pitch": (-135.4, 159.3),
    "left_shoulder_roll": (-137.4, 2.5),
    "right_shoulder_roll": (-2.5, 137.4),
}


# -- Dance (Mjlab-Dance-Xiaoxiaobai) --
# Same motors as the walk; the arms use their working range on the robot (inside the measured
# stops of GETUP_JOINT_RANGES_DEG and the gesture ranges) instead of robot.xml's +-50 deg.
# Widened by 1/0.9 like robot.xml, so the 0.9 soft limits land on these ranges.
DANCE_ARM_RANGES_DEG: dict[str, tuple[float, float]] = {
    "right_shoulder_pitch": (-120.0, 145.0),
    "left_shoulder_pitch": (-145.0, 120.0),
    "right_shoulder_roll": (0.0, 125.0),
    "left_shoulder_roll": (-125.0, 0.0),
    "right_elbow": (-85.0, 0.0),
    "left_elbow": (0.0, 85.0),
}


def get_dance_spec() -> mujoco.MjSpec:
    spec = get_spec()
    for joint in spec.joints:
        if joint.name in DANCE_ARM_RANGES_DEG:
            lo, hi = np.deg2rad(DANCE_ARM_RANGES_DEG[joint.name])
            mid, half = (lo + hi) / 2, (hi - lo) / 2 / 0.9
            joint.range = (mid - half, mid + half)
    return spec


XIAOXIAOBAI_DANCE_ROBOT_CFG = replace(XIAOXIAOBAI_ROBOT_CFG, spec_fn=get_dance_spec)


def _fibonacci_directions(n: int) -> np.ndarray:
    i = np.arange(n)
    y = 1 - 2 * (i + 0.5) / n
    radius = np.sqrt(np.maximum(0.0, 1 - y * y))
    phi = np.pi * (1 + 5**0.5) * i
    return np.stack([radius * np.cos(phi), y, radius * np.sin(phi)], axis=1)


def _support_hull(points: np.ndarray, directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convex hull of the support points of ``points`` in ``directions``.

    The hull uses only points of the mesh, so it never adds volume outside the part.
    96 directions leaves the real surface a couple of millimetres outside."""
    from scipy.spatial import ConvexHull

    support = points[(points @ directions.T).argmax(axis=0)]
    support = np.unique(np.round(support, 5), axis=0)
    hull = ConvexHull(support)
    remap = {int(old): new for new, old in enumerate(hull.vertices)}
    faces = np.array([[remap[int(i)] for i in simplex] for simplex in hull.simplices], dtype=np.int32)
    return support[hull.vertices], faces


def _quat_rotate(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=float)
    vec = np.atleast_2d(np.asarray(vec, dtype=float))
    w, xyz = quat[0], quat[1:]
    turned = 2 * np.cross(xyz, vec)
    return np.squeeze(vec + w * turned + np.cross(xyz, turned))


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


_FOOT_SOLE = r"^(left|right)_foot_collision$"


def _add_ground_hulls(spec: mujoco.MjSpec) -> None:
    """Add ``<body>_ground_collision``: the convex hull of all visual meshes (shell and motors)
    of each body that has a collision box.

    Against a flat floor a body only ever touches its convex hull, so the hull is the exact
    ground shape. It must not touch the other links: the hull fills concave pockets (the trunk
    hull swallows the hips by ~47 mm at the ready pose), so self-contact stays on the boxes.
    GETUP_COLLISION sets the contact bits that keep the two apart."""
    model = spec.compile()
    directions = _fibonacci_directions(96)
    for body in spec.bodies:
        if not any(g.name.endswith("_collision") for g in body.geoms):
            continue
        points = []
        for visual in [g for g in body.geoms if g.name.endswith("_visual") and g.meshname]:
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, visual.name)
            mesh_id = int(model.geom_dataid[geom_id])
            adr, n = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
            local = np.asarray(model.mesh_vert[adr : adr + n])
            pos = np.asarray(model.geom_pos[geom_id])
            points.append(pos + _quat_rotate(np.asarray(model.geom_quat[geom_id]), local))
        if not points:
            continue
        points = np.concatenate(points)
        # Feet: keep the hull above the sole box, so standing touches only the walk's sole.
        sole = next((g for g in body.geoms if re.fullmatch(_FOOT_SOLE, g.name)), None)
        if sole is not None:
            points = points[points[:, 2] > sole.pos[2] + sole.size[2]]
        verts, faces = _support_hull(points, directions)
        mesh = spec.add_mesh()
        mesh.name = f"{body.name}_ground_hull"
        mesh.uservert = verts.reshape(-1)
        mesh.userface = faces.reshape(-1)
        geom = body.add_geom()
        geom.name = f"{body.name}_ground_collision"
        geom.type = mujoco.mjtGeom.mjGEOM_MESH
        geom.meshname = mesh.name
        geom.group = 3
    # spec.geom(name) only finds geoms named after creation once the spec is compiled again,
    # and CollisionCfg looks them up by name.
    spec.compile()


# Weighed on the robot once wired. The model's parts add up to 1.072 kg; the wiring runs
# through the whole body, so every body's mass and inertia are scaled up alike.
GETUP_TOTAL_MASS = 1.133


def get_getup_spec() -> mujoco.MjSpec:
    spec = get_spec()
    for joint in spec.joints:
        if joint.name in GETUP_JOINT_RANGES_DEG:
            joint.range = np.deg2rad(GETUP_JOINT_RANGES_DEG[joint.name])
    scale = GETUP_TOTAL_MASS / spec.compile().body_subtreemass[1]
    for body in spec.bodies[1:]:
        body.mass *= scale
        body.inertia = body.inertia * scale
        body.fullinertia = body.fullinertia * scale
    _add_ground_hulls(spec)
    return spec


# Contact bits (the floor is contype 1 / conaffinity 1):
#   *_ground_collision  mesh hulls, floor only        contype 1, conaffinity 0
#   foot soles          the walk's boxes, floor + self contype 3, conaffinity 2
#   other boxes         self-contact only             contype 2, conaffinity 2
# The soles stay the walk's boxes so standing and the hand-over match the walk exactly.
# Getting up pushes on the ground with the trunk, arms and knees, so every floor contact
# gets friction (condim 3) instead of only the feet.
GETUP_COLLISION = CollisionCfg(
    geom_names_expr=(r".*_collision",),
    contype={r".*_ground_collision$": 1, _FOOT_SOLE: 3, r".*_collision": 2},
    conaffinity={r".*_ground_collision$": 0, r".*_collision": 2},
    condim={r".*_collision": 3},
    # Above the terrain's 0, so the randomised robot friction is the contact friction rather
    # than max(robot, terrain = 1.0).
    priority={_FOOT_SOLE: 1, r".*_ground_collision$": 1},
    friction={_FOOT_SOLE: (1.0,), r".*_collision": (0.6,)},
)

# Same peak torque (current limit) as the walk's 125, reached at ~100 deg of command lead
# instead of ~160: the joints hold the support poses instead of sagging. The robot switches
# kp when it hands over between the get-up and walk policies.
GETUP_KP = 200

# -- Drag (Mjlab-Drag-Xiaoxiaobai) --
# Walking backward pulling a load taped to both hands, the arms straight out in front.
# robot.xml holds shoulder pitch to the walk's +-50 deg; the motors turn +-150
# (FREE_ARM_RANGES_DEG). Forward is + on the right shoulder, - on the left. The arms hold
# this pose (the policy drives the legs only); the right side's angles, mirrored on the left.
DRAG_SHOULDER_PITCH_DEG = 90.0
DRAG_SHOULDER_ROLL_DEG = 0.0
DRAG_ELBOW_DEG = 0.0
_DRAG_SHOULDER_PITCH_RANGE_DEG = (-50.0, 100.0)  # right; working range, as soft limits
# The forearm tip in the radius / radius_2 frames: the lowest point of the hanging arm.
DRAG_HAND_SITE = (0.0, 0.0, -0.069)


def get_drag_spec() -> mujoco.MjSpec:
    spec = get_spec()
    lo, hi = np.deg2rad(_DRAG_SHOULDER_PITCH_RANGE_DEG)
    # Widened by 1/0.9 so the soft limits land on the working range, as robot.xml.
    centre, half = (lo + hi) / 2, (hi - lo) / 2 / 0.9
    for joint in spec.joints:
        if joint.name == "right_shoulder_pitch":
            joint.range = [centre - half, centre + half]
        elif joint.name == "left_shoulder_pitch":
            joint.range = [-centre - half, -centre + half]
    for body_name, side in (("radius", "right"), ("radius_2", "left")):
        body = next(b for b in spec.bodies if b.name == body_name)
        body.add_site(name=f"{side}_hand", pos=list(DRAG_HAND_SITE))
    return spec


DRAG_HOME_FRAME = replace(
    HOME_FRAME,
    joint_pos={
        **HOME_FRAME.joint_pos,
        "right_shoulder_pitch": float(np.deg2rad(DRAG_SHOULDER_PITCH_DEG)),
        "left_shoulder_pitch": float(np.deg2rad(-DRAG_SHOULDER_PITCH_DEG)),
        "right_shoulder_roll": float(np.deg2rad(DRAG_SHOULDER_ROLL_DEG)),
        "left_shoulder_roll": float(np.deg2rad(-DRAG_SHOULDER_ROLL_DEG)),
        "right_elbow": float(np.deg2rad(DRAG_ELBOW_DEG)),
        "left_elbow": float(np.deg2rad(-DRAG_ELBOW_DEG)),
    },
)

XIAOXIAOBAI_DRAG_ROBOT_CFG = replace(XIAOXIAOBAI_ROBOT_CFG, spec_fn=get_drag_spec, init_state=DRAG_HOME_FRAME)

XIAOXIAOBAI_GETUP_ROBOT_CFG = replace(
    XIAOXIAOBAI_ROBOT_CFG,
    spec_fn=get_getup_spec,
    collisions=(GETUP_COLLISION,),
    articulation=replace(XIAOXIAOBAI_ROBOT_CFG.articulation, actuators=(replace(actuators, kp_fw=GETUP_KP),)),
)

# -- Skate task only --
# Roller skates measured in modelforRLV6.3dm, in the foot body frame (origin on the ankle-roll
# axis, x forward, z up). Each skate weighs 22.6 g; its four wheels spin freely on bearings
# about the lateral axis through their centres.
SKATE_MASS = 0.0226
SKATE_WHEEL_MASS = 0.0025
SKATE_WHEEL_RADIUS = 0.01015
SKATE_WHEEL_HALF_WIDTH = 0.00325
SKATE_WHEEL_Z = -0.02995
# Wheel centres (x, |y|): the rear axle is 2.9 mm wider than the front one.
SKATE_WHEELS = {"front": (0.02322, 0.02979), "rear": (-0.04673, 0.03123)}
SKATE_DECK_POS = (-0.01195, 0.0, -0.02717)
SKATE_DECK_HALF = (0.0497, 0.03148, 0.00564)
SKATE_SOLE_Z = SKATE_WHEEL_Z - SKATE_WHEEL_RADIUS
# The skates stand the robot 12.6 mm higher than its bare soles.
SKATE_LIFT = -0.0274875 - SKATE_SOLE_Z
# The wheel's own spin inertia is ~1.3e-7 kg m^2; the bearings' and hubs' add a little.
SKATE_WHEEL_ARMATURE = 1e-7
# Smooth miniature bearings; the floor's rolling resistance is a contact property (below).
SKATE_BEARING_FRICTION = 2e-5
SKATE_WHEEL_JOINTS = r".*_wheel_.*"
NOT_WHEEL = r"^(?!.*_wheel_).*$"

_SKATE_FEET = {"right": "foot", "left": "foot_2"}
_SKATE_RGBA = {"deck": (0.15, 0.15, 0.18, 1.0), "wheel": (0.9, 0.85, 0.2, 1.0)}


def _add_skate(spec: mujoco.MjSpec, side: str) -> None:
    # spec.body(name) goes stale once bodies are added without recompiling.
    foot = next(b for b in spec.bodies if b.name == _SKATE_FEET[side])
    site = next(s for s in spec.sites if s.name == f"{side}_foot")
    # Foot sites on the wheels' contact line, where the sole sat on the bare robot.
    site.pos = [(SKATE_WHEELS["front"][0] + SKATE_WHEELS["rear"][0]) / 2, 0.0, SKATE_SOLE_Z]

    wheel_mass = 4 * SKATE_WHEEL_MASS
    deck_mass = SKATE_MASS - wheel_mass
    half = np.asarray(SKATE_DECK_HALF)
    skate = foot.add_body(name=f"{side}_skate", pos=[0.0, 0.0, 0.0])
    skate.mass = deck_mass
    skate.ipos = list(SKATE_DECK_POS)
    skate.inertia = [deck_mass / 3 * (half[1] ** 2 + half[2] ** 2),
                     deck_mass / 3 * (half[0] ** 2 + half[2] ** 2),
                     deck_mass / 3 * (half[0] ** 2 + half[1] ** 2)]
    skate.explicitinertial = True
    for suffix, group, contype in (("collision", 3, 1), ("visual", 2, 0)):
        skate.add_geom(
            name=f"{side}_skate_{suffix}", type=mujoco.mjtGeom.mjGEOM_BOX, pos=list(SKATE_DECK_POS),
            size=list(half), group=group, contype=contype, conaffinity=contype,
            rgba=list(_SKATE_RGBA["deck"]),
        )

    m, r, h = SKATE_WHEEL_MASS, SKATE_WHEEL_RADIUS, SKATE_WHEEL_HALF_WIDTH
    axle = 0.5 * m * r**2
    across = m * (3 * r**2 + (2 * h) ** 2) / 12
    # Cylinder axis (z) turned onto the lateral axis.
    on_axle = [np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0]
    for end, (x, y) in SKATE_WHEELS.items():
        for lateral, sign in (("in", -1.0 if side == "left" else 1.0), ("out", 1.0 if side == "left" else -1.0)):
            name = f"{side}_wheel_{end}_{lateral}"
            wheel = skate.add_body(name=name, pos=[x, sign * y, SKATE_WHEEL_Z])
            # add_body(pos=...) also sets ipos to pos, which would hang the mass off the axle.
            wheel.ipos = [0.0, 0.0, 0.0]
            wheel.mass = m
            wheel.inertia = [across, axle, across]
            wheel.explicitinertial = True
            wheel.add_joint(
                name=name, type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0.0, 1.0, 0.0],
                armature=SKATE_WHEEL_ARMATURE, frictionloss=SKATE_BEARING_FRICTION,
            )
            for suffix, group, contype in (("collision", 3, 1), ("visual", 2, 0)):
                wheel.add_geom(
                    name=f"{name}_{suffix}", type=mujoco.mjtGeom.mjGEOM_CYLINDER, quat=on_axle,
                    size=[r, h, 0.0], group=group, contype=contype, conaffinity=contype,
                    rgba=list(_SKATE_RGBA["wheel"]),
                )
            # The inner wheels clear the sole box by 1 mm sideways.
            spec.add_exclude(bodyname1=foot.name, bodyname2=name)


def get_skate_spec() -> mujoco.MjSpec:
    # The backpack goes on after the scaling to the wired robot's weight: it was weighed alone.
    spec = get_spec(backpack=False)
    scale = GETUP_TOTAL_MASS / spec.compile().body_subtreemass[1]
    for body in spec.bodies[1:]:
        body.mass *= scale
        body.inertia = body.inertia * scale
        body.fullinertia = body.fullinertia * scale
    if _BACKPACK:
        _add_backpack(spec)
    for side in _SKATE_FEET:
        _add_skate(spec, side)
    spec.compile()
    return spec


# Wheels: polyurethane on tile or carpet; the randomised wheel friction is the contact
# friction (priority above the terrain's). condim 6 adds rolling resistance proportional to
# the wheel's load: friction[2] = c_rr * wheel radius. The deck only touches when the skate
# tips over.
SKATE_COLLISION = CollisionCfg(
    geom_names_expr=(r".*_collision",),
    condim={r".*_wheel_.*_collision$": 6, r".*_skate_collision$": 3, r"^(left|right)_foot_collision$": 3,
            r".*_collision": 1},
    priority={r".*_wheel_.*_collision$": 1, r".*_skate_collision$": 1, r"^(left|right)_foot_collision$": 1},
    friction={r".*_wheel_.*_collision$": (0.8, 1e-4, 1e-4), r".*_skate_collision$": (0.5,),
              r"^(left|right)_foot_collision$": (1.0,)},
)

SKATE_HOME_FRAME = replace(HOME_FRAME, pos=(0.0, 0.0, HOME_FRAME.pos[2] + SKATE_LIFT))

XIAOXIAOBAI_SKATE_ROBOT_CFG = replace(
    XIAOXIAOBAI_ROBOT_CFG,
    spec_fn=get_skate_spec,
    init_state=SKATE_HOME_FRAME,
    collisions=(SKATE_COLLISION,),
    articulation=replace(
        XIAOXIAOBAI_ROBOT_CFG.articulation,
        actuators=(replace(actuators, target_names_expr=(r"^(?!head_wobble|leg_flex|.*_wheel_).*$",)),),
    ),
)

if __name__ == "__main__":
    import mujoco.viewer as viewer
    from mjlab.scene import Scene, SceneCfg
    from mjlab.terrains import TerrainEntityCfg

    SCENE_CFG = SceneCfg(
        terrain=TerrainEntityCfg(terrain_type="plane"),
        entities={"robot": XIAOXIAOBAI_ROBOT_CFG},
    )

    scene = Scene(SCENE_CFG, device="cuda:0")
    model = scene.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("init_state").id)
    viewer.launch(model, data=data)
