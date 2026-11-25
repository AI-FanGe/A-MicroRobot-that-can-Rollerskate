"""Convert modelforRLV6.3dm into the Xiaoxiaobai MJCF (robot.xml + link OBJ meshes).

Runs outside Rhino with the ``rhino3dm`` package (no mujoco needed):

    python scripts/rhino3dm_to_mjcf.py \
        --src ../modelforRLV6.3dm --out src/mjlab_xiaoxiaobai/robot/xiaoxiaobai

Conventions taken from the 3dm:
  * ``joint_<name>`` points give the joint centers, ``axis_<name>`` lines the axes.
  * Part masses are read from the object names, e.g. ``右小腿（36.4g）``; every servo
    instance ``<id>（18.3g）`` is a motor.
  * The trunk frame origin is the midpoint of the two hip-yaw joints, all body frames
    are world-aligned in the zero pose.
  * Joint positive directions follow the previous robot.xml so motor signs on the
    real robot keep their meaning.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rhino3dm

UNNAMED_DENSITY_G_CM3 = 1.0
MOTOR_RGBA = "0.301961 0.301961 0.301961 1"
PART_RGBA = "0.55 0.55 0.55 1"

ARMATURE_CLASS = "xml330_actuator"


@dataclass
class LinkSpec:
    name: str
    parent: str | None
    joint: str | None
    marker: str | None
    axis: tuple[float, float, float] | None
    range: tuple[float, float] | None
    parts: list[str]
    motors: list[int] = field(default_factory=list)
    unnamed_near: list[tuple[float, float, float]] = field(default_factory=list)
    horns_near: list[tuple[float, float, float]] = field(default_factory=list)


def _p(name: str) -> str:
    return rf"^{name}（[0-9.]+`?g?）$|^{name}[0-9.]+g$"


DEG = np.pi / 180.0

LINKS: list[LinkSpec] = [
    LinkSpec(
        "trunk", None, None, None, None, None,
        parts=[_p("骨盆"), _p("上半驱干"), _p("控制板"), r"^电池[0-9.]+g$",
               _p("轴承腿右"), _p("轴承腿左"), _p("轴承肩右"), _p("轴承肩左"), _p("轴承头"), r"^腰部 ?[0-9.]+g$"],
        motors=[1, 7, 13, 16, 19],
    ),
    LinkSpec(
        "head", "trunk", "head", "joint_head1", (0, 0, 1), (-1.57079633, 1.57079633),
        parts=[_p("脖子"), _p("头"), _p("屏幕"), r"^头部固定板[0-9.]+g$", r"^头步固定板[0-9.]+g$",
               r"^扬声器 ?[0-9.]+g$", r"^esp32 ?[0-9.]+g$"],
        motors=[20],
    ),
    LinkSpec("shoulder", "trunk", "right_shoulder_pitch", "joint_right_shoulder_pitch", (0, -1, 0),
             (-0.785398163, 0.785398163), parts=[_p("右肩")]),
    LinkSpec("humerus__configuration_right", "shoulder", "right_shoulder_roll", "joint_right_shoulder_roll",
             (-1, 0, 0), (0.0, 0.785398163), parts=[_p("humerus右")], motors=[14, 15]),
    LinkSpec("radius", "humerus__configuration_right", "right_elbow", "joint_right_elbow", (0, 1, 0),
             (-1.74532925, 0.0), parts=[_p("右手")]),
    LinkSpec("hip", "trunk", "right_hip_yaw", "joint_right_hip_yaw", (0, 0, -1),
             (-0.488692191, 1.04719755), parts=[_p("hip右")], horns_near=[(-29.28, -30.0, 156.35)]),
    LinkSpec("hip_block__configuration_right", "hip", "right_hip_roll", "joint_right_hip_roll", (-1, 0, 0),
             (-0.27925268, 0.523598776), parts=[_p("hip_dbl_block右")], motors=[2, 3]),
    LinkSpec("femur__configuration_right", "hip_block__configuration_right", "right_hip_pitch",
             "joint_right_hip_pitch", (0, 1, 0), (-1.3962634, 1.3962634), parts=[_p("右大腿")]),
    # The knee may hyperextend 10 deg past the straight-leg neutral, so the ready pose
    # sits inside the working range instead of on its edge.
    LinkSpec("tibia__configuration_right", "femur__configuration_right", "right_knee", "joint_right_knee",
             (0, 1, 0), (-0.174532925, 1.48352986), parts=[_p("右小腿")], motors=[4]),
    LinkSpec("ankle_block__configuration_right", "tibia__configuration_right", "right_ankle_pitch",
             "joint_right_ankle_pitch", (0, 1, 0), (-0.436332313, 0.698131701),
             parts=[_p("ankle_dbl_block右")], motors=[5, 6]),
    LinkSpec("foot", "ankle_block__configuration_right", "right_ankle_roll", "joint_right_ankle_roll",
             (-1, 0, 0), (-0.610865238, 0.610865238), parts=[_p("右脚")], horns_near=[(-29.28, -30.0, 27.33)]),
    LinkSpec("shoulder_2", "trunk", "left_shoulder_pitch", "joint_left_shoulder_pitch", (0, 1, 0),
             (-0.785398163, 0.785398163), parts=[_p("左肩")]),
    LinkSpec("humerus__configuration_left", "shoulder_2", "left_shoulder_roll", "joint_left_shoulder_roll",
             (-1, 0, 0), (-0.785398163, 0.0), parts=[_p("humerus左")], motors=[17, 18]),
    LinkSpec("radius_2", "humerus__configuration_left", "left_elbow", "joint_left_elbow", (0, -1, 0),
             (0.0, 1.74532925), parts=[_p("左手")]),
    LinkSpec("hip_2", "trunk", "left_hip_yaw", "joint_left_hip_yaw", (0, 0, -1),
             (-1.04719755, 0.488692191), parts=[_p("hip左")], horns_near=[(-29.28, 30.0, 156.35)]),
    LinkSpec("hip_block__configuration_left", "hip_2", "left_hip_roll", "joint_left_hip_roll", (-1, 0, 0),
             (-0.523598776, 0.27925268), parts=[_p("hip_dbl_block左")], motors=[8, 9]),
    LinkSpec("femur__configuration_left", "hip_block__configuration_left", "left_hip_pitch",
             "joint_left_hip_pitch", (0, -1, 0), (-1.3962634, 1.3962634), parts=[_p("左大腿")]),
    LinkSpec("tibia__configuration_left", "femur__configuration_left", "left_knee", "joint_left_knee",
             (0, -1, 0), (-1.48352986, 0.174532925), parts=[_p("左小腿")], motors=[10]),
    LinkSpec("ankle_block__configuration_left", "tibia__configuration_left", "left_ankle_pitch",
             "joint_left_ankle_pitch", (0, -1, 0), (-0.698131701, 0.436332313),
             parts=[_p("ankle_dbl_block左")], motors=[11, 12]),
    LinkSpec("foot_2", "ankle_block__configuration_left", "left_ankle_roll", "joint_left_ankle_roll",
             (-1, 0, 0), (-0.610865238, 0.610865238), parts=[_p("左脚")], horns_near=[(-29.28, 30.0, 27.33)]),
]

FOOT_LINKS = {"foot": "right", "foot_2": "left"}

# The hip-yaw bracket wraps around the femur through the crossed roll/pitch axes, so
# their boxes always overlap although the real parts never touch.
CONTACT_EXCLUDES = [
    ("hip", "femur__configuration_right"),
    ("hip_2", "femur__configuration_left"),
]

# LinkSpec ranges are the working ranges the policy is penalised for leaving, not hard
# stops. The MJCF range is widened so that mjlab's soft limit lands exactly on them;
# must match soft_joint_pos_limit_factor in xiaoxiaobai_constants.py.
SOFT_LIMIT_FACTOR = 0.9

# Collision boxes are the link's printed-part bounding box shrunk by this factor so
# that neighbouring links do not touch in the nominal poses.
COLLISION_SHRINK = 0.7


# ---------------------------------------------------------------- geometry ---


def xform_to_np(xf: rhino3dm.Transform) -> np.ndarray:
    return np.array([[getattr(xf, f"M{i}{j}") for j in range(4)] for i in range(4)])


def mesh_to_np(mesh: rhino3dm.Mesh) -> tuple[np.ndarray, np.ndarray]:
    verts = np.array([[v.X, v.Y, v.Z] for v in mesh.Vertices], dtype=float)
    faces = []
    for i in range(len(mesh.Faces)):
        f = mesh.Faces[i]
        if f[2] == f[3]:
            faces.append((f[0], f[1], f[2]))
        else:
            faces.append((f[0], f[1], f[2]))
            faces.append((f[0], f[2], f[3]))
    return verts, np.array(faces, dtype=np.int64).reshape(-1, 3)


class Doc:
    def __init__(self, path: Path) -> None:
        self.model = rhino3dm.File3dm.Read(str(path))
        if self.model is None:
            raise SystemExit(f"Cannot read {path}")
        self.objects = {str(o.Attributes.Id): o for o in self.model.Objects}
        self.idefs = {str(d.Id): d for d in self.model.InstanceDefinitions}
        self.top = [o for o in self.model.Objects if str(o.Attributes.Mode).endswith("Normal")]

    def named(self, pattern: str) -> list:
        rx = re.compile(pattern)
        return [o for o in self.top if rx.search(o.Attributes.Name or "")]

    def point(self, name: str) -> np.ndarray:
        hits = [o for o in self.top if (o.Attributes.Name or "") == name]
        if len(hits) != 1:
            raise SystemExit(f"Expected one marker {name!r}, found {len(hits)}")
        bb = hits[0].Geometry.GetBoundingBox()
        return np.array([(bb.Min.X + bb.Max.X) / 2, (bb.Min.Y + bb.Max.Y) / 2, (bb.Min.Z + bb.Max.Z) / 2])

    def meshes(self, geometry, xform: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        out = []
        if isinstance(geometry, rhino3dm.InstanceReference):
            idef = self.idefs.get(str(geometry.ParentIdefId))
            if idef is None:
                return out
            nested = xform @ xform_to_np(geometry.Xform)
            for oid in idef.GetObjectIds():
                obj = self.objects.get(str(oid))
                if obj is not None:
                    out.extend(self.meshes(obj.Geometry, nested))
            return out
        pieces = []
        if isinstance(geometry, rhino3dm.Mesh):
            pieces = [geometry]
        elif isinstance(geometry, rhino3dm.Extrusion):
            m = geometry.GetMesh(rhino3dm.MeshType.Any)
            if m is not None:
                pieces = [m]
            else:
                brep = geometry.ToBrep(True)
                pieces = [f.GetMesh(rhino3dm.MeshType.Any) for f in brep.Faces] if brep else []
        elif isinstance(geometry, rhino3dm.Brep):
            pieces = [f.GetMesh(rhino3dm.MeshType.Any) for f in geometry.Faces]
        for piece in pieces:
            if piece is None or len(piece.Vertices) == 0:
                continue
            v, f = mesh_to_np(piece)
            vh = np.c_[v, np.ones(len(v))] @ xform.T
            out.append((vh[:, :3], f))
        return out

    def object_mesh(self, obj) -> tuple[np.ndarray, np.ndarray]:
        chunks = self.meshes(obj.Geometry, np.eye(4))
        if not chunks:
            raise SystemExit(f"No mesh for {obj.Attributes.Name!r}")
        return merge(chunks)


def merge(chunks):
    verts, faces, off = [], [], 0
    for v, f in chunks:
        verts.append(v)
        faces.append(f + off)
        off += len(v)
    return np.vstack(verts), np.vstack(faces)


def mass_from_name(name: str) -> float:
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)`?g?）?$", name.replace("（", "(").replace("）", ")").rstrip(")"))
    if not m:
        raise SystemExit(f"No mass in name {name!r}")
    return float(m.group(1)) * 1e-3


def mesh_volume_cm3(v, f) -> float:
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0) / 1000.0


def shell_inertia(v_m: np.ndarray, f: np.ndarray, mass: float) -> tuple[np.ndarray, np.ndarray]:
    """Mass spread uniformly over the surface. Returns (com, inertia about com)."""
    a, b, c = v_m[f[:, 0]], v_m[f[:, 1]], v_m[f[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    total = area.sum()
    if total <= 0:
        raise SystemExit("Degenerate mesh")
    centroid = (a + b + c) / 3.0
    com = (area[:, None] * centroid).sum(0) / total
    s = a + b + c
    second = (area[:, None, None] / 12.0) * (
        np.einsum("ni,nj->nij", a, a) + np.einsum("ni,nj->nij", b, b) + np.einsum("ni,nj->nij", c, c)
        + np.einsum("ni,nj->nij", s, s)
    )
    cov = second.sum(0) * (mass / total) - mass * np.outer(com, com)
    inertia = np.trace(cov) * np.eye(3) - cov
    return com, inertia


def combine(bodies: list[tuple[float, np.ndarray, np.ndarray]]):
    mass = sum(m for m, _, _ in bodies)
    com = sum(m * c for m, c, _ in bodies) / mass
    inertia = np.zeros((3, 3))
    for m, c, i in bodies:
        d = c - com
        inertia += i + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    return mass, com, inertia


def write_obj(path: Path, v_m: np.ndarray, f: np.ndarray) -> None:
    lines = ["# Generated by scripts/rhino3dm_to_mjcf.py; units: m, link frame"]
    lines += [f"v {x:.7g} {y:.7g} {z:.7g}" for x, y, z in v_m]
    lines += [f"f {i + 1} {j + 1} {k + 1}" for i, j, k in f]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(x) -> str:
    return " ".join(f"{float(v):.9g}" for v in np.ravel(x))


# ------------------------------------------------------------------- build ---


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Directory for robot.xml and the mesh folder")
    parser.add_argument("--meshdir", default="assets_outputmodel")
    args = parser.parse_args()

    doc = Doc(args.src)
    mesh_dir = args.out / args.meshdir
    mesh_dir.mkdir(parents=True, exist_ok=True)

    right_yaw = doc.point("joint_right_hip_yaw")
    left_yaw = doc.point("joint_left_hip_yaw")
    origin_mm = 0.5 * (right_yaw + left_yaw)

    def to_m(p_mm):
        return (np.asarray(p_mm) - origin_mm) * 1e-3

    used: set[str] = set()
    unnamed = [o for o in doc.top if not (o.Attributes.Name or "") and isinstance(o.Geometry, rhino3dm.Mesh)]
    horns = doc.named(r"^DC15_A01_HORN_IDLE2_DUMMY$")

    def nearest(objs, target):
        def dist(o):
            bb = o.Geometry.GetBoundingBox()
            c = np.array([(bb.Min.X + bb.Max.X) / 2, (bb.Min.Y + bb.Max.Y) / 2, (bb.Min.Z + bb.Max.Z) / 2])
            return np.linalg.norm(c - np.array(target))
        best = min(objs, key=dist)
        if dist(best) > 1.0:
            raise SystemExit(f"No object near {target}")
        return best

    link_pos_world = {}
    for spec in LINKS:
        link_pos_world[spec.name] = to_m(doc.point(spec.marker)) if spec.marker else np.zeros(3)

    # Each joint must sit on an axis line of the same kind (the 3dm has a few left-side
    # axes labelled "right"), and the XML axis must be parallel to it.
    axis_lines = []
    for o in doc.top:
        name = o.Attributes.Name or ""
        if name.startswith("axis_") and isinstance(o.Geometry, (rhino3dm.PolylineCurve, rhino3dm.LineCurve)):
            g = o.Geometry
            a = np.array([g.PointAtStart.X, g.PointAtStart.Y, g.PointAtStart.Z])
            b = np.array([g.PointAtEnd.X, g.PointAtEnd.Y, g.PointAtEnd.Z])
            kind = re.sub(r"^axis_(left_|right_)?", "", name)
            axis_lines.append((kind, name, a, b))
    for spec in LINKS:
        if not spec.joint:
            continue
        kind = "head1" if spec.joint == "head" else re.sub(r"^(left|right)_", "", spec.joint)
        p = doc.point(spec.marker)

        def line_dist(line):
            _, _, a, b = line
            u = (b - a) / np.linalg.norm(b - a)
            return np.linalg.norm(np.cross(p - a, u))

        candidates = [line for line in axis_lines if line[0] == kind]
        best = min(candidates, key=line_dist)
        u = (best[3] - best[2]) / np.linalg.norm(best[3] - best[2])
        if line_dist(best) > 0.5 or abs(abs(np.dot(u, spec.axis)) - 1.0) > 1e-3:
            raise SystemExit(f"{spec.joint}: axis {spec.axis} does not match {best[1]} (dist {line_dist(best):.2f} mm, dir {u})")

    report = []
    body_xml: dict[str, list[str]] = {}
    foot_geoms: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    imu_site = None

    for spec in LINKS:
        origin = link_pos_world[spec.name]
        inertial_parts = []
        part_chunks = []
        motor_chunks = []
        names = []

        for pattern in spec.parts:
            hits = doc.named(pattern)
            if len(hits) != 1:
                raise SystemExit(f"{spec.name}: pattern {pattern!r} matched {[h.Attributes.Name for h in hits]}")
            obj = hits[0]
            used.add(str(obj.Attributes.Id))
            v, f = doc.object_mesh(obj)
            mass = mass_from_name(obj.Attributes.Name)
            vm = v * 1e-3 - origin_mm * 1e-3 - origin
            com, inertia = shell_inertia(vm, f, mass)
            inertial_parts.append((mass, com, inertia))
            part_chunks.append((vm, f))
            names.append(f"{obj.Attributes.Name}")

        for target in spec.unnamed_near:
            obj = nearest(unnamed, target)
            used.add(str(obj.Attributes.Id))
            v, f = doc.object_mesh(obj)
            mass = mesh_volume_cm3(v, f) * UNNAMED_DENSITY_G_CM3 * 1e-3
            vm = v * 1e-3 - origin_mm * 1e-3 - origin
            com, inertia = shell_inertia(vm, f, mass)
            inertial_parts.append((mass, com, inertia))
            part_chunks.append((vm, f))
            names.append(f"<unnamed {mass * 1e3:.1f}g est>")

        for target in spec.horns_near:
            obj = nearest(horns, target)
            used.add(str(obj.Attributes.Id))
            v, f = doc.object_mesh(obj)
            motor_chunks.append((v * 1e-3 - origin_mm * 1e-3 - origin, f))

        for motor_id in spec.motors:
            hits = doc.named(rf"^{motor_id}（[0-9.]+g）$")
            if len(hits) != 1:
                raise SystemExit(f"motor {motor_id}: found {len(hits)}")
            obj = hits[0]
            used.add(str(obj.Attributes.Id))
            v, f = doc.object_mesh(obj)
            mass = mass_from_name(obj.Attributes.Name)
            vm = v * 1e-3 - origin_mm * 1e-3 - origin
            com, inertia = shell_inertia(vm, f, mass)
            inertial_parts.append((mass, com, inertia))
            motor_chunks.append((vm, f))
            names.append(f"motor {motor_id}")

        mass, com, inertia = combine(inertial_parts)
        report.append((spec.name, mass, names))

        pv, pf = merge(part_chunks)
        write_obj(mesh_dir / f"{spec.name}.obj", pv, pf)
        if motor_chunks:
            mv, mf = merge(motor_chunks)
            write_obj(mesh_dir / f"{spec.name}_motors.obj", mv, mf)

        lines = []
        if spec.joint:
            mid, half = (spec.range[0] + spec.range[1]) / 2, (spec.range[1] - spec.range[0]) / 2
            lo, hi = mid - half / SOFT_LIMIT_FACTOR, mid + half / SOFT_LIMIT_FACTOR
            lines.append(
                f'<joint axis="{fmt(spec.axis)}" name="{spec.joint}" type="hinge" range="{lo:.9g} {hi:.9g}" class="{ARMATURE_CLASS}" />'
            )
        lines.append(
            f'<inertial pos="{fmt(com)}" mass="{mass:.9g}" fullinertia="{fmt([inertia[0, 0], inertia[1, 1], inertia[2, 2], inertia[0, 1], inertia[0, 2], inertia[1, 2]])}" />'
        )
        lines.append(f'<geom name="{spec.name}_visual" class="visual" type="mesh" mesh="{spec.name}_mesh" rgba="{PART_RGBA}" />')
        if motor_chunks:
            lines.append(
                f'<geom name="{spec.name}_motors_visual" class="visual" type="mesh" mesh="{spec.name}_motors_mesh" rgba="{MOTOR_RGBA}" />'
            )

        if spec.name in FOOT_LINKS:
            side = FOOT_LINKS[spec.name]
            zmin = pv[:, 2].min()
            sole = pv[pv[:, 2] < zmin + 0.002]
            lo_xy, hi_xy = sole[:, :2].min(0), sole[:, :2].max(0)
            center = np.r_[(lo_xy + hi_xy) / 2, zmin + 0.003]
            half = np.r_[(hi_xy - lo_xy) / 2, 0.003]
            lines.append(f'<geom name="{side}_foot_collision" class="collision" type="box" pos="{fmt(center)}" size="{fmt(half)}" />')
            lines.append(f'<site group="3" name="{side}_foot" pos="{fmt([center[0], center[1], zmin])}" quat="1 0 0 0" />')
            foot_geoms[side] = (center, half)
        else:
            lo, hi = pv.min(0), pv.max(0)
            center = (lo + hi) / 2
            half = (hi - lo) / 2 * COLLISION_SHRINK
            lines.append(f'<geom name="{spec.name}_collision" class="collision" type="box" pos="{fmt(center)}" size="{fmt(half)}" />')

        if spec.name == "trunk":
            board = doc.named(_p("控制板"))[0]
            chunks = []
            imu_xf = None

            def find_imu(geometry, xform):
                nonlocal imu_xf
                if isinstance(geometry, rhino3dm.InstanceReference):
                    idef = doc.idefs[str(geometry.ParentIdefId)]
                    nested = xform @ xform_to_np(geometry.Xform)
                    for oid in idef.GetObjectIds():
                        o = doc.objects.get(str(oid))
                        if o is None:
                            continue
                        if (o.Attributes.Name or "") == "IMU":
                            chunks.extend(doc.meshes(o.Geometry, nested))
                            imu_xf = nested
                        else:
                            find_imu(o.Geometry, nested)

            find_imu(board.Geometry, np.eye(4))
            if not chunks:
                raise SystemExit("IMU not found inside 控制板")
            iv, _ = merge(chunks)
            imu_center = (iv.min(0) + iv.max(0)) / 2 * 1e-3 - origin_mm * 1e-3
            imu_size = iv.max(0) - iv.min(0)
            imu_site = (imu_center, imu_size, imu_xf)
            lines.append(f'<site group="3" name="body" pos="0 0 0" quat="1 0 0 0" />')
            lines.append(f'<site group="3" name="imu" pos="{fmt(imu_center)}" quat="0.5 -0.5 -0.5 0.5" />')

        body_xml[spec.name] = lines

    # Anything with a mass in its name that was not assigned is a modelling mistake.
    leftovers = [
        o.Attributes.Name for o in doc.top
        if str(o.Attributes.Id) not in used and re.search(r"[0-9.]+`?g?）$|[0-9.]+g$", o.Attributes.Name or "")
    ]
    if leftovers:
        raise SystemExit(f"Unassigned weighted objects: {leftovers}")

    children: dict[str, list[LinkSpec]] = {}
    for spec in LINKS:
        if spec.parent:
            children.setdefault(spec.parent, []).append(spec)

    def emit(spec: LinkSpec, depth: int) -> list[str]:
        pad = "  " * depth
        if spec.parent is None:
            out = [f'{pad}<body name="{spec.name}" pos="0 0 0" quat="1 0 0 0">', f'{pad}  <freejoint name="trunk_freejoint" />']
        else:
            rel = link_pos_world[spec.name] - link_pos_world[spec.parent]
            out = [f'{pad}<body name="{spec.name}" pos="{fmt(rel)}" quat="1 0 0 0">']
        out += [f"{pad}  {line}" for line in body_xml[spec.name]]
        for child in children.get(spec.name, []):
            out += emit(child, depth + 1)
        out.append(f"{pad}</body>")
        return out

    assets = []
    for spec in LINKS:
        assets.append(f'    <mesh name="{spec.name}_mesh" file="{spec.name}.obj" />')
        if (mesh_dir / f"{spec.name}_motors.obj").exists():
            assets.append(f'    <mesh name="{spec.name}_motors_mesh" file="{spec.name}_motors.obj" />')

    actuated = [s.joint for s in LINKS if s.joint]
    xml = f"""<mujoco model="xiaoxiaobai">
  <compiler angle="radian" meshdir="{args.meshdir}" autolimits="true" />
  <default>
    <default class="xiaoxiaobai">
      <joint frictionloss="0.1" armature="0.005" />
      <position kp="50" dampratio="1" />
      <default class="visual">
        <geom contype="0" conaffinity="0" group="2" />
      </default>
      <default class="collision">
        <geom group="3" />
      </default>
    </default>
  </default>
  <default>
    <default class="xml330_actuator">
      <joint damping="0.041" frictionloss="0.013" armature="0.0018" />
      <position kp="0.277" kv="0.0" forcerange="-0.77 0.77" />
    </default>
  </default>
  <sensor>
    <framequat name="orientation" objtype="site" noise="0.001" objname="imu" />
    <gyro name="angular-velocity" site="imu" noise="0.005" />
    <gyro name="imu_ang_vel" site="imu" />
    <velocimeter name="imu_lin_vel" site="imu" />
    <accelerometer name="imu_accel" site="imu" />
    <subtreeangmom name="root_angmom" body="trunk" />
  </sensor>
  <asset>
{chr(10).join(assets)}
  </asset>
  <worldbody>
{chr(10).join(emit(LINKS[0], 2))}
  </worldbody>
  <contact>
{chr(10).join(f'    <exclude body1="{a}" body2="{b}" />' for a, b in CONTACT_EXCLUDES)}
  </contact>
  <actuator>
{chr(10).join(f'    <position class="{ARMATURE_CLASS}" name="{j}" joint="{j}" inheritrange="1" />' for j in actuated)}
  </actuator>
</mujoco>
"""
    (args.out / "robot.xml").write_text(xml, encoding="utf-8")

    total = sum(m for _, m, _ in report)
    print(f"trunk origin (Rhino mm): {fmt(origin_mm)}")
    for name, mass, parts in report:
        print(f"  {name:36s} {mass * 1e3:7.1f} g  <- {', '.join(parts)}")
    print(f"total mass: {total * 1e3:.1f} g")
    c, s, xf = imu_site
    print(f"imu site (trunk frame, m): {fmt(c)}  IMU box size (mm): {fmt(s)}")
    print("IMU instance rotation (columns = IMU-local x,y,z in world):")
    print(np.round(xf[:3, :3], 4))
    for side, (center, half) in foot_geoms.items():
        print(f"{side} foot box center {fmt(center)} half {fmt(half)}")


if __name__ == "__main__":
    main()
