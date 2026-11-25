# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

"""Turn the Rhino CAD model into a small three.js-ready model for the web UI's virtual robot.

Reads modelforRLV6.3dm with the link layout of mjlab_xiaoxiaobai's rhino3dm_to_mjcf.py (the
same file that generates src/model/mjcf/robot.xml, so frames and joint axes match), and
keeps every part separate so the UI can label and highlight motors, boards and sensors.

Writes webui/model/xiaoxiaobai.json (body tree, joints, parts, labelled components) and
xiaoxiaobai.bin (int16 positions, int8 crease-angle normals, uint16/uint32 indices).
Run from the repo root on a PC (`make webui-model`):

    uv run --no-project --python 3.12 --with rhino3dm --with fast-simplification --with numpy \
        python scripts/build_webui_model.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import fast_simplification
import numpy as np
import rhino3dm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from constants import MOTOR_TO_ID, NEUTRAL_POSE  # noqa: E402

# Head pitch (the second neck motor) is driven by the upper-body tools only.
EXTRA_MOTORS = {"head_pitch": 20}
WELD_M = 5e-6
CREASE_DEG = 32.0
AGGRESSIVENESS = 5


def load_converter(path: Path):
    spec = importlib.util.spec_from_file_location("rhino3dm_to_mjcf", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ mesh ops


def weld(v: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Merge the per-face copies Rhino keeps along Brep edges, drop degenerate/duplicate faces.

    Decimating the unwelded soup tears holes along every CAD face boundary.
    """
    key = np.round(v / WELD_M).astype(np.int64)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.ravel()
    nv = v[first]
    nf = inverse[f]
    nf = nf[(nf[:, 0] != nf[:, 1]) & (nf[:, 1] != nf[:, 2]) & (nf[:, 0] != nf[:, 2])]
    _, keep = np.unique(np.sort(nf, axis=1), axis=0, return_index=True)
    nf = nf[np.sort(keep)]
    used = np.unique(nf)
    remap = np.full(len(nv), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return nv[used], remap[nf]


def decimate(v: np.ndarray, f: np.ndarray, keep: float, min_faces: int) -> tuple[np.ndarray, np.ndarray]:
    target = max(min_faces, int(len(f) * keep))
    if target >= len(f):
        return v, f
    out_v, out_f = fast_simplification.simplify(
        v.astype(np.float32), f.astype(np.int32), 1.0 - target / len(f), agg=AGGRESSIVENESS
    )
    return weld(out_v.astype(np.float64), out_f.astype(np.int64))


def crease_split(v: np.ndarray, f: np.ndarray, crease_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Smooth normals inside a surface, hard edges where faces meet at more than crease_deg."""
    raw = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    length = np.linalg.norm(raw, axis=1)
    good = length > 1e-12
    f, raw, length = f[good], raw[good], length[good]
    unit = raw / length[:, None]

    corner_vertex = f.ravel()
    corner_face = np.repeat(np.arange(len(f)), 3)
    order = np.argsort(corner_vertex, kind="stable")
    sv, sf = corner_vertex[order], corner_face[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sv)) + 1]
    sizes = np.diff(np.r_[starts, len(sv)])
    group = np.repeat(np.arange(len(starts)), sizes)
    k = sizes[group]
    a = np.repeat(np.arange(len(sv)), k)
    offset = np.arange(len(a)) - np.repeat(np.cumsum(k) - k, k)
    b = starts[group[a]] + offset
    fa, fb = sf[a], sf[b]
    similar = np.einsum("ij,ij->i", unit[fa], unit[fb]) >= np.cos(np.radians(crease_deg))
    acc = np.zeros((len(sv), 3))
    np.add.at(acc, a[similar], raw[fb[similar]])
    acc /= np.linalg.norm(acc, axis=1, keepdims=True)
    normals = np.empty_like(acc)
    normals[order] = acc

    q = np.round(normals * 127).astype(np.int64)
    key = np.c_[corner_vertex, q]
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    out_v = v[corner_vertex[first]]
    out_n = q[first].astype(np.int8)
    out_f = inverse.ravel().reshape(-1, 3)
    return out_v, out_n, out_f


class Blob:
    def __init__(self) -> None:
        self.data = bytearray()

    def add(self, array: np.ndarray) -> int:
        while len(self.data) % 4:
            self.data.append(0)
        offset = len(self.data)
        self.data += array.tobytes()
        return offset


def pack_mesh(blob: Blob, v: np.ndarray, f: np.ndarray, keep: float, min_faces: int) -> dict:
    v, f = weld(v, f)
    before = len(f)
    v, f = decimate(v, f, keep, min_faces)
    v, n, f = crease_split(v, f, CREASE_DEG)
    lo, hi = v.min(axis=0), v.max(axis=0)
    scale = np.where(hi - lo > 1e-12, (hi - lo) / 65534.0, 1.0)
    q = np.round((v - lo) / scale - 32767).clip(-32767, 32767).astype("<i2")
    index_type = "uint16" if len(v) < 65536 else "uint32"
    return {
        "vertices": int(len(v)),
        "triangles": int(len(f)),
        "source_triangles": before,
        "position": {"offset": blob.add(q), "origin": (lo + 32767 * scale).round(9).tolist(), "scale": scale.round(12).tolist()},
        "normal": {"offset": blob.add(n)},
        "index": {"offset": blob.add(f.astype("<u2" if index_type == "uint16" else "<u4")), "type": index_type},
    }


# ------------------------------------------------------------------ parts


def kind_of(name: str) -> str:
    rules = [
        (r"^\d+（", "motor"),
        (r"^电池", "battery"),
        (r"^esp32", "esp32"),
        (r"^扬声器", "speaker"),
        (r"^屏幕", "screen"),
        (r"^轴承|^washer|^DC15", "metal"),
        (r"^raspberry_pi", "pi"),
        (r"^IMU$", "imu"),
    ]
    for pattern, kind in rules:
        if re.search(pattern, name):
            return kind
    return "shell"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", type=Path, default=REPO.parent / "modelforRLV6.3dm")
    parser.add_argument("--converter", type=Path, default=REPO.parent / "mjlab_xiaoxiaobai/scripts/rhino3dm_to_mjcf.py",
                        help="rhino3dm_to_mjcf.py with the link layout (LINKS)")
    parser.add_argument("--out", type=Path, default=REPO / "webui/model")
    parser.add_argument("--keep", type=float, default=0.3, help="fraction of faces kept per printed part")
    parser.add_argument("--min-faces", type=int, default=400)
    args = parser.parse_args()

    conv = load_converter(args.converter)
    doc = conv.Doc(args.src)
    origin_mm = 0.5 * (doc.point("joint_right_hip_yaw") + doc.point("joint_left_hip_yaw"))
    link_pos = {s.name: (doc.point(s.marker) - origin_mm) * 1e-3 if s.marker else np.zeros(3) for s in conv.LINKS}
    body_index = {s.name: i for i, s in enumerate(conv.LINKS)}

    def to_link(v_mm: np.ndarray, link: str) -> np.ndarray:
        return (v_mm - origin_mm) * 1e-3 - link_pos[link]

    bodies = [{
        "name": s.name,
        "parent": body_index[s.parent] if s.parent else -1,
        "pos": (link_pos[s.name] - (link_pos[s.parent] if s.parent else 0)).round(7).tolist(),
        "quat": [1, 0, 0, 0],
    } for s in conv.LINKS]
    joints = [{
        "name": s.joint,
        "body": body_index[s.name],
        "axis": list(s.axis),
        "pos": [0, 0, 0],
        "range": [round(s.range[0], 5), round(s.range[1], 5)],
        "motor_id": MOTOR_TO_ID.get(s.joint),
        "neutral": round(float(NEUTRAL_POSE.get(s.joint, 0.0)), 5),
    } for s in conv.LINKS if s.joint]

    blob = Blob()
    meshes: list[dict] = []
    parts: list[dict] = []
    lowest = np.inf

    def add_part(name: str, kind: str, link: str, v_link: np.ndarray, f: np.ndarray, keep: float, min_faces: int) -> int:
        nonlocal lowest
        meshes.append(pack_mesh(blob, v_link, f, keep, min_faces))
        lo, hi = v_link.min(0), v_link.max(0)
        if link.startswith("foot"):
            lowest = min(lowest, float((v_link[:, 2] + link_pos[link][2]).min()))
        parts.append({
            "name": name, "kind": kind, "body": body_index[link], "mesh": len(meshes) - 1,
            "center": ((lo + hi) / 2).round(6).tolist(), "size": (hi - lo).round(6).tolist(),
        })
        return len(parts) - 1

    # Joint -> CAD motor: the motor whose housing holds the joint center.
    motor_objs = {int(m.group(1)): o for o in doc.top if (m := re.match(r"^(\d+)（", o.Attributes.Name or ""))}
    motor_world = {cad: doc.object_mesh(o)[0] for cad, o in motor_objs.items()}
    joint_names = [o.Attributes.Name[6:] for o in doc.top if (o.Attributes.Name or "").startswith("joint_")]
    runtime_id: dict[int, tuple[int, str]] = {}
    for joint in joint_names:
        p = doc.point(f"joint_{joint}")
        cad = min(motor_world, key=lambda c: np.linalg.norm(motor_world[c] - p, axis=1).min())
        name = {"head1": "head", "head2": "head_pitch"}.get(joint, joint)
        rid = MOTOR_TO_ID.get(name) or EXTRA_MOTORS.get(name)
        if rid is None or cad in runtime_id:
            raise SystemExit(f"cannot map joint {joint} (CAD motor {cad})")
        runtime_id[cad] = (rid, name)

    # Motors are two shared block meshes placed by their instance transforms.
    motor_meshes: dict[str, int] = {}
    components: list[dict] = []
    total_in = 0
    for spec in conv.LINKS:
        for pattern in spec.parts:
            obj = doc.named(pattern)[0]
            name = obj.Attributes.Name
            if name.startswith("控制板"):
                add_board(doc, conv, spec.name, obj, to_link, add_part, components)
                continue
            v, f = doc.object_mesh(obj)
            total_in += len(f)
            kind = kind_of(name)
            box = kind in ("battery", "esp32", "speaker")
            index = add_part(name, kind, spec.name, to_link(v, spec.name), f, 1.0 if box else args.keep, args.min_faces)
            if kind in ("battery", "esp32", "speaker", "screen"):
                components.append({"id": kind, "kind": kind, "part": index})
        for target in spec.horns_near:
            horns = doc.named(r"^DC15_A01_HORN_IDLE2_DUMMY$")
            obj = min(horns, key=lambda o: np.linalg.norm(doc.object_mesh(o)[0].mean(0) - np.array(target)))
            v, f = doc.object_mesh(obj)
            add_part("horn", "metal", spec.name, to_link(v, spec.name), f, 0.4, 300)
        for cad in spec.motors:
            obj = motor_objs[cad]
            idef = str(obj.Geometry.ParentIdefId)
            xf = conv.xform_to_np(obj.Geometry.Xform)
            if idef not in motor_meshes:
                local = (np.linalg.inv(xf) @ np.c_[motor_world[cad], np.ones(len(motor_world[cad]))].T).T[:, :3]
                _, f = doc.object_mesh(obj)
                meshes.append(pack_mesh(blob, local * 1e-3, f, 0.22, 600))
                motor_meshes[idef] = len(meshes) - 1
            matrix = np.eye(4)
            matrix[:3, :3] = xf[:3, :3]
            matrix[:3, 3] = to_link(xf[:3, 3], spec.name)
            world = motor_world[cad]
            v_link = to_link(world, spec.name)
            lo, hi = v_link.min(0), v_link.max(0)
            rid, joint = runtime_id[cad]
            parts.append({
                "name": f"motor {rid}", "kind": "motor", "body": body_index[spec.name], "mesh": motor_meshes[idef],
                "matrix": matrix.T.ravel().round(9).tolist(),
                "center": ((lo + hi) / 2).round(6).tolist(), "size": (hi - lo).round(6).tolist(),
            })
            components.append({"id": f"motor:{rid}", "kind": "motor", "motor_id": rid, "joint": joint, "part": len(parts) - 1})

    screen = next(p for p in parts if p["kind"] == "screen")
    face = {
        "body": screen["body"],
        "front_x": round(screen["center"][0] + screen["size"][0] / 2, 6),
        "center": [screen["center"][1], screen["center"][2]],
        "size": [screen["size"][1], screen["size"][2]],
    }
    components.sort(key=lambda c: (c["kind"] == "motor", c.get("motor_id", 0)))

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "xiaoxiaobai.bin").write_bytes(bytes(blob.data))
    manifest = {
        "version": 2,
        "source": args.src.name,
        "ground_z": round(-lowest, 5),
        "bodies": bodies,
        "joints": joints,
        "meshes": meshes,
        "parts": parts,
        "components": components,
        "face": face,
        "bin": "xiaoxiaobai.bin",
        "bytes": len(blob.data),
    }
    (args.out / "xiaoxiaobai.json").write_text(json.dumps(manifest, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    tri = sum(m["triangles"] for m in meshes)
    print(f"{len(bodies)} bodies, {len(joints)} joints, {len(parts)} parts, {len(meshes)} meshes, {len(components)} labelled")
    print(f"triangles {tri} (from {sum(m['source_triangles'] for m in meshes)} welded), {len(blob.data) / 1e6:.2f} MB, ground {-lowest:.4f} m")


def add_board(doc, conv, link: str, obj, to_link, add_part, components: list[dict]) -> None:
    """The control board block: Raspberry Pi, IMU breakout and spacers, kept apart for labels."""
    pieces: list[tuple[str, np.ndarray, np.ndarray]] = []

    def walk(geometry, xform, name):
        if isinstance(geometry, rhino3dm.InstanceReference):
            idef = doc.idefs[str(geometry.ParentIdefId)]
            nested = xform @ conv.xform_to_np(geometry.Xform)
            for oid in idef.GetObjectIds():
                child = doc.objects.get(str(oid))
                if child is not None:
                    walk(child.Geometry, nested, child.Attributes.Name or name)
            return
        chunks = doc.meshes(geometry, xform)
        if chunks:
            v, f = conv.merge(chunks)
            pieces.append((name, v, f))

    walk(obj.Geometry, np.eye(4), "")
    imu = next(v for name, v, _ in pieces if name == "IMU")
    imu_center = (imu.min(0) + imu.max(0)) / 2
    groups: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for name, v, f in pieces:
        center = (v.min(0) + v.max(0)) / 2
        if name.startswith("raspberry_pi"):
            key = "pi"
        elif name == "IMU" or (not name and np.linalg.norm(center[[0, 2]] - imu_center[[0, 2]]) < 3 and np.ptp(v[:, 1]) < 30):
            key = "imu"
        elif name == "washer":
            key = "metal"
        else:
            key = "pcb"
        groups.setdefault(key, []).append((v, f))
    for key, chunks in groups.items():
        v, f = conv.merge(chunks)
        keep = {"pi": 0.12, "imu": 1.0, "metal": 0.4, "pcb": 1.0}[key]
        index = add_part({"pi": "raspberry_pi_zero_2W", "imu": "IMU", "metal": "spacers", "pcb": "board"}[key],
                         key, link, to_link(v, link), f, keep, 300)
        if key in ("pi", "imu"):
            components.append({"id": key, "kind": key, "part": index})


if __name__ == "__main__":
    main()
