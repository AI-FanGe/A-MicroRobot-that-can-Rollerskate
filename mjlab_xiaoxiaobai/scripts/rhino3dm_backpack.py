"""Extract the charging backpack (电池100g, 电池背包30g) from modelforRLV7.3dm.

V7 is V6 plus these two objects on the robot's back. robot.xml stays as it is (it carries
hand-added joints the converter does not know); this writes the two meshes in the trunk frame
and backpack.json, which robot/xiaoxiaobai_constants.py folds into the trunk with
XIAOXIAOBAI_BACKPACK=1:

    python scripts/rhino3dm_backpack.py \
        --src ../modelforRLV7.3dm --out src/mjlab_xiaoxiaobai/robot/xiaoxiaobai

The battery is a cuboid, taken as a solid box over its bounding box; the bag is fabric, its
mass spread over its surface. Both meshes are open, so their volume is not usable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rhino3dm_to_mjcf import COLLISION_SHRINK, Doc, shell_inertia, write_obj

PARTS = [  # object name, mesh file, mass [kg], solid box (else shell)
    ("电池100g", "backpack_battery", 0.100, True),
    ("电池背包30g", "backpack_bag", 0.030, False),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--meshdir", default="assets_outputmodel")
    args = parser.parse_args()

    doc = Doc(args.src)
    origin_mm = 0.5 * (doc.point("joint_right_hip_yaw") + doc.point("joint_left_hip_yaw"))
    parts, all_v = [], []
    for name, mesh, mass, solid in PARTS:
        hits = [o for o in doc.top if (o.Attributes.Name or "") == name]
        if len(hits) != 1:
            raise SystemExit(f"Expected one {name!r}, found {len(hits)}")
        v, f = doc.object_mesh(hits[0])
        vm = (v - origin_mm) * 1e-3
        write_obj(args.out / args.meshdir / f"{mesh}.obj", vm, f)
        if solid:
            lo, hi = vm.min(0), vm.max(0)
            com, d = (lo + hi) / 2, hi - lo
            inertia = mass / 12.0 * np.diag([d[1] ** 2 + d[2] ** 2, d[0] ** 2 + d[2] ** 2, d[0] ** 2 + d[1] ** 2])
        else:
            com, inertia = shell_inertia(vm, f, mass)
        parts.append({"name": name, "mesh": mesh, "mass": mass, "com": com.tolist(), "inertia": inertia.tolist()})
        all_v.append(vm)
        print(f"{name}: {mass * 1e3:.0f} g, com {np.round(com * 1e3, 1)} mm (trunk frame)")
    v = np.vstack(all_v)
    lo, hi = v.min(0), v.max(0)
    collision = {"pos": ((lo + hi) / 2).tolist(), "size": ((hi - lo) / 2 * COLLISION_SHRINK).tolist()}
    out = {"source": args.src.name, "parts": parts, "collision": collision}
    (args.out / "backpack.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out / 'backpack.json'}")


if __name__ == "__main__":
    main()
