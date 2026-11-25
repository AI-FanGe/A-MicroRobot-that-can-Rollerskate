from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import mujoco.viewer


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XML = REPO_ROOT / "src" / "mjlab_xiaoxiaobai" / "robot" / "xiaoxiaobai" / "robot.xml"
DEFAULT_LIMITS = REPO_ROOT / "joint_limit_calibration" / "manual_adjusted" / "joint_limits_selected.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "joint_limit_calibration" / "slider_view"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Open MuJoCo's native viewer with actuator sliders using the current joint "
            "limits from robot.xml."
        )
    )
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--limits-json", type=Path, default=DEFAULT_LIMITS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--fix-base",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove the trunk freejoint in a temporary XML so the robot stays fixed while checking joints.",
    )
    parser.add_argument(
        "--wide-ranges",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Temporarily widen hinge ranges to +/- pi. Disabled by default to use current limits.",
    )
    return parser.parse_args()


def prepare_xml(xml_path: Path, output_dir: Path, fix_base: bool, wide_ranges: bool) -> Path:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    compiler = root.find("compiler")
    if compiler is not None:
        meshdir = compiler.get("meshdir")
        if meshdir and not Path(meshdir).is_absolute():
            compiler.set("meshdir", str((xml_path.parent / meshdir).resolve()))

    if fix_base:
        for body in root.iter("body"):
            for freejoint in list(body.findall("freejoint")):
                body.remove(freejoint)

    if wide_ranges:
        for joint in root.iter("joint"):
            if joint.get("type", "hinge") == "hinge" and joint.get("name"):
                joint.set("range", f"{-math.pi:.9g} {math.pi:.9g}")

    output_dir.mkdir(parents=True, exist_ok=True)
    prepared_xml = output_dir / "robot_slider_view.xml"
    tree.write(prepared_xml, encoding="utf-8", xml_declaration=True)
    return prepared_xml


def load_home_positions(limits_json: Path) -> dict[str, float]:
    if not limits_json.exists():
        return {}

    payload = json.loads(limits_json.read_text(encoding="utf-8"))
    homes: dict[str, float] = {}
    for joint_name, values in payload.get("limits", {}).items():
        if "home_rad" in values:
            homes[joint_name] = float(values["home_rad"])
        elif "home_deg" in values:
            homes[joint_name] = math.radians(float(values["home_deg"]))
    return homes


def apply_home_pose(model: mujoco.MjModel, data: mujoco.MjData, homes: dict[str, float]) -> None:
    for joint_name, qpos in homes.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0 or model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
            continue

        qposadr = int(model.jnt_qposadr[joint_id])
        data.qpos[qposadr] = qpos

        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
        if actuator_id >= 0:
            data.ctrl[actuator_id] = qpos

    mujoco.mj_forward(model, data)


def main() -> None:
    args = parse_args()
    xml_path = args.xml.resolve()
    output_dir = args.output_dir.resolve()
    prepared_xml = prepare_xml(xml_path, output_dir, args.fix_base, args.wide_ranges)

    model = mujoco.MjModel.from_xml_path(str(prepared_xml))
    data = mujoco.MjData(model)
    apply_home_pose(model, data, load_home_positions(args.limits_json.resolve()))

    print(f"[INFO] Source XML: {xml_path}", flush=True)
    print(f"[INFO] Loaded XML: {prepared_xml}", flush=True)
    print("[INFO] Use MuJoCo viewer's Control panel sliders to move the joints.", flush=True)
    print("[INFO] Current robot.xml joint ranges are used unless --wide-ranges is passed.", flush=True)

    mujoco.viewer.launch(model, data=data)


if __name__ == "__main__":
    main()
