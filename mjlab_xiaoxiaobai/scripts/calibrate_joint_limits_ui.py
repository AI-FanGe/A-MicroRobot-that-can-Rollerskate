from __future__ import annotations

import argparse
import json
import math
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import mujoco.viewer

try:
    from mujoco.glfw import glfw
except Exception:  # pragma: no cover - depends on mujoco wheel extras
    glfw = None


DEFAULT_XML = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "mjlab_xiaoxiaobai"
    / "robot"
    / "xiaoxiaobai"
    / "robot.xml"
)


@dataclass
class JointControl:
    actuator_id: int
    joint_id: int
    actuator_name: str
    joint_name: str
    qposadr: int
    target: float
    home: float
    observed_min: float
    observed_max: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Open a MuJoCo viewer for manually moving actuated joints and recording "
            "observed min/max joint ranges."
        )
    )
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("joint_limit_calibration"),
    )
    parser.add_argument(
        "--wide-ranges",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use a temporary XML with hinge ranges widened to +/- pi.",
    )
    parser.add_argument(
        "--fix-base",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep the floating base fixed while calibrating joint limits.",
    )
    parser.add_argument("--step-deg", type=float, default=1.0)
    parser.add_argument("--coarse-step-deg", type=float, default=5.0)
    return parser.parse_args()


def make_wide_range_xml(xml_path: Path, output_dir: Path) -> Path:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    compiler = root.find("compiler")
    if compiler is not None:
        meshdir = compiler.get("meshdir")
        if meshdir and not Path(meshdir).is_absolute():
            compiler.set("meshdir", str((xml_path.parent / meshdir).resolve()))

    for joint in root.iter("joint"):
        if joint.get("type", "hinge") == "hinge" and joint.get("name"):
            joint.set("range", f"{-math.pi:.9g} {math.pi:.9g}")

    output_dir.mkdir(parents=True, exist_ok=True)
    wide_xml = output_dir / "robot_calibration_wide_ranges.xml"
    tree.write(wide_xml, encoding="utf-8", xml_declaration=True)
    return wide_xml


def object_name(model: mujoco.MjModel, obj_type: mujoco.mjtObj, obj_id: int) -> str:
    return mujoco.mj_id2name(model, obj_type, obj_id) or f"{obj_type.name}_{obj_id}"


def build_controls(model: mujoco.MjModel, data: mujoco.MjData) -> list[JointControl]:
    controls: list[JointControl] = []
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        if joint_id < 0:
            continue
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        qposadr = int(model.jnt_qposadr[joint_id])
        actuator_name = object_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        joint_name = object_name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        home = float(data.qpos[qposadr])
        controls.append(
            JointControl(
                actuator_id=actuator_id,
                joint_id=joint_id,
                actuator_name=actuator_name,
                joint_name=joint_name,
                qposadr=qposadr,
                target=home,
                home=home,
                observed_min=home,
                observed_max=home,
            )
        )
    return controls


def load_model(xml_path: Path, output_dir: Path, wide_ranges: bool) -> tuple[mujoco.MjModel, Path]:
    if wide_ranges:
        xml_path = make_wide_range_xml(xml_path, output_dir)
    return mujoco.MjModel.from_xml_path(str(xml_path)), xml_path


def reset_to_home(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "init_state")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)


def save_outputs(
    output_dir: Path,
    controls: list[JointControl],
    source_xml: Path,
    loaded_xml: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    limits = {}
    for control in controls:
        limits[control.joint_name] = {
            "min_rad": control.observed_min,
            "max_rad": control.observed_max,
            "min_deg": math.degrees(control.observed_min),
            "max_deg": math.degrees(control.observed_max),
            "home_rad": control.home,
            "home_deg": math.degrees(control.home),
            "actuator": control.actuator_name,
        }

    payload = {
        "source_xml": str(source_xml.resolve()),
        "loaded_xml": str(loaded_xml.resolve()),
        "note": "Observed ranges are recorded from manual joint target movement in the UI.",
        "limits": limits,
    }
    json_path = output_dir / "joint_limits_observed.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    snippet_lines = [
        "<!-- Paste these range values onto the matching joints in robot.xml after review. -->"
    ]
    for control in controls:
        snippet_lines.append(
            f'<joint name="{control.joint_name}" '
            f'range="{control.observed_min:.9g} {control.observed_max:.9g}"/>'
        )
    snippet_path = output_dir / "robot_xml_joint_range_snippet.xml"
    snippet_path.write_text("\n".join(snippet_lines) + "\n", encoding="utf-8")

    print(f"\n[SAVED] {json_path}", flush=True)
    print(f"[SAVED] {snippet_path}\n", flush=True)


def print_help() -> None:
    print(
        """
Joint limit calibration controls:
  [ / ]        previous / next joint
  Left/Right   decrease / increase selected joint target by fine step
  Down/Up      decrease / increase selected joint target by coarse step
  R            reset selected joint target to home
  T            reset all joint targets to home
  C            clear observed min/max for selected joint
  S            save observed ranges to JSON and XML snippet
  P            print selected joint status
  H            print this help

Move each joint until it reaches the mechanical/collision limit you accept.
The script records observed min/max automatically while you move.
"""
    , flush=True)


def main() -> None:
    args = parse_args()
    xml_path = args.xml.resolve()
    output_dir = args.output_dir.resolve()
    model, loaded_xml = load_model(xml_path, output_dir, args.wide_ranges)
    data = mujoco.MjData(model)
    reset_to_home(model, data)

    controls = build_controls(model, data)
    if not controls:
        raise RuntimeError("No hinge-joint actuators found.")

    root_qpos = data.qpos[:7].copy() if model.nq >= 7 else None
    root_qvel = data.qvel[:6].copy() if model.nv >= 6 else None
    fine_step = math.radians(args.step_deg)
    coarse_step = math.radians(args.coarse_step_deg)
    selected = {"index": 0}

    def selected_control() -> JointControl:
        return controls[selected["index"]]

    def print_status() -> None:
        control = selected_control()
        q = float(data.qpos[control.qposadr])
        print(
            f"[{selected['index'] + 1:02d}/{len(controls):02d}] {control.joint_name} "
            f"q={math.degrees(q):8.2f} deg target={math.degrees(control.target):8.2f} deg "
            f"observed=[{math.degrees(control.observed_min):8.2f}, "
            f"{math.degrees(control.observed_max):8.2f}] deg"
        , flush=True)

    def key_callback(keycode: int) -> None:
        nonlocal controls
        control = selected_control()
        if glfw is not None and keycode == glfw.KEY_RIGHT:
            control.target += fine_step
        elif glfw is not None and keycode == glfw.KEY_LEFT:
            control.target -= fine_step
        elif glfw is not None and keycode == glfw.KEY_UP:
            control.target += coarse_step
        elif glfw is not None and keycode == glfw.KEY_DOWN:
            control.target -= coarse_step
        elif keycode == ord("]"):
            selected["index"] = (selected["index"] + 1) % len(controls)
            print_status()
        elif keycode == ord("["):
            selected["index"] = (selected["index"] - 1) % len(controls)
            print_status()
        elif keycode in (ord("R"), ord("r")):
            control.target = control.home
            print_status()
        elif keycode in (ord("T"), ord("t")):
            for item in controls:
                item.target = item.home
            print("[RESET] all targets set to home", flush=True)
        elif keycode in (ord("C"), ord("c")):
            q = float(data.qpos[control.qposadr])
            control.observed_min = q
            control.observed_max = q
            print_status()
        elif keycode in (ord("S"), ord("s")):
            save_outputs(output_dir, controls, xml_path, loaded_xml)
        elif keycode in (ord("P"), ord("p")):
            print_status()
        elif keycode in (ord("H"), ord("h")):
            print_help()

    print_help()
    print(f"[INFO] Source XML: {xml_path}", flush=True)
    print(f"[INFO] Loaded XML: {loaded_xml}", flush=True)
    print(f"[INFO] Output dir: {output_dir}", flush=True)
    print_status()

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        last_print = time.time()
        while viewer.is_running():
            if args.fix_base and root_qpos is not None:
                data.qpos[:7] = root_qpos
            if args.fix_base and root_qvel is not None:
                data.qvel[:6] = root_qvel

            for control in controls:
                data.ctrl[control.actuator_id] = control.target

            mujoco.mj_step(model, data)

            for control in controls:
                q = float(data.qpos[control.qposadr])
                control.observed_min = min(control.observed_min, q)
                control.observed_max = max(control.observed_max, q)

            now = time.time()
            if now - last_print > 5.0:
                print_status()
                last_print = now

            viewer.sync()
            time.sleep(model.opt.timestep)

    save_outputs(output_dir, controls, xml_path, loaded_xml)


if __name__ == "__main__":
    main()
