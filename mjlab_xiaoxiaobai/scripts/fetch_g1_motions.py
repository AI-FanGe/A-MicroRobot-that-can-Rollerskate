"""Download G1 motion clips from Hugging Face and write them in the LAFAN1 G1 CSV layout (30 fps,
root xyz, root quat xyzw, 29 joints) that dance_feasibility.py / retarget_g1_to_xiaoxiaobai.py read.

Sources:
  - CMRobot/MotionDecode (ChingMu optical mocap retargeted to G1): 120 fps CSV with a header,
    root quat wxyz. Takes come in groups of 5 of the same move; --moves picks that many moves
    (the first take of each) per category, at least one per subcategory.
  - openhe/g1-retargeted-motions: PHC-style pickles, 30 fps, 23 DOF (no waist roll/pitch, no
    wrist pitch/yaw: those are written as 0), root quat xyzw.

usage: fetch_g1_motions.py [--out data/motions/web] [--moves 3]
Writes <out>/<name>.csv and <out>/index.json ({name: {"label", "source"}}).
"""
import argparse
import io
import json
import pickle
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

HF = "https://huggingface.co"
FPS = 30
MOTIONDECODE = {
    # (category path under samples/, label)
    "1.14.Dance_and_Performance/1.14.3.Pop_Dance/1.14.3.1.Locking": "流行舞·Locking",
    "1.14.Dance_and_Performance/1.14.3.Pop_Dance/1.14.3.2.Hip_Hop": "流行舞·Hip-hop",
    "1.14.Dance_and_Performance/1.14.3.Pop_Dance/1.14.3.3.Waacking_Pose_Line_Throw": "流行舞·Waacking",
    "1.14.Dance_and_Performance/1.14.3.Pop_Dance/1.14.3.4.Jazz_Dance": "流行舞·爵士",
    "1.14.Dance_and_Performance/1.14.2.Rhythm_Movement/1.14.2.1.Upper_Body_Rhythm": "律动·上身",
    "1.14.Dance_and_Performance/1.14.2.Rhythm_Movement/1.14.2.2.Lower_Body_Rhythm": "律动·下身",
    "1.14.Dance_and_Performance/1.14.2.Rhythm_Movement/1.14.2.3.Full_Body_Rhythm": "律动·全身",
    "1.14.Dance_and_Performance/1.14.4.Emotional_Expression_Action/1.14.4.1.Joy": "表演·欢乐",
    "1.14.Dance_and_Performance/1.14.4.Emotional_Expression_Action/1.14.4.4.Surprise": "表演·惊喜",
    "1.14.Dance_and_Performance/1.14.1.Stage_Performance_Basics/1.14.1.5.Curtain_Call": "舞台·谢幕",
    "1.14.Dance_and_Performance/1.14.1.Stage_Performance_Basics/1.14.1.2.Interaction_Action": "舞台·互动",
    "1.14.Dance_and_Performance/1.14.8.Folk_Dance/1.14.8.1.Mongolian_Elements": "民族舞·蒙古",
    "1.14.Dance_and_Performance/1.14.8.Folk_Dance/1.14.8.2.Tibetan": "民族舞·藏族",
    "1.14.Dance_and_Performance/1.14.8.Folk_Dance/1.14.8.3.Uyghur": "民族舞·新疆",
    "1.14.Dance_and_Performance/1.14.8.Folk_Dance/1.14.8.4.Korean_Ethnic": "民族舞·朝鲜族",
    "1.14.Dance_and_Performance/1.14.8.Folk_Dance/1.14.8.5.Dai_Ethnic": "民族舞·傣族",
    "1.10.Emotional_Expressions_and_Social_Behaviors/1.10.8.Performative_Emotion": "情绪·表演性",
    "1.10.Emotional_Expressions_and_Social_Behaviors/1.10.1.Positive_Emotion": "情绪·积极",
    "1.10.Emotional_Expressions_and_Social_Behaviors/1.10.6.Dramatic_Emotional_Gestures": "情绪·夸张",
    "1.8.Social_and_Interpersonal_Interaction/1.8.1.Basic_Greetings_and_Etiquette": "社交·打招呼",
    "2.4.Competitive_Interaction/2.4.1.Box_Ing": "拳击",
}
OPENHE = {
    "kungfu_retargeted/Bruce_Lee_pose.pkl": "功夫·李小龙",
    "kungfu_retargeted/Charleston_dance.pkl": "查尔斯顿舞",
    "kungfu_retargeted/Hooks_punch.pkl": "功夫·勾拳",
    "kungfu_retargeted/Horse-stance_punch.pkl": "功夫·马步冲拳",
    "dance_db_retargeted/Andria_Happy_v1_C3D.pkl": "情绪·开心",
    "dance_db_retargeted/Andria_Excited_v1_C3D.pkl": "情绪·兴奋",
    "dance_db_retargeted/Andria_Relaxed_v1_C3D.pkl": "情绪·放松",
    "dance_db_retargeted/AnnaCortesi_RandB_C3D.pkl": "R&B",
    "dance_db_retargeted/ANDREAS_Sta_tria_Syrtos.pkl": "希腊舞·Syrtos",
    "dance_db_retargeted/ANDREAS_Hasaposerviko.pkl": "希腊舞·Hasaposerviko",
}
# G1 29-DOF columns the 23-DOF PHC layout fills (the rest stay 0).
DOF23_TO_29 = list(range(12)) + [12] + [15, 16, 17, 18, 19] + [22, 23, 24, 25, 26]


def get_json(url: str):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def tree(repo: str, path: str) -> list[dict]:
    return get_json(f"{HF}/api/datasets/{repo}/tree/main/{urllib.parse.quote(path)}")


def download(repo: str, path: str, tries: int = 4) -> bytes:
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(f"{HF}/datasets/{repo}/resolve/main/{urllib.parse.quote(path)}", timeout=120) as r:
                return r.read()
        except OSError:
            if attempt == tries - 1:
                raise
            time.sleep(3 * (attempt + 1))


def pick(repo: str, path: str, moves: int) -> list[str]:
    """First take of `moves` moves under path, spread over its subfolders (one level down)."""
    entries = tree(repo, path)
    groups = [sorted(y["path"] for y in tree(repo, x["path"]) if y["path"].endswith(".csv"))
              for x in entries if x["type"] == "directory"]
    groups.append(sorted(x["path"] for x in entries if x["path"].endswith(".csv")))
    groups = [g for g in groups if g]
    per = -(-moves // max(1, len(groups)))
    return [f for g in groups for f in g[::5][:per]][:max(moves, len(groups))]


def save(out: Path, name: str, rows: np.ndarray) -> None:
    np.savetxt(out / f"{name}.csv", rows, delimiter=",", fmt="%.6f")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/motions/web")
    ap.add_argument("--moves", type=int, default=3)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index_path = out / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}

    for cat, label in MOTIONDECODE.items():
        try:
            files = pick("CMRobot/MotionDecode", f"samples/{cat}", args.moves)
        except OSError as exc:
            print(f"skip {label}: {exc}")
            continue
        for k, f in enumerate(files):
            name = Path(f).stem
            if not (out / f"{name}.csv").exists():
                text = download("CMRobot/MotionDecode", f).decode()
                raw = np.loadtxt(text.splitlines()[1:], delimiter=",")[::4]  # 120 -> 30 fps
                save(out, name, np.concatenate([raw[:, :3], raw[:, [4, 5, 6, 3]], raw[:, 7:]], axis=1))
            index[name] = {"label": f"{label} {k + 1}", "source": f"CMRobot/MotionDecode {f}"}
            print(f"{name}: {index[name]['label']}")
        index_path.write_text(json.dumps(index, ensure_ascii=False, indent=1))

    for f, label in OPENHE.items():
        name = Path(f).stem
        if not (out / f"{name}.csv").exists():
            blob = download("openhe/g1-retargeted-motions", f)
            try:
                data = pickle.loads(blob)
            except (ModuleNotFoundError, pickle.UnpicklingError):  # joblib dumps: uvx --with numpy --with joblib python ...
                import joblib
                data = joblib.load(io.BytesIO(blob))
            m = next(iter(data.values()))
            assert int(m.get("fps", FPS)) == FPS, m.get("fps")
            dof = np.zeros((len(m["dof"]), 29))
            dof[:, DOF23_TO_29] = m["dof"]
            save(out, name, np.concatenate([m["root_trans_offset"], m["root_rot"], dof], axis=1))
        index[name] = {"label": label, "source": f"openhe/g1-retargeted-motions {f}"}
        print(f"{name}: {label}")
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
