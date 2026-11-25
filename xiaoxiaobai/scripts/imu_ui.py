#!/usr/bin/env python3
"""Live IMU attitude UI.

Run this on your Ubuntu PC. It SSHs into the controller, reads the IMU, and
opens a local window with roll / pitch / yaw.

    python3 scripts/imu_ui.py
    python3 scripts/imu_ui.py --host 192.168.2.42 --user user
    python3 scripts/imu_ui.py --host xiaoxiaobai --password 'your_password'
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from typing import Any

try:
    import tkinter as tk
    from tkinter import scrolledtext
except ImportError:
    print(
        "需要 tkinter。Ubuntu 上执行:  sudo apt install python3-tk",
        file=sys.stderr,
    )
    sys.exit(1)


LOG_PATH = "/tmp/xiaoxiaobai_imu_ui.log"

REMOTE_STREAMER = r"""
import json, math, os, sys, time, traceback
from pathlib import Path

def emit(payload):
    print(json.dumps(payload), flush=True)

try:
    home = Path.home()
    candidates = [
        Path.cwd(),
        Path.cwd() / "xiaoxiaobai",
        home / "xiaoxiaobai",
        home / "bot" / "xiaoxiaobai",
    ]
    root = next((p for p in candidates if (p / "src" / "imu_reader.py").is_file()), None)
    if root is None:
        emit({"error": "xiaoxiaobai src/imu_reader.py not found", "home": str(home), "cwd": os.getcwd()})
        sys.exit(1)

    sys.path.insert(0, str(root / "src"))
    os.chdir(root)

    from constants import IMU_I2C_BUS
    import bmi088.bmi088 as bmi_mod
    from smbus3 import SMBus
    import glob, subprocess, grp, pwd

    def i2c_present(bus, addr):
        for fn in ("write_quick", "read_byte"):
            try:
                getattr(bus, fn)(addr)
                return True
            except Exception:
                pass
        try:
            bus.read_byte_data(addr, 0x00)
            return True
        except Exception:
            return False

    def scan_bus(n):
        try:
            bus = SMBus(n)
        except Exception as exc:
            return {"open_error": str(exc)}
        found = []
        for addr in range(0x03, 0x78):
            if i2c_present(bus, addr):
                found.append("0x%02x" % addr)
        try:
            bus.close()
        except Exception:
            pass
        return found

    i2c_nodes = sorted(glob.glob("/dev/i2c-*"))
    spi_nodes = sorted(glob.glob("/dev/spidev*"))
    adapters = {}
    for path in glob.glob("/sys/class/i2c-adapter/i2c-*/name"):
        try:
            adapters[path.split("/")[-2]] = Path(path).read_text().strip()
        except Exception:
            pass
    detect = {}
    for node in i2c_nodes:
        n = int(node.split("-")[-1])
        detect[str(n)] = scan_bus(n)
        try:
            out = subprocess.check_output(["i2cdetect", "-y", str(n)], text=True, timeout=8)
            detect[str(n) + "_i2cdetect"] = out
        except Exception:
            pass

    acc = gyro = used_bus = None
    for node in i2c_nodes:
        n = int(node.split("-")[-1])
        found = detect.get(str(n)) or []
        if not isinstance(found, list):
            continue
        nums = [int(x, 16) for x in found]
        a = next((x for x in (0x19, 0x18) if x in nums), None)
        g = next((x for x in (0x68, 0x69) if x in nums), None)
        if a is not None:
            acc, gyro, used_bus = a, g, n
            if n == IMU_I2C_BUS:
                break

    emit({"debug": "i2c scan", "i2c_nodes": i2c_nodes, "spi_nodes": spi_nodes, "adapters": adapters, "detect": {k: v for k, v in detect.items() if not k.endswith("_i2cdetect")}, "picked": {"bus": used_bus, "acc": acc, "gyro": gyro}})

    if acc is None:
        lines = [
            "IMU not found on any I2C bus.",
            "i2c nodes: " + (", ".join(i2c_nodes) or "none"),
            "spi nodes: " + (", ".join(spi_nodes) or "none"),
            "adapters: " + json.dumps(adapters),
        ]
        for k, v in detect.items():
            lines.append("bus %s: %s" % (k, v if not isinstance(v, str) else v.strip().replace("\n", " | ")))
        try:
            me = pwd.getpwuid(os.getuid()).pw_name
            groups = [g.gr_name for g in grp.getgrall() if me in g.gr_mem] + [grp.getgrgid(os.getgid()).gr_name]
            lines.append("user %s groups: %s" % (me, ",".join(groups)))
        except Exception:
            pass
        emit({"error": "\n".join(lines)})
        sys.exit(1)

    bus = SMBus(used_bus)
    try:
        bmi_mod.ACC_ADDRESS = acc
        bmi_mod.GYRO_ADDRESS = gyro if gyro is not None else 0x68
        bus.write_byte_data(acc, 0x7D, 0x04)
        time.sleep(0.05)
        try:
            bus.write_byte_data(acc, 0x7C, 0x00)
        except OSError:
            pass
    finally:
        bus.close()

    from imu_reader import ThreadedIMUReader, imu_quat_to_body

    def euler_deg(w, x, y, z):
        roll = math.degrees(math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))))
        yaw = math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        return roll, pitch, yaw

    emit({"debug": "imu stream starting", "root": str(root), "bus": used_bus, "acc": hex(acc), "gyro": hex(bmi_mod.GYRO_ADDRESS)})
    reader = ThreadedIMUReader(i2c_bus=used_bus, frequency_hz=200.0)
    reader.start()
    try:
        while True:
            sample = reader.get_latest()
            w, x, y, z = sample.quat
            bw, bx, by, bz = imu_quat_to_body((w, x, y, z))
            roll, pitch, yaw = euler_deg(w, x, y, z)
            b_roll, b_pitch, b_yaw = euler_deg(bw, bx, by, bz)
            age_ms = max(0.0, (time.perf_counter() - sample.timestamp_s) * 1000.0)
            emit({
                "valid": bool(sample.valid),
                "age_ms": round(age_ms, 1),
                "errors": int(sample.error_count),
                "roll": round(roll, 2),
                "pitch": round(pitch, 2),
                "yaw": round(yaw, 2),
                "body_roll": round(b_roll, 2),
                "body_pitch": round(b_pitch, 2),
                "body_yaw": round(b_yaw, 2),
                "acc": [round(v, 4) for v in sample.acc],
                "gyro": [round(v, 4) for v in sample.gyro],
            })
            time.sleep(0.05)
    finally:
        reader.stop()
except Exception:
    emit({"error": traceback.format_exc()})
    sys.exit(1)
"""

SSH_NOISE = (
    "permanently added",
    "warning: ",
    "debug1:",
    "debug2:",
    "debug3:",
)


def _log(line: str) -> None:
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(line.rstrip() + "\n")
    except OSError:
        pass


class ImuStream:
    def __init__(self, host: str, user: str | None, password: str | None) -> None:
        self.host = f"{user}@{host}" if user and "@" not in host else host
        self.password = password
        self.proc: subprocess.Popen[str] | None = None
        self._askpass: str | None = None
        self.latest: dict[str, Any] = {}
        self.error: str | None = None
        self.connected = False
        self._stop = threading.Event()

    def start(self) -> None:
        try:
            os.remove(LOG_PATH)
        except OSError:
            pass
        _log(f"connecting to {self.host}")
        env = os.environ.copy()
        extra_opts: list[str] = []
        if self.password:
            handle, path = tempfile.mkstemp(prefix="imu_askpass_", text=True)
            os.close(handle)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("#!/bin/sh\n")
                fh.write("cat <<'EOF'\n")
                fh.write(self.password + "\n")
                fh.write("EOF\n")
            os.chmod(path, 0o700)
            self._askpass = path
            env["DISPLAY"] = env.get("DISPLAY") or ":0"
            env["SSH_ASKPASS"] = path
            env["SSH_ASKPASS_REQUIRE"] = "force"
            extra_opts += [
                "-o",
                "PreferredAuthentications=password",
                "-o",
                "PubkeyAuthentication=no",
            ]

        remote = (
            "cat > /tmp/xiaoxiaobai_imu_stream.py && "
            "cd \"$HOME/xiaoxiaobai\" && "
            "if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python3; fi && "
            "PYTHONPATH=src \"$PY\" -u /tmp/xiaoxiaobai_imu_stream.py"
        )
        cmd = [
            "ssh",
            "-T",
            "-o",
            "ConnectTimeout=8",
            "-o",
            "ServerAliveInterval=5",
            "-o",
            "StrictHostKeyChecking=accept-new",
            *extra_opts,
            self.host,
            remote,
        ]
        _log("ssh " + " ".join(cmd[:-1] + ["<remote-python>"]))
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        assert self.proc.stdin is not None
        self.proc.stdin.write(REMOTE_STREAMER)
        self.proc.stdin.close()
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()
        threading.Thread(target=self._watch_exit, daemon=True).start()

    def _append_error(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        if any(token in text.lower() for token in SSH_NOISE):
            _log("ssh-noise: " + text)
            return
        _log("error: " + text)
        if self.error:
            self.error = self.error + "\n" + text
        else:
            self.error = text

    def _read_stdout(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        for line in self.proc.stdout:
            if self._stop.is_set():
                break
            line = line.strip()
            if not line:
                continue
            _log("stdout: " + line[:500])
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                self._append_error(line)
                continue
            if payload.get("error") and "roll" not in payload:
                self._append_error(str(payload["error"]))
                continue
            if payload.get("debug"):
                _log("debug: " + line)
                continue
            self.latest = payload
            self.connected = True

    def _read_stderr(self) -> None:
        assert self.proc is not None and self.proc.stderr is not None
        for line in self.proc.stderr:
            if self._stop.is_set():
                break
            self._append_error(line.rstrip())

    def _watch_exit(self) -> None:
        if self.proc is None:
            return
        code = self.proc.wait()
        _log(f"ssh exit {code}")
        if code not in (0, None) and not self.latest:
            self._append_error(f"SSH/远程进程退出码 {code}")

    def stop(self) -> None:
        self._stop.set()
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self._askpass:
            try:
                os.remove(self._askpass)
            except OSError:
                pass


class AttitudeView:
    def __init__(self, stream: ImuStream) -> None:
        self.stream = stream
        self.root = tk.Tk()
        self.root.title("Xiaoxiaobai IMU")
        self.root.geometry("900x640+80+80")
        self.root.minsize(720, 520)
        self.root.configure(bg="#11141a")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.status = tk.StringVar(value="正在连接主控 …")
        tk.Label(
            self.root,
            textvariable=self.status,
            fg="#9aa4b2",
            bg="#11141a",
            font=("Sans", 12),
        ).pack(pady=(12, 4))

        self.log_box = scrolledtext.ScrolledText(
            self.root,
            height=6,
            wrap="word",
            bg="#1a1f29",
            fg="#ffb4a8",
            insertbackground="#ffb4a8",
            font=("Sans", 11),
            relief="flat",
        )
        self.log_box.pack(fill="x", padx=24, pady=(0, 8))
        self.log_box.insert("1.0", f"报错会显示在这里，同时写入 {LOG_PATH}\n")
        self.log_box.configure(state="disabled")

        row = tk.Frame(self.root, bg="#11141a")
        row.pack(fill="x", padx=24, pady=8)
        self.roll_var = tk.StringVar(value="—")
        self.pitch_var = tk.StringVar(value="—")
        self.yaw_var = tk.StringVar(value="—")
        self._metric(row, "ROLL", self.roll_var, "#ff6b6b").pack(side="left", expand=True, fill="x")
        self._metric(row, "PITCH", self.pitch_var, "#4ecdc4").pack(side="left", expand=True, fill="x")
        self._metric(row, "YAW", self.yaw_var, "#ffe66d").pack(side="left", expand=True, fill="x")

        self.canvas = tk.Canvas(self.root, bg="#0b0d12", highlightthickness=0, height=280)
        self.canvas.pack(fill="both", expand=True, padx=24, pady=(4, 8))

        self.detail = tk.StringVar(value="等待 IMU 数据")
        tk.Label(
            self.root,
            textvariable=self.detail,
            fg="#6f7a88",
            bg="#11141a",
            font=("Sans", 11),
            justify="left",
        ).pack(anchor="w", padx=28, pady=(0, 16))

        self._shown_error: str | None = None
        self.root.after(50, self._tick)

    def _set_log(self, text: str) -> None:
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.insert("1.0", text)
        self.log_box.configure(state="disabled")

    def _metric(self, parent: tk.Widget, title: str, var: tk.StringVar, color: str) -> tk.Frame:
        box = tk.Frame(parent, bg="#1a1f29", padx=12, pady=10)
        tk.Label(box, text=title, fg=color, bg="#1a1f29", font=("Sans", 11, "bold")).pack()
        tk.Label(box, textvariable=var, fg="#f4f7fb", bg="#1a1f29", font=("Sans", 28, "bold")).pack()
        return box

    def _tick(self) -> None:
        data = self.stream.latest
        if self.stream.error and self.stream.error != self._shown_error:
            self._shown_error = self.stream.error
            self._set_log(self.stream.error)
        if self.stream.error and not data:
            self.status.set("连接失败 / IMU 读取失败")
            self.detail.set(f"完整日志: {LOG_PATH}")
        elif not data:
            self.status.set("已 SSH，等待 IMU 第一帧 …")
        else:
            valid = "有效" if data.get("valid") else "无效"
            self.status.set(
                f"IMU  {valid}    延迟 {data.get('age_ms', 0):.0f} ms    错误 {data.get('errors', 0)}"
            )
            self.roll_var.set(f"{data.get('roll', 0):+.1f}°")
            self.pitch_var.set(f"{data.get('pitch', 0):+.1f}°")
            self.yaw_var.set(f"{data.get('yaw', 0):+.1f}°")
            acc = data.get("acc") or [0, 0, 0]
            gyro = data.get("gyro") or [0, 0, 0]
            self.detail.set(
                f"机体系  roll={data.get('body_roll', 0):+.1f}°  "
                f"pitch={data.get('body_pitch', 0):+.1f}°  "
                f"yaw={data.get('body_yaw', 0):+.1f}°    "
                f"acc=({acc[0]:+.2f}, {acc[1]:+.2f}, {acc[2]:+.2f}) g    "
                f"gyro=({gyro[0]:+.2f}, {gyro[1]:+.2f}, {gyro[2]:+.2f}) rad/s"
            )
            self._draw_horizon(
                float(data.get("roll", 0.0)),
                float(data.get("pitch", 0.0)),
                float(data.get("yaw", 0.0)),
            )
        self.root.after(50, self._tick)

    def _draw_horizon(self, roll_deg: float, pitch_deg: float, yaw_deg: float) -> None:
        c = self.canvas
        c.delete("all")
        w = max(c.winfo_width(), 1)
        h = max(c.winfo_height(), 1)
        cx, cy = w / 2.0, h / 2.0
        radius = min(w, h) * 0.40
        roll = math.radians(roll_deg)
        pitch_shift = max(-radius, min(radius, (pitch_deg / 45.0) * radius * 0.65))

        def rot(px: float, py: float) -> tuple[float, float]:
            x, y = px - cx, py - cy
            return (
                cx + x * math.cos(roll) - y * math.sin(roll),
                cy + x * math.sin(roll) + y * math.cos(roll),
            )

        c.create_rectangle(0, 0, w, h, fill="#0b0d12", outline="")
        horizon = cy + pitch_shift
        sky = [rot(-w, -h), rot(w * 2, -h), rot(w * 2, horizon), rot(-w, horizon)]
        ground = [rot(-w, horizon), rot(w * 2, horizon), rot(w * 2, h * 2), rot(-w, h * 2)]
        c.create_polygon(*[v for p in sky for v in p], fill="#2b6cb0", outline="")
        c.create_polygon(*[v for p in ground for v in p], fill="#8d5a2b", outline="")

        c.create_oval(cx - radius, cy - radius, cx + radius, cy + radius, outline="#e8eef7", width=3)
        c.create_line(cx - 52, cy, cx - 12, cy, fill="#f4f7fb", width=3)
        c.create_line(cx + 12, cy, cx + 52, cy, fill="#f4f7fb", width=3)
        c.create_oval(cx - 4, cy - 4, cx + 4, cy + 4, fill="#f4f7fb", outline="")

        heading = ((yaw_deg + 180.0) % 360.0) - 180.0
        bar_y = h - 36
        c.create_rectangle(40, bar_y - 16, w - 40, bar_y + 16, fill="#1a1f29", outline="#3a4250")
        c.create_text(
            w / 2,
            bar_y,
            text=f"YAW  {heading:+.1f}°",
            fill="#ffe66d",
            font=("Sans", 13, "bold"),
        )

    def run(self) -> None:
        self.root.mainloop()

    def _on_close(self) -> None:
        self.stream.stop()
        self.root.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SSH to the controller and show IMU roll/pitch/yaw")
    parser.add_argument("--host", default="xiaoxiaobai", help="SSH host, default xiaoxiaobai")
    parser.add_argument("--user", default=None, help="SSH user, e.g. user")
    parser.add_argument("--password", default=None, help="SSH password if you have no key")
    return parser.parse_args()


def main() -> None:
    if shutil.which("ssh") is None:
        print("找不到 ssh 命令", file=sys.stderr)
        sys.exit(1)
    args = parse_args()
    stream = ImuStream(host=args.host, user=args.user, password=args.password)
    stream.start()
    AttitudeView(stream).run()
    stream.stop()


if __name__ == "__main__":
    main()
