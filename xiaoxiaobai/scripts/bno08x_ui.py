#!/usr/bin/env python3
"""Temporary IMU test UI.

Does NOT change robot control / IMU code. Run on the Ubuntu PC; it SSHs
into the controller, reads the module over the existing I2C1 pins, and opens a
local window with roll / pitch / yaw plus a rotating cube.

Wiring (same I2C pins as before):

    测试 IMU        主控 40-pin
    --------         --------------
    VIN / 3V3        pin 1  3.3V     (do not use 5V)
    GND              pin 6  GND
    SCL              pin 5  GPIO3 / SCL
    SDA              pin 3  GPIO2 / SDA

Leave PS0/PS1 in I2C mode (both low / default on most of these boards).
I2C address is usually 0x4A, sometimes 0x4B.

    python3 scripts/bno08x_ui.py --host 192.168.2.42 --user user --password '...'
    make bno08x-ui HOST=xiaoxiaobai
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
    print("需要 tkinter。Ubuntu 上执行:  sudo apt install python3-tk", file=sys.stderr)
    sys.exit(1)


LOG_PATH = "/tmp/xiaoxiaobai_bno08x_ui.log"

REMOTE_STREAMER = r"""
import fcntl, glob, json, math, os, struct, sys, time, traceback

def emit(payload):
    print(json.dumps(payload), flush=True)

I2C_SLAVE = 0x0703
I2C_SLAVE_FORCE = 0x0706
CHAN_EXE = 1
CHAN_CONTROL = 2
CHAN_REPORTS = 3
SET_FEATURE = 0xFD
REPORT_GAME_RV = 0x08
REPORT_RV = 0x05
QUAT_SCALE = 1.0 / (1 << 14)

class Bno08xI2c:
    def __init__(self, bus, addr):
        self.fd = os.open("/dev/i2c-%d" % bus, os.O_RDWR)
        try:
            fcntl.ioctl(self.fd, I2C_SLAVE, addr)
        except OSError:
            fcntl.ioctl(self.fd, I2C_SLAVE_FORCE, addr)
        self.seq = [0] * 6
        self.addr = addr
        self.bus = bus

    def close(self):
        try:
            os.close(self.fd)
        except OSError:
            pass

    def send(self, channel, data):
        data = bytes(data)
        length = len(data) + 4
        pkt = struct.pack("<HBB", length, channel, self.seq[channel] & 0xFF) + data
        self.seq[channel] = (self.seq[channel] + 1) & 0xFF
        os.write(self.fd, pkt)

    def read_packet(self):
        header = os.read(self.fd, 4)
        if len(header) < 4:
            return None
        raw_len, channel, seqn = struct.unpack("<HBB", header)
        total = raw_len & 0x7FFF
        if total == 0 or total == 0x7FFF:
            return None
        if channel > 5:
            return None
        self.seq[channel] = seqn
        buf = os.read(self.fd, total)
        if len(buf) < 4:
            return None
        payload = buf[4:] if len(buf) >= total else buf
        if len(buf) == total - 4:
            payload = buf
        return channel, payload

    def drain(self, seconds):
        deadline = time.time() + seconds
        while time.time() < deadline:
            try:
                pkt = self.read_packet()
            except OSError:
                time.sleep(0.01)
                continue
            if pkt is None:
                time.sleep(0.005)

    def enable_game_rv(self, interval_us=20000):
        report = bytearray(17)
        report[0] = SET_FEATURE
        report[1] = REPORT_GAME_RV
        struct.pack_into("<I", report, 5, interval_us)
        self.send(CHAN_CONTROL, report)

    def parse_quat(self, payload):
        i = 0
        found = None
        while i < len(payload):
            rid = payload[i]
            if rid == 0xFB:
                i += 5
                continue
            if rid == REPORT_GAME_RV and i + 12 <= len(payload):
                iq, jq, kq, rq = struct.unpack_from("<hhhh", payload, i + 4)
                found = (rq * QUAT_SCALE, iq * QUAT_SCALE, jq * QUAT_SCALE, kq * QUAT_SCALE)
                i += 12
                continue
            if rid == REPORT_RV and i + 14 <= len(payload):
                iq, jq, kq, rq = struct.unpack_from("<hhhh", payload, i + 4)
                found = (rq * QUAT_SCALE, iq * QUAT_SCALE, jq * QUAT_SCALE, kq * QUAT_SCALE)
                i += 14
                continue
            break
        return found


def i2c_ack(bus, addr):
    try:
        fd = os.open("/dev/i2c-%d" % bus, os.O_RDWR)
    except OSError:
        return False
    try:
        try:
            fcntl.ioctl(fd, I2C_SLAVE, addr)
        except OSError:
            fcntl.ioctl(fd, I2C_SLAVE_FORCE, addr)
        os.read(fd, 4)
        return True
    except OSError:
        return False
    finally:
        os.close(fd)

try:
    nodes = sorted(glob.glob("/dev/i2c-*"))
    buses = []
    for node in nodes:
        try:
            buses.append(int(node.split("-")[-1]))
        except ValueError:
            pass
    found = []
    for bus in buses:
        for addr in (0x4A, 0x4B):
            if i2c_ack(bus, addr):
                found.append((bus, addr))
    emit({"debug": "bno08x scan", "i2c_nodes": nodes, "found": [{"bus": b, "addr": hex(a)} for b, a in found]})
    if not found:
        emit({"error": "测试 IMU 未找到。在 %s 上查找 0x4A/0x4B。检查 VIN=3.3V、GND、SDA=pin3、SCL=pin5、I2C 模式。" % (", ".join(nodes) or "no /dev/i2c-*")})
        sys.exit(1)

    bus, addr = found[0]
    imu = Bno08xI2c(bus, addr)
    try:
        imu.send(CHAN_EXE, [1])
        time.sleep(0.8)
        imu.drain(0.4)
        imu.enable_game_rv(20000)
        time.sleep(0.2)
        last_q = (1.0, 0.0, 0.0, 0.0)
        errors = 0
        while True:
            try:
                pkt = imu.read_packet()
            except OSError as exc:
                errors += 1
                emit({"error": "I2C read failed: %s (控制器硬件 I2C 常常撑不住这颗 IMU 的时钟拉伸). Try slower I2C or i2c-gpio overlay." % exc})
                time.sleep(0.05)
                continue
            if pkt is None:
                time.sleep(0.005)
                continue
            channel, payload = pkt
            if channel in (CHAN_REPORTS, 4) and payload:
                q = imu.parse_quat(payload)
                if q is not None:
                    last_q = q
            w, x, y, z = last_q
            n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
            w, x, y, z = w / n, x / n, y / n, z / n
            roll = math.degrees(math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
            pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))))
            yaw = math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
            emit({
                "valid": True,
                "bus": bus,
                "addr": hex(addr),
                "errors": errors,
                "roll": round(roll, 2),
                "pitch": round(pitch, 2),
                "yaw": round(yaw, 2),
                "quat": [round(w, 5), round(x, 5), round(y, 5), round(z, 5)],
            })
            time.sleep(0.03)
    finally:
        imu.close()
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
            handle, path = tempfile.mkstemp(prefix="bno08x_askpass_", text=True)
            os.close(handle)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("#!/bin/sh\ncat <<'EOF'\n")
                fh.write(self.password + "\nEOF\n")
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
        remote = "python3 -u -"
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
        self.error = (self.error + "\n" + text) if self.error else text

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
                continue
            self.latest = payload

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
        self.root.title("IMU 测试")
        self.root.geometry("900x680+80+80")
        self.root.minsize(720, 540)
        self.root.configure(bg="#11141a")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.status = tk.StringVar(value="正在连接主控上的测试 IMU …")
        tk.Label(self.root, textvariable=self.status, fg="#9aa4b2", bg="#11141a", font=("Sans", 12)).pack(pady=(12, 4))

        self.log_box = scrolledtext.ScrolledText(
            self.root, height=5, wrap="word", bg="#1a1f29", fg="#ffb4a8",
            font=("Sans", 11), relief="flat",
        )
        self.log_box.pack(fill="x", padx=24, pady=(0, 8))
        self.log_box.insert("1.0", f"测试脚本。报错显示在这里，并写入 {LOG_PATH}\n接线: VIN=3.3V  GND  SDA=Pin3  SCL=Pin5\n")
        self.log_box.configure(state="disabled")

        row = tk.Frame(self.root, bg="#11141a")
        row.pack(fill="x", padx=24, pady=8)
        self.roll_var = tk.StringVar(value="—")
        self.pitch_var = tk.StringVar(value="—")
        self.yaw_var = tk.StringVar(value="—")
        self._metric(row, "ROLL", self.roll_var, "#ff6b6b").pack(side="left", expand=True, fill="x")
        self._metric(row, "PITCH", self.pitch_var, "#4ecdc4").pack(side="left", expand=True, fill="x")
        self._metric(row, "YAW", self.yaw_var, "#ffe66d").pack(side="left", expand=True, fill="x")

        self.canvas = tk.Canvas(self.root, bg="#0b0d12", highlightthickness=0, height=320)
        self.canvas.pack(fill="both", expand=True, padx=24, pady=(4, 8))

        self.detail = tk.StringVar(value="等待 IMU 四元数")
        tk.Label(self.root, textvariable=self.detail, fg="#6f7a88", bg="#11141a", font=("Sans", 11)).pack(
            anchor="w", padx=28, pady=(0, 16)
        )
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
            self.status.set("已 SSH，正在复位并启用 Game Rotation Vector …")
        else:
            self.status.set(
                f"IMU  bus {data.get('bus')}  {data.get('addr')}    错误 {data.get('errors', 0)}"
            )
            self.roll_var.set(f"{data.get('roll', 0):+.1f}°")
            self.pitch_var.set(f"{data.get('pitch', 0):+.1f}°")
            self.yaw_var.set(f"{data.get('yaw', 0):+.1f}°")
            q = data.get("quat") or [1, 0, 0, 0]
            self.detail.set(f"quat wxyz = ({q[0]:+.3f}, {q[1]:+.3f}, {q[2]:+.3f}, {q[3]:+.3f})")
            self._draw(
                float(data.get("roll", 0.0)),
                float(data.get("pitch", 0.0)),
                float(data.get("yaw", 0.0)),
                q,
            )
        self.root.after(40, self._tick)

    def _draw(self, roll_deg: float, pitch_deg: float, yaw_deg: float, quat: list[float]) -> None:
        c = self.canvas
        c.delete("all")
        w = max(c.winfo_width(), 1)
        h = max(c.winfo_height(), 1)
        c.create_rectangle(0, 0, w, h, fill="#0b0d12", outline="")
        self._draw_horizon(c, w * 0.42, h, roll_deg, pitch_deg, yaw_deg)
        self._draw_cube(c, w * 0.72, h * 0.48, min(w, h) * 0.22, quat)

    def _draw_horizon(self, c: tk.Canvas, cx: float, h: float, roll_deg: float, pitch_deg: float, yaw_deg: float) -> None:
        cy = h / 2.0
        radius = min(cx, h) * 0.38
        roll = math.radians(roll_deg)
        pitch_shift = max(-radius, min(radius, (pitch_deg / 45.0) * radius * 0.65))

        def rot(px: float, py: float) -> tuple[float, float]:
            x, y = px - cx, py - cy
            return (
                cx + x * math.cos(roll) - y * math.sin(roll),
                cy + x * math.sin(roll) + y * math.cos(roll),
            )

        horizon = cy + pitch_shift
        sky = [rot(-800, -800), rot(800, -800), rot(800, horizon), rot(-800, horizon)]
        ground = [rot(-800, horizon), rot(800, horizon), rot(800, 800), rot(-800, 800)]
        c.create_polygon(*[v for p in sky for v in p], fill="#2b6cb0", outline="")
        c.create_polygon(*[v for p in ground for v in p], fill="#8d5a2b", outline="")
        c.create_oval(cx - radius, cy - radius, cx + radius, cy + radius, outline="#e8eef7", width=3)
        c.create_line(cx - 40, cy, cx + 40, cy, fill="#f4f7fb", width=2)
        heading = ((yaw_deg + 180.0) % 360.0) - 180.0
        c.create_text(cx, h - 28, text=f"YAW {heading:+.1f}°", fill="#ffe66d", font=("Sans", 12, "bold"))

    def _draw_cube(self, c: tk.Canvas, cx: float, cy: float, scale: float, quat: list[float]) -> None:
        wq, xq, yq, zq = (quat + [0, 0, 0, 0])[:4]
        n = math.sqrt(wq * wq + xq * xq + yq * yq + zq * zq) or 1.0
        wq, xq, yq, zq = wq / n, xq / n, yq / n, zq / n

        def rotate(v: tuple[float, float, float]) -> tuple[float, float, float]:
            vx, vy, vz = v
            tx = 2 * (yq * vz - zq * vy)
            ty = 2 * (zq * vx - xq * vz)
            tz = 2 * (xq * vy - yq * vx)
            return (
                vx + wq * tx + (yq * tz - zq * ty),
                vy + wq * ty + (zq * tx - xq * tz),
                vz + wq * tz + (xq * ty - yq * tx),
            )

        corners = [
            (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
            (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1),
        ]
        pts = []
        for corner in corners:
            x, y, z = rotate(corner)
            pts.append((cx + x * scale, cy - y * scale, z))
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        for a, b in edges:
            c.create_line(pts[a][0], pts[a][1], pts[b][0], pts[b][1], fill="#7ee0c8", width=2)
        c.create_text(cx, cy + scale + 22, text="转动模块，立方体应跟着转", fill="#9aa4b2", font=("Sans", 11))

    def run(self) -> None:
        self.root.mainloop()

    def _on_close(self) -> None:
        self.stream.stop()
        self.root.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Temporary IMU test UI (does not touch robot code)")
    parser.add_argument("--host", default="xiaoxiaobai")
    parser.add_argument("--user", default=None)
    parser.add_argument("--password", default=None)
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
