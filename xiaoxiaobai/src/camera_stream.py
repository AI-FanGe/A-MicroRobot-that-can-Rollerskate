# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

"""Live view of the head-board camera, served as MJPEG on 127.0.0.1.

The process that owns the ESP32 link (the display loop) publishes JPEG frames into a
CameraHub. The camera runs only while a viewer holds the stream, or for the single frame
a one-shot capture asks for; then it is switched off. `make camera` tunnels the port over
SSH and opens the page.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from esp32_link import CAMERA_SIZES

CAMERA_PORT_ENV = "XIAOXIAOBAI_CAMERA_PORT"
CAMERA_PORT = 8090
DEFAULT_SIZE = "qvga"
DEFAULT_QUALITY = 12
KEEPALIVE_S = 1.0
STREAM_WAIT_S = 2.0
# The firmware takes the frame interval in 10 ms units in one byte.
MAX_INTERVAL_MS = 2550


def camera_port() -> int:
    return int(os.environ.get(CAMERA_PORT_ENV) or CAMERA_PORT)


class CameraHub:
    """Latest JPEG plus the camera settings the current viewers ask for."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._jpeg: bytes | None = None
        self._seq = 0
        self._frame_times: deque[float] = deque(maxlen=30)
        self._viewers = 0
        # Minimum time between frames each viewer asked for (0 = as fast as possible).
        self._intervals: list[int] = []
        self._size = DEFAULT_SIZE
        self._quality = DEFAULT_QUALITY
        self._sent: tuple[bool, int, int, int] | None = None
        self._next_keepalive = 0.0
        self.closed = False
        self.state = "ESP32 not connected"
        self.link_stats: dict[str, int] = {}

    def publish(self, jpeg: bytes) -> None:
        with self._cond:
            self._jpeg = jpeg
            self._seq += 1
            self._frame_times.append(time.monotonic())
            self._cond.notify_all()

    def wait_frame(self, after_seq: int, timeout_s: float) -> tuple[int, bytes] | None:
        with self._cond:
            self._cond.wait_for(lambda: self.closed or (self._seq != after_seq and self._jpeg is not None), timeout_s)
            if self.closed or self._jpeg is None or self._seq == after_seq:
                return None
            return self._seq, self._jpeg

    @property
    def watching(self) -> bool:
        return self._viewers > 0

    def latest(self) -> bytes | None:
        with self._cond:
            return self._jpeg

    def capture(self, timeout_s: float, size: str = DEFAULT_SIZE, quality: int = DEFAULT_QUALITY) -> bytes | None:
        """Switch the camera on, wait for one JPEG taken after this call, then release it.

        A stream that is already open stays open: this only adds a viewer for the wait.
        """
        with self._cond:
            start_seq = self._seq
        with self.viewer(size, quality, 0):
            deadline = time.monotonic() + timeout_s
            while not self.closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                got = self.wait_frame(start_seq, min(0.5, remaining))
                if got is not None:
                    return got[1]
            return None

    @contextmanager
    def viewer(self, size: str, quality: int, interval_ms: int = 0) -> Iterator[None]:
        with self._cond:
            self._viewers += 1
            self._intervals.append(interval_ms)
            self._size, self._quality = size, quality
        try:
            yield
        finally:
            with self._cond:
                self._viewers -= 1
                self._intervals.remove(interval_ms)

    def command(self, now: float) -> tuple[bool, int, int, int] | None:
        """Camera request (enabled, size code, quality, interval ms) the link owner should send now, if any."""
        with self._cond:
            interval_ms = min(self._intervals, default=0)
            wanted = (self._viewers > 0, CAMERA_SIZES[self._size][0], self._quality, interval_ms)
        if wanted[0]:
            if wanted == self._sent and now < self._next_keepalive:
                return None
            self._next_keepalive = now + KEEPALIVE_S
        elif self._sent is None or not self._sent[0]:
            return None
        self._sent = wanted
        return wanted

    def link_changed(self) -> None:
        """Forget what the previous link was told, so the next command() resends it."""
        self._sent = None

    def close(self) -> None:
        with self._cond:
            self.closed = True
            self._cond.notify_all()

    def stats(self) -> dict[str, object]:
        with self._cond:
            times = list(self._frame_times)
            fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 and times[-1] > times[0] else 0.0
            if times and time.monotonic() - times[-1] > 2.0:
                fps = 0.0
            return {
                "state": self.state,
                "viewers": self._viewers,
                "size": self._size,
                "quality": self._quality,
                "fps": round(fps, 1),
                "frame_bytes": len(self._jpeg) if self._jpeg else 0,
                **self.link_stats,
            }


PAGE = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>小小白 摄像头</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  html, body { margin: 0; height: 100%; background: #111; color: #ddd; font: 14px sans-serif; }
  body { display: flex; flex-direction: column; }
  header { display: flex; gap: 8px; align-items: center; padding: 8px; flex-wrap: wrap; }
  button { background: #333; color: #eee; border: 1px solid #555; border-radius: 4px; padding: 4px 10px; cursor: pointer; }
  button.on { background: #2a6; border-color: #2a6; }
  #info { margin-left: auto; font-family: monospace; }
  main { flex: 1; display: flex; align-items: center; justify-content: center; overflow: hidden; }
  img { max-width: 100%; max-height: 100%; image-rendering: auto; }
</style></head><body>
<header>
  <span>分辨率</span><span id="sizes"></span>
  <button id="rotate">旋转 90°</button>
  <a id="snap" href="/snapshot.jpg" download="xiaoxiaobai.jpg"><button>截图</button></a>
  <span id="info">连接中…</span>
</header>
<main><img id="view" alt="camera"></main>
<script>
const params = new URLSearchParams(location.search);
let size = params.get("size") || "__SIZE__";
const quality = params.get("quality") || "__QUALITY__";
let rotation = Number(localStorage.getItem("xiaoxiaobaiCameraRotation") || 0);
const view = document.getElementById("view");
function start() {
  view.src = `/stream?size=${size}&quality=${quality}&t=${Date.now()}`;
  for (const b of document.querySelectorAll("#sizes button")) b.classList.toggle("on", b.dataset.size === size);
}
function rotate() {
  view.style.transform = `rotate(${rotation}deg)`;
  localStorage.setItem("xiaoxiaobaiCameraRotation", rotation);
}
for (const s of __SIZES__) {
  const b = document.createElement("button");
  b.textContent = s.toUpperCase(); b.dataset.size = s;
  b.onclick = () => { size = s; start(); };
  document.getElementById("sizes").appendChild(b);
}
document.getElementById("rotate").onclick = () => { rotation = (rotation + 90) % 360; rotate(); };
view.onerror = () => setTimeout(start, 1000);
async function poll() {
  try {
    const s = await (await fetch("/status")).json();
    let text = `${s.state} | ${s.fps} fps | ${(s.frame_bytes / 1024).toFixed(1)} KB/帧`;
    if (s.dropped_frames || s.bad_packets) text += ` | 丢帧 ${s.dropped_frames || 0} 坏包 ${s.bad_packets || 0}`;
    document.getElementById("info").textContent = text;
  } catch (e) {
    document.getElementById("info").textContent = "与机器人断开";
  }
}
rotate(); start(); poll(); setInterval(poll, 1000);
</script></body></html>
"""


class _Handler(BaseHTTPRequestHandler):
    hub: CameraHub

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        url = urlsplit(self.path)
        query = parse_qs(url.query)
        if url.path == "/":
            self._send(200, "text/html; charset=utf-8", self._page().encode())
        elif url.path == "/status":
            self._send(200, "application/json", json.dumps(self.hub.stats()).encode())
        elif url.path == "/snapshot.jpg":
            size, quality, _ = self._settings(query)
            # fresh=1 never reuses a frame taken before this request (a voice turn wants now).
            fresh = query.get("fresh", ["0"])[0] == "1"
            jpeg = None if fresh or not self.hub.watching else self.hub.latest()
            if jpeg is None:
                jpeg = self.hub.capture(4.0, size, quality)
            if jpeg is None:
                self._send(503, "text/plain; charset=utf-8", "no camera frame yet\n".encode())
            else:
                self._send(200, "image/jpeg", jpeg)
        elif url.path == "/stream":
            self._stream(*self._settings(query))
        else:
            self._send(404, "text/plain; charset=utf-8", b"not found\n")

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - http.server API
        pass

    def _page(self) -> str:
        return (
            PAGE.replace("__SIZES__", json.dumps(list(CAMERA_SIZES)))
            .replace("__SIZE__", DEFAULT_SIZE)
            .replace("__QUALITY__", str(DEFAULT_QUALITY))
        )

    @staticmethod
    def _settings(query: dict[str, list[str]]) -> tuple[str, int, int]:
        size = query.get("size", [DEFAULT_SIZE])[0].lower()
        if size not in CAMERA_SIZES:
            size = DEFAULT_SIZE
        try:
            quality = max(6, min(63, int(query.get("quality", [DEFAULT_QUALITY])[0])))
        except ValueError:
            quality = DEFAULT_QUALITY
        try:
            interval_ms = max(0, min(MAX_INTERVAL_MS, int(query.get("interval_ms", ["0"])[0])))
        except ValueError:
            interval_ms = 0
        return size, quality, interval_ms

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _stream(self, size: str, quality: int, interval_ms: int) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        seq = 0
        with self.hub.viewer(size, quality, interval_ms):
            try:
                while not self.hub.closed:
                    got = self.hub.wait_frame(seq, STREAM_WAIT_S)
                    if got is None:
                        continue
                    seq, jpeg = got
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpeg) + jpeg + b"\r\n"
                    )
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass


class CameraServer:
    def __init__(self, hub: CameraHub, port: int) -> None:
        handler = type("CameraHandler", (_Handler,), {"hub": hub})
        self._hub = hub
        self._httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
        self._httpd.daemon_threads = True
        self._thread = threading.Thread(target=self._httpd.serve_forever, name="xiaoxiaobai-camera-http", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._hub.close()
        self._httpd.shutdown()
        self._httpd.server_close()
