# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

"""Keep the ESP32 camera stream up for `make camera`.

While `make run` (or the display service) owns the ESP32, its display loop already serves
the stream and this script only holds the SSH tunnel open. Otherwise it runs the display
loop itself (face + speaker + camera) until Ctrl+C, and takes over when the other owner exits.
"""

from __future__ import annotations

import argparse
import os
import socket
import threading
import time

from camera_stream import CAMERA_PORT_ENV, camera_port
from display_smile import DisplayAlreadyRunning, display_lock, display_worker
from esp32_link import CAMERA_BACKENDS, ESP32_BACKEND_ENV, esp32_backend

POLL_S = 1.0
# scripts/camera_view.py opens the browser when it sees this line.
READY_MARKER = "Camera: stream ready"
READY_TIMEOUT_S = 10.0


def _serving(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def _announce_when_serving(port: int) -> None:
    deadline = time.monotonic() + READY_TIMEOUT_S
    while time.monotonic() < deadline:
        if _serving(port):
            print(f"{READY_MARKER} on port {port}", flush=True)
            return
        time.sleep(0.2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=camera_port(), help="local HTTP port (default %(default)s)")
    args = parser.parse_args()
    backend = esp32_backend()
    if backend not in CAMERA_BACKENDS:
        raise SystemExit(f"The camera needs {ESP32_BACKEND_ENV}=uart (or usb), got {backend!r}.")
    os.environ[CAMERA_PORT_ENV] = str(args.port)

    announced = None
    try:
        while True:
            if _serving(args.port):
                if announced != "shared":
                    print("Camera: using the running display loop. Ctrl+C to close.", flush=True)
                    print(f"{READY_MARKER} on port {args.port}", flush=True)
                    announced = "shared"
                time.sleep(POLL_S)
                continue
            try:
                with display_lock():
                    pass
            except DisplayAlreadyRunning:
                if announced != "blocked":
                    print(
                        "Camera: another display loop holds the ESP32 but serves no camera: the "
                        "xiaoxiaobai-display service or a `make run` started before this update, or one "
                        f"started with {ESP32_BACKEND_ENV}=ble/spi. Restart it "
                        "(`sudo systemctl restart xiaoxiaobai-display`); waiting for it to exit.",
                        flush=True,
                    )
                    announced = "blocked"
                time.sleep(POLL_S)
                continue
            if announced != "own":
                print("Camera: no display loop running, starting one here. Ctrl+C to stop.", flush=True)
                threading.Thread(target=_announce_when_serving, args=(args.port,), daemon=True).start()
                announced = "own"
            try:
                display_worker(threading.Event())
            except DisplayAlreadyRunning:
                time.sleep(POLL_S)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
