#!/usr/bin/env python3
"""`make camera`: tunnel the robot's camera stream over SSH and open it in the browser.

Runs src/camera.py on the robot through `ssh -L`, relays its output, and opens
http://127.0.0.1:<port>/ once it reports the stream is up. Ctrl+C closes the tunnel.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import webbrowser

SIZES = ("qqvga", "qvga", "hvga", "vga")
# Printed by src/camera.py.
READY_MARKER = b"Camera: stream ready"


def _say(message: str) -> None:
    # ssh -tt puts this terminal in raw mode, so newlines need an explicit carriage return.
    sys.stdout.write(message + "\r\n")
    sys.stdout.flush()


def _open_browser(url: str) -> None:
    opener = shutil.which("xdg-open") or shutil.which("open")
    if opener:
        subprocess.Popen([opener, url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        webbrowser.open(url)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="xiaoxiaobai")
    parser.add_argument("--port", type=int, default=8090, help="port on this computer and on the robot")
    parser.add_argument("--size", choices=SIZES, default="qvga", help="initial resolution (default %(default)s)")
    parser.add_argument("--quality", type=int, default=12, help="JPEG quality 6-63, lower is sharper (default %(default)s)")
    parser.add_argument("--no-browser", action="store_true", help="only open the tunnel")
    args = parser.parse_args()

    if shutil.which("ssh") is None:
        print("ssh not found", file=sys.stderr)
        return 1
    remote = f"bash -l -c 'cd xiaoxiaobai && PYTHONPATH=src .venv/bin/python src/camera.py --port {args.port}'"
    proc = subprocess.Popen(
        [
            "ssh",
            "-tt",
            "-o",
            "ExitOnForwardFailure=yes",
            "-L",
            f"{args.port}:127.0.0.1:{args.port}",
            args.host,
            remote,
        ],
        stdout=subprocess.PIPE,
    )
    url = f"http://127.0.0.1:{args.port}/?size={args.size}&quality={args.quality}"
    opened = False
    tail = b""
    try:
        while chunk := os.read(proc.stdout.fileno(), 4096):
            sys.stdout.buffer.write(chunk)
            sys.stdout.flush()
            tail = (tail + chunk)[-256:]
            if not opened and READY_MARKER in tail:
                opened = True
                _say(f"Camera: {url}")
                if not args.no_browser:
                    _open_browser(url)
        return proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        return proc.wait()


if __name__ == "__main__":
    sys.exit(main())
