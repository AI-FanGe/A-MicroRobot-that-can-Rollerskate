"""`make run`: start the robot API + web UI on the robot over SSH and open it in this computer's browser.

The browser goes through an SSH tunnel (http://localhost:<port>/), so it works whatever the robot's
address is. Ctrl-C stops the robot (torque off) and the service (scripts/robot_run.sh).
"""

from __future__ import annotations

import argparse
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser

READY_TIMEOUT_S = 120.0


def free_local_port(preferred: int) -> int:
    for port in (preferred, 0):
        with socket.socket() as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return sock.getsockname()[1]
    raise RuntimeError("no free local port")


def open_browser(url: str) -> None:
    if sys.platform.startswith("linux") and shutil.which("xdg-open"):
        subprocess.Popen(["xdg-open", url], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)
    else:
        webbrowser.open(url)


def open_when_ready(url: str, ssh: subprocess.Popen) -> None:
    deadline = time.time() + READY_TIMEOUT_S
    while time.time() < deadline and ssh.poll() is None:
        try:
            with urllib.request.urlopen(url + "api/v1/mc/action", timeout=2) as reply:
                if reply.status == 200:
                    break
        except OSError:
            pass
        time.sleep(0.5)
    else:
        return
    # ssh -tt puts this terminal in raw mode: lines need their carriage return.
    print(f"\r\n网页已打开：{url}（手机等设备用上面日志里的机器人地址）\r\n", end="", flush=True)
    open_browser(url)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="xiaoxiaobai")
    parser.add_argument("--port", type=int, default=8000, help="API port on the robot")
    parser.add_argument("--env", default="", help='environment for the robot side, e.g. "XIAOXIAOBAI_DISPLAY_BACKEND=usb"')
    args = parser.parse_args()

    local = free_local_port(args.port)
    url = f"http://localhost:{local}/"
    remote = f"bash -l -c 'cd xiaoxiaobai && {args.env} bash scripts/robot_run.sh {args.port}'"
    ssh = subprocess.Popen(["ssh", "-tt", "-o", "ExitOnForwardFailure=yes",
                            "-L", f"{local}:localhost:{args.port}", args.host, remote])
    threading.Thread(target=open_when_ready, args=(url, ssh), daemon=True).start()
    while True:
        try:
            return ssh.wait()
        except KeyboardInterrupt:
            continue  # Ctrl-C belongs to the robot side, which stops the robot first


if __name__ == "__main__":
    sys.exit(main())
