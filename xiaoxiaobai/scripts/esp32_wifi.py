#!/usr/bin/env python3
"""Reach the head board directly over Wi-Fi (no controller in the data path).

    ip      print the board's address
    camera  open its camera page in the browser
    ota     upload a firmware .bin with espota (ArduinoOTA)

The board is found as xiaoxiaobai-face.local (mDNS); when that does not resolve, the address
the board reported to the Pi over the UART is read from /tmp/xiaoxiaobai_esp32.json via SSH.
"""

from __future__ import annotations

import argparse
import json
import shutil
import socket
import subprocess
import sys
import urllib.request
import webbrowser


def _answers(address: str) -> bool:
    try:
        with urllib.request.urlopen(f"http://{address}/status", timeout=3) as response:
            return response.status == 200
    except OSError:
        return False


def _from_pi(host: str) -> str | None:
    try:
        out = subprocess.run(
            ["ssh", host, "cat /tmp/xiaoxiaobai_esp32.json"], check=True, capture_output=True, text=True, timeout=30
        ).stdout
        return json.loads(out).get("ip")
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError):
        return None


def find_esp32(name: str, pi_host: str) -> str:
    try:
        address = socket.gethostbyname(name)
        if _answers(address):
            return address
    except OSError:
        pass
    print(f"{name} not reachable, asking {pi_host} for the board's address...", file=sys.stderr)
    address = _from_pi(pi_host)
    if address and _answers(address):
        return address
    raise SystemExit(
        "ESP32 not found on Wi-Fi. Check `make run`/display-service output for 'ESP32 on Wi-Fi at ...', "
        "and that `make wifi-sync-enable` was run once on the Pi."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--esp32", default="xiaoxiaobai-face.local", help="board hostname or IP")
    parser.add_argument("--host", default="xiaoxiaobai", help="Pi SSH host, for the address fallback")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ip")
    camera = sub.add_parser("camera")
    camera.add_argument("--size", choices=("qqvga", "qvga", "hvga", "vga"), default="vga")
    camera.add_argument("--quality", type=int, default=12, help="JPEG quality 6-63, lower is sharper")
    ota = sub.add_parser("ota")
    ota.add_argument("binary")
    ota.add_argument("--espota", required=True, help="path to espota.py from the ESP32 Arduino core")
    ota.add_argument("--password", default="xiaoxiaobai")
    args = parser.parse_args()

    address = find_esp32(args.esp32, args.host)
    if args.command == "ip":
        print(address)
    elif args.command == "camera":
        url = f"http://{address}/?size={args.size}&quality={args.quality}"
        print(f"Camera (Wi-Fi direct): {url}")
        opener = shutil.which("xdg-open") or shutil.which("open")
        if opener:
            subprocess.Popen([opener, url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            webbrowser.open(url)
    else:
        print(f"Uploading {args.binary} to {address} ...")
        cmd = [sys.executable, args.espota, "-i", address, "-p", "3232", f"--auth={args.password}", "-f", args.binary, "-r"]
        raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
