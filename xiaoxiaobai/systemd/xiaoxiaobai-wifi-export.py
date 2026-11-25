#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe
"""Export the Pi's saved Wi-Fi networks for the ESP32 face board.

Only root can read NetworkManager secrets, so this runs as root (installed by
systemd/install-wifi-export.sh) and writes /run/xiaoxiaobai/wifi.json readable by the robot
user's group. The display loop forwards the list to the ESP32 over the UART, so the board
always joins the same networks as the Pi.
"""

from __future__ import annotations

import argparse
import grp
import json
import os
import subprocess
import tempfile

OUT_DIR = "/run/xiaoxiaobai"
OUT_PATH = os.path.join(OUT_DIR, "wifi.json")
# WPA/WPA2/WPA3 personal and open networks; enterprise (802.1X) and WEP are skipped.
SUPPORTED_KEY_MGMT = {"", "wpa-psk", "sae"}


def nmcli(*args: str) -> str:
    return subprocess.run(
        ["nmcli", "--escape", "no", *args], check=True, capture_output=True, text=True
    ).stdout.rstrip("\n")


def field(uuid: str, name: str) -> str:
    return nmcli("-s", "-g", name, "connection", "show", "uuid", uuid)


def saved_networks() -> list[dict[str, str]]:
    rows = []
    for line in nmcli("-g", "UUID,TYPE,ACTIVE,AUTOCONNECT-PRIORITY", "connection", "show").splitlines():
        uuid, kind, active, priority = line.split(":")
        if kind != "802-11-wireless" or field(uuid, "802-11-wireless.mode") not in ("", "infrastructure"):
            continue
        key_mgmt = field(uuid, "802-11-wireless-security.key-mgmt")
        psk = field(uuid, "802-11-wireless-security.psk") if key_mgmt else ""
        if key_mgmt not in SUPPORTED_KEY_MGMT or (key_mgmt and not psk):
            continue
        rows.append((active != "yes", -int(priority or 0), field(uuid, "802-11-wireless.ssid"), psk))
    rows.sort()
    networks: list[dict[str, str]] = []
    for _, _, ssid, psk in rows:
        if ssid and all(n["ssid"] != ssid for n in networks):
            networks.append({"ssid": ssid, "psk": psk})
    return networks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", required=True, help="group allowed to read the export")
    args = parser.parse_args()
    gid = grp.getgrnam(args.group).gr_gid

    os.makedirs(OUT_DIR, exist_ok=True)
    os.chown(OUT_DIR, 0, gid)
    os.chmod(OUT_DIR, 0o750)
    networks = saved_networks()
    fd, tmp = tempfile.mkstemp(dir=OUT_DIR, prefix=".wifi-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"networks": networks}, handle, ensure_ascii=False)
            handle.write("\n")
        os.chown(tmp, 0, gid)
        os.chmod(tmp, 0o640)
        os.replace(tmp, OUT_PATH)
    except BaseException:
        os.unlink(tmp)
        raise
    print(f"Exported {len(networks)} Wi-Fi network(s) to {OUT_PATH}: {', '.join(n['ssid'] for n in networks)}")


if __name__ == "__main__":
    main()
