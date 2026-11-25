#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe
#
# Install (or remove, with --uninstall) the Wi-Fi export that lets the ESP32 face board
# follow the Pi's Wi-Fi: a root oneshot rewrites /run/xiaoxiaobai/wifi.json at boot and
# whenever NetworkManager's saved connections change. Run on the Pi as root from the
# repo root, e.g.:
#   sudo bash systemd/install-wifi-export.sh

set -euo pipefail

BIN=/usr/local/sbin/xiaoxiaobai-wifi-export
SERVICE=/etc/systemd/system/xiaoxiaobai-wifi-export.service
WATCH=/etc/systemd/system/xiaoxiaobai-wifi-export.path

if [[ "${1:-}" == "--uninstall" ]]; then
    systemctl disable --now xiaoxiaobai-wifi-export.path xiaoxiaobai-wifi-export.service 2>/dev/null || true
    rm -f "$BIN" "$SERVICE" "$WATCH"
    rm -rf /run/xiaoxiaobai
    systemctl daemon-reload
    echo "Wi-Fi export removed; the ESP32 keeps the networks it already has."
    exit 0
fi

USER_NAME=${SUDO_USER:-$(whoami)}
GROUP_NAME=$(id -gn "$USER_NAME")
REPO=$(cd "$(dirname "$0")/.." && pwd)

# Root runs a root-owned copy, never the user-writable one in the repo.
install -m 0755 -o root -g root "$REPO/systemd/xiaoxiaobai-wifi-export.py" "$BIN"

cat > "$SERVICE" <<EOF
[Unit]
Description=Export the Pi's Wi-Fi networks for the Xiaoxiaobai ESP32 face board
After=NetworkManager.service
Wants=NetworkManager.service

[Service]
Type=oneshot
ExecStart=$BIN --group $GROUP_NAME
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

cat > "$WATCH" <<EOF
[Unit]
Description=Re-export the Xiaoxiaobai Wi-Fi list when the Pi's connections change

[Path]
PathChanged=/etc/NetworkManager/system-connections
PathChanged=/run/NetworkManager/system-connections
PathChanged=/etc/netplan
Unit=xiaoxiaobai-wifi-export.service

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable xiaoxiaobai-wifi-export.service
systemctl enable --now xiaoxiaobai-wifi-export.path
systemctl start xiaoxiaobai-wifi-export.service
journalctl -u xiaoxiaobai-wifi-export.service -n 1 --no-pager -o cat
# A new robot has no display service yet; try-restart then exits 5 and would abort app-enable.
systemctl try-restart xiaoxiaobai-display.service 2>/dev/null || true
echo "Wi-Fi export installed (group=$GROUP_NAME): the ESP32 now follows the Pi's saved Wi-Fi networks."
