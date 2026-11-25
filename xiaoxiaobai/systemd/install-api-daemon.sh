#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe
#
# Install (or remove, with --uninstall) the boot-time robot API + web UI service.
# Run on the Pi as root from the repo root, e.g.:
#   sudo bash systemd/install-api-daemon.sh [--port 8000]

set -euo pipefail

UNIT=/etc/systemd/system/xiaoxiaobai-api.service

if [[ "${1:-}" == "--uninstall" ]]; then
    systemctl disable --now xiaoxiaobai-api.service 2>/dev/null || true
    rm -f "$UNIT"
    systemctl daemon-reload
    echo "Robot API service removed."
    exit 0
fi

PORT=8000
if [[ "${1:-}" == "--port" ]]; then
    PORT=${2:?--port needs a value}
fi

USER_NAME=${SUDO_USER:-$(whoami)}
REPO=$(cd "$(dirname "$0")/.." && pwd)

# KillMode=process: restarting the API (e.g. after `make sync`) must not kill the control
# loop or display it spawned; the new API instance adopts them through their pid files.
# No network-online.target: on a first boot without Wi-Fi the phone app needs the
# Bluetooth link right away, not after NetworkManager gives up waiting.
# CAP_NET_ADMIN: the phone app's BLE advertisement goes straight to the kernel when
# bluetoothd's own advertising API is broken (src/api/ble_mgmt.py).
cat > "$UNIT" <<EOF
[Unit]
Description=Xiaoxiaobai robot API and web UI
After=network.target bluetooth.target
Wants=bluetooth.target

[Service]
Type=simple
User=$USER_NAME
WorkingDirectory=$REPO
Environment=PYTHONPATH=$REPO/src
Environment=XIAOXIAOBAI_API_PORT=$PORT
ExecStart=$REPO/.venv/bin/python -u -m api
AmbientCapabilities=CAP_NET_ADMIN
KillMode=process
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable xiaoxiaobai-api.service
systemctl restart xiaoxiaobai-api.service
echo "Robot API service installed and started (user=$USER_NAME, repo=$REPO)."
for ip in $(hostname -I); do
    [[ "$ip" == *:* ]] || echo "  web UI: http://$ip:$PORT/"
done
echo "It stays enabled across reboots until 'make api-disable'."
