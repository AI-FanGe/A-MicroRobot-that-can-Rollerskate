#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe
#
# Install (or remove, with --uninstall) the boot-time LCD expression service.
# Run on the Pi as root from the repo root, e.g.:
#   sudo bash systemd/install-display-daemon.sh

set -euo pipefail

UNIT=/etc/systemd/system/xiaoxiaobai-display.service

if [[ "${1:-}" == "--uninstall" ]]; then
    systemctl disable --now xiaoxiaobai-display.service 2>/dev/null || true
    rm -f "$UNIT"
    systemctl daemon-reload
    echo "LCD expression service removed."
    exit 0
fi

USER_NAME=${SUDO_USER:-$(whoami)}
REPO=$(cd "$(dirname "$0")/.." && pwd)

cat > "$UNIT" <<EOF
[Unit]
Description=Xiaoxiaobai LCD expression player
After=multi-user.target

[Service]
Type=simple
User=$USER_NAME
WorkingDirectory=$REPO
Environment=PYTHONPATH=$REPO/src
ExecStart=$REPO/.venv/bin/python -u src/display_smile.py
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now xiaoxiaobai-display.service
echo "LCD expression service installed and started (user=$USER_NAME, repo=$REPO)."
echo "It stays enabled across reboots until 'make display-disable'."
