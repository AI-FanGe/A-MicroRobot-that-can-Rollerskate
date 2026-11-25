#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe
#
# Prepare the Pi for the 小小白 phone app (or undo it, with --uninstall): Bluetooth on at
# every boot, the API user may advertise through BlueZ and change Wi-Fi through
# NetworkManager, the ESP32 follows the Pi's Wi-Fi, and the API starts at boot.
# Run on the Pi as root from the repo root, e.g.:
#   sudo bash systemd/install-app.sh [--port 8000]

set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
USER_NAME=${SUDO_USER:-$(whoami)}
RULES=/etc/polkit-1/rules.d/50-xiaoxiaobai-network.rules
PKLA=/etc/polkit-1/localauthority/50-local.d/50-xiaoxiaobai-network.pkla

if [[ "${1:-}" == "--uninstall" ]]; then
    rm -f "$RULES" "$PKLA"
    echo "Phone app permissions removed (the API service and Bluetooth stay as they are)."
    echo "Run 'make api-disable' to stop the API too."
    exit 0
fi

PORT=8000
if [[ "${1:-}" == "--port" ]]; then
    PORT=${2:?--port needs a value}
fi

if ! command -v bluetoothctl >/dev/null; then
    apt-get install -y bluez
fi
for config in /boot/firmware/config.txt /boot/config.txt; do
    if [[ -f "$config" ]] && grep -qE '^\s*dtoverlay=(pi3-)?disable-bt' "$config"; then
        echo "WARNING: $config disables the on-board Bluetooth (dtoverlay=disable-bt); remove that line and reboot."
    fi
done

rfkill unblock bluetooth 2>/dev/null || true
systemctl enable --now bluetooth.service
systemctl enable hciuart.service 2>/dev/null || true

# BlueZ's D-Bus policy lets the bluetooth group register GATT services and advertisements.
for group in bluetooth netdev; do
    if getent group "$group" >/dev/null; then
        usermod -aG "$group" "$USER_NAME"
    fi
done

# The API runs without a login session, so NetworkManager asks polkit; allow this user.
if [[ -d /etc/polkit-1/rules.d ]] && pkaction --version 2>/dev/null | grep -qvE ' 0\.10[0-5]'; then
    cat > "$RULES" <<EOF
// Installed by xiaoxiaobai systemd/install-app.sh: the robot API sets up Wi-Fi for the phone app.
polkit.addRule(function (action, subject) {
    if (action.id.indexOf("org.freedesktop.NetworkManager.") === 0 && subject.user === "$USER_NAME") {
        return polkit.Result.YES;
    }
});
EOF
else
    mkdir -p "$(dirname "$PKLA")"
    cat > "$PKLA" <<EOF
[Xiaoxiaobai robot API: Wi-Fi setup for the phone app]
Identity=unix-user:$USER_NAME
Action=org.freedesktop.NetworkManager.*
ResultAny=yes
ResultInactive=yes
ResultActive=yes
EOF
fi

# The ESP32 face board joins whatever network the app gives the Pi.
bash "$REPO/systemd/install-wifi-export.sh"
bash "$REPO/systemd/install-api-daemon.sh" --port "$PORT"

echo
echo "Ready for the 小小白 app: Bluetooth comes on at boot; open the app near the robot and add it."
