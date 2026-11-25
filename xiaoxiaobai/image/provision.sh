#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Second half of image/build.sh: runs as root inside the arm64 image (chroot), with this repo
# staged at /tmp/xiaoxiaobai. Leaves the robot in the state of `make setup` + `make app-enable`
# + `make music-setup` on a freshly flashed Raspberry Pi OS Lite.

set -euo pipefail

USER_NAME=user
PASSWORD=password
HOST_NAME=xiaoxiaobai
REPO=/home/$USER_NAME/xiaoxiaobai
export DEBIAN_FRONTEND=noninteractive

# systemd is not running in the chroot: let the install scripts enable units, skip the rest.
SHIM=$(mktemp -d)
cat > "$SHIM/systemctl" <<'EOF'
#!/bin/bash
args=()
for a in "$@"; do
    [[ $a == --now ]] || args+=("$a")
done
for a in "${args[@]}"; do
    case $a in
        enable | disable | mask | unmask | is-enabled) exec /usr/bin/systemctl "${args[@]}" ;;
    esac
done
exit 0
EOF
printf '#!/bin/sh\nexit 0\n' > "$SHIM/journalctl"
chmod 755 "$SHIM"/*
export PATH=$SHIM:$PATH

echo "--- apt: python3-dev (spidev builds from source), i2c-tools, ffmpeg (music page)"
apt-get update
apt-get install -y --no-install-recommends python3-dev libc6-dev i2c-tools ffmpeg

echo "--- user $USER_NAME / $PASSWORD"
# The image ships a placeholder "pi" that userconfig.service renames interactively on first
# boot; rename it here instead and drop that dialog.
if id pi >/dev/null 2>&1; then
    usermod -l "$USER_NAME" pi
    groupmod -n "$USER_NAME" pi
    usermod -d "/home/$USER_NAME" "$USER_NAME"
    sed -i "s/^pi:/$USER_NAME:/" /etc/subuid /etc/subgid
fi
if [[ -d /home/pi ]]; then
    mv /home/pi "/home/$USER_NAME"
fi
if [[ ! -d /home/$USER_NAME ]]; then
    cp -rT /etc/skel "/home/$USER_NAME"
fi
chown -R "$USER_NAME:$USER_NAME" "/home/$USER_NAME"
usermod -s /bin/bash "$USER_NAME"
echo "$USER_NAME:$PASSWORD" | chpasswd
for group in adm dialout cdrom audio users sudo video plugdev input gpio spi i2c netdev bluetooth render; do
    if getent group "$group" >/dev/null; then
        usermod -aG "$group" "$USER_NAME"
    fi
done
# Same rule Raspberry Pi OS gives its first user; the API powers off / reboots with `sudo -n`.
echo "$USER_NAME ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/010_xiaoxiaobai-nopasswd
chmod 440 /etc/sudoers.d/010_xiaoxiaobai-nopasswd
visudo -cf /etc/sudoers.d/010_xiaoxiaobai-nopasswd
systemctl disable userconfig.service
systemctl enable getty@tty1.service
rm -rf /var/lib/userconf-pi /etc/ssh/sshd_config.d/rename_user.conf

echo "--- hostname $HOST_NAME, time zone, SSH"
echo "$HOST_NAME" > /etc/hostname
sed -i "s/^127\.0\.1\.1.*/127.0.1.1\t$HOST_NAME/" /etc/hosts
ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
echo Asia/Shanghai > /etc/timezone
systemctl enable ssh.service
# Background apt runs steal CPU from the 50 Hz control loop.
systemctl disable apt-daily.timer apt-daily-upgrade.timer man-db.timer

echo "--- I2C device node, Wi-Fi power save off"
echo i2c-dev > /etc/modules-load.d/xiaoxiaobai.conf
# Power save makes SSH and the web UI stall while the robot is idle.
cat > /etc/NetworkManager/conf.d/50-xiaoxiaobai-wifi-powersave.conf <<'EOF'
[connection]
wifi.powersave=2
EOF

echo "--- cloud-init: keep the hostname, re-apply network-config when it changes"
cat > /etc/cloud/cloud.cfg.d/90_xiaoxiaobai.cfg <<'EOF'
# The image sets the hostname itself; keep it (and any later rename) when cloud-init runs again.
preserve_hostname: true
EOF
# cloud-init reads /boot/firmware/network-config only on its first boot. When the file has
# changed since the last boot, reset cloud-init and reboot once so the new Wi-Fi is applied.
cat > /usr/local/sbin/xiaoxiaobai-netcfg-reapply <<'EOF'
#!/bin/sh
set -u
CFG=/boot/firmware/network-config
STAMP=/var/lib/xiaoxiaobai/netcfg.sha256
[ -f "$CFG" ] || exit 0
mkdir -p "$(dirname "$STAMP")"
new=$(sha256sum "$CFG" | cut -d' ' -f1)
old=$(cat "$STAMP" 2>/dev/null || true)
echo "$new" > "$STAMP"
if [ -n "$old" ] && [ "$new" != "$old" ]; then
    cloud-init clean --logs
    systemctl reboot
fi
EOF
chmod 755 /usr/local/sbin/xiaoxiaobai-netcfg-reapply
cat > /etc/systemd/system/xiaoxiaobai-netcfg-reapply.service <<'EOF'
[Unit]
Description=Re-apply /boot/firmware/network-config when it changes
After=cloud-final.service
ConditionPathExists=/boot/firmware/network-config

[Service]
Type=oneshot
ExecStart=/usr/local/sbin/xiaoxiaobai-netcfg-reapply

[Install]
WantedBy=multi-user.target
EOF
systemctl enable xiaoxiaobai-netcfg-reapply.service

echo "--- code in $REPO, uv sync"
mv /tmp/xiaoxiaobai "$REPO"
chown -R "$USER_NAME:$USER_NAME" "$REPO"
runuser -u "$USER_NAME" -- env HOME="/home/$USER_NAME" \
    UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never \
    bash -c "cd '$REPO' && uv sync --frozen --python /usr/bin/python3"
rm -rf /tmp/uv-cache

echo "--- boot services (make app-enable)"
(cd "$REPO" && SUDO_USER=$USER_NAME bash systemd/install-app.sh)

apt-get clean
rm -rf /var/lib/apt/lists/* "$SHIM"

echo "--- check"
for unit in ssh xiaoxiaobai-api bluetooth xiaoxiaobai-wifi-export.path xiaoxiaobai-netcfg-reapply; do
    printf '%-32s %s\n' "$unit" "$(/usr/bin/systemctl is-enabled "$unit")"
done
printf '%-32s %s\n' userconfig "$(/usr/bin/systemctl is-enabled userconfig || true)"
id "$USER_NAME"
runuser -u "$USER_NAME" -- "$REPO/.venv/bin/python" -c "import numpy, onnxruntime, placo, rustypot, spidev; print('venv ok')"
