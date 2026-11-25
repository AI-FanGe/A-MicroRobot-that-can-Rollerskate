#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Build the ready-to-flash 小小白 SD-card image from the official Raspberry Pi OS Lite (64-bit):
# this repo with its .venv, the boot services of `make app-enable`, ffmpeg, I2C / SPI / UART
# set up, user "user" / password "password", hostname xiaoxiaobai. Run as root on a Linux
# computer, e.g. from the repo root:
#   sudo bash image/build.sh        (or `make image`)
# An x86 computer runs the image's arm64 programs through qemu: sudo apt install qemu-user-static
# Output: build/image/xiaoxiaobai-<date>.img.xz and its .sha256.

set -euo pipefail

BASE_URL=https://downloads.raspberrypi.com/raspios_lite_arm64/images/raspios_lite_arm64-2026-10-06/2026-10-06-raspios-trixie-arm64-lite.img.xz
BASE_SHA256=483db18a48da399b5b7022ffae9a07bc6d99daab30e593aedac09b69ff422843
UV_VERSION=0.12.5
UV_URL=https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-aarch64-unknown-linux-gnu.tar.gz
UV_SHA256=9bf43b4d1a07665bf64d4c4e710930b382321a785e0eb10aac07f46471f86a31
# Room for apt packages and the .venv while building. The image is shrunk afterwards to its
# content plus SLACK_MB, and grows to the whole card on first boot.
GROW_MB=4096
SLACK_MB=256

REPO=$(cd "$(dirname "$0")/.." && pwd)
HERE=$REPO/image
OUT=${OUT:-$REPO/build/image}
NAME=${NAME:-xiaoxiaobai-$(date +%Y%m%d)}
IMG=$OUT/$NAME.img
MNT=$OUT/mnt
LOOP=
QEMU_COPY=

die() {
    echo "error: $*" >&2
    exit 1
}

step() {
    echo
    echo "==> $*"
}

[[ $EUID -eq 0 ]] || die "run as root: sudo bash image/build.sh"
for cmd in curl xz sha256sum sfdisk partx losetup e2fsck resize2fs dumpe2fs rsync chroot fstrim tar mountpoint; do
    command -v "$cmd" >/dev/null || die "missing command: $cmd"
done

if [[ $(uname -m) != aarch64 ]]; then
    BINFMT=/proc/sys/fs/binfmt_misc/qemu-aarch64
    [[ -f $BINFMT ]] || die "this computer cannot run arm64 programs; install qemu first: sudo apt install qemu-user-static"
    # Without the F (fix-binary) flag the kernel looks for the interpreter inside the chroot.
    if ! grep -q '^flags:.*F' "$BINFMT"; then
        QEMU_COPY=$(awk '/^interpreter/ {print $2}' "$BINFMT")
    fi
fi

cleanup() {
    set +e
    if mountpoint -q "$MNT"; then
        umount -R "$MNT"
    fi
    if [[ -n $LOOP ]]; then
        losetup -d "$LOOP"
    fi
}
trap cleanup EXIT

# Download once into OUT; a wrong checksum (truncated download, changed upstream file) stops here.
fetch() {
    local url=$1 sha=$2 dest
    dest=$OUT/$(basename "$url")
    if [[ ! -f $dest ]]; then
        curl -fL --retry 3 -o "$dest.part" "$url"
        mv "$dest.part" "$dest"
    fi
    echo "$sha  $dest" | sha256sum -c --quiet - || die "checksum mismatch: $dest (delete it to download again)"
    echo "$dest"
}

mkdir -p "$OUT" "$MNT"

step "Downloading Raspberry Pi OS Lite and uv $UV_VERSION"
BASE=$(fetch "$BASE_URL" "$BASE_SHA256")
UV_TGZ=$(fetch "$UV_URL" "$UV_SHA256")

step "Unpacking into $IMG (+$GROW_MB MB)"
xz -dcT0 "$BASE" > "$IMG"
truncate -s "+${GROW_MB}M" "$IMG"
echo ", +" | sfdisk -q --no-reread --no-tell-kernel -N 2 "$IMG"

LOOP=$(losetup -Pf --show "$IMG")
BOOT_DEV=${LOOP}p1
ROOT_DEV=${LOOP}p2
for _ in $(seq 50); do
    [[ -b $ROOT_DEV ]] && break
    sleep 0.1
done
[[ -b $ROOT_DEV ]] || die "no partition device for $LOOP"
e2fsck -fy "$ROOT_DEV" >/dev/null || [[ $? -le 1 ]]
resize2fs "$ROOT_DEV" >/dev/null

mount "$ROOT_DEV" "$MNT"
mount "$BOOT_DEV" "$MNT/boot/firmware"

step "Boot partition: interfaces, cloud-init user-data and network-config"
cat >> "$MNT/boot/firmware/config.txt" <<'EOF'
# 小小白: IMU on I2C-1 (400 kHz), face LCD on SPI0, ESP32 head board on the PL011 UART
# (/dev/serial0, 3 Mbps), which moves Bluetooth to the mini UART.
dtparam=i2c_arm=on
dtparam=i2c_arm_baudrate=400000
dtparam=spi=on
enable_uart=1
dtoverlay=miniuart-bt
EOF
# No serial login console: it would write onto the ESP32 link. Wi-Fi country until
# network-config (or the phone app) sets another one.
sed -i -e 's/console=serial0,[0-9]* //' -e 's/$/ cfg80211.ieee80211_regdom=CN/' "$MNT/boot/firmware/cmdline.txt"
install -m 0644 "$HERE/user-data" "$HERE/network-config" "$MNT/boot/firmware/"

step "Staging the code and uv"
rsync -a --no-owner --no-group \
    --exclude='.git' \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='cad' \
    --exclude='docs' \
    --exclude='firmware' \
    --exclude='build' \
    --exclude='logs' \
    --exclude='src/debug' \
    --exclude='src/sim' \
    --exclude='src/model/mjcf' \
    --exclude='android' \
    --exclude='image' \
    --exclude='.console-data' \
    "$REPO/" "$MNT/tmp/xiaoxiaobai/"
tar -xzf "$UV_TGZ" -C "$MNT/usr/local/bin" --strip-components=1 --no-same-owner
install -m 0755 "$HERE/provision.sh" "$MNT/tmp/provision.sh"

step "Provisioning inside the image (arm64)"
MACHINE_ID=$(cat "$MNT/etc/machine-id")
RESOLV_BACKUP=$MNT/etc/resolv.conf.xiaoxiaobai
if [[ -e $MNT/etc/resolv.conf || -L $MNT/etc/resolv.conf ]]; then
    mv "$MNT/etc/resolv.conf" "$RESOLV_BACKUP"
fi
if [[ -s /run/systemd/resolve/resolv.conf ]]; then
    cp /run/systemd/resolve/resolv.conf "$MNT/etc/resolv.conf"
else
    cp -L /etc/resolv.conf "$MNT/etc/resolv.conf"
fi
printf '#!/bin/sh\nexit 101\n' > "$MNT/usr/sbin/policy-rc.d"
chmod 755 "$MNT/usr/sbin/policy-rc.d"
if [[ -n $QEMU_COPY ]]; then
    install -D -m 0755 "$QEMU_COPY" "$MNT$QEMU_COPY"
fi
mount -t proc proc "$MNT/proc"
mount -t sysfs sys "$MNT/sys"
mount --bind /dev "$MNT/dev"
mount --bind /dev/pts "$MNT/dev/pts"

chroot "$MNT" /usr/bin/env -i \
    HOME=/root LANG=C.UTF-8 TERM="${TERM:-dumb}" \
    PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    bash /tmp/provision.sh

umount "$MNT/dev/pts" "$MNT/dev" "$MNT/sys" "$MNT/proc"

step "Depersonalizing and cleaning up"
rm -f "$MNT/usr/sbin/policy-rc.d" "$MNT/etc/resolv.conf"
if [[ -e $RESOLV_BACKUP || -L $RESOLV_BACKUP ]]; then
    mv "$RESOLV_BACKUP" "$MNT/etc/resolv.conf"
fi
if [[ -n $QEMU_COPY ]]; then
    rm -f "$MNT$QEMU_COPY"
fi
# Every robot gets its own machine-id and SSH host keys on first boot (ConditionFirstBoot).
printf '%s\n' "$MACHINE_ID" > "$MNT/etc/machine-id"
rm -f "$MNT"/etc/ssh/ssh_host_*
rm -rf "$MNT"/tmp/* "$MNT"/var/tmp/* "$MNT"/var/cache/apt/*.bin "$MNT"/var/cache/apt/archives/*.deb
rm -rf "$MNT"/root/.cache "$MNT"/home/user/.cache "$MNT"/root/.bash_history "$MNT"/home/user/.bash_history
find "$MNT/var/log" -type f -delete
fstrim "$MNT" >/dev/null || true
umount "$MNT/boot/firmware"
umount "$MNT"

step "Shrinking the root partition"
e2fsck -fy "$ROOT_DEV" >/dev/null || [[ $? -le 1 ]]
resize2fs -M "$ROOT_DEV" >/dev/null
BLOCKS=$(dumpe2fs -h "$ROOT_DEV" 2>/dev/null | awk -F': *' '/^Block count/ {print $2}')
BLOCK_SIZE=$(dumpe2fs -h "$ROOT_DEV" 2>/dev/null | awk -F': *' '/^Block size/ {print $2}')
FS_BYTES=$((BLOCKS * BLOCK_SIZE + SLACK_MB * 1024 * 1024))
resize2fs "$ROOT_DEV" "$((FS_BYTES / 1024))K" >/dev/null
losetup -d "$LOOP"
LOOP=
START=$(partx -g -o START -n 2 "$IMG" | tr -d ' ')
SECTORS=$((FS_BYTES / 512))
echo "$START,$SECTORS" | sfdisk -q --no-reread --no-tell-kernel -N 2 "$IMG"
truncate -s $(((START + SECTORS) * 512)) "$IMG"

step "Compressing"
xz -fT0 "$IMG"
(cd "$OUT" && sha256sum "$NAME.img.xz" > "$NAME.img.xz.sha256")
if [[ -n ${SUDO_UID:-} ]]; then
    chown "$SUDO_UID:$SUDO_GID" "$OUT" "$OUT"/*
fi
rmdir "$MNT"

echo
echo "Done: $IMG.xz ($(du -h "$IMG.xz" | cut -f1))"
