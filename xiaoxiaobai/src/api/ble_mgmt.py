"""BLE advertising through the kernel's Bluetooth management socket, bypassing bluetoothd.

BlueZ 5.82 sends MGMT_OP_ADD_EXT_ADV_DATA 8 bytes too long, and kernels with the stable fix
"Bluetooth: MGMT: validate Add Extended Advertising Data length" reject it, so every
LEAdvertisingManager1.RegisterAdvertisement fails with "Invalid Parameters (0x0d)" (Debian
trixie on a Pi, mid-2026). The legacy MGMT_OP_ADD_ADVERTISING still works. bluetoothd keeps
serving the GATT application; only the advertisement is added here. Needs CAP_NET_ADMIN
(the xiaoxiaobai-api unit from `make app-enable` grants it).
"""

from __future__ import annotations

import ctypes
import os
import select
import socket
import struct
import time
import uuid

AF_BLUETOOTH = 31
BTPROTO_HCI = 1
HCI_DEV_NONE = 0xFFFF
HCI_CHANNEL_CONTROL = 3

OP_ADD_ADVERTISING = 0x003E
OP_REMOVE_ADVERTISING = 0x003F
EV_CMD_COMPLETE = 0x0001
EV_CMD_STATUS = 0x0002

FLAG_CONNECTABLE = 1 << 0
FLAG_DISCOV = 1 << 1

STATUS_PERMISSION_DENIED = 0x14
STATUS_TEXT = {0x01: "未知命令", 0x03: "失败", 0x0B: "忙", 0x0C: "不支持", 0x0D: "参数无效",
               0x0F: "被拒绝", 0x11: "适配器不存在", STATUS_PERMISSION_DENIED: "权限不够"}


class MgmtError(RuntimeError):
    def __init__(self, op: int, status: int) -> None:
        self.status = status
        if status == STATUS_PERMISSION_DENIED:
            text = "蓝牙广播需要 CAP_NET_ADMIN 权限：在电脑上运行一次 make app-enable"
        else:
            text = f"蓝牙管理命令 0x{op:04x} 失败：{STATUS_TEXT.get(status, f'0x{status:02x}')}"
        super().__init__(text)


class _SockaddrHci(ctypes.Structure):
    _fields_ = [("family", ctypes.c_ushort), ("dev", ctypes.c_ushort), ("channel", ctypes.c_ushort)]


def _open() -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.socket(AF_BLUETOOTH, socket.SOCK_RAW | socket.SOCK_CLOEXEC, BTPROTO_HCI)
    if fd < 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))
    addr = _SockaddrHci(AF_BLUETOOTH, HCI_DEV_NONE, HCI_CHANNEL_CONTROL)
    if libc.bind(fd, ctypes.byref(addr), ctypes.sizeof(addr)) < 0:
        err = ctypes.get_errno()
        os.close(fd)
        raise OSError(err, os.strerror(err))
    return fd


def _command(index: int, op: int, params: bytes, timeout: float = 3.0) -> bytes:
    fd = _open()
    try:
        os.write(fd, struct.pack("<HHH", op, index, len(params)) + params)
        deadline = time.monotonic() + timeout
        while (left := deadline - time.monotonic()) > 0:
            if not select.select([fd], [], [], left)[0]:
                break
            data = os.read(fd, 1024)
            if len(data) < 9:
                continue
            event, ev_index, _ = struct.unpack_from("<HHH", data)
            ev_op, status = struct.unpack_from("<HB", data, 6)
            if event in (EV_CMD_COMPLETE, EV_CMD_STATUS) and ev_index == index and ev_op == op:
                if status:
                    raise MgmtError(op, status)
                return data[9:]
        raise TimeoutError(f"蓝牙管理命令 0x{op:04x} 没有回应")
    finally:
        os.close(fd)


def _ad(kind: int, payload: bytes) -> bytes:
    return bytes([len(payload) + 1, kind]) + payload


def add(index: int, instance: int, service_uuid: str, name: str) -> None:
    """Connectable, discoverable advertisement: the service UUID, with the name in the scan response."""
    adv = _ad(0x07, uuid.UUID(service_uuid).bytes[::-1])
    scan = _ad(0x09, name.encode()[:29])
    params = struct.pack("<BIHHBB", instance, FLAG_CONNECTABLE | FLAG_DISCOV, 0, 0, len(adv), len(scan))
    _command(index, OP_ADD_ADVERTISING, params + adv + scan)


def remove(index: int, instance: int) -> None:
    _command(index, OP_REMOVE_ADVERTISING, bytes([instance]))
