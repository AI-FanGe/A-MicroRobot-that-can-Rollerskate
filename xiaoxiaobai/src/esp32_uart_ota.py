# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

"""Flash the ESP32 face firmware over the Pi's UART.

The image is written into the inactive OTA slot (the one Wi-Fi OTA uses too) and checked
before the board reboots into it, so a broken transfer leaves the current firmware in place.
Wi-Fi OTA stays available; this path is the one that works whenever the Pi is reachable,
even when the ESP32 itself has no Wi-Fi.

    python src/esp32_uart_ota.py /tmp/xiao_face.ino.bin

When the display loop already owns the UART, the image is handed to it through
OTA_REQUEST_PATH. Otherwise this process opens /dev/serial0 itself, stopping the
xiaoxiaobai-display service first if that is what holds the port.
"""

from __future__ import annotations

import fcntl
import json
import os
import struct
import subprocess
import sys
import threading
import time
import zlib
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from esp32_link import (
    OTA_ABORT,
    OTA_ACK,
    OTA_BEGIN,
    OTA_CHUNK_BYTES,
    OTA_DATA,
    OTA_DONE,
    OTA_END,
    OTA_ERROR,
    Esp32UartLink,
    esp32_uart_baud,
    esp32_uart_path,
)

# app0 / app1 in the default_8MB partition table (kOtaPartitionBytes in the firmware).
OTA_PARTITION_BYTES = 0x330000
OTA_REQUEST_PATH = Path("/tmp/xiaoxiaobai_esp32_ota.json")
OTA_STATUS_PATH = Path("/tmp/xiaoxiaobai_esp32_ota.status")
# display_smile.DISPLAY_LOCK_PATH; held the whole time the display loop owns the UART.
DISPLAY_LOCK_PATH = Path("/tmp/xiaoxiaobai_display.lock")
DISPLAY_SERVICE = "xiaoxiaobai-display"
# The first ack waits out the OTA slot erase; a data ack is one flash page.
BEGIN_TIMEOUT_S = 45.0
# A chunk on a 64 KB boundary waits for that block to erase.
DATA_TIMEOUT_S = 6.0
END_TIMEOUT_S = 5.0
DATA_ATTEMPTS = 4
# How long to wait for a running display loop to pick the request up.
DISPLAY_PICKUP_S = 4.0
# Give up only after the byte offset stops moving. A fixed budget (30 s + size/20 KB/s)
# ran out around 93% of a 1.3 MB image: each 507-byte packet is a round trip, about
# 13 KB/s, so the image needs longer than that budget. An erase of a 64 KB block plus
# the data retries fits in STALL_S; BEGIN_TIMEOUT_S * 2 is the longest silent erase.
STALL_S = BEGIN_TIMEOUT_S * 2 + DATA_TIMEOUT_S


class OtaError(RuntimeError):
    pass


class _Acks:
    def __init__(self) -> None:
        self._event = threading.Event()
        self._payload = b""

    def __call__(self, payload: bytes) -> None:
        self._payload = payload
        self._event.set()

    def wait(self, timeout: float) -> bytes | None:
        if not self._event.wait(timeout):
            return None
        self._event.clear()
        return self._payload


def _reply(payload: bytes) -> tuple[int, int, str]:
    if len(payload) < 5:
        raise OtaError(f"short OTA status ({len(payload)} bytes)")
    state, offset = struct.unpack_from("<BI", payload)
    return state, offset, payload[5:].decode(errors="replace")


def _transact(link: Esp32UartLink, acks: _Acks, packet: bytes, timeout: float, attempts: int) -> tuple[int, int, str]:
    last = "no reply"
    for _ in range(attempts):
        link.ota(packet)
        payload = acks.wait(timeout)
        if payload is None:
            last = "no reply"
            continue
        state, offset, text = _reply(payload)
        if state == OTA_ERROR:
            raise OtaError(text or "the ESP32 rejected the update")
        return state, offset, text
    raise OtaError(
        f"{last} over the UART. The board is running firmware without UART updates: "
        "flash this build once with `make firmware-ota` (ESP32 on Wi-Fi) or `make firmware-upload` (USB). "
        "After that, `make firmware-uart` works whenever the Pi is on the network."
    )


def flash_link(link: Esp32UartLink, image: bytes, progress: Callable[[int, int], None] | None = None) -> None:
    """Write `image` through an open link. Raises OtaError; the ESP32 reboots on success."""
    if not image or len(image) > OTA_PARTITION_BYTES:
        raise OtaError(f"image is {len(image)} bytes; the OTA slot holds at most {OTA_PARTITION_BYTES}")
    acks = _Acks()
    previous = link.on_ota
    link.on_ota = acks
    size = len(image)
    try:
        crc = zlib.crc32(image) & 0xFFFFFFFF
        state, offset, _ = _transact(link, acks, struct.pack("<BII", OTA_BEGIN, size, crc), BEGIN_TIMEOUT_S, 2)
        if state != OTA_ACK or offset != 0:
            raise OtaError(f"unexpected answer to the start of the update (state {state}, offset {offset})")
        if progress is not None:
            progress(0, size)
        sent = 0
        while sent < size:
            chunk = image[sent : sent + OTA_CHUNK_BYTES]
            state, offset, _ = _transact(link, acks, struct.pack("<BI", OTA_DATA, sent) + chunk, DATA_TIMEOUT_S, DATA_ATTEMPTS)
            if state != OTA_ACK:
                raise OtaError(f"unexpected answer while writing (state {state})")
            if offset > size or offset < sent:
                raise OtaError(f"ESP32 asked to resume at {offset}, outside {sent}..{size}")
            sent = offset
            if progress is not None:
                progress(sent, size)
        for _ in range(4):
            link.ota(bytes((OTA_END,)))
            payload = acks.wait(END_TIMEOUT_S)
            if payload is None:
                continue
            state, _, text = _reply(payload)
            if state == OTA_ERROR:
                raise OtaError(text or "the ESP32 rejected the image")
            if state == OTA_DONE:
                return
        raise OtaError("the ESP32 accepted every byte but did not confirm the reboot")
    except OtaError:
        try:
            link.ota(bytes((OTA_ABORT,)))
        except OSError:
            pass
        raise
    finally:
        link.on_ota = previous


def write_status(state: str, offset: int, size: int, message: str = "") -> None:
    payload = json.dumps({"state": state, "offset": offset, "size": size, "message": message})
    temporary = OTA_STATUS_PATH.with_suffix(".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(OTA_STATUS_PATH)


def take_ota_request() -> Path | None:
    """The firmware image a flasher asked the display loop to send, if there is one."""
    try:
        request = json.loads(OTA_REQUEST_PATH.read_text(encoding="utf-8"))
        OTA_REQUEST_PATH.unlink()
    except (OSError, json.JSONDecodeError):
        return None
    path = request.get("path")
    return Path(path) if isinstance(path, str) and path else None


def run_ota_request(link: Esp32UartLink, path: Path) -> None:
    """Called by the display loop, which owns the UART."""
    try:
        image = path.read_bytes()
    except OSError as exc:
        write_status("error", 0, 0, f"cannot read {path} ({exc.strerror})")
        return
    last_report = -1

    def progress(offset: int, size: int) -> None:
        nonlocal last_report
        write_status("running", offset, size)
        step = offset // (64 * 1024)
        if offset == 0 or offset == size or step != last_report:
            last_report = step
            print(f"UART OTA {offset / size:.0%} ({offset}/{size})", flush=True)

    try:
        write_status("running", 0, len(image))
        flash_link(link, image, progress)
    except (OtaError, OSError) as exc:
        write_status("error", 0, len(image), str(exc))
        print(f"UART OTA failed: {exc}", flush=True)
        return
    write_status("done", len(image), len(image))
    print("UART OTA done, ESP32 rebooting", flush=True)


def _read_status() -> dict | None:
    try:
        return json.loads(OTA_STATUS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _flash_via_display(image_path: Path) -> bool:
    """True when the running display loop took the request (it may still fail afterwards)."""
    OTA_STATUS_PATH.unlink(missing_ok=True)
    temporary = OTA_REQUEST_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps({"path": str(image_path)}), encoding="utf-8")
    temporary.replace(OTA_REQUEST_PATH)
    deadline = time.monotonic() + DISPLAY_PICKUP_S
    while time.monotonic() < deadline:
        if _read_status() is not None:
            return True
        time.sleep(0.1)
    if not OTA_REQUEST_PATH.exists():
        # The display loop already took the request; give it a moment to report.
        deadline = time.monotonic() + DISPLAY_PICKUP_S
        while time.monotonic() < deadline:
            if _read_status() is not None:
                return True
            time.sleep(0.1)
        raise OtaError("the display loop took the update but never reported back")
    OTA_REQUEST_PATH.unlink(missing_ok=True)
    return False


def _wait_for_display(size: int) -> None:
    """Block until the display loop finishes. The clock resets whenever another chunk lands."""
    stall_deadline = time.monotonic() + STALL_S
    last = -1
    last_step = -1
    while True:
        status = _read_status()
        if status is not None:
            state = status.get("state")
            offset = int(status.get("offset") or 0)
            if offset != last and state == "running":
                last = offset
                stall_deadline = time.monotonic() + STALL_S
                step = offset // (64 * 1024)
                if step != last_step:
                    last_step = step
                    print(f"UART OTA {offset / size:.0%} ({offset}/{size})", flush=True)
            if state == "done":
                print("UART OTA done, ESP32 rebooting", flush=True)
                return
            if state == "error":
                raise OtaError(status.get("message") or "the display loop reported a failure")
        if time.monotonic() > stall_deadline:
            raise OtaError("the display loop stopped reporting; the update did not finish")
        time.sleep(0.2)


def _display_lock_held() -> bool:
    """True when a display loop has the UART. Opening /dev/serial0 cannot tell: a second open succeeds."""
    fd = os.open(DISPLAY_LOCK_PATH, os.O_CREAT | os.O_RDWR, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return True
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)
    return False


@contextmanager
def _hold_display_lock() -> Iterator[None]:
    fd = os.open(DISPLAY_LOCK_PATH, os.O_CREAT | os.O_RDWR, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise OtaError("the display loop still has the UART; stop `make run` and retry") from None
    try:
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _service_active() -> bool:
    return subprocess.run(["systemctl", "is-active", "--quiet", DISPLAY_SERVICE], check=False).returncode == 0


def _systemctl(action: str) -> None:
    subprocess.run(["sudo", "systemctl", action, DISPLAY_SERVICE], check=False)


def _flash_direct(image: bytes) -> None:
    stopped = False
    if _service_active():
        print(f"stopping {DISPLAY_SERVICE} so nothing else writes the UART (sudo)...", flush=True)
        _systemctl("stop")
        stopped = True
    deadline = time.monotonic() + 3.0
    while _display_lock_held() and time.monotonic() < deadline:
        time.sleep(0.1)
    try:
        with _hold_display_lock():
            link = Esp32UartLink(esp32_uart_path(), esp32_uart_baud())
            try:
                last = -1

                def progress(offset: int, size: int) -> None:
                    nonlocal last
                    step = offset // (64 * 1024)
                    if offset == 0 or offset == size or step != last:
                        last = step
                        print(f"UART OTA {offset / size:.0%} ({offset}/{size})", flush=True)

                flash_link(link, image, progress)
                print("UART OTA done, ESP32 rebooting", flush=True)
            finally:
                link.close()
    finally:
        if stopped:
            print(f"starting {DISPLAY_SERVICE} again", flush=True)
            _systemctl("start")


def flash_file(path: Path) -> None:
    try:
        image = path.read_bytes()
    except OSError as exc:
        raise OtaError(f"cannot read {path} ({exc.strerror})") from exc
    if not image or len(image) > OTA_PARTITION_BYTES:
        raise OtaError(f"{path} is {len(image)} bytes; the OTA slot holds at most {OTA_PARTITION_BYTES}")
    print(f"ESP32 UART update, {len(image)} bytes, via {esp32_uart_path()}", flush=True)
    try:
        if _display_lock_held():
            print("display loop has the UART; asking it to write the image", flush=True)
            if _flash_via_display(path.resolve()):
                _wait_for_display(len(image))
                return
            print("this display loop cannot write firmware; taking the UART over", flush=True)
        _flash_direct(image)
    except OSError as exc:
        raise OtaError(f"UART error: {exc}") from exc


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} FIRMWARE.bin")
    try:
        flash_file(Path(sys.argv[1]))
    except OtaError as exc:
        print(f"UART OTA failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
