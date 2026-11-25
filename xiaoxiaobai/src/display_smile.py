# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

"""Animated pixel-smile display for the 2 inch SPI LCD.

By default the LCD hangs off the head board (firmware/xiao_face): this loop
animates the face (face_engine.FaceAnimator, lips following what the speaker plays), streams
its 64x48 LED frames over the Pi's UART, forwards speaker requests and serves the board's camera (camera_stream.py, `make camera`). XIAOXIAOBAI_DISPLAY_BACKEND=ble uses
Bluetooth LE instead (no camera), =usb the ESP32's USB serial, and =spi drives an LCD
wired straight to the Pi's SPI header.
"""

from __future__ import annotations

import fcntl
import os
import signal
import struct
import subprocess
import sys
import threading
import time
import zlib
from contextlib import contextmanager
from dataclasses import dataclass
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

import numpy as np

from camera_stream import CameraHub, CameraServer, camera_port
from esp32_uart_ota import run_ota_request, take_ota_request
from esp32_link import (
    CAMERA_BACKENDS,
    ESP32_HOSTNAME,
    STATUS_CAMERA_FAILED,
    STATUS_CAMERA_ON,
    STATUS_MIC_FAILED,
    STATUS_OTA,
    STATUS_PSRAM,
    WIFI_EXPORT_PATH,
    WIFI_JOINING,
    WIFI_SCANNING,
    describe_wifi_diag,
    esp32_backend,
    load_wifi_payload,
    open_esp32_link,
    read_brightness,
    take_sound_request,
    write_esp32_info,
)
from voice_bridge import VoiceBridge, voice_socket_path
from board import BoardPlayer, Esp32Board
from expressions import (
    EXPRESSION_FRAME_DT_S,
    LIT_THRESHOLD,
    IdleCycle,
    new_animator,
    read_expression_name,
    render_expression_frame,
    render_led_rgb,
)


# Native LCD GRAM is 240x320. Frames are composed landscape (320x240) then
# rotated into that native buffer so a physically sideways module stays filled.
PANEL_WIDTH = 240
PANEL_HEIGHT = 320
WIDTH = PANEL_HEIGHT
HEIGHT = PANEL_WIDTH
# PIL counterclockwise degrees. 270 = previous 90 plus 180.
CONTENT_ROTATION_DEGREES = 270

SPI_BUS = 0
SPI_DEVICE = 0  # CE0 / GPIO8 / physical pin 24

DC_PIN = 25
RESET_PIN = 27
BACKLIGHT_PIN = 18
DISPLAY_LOCK_PATH = Path("/tmp/xiaoxiaobai_display.lock")
FRAME_INTERVAL_S = EXPRESSION_FRAME_DT_S
# The Pi renders the LED look itself for an LCD on its own SPI header, which costs more per frame.
LCD_FRAME_INTERVAL_S = 0.08
ESP32_RETRY_MIN_S = 1.0
# Bluetooth scans share the radio with the gamepad, so back off while the board is absent.
ESP32_RETRY_MAX_S = 30.0
# The firmware sends a status packet every second over the UART.
ESP32_SILENT_S = 3.0
WIFI_EXPORT_POLL_S = 5.0
WIFI_RESEND_S = 5.0
BRIGHTNESS_POLL_S = 0.2
# The ESP32 keeps the brightness in RAM only: resent so a rebooted board gets it back.
BRIGHTNESS_RESEND_S = 5.0


class DisplayAlreadyRunning(RuntimeError):
    pass


class _Image(Protocol):
    size: tuple[int, int]

    def convert(self, mode: str) -> "_Image":
        ...


class _GpioBackend(Protocol):
    HIGH: int
    LOW: int

    def output(self, pin: int, value: int) -> None:
        ...

    def cleanup(self, pins: list[int]) -> None:
        ...


class _RpiGpioBackend:
    def __init__(self, pins: list[int]) -> None:
        import RPi.GPIO as GPIO

        self._gpio = GPIO
        self.HIGH = GPIO.HIGH
        self.LOW = GPIO.LOW

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        for pin in pins:
            GPIO.setup(pin, GPIO.OUT)

    def output(self, pin: int, value: int) -> None:
        self._gpio.output(pin, value)

    def cleanup(self, pins: list[int]) -> None:
        self._gpio.cleanup(pins)


class _SysfsGpioBackend:
    """Fallback GPIO backend that does not need extra Python packages."""

    HIGH = 1
    LOW = 0

    def __init__(self, pins: list[int]) -> None:
        self._pins = sorted(set(pins))
        for pin in self._pins:
            gpio_dir = Path(f"/sys/class/gpio/gpio{pin}")
            if not gpio_dir.exists():
                Path("/sys/class/gpio/export").write_text(f"{pin}\n", encoding="ascii")
                time.sleep(0.05)
            (gpio_dir / "direction").write_text("out\n", encoding="ascii")

    def output(self, pin: int, value: int) -> None:
        Path(f"/sys/class/gpio/gpio{pin}/value").write_text(f"{value}\n", encoding="ascii")

    def cleanup(self, pins: list[int]) -> None:
        for pin in pins:
            try:
                Path("/sys/class/gpio/unexport").write_text(f"{pin}\n", encoding="ascii")
            except OSError:
                pass


def _ioc(direction: int, request_type: int, number: int, size: int) -> int:
    return (direction << 30) | (size << 16) | (request_type << 8) | number


GPIO_GET_LINEHANDLE_IOCTL = _ioc(1 | 2, 0xB4, 0x03, 364)
GPIOHANDLE_SET_LINE_VALUES_IOCTL = _ioc(1 | 2, 0xB4, 0x09, 64)
GPIOHANDLE_REQUEST_OUTPUT = 1 << 1


class _CdevGpioBackend:
    """GPIO character-device backend for newer Raspberry Pi kernels."""

    HIGH = 1
    LOW = 0

    def __init__(self, pins: list[int]) -> None:
        self._pins = list(dict.fromkeys(pins))
        self._values = {pin: 0 for pin in self._pins}
        self._chip_fd: int | None = None
        self._line_fd: int | None = None

        last_error: OSError | None = None
        for chip in sorted(Path("/dev").glob("gpiochip*")):
            try:
                self._open_chip(chip)
                return
            except OSError as exc:
                last_error = exc
                self.close()

        if last_error is not None:
            raise last_error
        raise FileNotFoundError("no /dev/gpiochip* device found")

    def output(self, pin: int, value: int) -> None:
        self._values[pin] = 1 if value else 0
        values = [0] * 64
        for index, gpio_pin in enumerate(self._pins):
            values[index] = self._values[gpio_pin]
        data = struct.pack("64B", *values)
        fcntl.ioctl(self._line_fd, GPIOHANDLE_SET_LINE_VALUES_IOCTL, data)

    def cleanup(self, pins: list[int]) -> None:
        _ = pins
        self.close()

    def close(self) -> None:
        for fd_name in ("_line_fd", "_chip_fd"):
            fd = getattr(self, fd_name)
            if fd is not None:
                os.close(fd)
                setattr(self, fd_name, None)

    def _open_chip(self, chip: Path) -> None:
        chip_fd = os.open(chip, os.O_RDONLY)
        offsets = self._pins + [0] * (64 - len(self._pins))
        default_values = [0] * 64
        label = b"xiaoxiaobai-display"

        request = struct.pack(
            "64I I 64B 32s I i",
            *offsets,
            GPIOHANDLE_REQUEST_OUTPUT,
            *default_values,
            label,
            len(self._pins),
            0,
        )
        request = bytearray(request)
        fcntl.ioctl(chip_fd, GPIO_GET_LINEHANDLE_IOCTL, request, True)
        line_fd = struct.unpack_from("i", request, 360)[0]

        self._chip_fd = chip_fd
        self._line_fd = line_fd


def _open_gpio(pins: list[int]) -> _GpioBackend:
    try:
        return _RpiGpioBackend(pins)
    except Exception:
        pass

    try:
        return _CdevGpioBackend(pins)
    except Exception:
        return _SysfsGpioBackend(pins)


def _env_flag_enabled(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


@contextmanager
def display_lock() -> Iterator[None]:
    fd = os.open(DISPLAY_LOCK_PATH, os.O_CREAT | os.O_RDWR, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(fd)
        raise DisplayAlreadyRunning("another display animation is already running") from exc

    try:
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


@dataclass
class DisplayAnimation:
    process: subprocess.Popen

    def stop(self, timeout_s: float = 2.0) -> None:
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGINT)
        try:
            self.process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            self.process.kill()


class SpiLcd:
    """Small SPI driver for the 2 inch 240x320 LCD."""

    def __init__(
        self,
        width: int = WIDTH,
        height: int = HEIGHT,
        panel_width: int = PANEL_WIDTH,
        panel_height: int = PANEL_HEIGHT,
        content_rotation_degrees: int = CONTENT_ROTATION_DEGREES,
        spi_bus: int = SPI_BUS,
        spi_device: int = SPI_DEVICE,
        dc_pin: int = DC_PIN,
        reset_pin: int = RESET_PIN,
        backlight_pin: int = BACKLIGHT_PIN,
        spi_speed_hz: int = 40_000_000,
    ) -> None:
        import spidev

        self.width = width
        self.height = height
        self.panel_width = panel_width
        self.panel_height = panel_height
        self.content_rotation_degrees = content_rotation_degrees
        self.dc_pin = dc_pin
        self.reset_pin = reset_pin
        self.backlight_pin = backlight_pin
        self.gpio = _open_gpio([self.dc_pin, self.reset_pin, self.backlight_pin])

        self.spi = spidev.SpiDev()
        self.spi.open(spi_bus, spi_device)
        self.spi.max_speed_hz = spi_speed_hz
        self.spi.mode = 0

    def init(self) -> None:
        self._reset()

        self.command(0x36)  # MADCTL: native portrait, RGB order
        self.data([0x00])

        self.command(0x3A)  # COLMOD: 16-bit RGB565
        self.data([0x55])

        self.command(0x21)  # Display inversion on

        self.command(0x2A)  # Column address: native 0-239
        self.data([0x00, 0x00, 0x00, 0xEF])

        self.command(0x2B)  # Row address: native 0-319
        self.data([0x00, 0x00, 0x01, 0x3F])

        self.command(0xB2)  # Porch control
        self.data([0x0C, 0x0C, 0x00, 0x33, 0x33])

        self.command(0xB7)  # Gate control
        self.data([0x35])

        self.command(0xBB)  # VCOM setting
        self.data([0x1F])

        self.command(0xC0)  # LCM control
        self.data([0x2C])

        self.command(0xC2)  # VDV and VRH command enable
        self.data([0x01])

        self.command(0xC3)  # VRH set
        self.data([0x12])

        self.command(0xC4)  # VDV set
        self.data([0x20])

        self.command(0xC6)  # Frame rate control
        self.data([0x0F])

        self.command(0xD0)  # Power control
        self.data([0xA4, 0xA1])

        self.command(0xE0)  # Positive voltage gamma
        self.data([0xD0, 0x08, 0x11, 0x08, 0x0C, 0x15, 0x39, 0x33, 0x50, 0x36, 0x13, 0x14, 0x29, 0x2D])

        self.command(0xE1)  # Negative voltage gamma
        self.data([0xD0, 0x08, 0x10, 0x08, 0x06, 0x06, 0x39, 0x44, 0x51, 0x0B, 0x16, 0x14, 0x2F, 0x31])

        self.command(0x11)  # Sleep out
        time.sleep(0.12)
        self.command(0x29)  # Display on
        self.backlight(True)
        self._fill_native((0, 0, 0))

    def command(self, value: int) -> None:
        self.gpio.output(self.dc_pin, self.gpio.LOW)
        self.spi.writebytes([value])

    def data(self, values: list[int] | bytes | bytearray) -> None:
        self.gpio.output(self.dc_pin, self.gpio.HIGH)
        self._write(values)

    def display(self, image: _Image) -> None:
        if image.size != (self.width, self.height):
            raise ValueError(f"image size must be {(self.width, self.height)}, got {image.size}")

        native = image.rotate(self.content_rotation_degrees, expand=True)
        if native.size != (self.panel_width, self.panel_height):
            raise ValueError(
                f"rotated image size must be {(self.panel_width, self.panel_height)}, got {native.size}"
            )
        self.show_rgb565(_rgb565_bytes(native))

    def show_rgb565(self, payload: bytes | bytearray) -> None:
        expected = self.panel_width * self.panel_height * 2
        if len(payload) != expected:
            raise ValueError(f"RGB565 payload must be {expected} bytes, got {len(payload)}")
        self._set_window(0, 0, self.panel_width - 1, self.panel_height - 1)
        self.gpio.output(self.dc_pin, self.gpio.HIGH)
        self._write(payload)

    def _fill_native(self, color: tuple[int, int, int]) -> None:
        from PIL import Image

        self._show_native(Image.new("RGB", (self.panel_width, self.panel_height), color))

    def _show_native(self, image: _Image) -> None:
        self.command(0x36)
        self.data([0x00])
        self._set_window(0, 0, self.panel_width - 1, self.panel_height - 1)
        self.gpio.output(self.dc_pin, self.gpio.HIGH)
        self._write(_rgb565_bytes(image))

    def backlight(self, enabled: bool) -> None:
        self.gpio.output(self.backlight_pin, self.gpio.HIGH if enabled else self.gpio.LOW)

    def close(self) -> None:
        try:
            self.backlight(False)
            self.spi.close()
        finally:
            self.gpio.cleanup([self.dc_pin, self.reset_pin, self.backlight_pin])

    def _reset(self) -> None:
        self.gpio.output(self.reset_pin, self.gpio.HIGH)
        time.sleep(0.01)
        self.gpio.output(self.reset_pin, self.gpio.LOW)
        time.sleep(0.01)
        self.gpio.output(self.reset_pin, self.gpio.HIGH)
        time.sleep(0.12)

    def _set_window(self, x0: int, y0: int, x1: int, y1: int) -> None:
        self.command(0x2A)  # Column address set
        self.data([x0 >> 8, x0 & 0xFF, x1 >> 8, x1 & 0xFF])

        self.command(0x2B)  # Row address set
        self.data([y0 >> 8, y0 & 0xFF, y1 >> 8, y1 & 0xFF])

        self.command(0x2C)  # Memory write

    def _write(self, values: list[int] | bytes | bytearray, chunk_size: int = 4096) -> None:
        if isinstance(values, list):
            for start in range(0, len(values), chunk_size):
                self.spi.writebytes(values[start : start + chunk_size])
            return
        view = memoryview(values)
        for start in range(0, len(view), chunk_size):
            self.spi.writebytes2(view[start : start + chunk_size])


def _rgb565_bytes(image: _Image) -> bytes:
    return _rgb565_from_array(np.asarray(image.convert("RGB"), dtype=np.uint8))


def _rgb565_from_array(rgb: np.ndarray) -> bytes:
    pixels = rgb.astype(np.uint16, copy=False)
    rgb565 = ((pixels[:, :, 0] & 0xF8) << 8) | ((pixels[:, :, 1] & 0xFC) << 3) | (pixels[:, :, 2] >> 3)
    return rgb565.byteswap().tobytes()


def _encode_native_frame(levels: bytes, rotation_degrees: int) -> bytes:
    rgb = render_led_rgb(levels)
    turns = (rotation_degrees // 90) % 4
    if turns:
        rgb = np.rot90(rgb, turns)
    return _rgb565_from_array(rgb)


def render_smile_frame(frame: int, width: int = WIDTH, height: int = HEIGHT, name: str | None = None):
    """Render one LED-matrix expression frame for the landscape panel."""
    return render_expression_frame(name or read_expression_name(), frame, width, height)


def _face_mask_bytes(levels: bytes) -> bytes:
    """1-bit mask for firmware that predates MSG_FACE_LEVELS."""
    lit = (np.frombuffer(levels, dtype=np.uint8) & 31) >= LIT_THRESHOLD
    return np.packbits(lit, bitorder="little").tobytes()


def _log(message: str) -> None:
    print(message, end="\r\n", flush=True)


def _start_camera_server(hub: CameraHub) -> CameraServer | None:
    port = camera_port()
    try:
        return CameraServer(hub, port)
    except OSError as exc:
        _log(f"Display: camera stream off, cannot listen on 127.0.0.1:{port} ({exc.strerror})")
        return None


def _esp32_state(link, connected_at: float, now: float, watching: bool) -> str:
    if not link.reports_status:
        return f"ESP32 on {link.description}"
    if link.last_status_s is None and now - connected_at < ESP32_SILENT_S:
        return "waiting for ESP32"
    if link.last_status_s is None or now - link.last_status_s > ESP32_SILENT_S:
        return (
            f"ESP32 not answering on {link.description}: check the TX/RX/GND wires "
            "and that the current firmware is flashed"
        )
    status = link.status or 0
    if status & STATUS_OTA:
        return "ESP32 receiving an OTA firmware update"
    if status & STATUS_MIC_FAILED:
        return "microphone failed to start: check the Sense expansion board and its connector"
    if status & STATUS_CAMERA_ON:
        return "camera streaming"
    if status & STATUS_CAMERA_FAILED:
        if not status & STATUS_PSRAM:
            return "camera off: PSRAM disabled, reflash with Arduino IDE Tools > PSRAM > OPI PSRAM"
        return "camera failed to start: check the Sense camera module and its connector"
    return "camera starting" if watching else "ESP32 ready"


class _BrightnessSync:
    """Sends the backlight brightness set in the web UI (esp32_link.write_brightness)."""

    def __init__(self) -> None:
        self._percent = read_brightness()
        self._next_read = 0.0
        self._sent: tuple[object, int] | None = None
        self._next_send = 0.0

    def tick(self, link, now: float) -> None:
        if now >= self._next_read:
            self._next_read = now + BRIGHTNESS_POLL_S
            self._percent = read_brightness()
        if self._sent != (link, self._percent) or now >= self._next_send:
            link.brightness(self._percent)
            self._sent = (link, self._percent)
            self._next_send = now + BRIGHTNESS_RESEND_S


class _WifiSync:
    """Keeps the ESP32's Wi-Fi list equal to the Pi's and reports where the board is."""

    def __init__(self) -> None:
        self._payload: bytes | None = None
        self._next_read = 0.0
        self._next_send = 0.0
        self._ip: str | None = None
        self._warned = False
        self._problem: tuple | None = None

    def tick(self, link, now: float) -> None:
        if not link.reports_status or link.wifi_crc is None:
            return
        if now >= self._next_read:
            self._next_read = now + WIFI_EXPORT_POLL_S
            self._payload = load_wifi_payload()
            if self._payload is None and not self._warned:
                _log(
                    f"Display: no Wi-Fi export at {WIFI_EXPORT_PATH}; run `make wifi-sync-enable` "
                    "once so the ESP32 joins the Pi's Wi-Fi"
                )
                self._warned = True
        if self._payload is not None and zlib.crc32(self._payload) != link.wifi_crc and now >= self._next_send:
            link.wifi(self._payload)
            self._next_send = now + WIFI_RESEND_S
            _log(f"Display: sent the Pi's {self._payload[0]} Wi-Fi network(s) to the ESP32")
        diag = link.wifi_diag
        if diag and diag[0] not in (WIFI_SCANNING, WIFI_JOINING):
            problem = describe_wifi_diag(diag)
            key = (diag[0], diag[1] == 0, diag[3]) if problem else None
            if key != self._problem:
                self._problem = key
                if problem:
                    _log(f"Display: {problem}")
        if link.wifi_ip != self._ip:
            self._ip = link.wifi_ip
            if self._ip:
                _log(f"Display: ESP32 on Wi-Fi at {self._ip} (http://{ESP32_HOSTNAME}.local/)")
            else:
                _log("Display: ESP32 left Wi-Fi")
            try:
                write_esp32_info(self._ip, link.wifi_rssi)
            except OSError:
                pass


def _start_voice_bridge() -> VoiceBridge | None:
    try:
        return VoiceBridge()
    except OSError as exc:
        _log(f"Display: voice bridge off, cannot listen on {voice_socket_path()} ({exc.strerror})")
        return None


def _esp32_worker(stop_event: threading.Event) -> None:
    idle = IdleCycle()
    animator = new_animator(idle.resolve(read_expression_name(), time.monotonic()), time.monotonic())
    board = BoardPlayer(log=_log)
    board_out = Esp32Board(log=_log)
    hub = CameraHub()
    wifi_sync = _WifiSync()
    brightness = _BrightnessSync()
    server = _start_camera_server(hub) if esp32_backend() in CAMERA_BACKENDS else None
    voice = _start_voice_bridge() if esp32_backend() in CAMERA_BACKENDS else None
    voice_connected = False
    link = None
    connected_at = 0.0
    retry_s = ESP32_RETRY_MIN_S
    next_connect = 0.0
    last_error: str | None = None
    try:
        while not stop_event.is_set():
            tick_start = time.monotonic()
            animator.set_expression(idle.resolve(read_expression_name(), tick_start), tick_start)

            if link is None and tick_start >= next_connect:
                try:
                    link = open_esp32_link(on_frame=hub.publish)
                    connected_at = time.monotonic()
                    animator.wake(connected_at)
                    hub.link_changed()
                    if hasattr(link, "on_speaker"):
                        link.on_speaker = lambda db, zcr: animator.feed_speech(db, zcr, time.monotonic())
                    if voice is not None and link.supports_audio:
                        voice.attach(link)
                    _log(f"Display: ESP32 link open via {link.description}")
                    last_error = None
                    retry_s = ESP32_RETRY_MIN_S
                except OSError as exc:
                    if str(exc) != last_error:
                        _log(f"Display: {exc}; retrying")
                        last_error = str(exc)
                        hub.state = str(exc)
                    next_connect = time.monotonic() + retry_s
                    retry_s = min(retry_s * 2, ESP32_RETRY_MAX_S)
                    tick_start = time.monotonic()

            if link is not None:
                try:
                    ota_image = take_ota_request()
                    if ota_image is not None:
                        _log(f"Display: UART firmware update from {ota_image.name}")
                        run_ota_request(link, ota_image)
                        continue
                    board.poll(tick_start)
                    board_out.tick(link, board.picture(tick_start), tick_start)
                    face = animator.frame(tick_start)
                    if link.face_levels:
                        link.show_levels(face)
                    else:
                        link.show_face(_face_mask_bytes(face))
                    sound = take_sound_request()
                    if sound is not None:
                        link.play(*sound)
                    if link.supports_camera:
                        command = hub.command(tick_start)
                        if command is not None:
                            link.camera(*command)
                    if voice is not None and link.supports_audio:
                        voice.tick(link, tick_start)
                    wifi_sync.tick(link, tick_start)
                    brightness.tick(link, tick_start)
                except OSError as exc:
                    _log(f"Display: ESP32 link lost ({exc}); retrying")
                    last_error = str(exc)
                    hub.state = f"ESP32 link lost ({exc})"
                    if voice is not None:
                        voice.detach()
                    try:
                        link.close()
                    except OSError:
                        pass
                    link = None
                    next_connect = time.monotonic() + ESP32_RETRY_MIN_S

            if link is not None:
                state = _esp32_state(link, connected_at, time.monotonic(), hub.watching)
                if link.reports_status and state != hub.state and state != "waiting for ESP32":
                    _log(f"Display: {state}")
                hub.state = state
                if link.supports_camera:
                    hub.link_stats = {"dropped_frames": link.dropped_frames, "bad_packets": link.bad_packets}

            if voice is not None and voice.connected != voice_connected:
                voice_connected = voice.connected
                _log(f"Display: voice client {'connected' if voice_connected else 'disconnected'}")

            interval = 1.0 / link.face_fps if link is not None else FRAME_INTERVAL_S
            leftover = interval - (time.monotonic() - tick_start)
            if leftover > 0:
                stop_event.wait(leftover)
    finally:
        if server is not None:
            server.close()
        if voice is not None:
            voice.close()
        if link is not None:
            try:
                link.backlight(False)
            except OSError:
                pass
            if link.supports_audio:
                try:
                    link.mic(False)
                    link.voice_stop()
                except OSError:
                    pass
            if link.supports_camera:
                try:
                    link.camera(False, 0, 0)
                except OSError:
                    pass
            try:
                link.close()
            except OSError:
                pass


def display_worker(stop_event: threading.Event) -> None:
    with display_lock():
        if esp32_backend() == "spi":
            _lcd_worker(stop_event)
        else:
            _esp32_worker(stop_event)


def _lcd_worker(stop_event: threading.Event) -> None:
    lcd = SpiLcd()
    lcd.init()
    idle = IdleCycle()
    animator = new_animator(idle.resolve(read_expression_name(), time.monotonic()), time.monotonic())
    animator.wake(time.monotonic())
    board = BoardPlayer(log=_log)
    shown = None
    try:
        while not stop_event.is_set():
            tick_start = time.monotonic()
            animator.set_expression(idle.resolve(read_expression_name(), tick_start), tick_start)
            board.poll(tick_start)
            picture = board.picture(tick_start)
            face = animator.frame(tick_start)
            if picture is None:
                shown = None
                lcd.show_rgb565(_encode_native_frame(face, lcd.content_rotation_degrees))
            elif picture[0] != shown:
                shown = picture[0]
                lcd.display(picture[1])
            leftover = LCD_FRAME_INTERVAL_S - (time.monotonic() - tick_start)
            if leftover > 0:
                stop_event.wait(leftover)
    finally:
        lcd.close()


def start_smile_animation() -> DisplayAnimation | None:
    """Start the face animation in a background process (exits at once if the display service runs).

    A process of its own, at a lower priority, so drawing the face never holds up the control loop.
    Set XIAOXIAOBAI_DISPLAY=0 to disable it without changing code.
    """
    if not _env_flag_enabled("XIAOXIAOBAI_DISPLAY", default=True):
        return None
    try:
        process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve())], preexec_fn=lambda: os.nice(5))
    except OSError as exc:
        print(f"Display disabled: {exc}", flush=True)
        return None
    return DisplayAnimation(process=process)


if __name__ == "__main__":
    event = threading.Event()
    try:
        display_worker(event)
    except KeyboardInterrupt:
        event.set()
    except DisplayAlreadyRunning:
        raise SystemExit(0)
    except Exception as exc:  # noqa: BLE001 - foreground service should show the hardware error
        print(f"Display disabled: {exc}", flush=True)
        raise SystemExit(1) from exc
