# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

"""Link to the head board that drives the face LCD, the speaker, the camera and the microphone.

The default transport is the Pi's hardware UART (GPIO14/15, 3 Mbaud), which needs no
network and also carries the camera's JPEG frames and the microphone's PCM back. Bluetooth LE (face + speaker
only) and USB serial (bench debugging) carry the same byte stream, which must match
firmware/xiao_face/protocol.h:
    A5 5A | type u8 | length u16 LE | payload | XOR of payload bytes
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import fcntl
import json
import os
import select
import struct
import termios
import threading
import time
import tty
import zlib
from collections.abc import Callable
from functools import reduce
from operator import xor
from pathlib import Path

import numpy as np

ESP32_BACKEND_ENV = "XIAOXIAOBAI_DISPLAY_BACKEND"
ESP32_BLE_ADDRESS_ENV = "XIAOXIAOBAI_ESP32_BLE_ADDRESS"
ESP32_PORT_ENV = "XIAOXIAOBAI_ESP32_PORT"
ESP32_PORT_PATTERNS = ("*Espressif*", "*XIAO*", "*Seeed*")
SERIAL_BY_ID = Path("/dev/serial/by-id")

ESP32_UART_ENV = "XIAOXIAOBAI_ESP32_UART"
ESP32_UART_DEFAULT = "/dev/serial0"
ESP32_UART_BAUD_ENV = "XIAOXIAOBAI_ESP32_UART_BAUD"
# Must match kLinkBaud in firmware/xiao_face/xiao_face.ino.
ESP32_UART_BAUD = 3_000_000
CAMERA_BACKENDS = ("uart", "usb")

BLE_SERVICE_UUID = "8f6a0001-3c1e-4d2a-9b7e-6d6963726f62"
BLE_RX_UUID = "8f6a0002-3c1e-4d2a-9b7e-6d6963726f62"
BLE_SCAN_TIMEOUT_S = 4.0
BLE_CONNECT_TIMEOUT_S = 10.0
BLE_WRITE_TIMEOUT_S = 2.0

MAGIC = b"\xa5\x5a"
MSG_FACE = 0x01
MSG_TONES = 0x02
MSG_BACKLIGHT = 0x03
MSG_FACE_RLE = 0x04
MSG_CAMERA = 0x05
MSG_WIFI = 0x06
MSG_PCM = 0x07
MSG_PCM_FLUSH = 0x08
MSG_MIC = 0x09
MSG_VOICE_CONFIG = 0x0A
MSG_VOICE_STOP = 0x0B
MSG_VOICE_KEEPALIVE = 0x0C
MSG_OTA = 0x0D
MSG_FACE_LEVELS = 0x0E
# offset u16, total u16, then a slice of an Omni session.update carrying knowledge-base passages for
# what the user just said: the ESP32 sends it and, if the reply has not started, has it answered again
MSG_VOICE_CONTEXT = 0x0F
# Full-screen picture instead of the face (image_packets); the ESP32 answers kImageEnd with MSG_IMAGE_STATUS.
MSG_IMAGE = 0x10
MSG_BRIGHTNESS = 0x11  # backlight u8 1-100 %
MSG_CAMERA_CHUNK = 0x81
MSG_STATUS = 0x82
MSG_MIC_CHUNK = 0x83
MSG_VOICE_EVENT = 0x84
MSG_OTA_STATUS = 0x85
MSG_SPEAKER = 0x86
MSG_IMAGE_STATUS = 0x87  # shown u8, crc32 u32 of the picture
# MSG_FACE_LEVELS: band index | FACE_COMMIT on the frame's last band, then zero_rle(band).
FACE_COMMIT = 0x80
FACE_BANDS = 16
FACE_BAND_BYTES = 64 * 48 // FACE_BANDS
# Unchanged bands are skipped, but the whole frame goes out this often (the ESP32 may have rebooted).
FACE_FULL_RESEND_S = 2.0
# MSG_SPEAKER: loudness u8 = dB * SPEAKER_DB_SCALE, zero-crossing rate u8 = rate * 255.
SPEAKER_DB_SCALE = 2.83
# Feature byte of MSG_STATUS (kFeature* in protocol.h)
FEATURE_FACE_LEVELS = 1 << 0
FEATURE_IMAGE = 1 << 1
FEATURE_BRIGHTNESS = 1 << 2
# MSG_IMAGE ops (kImage* in protocol.h)
IMAGE_BEGIN, IMAGE_DATA, IMAGE_END, IMAGE_HIDE = 1, 2, 3, 4
# MSG_OTA payload (kOta* in protocol.h): op, then begin's size+crc or data's offset+bytes.
OTA_BEGIN, OTA_DATA, OTA_END, OTA_ABORT = 1, 2, 3, 4
OTA_ACK, OTA_ERROR, OTA_DONE = 1, 2, 3
MAX_PAYLOAD = 512
# Bytes of firmware per MSG_OTA data packet (op u8 + offset u32 share the payload).
OTA_CHUNK_BYTES = MAX_PAYLOAD - 5
IMAGE_CHUNK_BYTES = MAX_PAYLOAD - 5

# MSG_VOICE_EVENT kinds (kVoiceEvent* in protocol.h), followed by UTF-8 text.
(
    VOICE_EVENT_STATUS,
    VOICE_EVENT_ERROR,
    VOICE_EVENT_SPEECH_STARTED,
    VOICE_EVENT_SPEECH_STOPPED,
    VOICE_EVENT_REPLY_TEXT,
    VOICE_EVENT_REPLY_DONE,
    VOICE_EVENT_USER_TEXT,
    VOICE_EVENT_PLAYING,
    VOICE_EVENT_LATENCY,
    VOICE_EVENT_INTERRUPTED,
    VOICE_EVENT_NEED_CONFIG,
    VOICE_EVENT_DEBUG,
) = range(1, 13)
# MSG_VOICE_CONFIG payload: offset u16, total u16, then that slice of the config.
VOICE_CONFIG_SLICE = MAX_PAYLOAD - 4

# kPcmSampleRate / kMicSampleRate in protocol.h; both s16le mono.
SPEAKER_SAMPLE_RATE = 24000
# kSpeechGain / kSpeechCeiling in xiao_face.ino: every MSG_PCM stream is lifted by the gain and
# peak-limited at the ceiling (full scale = 1), which suits speech but squashes mastered music.
SPEAKER_PCM_GAIN = 2.8
SPEAKER_PCM_CEILING = 0.97
MIC_SAMPLE_RATE = 16000
MAX_INCOMING_PAYLOAD = 4096
WRITE_TIMEOUT_S = 1.0

STATUS_FRAMEBUFFER = 1 << 0
STATUS_PSRAM = 1 << 1
STATUS_CAMERA_ON = 1 << 2
STATUS_CAMERA_FAILED = 1 << 3
STATUS_WIFI = 1 << 4
STATUS_OTA = 1 << 5
STATUS_MIC_ON = 1 << 6
STATUS_MIC_FAILED = 1 << 7

# Wi-Fi state byte of MSG_STATUS (kWifi* in protocol.h)
WIFI_NO_CONFIG, WIFI_SCANNING, WIFI_NOT_FOUND, WIFI_JOINING, WIFI_JOIN_FAILED, WIFI_CONNECTED = range(6)
# ESP-IDF wifi_err_reason_t values worth naming
WIFI_REASONS = {
    2: "authentication expired",
    15: "4-way handshake timeout (wrong password?)",
    200: "beacon timeout (signal lost)",
    201: "network not found",
    202: "authentication failed (wrong password?)",
    203: "association failed",
    204: "handshake timeout (wrong password?)",
    205: "connection failed",
    210: "no AP with compatible security",
    211: "AP security below the accepted threshold",
}


def describe_wifi_diag(diag: tuple[int, int, int, int]) -> str | None:
    """One line on why the ESP32 is not on Wi-Fi, or None when there is nothing to report."""
    state, seen, rssi, reason = diag
    if state == WIFI_NOT_FOUND:
        if seen == 0:
            return "ESP32 Wi-Fi scan found no networks at all: is the antenna plugged into the head board's U.FL socket?"
        return f"ESP32 Wi-Fi scan saw {seen} networks but none of the Pi's saved ones (they must be 2.4 GHz)"
    if state == WIFI_JOIN_FAILED:
        why = WIFI_REASONS.get(reason, f"reason {reason}") if reason else "timed out"
        weak = " - weak signal, check the antenna" if rssi and rssi < -80 else ""
        return f"ESP32 could not join Wi-Fi ({rssi} dBm): {why}{weak}"
    return None

# Written by systemd/xiaoxiaobai-wifi-export.py (root) from the Pi's NetworkManager profiles.
WIFI_EXPORT_PATH = Path("/run/xiaoxiaobai/wifi.json")
MAX_WIFI_NETWORKS = 8
ESP32_HOSTNAME = "xiaoxiaobai-face"
ESP32_INFO_PATH = Path("/tmp/xiaoxiaobai_esp32.json")

# Size code sent to the firmware (cameraFrameSize) -> pixels.
CAMERA_SIZES = {"qqvga": (0, 160, 120), "qvga": (1, 320, 240), "hvga": (2, 480, 320), "vga": (3, 640, 480)}
CAMERA_CHUNK_HEADER = struct.Struct("<HII")
MAX_JPEG_BYTES = 256 * 1024

BRIGHTNESS_PATH = Path("/tmp/xiaoxiaobai_brightness.json")
DEFAULT_BRIGHTNESS = 100
MIN_BRIGHTNESS = 10

SOUND_REQUEST_PATH = Path("/tmp/xiaoxiaobai_sound.json")
SOUND_REQUEST_MAX_AGE_S = 5.0
SPEAKER_TEST_VOLUME = 30
# (frequency Hz, duration ms); 0 Hz is a rest. C5-E5-G5-C6 arpeggio.
SPEAKER_TEST_NOTES: tuple[tuple[int, int], ...] = ((523, 160), (659, 160), (784, 160), (1047, 320))
MAX_NOTES = 32

Notes = list[tuple[int, int]] | tuple[tuple[int, int], ...]


def encode_packet(kind: int, payload: bytes) -> bytes:
    return MAGIC + struct.pack("<BH", kind, len(payload)) + payload + bytes([reduce(xor, payload, 0)])


def zero_rle(data: bytes) -> bytes:
    """0x00 n -> n zero bytes (n = 1..255); any other byte is a literal."""
    out = bytearray()
    i = 0
    while i < len(data):
        if data[i]:
            out.append(data[i])
            i += 1
            continue
        j = i
        while j < len(data) and data[j] == 0 and j - i < 255:
            j += 1
        out += bytes((0, j - i))
        i = j
    return bytes(out)


def pixel_rle(pixels: bytes) -> bytes:
    """2-byte pixels packed for MSG_IMAGE (protocol::unpackPixelRle): control byte c, then
    c & 0x80 ? one pixel repeated (c & 0x7F) + 1 times : c + 1 literal pixels."""
    px = np.frombuffer(pixels, dtype=np.uint16)
    starts = np.flatnonzero(np.r_[True, px[1:] != px[:-1]])
    lengths = np.diff(np.r_[starts, len(px)])
    out = bytearray()
    literal = None

    def flush(end: int) -> None:
        for at in range(literal, end, 128):
            count = min(128, end - at)
            out.append(count - 1)
            out.extend(pixels[at * 2 : (at + count) * 2])

    for start, length in zip(starts.tolist(), lengths.tolist()):
        if length == 1:
            if literal is None:
                literal = start
            continue
        if literal is not None:
            flush(start)
            literal = None
        for at in range(start, start + length, 128):
            out.append(0x80 | (min(128, start + length - at) - 1))
            out.extend(pixels[at * 2 : at * 2 + 2])
    if literal is not None:
        flush(len(px))
    return bytes(out)


def image_packets(native: bytes, seconds: float) -> tuple[int, list[bytes]]:
    """MSG_IMAGE payloads putting up `native` (the panel's big-endian RGB565, as the face renderer
    lays it out) for `seconds`, and the crc32 the ESP32's MSG_IMAGE_STATUS answers with."""
    packed = pixel_rle(native)
    crc = zlib.crc32(packed)
    packets = [struct.pack("<BIII", IMAGE_BEGIN, len(packed), crc, max(1, round(seconds * 1000)))]
    for offset in range(0, len(packed), IMAGE_CHUNK_BYTES):
        packets.append(struct.pack("<BI", IMAGE_DATA, offset) + packed[offset : offset + IMAGE_CHUNK_BYTES])
    packets.append(bytes([IMAGE_END]))
    return crc, packets


class StreamParser:
    """Splits the ESP32 -> Pi byte stream into (type, payload) packets, skipping corrupt ones."""

    def __init__(self, max_payload: int = MAX_INCOMING_PAYLOAD) -> None:
        self._buf = bytearray()
        self._max_payload = max_payload
        self.bad_packets = 0

    def feed(self, data: bytes) -> list[tuple[int, bytes]]:
        buf = self._buf
        buf += data
        packets: list[tuple[int, bytes]] = []
        pos = 0
        while True:
            start = buf.find(MAGIC, pos)
            if start < 0:
                pos = len(buf) - 1 if buf.endswith(MAGIC[:1]) else len(buf)
                break
            if len(buf) - start < 5:
                pos = start
                break
            kind, length = struct.unpack_from("<BH", buf, start + 2)
            if length > self._max_payload:
                self.bad_packets += 1
                pos = start + 1
                continue
            end = start + 5 + length + 1
            if len(buf) < end:
                pos = start
                break
            payload = bytes(buf[start + 5 : end - 1])
            if int(np.bitwise_xor.reduce(np.frombuffer(payload, dtype=np.uint8))) != buf[end - 1]:
                self.bad_packets += 1
                pos = start + 1
                continue
            packets.append((kind, payload))
            pos = end
        del buf[:pos]
        return packets


class JpegAssembler:
    """Rebuilds JPEG frames from MSG_CAMERA_CHUNK payloads; a gap drops the whole frame."""

    def __init__(self) -> None:
        self._seq: int | None = None
        self._total = 0
        self._buf = bytearray()
        self.dropped = 0

    def feed(self, payload: bytes) -> bytes | None:
        if len(payload) < CAMERA_CHUNK_HEADER.size:
            return None
        seq, offset, total = CAMERA_CHUNK_HEADER.unpack_from(payload)
        data = payload[CAMERA_CHUNK_HEADER.size :]
        if offset == 0:
            if self._seq is not None:
                self.dropped += 1
            if total > MAX_JPEG_BYTES:
                self._seq = None
                self.dropped += 1
                return None
            self._seq, self._total, self._buf = seq, total, bytearray(data)
        elif self._seq != seq or offset != len(self._buf):
            if self._seq is not None:
                self.dropped += 1
            self._seq = None
            return None
        else:
            self._buf += data
        if len(self._buf) < self._total:
            return None
        self._seq = None
        if len(self._buf) != self._total:
            self.dropped += 1
            return None
        return bytes(self._buf)


class _Esp32Protocol:
    description = "ESP32"
    supports_camera = False
    # Streamed speaker PCM and the microphone (they need the UART / USB bandwidth).
    supports_audio = False
    # Whether the firmware sends its 1 s MSG_STATUS heartbeat on this transport.
    reports_status = False
    # FEATURE_* byte of the last MSG_STATUS; None until one arrives (or on links without it).
    features: int | None = None
    # (shown, crc32) of the last MSG_IMAGE_STATUS.
    image_status: tuple[bool, int] | None = None
    # Face frames per second the transport keeps up with.
    face_fps = 25.0
    _face_sent: list[bytes] | None = None
    _face_full_s = 0.0

    def show_face(self, mask: bytes) -> None:
        """1-bit 64x48 mask (firmware without FEATURE_FACE_LEVELS)."""
        packed = zero_rle(mask)
        if len(packed) < len(mask):
            self._send(MSG_FACE_RLE, packed, reliable=False)
        else:
            self._send(MSG_FACE, mask, reliable=False)

    @property
    def face_levels(self) -> bool:
        """Whether the firmware takes MSG_FACE_LEVELS (assumed until a status says otherwise)."""
        return self.features is None or bool(self.features & FEATURE_FACE_LEVELS)

    def show_levels(self, frame: bytes) -> None:
        """One 64x48 frame of LED levels (face_engine.Canvas.to_bytes); only changed bands go out."""
        now = time.monotonic()
        sent = self._face_sent
        full = sent is None or now - self._face_full_s >= FACE_FULL_RESEND_S
        bands = [frame[i * FACE_BAND_BYTES : (i + 1) * FACE_BAND_BYTES] for i in range(FACE_BANDS)]
        changed = [i for i, band in enumerate(bands) if full or band != sent[i]]
        if not changed:
            return
        if full:
            self._face_full_s = now
        self._face_sent = bands
        for i in changed:
            head = i | (FACE_COMMIT if i == changed[-1] else 0)
            self._send(MSG_FACE_LEVELS, bytes([head]) + zero_rle(bands[i]), reliable=False)

    @property
    def supports_image(self) -> bool:
        """Whether the firmware takes MSG_IMAGE (only known once a status arrives)."""
        return self.features is not None and bool(self.features & FEATURE_IMAGE)

    def image(self, payload: bytes) -> None:
        """One MSG_IMAGE packet (image_packets, or IMAGE_HIDE)."""
        self._send(MSG_IMAGE, payload, reliable=True)

    def backlight(self, enabled: bool) -> None:
        self._send(MSG_BACKLIGHT, bytes([1 if enabled else 0]), reliable=True)

    def brightness(self, percent: int) -> None:
        """Backlight brightness (firmware with FEATURE_BRIGHTNESS; older firmware ignores it)."""
        self._send(MSG_BRIGHTNESS, bytes([clamp_brightness(percent)]), reliable=True)

    def play(self, notes: Notes, volume: int) -> None:
        payload = bytearray([max(0, min(100, int(volume)))])
        for freq_hz, duration_ms in list(notes)[:MAX_NOTES]:
            payload += struct.pack("<HH", max(0, min(20000, int(freq_hz))), max(0, min(10000, int(duration_ms))))
        self._send(MSG_TONES, bytes(payload), reliable=True)

    def camera(self, enabled: bool, size_code: int, quality: int, interval_ms: int = 0) -> None:
        """Start/refresh (resend at least every second) or stop the camera stream.

        `interval_ms` > 0 caps the frame rate (10 ms steps, up to 2.55 s between frames).
        """
        payload = bytes([1 if enabled else 0, size_code, max(6, min(63, quality))])
        if interval_ms > 0:
            payload += bytes([max(1, min(255, round(interval_ms / 10)))])
        self._send(MSG_CAMERA, payload, reliable=True)

    def pcm(self, samples: bytes) -> None:
        """Queue s16le mono SPEAKER_SAMPLE_RATE samples (at most MAX_PAYLOAD bytes) for the speaker."""
        self._send(MSG_PCM, samples, reliable=False)

    def pcm_flush(self) -> None:
        """Drop the speaker samples queued on the ESP32."""
        self._send(MSG_PCM_FLUSH, b"", reliable=True)

    def mic(self, enabled: bool) -> None:
        """Start/refresh (resend at least every second) or stop the microphone stream."""
        self._send(MSG_MIC, bytes([1 if enabled else 0]), reliable=True)

    def voice_config(self, payload: bytes) -> None:
        """Forward one MSG_VOICE_CONFIG slice (the ESP32 starts its own Qwen session once complete)."""
        self._send(MSG_VOICE_CONFIG, payload, reliable=True)

    def voice_context(self, payload: bytes) -> None:
        """Forward one MSG_VOICE_CONTEXT slice."""
        self._send(MSG_VOICE_CONTEXT, payload, reliable=True)

    def voice_keepalive(self) -> None:
        """Keep the direct voice session running (resend at least every 5 s)."""
        self._send(MSG_VOICE_KEEPALIVE, b"", reliable=True)

    def voice_stop(self) -> None:
        self._send(MSG_VOICE_STOP, b"", reliable=True)

    def ota(self, payload: bytes) -> None:
        """One UART firmware-update packet (begin / data / end / abort)."""
        self._send(MSG_OTA, payload, reliable=True)

    def wifi(self, payload: bytes) -> None:
        """Hand over the networks to join (encode_wifi_networks); the ESP32 stores them in flash."""
        self._send(MSG_WIFI, payload, reliable=True)

    def close(self) -> None:
        raise NotImplementedError

    def _send(self, kind: int, payload: bytes, reliable: bool) -> None:
        raise NotImplementedError


# ---------------------------------------------------------------- UART / USB serial

FrameCallback = Callable[[bytes], None]
PcmCallback = Callable[[bytes], None]


def _termios_speed(baud: int) -> int:
    speed = getattr(termios, f"B{baud}", None)
    if speed is None:
        raise ValueError(f"unsupported baud rate {baud}")
    return speed


class _Esp32SerialLink(_Esp32Protocol):
    """Raw tty to the ESP32 plus a reader thread for its status, camera and microphone packets."""

    supports_camera = True
    supports_audio = True

    def __init__(self, path: str, baud: int, description: str, on_frame: FrameCallback | None) -> None:
        self.description = description
        # Receives each microphone chunk (s16le mono MIC_SAMPLE_RATE) on the reader thread.
        self.on_mic: PcmCallback | None = None
        # Receives each MSG_VOICE_EVENT payload (kind byte + text) on the reader thread.
        self.on_voice_event: Callable[[bytes], None] | None = None
        # Receives each MSG_OTA_STATUS payload on the reader thread.
        self.on_ota: Callable[[bytes], None] | None = None
        # Receives the speaker's loudness (dB) and zero-crossing rate (0..1) on the reader thread.
        self.on_speaker: Callable[[float, float], None] | None = None
        self.status: int | None = None
        self.last_status_s: float | None = None
        self.wifi_ip: str | None = None
        self.wifi_rssi: int | None = None
        self.wifi_crc: int | None = None
        # (WIFI_* state, networks seen in the last scan, best saved network RSSI, disconnect reason)
        self.wifi_diag: tuple[int, int, int, int] | None = None
        self.frames = 0
        self._on_frame = on_frame
        self._parser = StreamParser()
        self._jpeg = JpegAssembler()
        self._read_error: OSError | None = None
        self._closing = threading.Event()
        # The display loop and the voice bridge both send; packets must not interleave.
        self._tx_lock = threading.Lock()
        self._fd = os.open(path, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        try:
            tty.setraw(self._fd)
            attrs = termios.tcgetattr(self._fd)
            attrs[2] |= termios.CLOCAL | termios.CREAD
            attrs[2] &= ~termios.CRTSCTS
            attrs[4] = attrs[5] = _termios_speed(baud)
            termios.tcsetattr(self._fd, termios.TCSANOW, attrs)
            termios.tcflush(self._fd, termios.TCIOFLUSH)
            # A second open of a tty still succeeds and steals bytes, unless this is set.
            try:
                fcntl.ioctl(self._fd, termios.TIOCEXCL)
            except OSError:
                pass
        except Exception:
            os.close(self._fd)
            raise
        self._reader = threading.Thread(target=self._read_loop, name="xiaoxiaobai-esp32-rx", daemon=True)
        self._reader.start()

    @property
    def dropped_frames(self) -> int:
        return self._jpeg.dropped

    @property
    def bad_packets(self) -> int:
        return self._parser.bad_packets

    def close(self) -> None:
        self._closing.set()
        self._reader.join(timeout=1.0)
        try:
            termios.tcdrain(self._fd)
        except termios.error:
            pass
        finally:
            os.close(self._fd)

    def _send(self, kind: int, payload: bytes, reliable: bool) -> None:
        if self._read_error is not None:
            raise ConnectionError(f"ESP32 on {self.description}: {self._read_error}")
        view = memoryview(encode_packet(kind, payload))
        with self._tx_lock:
            while view:
                if not select.select([], [self._fd], [], WRITE_TIMEOUT_S)[1]:
                    raise TimeoutError(f"ESP32 on {self.description} stopped accepting data")
                view = view[os.write(self._fd, view) :]

    def _read_loop(self) -> None:
        while not self._closing.is_set():
            try:
                if not select.select([self._fd], [], [], 0.2)[0]:
                    continue
                data = os.read(self._fd, 65536)
            except BlockingIOError:
                continue
            except OSError as exc:
                self._read_error = exc
                return
            if not data:
                self._read_error = ConnectionError("port hung up")
                return
            for kind, payload in self._parser.feed(data):
                self._dispatch(kind, payload)

    def _dispatch(self, kind: int, payload: bytes) -> None:
        if kind == MSG_STATUS and payload:
            self.status = payload[0]
            if len(payload) >= 10:
                ip, rssi, crc = struct.unpack_from("<4sbI", payload, 1)
                online = bool(payload[0] & STATUS_WIFI)
                self.wifi_ip = ".".join(map(str, ip)) if online else None
                self.wifi_rssi = rssi if online else None
                self.wifi_crc = crc
            if len(payload) >= 14:
                self.wifi_diag = struct.unpack_from("<BBbB", payload, 10)
            self.features = payload[14] if len(payload) >= 15 else 0
            self.last_status_s = time.monotonic()
        elif kind == MSG_SPEAKER and len(payload) >= 2:
            on_speaker = self.on_speaker
            if on_speaker is not None:
                on_speaker(payload[0] / SPEAKER_DB_SCALE, payload[1] / 255.0)
        elif kind == MSG_IMAGE_STATUS and len(payload) >= 5:
            self.image_status = (bool(payload[0]), struct.unpack_from("<I", payload, 1)[0])
        elif kind == MSG_CAMERA_CHUNK:
            frame = self._jpeg.feed(payload)
            if frame is not None:
                self.frames += 1
                if self._on_frame is not None:
                    self._on_frame(frame)
        elif kind == MSG_MIC_CHUNK:
            on_mic = self.on_mic
            if on_mic is not None:
                on_mic(payload)
        elif kind == MSG_VOICE_EVENT and payload:
            on_voice_event = self.on_voice_event
            if on_voice_event is not None:
                on_voice_event(payload)
        elif kind == MSG_OTA_STATUS and payload:
            on_ota = self.on_ota
            if on_ota is not None:
                on_ota(payload)


class Esp32UartLink(_Esp32SerialLink):
    """Controller GPIO14 (TXD) -> head board D9, controller GPIO15 (RXD) <- head board D2, plus GND."""

    reports_status = True

    def __init__(self, path: str, baud: int = ESP32_UART_BAUD, on_frame: FrameCallback | None = None) -> None:
        try:
            super().__init__(path, baud, f"UART {path}", on_frame)
        except PermissionError as exc:
            raise PermissionError(f"cannot open {path} ({exc.strerror}); run `sudo usermod -aG dialout $USER`") from exc


def esp32_uart_path() -> str:
    return os.environ.get(ESP32_UART_ENV) or ESP32_UART_DEFAULT


def esp32_uart_baud() -> int:
    return int(os.environ.get(ESP32_UART_BAUD_ENV) or ESP32_UART_BAUD)


def esp32_ports() -> list[str]:
    override = os.environ.get(ESP32_PORT_ENV)
    if override:
        return [override] if Path(override).exists() else []
    if not SERIAL_BY_ID.is_dir():
        return []
    ports: list[str] = []
    for pattern in ESP32_PORT_PATTERNS:
        ports.extend(str(path) for path in sorted(SERIAL_BY_ID.glob(pattern)) if str(path) not in ports)
    return ports


def find_esp32_port() -> str | None:
    ports = esp32_ports()
    return ports[0] if ports else None


def esp32_device_paths() -> set[str]:
    """Resolved tty paths the ESP32 may be on, so motor-port scans can skip them."""
    paths = {str(Path(port).resolve()) for port in esp32_ports()}
    if esp32_backend() == "uart":
        paths.add(str(Path(esp32_uart_path()).resolve()))
    return paths


class Esp32UsbLink(_Esp32SerialLink):
    def __init__(self, port: str, on_frame: FrameCallback | None = None) -> None:
        super().__init__(port, 115200, f"USB {port}", on_frame)


# ---------------------------------------------------------------- Bluetooth LE


class Esp32BleLink(_Esp32Protocol):
    """GATT client (bleak / BlueZ) driven from a private asyncio loop, exposed synchronously."""

    face_fps = 10.0

    def __init__(self, address: str | None = None) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="xiaoxiaobai-esp32-ble", daemon=True)
        self._thread.start()
        self._client = None
        self._char = None
        try:
            self._run(self._connect(address), BLE_SCAN_TIMEOUT_S + BLE_CONNECT_TIMEOUT_S + 2.0)
        except BaseException:
            self._stop_loop()
            raise
        self.description = f"Bluetooth {self._client.address}"

    def close(self) -> None:
        try:
            if self._client is not None:
                self._run(self._client.disconnect(), 3.0)
        except OSError:
            pass
        finally:
            self._stop_loop()

    def _send(self, kind: int, payload: bytes, reliable: bool) -> None:
        if self._client is None or not self._client.is_connected:
            raise ConnectionError("ESP32 Bluetooth link dropped")
        self._run(self._write(encode_packet(kind, payload), reliable), BLE_WRITE_TIMEOUT_S)

    async def _connect(self, address: str | None) -> None:
        from bleak import BleakClient, BleakScanner

        target = address
        if target is None:
            target = await BleakScanner.find_device_by_filter(
                lambda _device, adv: BLE_SERVICE_UUID in adv.service_uuids, timeout=BLE_SCAN_TIMEOUT_S
            )
            if target is None:
                raise ConnectionError("ESP32 face board not found over Bluetooth")
        client = BleakClient(target, timeout=BLE_CONNECT_TIMEOUT_S)
        await client.connect()
        char = client.services.get_characteristic(BLE_RX_UUID)
        if char is None:
            await client.disconnect()
            raise ConnectionError("Bluetooth device has no Xiaoxiaobai face characteristic")
        self._client, self._char = client, char

    async def _write(self, packet: bytes, reliable: bool) -> None:
        if reliable:
            await self._client.write_gatt_char(self._char, packet, response=True)
            return
        chunk = max(20, self._char.max_write_without_response_size)
        for start in range(0, len(packet), chunk):
            await self._client.write_gatt_char(self._char, packet[start : start + chunk], response=False)

    def _run(self, coro, timeout_s: float):
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout_s)
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise TimeoutError("ESP32 Bluetooth operation timed out") from None
        except OSError:
            raise
        except Exception as exc:  # noqa: BLE001 - bleak/D-Bus errors become link errors
            raise ConnectionError(f"ESP32 Bluetooth error: {exc}") from exc

    def _stop_loop(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=2.0)
        if not self._thread.is_alive():
            self._loop.close()


def esp32_backend() -> str:
    return os.environ.get(ESP32_BACKEND_ENV, "uart").strip().lower()


def open_esp32_link(backend: str | None = None, on_frame: FrameCallback | None = None) -> _Esp32Protocol:
    """Connect to the face board; raises OSError when it cannot be reached.

    `on_frame` receives each camera JPEG (UART / USB only), on the link's reader thread.
    """
    backend = backend or esp32_backend()
    if backend == "uart":
        return Esp32UartLink(esp32_uart_path(), esp32_uart_baud(), on_frame)
    if backend == "usb":
        port = find_esp32_port()
        if port is None:
            raise FileNotFoundError("ESP32 face board not found on USB")
        return Esp32UsbLink(port, on_frame)
    if backend == "ble":
        return Esp32BleLink(os.environ.get(ESP32_BLE_ADDRESS_ENV) or None)
    raise ValueError(f"{ESP32_BACKEND_ENV} must be uart, ble, usb or spi (got {backend!r})")


# ---------------------------------------------------------------- Wi-Fi hand-over


def encode_wifi_networks(networks: list[tuple[str, str]]) -> bytes:
    """MSG_WIFI payload: count, then (ssid_len, ssid, psk_len, psk); skips what the ESP32 cannot join."""
    payload = bytearray([0])
    for ssid, psk in networks:
        ssid_bytes, psk_bytes = ssid.encode(), psk.encode()
        if not 1 <= len(ssid_bytes) <= 32 or len(psk_bytes) > 64 or payload[0] == MAX_WIFI_NETWORKS:
            continue
        payload += bytes([len(ssid_bytes)]) + ssid_bytes + bytes([len(psk_bytes)]) + psk_bytes
        payload[0] += 1
    return bytes(payload)


def load_wifi_payload(path: Path = WIFI_EXPORT_PATH) -> bytes | None:
    """The Pi's networks as a MSG_WIFI payload, or None when the export is missing or unreadable."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return encode_wifi_networks([(str(n["ssid"]), str(n.get("psk", ""))) for n in data["networks"]])
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return None


def write_esp32_info(ip: str | None, rssi: int | None, path: Path = ESP32_INFO_PATH) -> None:
    """Where the ESP32 is on Wi-Fi, for `make esp32-ip` when mDNS does not resolve."""
    payload = {"ts": time.time(), "hostname": f"{ESP32_HOSTNAME}.local", "ip": ip, "rssi": rssi}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------- screen brightness


def clamp_brightness(percent: object) -> int:
    try:
        value = int(round(float(percent)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return DEFAULT_BRIGHTNESS
    return max(MIN_BRIGHTNESS, min(100, value))


def read_brightness(path: Path = BRIGHTNESS_PATH) -> int:
    """The backlight brightness the API asked for (the display loop polls this file)."""
    try:
        return clamp_brightness(json.loads(path.read_text(encoding="utf-8"))["percent"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return DEFAULT_BRIGHTNESS


def write_brightness(percent: int, path: Path = BRIGHTNESS_PATH) -> int:
    value = clamp_brightness(percent)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"percent": value}) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return value


# ---------------------------------------------------------------- sound requests


def request_sound(notes: Notes = SPEAKER_TEST_NOTES, volume: int = SPEAKER_TEST_VOLUME, path: Path = SOUND_REQUEST_PATH) -> None:
    """Queue a sound for whichever process owns the ESP32 link (the display loop polls this file)."""
    payload = {"ts": time.time(), "volume": int(volume), "notes": [[int(f), int(ms)] for f, ms in notes]}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def take_sound_request(path: Path = SOUND_REQUEST_PATH) -> tuple[list[tuple[int, int]], int] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        path.unlink()
    except (OSError, json.JSONDecodeError):
        return None
    try:
        if time.time() - float(data["ts"]) > SOUND_REQUEST_MAX_AGE_S:
            return None
        notes = [(int(f), int(ms)) for f, ms in data["notes"]]
        return notes, int(data.get("volume", SPEAKER_TEST_VOLUME))
    except (KeyError, TypeError, ValueError):
        return None
