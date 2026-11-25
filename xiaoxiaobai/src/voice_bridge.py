# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

"""Local socket that gives one voice client the ESP32's speaker, microphone and voice session.

The display loop owns the ESP32 link; a voice client (src/voice_chat.py) connects to
VOICE_SOCKET_PATH and exchanges the same packets as the UART (esp32_link.encode_packet).
Two routes:

- Pi route: the client sends MSG_MIC 1 and receives every MSG_MIC_CHUNK, and sends
  MSG_PCM / MSG_PCM_FLUSH, forwarded to the ESP32 as soon as they arrive.
- ESP32 route: the client sends MSG_VOICE_CONFIG slices and the ESP32 talks to Qwen itself;
  the client receives its MSG_VOICE_EVENT reports.

The bridge keeps whatever the client asked for alive and stops it when the client leaves.
"""

from __future__ import annotations

import os
import socket
import threading
from pathlib import Path

from esp32_link import (
    MSG_MIC,
    MSG_MIC_CHUNK,
    MSG_PCM,
    MSG_PCM_FLUSH,
    MSG_VOICE_CONFIG,
    MSG_VOICE_CONTEXT,
    MSG_VOICE_EVENT,
    MSG_VOICE_STOP,
    StreamParser,
    encode_packet,
)

VOICE_SOCKET_ENV = "XIAOXIAOBAI_VOICE_SOCKET"
VOICE_SOCKET_PATH = Path("/tmp/xiaoxiaobai_voice.sock")
KEEPALIVE_S = 1.0
# A client that cannot take 20 ms of microphone audio this fast is dropped.
CLIENT_SEND_TIMEOUT_S = 0.2


def voice_socket_path() -> Path:
    return Path(os.environ.get(VOICE_SOCKET_ENV) or VOICE_SOCKET_PATH)


class VoiceBridge:
    def __init__(self, path: Path | None = None) -> None:
        self._path = path or voice_socket_path()
        self._path.unlink(missing_ok=True)
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(str(self._path))
        # The display service and the voice client may run as different users.
        os.chmod(self._path, 0o666)
        self._server.listen(1)
        self._lock = threading.Lock()
        self._client: socket.socket | None = None
        self._link = None
        # What the current client asked for, and what the ESP32 was last told.
        self._wants_mic = False
        self._wants_voice = False
        self._mic_sent = False
        self._voice_sent = False
        self._next_keepalive = 0.0
        self._closed = threading.Event()
        self.pcm_packets = 0
        self.mic_packets = 0
        self._accept_thread = threading.Thread(target=self._accept_loop, name="xiaoxiaobai-voice-accept", daemon=True)
        self._accept_thread.start()

    @property
    def connected(self) -> bool:
        return self._client is not None

    def attach(self, link) -> None:
        """Route the microphone and voice events of a (re)opened link to the client."""
        with self._lock:
            self._link = link
            self._mic_sent = self._voice_sent = False
        link.on_mic = self._on_mic
        link.on_voice_event = self._on_voice_event

    def detach(self) -> None:
        with self._lock:
            if self._link is not None:
                self._link.on_mic = None
                self._link.on_voice_event = None
            self._link = None

    def tick(self, link, now: float) -> None:
        """Keeps the client's microphone / voice session running; call from the link owner."""
        connected = self.connected
        wants_mic = connected and self._wants_mic
        wants_voice = connected and self._wants_voice
        keepalive = now >= self._next_keepalive
        if keepalive:
            self._next_keepalive = now + KEEPALIVE_S
        if wants_mic and (keepalive or not self._mic_sent):
            link.mic(True)
            self._mic_sent = True
        elif not wants_mic and self._mic_sent:
            link.mic(False)
            link.pcm_flush()
            self._mic_sent = False
        if wants_voice and (keepalive or not self._voice_sent):
            link.voice_keepalive()
            self._voice_sent = True
        elif not wants_voice and self._voice_sent:
            link.voice_stop()
            link.pcm_flush()
            self._voice_sent = False

    def close(self) -> None:
        self._closed.set()
        self.detach()
        try:
            self._server.close()
        finally:
            self._path.unlink(missing_ok=True)
        self._drop_client()

    def _accept_loop(self) -> None:
        while not self._closed.is_set():
            try:
                client, _ = self._server.accept()
            except OSError:
                return
            client.settimeout(CLIENT_SEND_TIMEOUT_S)
            self._drop_client()
            with self._lock:
                self._client = client
            threading.Thread(target=self._client_loop, args=(client,), name="xiaoxiaobai-voice-client", daemon=True).start()

    def _client_loop(self, client: socket.socket) -> None:
        parser = StreamParser()
        while not self._closed.is_set() and self._client is client:
            try:
                data = client.recv(65536)
            except TimeoutError:
                continue
            except OSError:
                break
            if not data:
                break
            for kind, payload in parser.feed(data):
                if kind == MSG_MIC:
                    self._wants_mic = payload[:1] == b"\x01"
                    continue
                if kind == MSG_VOICE_STOP:
                    self._wants_voice = False
                    continue
                link = self._link
                if link is None:
                    continue
                try:
                    if kind == MSG_PCM:
                        link.pcm(payload)
                        self.pcm_packets += 1
                    elif kind == MSG_PCM_FLUSH:
                        link.pcm_flush()
                    elif kind == MSG_VOICE_CONFIG:
                        link.voice_config(payload)
                        self._wants_voice = True
                    elif kind == MSG_VOICE_CONTEXT:
                        link.voice_context(payload)
                except OSError:
                    pass  # the display loop notices the dead link and reopens it
        if self._client is client:
            self._drop_client()

    def _on_mic(self, pcm: bytes) -> None:
        if self._client is not None and self._wants_mic:
            self._to_client(encode_packet(MSG_MIC_CHUNK, pcm))
            self.mic_packets += 1

    def _on_voice_event(self, payload: bytes) -> None:
        if self._client is not None and self._wants_voice:
            self._to_client(encode_packet(MSG_VOICE_EVENT, payload))

    def _to_client(self, packet: bytes) -> None:
        client = self._client
        if client is None:
            return
        try:
            client.sendall(packet)
        except OSError:
            self._drop_client()

    def _drop_client(self) -> None:
        with self._lock:
            client, self._client = self._client, None
            self._wants_mic = self._wants_voice = False
        if client is not None:
            try:
                client.close()
            except OSError:
                pass
