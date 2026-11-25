# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

"""A local 24 kHz mono 16-bit wav (the dance soundtrack) played from the API process itself.

Same pacing, packets and events as src/music_player.py, but no new Python process and no ffmpeg:
on the controller those take seconds to start and ~45 MB the control loop needs (it swapped and
stalled when the dance music started). Looks like the subprocess.Popen MusicService drives:
stdin takes the player's control lines, stdout yields its JSON event lines.
"""

from __future__ import annotations

import json
import queue
import socket
import subprocess
import threading
import time
import wave
from pathlib import Path


class _Commands:
    def __init__(self, player: LocalPlayer) -> None:
        self._player = player

    def write(self, text: str) -> None:
        for line in text.splitlines():
            self._player.command(line.strip())

    def flush(self) -> None:
        pass


class LocalPlayer:
    def __init__(self, path: Path, volume: int) -> None:
        self.stdin = _Commands(self)
        self._lines: queue.Queue[str | None] = queue.Queue()
        self.stdout = iter(self._lines.get, None)
        self.returncode: int | None = None
        self._volume = max(0, min(100, volume))
        self._paused = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(path,), name="xiaoxiaobai-local-music", daemon=True)
        self._thread.start()

    # ---- the Popen side ---------------------------------------------------------------------

    def command(self, line: str) -> None:
        cmd, _, arg = line.partition(" ")
        if cmd == "pause":
            self._paused = True
        elif cmd == "resume":
            self._paused = False
        elif cmd == "volume":
            try:
                self._volume = max(0, min(100, int(arg)))
            except ValueError:
                pass
        elif cmd == "stop":
            self._stop.set()

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise subprocess.TimeoutExpired("local player", timeout or 0.0)
        return self.returncode or 0

    def kill(self) -> None:
        self._stop.set()

    # ---- the player thread ------------------------------------------------------------------

    def _emit(self, event: str, **fields: object) -> None:
        self._lines.put(json.dumps({"event": event, **fields}, ensure_ascii=False))

    def _run(self, path: Path) -> None:
        code = 1
        try:
            code = self._play(path)
        except Exception as exc:  # noqa: BLE001 - reported to the service
            self._emit("error", message=f"配乐播放失败：{exc}")
        finally:
            self.returncode = code
            self._lines.put(None)

    def _play(self, path: Path) -> int:
        import numpy as np

        from esp32_link import MSG_PCM, MSG_PCM_FLUSH, SPEAKER_SAMPLE_RATE, encode_packet
        from music_player import BYTES_PER_S, DEFAULT_LEAD_S, OUTPUT_LATENCY_S, PACKET_BYTES, UNITY
        from voice_bridge import voice_socket_path

        with wave.open(str(path), "rb") as wav:
            if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, SPEAKER_SAMPLE_RATE):
                self._emit("error", message=f"解码失败：配乐要是 {SPEAKER_SAMPLE_RATE} Hz 单声道 16 位 wav")
                return 1
            pcm = wav.readframes(wav.getnframes())
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.connect(str(voice_socket_path()))
        except OSError as exc:
            sock.close()
            self._emit("error", message=f"连不上语音桥（{exc.strerror}）：显示服务没在运行？")
            return 1

        def send(kind: int, payload: bytes = b"") -> None:
            try:
                sock.sendall(encode_packet(kind, payload))
            except OSError:
                raise RuntimeError("扬声器被别的程序接管了（语音对话？）") from None

        def heard() -> int:
            return max(0, sent - int(max(0.0, play_end - time.monotonic()) * BYTES_PER_S)) & ~1

        sent, play_end, started, paused, next_progress = 0, 0.0, False, False, 0.0
        try:
            while not self._stop.is_set():
                now = time.monotonic()
                if self._paused != paused:
                    if self._paused:
                        # What was queued but not heard yet is sent again on resume.
                        sent = heard()
                        send(MSG_PCM_FLUSH)
                        play_end = 0.0
                        self._emit("paused", pos=round(sent / BYTES_PER_S, 1))
                    else:
                        self._emit("resumed", pos=round(sent / BYTES_PER_S, 1))
                    paused = self._paused
                if paused:
                    time.sleep(0.05)
                    continue
                if started and now >= next_progress:
                    self._emit("progress", pos=round(heard() / BYTES_PER_S, 1))
                    next_progress = now + 1.0
                play_end = max(play_end, now)
                ahead = play_end - now
                if ahead > DEFAULT_LEAD_S:
                    time.sleep(min(0.05, ahead - DEFAULT_LEAD_S))
                    continue
                if sent >= len(pcm):
                    break
                raw = pcm[sent:sent + PACKET_BYTES]
                scaled = np.frombuffer(raw, dtype="<i2").astype(np.float32) * (self._volume / 100.0 * UNITY)
                send(MSG_PCM, np.rint(scaled).astype("<i2").tobytes())
                sent += len(raw)
                play_end += len(raw) / BYTES_PER_S
                if not started:
                    started = True
                    self._emit("started", heard_at=round(time.time() + OUTPUT_LATENCY_S, 3))
            if self._stop.is_set():
                send(MSG_PCM_FLUSH)
                self._emit("stopped", pos=round(heard() / BYTES_PER_S, 1))
                return 0
            while not self._stop.is_set() and time.monotonic() < play_end:
                time.sleep(0.05)
            self._emit("done", pos=round(sent / BYTES_PER_S, 1))
            return 0
        except RuntimeError as exc:
            self._emit("error", message=str(exc))
            return 1
        finally:
            sock.close()
