# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

"""Plays one song on the robot's speaker: download, decode with ffmpeg, pace to the ESP32.

    python3 src/music_player.py --url https://... [--volume 60] [--sink bridge|null] [--no-analysis]

The song is downloaded in the background (a pause cannot time the HTTP stream out), decoded
by ffmpeg to 24 kHz mono s16le, and sent through the voice bridge (src/voice_bridge.py) just
ahead of playback like voice replies, so pausing or stopping is near-instant. The ESP32
reports the speaker loudness back, so the face moves its mouth to the music.

About once a second a "beat" event gives the beat grid in wall-clock time, for the upper body
to dance to. Once the song is downloaded, src/song_analysis.py maps the whole of it (niced, a
few seconds; kept in SONG_CACHE_ENV's directory, so a song is mapped once) and the grid comes
from the map, with where the song is in it: which dance beat, which section, what comes next.
Until then the decoded audio's online beat tracker (src/beat_tracker.py) gives it.

Control lines on stdin: pause, resume, volume <0-100>, stop.
JSON lines on stdout: {"event": "started"|"progress"|"beat"|"analysed"|"paused"|"resumed"|"stopped"|"done"|"error", ...};
started: {"heard_at": time.time() the first sample is heard};
beat: {"t0": time.time() of a beat as heard, "period" [s], "bpm" (the song's tempo), "confidence", "energy",
"busy", "bar_beats" (periods per bar), and either "downbeat": time.time() of a bar's first beat,
"bar_confidence", "analysing" (the map is on its way) from the tracker, or from the map "index" (t0's
dance beat), "end" (the dance beats in the song), "section" and "next" ({"start", "end" (dance beats),
"label", "variant" (how many of the same label came right before), "energy", "busy", "kind"}; null
past the end) and "song" (the map's key).
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from collections import deque
from pathlib import Path

import numpy as np

from beat_tracker import BeatTracker
from esp32_link import (
    MAX_PAYLOAD,
    MSG_PCM,
    MSG_PCM_FLUSH,
    SPEAKER_PCM_CEILING,
    SPEAKER_PCM_GAIN,
    SPEAKER_SAMPLE_RATE,
    encode_packet,
)
from voice_bridge import voice_socket_path

FFMPEG_ENV = "XIAOXIAOBAI_FFMPEG"
LEAD_ENV = "XIAOXIAOBAI_MUSIC_LEAD_S"
# Audio queued on the ESP32 ahead of playback: rides out a busy Pi, and is what a pause cuts.
DEFAULT_LEAD_S = 0.4
PACKET_BYTES = MAX_PAYLOAD - MAX_PAYLOAD % 2
READ_BYTES = PACKET_BYTES * 8
BYTES_PER_S = SPEAKER_SAMPLE_RATE * 2
DOWNLOAD_CHUNK = 64 * 1024
DOWNLOAD_MAX = 60 * 1024 * 1024
# Undoes the ESP32's speech boost: a full-scale song comes out at the limiter's ceiling, so the
# limiter never acts and volume 100 is the song as mastered.
UNITY = SPEAKER_PCM_CEILING / SPEAKER_PCM_GAIN
# From a packet's queued end to the speaker: the ESP32's I2S DMA buffers.
OUTPUT_LATENCY_S = 0.05
SONG_CACHE_ENV = "XIAOXIAOBAI_SONG_CACHE"
SONG_CACHE_KEEP = 300
ANALYSIS_TIMEOUT_S = 240.0
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"


def ffmpeg_path() -> str | None:
    return os.environ.get(FFMPEG_ENV) or shutil.which("ffmpeg")


def emit(event: str, **fields) -> None:
    sys.stdout.write(json.dumps({"event": event, **fields}, ensure_ascii=False) + "\n")
    sys.stdout.flush()


class Download:
    """The whole file, fetched in the background; readers wait for the bytes they need."""

    def __init__(self, url: str) -> None:
        self.data = bytearray()
        self.done = False
        self.error: str | None = None
        self._cond = threading.Condition()
        threading.Thread(target=self._run, args=(url,), daemon=True).start()

    def _run(self, url: str) -> None:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Referer": "https://music.163.com/"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                while True:
                    chunk = resp.read(DOWNLOAD_CHUNK)
                    if not chunk:
                        break
                    with self._cond:
                        self.data += chunk
                        self._cond.notify_all()
                    if len(self.data) > DOWNLOAD_MAX:
                        raise OSError("文件太大")
        except Exception as exc:  # noqa: BLE001 - reported to the service
            self.error = f"下载失败：{getattr(exc, 'reason', exc)}"
        with self._cond:
            self.done = True
            self._cond.notify_all()

    def wait(self, stop: threading.Event) -> bool:
        """Until the whole file is in (True) or it failed or stop is set (False)."""
        with self._cond:
            while not self.done and not stop.is_set():
                self._cond.wait(1.0)
        return self.done and self.error is None and bool(self.data)

    def read_from(self, offset: int) -> bytes | None:
        """Bytes from offset on, waiting for more; None at the end."""
        with self._cond:
            while len(self.data) <= offset and not self.done:
                self._cond.wait(1.0)
            if len(self.data) <= offset:
                return None
            return bytes(self.data[offset:offset + DOWNLOAD_CHUNK])


def feed_decoder(download: Download, proc: subprocess.Popen) -> None:
    offset = 0
    try:
        while proc.stdin is not None:
            chunk = download.read_from(offset)
            if chunk is None:
                break
            proc.stdin.write(chunk)
            offset += len(chunk)
    except OSError:
        pass
    finally:
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except OSError:
            pass


class Player:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.gain = max(0, min(100, args.volume)) / 100.0
        self.want_pause = False
        self.stopped = threading.Event()
        self.lead_s = float(os.environ.get(LEAD_ENV) or DEFAULT_LEAD_S)
        self.sock: socket.socket | None = None
        # Bytes of decoded audio sent so far, and when the last of them will have played.
        self.sent = 0
        self.play_end = 0.0
        self.tracker: BeatTracker | None = None if args.no_analysis else BeatTracker()
        self._odd = b""  # a decoded byte waiting for its other half
        self.song_map: dict | None = None
        self.song_key = ""
        self.analysing = False
        self._analysis: subprocess.Popen | None = None

    def track(self, chunk: bytes) -> None:
        if self.tracker is None:
            return
        data, self._odd = self._odd + chunk, b""
        if len(data) % 2:
            data, self._odd = data[:-1], data[-1:]
        try:
            self.tracker.feed(np.frombuffer(data, dtype="<i2"))
        except Exception as exc:  # noqa: BLE001 - the music goes on without the beat
            self.tracker = None
            print(f"beat tracker off: {exc}", file=sys.stderr, flush=True)

    def analyse(self, download: Download) -> None:
        """Maps the whole song once it is downloaded (or finds its map in the cache)."""
        self.analysing = True
        started = time.monotonic()
        try:
            if not download.wait(self.stopped):
                return
            data = bytes(download.data)
            key = hashlib.sha1(data[: 1 << 18]).hexdigest()[:16] + f"{len(data):x}"
            cache = Path(os.environ.get(SONG_CACHE_ENV) or Path.home() / ".cache" / "xiaoxiaobai" / "songs")
            path = cache / f"{key}.json"
            if not path.exists():
                cache.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as fh:
                    fh.write(data)
                del data
                try:
                    # Niced: the control loop shares these four small cores.
                    nice = shutil.which("nice")
                    self._analysis = subprocess.Popen(
                        [*([nice, "-n", "19"] if nice else []), sys.executable,
                         str(Path(__file__).with_name("song_analysis.py")), fh.name, "--out", str(path)],
                        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                    _, err = self._analysis.communicate(timeout=ANALYSIS_TIMEOUT_S)
                    if self._analysis.returncode != 0:
                        if not self.stopped.is_set():
                            print(f"song analysis failed: {err.decode('utf-8', 'replace')[-300:]}", file=sys.stderr, flush=True)
                        return
                finally:
                    os.unlink(fh.name)
                for old in sorted(cache.glob("*.json"), key=lambda p: p.stat().st_mtime)[:-SONG_CACHE_KEEP]:
                    old.unlink(missing_ok=True)
            else:
                path.touch()
            song_map = json.loads(path.read_text(encoding="utf-8"))
            if not song_map or not song_map.get("grid"):
                return
            run, label = 0, None
            for s in song_map["sections"]:
                run = run + 1 if s["label"] == label else 0
                s["variant"], label = run, s["label"]
            self.song_key, self.song_map = key, song_map
            emit("analysed", tempo=song_map["tempo"], sections=[f"{s['label']}:{s['kind']}" for s in song_map["sections"]],
                 seconds=round(time.monotonic() - started, 1))
        except Exception as exc:  # noqa: BLE001 - the dance goes on with the online tracker
            print(f"song analysis failed: {exc}", file=sys.stderr, flush=True)
        finally:
            self.analysing = False

    def emit_beat(self) -> None:
        # Byte offset 2 * sample plays when the bytes before it have: play_end less what is
        # queued after it (play_end is behind now with nothing queued, e.g. right after a resume).
        play_end = max(self.play_end, time.monotonic())
        to_wall = time.time() - time.monotonic() + OUTPUT_LATENCY_S

        def heard_at(sample: int) -> float:
            return round(play_end - (self.sent - 2 * sample) / BYTES_PER_S + to_wall, 3)

        if self.song_map is not None:
            self.emit_planned(self.song_map, heard_at)
            return
        if self.tracker is None:
            return
        try:
            grid = self.tracker.estimate()
        except Exception as exc:  # noqa: BLE001
            self.tracker = None
            print(f"beat tracker off: {exc}", file=sys.stderr, flush=True)
            return
        if grid is None:
            return
        emit("beat", t0=heard_at(grid["beat_sample"]), period=round(60.0 / grid["bpm"], 4),
             bpm=round(grid["tempo"], 1), confidence=round(grid["confidence"], 2), energy=round(grid["energy"], 2),
             downbeat=heard_at(grid["downbeat_sample"]), bar_confidence=round(grid["bar_confidence"], 2),
             bar_beats=grid["bar_beats"], busy=round(grid["busy"], 2), analysing=self.analysing)

    def emit_planned(self, song_map: dict, heard_at) -> None:
        """The beat event from the song's map, at the dance beat being heard next."""
        grid, sections = song_map["grid"], song_map["sections"]
        k = bisect.bisect_left(grid, self.heard() / BYTES_PER_S)
        last = len(grid) - 1
        a, b = max(0, min(k, last) - 2), min(last, k + 2)
        period = (grid[b] - grid[a]) / (b - a) if b > a else 0.5
        t = grid[k] if k <= last else grid[last] + (k - last) * period
        current = next((i for i, s in enumerate(sections) if s["start"] <= k < s["end"]), None)
        section = sections[current] if current is not None else None
        following = sections[current + 1] if current is not None and current + 1 < len(sections) else None
        energy = section["energy"] if section else 0.0
        emit("beat", t0=heard_at(round(t * SPEAKER_SAMPLE_RATE)), period=round(period, 4), bpm=song_map["tempo"],
             confidence=1.0, energy=energy, busy=section["busy"] if section else 0.0, bar_beats=song_map["bar_beats"],
             index=k, end=len(grid), section=section, next=following, song=self.song_key)

    def heard(self) -> int:
        queued = max(0.0, self.play_end - time.monotonic())
        return max(0, self.sent - int(queued * BYTES_PER_S)) & ~1

    def open_sink(self) -> None:
        if self.args.sink == "null":
            return
        path = voice_socket_path()
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self.sock.connect(str(path))
        except OSError as exc:
            raise RuntimeError(f"连不上语音桥 {path}（{exc.strerror}）：显示服务没在运行？") from None

    def send(self, kind: int, payload: bytes = b"") -> None:
        if self.sock is None:
            return
        try:
            self.sock.sendall(encode_packet(kind, payload))
        except OSError:
            raise RuntimeError("扬声器被别的程序接管了（语音对话？）") from None

    def read_commands(self) -> None:
        for line in sys.stdin:
            cmd, _, arg = line.strip().partition(" ")
            if cmd == "pause":
                self.want_pause = True
            elif cmd == "resume":
                self.want_pause = False
            elif cmd == "volume":
                try:
                    self.gain = max(0, min(100, int(arg))) / 100.0
                except ValueError:
                    pass
            elif cmd == "stop":
                break
        self.stopped.set()

    def run(self) -> int:
        ffmpeg = ffmpeg_path()
        if not ffmpeg:
            emit("error", message="机器人上没有 ffmpeg：运行 make music-setup 安装")
            return 1
        try:
            self.open_sink()
        except RuntimeError as exc:
            emit("error", message=str(exc))
            return 1
        download = Download(self.args.url)
        proc = subprocess.Popen(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
             "-f", "s16le", "-ac", "1", "-ar", str(SPEAKER_SAMPLE_RATE), "pipe:1"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        threading.Thread(target=feed_decoder, args=(download, proc), daemon=True).start()
        threading.Thread(target=self.read_commands, daemon=True).start()
        if not self.args.no_analysis:
            threading.Thread(target=self.analyse, args=(download,), daemon=True).start()
        try:
            return self._play(proc, download)
        except RuntimeError as exc:
            emit("error", message=str(exc))
            return 1
        finally:
            self.stopped.set()
            for child in (proc, self._analysis):
                if child is not None and child.poll() is None:
                    try:
                        child.kill()
                    except OSError:
                        pass
            if self.sock is not None:
                self.sock.close()

    def _play(self, proc: subprocess.Popen, download: Download) -> int:
        pending = bytearray()  # decoded, not sent yet
        recent: deque[tuple[int, bytes]] = deque()  # (offset, unscaled packet) possibly still queued on the ESP32
        paused = started = eof = False
        next_progress = 0.0
        while not self.stopped.is_set():
            now = time.monotonic()
            if self.want_pause and not paused:
                heard = self.heard()
                self.send(MSG_PCM_FLUSH)
                # Whatever was queued but not heard yet goes back in front of the queue.
                back = bytearray()
                for offset, raw in recent:
                    if offset + len(raw) > heard:
                        back += raw[max(0, heard - offset):]
                pending[:0] = back
                recent.clear()
                self.sent, self.play_end, paused = heard, 0.0, True
                emit("paused", pos=round(heard / BYTES_PER_S, 1))
            if paused:
                if self.want_pause:
                    time.sleep(0.05)
                    continue
                paused = False
                emit("resumed", pos=round(self.sent / BYTES_PER_S, 1))
            if started and now >= next_progress:
                emit("progress", pos=round(self.heard() / BYTES_PER_S, 1))
                self.emit_beat()
                next_progress = now + 1.0
            self.play_end = max(self.play_end, now)
            ahead = self.play_end - now
            if ahead > self.lead_s:
                time.sleep(min(0.05, ahead - self.lead_s))
                continue
            if len(pending) < PACKET_BYTES and not eof:
                chunk = proc.stdout.read1(READ_BYTES) if proc.stdout else b""
                if chunk:
                    pending += chunk
                    self.track(chunk)
                    continue
                eof = True
            cut = min(len(pending), PACKET_BYTES) & ~1
            if cut <= 0:
                break
            raw = bytes(pending[:cut])
            del pending[:cut]
            scaled = np.frombuffer(raw, dtype="<i2").astype(np.float32) * (self.gain * UNITY)
            self.send(MSG_PCM, np.rint(scaled).astype("<i2").tobytes())
            recent.append((self.sent, raw))
            while recent and recent[0][0] + len(recent[0][1]) < self.sent - int((self.lead_s + 1.0) * BYTES_PER_S):
                recent.popleft()
            self.sent += cut
            self.play_end += cut / BYTES_PER_S
            if not started:
                started = True
                emit("started", heard_at=round(time.time() + OUTPUT_LATENCY_S, 3))
        if self.stopped.is_set():
            self.send(MSG_PCM_FLUSH)
            emit("stopped", pos=round(self.heard() / BYTES_PER_S, 1))
            return 0
        proc.wait(timeout=5)
        if download.error or not started:
            err = proc.stderr.read().decode("utf-8", "replace").strip()[-200:] if proc.stderr else ""
            emit("error", message=download.error or f"解码失败：{err or proc.returncode}")
            return 1
        while not self.stopped.is_set() and time.monotonic() < self.play_end:
            time.sleep(0.05)
        emit("done", pos=round(self.sent / BYTES_PER_S, 1))
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", required=True)
    parser.add_argument("--volume", type=int, default=60)
    parser.add_argument("--sink", choices=("bridge", "null"), default="bridge", help="null: decode and pace without a speaker")
    parser.add_argument("--no-analysis", action="store_true",
                        help="no beat tracking or song map (the dance soundtrack: it already follows the dance)")
    return Player(parser.parse_args()).run()


if __name__ == "__main__":
    sys.exit(main())
