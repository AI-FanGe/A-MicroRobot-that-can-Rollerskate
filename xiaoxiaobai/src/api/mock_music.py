# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

"""Stand-in for music_player.py in mock mode: same stdin commands and stdout events, no audio.
A song "plays" for at most MOCK_SECONDS so that queues move on quickly. Its beats come as from
the online tracker for MOCK_ANALYSIS_S, then from a song map of MOCK_SECTIONS."""

import argparse
import json
import math
import sys
import threading
import time

MOCK_SECONDS = 20.0
MOCK_BPM = 120.0
MOCK_ANALYSIS_S = 3.0
MOCK_SECTIONS = [  # (beats, label, energy, busy, kind)
    (8, "A", 0.3, 0.2, "intro"), (16, "B", 0.7, 0.4, "main"), (16, "C", 1.0, 0.5, "main"),
    (16, "B", 0.7, 0.4, "main"), (16, "C", 1.0, 0.5, "main"), (8, "D", 0.3, 0.1, "outro"),
]


def mock_sections() -> list[dict]:
    out, start, run, label = [], 0, 0, None
    for beats, name, energy, busy, kind in MOCK_SECTIONS:
        run = run + 1 if name == label else 0
        label = name
        out.append({"start": start, "end": start + beats, "label": name, "variant": run, "energy": energy,
                    "busy": busy, "kind": kind})
        start += beats
    return out


def emit(event: str, **fields) -> None:
    print(json.dumps({"event": event, **fields}, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--volume", type=int, default=60)
    parser.add_argument("--sink", default="null")
    parser.add_argument("--seconds", type=float, default=MOCK_SECONDS)
    parser.add_argument("--no-analysis", action="store_true")
    args = parser.parse_args()
    state = {"paused": False, "stop": False}

    def commands() -> None:
        for line in sys.stdin:
            cmd = line.strip().split(" ")[0]
            if cmd == "pause":
                state["paused"] = True
            elif cmd == "resume":
                state["paused"] = False
            elif cmd == "stop":
                break
        state["stop"] = True

    threading.Thread(target=commands, daemon=True).start()
    time.sleep(0.3)
    emit("started", heard_at=round(time.time(), 3))
    first_beat = time.time()
    pos, was_paused, last = 0.0, False, time.monotonic()
    next_beat_event = 0.0
    while pos < min(args.seconds, MOCK_SECONDS):
        if state["stop"]:
            emit("stopped", pos=round(pos, 1))
            return
        if state["paused"] != was_paused:
            was_paused = state["paused"]
            emit("paused" if was_paused else "resumed", pos=round(pos, 1))
        time.sleep(0.25)
        now = time.monotonic()
        if not state["paused"]:
            pos += now - last
        last = now
        emit("progress", pos=round(pos, 1))
        if not state["paused"] and pos >= next_beat_event:
            next_beat_event = pos + 1.0
            period = 60.0 / MOCK_BPM
            if pos < MOCK_ANALYSIS_S:
                emit("beat", t0=round(first_beat, 3), period=period, bpm=MOCK_BPM, confidence=0.9, energy=0.8,
                     downbeat=round(first_beat, 3), bar_confidence=0.8, bar_beats=4, busy=0.4, analysing=True)
                continue
            sections = mock_sections()
            index = math.ceil((time.time() - first_beat) / period)
            current = next((i for i, s in enumerate(sections) if s["start"] <= index < s["end"]), None)
            section = sections[current] if current is not None else None
            following = sections[current + 1] if current is not None and current + 1 < len(sections) else None
            emit("beat", t0=round(first_beat + index * period, 3), period=period, bpm=MOCK_BPM, confidence=1.0,
                 energy=section["energy"] if section else 0.0, busy=section["busy"] if section else 0.0, bar_beats=4,
                 index=index, end=sections[-1]["end"], section=section, next=following, song="mock")
    emit("done", pos=round(pos, 1))


if __name__ == "__main__":
    main()
