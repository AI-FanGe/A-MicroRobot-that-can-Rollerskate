# SPDX-License-Identifier: GPL-3.0-or-later

"""The whole song's beats, bars and sections, for the upper body to dance to its structure.

    python3 src/song_analysis.py song.mp3           # prints the map as JSON
    song_map = analyse(int16_samples_at_RATE)

The music player runs this (niced, on the downloaded file) while the song starts playing; the
online beat tracker (src/beat_tracker.py) covers the first seconds until the map is ready.

- Beats: the onset strength (spectral flux of log bands, 100 frames/s), a global tempo from its
  autocorrelation (weighted toward PRIOR_BPM), then dynamic programming over the whole song for
  the beat sequence that best fits both (Ellis 2007), so every beat lands on the music.
- Bars: which of every four beats starts a bar, from where the bass hits and the chords change.
- Sections: bars compared with each other by harmony (chroma) and timbre (log bands); the
  novelty along that self-similarity, plus loudness changes, gives boundaries, chosen by dynamic
  programming that also favours sections of 4, 8 and 16 bars. Sections that repeat (a second
  chorus) get the same label. Each has its loudness against the song's loudest section and how
  busy it is between the beats; quiet ones at the start and end are the intro and outro, and
  quiet ones in between breaks.
- The dance grid: the beats folded under DANCE_BPM_MAX (every second beat for a fast song,
  starting on the bars' first beats), what the robot dances to.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
import subprocess
import sys

import numpy as np

RATE = 12000
HOP = 120                  # 10 ms
FPS = RATE / HOP
WIN = 512
BANDS = 24
LOW_HZ = 200.0
CHROMA_WIN = 4096
CHROMA_HOP = 1200          # 10 frames/s
CHROMA_HZ = (65.0, 2100.0)
CHUNK_FRAMES = 512         # frames per FFT batch (the chroma's: an eighth): small in memory
BPM_RANGE = (60.0, 180.0)
PRIOR_BPM = 110.0
PRIOR_OCTAVES = 0.9
TEMPO_WINDOW_S, TEMPO_HOP_S = 8.0, 2.0
TIGHTNESS = 100.0          # how hard the beat sequence holds to the tempo
BAR_BEATS = 4
DANCE_BPM_MAX = 120.0
FOLD_SLACK = 0.05
NOVELTY_BARS = 4           # half the checkerboard kernel
# Section lengths [bars] the segmentation favours, and what any other length costs.
LENGTH_PRIOR = {4: 0.5, 8: 0.8, 16: 0.7, 12: 0.3, 24: 0.4, 32: 0.4, 2: -0.5, 6: -0.2, 10: -0.4, 20: 0.0}
LENGTH_OTHER = -0.8
MIN_SECTION_BARS = 2
MAX_SECTION_BARS = 32
BOUNDARY_WEIGHT = 2.0
SECTION_COST = 1.0
SILENT_DB = -30.0          # under the song's loudest: no music (the start, the end)
VERSION = 1


def decode(path: str, ffmpeg: str | None = None) -> np.ndarray:
    ffmpeg = ffmpeg or os.environ.get("XIAOXIAOBAI_FFMPEG") or shutil.which("ffmpeg") or "ffmpeg"
    out = subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-i", path, "-f", "s16le", "-ac", "1",
                          "-ar", str(RATE), "pipe:1"], capture_output=True, check=True).stdout
    return np.frombuffer(out[: len(out) // 2 * 2], dtype="<i2")


def _frames(x: np.ndarray, win: int, hop: int, start: int, count: int) -> np.ndarray:
    """Frames of the int16 song, as float in [-1, 1) (only this chunk's: the song stays int16)."""
    seg = x[start * hop: (start + count - 1) * hop + win].astype(np.float32) / 32768.0
    if len(seg) < (count - 1) * hop + win:
        seg = np.pad(seg, (0, (count - 1) * hop + win - len(seg)))
    return np.lib.stride_tricks.sliding_window_view(seg, win)[::hop][:count]


def _features(x: np.ndarray) -> dict[str, np.ndarray]:
    """Per 10 ms frame: onset strength, its bass part, RMS and the log band spectrum; per 100 ms
    frame: chroma. Computed in chunks, so a long song stays small in memory."""
    n = max(0, (len(x) - WIN) // HOP + 1)
    window = np.hanning(WIN).astype(np.float32)
    bins = WIN // 2 + 1
    edges = np.unique(np.geomspace(2, bins - 1, BANDS + 1).astype(int))
    bands = np.zeros((bins, len(edges) - 1), dtype=np.float32)
    for i, (lo, hi) in enumerate(itertools.pairwise(edges)):
        bands[lo:hi, i] = 1.0 / (hi - lo)
    low = max(1, int(np.sum(edges[1:] <= LOW_HZ / (RATE / WIN))))
    onset = np.zeros(n, dtype=np.float32)
    bass = np.zeros(n, dtype=np.float32)
    rms = np.zeros(n, dtype=np.float32)
    spec = np.zeros((n, bands.shape[1]), dtype=np.float32)
    prev = None
    for start in range(0, n, CHUNK_FRAMES):
        count = min(CHUNK_FRAMES, n - start)
        fr = _frames(x, WIN, HOP, start, count)
        s = np.log1p(100.0 * (np.abs(np.fft.rfft(fr * window, axis=1)).astype(np.float32) @ bands))
        rise = np.maximum(np.diff(np.vstack([s[:1] if prev is None else prev[None, :], s]), axis=0), 0.0)
        prev = s[-1]
        onset[start:start + count] = rise.mean(axis=1)
        bass[start:start + count] = rise[:, :low].mean(axis=1)
        rms[start:start + count] = np.sqrt((fr ** 2).mean(axis=1))
        spec[start:start + count] = s
    # Chroma
    nc = max(0, (len(x) - CHROMA_WIN) // CHROMA_HOP + 1)
    freqs = np.fft.rfftfreq(CHROMA_WIN, 1.0 / RATE)
    keep = np.nonzero((freqs >= CHROMA_HZ[0]) & (freqs <= CHROMA_HZ[1]))[0]
    pitch = np.rint(12.0 * np.log2(freqs[keep] / 440.0)).astype(int) % 12
    to_chroma = np.zeros((len(keep), 12), dtype=np.float32)
    to_chroma[np.arange(len(keep)), pitch] = 1.0
    cwin = np.hanning(CHROMA_WIN).astype(np.float32)
    chroma = np.zeros((nc, 12), dtype=np.float32)
    for start in range(0, nc, CHUNK_FRAMES // 8):
        count = min(CHUNK_FRAMES // 8, nc - start)
        fr = _frames(x, CHROMA_WIN, CHROMA_HOP, start, count)
        mag = np.abs(np.fft.rfft(fr * cwin, axis=1))[:, keep].astype(np.float32)
        c = np.log1p(10.0 * mag) @ to_chroma
        chroma[start:start + count] = c / np.maximum(c.max(axis=1, keepdims=True), 1e-6)
    return {"onset": onset, "bass": bass, "rms": rms, "spec": spec, "chroma": chroma}


def _local(x: np.ndarray, width: int = 31) -> np.ndarray:
    """x above its local mean: loud passages do not swamp quiet ones."""
    return np.maximum(x - np.convolve(x, np.ones(width) / width, mode="same"), 0.0)


def _tempo(onset: np.ndarray) -> float:
    """Frames per beat: the autocorrelation of the onsets, averaged over windows, times the prior."""
    win, hop = int(TEMPO_WINDOW_S * FPS), int(TEMPO_HOP_S * FPS)
    lo, hi = int(60.0 * FPS / BPM_RANGE[1]), int(np.ceil(60.0 * FPS / BPM_RANGE[0]))
    total = np.zeros(hi + 2)
    for start in range(0, max(1, len(onset) - win + 1), hop):
        x = onset[start:start + win]
        x = x - x.mean()
        if len(x) < 2 * hi:
            continue
        spectrum = np.fft.rfft(x, 2 * len(x))
        ac = np.fft.irfft(spectrum * np.conj(spectrum))[: hi + 2]
        if ac[0] > 1e-9:
            total += ac / ac[0]
    lags = np.arange(lo, hi + 1)
    prior = np.exp(-0.5 * (np.log2(60.0 * FPS / lags / PRIOR_BPM) / PRIOR_OCTAVES) ** 2)
    best = int(lags[np.argmax(np.maximum(total[lags], 0.0) * prior)])
    a, b, c = total[best - 1], total[best], total[best + 1]
    bend = a - 2 * b + c
    return best + (float(np.clip(0.5 * (a - c) / bend, -0.5, 0.5)) if bend < 0 else 0.0)


def _beats(onset: np.ndarray, period: float) -> np.ndarray:
    """Beat frames: the dynamic programming beat tracker, vectorized over half-beat blocks (a
    frame's predecessors are all at least half a beat back, so a block depends only on earlier
    ones)."""
    sd = onset.std()
    if sd <= 1e-9:
        return np.zeros(0, dtype=int)
    o = onset / sd
    k = np.arange(-int(period), int(period) + 1)
    score = np.convolve(o, np.exp(-0.5 * (k * 32.0 / period) ** 2), mode="same")
    n = len(score)
    lags = np.arange(max(1, round(period / 2)), round(2 * period) + 1)
    penalty = -TIGHTNESS * np.log(lags / period) ** 2
    cum = np.zeros(n)
    back = np.full(n, -1, dtype=int)
    block = int(lags[0])
    for start in range(0, n, block):
        t = np.arange(start, min(start + block, n))
        prev = t[:, None] - lags[None, :]
        cand = np.where(prev >= 0, cum[np.maximum(prev, 0)] + penalty[None, :], -np.inf)
        best = np.argmax(cand, axis=1)
        value = cand[np.arange(len(t)), best]
        ok = np.isfinite(value)
        cum[t] = score[t] + np.where(ok, value, 0.0)
        back[t] = np.where(ok, prev[np.arange(len(t)), best], -1)
    # The last beat: the latest local maximum of the cumulative score that is a real one.
    peaks = np.nonzero((cum[1:-1] > cum[:-2]) & (cum[1:-1] >= cum[2:]))[0] + 1
    if not len(peaks):
        return np.zeros(0, dtype=int)
    median = np.median(cum[peaks])
    last = int(peaks[np.nonzero(cum[peaks] >= 0.5 * median)[0][-1]])
    beats = [last]
    while back[beats[-1]] >= 0:
        beats.append(int(back[beats[-1]]))
    beats = np.array(beats[::-1])
    # Beats where the music has not started yet, or has stopped, go.
    strength = score[beats]
    level = 0.5 * np.sqrt(np.mean(np.convolve(strength, np.hanning(5) / 2.5, mode="same") ** 2))
    strong = np.nonzero(np.convolve(strength, np.hanning(5) / 2.5, mode="same") > level)[0]
    return beats[strong[0]: strong[-1] + 1] if len(strong) else beats


def _at(x: np.ndarray, frames: np.ndarray, reach: int = 3) -> np.ndarray:
    """The strongest value of x within reach frames of each frame."""
    idx = np.clip(frames[:, None] + np.arange(-reach, reach + 1)[None, :], 0, len(x) - 1)
    return x[idx].max(axis=1)


def _z(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return (x - x.mean()) / sd if sd > 1e-9 else np.zeros_like(x)


def _span_mean(x: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """The mean of x's rows over each [start, end)."""
    c = np.vstack([np.zeros((1,) + x.shape[1:]), np.cumsum(x, axis=0)])
    starts = np.clip(starts, 0, len(x) - 1)
    ends = np.clip(np.maximum(ends, starts + 1), 1, len(x))
    return (c[ends] - c[starts]) / (ends - starts)[:, None]


def _downbeat(f: dict[str, np.ndarray], beats: np.ndarray) -> tuple[int, float]:
    """(the first beat that starts a bar, how clearly: 0-1): the bass hits and chord changes."""
    if len(beats) < 4 * BAR_BEATS:
        return 0, 0.0
    bass = _at(_local(f["bass"]), beats)
    hit = _at(_local(f["onset"]), beats)
    cb = np.rint(beats / FPS * RATE / CHROMA_HOP).astype(int)
    per_beat = _span_mean(f["chroma"], cb, np.append(cb[1:], cb[-1] + 1))
    change = np.zeros(len(beats))
    for i in range(2, len(beats) - 2):
        a, b = per_beat[i - 2:i].mean(axis=0), per_beat[i:i + 2].mean(axis=0)
        change[i] = 1.0 - float(a @ b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-9)
    s = _z(bass) + _z(change) + 0.5 * _z(hit)
    strength = np.array([s[k::BAR_BEATS].mean() for k in range(BAR_BEATS)])
    order = np.argsort(strength)[::-1]
    return int(order[0]), float(np.clip((strength[order[0]] - strength[order[1]]) / 0.5, 0.0, 1.0))


def _segment(novelty: np.ndarray) -> list[int]:
    """Bar indices where sections start (with the end), by dynamic programming."""
    m = len(novelty) - 1
    best = np.full(m + 1, -np.inf)
    best[0] = 0.0
    prev = np.zeros(m + 1, dtype=int)
    for b in range(1, m + 1):
        for length in range(1, min(MAX_SECTION_BARS, b) + 1):
            a = b - length
            if not np.isfinite(best[a]) or (length < MIN_SECTION_BARS and 0 < a and b < m):
                continue
            # The first and last sections may be any length (a pickup, a fade).
            prior = LENGTH_PRIOR.get(length, LENGTH_OTHER) if 0 < a and b < m else 0.0
            value = best[a] + prior - SECTION_COST + (BOUNDARY_WEIGHT * novelty[b] if b < m else 0.0)
            if value > best[b]:
                best[b], prev[b] = value, a
    cuts = [m]
    while cuts[-1] > 0:
        cuts.append(int(prev[cuts[-1]]))
    return cuts[::-1]


def _sections(f: dict[str, np.ndarray], beats: np.ndarray, first: int) -> tuple[list[dict], dict]:
    """Sections over the bars (starting at beat `first`), and what they were found from (the
    bars' self-similarity, novelty and loudness)."""
    starts = beats[first::BAR_BEATS]
    if len(starts) < 4:
        return [], {}
    ends = np.append(starts[1:], beats[-1] + int(np.median(np.diff(beats))))
    nb = len(starts)
    cs, ce = (np.rint(v / FPS * RATE / CHROMA_HOP).astype(int) for v in (starts, ends))
    chroma = _span_mean(f["chroma"], cs, ce)
    timbre = _span_mean(f["spec"], starts, ends)
    timbre -= timbre.mean(axis=0)
    loud = 20.0 * np.log10(np.maximum(_span_mean(f["rms"][:, None], starts, ends)[:, 0], 1e-6))

    def similarity(v: np.ndarray) -> np.ndarray:
        v = np.hstack([v, np.vstack([v[1:], v[-1:]])])  # each bar with the next: less noisy
        v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
        return v @ v.T

    ssm = 0.5 * similarity(chroma - chroma.mean(axis=0)) + 0.5 * similarity(timbre)
    k = NOVELTY_BARS
    g = np.exp(-0.5 * (np.linspace(-1.5, 1.5, 2 * k)) ** 2)
    kernel = np.outer(g, g) * np.outer(np.r_[-np.ones(k), np.ones(k)], np.r_[-np.ones(k), np.ones(k)])
    padded = np.pad(ssm, k, mode="edge")
    novelty = np.zeros(nb + 1)
    for b in range(1, nb):
        novelty[b] = max(0.0, float((padded[b:b + 2 * k, b:b + 2 * k] * kernel).sum()))
    novelty /= max(novelty.max(), 1e-9)
    for b in range(2, nb - 1):  # loudness steps mark sections too
        before, after = loud[max(0, b - k):b].mean(), loud[b:b + k].mean()
        novelty[b] += 0.5 * min(1.0, abs(after - before) / 6.0)
    # Only peaks: a change spread over a few bars is one boundary. Next to a peak a boundary
    # still scores most of it: a fill bar before a chorus peaks the novelty a bar early.
    peak = np.zeros_like(novelty)
    for b in range(1, nb):
        if novelty[b] >= novelty[max(0, b - 2):b + 3].max():
            peak[b] = max(peak[b], novelty[b])
            for n in (b - 1, b + 1):
                if 0 < n < nb:
                    peak[n] = max(peak[n], 0.8 * novelty[b])
    cuts = _segment(peak)
    # Labels: a section like an earlier one bar for bar is a repeat of it.
    level = float(np.mean(ssm) + np.std(ssm))
    sections: list[dict] = []
    for a, b in itertools.pairwise(cuts):
        label = None
        for s in sections:
            n = min(b - a, s["bars"][1] - s["bars"][0])
            same = float(np.mean([ssm[a + i, s["bars"][0] + i] for i in range(n)]))
            if same > level and abs((b - a) - (s["bars"][1] - s["bars"][0])) <= max(2, (b - a) // 2):
                label = s["label"]
                break
        if label is None:
            label = chr(ord("A") + len({s["label"] for s in sections}) % 26)
        sections.append({"bars": (a, b), "label": label, "db": float(loud[a:b].mean())})
    top = max(s["db"] for s in sections)
    for s in sections:
        s["energy"] = float(np.clip(1.0 + (s["db"] - top) / 12.0, 0.0, 1.0))
    return sections, {"ssm": ssm, "novelty": novelty, "loud": loud}


def _busy(f: dict[str, np.ndarray], beats: np.ndarray, a: int, b: int) -> float:
    """Onsets half way between the beats a..b against those on them."""
    on_frames = beats[a:b]
    if len(on_frames) < 3:
        return 0.0
    onset = _local(f["onset"])
    mid = ((beats[a:b - 1] + beats[a + 1:b]) // 2).astype(int)
    on = _at(onset, on_frames, 2).mean()
    return float(np.clip(_at(onset, mid, 2).mean() / on, 0.0, 1.0)) if on > 1e-9 else 0.0


def analyse(samples: np.ndarray) -> dict | None:
    """The song's map (see the module doc) from int16 samples at RATE, times in seconds of the
    audio; None for no beat."""
    x = np.asarray(samples, dtype=np.int16)
    if len(x) < 10 * RATE:
        return None
    f = _features(x)
    onset = _local(f["onset"])
    period = _tempo(onset)
    beat_frames = _beats(onset, period)
    if len(beat_frames) < 4 * BAR_BEATS:
        return None
    tempo = 60.0 * FPS / float(np.median(np.diff(beat_frames)))
    first, bar_confidence = _downbeat(f, beat_frames)
    sections, _ = _sections(f, beat_frames, first)
    step = 1
    while tempo / step > DANCE_BPM_MAX * (1 + FOLD_SLACK):
        step *= 2
    # The dance grid starts on the beat before the first bar that keeps it on the bars' beats.
    g0 = first % step
    grid = beat_frames[g0::step]
    per_bar = max(1, BAR_BEATS // step)
    to_grid = (first - g0) // step  # grid index of the first bar's first beat
    out_sections = []
    last_grid = len(grid) - 1
    for i, s in enumerate(sections):
        a, b = s["bars"]
        start = 0 if i == 0 else to_grid + a * per_bar
        end = last_grid + 1 if i == len(sections) - 1 else to_grid + b * per_bar
        e = s["energy"]
        kind = "main"
        if i == 0 and e < 0.6:
            kind = "intro"
        elif i == len(sections) - 1 and e < 0.6:
            kind = "outro"
        elif e < 0.35:
            kind = "break"
        beat_a, beat_b = first + a * BAR_BEATS, min(len(beat_frames), first + b * BAR_BEATS)
        out_sections.append({"start": int(start), "end": int(end), "label": s["label"], "energy": round(e, 2),
                             "busy": round(_busy(f, beat_frames, beat_a, beat_b), 2), "kind": kind})
    if not out_sections:
        out_sections = [{"start": 0, "end": last_grid + 1, "label": "A", "energy": 1.0, "busy": 0.3, "kind": "main"}]
    to_s = (lambda fr: round(float(fr * HOP + WIN // 2) / RATE, 3))
    return {
        "version": VERSION,
        "tempo": round(tempo, 1),
        "beats": [to_s(b) for b in beat_frames],
        "first_bar": int(first),
        "bar_confidence": round(bar_confidence, 2),
        "grid": [to_s(b) for b in grid],
        "bar_beats": per_bar,
        "first_grid_bar": int(to_grid),
        "sections": out_sections,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path")
    parser.add_argument("--out", help="write the map here (atomically) instead of printing it")
    args = parser.parse_args()
    song_map = analyse(decode(args.path))
    text = json.dumps(song_map, ensure_ascii=False)
    if args.out:
        tmp = args.out + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, args.out)
    else:
        sys.stdout.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
