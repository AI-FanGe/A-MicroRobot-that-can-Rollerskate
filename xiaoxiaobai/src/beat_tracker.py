# SPDX-License-Identifier: GPL-3.0-or-later

"""Online beat tracking of the song the music player decodes, for the upper body to sway to.

    tracker = BeatTracker()
    tracker.feed(int16_samples)        # as decoded, any chunk size
    grid = tracker.estimate()          # about once a second: None until there is enough audio

Onset strength is the spectral flux of a log-compressed spectrum (100 frames/s). The tempo
comes from the autocorrelation of the last TEMPO_WINDOW_S of it, weighted toward PRIOR_BPM (the
usual octave errors of beat trackers then mostly land on the tempo a listener would tap), and
the beat phase from where a comb of beats at that tempo lines up with the onsets best.

For the choreography on top: the bar (which beat of four the bass hits hardest, over the last
ENV_WINDOW_S) and how busy the music is between the beats (hi-hats, syncopation).
"""

from __future__ import annotations

import itertools
from collections import deque

import numpy as np

SAMPLE_RATE = 24000
HOP = 240                  # 10 ms
FPS = SAMPLE_RATE / HOP
WIN = 1024
MAX_HZ = 8000.0            # the flux above it is mostly hi-hat hiss
BANDS = 24
LOW_HZ = 200.0             # kick and bass, for the bar's first beat
ENV_WINDOW_S = 16.0
TEMPO_WINDOW_S = 8.0
PHASE_WINDOW_S = 4.0
BAR_BEATS = 4
MIN_ANALYSE_S = 3.0
BPM_RANGE = (60.0, 180.0)
# The beat grid handed out is folded under this, an octave above BPM_RANGE's floor (a 128 BPM
# song sways at 64, gently): octave errors then change nothing, and the comb picks the
# stronger of the two beat sets.
DANCE_BPM_MAX = 120.0
FOLD_SLACK = 0.08          # how far past the range a folding already chosen holds (no flip-flop at 120)
SHIFT_DECAY = 0.9          # per estimate (about a second): the bars' say in a folded grid's phase
SHIFT_SWITCH = 2.0         # how far the bars must outvote the current phase to move it
PRIOR_BPM = 110.0
PRIOR_OCTAVES = 0.9        # the prior's width (std of log2 tempo)
# A new tempo replaces the current one only after this many estimates in a row agree on it.
TEMPO_SWITCH = 3
TEMPO_SAME = 0.04          # relative difference still counted as the same tempo
LOUDNESS_WINDOW_S = 1.0
LOUDNESS_DECAY_S = 30.0    # of the song's loudest level, against which loudness is measured


class BeatTracker:
    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"BeatTracker expects {SAMPLE_RATE} Hz audio")
        self._window = np.hanning(WIN).astype(np.float32)
        self._bins = int(MAX_HZ / (SAMPLE_RATE / WIN))
        # Log-spaced bands, so that a kick (a few low bins) counts as much as a snare (most of them).
        edges = np.unique(np.geomspace(2, self._bins, BANDS + 1).astype(int))
        bands = np.zeros((self._bins, len(edges) - 1), dtype=np.float32)
        for i, (lo, hi) in enumerate(itertools.pairwise(edges)):
            bands[lo:hi, i] = 1.0 / (hi - lo)
        self._bands = bands
        self._low_bands = max(1, int(np.sum(edges[1:] <= LOW_HZ / (SAMPLE_RATE / WIN))))
        self._buf = np.zeros(0, dtype=np.float32)   # samples not framed yet (with the frame overlap)
        self._prev: np.ndarray | None = None         # the last frame's log spectrum
        self.frames = 0                              # frame i covers samples [i * HOP, i * HOP + WIN)
        self._env: deque[float] = deque(maxlen=int(ENV_WINDOW_S * FPS))
        self._low: deque[float] = deque(maxlen=int(ENV_WINDOW_S * FPS))
        self._rms: deque[float] = deque(maxlen=int(LOUDNESS_WINDOW_S * FPS))
        self._loudest = 1e-3
        self._period: float | None = None             # frames per beat
        self._candidate: tuple[float, int] | None = None
        self._folds: int | None = None                # the beat grid is 2 ** folds beats apart
        self._shift_votes: list[float] = []          # which song beat of a folded grid's starts bars
        self._shift = 0
        self._shift_period = 1.0                      # the song beat [frames] the votes are for

    def feed(self, samples: np.ndarray) -> None:
        """Mono samples (int16 or float in int16 scale), in order."""
        self._buf = np.concatenate([self._buf, np.asarray(samples, dtype=np.float32) / 32768.0])
        count = (len(self._buf) - WIN) // HOP + 1
        if count <= 0:
            return
        frames = np.lib.stride_tricks.sliding_window_view(self._buf, WIN)[::HOP][:count]
        power = np.abs(np.fft.rfft(frames * self._window, axis=1))[:, :self._bins] @ self._bands
        spectrum = np.log1p(100.0 * power)
        previous = spectrum[:1] if self._prev is None else self._prev[None, :]
        rise = np.maximum(np.diff(np.vstack([previous, spectrum]), axis=0), 0.0)
        self._prev = spectrum[-1]
        self._env.extend(rise.mean(axis=1).tolist())
        self._low.extend(rise[:, :self._low_bands].mean(axis=1).tolist())
        self._rms.extend(np.sqrt((frames ** 2).mean(axis=1)).tolist())
        self.frames += count
        self._buf = self._buf[count * HOP:]

    def estimate(self) -> dict[str, float] | None:
        """{"bpm" (folded under DANCE_BPM_MAX), "tempo" (as detected), "beat_sample" (a beat at bpm, as
        a sample index of the fed audio), "confidence" 0-1, "energy" 0-1 (loudness against the
        song's loudest), "downbeat_sample" (a beat that starts a bar of BAR_BEATS song beats),
        "bar_confidence" 0-1, "bar_beats" (beats at bpm per bar: 4, or fewer when folded), "busy" 0-1
        (onsets between the beats against those on them)};
        None before MIN_ANALYSE_S."""
        if len(self._env) < MIN_ANALYSE_S * FPS:
            return None
        onsets, low = self._above_local(self._env), self._above_local(self._low)
        period, peak = self._tempo(onsets[-int(TEMPO_WINDOW_S * FPS):])
        if period is None:
            return None
        period = self._settle(period)
        tempo = 60.0 * FPS / period
        held = None if self._folds is None else tempo / 2 ** self._folds
        if held is None or not DANCE_BPM_MAX / 2 * (1 - FOLD_SLACK) < held < DANCE_BPM_MAX * (1 + FOLD_SLACK):
            self._folds = 0
            while tempo / 2 ** self._folds > DANCE_BPM_MAX:
                self._folds += 1
        beat, comb = self._phase(onsets, period * 2 ** self._folds)
        downbeat, bar_confidence = self._bar(low, beat, period)
        bar_beats = max(1, BAR_BEATS // 2 ** self._folds)
        if self._folds:
            # A folded grid could sit on the off beats (2 and 4): move it onto the bars' beats, once
            # the bars have pointed the same way for a while.
            steps = 2 ** self._folds
            if len(self._shift_votes) != steps or abs(period / self._shift_period - 1.0) > TEMPO_SAME:
                self._shift_votes, self._shift, self._shift_period = [0.0] * steps, 0, period
            shift = round((downbeat - beat) / period) % steps
            self._shift_votes = [v * SHIFT_DECAY for v in self._shift_votes]
            self._shift_votes[shift] += bar_confidence
            best = max(range(steps), key=self._shift_votes.__getitem__)
            if self._shift_votes[best] > SHIFT_SWITCH * max(self._shift_votes[self._shift], 1.0):
                self._shift = best
            beat += self._shift * period
            period *= steps
        loud = float(np.sqrt(np.mean(np.square(self._rms))))
        decay = np.exp(-(len(self._rms) / FPS) / LOUDNESS_DECAY_S)
        self._loudest = max(loud, self._loudest * decay)
        return {
            "bpm": 60.0 * FPS / period,
            "tempo": tempo,
            "beat_sample": int(beat * HOP + WIN // 2),
            "confidence": float(np.clip(min(peak * 2.5, comb), 0.0, 1.0)),
            "energy": float(np.clip(loud / self._loudest, 0.0, 1.0)),
            "downbeat_sample": int(downbeat * HOP + WIN // 2),
            "bar_confidence": bar_confidence,
            "bar_beats": bar_beats,
            "busy": self._busy(onsets, beat, 60.0 * FPS / tempo),
        }

    @staticmethod
    def _above_local(env: deque[float]) -> np.ndarray:
        """Onsets above the local level, so that loud passages do not swamp the comb."""
        x = np.asarray(env)
        return np.maximum(x - np.convolve(x, np.ones(31) / 31, mode="same"), 0.0)

    def _at_beats(self, onsets: np.ndarray, beat: float, period: float, shift: float = 0.0) -> np.ndarray:
        """The strongest onset within 20 ms of each beat (plus shift beats) in the window, latest first."""
        n = len(onsets)
        first = self.frames - n  # the window's first frame
        count = int((n - 4) / period)
        index = np.rint(beat + shift * period - np.arange(count) * period - first).astype(int)
        index = index[(index >= 2) & (index < n - 2)]
        if not len(index):
            return np.zeros(0)
        return np.max(np.stack([onsets[index + d] for d in range(-2, 3)]), axis=0)

    def _bar(self, low: np.ndarray, beat: float, period: float) -> tuple[float, float]:
        """(a frame that starts a bar, how clearly the bass marks it: 0-1)."""
        accents = self._at_beats(low, beat, period)
        if len(accents) < 2 * BAR_BEATS:
            return beat, 0.0
        strength = np.array([accents[k::BAR_BEATS].mean() for k in range(BAR_BEATS)])
        mean = float(strength.mean())
        if mean <= 1e-9:
            return beat, 0.0
        k = int(np.argmax(strength))
        return beat - k * period, float(np.clip((strength[k] / mean - 1.0) / 0.6, 0.0, 1.0))

    def _busy(self, onsets: np.ndarray, beat: float, tempo_period: float) -> float:
        """Onsets half way between the song's beats against those on them: 0 bare, 1 as strong."""
        on = self._at_beats(onsets, beat, tempo_period)
        off = self._at_beats(onsets, beat, tempo_period, -0.5)
        if not len(on) or not len(off) or on.mean() <= 1e-9:
            return 0.0
        return float(np.clip(off.mean() / on.mean(), 0.0, 1.0))

    def _tempo(self, onsets: np.ndarray) -> tuple[float | None, float]:
        """(frames per beat, normalized autocorrelation peak)."""
        x = onsets - onsets.mean()
        n = len(x)
        spectrum = np.fft.rfft(x, 2 * n)
        ac = np.fft.irfft(spectrum * np.conj(spectrum))[:n]
        if ac[0] <= 1e-12:
            return None, 0.0
        ac /= ac[0]
        lo = int(np.floor(60.0 * FPS / BPM_RANGE[1]))
        hi = min(int(np.ceil(60.0 * FPS / BPM_RANGE[0])), n - 2)
        lags = np.arange(lo, hi + 1)
        bpm = 60.0 * FPS / lags
        prior = np.exp(-0.5 * (np.log2(bpm / PRIOR_BPM) / PRIOR_OCTAVES) ** 2)
        best = int(lags[np.argmax(np.maximum(ac[lags], 0.0) * prior)])
        a, b, c = ac[best - 1], ac[best], ac[best + 1]
        bend = a - 2 * b + c
        shift = 0.5 * (a - c) / bend if bend < 0 else 0.0
        return best + float(np.clip(shift, -0.5, 0.5)), float(max(b, 0.0))

    def _settle(self, period: float) -> float:
        """The tempo, kept steady: a different one is taken once it has held for TEMPO_SWITCH
        estimates; twice or half the current one is the same beat, and changes nothing."""
        current = self._period
        if current is None or abs(period - current) / current <= TEMPO_SAME:
            self._period = period if current is None else 0.7 * current + 0.3 * period
            self._candidate = None
            return self._period
        if any(abs(period / current - ratio) / ratio <= TEMPO_SAME for ratio in (2.0, 0.5)):
            self._candidate = None
            return current
        held, count = self._candidate or (period, 0)
        if abs(period - held) / held <= TEMPO_SAME:
            count += 1
        else:
            held, count = period, 1
        if count >= TEMPO_SWITCH:
            self._period, self._candidate = period, None
        else:
            self._candidate = (held, count)
        return self._period

    def _phase(self, onsets: np.ndarray, period: float) -> tuple[float, float]:
        """(the frame of the latest beat, how much the comb stands out: 0-1)."""
        recent = onsets[-int(PHASE_WINDOW_S * FPS):]
        smooth = np.convolve(recent, np.ones(5) / 5, mode="same")
        n = len(smooth)
        teeth = np.arange(int(n / period))
        weights = 0.85 ** teeth
        offsets = np.arange(int(np.ceil(period)))
        positions = np.rint(n - 1 - offsets[:, None] - teeth[None, :] * period).astype(int)
        valid = positions >= 0
        scores = (np.where(valid, smooth[np.clip(positions, 0, n - 1)], 0.0) * weights).sum(axis=1)
        best = int(np.argmax(scores))
        mean = float(scores.mean())
        stand_out = (float(scores[best]) / mean - 1.0) / 1.5 if mean > 1e-12 else 0.0
        beat = self.frames - 1 - best
        return float(beat), float(np.clip(stand_out, 0.0, 1.0))
