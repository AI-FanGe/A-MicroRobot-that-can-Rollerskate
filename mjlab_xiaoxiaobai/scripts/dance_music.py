"""Synthesize a soundtrack for the dance, on the beats of the motion itself.

usage: dance_music.py [--ref ../xiaoxiaobai/src/agents/dance_ref.npz] [--out ../xiaoxiaobai/src/agents/dance_music.wav]
                      [--grid FIRST:PERIOD] [--key major]

The beats come from the reference: an onset strength from the joint speeds (the legs weigh
most), a global tempo from its autocorrelation and dynamic programming for the beat sequence
that fits both (Ellis 2007), so the kicks land where the dancer pushes. The bar starts on the
beat of four with the strongest leg onsets. Before the dance proper (the blend-in from the
reference pose) the grid is extended backwards for a count-in; after it, one closing hit.

House at the motion's tempo: kick on every beat, claps on 2 and 4, offbeat hats, an offbeat
bass over Am-F-C-G and pumping chords, layered in by four bars; the robot's small speaker has
little under 150 Hz, so the low end sits an octave up. The file starts with PAD_S of silence
(the deployment starts the dance that long after the first sample is heard, which covers the
control loop's latency), then t = 0 is the dance's frame 0. 24 kHz mono 16-bit, the speaker's rate.
"""
import argparse
import json
import wave
from pathlib import Path

import numpy as np

SR = 24000
PAD_S = 0.5
BLEND_S = 1.5  # dance_motion_npz.py: the reference blends in from and out to the reference pose
AGENTS = Path(__file__).resolve().parents[2] / "xiaoxiaobai/src/agents"
LEGS = [3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 17]
ARMS = [0, 1, 2, 9, 10, 11]
KEYS = {
    "minor": ([(57, 60, 64), (53, 57, 60), (48, 52, 55), (55, 59, 62)], [45, 41, 48, 43]),  # Am F C G (MIDI)
    "major": ([(48, 52, 55), (55, 59, 62), (57, 60, 64), (53, 57, 60)], [48, 43, 45, 41]),  # C G Am F
}


CHORDS, BASS = list(KEYS["minor"][0]), list(KEYS["minor"][1])


def midi_hz(m: float) -> float:
    return 440.0 * 2 ** ((m - 69) / 12)


def beat_grid(ref: np.lib.npyio.NpzFile) -> tuple[np.ndarray, int, float]:
    fps = float(ref["fps"])
    v = np.abs(ref["joint_vel"])
    energy = 2.0 * v[:, LEGS].sum(1) + v[:, ARMS].sum(1)
    onset = np.maximum(np.diff(energy, prepend=energy[0]), 0.0)
    a, b = int(BLEND_S * fps), len(onset) - int(BLEND_S * fps)
    on = onset[a:b] / (onset[a:b].std() + 1e-9)

    ac = np.correlate(on - on.mean(), on - on.mean(), "full")[len(on) - 1:]
    bpms = np.arange(90.0, 160.0, 0.25)
    score = [sum(np.interp(k * 60 * fps / bpm, np.arange(len(ac)), ac) for k in range(1, 9)) for bpm in bpms]
    period = 60 * fps / bpms[int(np.argmax(score))]

    # Ellis 2007: best[t] = on[t] + max over prev of best[prev] - alpha * log2((t - prev) / period)^2
    n, alpha = len(on), 100.0
    best, back = on.copy(), np.full(n, -1)
    for t in range(n):
        lo, hi = int(t - 2 * period), int(t - period / 2)
        if hi <= 0:
            continue
        prev = np.arange(max(lo, 0), hi)
        cand = best[prev] - alpha * np.log2((t - prev) / period) ** 2
        k = int(np.argmax(cand))
        best[t], back[t] = on[t] + cand[k], prev[k]
    t = int(np.argmax(best[-int(period):])) + n - int(period)
    beats = []
    while t >= 0:
        beats.append(t)
        t = back[t]
    beats = (np.array(beats[::-1]) + a) / fps

    legs = np.maximum(np.diff(v[:, LEGS].sum(1), prepend=0), 0)
    strength = [legs[np.clip((beats[k::4] * fps).astype(int), 0, len(legs) - 1)].mean() for k in range(4)]
    return beats, int(np.argmax(strength)), 60 * fps / period


def env(n: int, attack: float, decay: float) -> np.ndarray:
    t = np.arange(n) / SR
    return np.minimum(t / max(attack, 1e-4), 1.0) * np.exp(-t / decay)


def lowpass(x: np.ndarray, hz: float) -> np.ndarray:
    a = np.exp(-2 * np.pi * hz / SR)
    y = np.empty_like(x)
    acc = 0.0
    for i, s in enumerate(x):
        acc = (1 - a) * s + a * acc
        y[i] = acc
    return y


def highpass(x: np.ndarray, hz: float) -> np.ndarray:
    return x - lowpass(x, hz)


def saw(freq: float, n: int, phase: float = 0.0) -> np.ndarray:
    t = np.arange(n) / SR
    return 2 * ((t * freq + phase) % 1.0) - 1


rng = np.random.default_rng(7)
_NOISE = rng.uniform(-1, 1, SR * 2)


def kick(n=int(0.35 * SR)) -> np.ndarray:
    t = np.arange(n) / SR
    freq = 70 + 160 * np.exp(-t / 0.03)
    body = np.sin(2 * np.pi * np.cumsum(freq) / SR) * np.exp(-t / 0.12)
    click = highpass(_NOISE[:n], 2000) * np.exp(-t / 0.004)
    return np.tanh(2.2 * body) + 0.4 * click


def clap(n=int(0.25 * SR)) -> np.ndarray:
    t = np.arange(n) / SR
    burst = sum(np.where(t >= d, np.exp(-(t - d) / 0.008), 0) for d in (0.0, 0.011, 0.022))
    tail = np.exp(-t / 0.09)
    noise = highpass(lowpass(_NOISE[SR:SR + n], 5000), 900)
    return noise * (0.8 * burst + 0.5 * tail)


def hat(open_: bool) -> np.ndarray:
    n = int((0.22 if open_ else 0.05) * SR)
    return highpass(_NOISE[:n], 7000) * env(n, 0.001, 0.07 if open_ else 0.012)


def crash(n=int(1.6 * SR)) -> np.ndarray:
    return highpass(_NOISE[:n] * 0.7 + _NOISE[SR // 3:SR // 3 + n] * 0.5, 4000) * env(n, 0.002, 0.55)


def tone(freqs, n: int, cutoff: float, attack: float, decay: float, detune: float = 0.006) -> np.ndarray:
    x = sum(saw(f * (1 + d), n, i * 0.37) for i, f in enumerate(freqs) for d in (-detune, detune))
    return lowpass(x / (2 * len(freqs)), cutoff) * env(n, attack, decay)


def place(track: np.ndarray, clip: np.ndarray, t: float, gain: float) -> None:
    i = int(round((t + PAD_S) * SR))
    if i >= len(track) or i + len(clip) <= 0:
        return
    j = max(0, -i)
    end = min(len(track), i + len(clip))
    track[max(i, 0):end] += gain * clip[j:j + end - max(i, 0)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path, default=AGENTS / "dance_ref.npz")
    ap.add_argument("--out", type=Path, default=AGENTS / "dance_music.wav")
    ap.add_argument("--grid", help="FIRST:PERIOD [s]: the counts of a choreographed routine instead of tracked beats")
    ap.add_argument("--key", choices=tuple(KEYS), default="minor")
    args = ap.parse_args()
    CHORDS[:], BASS[:] = KEYS[args.key]
    ref = np.load(args.ref)
    duration = len(ref["joint_pos"]) / float(ref["fps"])
    if args.grid:
        first, period = map(float, args.grid.split(":"))
        beats, bar_offset, bpm = np.arange(first, duration - 0.5 * period, period), 0, 60.0 / period
    else:
        beats, bar_offset, bpm = beat_grid(ref)
    period = float(np.median(np.diff(beats)))
    # Count-in: the grid continued backwards over the blend-in, starting on a bar.
    first_bar = bar_offset
    pre = [beats[first_bar] - k * period for k in range(1, 9) if beats[first_bar] - k * period > -0.05]
    end_t = duration - 0.25  # the closing hit, as the robot settles in the reference pose

    total = int((PAD_S + duration + 2.0) * SR)
    drums, bass, chords, lead, fx = (np.zeros(total) for _ in range(5))
    duck = np.ones(total)
    dk = 1 - 0.65 * np.exp(-np.arange(int(0.3 * SR)) / SR / 0.09)

    for t in sorted(pre):
        place(drums, hat(False), t, 0.35)
        place(drums, hat(False), t + period / 2, 0.2)
    riser = int(len(pre) * period * SR)
    place(fx, highpass(np.resize(_NOISE, riser) * np.linspace(0, 1, riser) ** 2, 3000), sorted(pre)[0] if pre else 0.0, 0.25)

    main_beats = beats[first_bar:]
    for k, t in enumerate(main_beats):
        bar, pos = divmod(k, 4)
        section = bar // 4  # 0 drums+bass, 1 + claps/chords, 2 + lead, 3+ full
        chord = bar % 4
        place(drums, kick(), t, 0.95)
        i = int(round((t + PAD_S) * SR))
        duck[i:i + len(dk)] = np.minimum(duck[i:i + len(dk)], dk[:max(0, min(len(dk), total - i))])
        place(drums, hat(True), t + period / 2, 0.28)
        if section >= 1:
            for s in (0.25, 0.75):
                place(drums, hat(False), t + s * period, 0.12)
        if section >= 1 and pos in (1, 3):
            place(drums, clap(), t, 0.55)
        if k % 16 == 0:
            place(fx, crash(), t, 0.35 if k else 0.5)
        note = BASS[chord] + (12 if pos % 2 else 0) + 12
        place(bass, tone([midi_hz(note)], int(period / 2 * SR), 900, 0.004, 0.12, 0.003), t + period / 2, 0.55)
        if section >= 1 and pos == 0:
            place(chords, tone([midi_hz(m + 12) for m in CHORDS[chord]], int(4 * period * SR), 2200, 0.02, 1.2), t, 0.5)
        if section >= 2:
            arp = [CHORDS[chord][j % 3] + 24 + 12 * (j // 3) for j in (0, 1, 2, 3)]
            for j in range(4 if section >= 3 else 2):
                m = arp[(pos * 4 + j) % 4]
                place(lead, tone([midi_hz(m)], int(period / 4 * SR), 3500, 0.003, 0.08, 0.002), t + j * period / 4, 0.3)

    # The robot settles into the reference pose: let it ring out, then a final hit.
    place(drums, kick(), end_t, 1.0)
    place(fx, crash(), end_t, 0.6)
    place(chords, tone([midi_hz(m + 12) for m in CHORDS[0]], int(2.0 * SR), 2000, 0.01, 0.8), end_t, 0.6)

    delay = int(0.75 * period * SR)
    lead[delay:] += 0.35 * lead[:-delay].copy()
    mix = drums + duck * (bass + 0.8 * chords + lead) + fx
    stop = int((PAD_S + main_beats[-1] + period) * SR)
    gate = np.ones(total)
    end_i = int((PAD_S + end_t) * SR)
    gate[stop:end_i] = np.linspace(1, 0.35, max(0, end_i - stop))
    mix[stop:end_i] *= gate[stop:end_i]
    mix[:int(PAD_S * SR)] = 0
    mix = np.tanh(1.4 * mix / np.max(np.abs(mix)))
    mix *= 0.95 / np.max(np.abs(mix))
    fade = int(0.8 * SR)
    mix[-fade:] *= np.linspace(1, 0, fade)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(args.out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((mix * 32767).astype("<i2").tobytes())
    print(json.dumps({"out": str(args.out), "bpm": round(bpm, 1), "beats": len(beats), "first_beat_s": round(beats[0], 3),
                      "bar_offset": bar_offset, "pad_s": PAD_S, "seconds": round(total / SR, 2)}))


if __name__ == "__main__":
    main()
