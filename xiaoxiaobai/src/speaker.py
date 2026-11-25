# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

"""Play a speaker test on the head board.

While `make run` (or the display service) is up, the request is handed to the display
loop that owns the ESP32 link; otherwise this script connects to the ESP32 itself.
"""

from __future__ import annotations

import argparse
import time

from display_smile import DisplayAlreadyRunning, display_lock
from esp32_link import (
    SOUND_REQUEST_PATH,
    SPEAKER_TEST_NOTES,
    SPEAKER_TEST_VOLUME,
    open_esp32_link,
    request_sound,
)

HANDOFF_TIMEOUT_S = 0.6


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", type=int, default=SPEAKER_TEST_VOLUME, help="0-100 (default %(default)s)")
    parser.add_argument("--freq", type=int, default=None, help="play a single tone at this frequency (Hz)")
    parser.add_argument("--ms", type=int, default=1000, help="duration of --freq in ms (default %(default)s)")
    args = parser.parse_args()
    notes = [(args.freq, args.ms)] if args.freq else list(SPEAKER_TEST_NOTES)

    request_sound(notes, args.volume)
    deadline = time.monotonic() + HANDOFF_TIMEOUT_S
    while time.monotonic() < deadline:
        if not SOUND_REQUEST_PATH.exists():
            print("Speaker test sent through the running display loop.")
            return
        time.sleep(0.05)
    SOUND_REQUEST_PATH.unlink(missing_ok=True)

    try:
        with display_lock():
            try:
                link = open_esp32_link()
            except OSError as exc:
                raise SystemExit(f"Cannot reach the ESP32: {exc}") from None
            try:
                link.play(notes, args.volume)
            finally:
                link.close()
    except DisplayAlreadyRunning:
        raise SystemExit(
            "The display loop is running but did not take the request: the ESP32 is probably "
            "not connected yet (check the `make run` output), XIAOXIAOBAI_DISPLAY_BACKEND=spi, or it is "
            "the xiaoxiaobai-display service still running older code (`sudo systemctl restart xiaoxiaobai-display`)."
        ) from None
    print(f"Speaker test sent directly via {link.description}.")


if __name__ == "__main__":
    main()
