# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

"""Memo boards on the face screen: a day's list, a note ("已记下"), a reminder or the clock, drawn
at the panel's full 320x240 in the web UI's look (dark cards, one accent colour), always on one
page: the text shrinks, then goes to two columns, then the last row says how many did not fit.

The API asks for one through a request file (request_board); the display loop shows it in place
of the face for a while, then the face comes back. On the ESP32 the picture goes over once as
MSG_IMAGE (Esp32Board) and the firmware holds it; a Pi-driven LCD gets it pushed directly.

Font: Noto Sans CJK SC (SIL OFL, fonts/OFL.txt), cut down to GB2312 and common symbols.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np

from esp32_link import IMAGE_HIDE, image_packets

BOARD_REQUEST_PATH = Path("/tmp/xiaoxiaobai_board.json")
REQUEST_MAX_AGE_S = 10.0
FONT_DIR = Path(__file__).resolve().parent / "fonts"
WIDTH, HEIGHT = 320, 240
# Shapes are drawn at this scale and shrunk (smooth edges); text goes on afterwards at 1x.
SUPERSAMPLE = 3
MIN_SECONDS, MAX_SECONDS = 4.0, 180.0
WEEKDAYS = "一二三四五六日"

# webui/css/app.css
BG = "#0b0d12"
PANEL = "#141821"
PANEL_2 = "#1a1f2a"
BORDER = "#252b39"
TEXT = "#e7e9ef"
TEXT_2 = "#b6bccb"
MUTED = "#7f889b"
OK = "#22c55e"
ACCENTS = {"blue": "#5b8cff", "green": OK, "pink": "#f472b6", "red": "#ef4444", "gold": "#f59e0b",
           "cyan": "#2dd4bf", "purple": "#a78bfa"}

MARGIN = 12  # screen edge to card
PAD = 14  # card edge to content
CARD_TOP = 62
COMPACT_CARD_TOP = 36  # under the one-line header of a crowded list
CARD_BOTTOM = HEIGHT - MARGIN
# List text sizes tried in turn (two lines per item down to WRAP_MIN_PX), then with the compact
# header and tight rows, then two columns.
LIST_SIZES = (19, 18, 17, 16, 15, 14)
COMPACT_SIZES = (15, 14, 13, 12)
WRAP_MIN_PX = 14
COLUMN_PX = 13

# Esp32Board: bytes of MSG_IMAGE per display tick (the ESP32 reads 16 KB at a time, and faces and
# speech share the UART), how long to wait for its answer, and how often to try.
IMAGE_BYTES_PER_TICK = 6000
IMAGE_ACK_S = 1.5
IMAGE_TRIES = 3


def request_board(spec: dict[str, Any], path: Path = BOARD_REQUEST_PATH) -> None:
    """Queue a board for the display loop; {"kind": "clear"} takes the current one down."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"ts": time.time(), **spec}, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def take_board_request(path: Path = BOARD_REQUEST_PATH) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        path.unlink()
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or time.time() - float(data.get("ts", 0)) > REQUEST_MAX_AGE_S:
        return None
    return data


@lru_cache(maxsize=32)
def font(size: float, bold: bool = False):
    from PIL import ImageFont

    return ImageFont.truetype(str(FONT_DIR / f"NotoSansSC-{'Bold' if bold else 'Regular'}.otf"), round(size))


def text_width(text: str, size: float, bold: bool = False) -> float:
    return font(size, bold).getlength(text)


def ellipsize(text: str, width: float, size: float, bold: bool = False) -> str:
    if text_width(text, size, bold) <= width:
        return text
    while text and text_width(text + "…", size, bold) > width:
        text = text[:-1]
    return text.rstrip() + "…"


def wrap(text: str, width: float, size: float, max_lines: int) -> list[str]:
    """Lines of at most `width` px, broken anywhere (Chinese) but at a space when there is one;
    what does not fit in `max_lines` ends in an ellipsis."""
    lines: list[str] = []
    rest = " ".join(text.split())
    while rest and len(lines) < max_lines:
        if len(lines) == max_lines - 1 or text_width(rest, size) <= width:
            lines.append(ellipsize(rest, width, size))
            break
        cut = 1
        while cut < len(rest) and text_width(rest[: cut + 1], size) <= width:
            cut += 1
        space = rest.rfind(" ", 0, cut + 1)
        if space > cut // 2:
            cut = space
        lines.append(rest[:cut].rstrip())
        rest = rest[cut:].lstrip()
    return lines or [""]


def _rgb(color: str) -> tuple[int, int, int]:
    return int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)


def mix(color: str, over: str, alpha: float) -> str:
    """`color` at `alpha` over `over`."""
    a, b = _rgb(color), _rgb(over)
    return "#" + "".join(f"{round(x * alpha + y * (1 - alpha)):02x}" for x, y in zip(a, b))


class Canvas:
    """Shapes at SUPERSAMPLE, text queued and drawn on the shrunk image; coordinates in screen px."""

    def __init__(self) -> None:
        from PIL import Image, ImageDraw

        self._big = Image.new("RGB", (WIDTH * SUPERSAMPLE, HEIGHT * SUPERSAMPLE), BG)
        self._draw = ImageDraw.Draw(self._big)
        self._texts: list[tuple[float, float, str, float, bool, str, str]] = []
        self._strikes: list[tuple[float, float, float, str]] = []

    def _s(self, *values: float) -> list[int]:
        return [round(v * SUPERSAMPLE) for v in values]

    def box(self, x0: float, y0: float, x1: float, y1: float, radius: float, fill: str, outline: str | None = None) -> None:
        self._draw.rounded_rectangle(self._s(x0, y0, x1, y1), radius * SUPERSAMPLE, fill=fill, outline=outline,
                                     width=SUPERSAMPLE if outline else 0)

    def circle(self, cx: float, cy: float, r: float, fill: str | None = None, outline: str | None = None,
               width: float = 1.5) -> None:
        self._draw.ellipse(self._s(cx - r, cy - r, cx + r, cy + r), fill=fill, outline=outline,
                           width=round(width * SUPERSAMPLE) if outline else 0)

    def line(self, points: list[tuple[float, float]], color: str, width: float = 1.0) -> None:
        self._draw.line([tuple(self._s(x, y)) for x, y in points], fill=color, width=round(width * SUPERSAMPLE),
                        joint="curve")

    def check(self, cx: float, cy: float, r: float, color: str) -> None:
        self.line([(cx - r * 0.45, cy + r * 0.02), (cx - r * 0.1, cy + r * 0.38), (cx + r * 0.5, cy - r * 0.35)],
                  color, max(1.4, r * 0.28))

    def text(self, x: float, y: float, text: str, size: float, color: str, bold: bool = False, anchor: str = "lm") -> float:
        """`anchor` as in PIL ("lm": x left, y middle); returns the width."""
        self._texts.append((x, y, text, size, bold, color, anchor))
        return text_width(text, size, bold)

    def strike(self, x0: float, x1: float, y: float, color: str) -> None:
        self._strikes.append((x0, x1, y, color))

    def pill(self, right: float, cy: float, text: str, color: str, size: float = 12) -> float:
        """A badge ending at `right`; returns its left edge."""
        w = text_width(text, size, True) + 18
        h = size + 11
        self.box(right - w, cy - h / 2, right, cy + h / 2, h / 2, mix(color, BG, 0.16))
        self.text(right - w / 2, cy, text, size, color, bold=True, anchor="mm")
        return right - w

    def image(self):
        from PIL import Image, ImageDraw

        out = self._big.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(out)
        for x0, x1, y, color in self._strikes:
            draw.line([(x0, y), (x1, y)], fill=color, width=1)
        for x, y, text, size, bold, color, anchor in self._texts:
            draw.text((x, y), text, font=font(size, bold), fill=color, anchor=anchor)
        return out


@dataclass
class _Row:
    item: dict[str, Any]
    lines: list[str]


@dataclass
class _Style:
    """How the rows of a list are laid out: px sizes, wrapped lines per item, the status mark
    (ring with a tick, or a small dot) and the width of the day/time label column."""

    size: int
    line_h: float
    pad: float
    label_size: int
    lines: int
    dot_r: float
    with_time: bool = True
    ring: bool = True
    label_w: float = 0.0

    @property
    def lead(self) -> float:
        return 2 * self.dot_r + 10 + (self.label_w + 10 if self.label_w else 0)


class Board:
    """One board, from its request until it times out."""

    def __init__(self, spec: dict[str, Any], now: float) -> None:
        self.spec = spec
        self.kind = spec.get("kind", "list")
        self.accent = ACCENTS.get(str(spec.get("title_color") or ""), ACCENTS["blue"])
        self.items = [i for i in spec.get("items") or [] if isinstance(i, dict)]
        try:
            self.tz = ZoneInfo(str(spec["tz"])) if spec.get("tz") else None
        except (ZoneInfoNotFoundError, ValueError):
            self.tz = None
        self.end = now + min(MAX_SECONDS, max(MIN_SECONDS, float(spec.get("seconds") or 30)))

    def _now(self) -> datetime:
        return datetime.now(self.tz) if self.tz else datetime.now()

    def key(self) -> str:
        """Changes when the picture has to be drawn again (the clock, each minute)."""
        return f"{self._now():%H:%M}" if self.kind == "clock" else ""

    def render(self):
        canvas = Canvas()
        if self.kind == "clock":
            self._clock(canvas)
        else:
            self._list(canvas)
        return canvas.image()

    # ---- list ----------------------------------------------------------------------------------

    def _header(self, c: Canvas, compact: bool) -> None:
        title = str(self.spec.get("title") or "备忘录")
        subtitle = str(self.spec.get("subtitle") or "")
        items = self.items
        open_count = sum(not i.get("done") for i in items)
        badge, badge_color = str(self.spec.get("badge") or ""), self.accent
        if not badge and items:
            badge, badge_color = (f"{open_count} 项待办", self.accent) if open_count else ("全部完成", OK)
        left, right = MARGIN + 13, WIDTH - MARGIN
        cy = 19 if compact else 29
        c.box(MARGIN, cy - (8 if compact else 12), MARGIN + 4, cy + (8 if compact else 12), 2, self.accent)
        if badge:
            right = c.pill(right, cy, badge, badge_color, 11 if compact else 12) - 8
        if compact:
            x = left + c.text(left, cy - 1, ellipsize(title, right - left, 18, True), 18, TEXT, bold=True) + 10
            if subtitle and right - x > 30:
                c.text(x, cy, ellipsize(subtitle, right - x, 12), 12, MUTED)
            return
        c.text(left, 22, ellipsize(title, right - left, 22, True), 22, TEXT, bold=True)
        if subtitle:
            c.text(left, 45, ellipsize(subtitle, right - left, 12), 12, MUTED)

    def _list(self, c: Canvas) -> None:
        items = self.items
        if len(items) <= 1:
            self._header(c, False)
            if items:
                self._hero(c, items[0])
            else:
                self._empty(c, str(self.spec.get("empty") or "没有安排"))
            return
        for compact, sizes in ((False, LIST_SIZES), (True, COMPACT_SIZES)):
            top = COMPACT_CARD_TOP if compact else CARD_TOP
            area = CARD_BOTTOM - top - 8
            for size in sizes:
                style = self._style(size, compact)
                width = WIDTH - 2 * MARGIN - 2 * PAD - style.lead
                rows = [_Row(i, wrap(str(i.get("text", "")), width, size, style.lines)) for i in items]
                height = sum(len(r.lines) * style.line_h + 2 * style.pad for r in rows)
                if height <= area:
                    # Rows share the spare room (up to a point) so the card fills the screen.
                    style.pad += min((area - height) / len(rows), size * 0.9) / 2
                    self._header(c, compact)
                    c.box(MARGIN, top, WIDTH - MARGIN, CARD_BOTTOM, 14, PANEL, BORDER)
                    self._rows(c, rows, MARGIN + PAD, WIDTH - MARGIN - PAD, top + 4, style)
                    return
        self._header(c, True)
        self._columns(c, items)

    def _style(self, size: int, compact: bool, with_time: bool = True) -> _Style:
        multi_day = any(i.get("day") for i in self.items)
        label_size = round(size * 0.82) if multi_day else size
        style = _Style(size=size, line_h=size + 5 if compact else round(size * 1.36),
                       pad=3 if compact else round(size * 0.36), label_size=label_size,
                       lines=2 if size >= WRAP_MIN_PX and not compact else 1, dot_r=size * 0.42, with_time=with_time)
        widths = [sum(text_width(t, label_size, True) for t, _ in parts) + 4 * (len(parts) - 1)
                  for parts in (self._labels(i, with_time) for i in self.items) if parts]
        style.label_w = max(widths, default=0)
        return style

    def _labels(self, item: dict[str, Any], with_time: bool = True) -> list[tuple[str, str]]:
        done = bool(item.get("done"))
        parts = []
        if item.get("day"):
            parts.append((str(item["day"]), MUTED if done else TEXT_2))
        if with_time and item.get("time"):
            parts.append((str(item["time"]), MUTED if done else self.accent))
        return parts

    def _rows(self, c: Canvas, rows: list[_Row], left: float, right: float, top: float, style: _Style) -> None:
        y = top
        for n, row in enumerate(rows):
            if n:
                c.line([(left, y), (right, y)], BORDER, 1)
            self._row(c, row, left, y + style.pad, style)
            y += len(row.lines) * style.line_h + 2 * style.pad

    def _row(self, c: Canvas, row: _Row, left: float, top: float, style: _Style) -> None:
        item = row.item
        done = bool(item.get("done"))
        mid = top + style.line_h / 2
        r = style.dot_r
        if not style.ring:
            c.circle(left + r, mid, r, fill=OK if done else self.accent)
        elif done:
            c.circle(left + r, mid, r, fill=OK)
            c.check(left + r, mid, r, BG)
        else:
            c.circle(left + r, mid, r, outline=mix(self.accent, PANEL, 0.75), width=max(1.4, style.size * 0.09))
        x = left + 2 * r + 10
        for text, color in self._labels(item, style.with_time):
            x += c.text(x, mid, text, style.label_size, color, bold=True) + 4
        x = left + style.lead
        for k, line in enumerate(row.lines):
            y = mid + k * style.line_h
            w = c.text(x, y, line, style.size, MUTED if done else TEXT)
            if done and line:
                c.strike(x, x + w, y + 1, MUTED)

    def _columns(self, c: Canvas, items: list[dict[str, Any]]) -> None:
        """Too many for one column: two, one line each, days without their times; the last row
        counts what is left out."""
        style = self._style(COLUMN_PX, True, with_time=not any(i.get("day") for i in items))
        style.ring, style.dot_r = False, 3.5
        top = COMPACT_CARD_TOP
        row_h = style.line_h + 2 * style.pad
        per_column = int((CARD_BOTTOM - top - 8) // row_h)
        shown = items if len(items) <= 2 * per_column else items[: 2 * per_column - 1]
        c.box(MARGIN, top, WIDTH - MARGIN, CARD_BOTTOM, 14, PANEL, BORDER)
        mid_x = WIDTH / 2
        c.line([(mid_x, top + 10), (mid_x, CARD_BOTTOM - 10)], BORDER, 1)
        for column, (left, right) in enumerate(((MARGIN + PAD - 4, mid_x - 9), (mid_x + 9, WIDTH - MARGIN - PAD + 4))):
            part = shown[column * per_column : (column + 1) * per_column]
            rows = [_Row(i, [ellipsize(str(i.get("text", "")), right - left - style.lead, style.size)]) for i in part]
            self._rows(c, rows, left, right, top + 4, style)
            if column and len(shown) < len(items):
                y = top + 4 + (per_column - 1) * row_h + row_h / 2
                c.text((left + right) / 2, y, f"还有 {len(items) - len(shown)} 项…", style.size, MUTED, anchor="mm")

    def _hero(self, c: Canvas, item: dict[str, Any]) -> None:
        """A single memo (just noted down, or due): large."""
        c.box(MARGIN, CARD_TOP, WIDTH - MARGIN, CARD_BOTTOM, 14, PANEL, BORDER)
        left, right = MARGIN + PAD + 4, WIDTH - MARGIN - PAD - 4
        done = bool(item.get("done"))
        lines = wrap(str(item.get("text", "")), right - left, 20, 3)
        label = bool(item.get("time") or item.get("day"))
        block = (44 if label else 0) + len(lines) * 30
        y = CARD_TOP + (CARD_BOTTOM - CARD_TOP - block) / 2
        if label:
            x = left
            if item.get("time"):
                x += c.text(x, y + 16, str(item["time"]), 34, MUTED if done else self.accent, bold=True) + 12
            if item.get("day"):
                w = text_width(item["day"], 13, True) + 16
                c.box(x, y + 16 - 12, x + w, y + 16 + 12, 12, PANEL_2, BORDER)
                c.text(x + w / 2, y + 16, str(item["day"]), 13, TEXT_2, bold=True, anchor="mm")
            y += 44
        if done:
            c.circle(right - 11, CARD_TOP + 22, 11, fill=OK)
            c.check(right - 11, CARD_TOP + 22, 11, BG)
        for k, line in enumerate(lines):
            w = c.text(left, y + 15 + k * 30, line, 20, MUTED if done else TEXT)
            if done:
                c.strike(left, left + w, y + 16 + k * 30, MUTED)

    def _empty(self, c: Canvas, text: str) -> None:
        c.box(MARGIN, CARD_TOP, WIDTH - MARGIN, CARD_BOTTOM, 14, PANEL, BORDER)
        cx, cy = WIDTH / 2, CARD_TOP + 58
        icon = mix(self.accent, PANEL, 0.85)
        c.box(cx - 19, cy - 15, cx + 19, cy + 17, 6, mix(self.accent, PANEL, 0.12), icon)
        c.line([(cx - 19, cy - 5), (cx + 19, cy - 5)], icon, 1)
        for dx in (-9, 9):
            c.line([(cx + dx, cy - 20), (cx + dx, cy - 11)], icon, 2.2)
        c.check(cx, cy + 6, 10, OK)
        c.text(cx, cy + 46, text, 17, TEXT, bold=True, anchor="mm")
        c.text(cx, cy + 70, "说「提醒我……」就能记下", 12, MUTED, anchor="mm")

    # ---- clock ---------------------------------------------------------------------------------

    def _clock(self, c: Canvas) -> None:
        now = self._now()
        x = MARGIN + 6
        hours, minutes = f"{now:%H}", f"{now:%M}"
        size = 64
        x += c.text(x, 50, hours, size, TEXT, bold=True)
        x += c.text(x + 2, 46, ":", size * 0.9, self.accent, bold=True) + 4
        c.text(x, 50, minutes, size, TEXT, bold=True)
        c.text(WIDTH - MARGIN - 6, 30, f"{now.month}月{now.day}日", 16, TEXT_2, bold=True, anchor="rm")
        c.text(WIDTH - MARGIN - 6, 54, f"星期{WEEKDAYS[now.weekday()]}", 14, MUTED, anchor="rm")

        top, bottom = 100, CARD_BOTTOM
        c.box(MARGIN, top, WIDTH - MARGIN, bottom, 14, PANEL, BORDER)
        left, right = MARGIN + PAD, WIDTH - MARGIN - PAD
        c.text(left, top + 20, "接下来", 13, TEXT_2, bold=True)
        items = self.items
        if items:
            c.pill(right, top + 20, f"今天还有 {len(items)} 项", self.accent, 11)
        c.line([(left, top + 38), (right, top + 38)], BORDER, 1)
        if not items:
            c.text(WIDTH / 2, (top + 38 + bottom) / 2, "今天没有待办了", 15, MUTED, anchor="mm")
            return
        style = self._style(15, True)
        style.pad = 6
        fits = int((bottom - top - 42) // (style.line_h + 2 * style.pad))
        rows = [_Row(i, [ellipsize(str(i.get("text", "")), right - left - style.lead, style.size)]) for i in items[:fits]]
        self._rows(c, rows, left, right, top + 40, style)


class BoardPlayer:
    """The display loop's side: picks up requests and hands out the picture while a board is up."""

    def __init__(self, path: Path = BOARD_REQUEST_PATH, log=print) -> None:
        self._path = path
        self._log = log
        self._board: Board | None = None
        self._key: str | None = None
        self._image = None
        self.version = 0

    def poll(self, now: float) -> None:
        spec = take_board_request(self._path)
        if spec is None:
            return
        self._board, self._key = None, None
        if spec.get("kind") == "clear":
            return
        try:
            self._board = Board(spec, now)
        except Exception as exc:  # noqa: BLE001 - a bad board must not take the face down
            self._log(f"Display: cannot show board ({exc})")

    def picture(self, now: float):
        """(version, image 320x240, monotonic end) of the board up now, or None; the version
        changes with every new picture."""
        board = self._board
        if board is None or now >= board.end:
            self._board = None
            return None
        key = board.key()
        if key != self._key:
            try:
                self._image = board.render()
            except Exception as exc:  # noqa: BLE001
                self._log(f"Display: cannot draw board ({exc})")
                self._board = None
                return None
            self._key = key
            self.version += 1
        return self.version, self._image, board.end


def native_rgb565(image) -> bytes:
    """A 320x240 picture in the ESP32 panel's layout (face_render.h): panel row x, column 239 - y,
    big-endian RGB565."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint16)[::-1].transpose(1, 0, 2)
    value = ((rgb[..., 0] & 0xF8) << 8) | ((rgb[..., 1] & 0xFC) << 3) | (rgb[..., 2] >> 3)
    return value.astype(">u2").tobytes()


class Esp32Board:
    """Puts the player's pictures up on the ESP32 over MSG_IMAGE, a few KB per tick, sent again if
    the ESP32 does not confirm; takes them down when the board ends."""

    def __init__(self, log=print) -> None:
        self._log = log
        self._link = None
        self._version: int | None = None
        self._queue: list[bytes] = []
        self._crc = 0
        self._waiting_since: float | None = None
        self._tries = 0
        self._warned = False

    def tick(self, link, picture, now: float) -> None:
        if link is not self._link:
            self._link, self._version, self._queue, self._waiting_since = link, None, [], None
        if picture is None:
            if self._version is not None:
                link.image(bytes([IMAGE_HIDE]))
                self._version, self._queue, self._waiting_since = None, [], None
            return
        if not link.supports_image:
            if not self._warned and link.features is not None:
                self._log("Display: the ESP32 firmware cannot show memo boards; update it (make firmware-uart)")
                self._warned = True
            return
        version, image, end = picture
        if version != self._version:
            self._version, self._tries = version, 0
            self._start(image, end, now)
        elif self._waiting_since is not None and not self._queue:
            status = link.image_status
            if status == (True, self._crc):
                self._waiting_since = None
            elif status == (False, self._crc) or now - self._waiting_since > IMAGE_ACK_S:
                if self._tries < IMAGE_TRIES:
                    self._start(image, end, now)
                else:
                    self._log("Display: the ESP32 did not take the memo board")
                    self._waiting_since = None
        budget = IMAGE_BYTES_PER_TICK
        while self._queue and budget > 0:
            packet = self._queue.pop(0)
            link.image(packet)
            budget -= len(packet)
            if not self._queue:
                self._waiting_since = now

    def _start(self, image, end: float, now: float) -> None:
        self._tries += 1
        # A second spare: the Pi takes it down itself, the ESP32's timer only matters if the Pi goes away.
        self._crc, self._queue = image_packets(native_rgb565(image), end - now + 1.0)
        self._link.image_status = None
        self._waiting_since = now


DEMOS = {
    "list": {"kind": "list", "title": "今天", "subtitle": "11月8日 周日", "items": [
        {"time": "09:30", "text": "给妈妈打电话"},
        {"time": "15:00", "text": "和小王开会讨论机器人外壳的配色和材料"},
        {"text": "买牛奶和鸡蛋", "done": True},
        {"text": "取快递"},
    ]},
    "week": {"kind": "list", "title": "本周", "subtitle": "11月9日 周一 – 11月15日 周日", "items": [
        {"day": "明天", "time": "08:00", "text": "晨跑"},
        {"day": "明天", "time": "10:00", "text": "牙医复诊"},
        {"day": "后天", "text": "交季度报告"},
        {"day": "11/12", "time": "19:30", "text": "看电影《星际穿越》重映"},
        {"day": "11/13", "text": "给猫买猫粮", "done": True},
        {"day": "11/14", "time": "14:00", "text": "团队周会"},
        {"day": "11/15", "text": "大扫除"},
    ]},
    "many": {"kind": "list", "title": "近期", "subtitle": "逾期和未来 7 天", "items": [
        {"time": f"{8 + n:02d}:00", "text": t, "done": n in (1, 4)} for n, t in enumerate(
            ["晨会", "回邮件", "写周报", "午饭约小李", "取快递", "修机器人左腿舵机", "整理桌面", "健身", "买菜",
             "给植物浇水", "看书半小时", "备份照片", "订机票", "复习英语", "打扫厨房", "洗衣服", "冥想"])]},
    "busy": {"kind": "list", "title": "明天", "subtitle": "11月9日 周一", "items": [
        {"time": f"{9 + n}:{'30' if n % 2 else '00'}", "text": t} for n, t in enumerate(
            ["站会", "评审机器人步态策略", "午饭", "给客户回电话确认交付时间", "健身房", "写日报", "买菜"])]},
    "note": {"kind": "list", "title": "已记下", "subtitle": "11月8日 周日", "title_color": "green", "badge": "1 项",
             "items": [{"day": "明天", "time": "15:00", "text": "开会"}]},
    "remind": {"kind": "list", "title": "提醒", "subtitle": "11月8日 周日", "title_color": "pink",
               "items": [{"time": "15:00", "text": "和小王开会讨论机器人外壳"}]},
    "empty": {"kind": "list", "title": "后天", "subtitle": "11月10日 周二", "items": []},
    "clock": {"kind": "clock", "items": [{"time": "15:00", "text": "和小王开会"}, {"text": "取快递"},
                                         {"time": "20:00", "text": "给妈妈打电话"}]},
}


if __name__ == "__main__":
    import sys

    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp")
    for name, spec in DEMOS.items():
        path = out_dir / f"board_{name}.png"
        start = time.perf_counter()
        Board(spec, 0.0).render().save(path)
        print(path, f"{(time.perf_counter() - start) * 1000:.0f} ms")
