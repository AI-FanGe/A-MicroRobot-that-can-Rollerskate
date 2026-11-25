# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

"""The robot's calendar and notebook: dated memos (to-dos, appointments) and undated notes.

Everything the user asks the robot to remember lands here, from the voice chat, the text chat
or the web UI. The memos go into the chat and voice prompts, are read out and shown on the face
screen when asked about a day ("看一下今天要做的事"), and export as CSV, Markdown, JSON or iCalendar.
"""

from __future__ import annotations

import csv
import io
import json
import re
import threading
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from api.memory import new_id
from api.protocol import ApiError, CommonState, Reason
from api.store import DATA_DIR, read_json, write_json_atomic

MEMO_PATH = DATA_DIR / "memo.json"
MAX_MEMOS = 2000
MAX_TEXT = 200
DEFAULT_TIMEZONE = "Asia/Shanghai"
WEEKDAYS = "一二三四五六日"
# A memo due this long ago still rings when the API was not running at the time.
REMIND_GRACE_S = 10 * 60
UPCOMING_DAYS = 7
OVERDUE_DAYS = 7

_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TIME = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)$")

# What the user says: asking about a day's memos, or telling the robot something to note down.
# Words that only mean the memos: a question with one is about them.
QUERY_TOPIC = re.compile(r"待办|代办|日程|行程|备忘|(?:有|的|啥|什么|哪些)安排|计划表")
# Words that are as often just chat (沙利文是干嘛的, 你能干啥, 这件事情你怎么看): about the memos
# only with a day named, and asked about the user (我), not the robot (你今天干嘛了).
QUERY_TOPIC_CHAT = re.compile(
    r"要做的|要干的|要办的|做什么|干什么|做啥|干啥|干嘛|什么事|啥事|哪些事|安排|计划|提醒|事情|事儿|任务|约会"
)
NOT_QUERY = re.compile(r"(?:干嘛|干啥|做什么|干什么|做啥)的")
QUERY_ASK = re.compile(r"看|查|显示|列|说说|读|念|告诉|什么|啥|哪些|几件|多少|有没有|吗|嘛")
CAPTURE = re.compile(
    r"提醒我|记住|记一下|记下|记着|帮我记|别忘|不要忘|加[个一]|添加|做完|完成了|办完|搞定|取消|删掉|删除|不用了|改到|改成|推迟|提前"
)
# "明天要做的事情是买菜": telling, not asking.
STATEMENT = re.compile(r"(?:事情|事儿|安排|计划|任务|待办|日程|行程|要做的|要干的|要办的)(?:是|有|就是|[:：])")
# Said while a board is on the screen: take it down (checked before everything else, so
# "关掉提醒" does not show the reminders again).
DISMISS = re.compile(r"关掉|关了|关闭|收起|收掉|退出|拿掉|撤掉|隐藏|不看了|不用看了|别显示|不用显示|看完了|知道了|知道啦|"
                     r"明白了|收到|返回|回到表情|换回")
DISMISS_MAX_CHARS = 12
# Asking the time or date now, not when something is (你几点睡觉, 明天几点开会).
CLOCK_ASK = re.compile(
    r"(?:现在|这会儿?)(?:是)?(?:几点|什么时[间候]|啥时候)|几点(?:钟)?了|什么时[间候]了|"
    r"(?:今天|今儿|今日)(?:是)?(?:几号|几月几|星期几|周几|礼拜几|什么日子)|今天的?日期|(?:几号|星期几|周几|礼拜几)了"
)

_CN = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_NUM = r"(\d{1,2}|[一二两三四五六七八九十〇零]{1,3})"
_RELATIVE = {"大前天": -3, "前天": -2, "昨天": -1, "昨日": -1, "今天": 0, "今日": 0, "今儿": 0, "今晚": 0,
             "明天": 1, "明日": 1, "明儿": 1, "明晚": 1, "后天": 2, "大后天": 3}
_RELATIVE_RE = re.compile("|".join(sorted(_RELATIVE, key=len, reverse=True)))
_WEEKDAY_RE = re.compile(r"(上|下|这|本)?个?(?:周|星期|礼拜)([一二三四五六日天1-7])")
_WEEK_RE = re.compile(r"(上|下|这|本)个?(?:周|星期|礼拜)|周末")
_MONTH_DAY_RE = re.compile(_NUM + r"月" + _NUM + r"[号日]")
_DAY_RE = re.compile(r"(?<![月\d])" + _NUM + r"[号日]")
_ISO_RE = re.compile(r"(\d{4})[-/.年](\d{1,2})[-/.月](\d{1,2})")
_SLASH_RE = re.compile(r"(?<!\d)(\d{1,2})/(\d{1,2})(?!\d)")
_WEEKDAY_INDEX = {**{c: i for i, c in enumerate(WEEKDAYS)}, "天": 6, **{str(i + 1): i for i in range(7)}}


def local_now(tz: str = "") -> datetime:
    if tz:
        try:
            return datetime.now(ZoneInfo(tz))
        except (ZoneInfoNotFoundError, ValueError):
            pass
    return datetime.now().astimezone()


def valid_timezone(tz: str) -> bool:
    try:
        ZoneInfo(tz)
        return True
    except (ZoneInfoNotFoundError, ValueError):
        return False


def weekday(d: date) -> str:
    return "周" + WEEKDAYS[d.weekday()]


def day_name(d: date, today: date) -> str:
    """今天 / 明天 / … within three days of today, else 11月9日."""
    offset = (d - today).days
    for word in ("大前天", "前天", "昨天", "今天", "明天", "后天", "大后天"):
        if _RELATIVE[word] == offset:
            return word
    return f"{d.month}月{d.day}日" if d.year == today.year else f"{d.year}年{d.month}月{d.day}日"


def short_date(d: date) -> str:
    return f"{d.month}/{d.day}"


def _cn_int(text: str) -> int | None:
    if text.isdigit():
        return int(text)
    if "十" in text:
        tens, _, ones = text.partition("十")
        return (_CN.get(tens, 1) if tens else 1) * 10 + (_CN.get(ones, 0) if ones else 0)
    return _CN.get(text) if len(text) == 1 else None


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def parse_days(text: str, today: date) -> tuple[date, date, str] | None:
    """The day (or days) `text` talks about: (first, last, label), or None when it names none."""
    found = _ISO_RE.search(text)
    if found:
        d = _safe_date(*(int(g) for g in found.groups()))
        if d:
            return d, d, day_name(d, today)
    found = _MONTH_DAY_RE.search(text)
    if found:
        month, day = _cn_int(found.group(1)), _cn_int(found.group(2))
        d = _safe_date(today.year, month or 0, day or 0)
        if d:
            return d, d, day_name(d, today)
    found = _SLASH_RE.search(text)
    if found:
        d = _safe_date(today.year, int(found.group(1)), int(found.group(2)))
        if d:
            return d, d, day_name(d, today)
    found = _WEEKDAY_RE.search(text)
    if found:
        monday = today - timedelta(days=today.weekday())
        shift = {"上": -7, "下": 7}.get(found.group(1) or "", 0)
        d = monday + timedelta(days=_WEEKDAY_INDEX[found.group(2)] + shift)
        prefix = {"上": "上", "下": "下"}.get(found.group(1) or "", "")
        return d, d, f"{prefix}{weekday(d)}"
    found = _WEEK_RE.search(text)
    if found:
        monday = today - timedelta(days=today.weekday())
        if found.group(0) == "周末":
            return monday + timedelta(days=5), monday + timedelta(days=6), "周末"
        shift = {"上": -7, "下": 7}.get(found.group(1), 0)
        label = {"上": "上周", "下": "下周"}.get(found.group(1), "本周")
        return monday + timedelta(days=shift), monday + timedelta(days=shift + 6), label
    offsets = [_RELATIVE[m.group(0)] for m in _RELATIVE_RE.finditer(text)]
    if offsets:
        first, last = today + timedelta(days=min(offsets)), today + timedelta(days=max(offsets))
        label = day_name(first, today) if first == last else f"{day_name(first, today)}到{day_name(last, today)}"
        return first, last, label
    found = _DAY_RE.search(text)
    if found:
        d = _safe_date(today.year, today.month, _cn_int(found.group(1)) or 0)
        if d:
            return d, d, day_name(d, today)
    return None


def _intent(text: str, today: date) -> str | None:
    if CAPTURE.search(text):
        return "capture"
    clear = bool(QUERY_TOPIC.search(text))
    chat = bool(QUERY_TOPIC_CHAT.search(text)) and not NOT_QUERY.search(text)
    if not (clear or chat):
        return None
    day = parse_days(text, today) is not None
    if not QUERY_ASK.search(text) and STATEMENT.search(text):
        return "capture" if clear or day or "我" in text else None
    if clear:
        return "query"
    about_me = "我" in text or "你" not in text
    return "query" if about_me and day else None


def parse_query(text: str, today: date) -> tuple[date | None, date | None, str] | None:
    """`text` asks what is on (看一下前天要做的事情): the days, (None, None, "近期") for no day in particular."""
    if _intent(text, today) != "query":
        return None
    return parse_days(text, today) or (None, None, "近期")


def wants_capture(text: str, today: date | None = None) -> bool:
    """`text` may add, finish or cancel a memo: worth asking the model what exactly."""
    return _intent(text, today or date.today()) == "capture"


def wants_dismiss(text: str) -> bool:
    """A short "关掉 / 收起 / 知道了": only meaningful while a board is up."""
    bare = re.sub(r"[\s，。！？、,.!?~～]", "", text)
    return len(bare) <= DISMISS_MAX_CHARS and bool(DISMISS.search(bare))


def asks_clock(text: str) -> bool:
    return bool(CLOCK_ASK.search(text)) and not QUERY_TOPIC.search(text) and not QUERY_TOPIC_CHAT.search(text)


def _normalize_time(value: Any) -> str | None:
    value = str(value or "").strip().replace("：", ":")
    if not value:
        return None
    found = _TIME.match(value)
    if not found:
        raise ApiError("时间格式是 HH:MM，例如 15:30", CommonState.INVALID)
    return f"{int(found.group(1)):02d}:{found.group(2)}"


def _normalize_date(value: Any) -> str | None:
    value = str(value or "").strip()
    if not value:
        return None
    if not _DATE.match(value) or _safe_date(*map(int, value.split("-"))) is None:
        raise ApiError("日期格式是 YYYY-MM-DD，例如 2026-11-08", CommonState.INVALID)
    return value


def sort_key(m: dict[str, Any]) -> tuple:
    return (m.get("date") or "9999-99-99", m.get("time") or "99:99", m["created"])


def describe(m: dict[str, Any], today: date, with_day: bool = True) -> str:
    """One memo as the prompts and the screen read it: 明天 周日 15:00 开会（已完成）."""
    parts = []
    if with_day and m.get("date"):
        d = date.fromisoformat(m["date"])
        parts.append(f"{day_name(d, today)} {weekday(d)}")
    if m.get("time"):
        parts.append(m["time"])
    parts.append(m["text"])
    return " ".join(parts) + ("（已完成）" if m.get("done") else "")


def read_items(path: Path = MEMO_PATH) -> list[dict[str, Any]]:
    items = read_json(path, [])
    return items if isinstance(items, list) else []


def in_days(items: list[dict[str, Any]], first: date | None, last: date | None, today: date) -> list[dict[str, Any]]:
    """Memos on first..last; with no days given, the open ones overdue or in the coming week, then the notes."""
    if first is None or last is None:
        start, end = (today - timedelta(days=OVERDUE_DAYS)).isoformat(), (today + timedelta(days=UPCOMING_DAYS - 1)).isoformat()
        dated = [m for m in items if m.get("date") and start <= m["date"] <= end and not m.get("done")]
        notes = [m for m in items if not m.get("date") and not m.get("done")]
        return sorted(dated, key=sort_key) + sorted(notes, key=lambda m: -m["created"])[:10]
    lo, hi = first.isoformat(), last.isoformat()
    return sorted((m for m in items if m.get("date") and lo <= m["date"] <= hi), key=sort_key)


def board_spec(items: list[dict[str, Any]], first: date | None, last: date | None, label: str, today: date,
               title: str = "", seconds: float = 30) -> dict[str, Any]:
    """What the face screen shows for these memos (board.py draws it)."""
    rows = board_rows(items, today, with_day=first is None or first != last)
    if title:  # a note about these memos ("已记下", "提醒"), not a day's list
        return {"kind": "list", "title": title, "subtitle": long_date(today), "badge": f"{len(rows)} 项",
                "items": rows, "seconds": seconds}
    if first is None:
        title, subtitle = label, f"逾期和未来 {UPCOMING_DAYS} 天"
    elif first != last:
        title, subtitle = label, f"{long_date(first)} – {long_date(last)}"
    elif label in ("大前天", "前天", "昨天", "今天", "明天", "后天", "大后天"):
        title, subtitle = label, long_date(first)
    else:
        title, subtitle = f"{first.month}月{first.day}日", weekday(first)
    return {"kind": "list", "title": title, "subtitle": subtitle, "items": rows, "seconds": seconds}


def board_rows(items: list[dict[str, Any]], today: date, with_day: bool) -> list[dict[str, Any]]:
    rows = []
    for m in items:
        day = ""
        if with_day and m.get("date"):
            d = date.fromisoformat(m["date"])
            day = day_name(d, today) if abs((d - today).days) <= 2 else short_date(d)
        rows.append({"day": day, "time": m.get("time") or "", "text": m["text"], "done": bool(m.get("done"))})
    return rows


def long_date(d: date) -> str:
    return f"{d.month}月{d.day}日 {weekday(d)}"


class MemoStore:
    def __init__(self, path: Path = MEMO_PATH, timezone: Callable[[], str] = lambda: "",
                 on_change: Callable[[str, dict[str, Any]], None] = lambda action, item: None) -> None:
        self._path = path
        self._timezone = timezone
        self._on_change = on_change
        self._lock = threading.Lock()
        self._items: list[dict[str, Any]] = read_items(path)

    def now(self) -> datetime:
        return local_now(self._timezone())

    def today(self) -> date:
        return self.now().date()

    def _save(self) -> None:
        write_json_atomic(self._path, self._items)

    def all(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(m) for m in self._items]

    def list(self, start: str | None = None, end: str | None = None, query: str = "", status: str = "all",
             notes: bool = True) -> list[dict[str, Any]]:
        start, end = _normalize_date(start), _normalize_date(end)
        needle = (query or "").strip().lower()
        out = []
        for m in self.all():
            if m.get("date"):
                if (start and m["date"] < start) or (end and m["date"] > end):
                    continue
            elif not notes:
                continue
            if status == "open" and m.get("done") or status == "done" and not m.get("done"):
                continue
            if needle and needle not in m["text"].lower():
                continue
            out.append(m)
        return sorted(out, key=sort_key)

    def get(self, memo_id: str) -> dict[str, Any]:
        with self._lock:
            for m in self._items:
                if m["id"] == memo_id:
                    return dict(m)
        raise ApiError("没有这条备忘", CommonState.INVALID, Reason.NOT_FOUND, 404)

    def add(self, text: str, day: str | None = None, at: str | None = None, source: str = "manual") -> dict[str, Any]:
        text = re.sub(r"\s+", " ", str(text or "")).strip()[:MAX_TEXT]
        if not text:
            raise ApiError("备忘内容不能为空", CommonState.INVALID)
        day, at = _normalize_date(day), _normalize_time(at)
        if at and not day:
            day = self.today().isoformat()
        now = time.time()
        with self._lock:
            memo = next((m for m in self._items if m["text"] == text and m.get("date") == day and m.get("time") == at), None)
            if memo is not None:
                memo["done"], memo["updated"] = False, now
            else:
                memo = {"id": new_id("memo"), "text": text, "date": day, "time": at, "done": False, "source": source,
                        "created": now, "updated": now, "reminded": False}
                self._items.append(memo)
                if len(self._items) > MAX_MEMOS:
                    # The oldest finished memos go first.
                    self._items.sort(key=lambda m: (not m.get("done"), m["updated"]))
                    del self._items[0]
            self._save()
            memo = dict(memo)
        self._on_change("add", memo)
        return memo

    def update(self, memo_id: str, values: dict[str, Any]) -> dict[str, Any]:
        changes: dict[str, Any] = {}
        if "text" in values:
            text = re.sub(r"\s+", " ", str(values["text"] or "")).strip()[:MAX_TEXT]
            if not text:
                raise ApiError("备忘内容不能为空", CommonState.INVALID)
            changes["text"] = text
        if "date" in values:
            changes["date"] = _normalize_date(values["date"])
        if "time" in values:
            changes["time"] = _normalize_time(values["time"])
        if "done" in values:
            changes["done"] = bool(values["done"])
        if "date" in changes or "time" in changes:
            changes["reminded"] = False
        today = self.today().isoformat()
        with self._lock:
            memo = next((m for m in self._items if m["id"] == memo_id), None)
            if memo is None:
                raise ApiError("没有这条备忘", CommonState.INVALID, Reason.NOT_FOUND, 404)
            memo.update(changes, updated=time.time())
            if memo.get("time") and not memo.get("date"):
                memo["date"] = today
            self._save()
            memo = dict(memo)
        self._on_change("update", memo)
        return memo

    def delete(self, memo_id: str) -> dict[str, Any]:
        with self._lock:
            memo = next((m for m in self._items if m["id"] == memo_id), None)
            if memo is None:
                raise ApiError("没有这条备忘", CommonState.INVALID, Reason.NOT_FOUND, 404)
            self._items.remove(memo)
            self._save()
        self._on_change("delete", memo)
        return memo

    def clear_done(self) -> int:
        with self._lock:
            before = len(self._items)
            self._items = [m for m in self._items if not m.get("done")]
            self._save()
            count = before - len(self._items)
        self._on_change("clear", {})
        return count

    def due(self) -> list[dict[str, Any]]:
        """Memos whose time has just come, each returned once."""
        now = self.now()
        out, changed = [], False
        with self._lock:
            for m in self._items:
                if m.get("done") or m.get("reminded") or not m.get("date") or not m.get("time"):
                    continue
                at = datetime.fromisoformat(f"{m['date']}T{m['time']}").replace(tzinfo=now.tzinfo)
                late = (now - at).total_seconds()
                if late >= 0:
                    m["reminded"] = changed = True
                    if late <= REMIND_GRACE_S:
                        out.append(dict(m))
            if changed:
                self._save()
        return out

    # ---- prompts ---------------------------------------------------------------------------

    def for_prompt(self, limit_chars: int = 600) -> str:
        """Overdue and coming memos plus the notes, a line each, for the chat and voice prompts."""
        today = self.today()
        items = in_days(self.all(), None, None, today)
        lines, used = [], 0
        for m in items:
            line = describe(m, today)
            if m.get("date") and m["date"] < today.isoformat():
                line += "（过期未完成）"
            elif not m.get("date"):
                line = "随手记：" + line
            if used + len(line) > limit_chars:
                break
            lines.append(line)
            used += len(line) + 1
        return "\n".join(lines)

    def days_text(self, first: date | None, last: date | None, label: str) -> str:
        today = self.today()
        items = in_days(self.all(), first, last, today)
        if not items:
            return f"{label}没有安排。"
        return "\n".join(describe(m, today, with_day=first is None or first != last) for m in items)

    # ---- export ----------------------------------------------------------------------------

    def export(self, fmt: str) -> tuple[bytes, str, str]:
        """(body, content type, file name) of every memo."""
        items = sorted(self.all(), key=sort_key)
        stamp = self.now().strftime("%Y%m%d")
        if fmt == "json":
            body = json.dumps(items, ensure_ascii=False, indent=2).encode()
            return body, "application/json; charset=utf-8", f"memo-{stamp}.json"
        if fmt == "csv":
            out = io.StringIO()
            writer = csv.writer(out)
            writer.writerow(["日期", "星期", "时间", "事项", "状态", "来源", "记录时间"])
            for m in items:
                d = date.fromisoformat(m["date"]) if m.get("date") else None
                writer.writerow([m.get("date") or "", weekday(d) if d else "", m.get("time") or "", m["text"],
                                 "已完成" if m.get("done") else "未完成", SOURCE_LABEL.get(m.get("source", ""), m.get("source", "")),
                                 datetime.fromtimestamp(m["created"]).strftime("%Y-%m-%d %H:%M")])
            # The BOM makes Excel read it as UTF-8.
            return ("\ufeff" + out.getvalue()).encode(), "text/csv; charset=utf-8", f"memo-{stamp}.csv"
        if fmt == "md":
            lines = [f"# 备忘录（导出于 {self.now():%Y-%m-%d %H:%M}）"]
            current = None
            for m in items:
                if m.get("date") != current:
                    current = m.get("date")
                    heading = f"{current} {weekday(date.fromisoformat(current))}" if current else "随手记"
                    lines += ["", f"## {heading}", ""]
                lines.append(f"- [{'x' if m.get('done') else ' '}] {m['time'] + ' ' if m.get('time') else ''}{m['text']}")
            return ("\n".join(lines).strip() + "\n").encode(), "text/markdown; charset=utf-8", f"memo-{stamp}.md"
        if fmt == "ics":
            return self._ics(items), "text/calendar; charset=utf-8", f"memo-{stamp}.ics"
        raise ApiError("format 只能是 csv / md / json / ics", CommonState.INVALID)

    def _ics(self, items: list[dict[str, Any]]) -> bytes:
        def escape(text: str) -> str:
            return text.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", "\\n")

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        tz = self._timezone() or ""
        lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//xiaoxiaobai//memo//ZH", "CALSCALE:GREGORIAN"]
        for m in items:
            uid = f"{m['id']}@xiaoxiaobai"
            if m.get("date"):
                day = m["date"].replace("-", "")
                summary = ("[已完成] " if m.get("done") else "") + m["text"]
                lines += ["BEGIN:VEVENT", f"UID:{uid}", f"DTSTAMP:{stamp}", f"SUMMARY:{escape(summary)}"]
                if m.get("time"):
                    start = f"{day}T{m['time'].replace(':', '')}00"
                    at = datetime.strptime(start, "%Y%m%dT%H%M%S") + timedelta(minutes=30)
                    zone = f";TZID={tz}" if tz else ""
                    lines += [f"DTSTART{zone}:{start}", f"DTEND{zone}:{at:%Y%m%dT%H%M%S}"]
                else:
                    end = (date.fromisoformat(m["date"]) + timedelta(days=1)).strftime("%Y%m%d")
                    lines += [f"DTSTART;VALUE=DATE:{day}", f"DTEND;VALUE=DATE:{end}"]
                lines.append("END:VEVENT")
            else:
                lines += ["BEGIN:VTODO", f"UID:{uid}", f"DTSTAMP:{stamp}", f"SUMMARY:{escape(m['text'])}",
                          f"STATUS:{'COMPLETED' if m.get('done') else 'NEEDS-ACTION'}", "END:VTODO"]
        lines.append("END:VCALENDAR")
        return ("\r\n".join(lines) + "\r\n").encode()


SOURCE_LABEL = {"manual": "网页", "chat": "文字聊天", "voice": "语音", "auto": "自动"}


def voice_context(heard: str, tz: str = "", path: Path = MEMO_PATH) -> str:
    """For the voice chat: the time when the user asks it, and the memos of the day they ask about."""
    now = local_now(tz)
    today = now.date()
    query = parse_query(heard, today)
    if query is None and not asks_clock(heard):
        return ""
    text = f"\n\n[现在时间] {now.year}年{now.month}月{now.day}日 {weekday(today)} {now:%H:%M}。"
    if query is not None:
        first, last, label = query
        items = in_days(read_items(path), first, last, today)
        listed = "\n".join(describe(m, today, with_day=first is None or first != last) for m in items[:12])
        text += (
            f"\n[备忘录·{label}]（用户在问{label}的安排，清单同时显示在你脸上的屏幕上。照着下面说，口语化，"
            "一件一件简短地说，最多说五件，多了就说还有几件；已完成的提一句就行；这里没有就说没有安排，不要编）\n"
            + (listed or f"{label}没有安排。")
        )
    return text
