# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

"""Music from NetEase Cloud Music on the robot's speaker: search, a play queue, voice requests
("放一首晴天"), and the NetEase account login (QR code, SMS code or cookie).

Songs play through src/music_player.py on the voice bridge. The bridge has one client, and
the voice chat would hear the music and talk over it, so a running voice chat is let finish
its reply, stopped for the music and started again when the music ends (setting
music.resume_voice). The player's beat grid goes to the motion controller, so that in
「走跑·上肢自由」 the arms and head dance to the song (settings music.groove / groove_size /
groove_offset_ms).
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import subprocess
import threading
import time
import urllib.parse
import wave
from pathlib import Path
from typing import Any

from api.backend import REPO, Backend
from api.events import EventBus
from api.local_player import LocalPlayer
from api.netease import (
    RISKY_NETWORK,
    SECURE_CAPTCHA,
    SLIDER_CAPTCHA_IDS,
    NeteaseClient,
    NeteaseError,
)
from api.protocol import ApiError, CommonState, Reason, response
from api.qr import qr_svg
from api.store import CONFIG_DIR, Settings, read_json, write_json_atomic

COOKIE_PATH = CONFIG_DIR / "netease_cookies.json"
QUEUE_MAX = 200
# A queue only needs enough playable songs; checking more costs a slower start.
URL_CHECK_MAX = 60
URL_TTL_S = 15 * 60
ACCOUNT_TTL_S = 600
VOICE_REPLY_WAIT_S = 15.0
VOICE_TAIL_S = 1.2
MAX_SKIPS = 8

# ---------------------------------------------------------------- what was said

_END = r"[\s,，。.!！?？~～]*"
STOP_RE = re.compile(
    r"(别|不要|不用|甭)再?(放|播|唱)(了|歌|音乐|$)|停(止|掉|下|一下)?(播放|放歌|音乐|歌|唱)|"
    r"(关|关掉|关闭)(一下)?(音乐|歌|播放)|不听了|别唱了|音乐停")
PAUSE_RE = re.compile(r"暂停(一下)?(播放|音乐|歌)?" + _END + "$|^(先)?停一下" + _END + "$")
RESUME_RE = re.compile(r"继续(播放|放|唱|听)|接着(放|播|唱|听)|恢复播放")
NEXT_RE = re.compile(r"下一首|换一首|换首|切歌|切一首|跳过这首|不好听")
PREV_RE = re.compile(r"上一首|前一首")
LOUDER_RE = re.compile(r"(大声|声音大|音量(调)?大|调大(音量|声音)|响一点|开大)")
QUIETER_RE = re.compile(r"(小声|声音小|音量(调)?小|调小(音量|声音)|轻一点|关小)")
PLAY_RE = re.compile(
    r"(?:给我|帮我|替我|你|请|再)?"
    r"(?:播放|放一?首|放点|来一?首|来点|唱一?首|我想听|想听|我要听|听一?首|点一?首|点播)"
    r"(?P<q>.*)")
# "放周杰伦的歌", "放个歌": a bare 放 ("把手放下") only counts with a song word after it.
PLAIN_PLAY_RE = re.compile(r"(?:给我|帮我|你|请|再)?放(?:一下|个|些)?(?P<q>.*?(?:歌|音乐|曲子)).*")
NOT_SONGS = re.compile(r"故事|笑话|你说|你讲|新闻|天气|广播|评书|相声|小说|声音|回答|解释|意见|看法|视频|电影|电视|动作|姿势|表情")
LEAD_FILLER = re.compile(r"^(一首|首|一下|下|一点|点|一个|个|一些|些|一曲|歌曲|歌|音乐)+")
TAIL_FILLER = re.compile(r"(吧|呗|啊|呀|嘛|好吗|好不好|可以吗|行吗|给我听|听听|来听|听|哈|了)*" + _END + "$")
GENERIC = re.compile(r"^(什么|点什么|随便|随便什么|随便来|一首|首|好听|好听点|热门|流行|最火|最近流行|最新|新|"
                     r"什么歌|歌|音乐|曲子)?(的|点的)?(歌|音乐|歌曲|曲子)?$")
LIKED = re.compile(r"我喜欢的|我收藏的|红心|我的歌单|喜欢的音乐|收藏的歌")
DAILY = re.compile(r"每日推荐|今日推荐|推荐的歌|推荐一首|给我推荐")


def song_query(text: str) -> str | None:
    """What to search for in a request to play something, "" for anything; None when it is not one."""
    found = PLAY_RE.search(text) or PLAIN_PLAY_RE.search(text)
    if found is None:
        return None
    query = found.group("q").strip()
    if NOT_SONGS.search(query):
        return None
    query = LEAD_FILLER.sub("", query)
    query = TAIL_FILLER.sub("", query).strip()
    query = re.sub(r"(这首歌|那首歌|这首|那首|的歌曲|的歌|歌曲)$", "", query)
    query = re.sub(r"(唱|演唱)?的", " ", query).strip() if re.search(r".的.", query) else query.rstrip("的")
    query = re.sub(r"[《》“”\"'「」]", " ", query)
    query = re.sub(r"\s+", " ", query).strip()
    return "" if GENERIC.match(query) else query


def parse_intent(text: str, active: bool) -> tuple[str, str] | None:
    """(action, query) for a music request; controls only count while music is on."""
    text = (text or "").strip()
    if not text:
        return None
    if active:
        for action, pattern in (("stop", STOP_RE), ("pause", PAUSE_RE), ("next", NEXT_RE), ("previous", PREV_RE),
                                ("louder", LOUDER_RE), ("quieter", QUIETER_RE)):
            if pattern.search(text):
                return action, ""
    if RESUME_RE.search(text):
        return ("resume", "") if active else None
    query = song_query(text)
    if query is None:
        return None
    if LIKED.search(text):
        return "liked", ""
    if DAILY.search(text):
        return "daily", ""
    return ("hot", "") if not query else ("play", query)


# ---------------------------------------------------------------- service

class MusicService:
    def __init__(self, backend: Backend, events: EventBus, settings: Settings, interaction: Any, voice: Any,
                 cookie_path: Path = COOKIE_PATH) -> None:
        self.backend = backend
        self.events = events
        self.settings = settings
        self.interaction = interaction
        self.voice = voice
        self.motion: Any = None  # set by the app: takes the beat grid (set_groove)
        self._cookie_path = cookie_path
        saved = read_json(cookie_path, {}).get("cookies") or {}
        self.client = NeteaseClient(saved)
        self._lock = threading.RLock()
        # Pending -462 verifications: their key (verify token or QR code) -> (since, the login step
        # to retry once verified, the challenge, the session it came with).
        self._verify: dict[str, tuple[float, Any, dict[str, Any], str]] = {}
        self._qr_flow = NeteaseClient.QR_FLOWS[0]
        self._queue: list[dict[str, Any]] = []
        self._index = -1
        self._label = ""
        self._state = "idle"  # idle | loading | playing | paused | error
        self._position = 0.0
        self._error: str | None = None
        self._notice: str | None = None
        self._proc: subprocess.Popen | None = None
        # Called once with time.time() the first sample is heard, or None if it never plays
        # (the dance waits on it: MotionController._dance_music_started).
        self._on_started: Any = None
        self._generation = 0
        self._voice_paused = False
        self._urls: dict[int, tuple[float, dict[str, Any]]] = {}
        self._account: tuple[float, dict[str, Any]] = (0.0, {})
        self._last_progress_event = 0.0
        if "deviceId" not in saved:
            self._save_cookies()  # keep the device id the client made up

    # ---- state ----------------------------------------------------------------------------

    def cfg(self) -> dict[str, Any]:
        return self.settings.section("music")

    @property
    def active(self) -> bool:
        return self._state in ("loading", "playing", "paused")

    def status(self) -> dict[str, Any]:
        with self._lock:
            track = self._queue[self._index] if 0 <= self._index < len(self._queue) else None
            return {
                "state": self._state,
                "track": track,
                "index": self._index,
                "queue_length": len(self._queue),
                "source": self._label,
                "position_s": round(self._position, 1),
                "volume": int(self.cfg().get("volume", 60)),
                "error": self._error,
                "notice": self._notice,
                "voice_paused": self._voice_paused,
                "ffmpeg": self.backend.mock or bool(os.environ.get("XIAOXIAOBAI_FFMPEG") or shutil.which("ffmpeg")),
                "logged_in": self.client.logged_in,
            }

    def queue(self) -> dict[str, Any]:
        with self._lock:
            return {"index": self._index, "source": self._label, "tracks": list(self._queue)}

    def _publish(self, **extra: Any) -> None:
        s = self.status()
        self.events.publish("music", state=s["state"], track=s["track"], index=s["index"], position_s=s["position_s"],
                            error=s["error"], notice=s["notice"], **extra)

    def _set(self, state: str | None = None, **fields: Any) -> None:
        with self._lock:
            if state is not None:
                self._state = state
            for key, value in fields.items():
                setattr(self, f"_{key}", value)
        self._publish()

    # ---- netease ----------------------------------------------------------------------------

    def _call(self, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except NeteaseError as exc:
            reason = Reason.NOT_CONFIGURED if exc.code == 301 else Reason.UPSTREAM
            raise ApiError(str(exc), CommonState.FAILURE, reason, 502) from None

    def _save_cookies(self) -> None:
        write_json_atomic(self._cookie_path, {"cookies": self.client.cookies, "saved": time.time()}, private=True)
        self._account = (0.0, {})

    def _check_urls(self, tracks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Marks playable / trial on the first URL_CHECK_MAX tracks (one request) and keeps the URLs."""
        head = tracks[:URL_CHECK_MAX]
        try:
            urls = self.client.song_urls([t["id"] for t in head], str(self.cfg().get("quality", "exhigh")))
        except NeteaseError:
            return tracks
        now = time.time()
        for t in head:
            info = urls.get(t["id"])
            if info is None:
                continue
            t["playable"] = bool(info.get("url"))
            t["trial"] = bool(info.get("trial"))
            if info.get("url"):
                self._urls[t["id"]] = (now, info)
        return tracks

    def _url(self, track_id: int) -> dict[str, Any]:
        cached = self._urls.get(track_id)
        if cached and time.time() - cached[0] < URL_TTL_S:
            return cached[1]
        info = self.client.song_url(track_id, str(self.cfg().get("quality", "exhigh")))
        if info.get("url"):
            self._urls[track_id] = (time.time(), info)
        return info

    def search(self, query: str, limit: int = 20) -> dict[str, Any]:
        query = (query or "").strip()
        if not query:
            raise ApiError("搜索词不能为空", CommonState.INVALID)
        tracks = self._call(self.client.search, query[:60], max(1, min(50, limit)))
        return {"query": query, "tracks": self._check_urls(tracks)}

    def playlist(self, kind: str) -> tuple[list[dict[str, Any]], str, str | None]:
        """(tracks, label, notice) of hot / daily / liked."""
        if kind == "liked":
            if not self.client.logged_in:
                raise ApiError("「我喜欢的音乐」要先登录网易云账号", CommonState.FAILURE, Reason.NOT_CONFIGURED)
            tracks = self._call(self.client.liked)
            random.shuffle(tracks)
            return tracks, "我喜欢的音乐", None
        if kind == "daily":
            if self.client.logged_in:
                return self._call(self.client.daily), "每日推荐", None
            tracks = self._call(self.client.hot, 50)
            random.shuffle(tracks)
            return tracks, "热歌榜", "每日推荐要登录网易云账号，先放热歌榜"
        tracks = self._call(self.client.hot, 50)
        random.shuffle(tracks)
        return tracks, "热歌榜", None

    # ---- account --------------------------------------------------------------------------

    def account(self, refresh: bool = False) -> dict[str, Any]:
        ts, cached = self._account
        if not refresh and cached and time.time() - ts < ACCOUNT_TTL_S:
            return cached
        try:
            info = self.client.account()
        except NeteaseError as exc:
            return {"logged_in": self.client.logged_in, "error": str(exc)}
        self._account = (time.time(), info)
        return info

    def qr_start(self) -> dict[str, Any]:
        key = self._call(self.client.qr_key, self._qr_flow)
        url = self.client.qr_url(key)
        return {"key": key, "url": url, "svg": qr_svg(url), "flow": self._qr_flow}

    def _qr_refused(self, flow: str) -> str:
        """Risk control refused this flow: the next QR code tries the other one."""
        flows = NeteaseClient.QR_FLOWS
        self._qr_flow = flows[(flows.index(flow) + 1) % len(flows)] if flow in flows else flows[0]
        return "网页版" if self._qr_flow == "web" else "客户端版"

    def qr_check(self, key: str) -> dict[str, Any]:
        if not key:
            raise ApiError("缺少 key", CommonState.INVALID)
        state = self._call(self.client.qr_check, key)
        messages = {800: "二维码过期了，点「刷新二维码」", 801: "等待扫码", 802: "已扫码，请在手机上确认登录", 803: "登录成功"}
        state["text"] = messages.get(state.get("code"), state.get("message") or f"状态 {state.get('code')}")
        challenge = state.pop("challenge", None)
        session = state.pop("session", "pc")
        code = state.get("code")
        if code == 803:
            self._qr_done(state)
        elif code in messages:
            pass
        elif challenge and (verify := self._challenge(challenge, lambda: self._qr_retry(key), session)):
            state["verify"] = verify
            state["text"] = "网易云要先做一次安全验证"
        else:
            other = self._qr_refused(str(state.get("flow")))
            why = "要网页验证码" if code == SECURE_CAPTCHA else (state.get("message") or "风控")
            state["text"] = (f"网易云拒绝了这次扫码登录（{code}：{why}）。再点「刷新二维码」会换{other}二维码试试，"
                             "也可以用验证码登录")
        return state

    def _qr_done(self, state: dict[str, Any]) -> None:
        self._save_cookies()
        state["account"] = self.account(refresh=True)
        self.events.publish("music_account", logged_in=True)

    def _qr_retry(self, key: str) -> dict[str, Any]:
        """The QR code checked again: still waiting (801 / 802), the web UI goes on polling it."""
        state = self.qr_check(key)
        return response(state, message="登录成功" if state.get("code") == 803 else state.get("text", ""))

    def sms_send(self, phone: str, country: str = "86") -> dict[str, Any]:
        phone = re.sub(r"\D", "", phone or "")
        if not 5 <= len(phone) <= 15:
            raise ApiError("手机号不对", CommonState.INVALID)
        country = country or "86"

        def send() -> dict[str, Any]:
            self.client.sms_send(phone, country)
            return response({"sent": True}, message="验证码已发送")
        return self._guarded(send)

    def sms_login(self, phone: str, code: str, country: str = "86") -> dict[str, Any]:
        phone = re.sub(r"\D", "", phone or "")
        code = re.sub(r"\D", "", code or "")
        if not phone or not code:
            raise ApiError("手机号和验证码都要填", CommonState.INVALID)
        country = country or "86"

        def login() -> dict[str, Any]:
            self.client.sms_login(phone, code, country)
            self._save_cookies()
            self.events.publish("music_account", logged_in=True)
            return response(self.account(refresh=True), message="登录成功")
        return self._guarded(login)

    # ---- the -462 risk-control verification ----

    def _guarded(self, step) -> dict[str, Any]:
        """Runs a login step; refused with -462, answers with the verification to do instead
        (data.verify) and runs the step again once it is done."""
        try:
            return step()
        except NeteaseError as exc:
            verify = exc.challenge and self._challenge(exc.challenge, lambda: self._guarded(step), exc.session)
            if not verify:
                message = str(exc)
                if exc.code == RISKY_NETWORK:
                    message += "。这是网易云按机器人的网络环境拦的：等一会儿再试、把机器人连到手机热点，或用「Cookie 登录」"
                raise ApiError(message, CommonState.FAILURE, Reason.UPSTREAM, 502) from None
            return response({"verify": verify}, message="网易云要先做一次安全验证")

    def _challenge(self, challenge: dict[str, Any], retry, session: str) -> dict[str, Any] | None:
        """The verification for a -462: kind "slider" (NECaptcha, solved in the web UI and handed to
        verify_slider), or "qr" (scanned with the NetEase app); None when there is none to do."""
        text = challenge.get("blockText") or "验证成功后，可进行下一步操作"
        verify_type = int(challenge.get("verifyType") or 0)
        if verify_type in SLIDER_CAPTCHA_IDS:
            key = str(challenge["verifyToken"])
            verify = {"key": key, "kind": "slider", "captcha_id": SLIDER_CAPTCHA_IDS[verify_type], "text": text}
        else:
            try:
                qr = self.client.verify_qr(challenge)
            except NeteaseError:
                return None
            key = qr["qr_code"]
            verify = {"key": key, "kind": "qr", "svg": qr_svg(qr["url"]), "url": qr["url"], "text": text}
        now = time.time()
        with self._lock:
            self._verify = {k: v for k, v in self._verify.items() if now - v[0] < 600}
            self._verify[key] = (now, retry, challenge, session)
        return verify

    def _pending(self, key: str) -> tuple[float, Any, dict[str, Any], str]:
        with self._lock:
            pending = self._verify.get(key)
        if pending is None:
            raise ApiError("这次安全验证已经过期，请重新操作", CommonState.INVALID)
        return pending

    def _retry(self, key: str, pending: tuple[float, Any, dict[str, Any], str]) -> dict[str, Any]:
        with self._lock:
            self._verify.pop(key, None)
        try:
            return pending[1]()
        except (NeteaseError, ApiError) as exc:
            return {"error": str(exc)}

    def verify_check(self, key: str) -> dict[str, Any]:
        """QR verifications: status 0 waiting, 10 scanned, 20 verified (the refused step is run
        again: "retry"), 21 expired."""
        pending = self._pending(key)
        status = self._call(self.client.verify_status, key)
        texts = {0: "用网易云 App 扫码验证", 10: "已扫码，请在手机上完成验证", 20: "验证通过", 21: "验证码过期了，请重新操作"}
        out: dict[str, Any] = {"status": status, "text": texts.get(status, f"状态 {status}")}
        if status == 21:
            with self._lock:
                self._verify.pop(key, None)
        if status == 20:
            out["retry"] = self._retry(key, pending)
        return out

    def verify_slider(self, key: str, validate: str) -> dict[str, Any]:
        """A slider solved in the web UI: NetEase checks it, then the refused step runs again ("retry")."""
        pending = self._pending(key)
        validate = str(validate or "").strip()
        if not validate:
            raise ApiError("缺少 validate", CommonState.INVALID)
        if not self._call(self.client.verify_slider, pending[2], validate, pending[3]):
            raise ApiError("网易云没认这次滑块验证，请再拖一次", CommonState.FAILURE, Reason.UPSTREAM, 502)
        return {"passed": True, "retry": self._retry(key, pending)}

    def cookie_login(self, text: str) -> dict[str, Any]:
        before = dict(self.client.cookies)
        try:
            self.client.use_cookie(text)
        except NeteaseError as exc:
            raise ApiError(str(exc), CommonState.INVALID) from None
        info = self.account(refresh=True)
        if not info.get("logged_in"):
            self.client.cookies = before
            raise ApiError("这个 Cookie 登录不上（无效或已过期）", CommonState.FAILURE, Reason.NOT_CONFIGURED)
        self._save_cookies()
        self.events.publish("music_account", logged_in=True)
        return response(info, message="登录成功")

    def logout(self) -> dict[str, Any]:
        self.client.logout()
        self._save_cookies()
        self.events.publish("music_account", logged_in=False)
        return response({"logged_in": False}, message="已退出网易云账号")

    # ---- playing --------------------------------------------------------------------------

    def play(self, body: dict[str, Any], origin: str = "app") -> dict[str, Any]:
        """{"query"} searches and plays the results; {"playlist": hot|daily|liked}; {"tracks", "index"}
        plays a list the web UI already has (search results); {"index"} jumps within the queue."""
        if not self.cfg().get("enabled", True):
            raise ApiError("音乐功能已关闭（设置 → 音乐）", CommonState.FAILURE, Reason.NOT_CONFIGURED)
        notice = None
        index = int(body.get("index") or 0)
        if body.get("query"):
            query = str(body["query"]).strip()
            tracks = self.search(query)["tracks"]
            label = f"搜索「{query}」"
            if not tracks:
                raise ApiError(f"网易云上没搜到「{query}」", CommonState.FAILURE, Reason.NOT_FOUND)
        elif body.get("playlist"):
            kind = str(body["playlist"])
            if kind not in ("hot", "daily", "liked"):
                raise ApiError("playlist 只能是 hot / daily / liked", CommonState.INVALID)
            tracks, label, notice = self.playlist(kind)
            tracks = self._check_urls(tracks)
            tracks = [t for t in tracks if t.get("playable") is not False] or tracks
        elif isinstance(body.get("tracks"), list):
            tracks = [self._clean_track(t) for t in body["tracks"][:QUEUE_MAX] if isinstance(t, dict) and t.get("id")]
            label = str(body.get("source") or "我点的歌")[:40]
        elif "index" in body:
            with self._lock:
                tracks, label = list(self._queue), self._label
        else:
            raise ApiError("要给 query、playlist、tracks 或 index", CommonState.INVALID)
        if not tracks:
            raise ApiError("列表是空的", CommonState.FAILURE, Reason.NOT_FOUND)
        index = max(0, min(len(tracks) - 1, index))
        playable = [t for t in tracks if t.get("playable", True)]
        if not playable:
            hint = "登录有黑胶 VIP 的网易云账号后再试" if not self.client.logged_in else "这些歌网易云不让播（版权或 VIP）"
            raise ApiError(f"找到了「{tracks[0]['name']}」，但播不了：{hint}", CommonState.FAILURE, Reason.NOT_FOUND)
        if not tracks[index].get("playable", True):
            index = tracks.index(playable[0])
        self._start(tracks[:QUEUE_MAX], index, label, notice, origin)
        return response(self.status(), state=CommonState.RUNNING, message=f"开始播放《{tracks[index]['name']}》")

    @staticmethod
    def _clean_track(t: dict[str, Any]) -> dict[str, Any]:
        out = {"id": int(t["id"]), "name": str(t.get("name") or "")[:80], "artist": str(t.get("artist") or "")[:80],
               "album": str(t.get("album") or "")[:80], "cover": str(t.get("cover") or "")[:300],
               "duration_s": float(t.get("duration_s") or 0), "vip": bool(t.get("vip"))}
        if "playable" in t:
            out["playable"] = bool(t["playable"])
        if "trial" in t:
            out["trial"] = bool(t["trial"])
        return out

    def play_dance(self, path: Path, on_started: Any, label: str = "跳舞") -> None:
        """Plays the dance soundtrack (a local file, no beat analysis); on_started as _on_started."""
        with wave.open(str(path), "rb") as w:
            duration = w.getnframes() / w.getframerate()
        track = {"id": 0, "name": f"{label}配乐", "artist": "Xiaoxiaobai", "album": "", "cover": "",
                 "duration_s": round(duration, 1), "vip": False, "local": path.resolve().as_uri()}
        self._start([track], 0, label, None, "dance", on_started=on_started)

    def _fire_started(self, heard_at: float | None) -> None:
        with self._lock:
            callback, self._on_started = self._on_started, None
        if callback is not None:
            callback(heard_at)

    def _start(self, tracks: list[dict[str, Any]], index: int, label: str, notice: str | None, origin: str,
               on_started: Any = None) -> None:
        self._fire_started(None)
        with self._lock:
            self._generation += 1
            generation = self._generation
            self._stop_proc()
            self._queue, self._index, self._label = tracks, index, label
            self._state, self._position, self._error, self._notice = "loading", 0.0, None, notice
            self._on_started = on_started
        self._publish(origin=origin)
        threading.Thread(target=self._run, args=(generation, index), name="xiaoxiaobai-music", daemon=True).start()

    def jump(self, step: int) -> dict[str, Any]:
        with self._lock:
            if not self._queue:
                raise ApiError("播放列表是空的", CommonState.FAILURE, Reason.TARGET_STATE)
            index = self._index + step
            if index < 0 or index >= len(self._queue):
                raise ApiError("已经是第一首了" if index < 0 else "已经是最后一首了", CommonState.FAILURE, Reason.TARGET_STATE)
            self._generation += 1
            generation = self._generation
            self._stop_proc()
            self._index, self._state, self._position, self._error = index, "loading", 0.0, None
        self._publish()
        threading.Thread(target=self._run, args=(generation, index, step), name="xiaoxiaobai-music", daemon=True).start()
        return response(self.status(), message=f"{'下' if step > 0 else '上'}一首")

    def pause(self) -> dict[str, Any]:
        if self._state != "playing":
            return response(self.status(), reason=Reason.TARGET_STATE, message="没在播放")
        self._command("pause")
        return response(self.status(), message="已暂停")

    def resume(self) -> dict[str, Any]:
        if self._state != "paused":
            return response(self.status(), reason=Reason.TARGET_STATE, message="没有暂停中的歌")
        self._command("resume")
        return response(self.status(), message="继续播放")

    def set_volume(self, volume: int) -> dict[str, Any]:
        volume = max(0, min(100, int(volume)))
        self.settings.update("music", {"volume": volume})
        self._command(f"volume {volume}")
        self._publish()
        return response({"volume": volume}, message=f"音量 {volume}")

    def stop(self, resume_voice: bool = True) -> dict[str, Any]:
        self._fire_started(None)
        with self._lock:
            was = self.active
            self._generation += 1
            self._stop_proc()
            self._state, self._position = "idle", 0.0
        self._publish()
        if resume_voice:
            self._resume_voice()
        else:
            self._voice_paused = False
        return response(self.status(), reason=Reason.SUCCESS if was else Reason.TARGET_STATE, message="已停止" if was else "没在播放")

    def close(self) -> None:
        with self._lock:
            self._generation += 1
            self._stop_proc()

    def _command(self, line: str) -> None:
        proc = self._proc
        if proc is None or proc.poll() is not None or proc.stdin is None:
            return
        try:
            proc.stdin.write(line + "\n")
            proc.stdin.flush()
        except OSError:
            pass

    def _stop_proc(self) -> None:
        self._groove(None)
        proc, self._proc = self._proc, None
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.stdin.write("stop\n")
            proc.stdin.flush()
        except (OSError, AttributeError):
            pass

        def reap() -> None:
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                except OSError:
                    pass

        threading.Thread(target=reap, daemon=True).start()

    # ---- voice chat hand-over -------------------------------------------------------------

    def _pause_voice(self, generation: int) -> None:
        """Lets a running voice chat finish its reply, then stops it so the speaker is free."""
        status = self.voice.status()
        if not status.get("running"):
            return
        deadline = time.monotonic() + VOICE_REPLY_WAIT_S
        while time.monotonic() < deadline and generation == self._generation:
            if self.voice.status().get("state") not in ("thinking", "speaking", "starting"):
                break
            time.sleep(0.2)
        time.sleep(VOICE_TAIL_S)
        if generation != self._generation:
            return
        self.voice.stop()
        self._voice_paused = True
        deadline = time.monotonic() + 6.0
        while time.monotonic() < deadline and self.voice.status().get("running"):
            time.sleep(0.1)
        # The bridge flushes the speaker once it notices the voice client left.
        time.sleep(0.4)

    def _resume_voice(self) -> None:
        if not self._voice_paused:
            return
        self._voice_paused = False
        if not self.cfg().get("resume_voice", True) or self.voice.status().get("running"):
            return
        try:
            self.voice.start()
        except ApiError as exc:
            self.events.alert("warning", f"音乐放完了，但语音对话没能恢复：{exc.message}")

    # ---- the player thread ----------------------------------------------------------------

    def _spawn(self, url: str, track: dict[str, Any]) -> subprocess.Popen | LocalPlayer:
        volume = str(int(self.cfg().get("dance_volume" if track.get("local") else "volume", 60)))
        if track.get("local") and not self.backend.mock:
            return LocalPlayer(Path(urllib.parse.unquote(urllib.parse.urlparse(url).path)), int(volume))
        if self.backend.mock:
            seconds = str(track.get("duration_s") or 20)
            args = [self.backend.python(), "-u", str(Path(__file__).with_name("mock_music.py")), "--url", url,
                    "--volume", volume, "--seconds", seconds]
        else:
            args = [self.backend.python(), "-u", "src/music_player.py", "--url", url, "--volume", volume]
        if track.get("local"):
            args.append("--no-analysis")
        log = self.backend.paths.log("music").open("a", encoding="utf-8")
        return subprocess.Popen(args, cwd=REPO, env=self.backend.env(PYTHONUNBUFFERED="1"), stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1, start_new_session=True)

    def _run(self, generation: int, index: int, step: int = 1) -> None:
        try:
            self._play_queue(generation, index, step)
        finally:
            if generation == self._generation:
                self._fire_started(None)

    def _play_queue(self, generation: int, index: int, step: int) -> None:
        if not self.backend.mock and not self.backend.ensure_display():
            self._set("error", error="显示/音频进程没有运行，放不了音乐")
            return
        self._pause_voice(generation)
        skips = 0
        direction = 1 if step >= 0 else -1
        while generation == self._generation and 0 <= index < len(self._queue):
            track = self._queue[index]
            with self._lock:
                self._index, self._position = index, 0.0
            try:
                info = {"url": track["local"]} if track.get("local") else self._url(track["id"])
            except NeteaseError as exc:
                self._set("error", error=str(exc))
                break
            if not info.get("url"):
                skips += 1
                if skips >= MAX_SKIPS:
                    self._set("error", error="连续好几首都播不了（版权或 VIP），停下了")
                    break
                index += direction
                continue
            notice = None
            if info.get("trial"):
                notice = f"《{track['name']}》是 VIP 歌曲，只能试听片段（登录黑胶 VIP 账号听完整版）"
            with self._lock:
                if generation != self._generation:
                    return
                self._proc = self._spawn(info["url"], track)
                proc = self._proc
                self._state, self._error = "loading", None
                if notice:
                    self._notice = notice
            self._publish()
            outcome = self._pump(proc, generation, track)
            if generation != self._generation:
                return
            if outcome == "fatal":
                break
            if outcome == "error":
                skips += 1
                if skips >= MAX_SKIPS:
                    break
            else:
                skips = 0
            index += 1
            direction = 1
        if generation != self._generation:
            return
        with self._lock:
            self._proc = None
            if self._state != "error":
                self._state, self._position = "idle", 0.0
        self._publish(finished=True)
        self._resume_voice()

    def _pump(self, proc: subprocess.Popen, generation: int, track: dict[str, Any]) -> str:
        """Follows the player's events until it exits: done / error / fatal / stopped."""
        outcome = "error"
        assert proc.stdout is not None
        for line in proc.stdout:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if generation != self._generation:
                continue
            kind = event.get("event")
            if "pos" in event:
                self._position = float(event["pos"])
            if kind == "beat":
                self._groove(event)
            elif kind in ("paused", "stopped", "done", "error"):
                self._groove(None)
            if kind == "started":
                self._fire_started(float(event.get("heard_at") or time.time()))
                self._set("playing")
                self._show(track)
            elif kind == "progress":
                now = time.monotonic()
                if now - self._last_progress_event >= 1.0:
                    self._last_progress_event = now
                    self._publish()
            elif kind == "paused":
                self._set("paused")
            elif kind == "resumed":
                self._set("playing")
            elif kind == "done":
                outcome = "done"
            elif kind == "stopped":
                outcome = "stopped"
            elif kind == "error":
                message = str(event.get("message") or "播放失败")
                fatal = "ffmpeg" in message or "语音桥" in message or "接管" in message
                outcome = "fatal" if fatal else "error"
                if "接管" in message:
                    # The voice chat was started by hand: it has the speaker now.
                    self._voice_paused = False
                    message = "语音对话开始了，音乐停了"
                self._set("error" if fatal else None, error=message)
                self.events.alert("warning", f"音乐：{message}")
        proc.wait()
        if generation == self._generation:
            self._groove(None)
        return outcome

    def _groove(self, beat: dict[str, Any] | None) -> None:
        """Hands the song's beat grid to the upper body (it sways to it in 「走跑·上肢自由」)."""
        if self.motion is None:
            return
        cfg = self.cfg()
        if beat is None or not cfg.get("groove", True):
            self.motion.set_groove(None)
            return
        # 动作时机: positive moves the dance earlier (a speaker or servos slower than assumed).
        early = max(-300, min(300, int(cfg.get("groove_offset_ms", 0)))) / 1000.0
        grid = {"t0": float(beat["t0"]) - early, "period": float(beat["period"]),
                "size": max(0, min(100, int(cfg.get("groove_size", 70)))) / 100.0,
                "bar_beats": int(beat.get("bar_beats") or 4), "analysing": bool(beat.get("analysing"))}
        for key in ("confidence", "energy", "bar_confidence", "busy"):
            grid[key] = float(beat.get(key) or 0.0)
        if beat.get("downbeat") is not None:
            grid["downbeat"] = float(beat["downbeat"]) - early
        if beat.get("index") is not None:
            grid.update(index=int(beat["index"]), end=int(beat.get("end") or 0), section=beat.get("section"),
                        next=beat.get("next"), song=beat.get("song"))
        self.motion.set_groove(grid)

    def _show(self, track: dict[str, Any]) -> None:
        if not self.cfg().get("show_on_screen", True):
            return
        try:
            self.interaction.show_board({
                "kind": "list", "title": "正在播放", "subtitle": track.get("artist", ""), "title_color": "pink",
                "items": [{"text": track.get("name", "")}] + ([{"text": track["album"]}] if track.get("album") else []),
                "seconds": 8,
            }, "music")
        except Exception:  # noqa: BLE001, S110 - the screen is a nicety
            pass

    # ---- voice / chat -----------------------------------------------------------------------

    def heard(self, text: str, origin: str = "voice") -> bool:
        """A sentence that asks for music (or controls it) is acted on in the background: True."""
        cfg = self.cfg()
        if not cfg.get("enabled", True) or not cfg.get("voice_control", True):
            return False
        intent = parse_intent(text, self.active)
        if intent is None:
            return False
        action, query = intent
        threading.Thread(target=self._act, args=(action, query, origin), name="xiaoxiaobai-music-voice", daemon=True).start()
        return True

    def _act(self, action: str, query: str, origin: str) -> None:
        if action in ("play", "hot", "daily", "liked"):
            self.events.publish("music", state="searching", query=query, origin=origin)
        try:
            result = self.do(action, query, origin)
            if result.get("message"):
                self.events.alert("info", f"音乐：{result['message']}")
        except ApiError as exc:
            self.events.alert("warning", f"音乐：{exc.message}")

    def do(self, action: str, query: str = "", origin: str = "interaction") -> dict[str, Any]:
        volume = int(self.cfg().get("volume", 60))
        if action == "play":
            return self.play({"query": query}, origin)
        if action in ("hot", "daily", "liked"):
            return self.play({"playlist": action}, origin)
        if action == "stop":
            return self.stop()
        if action == "pause":
            return self.pause()
        if action == "resume":
            return self.resume()
        if action == "next":
            return self.jump(1)
        if action == "previous":
            return self.jump(-1)
        if action == "louder":
            return self.set_volume(volume + 15)
        if action == "quieter":
            return self.set_volume(volume - 15)
        raise ApiError(f"不认识的音乐操作 {action}", CommonState.INVALID)
