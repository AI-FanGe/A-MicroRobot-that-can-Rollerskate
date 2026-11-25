# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

"""NetEase Cloud Music (music.163.com) web API: search, song URLs, playlists and login.

Requests go to the web client's /weapi endpoints, whose bodies are encrypted the way the web
page does it (AES-128-CBC twice, the second key sent RSA-encrypted). Standard library only:
AES is implemented below (encryption only, a few hundred bytes per request) and RSA is pow().

Logging in (QR code scanned with the NetEase app, SMS code, or a pasted MUSIC_U cookie) gives
the cookies that unlock VIP songs and the user's own playlists; without them search and free
songs still work.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

BASE = "https://music.163.com"
EAPI_BASE = "https://interfacepc.music.163.com"
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/124.0.0.0 Safari/537.36")
EAPI_USER_AGENT = "NeteaseMusic 9.0.90/5038 (iPhone; iOS 16.2; zh_CN)"
CLIENT = {"os": "pc", "appver": "3.1.17.204416", "osver": "Microsoft-Windows-10-Professional-build-19045-64bit",
          "channel": "netease"}
# The iPhone app's login page, for SMS logins: the web and desktop ones get -462 or
# 「当前登录存在安全风险」 from risk control (api-enhanced PR #201, 2026-06).
MOBILE_BASE = "https://interface3.music.163.com"
MOBILE = {"os": "iPhone OS", "appver": "9.5.37", "osver": "18.7.2", "buildver": "7010", "channel": "distribution"}
MOBILE_USER_AGENT = "neteasemusic/9.5.37 (iPhone; iOS 18.7.2; Scale/3.00)"
MOBILE_LOGIN_PAGE = {"os": "iOS", "fromPage": "RN", "rnBundleVersion": "0.0.5", "rnBundleName": "new-rn-login", "verifyId": 1}
# The web page's scan login, for QR codes (same PR): a browser's cookies and headers.
WEB_QR_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:152.0) Gecko/20100101 Firefox/152.0"
WEB_QR_HEADERS = {"Referer": BASE + "/", "Origin": BASE, "x-os": "web", "X-channelSource": "undefined",
                  "Nm-GCore-Status": "1"}
SECURE_CAPTCHA = 8821  # the web QR login wants the browser's NECaptcha solved first
RISKY_NETWORK = 8810  # 「您当前的网络环境存在安全风险」
# NetEase's verify page has the NECaptcha ids built in, per verifyType: 40 slider, 41 space.
SLIDER_CAPTCHA_IDS = {40: "aefd3e6914a546a2a7263b20a5e8304f", 41: "eb43b69cce7449c1b818b8a291b36f21"}
VERIFY_BASE = "https://st.music.163.com"
PC_USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Safari/537.36 "
                 "Chrome/91.0.4472.164 NeteaseMusicDesktop/3.1.29.205117")
TIMEOUT_S = 10
HOT_PLAYLIST = 3778678  # 热歌榜
QUALITIES = ("standard", "higher", "exhigh")  # 128k / 192k / 320k mp3

# ---------------------------------------------------------------- AES-128 (encrypt only)

_SBOX = bytes.fromhex(
    "637c777bf26b6fc53001672bfed7ab76ca82c97dfa5947f0add4a2af9ca472c0"
    "b7fd9326363ff7cc34a5e5f171d8311504c723c31896059a071280e2eb27b275"
    "09832c1a1b6e5aa0523bd6b329e32f8453d100ed20fcb15b6acbbe394a4c58cf"
    "d0efaafb434d338545f9027f503c9fa851a3408f929d38f5bcb6da2110fff3d2"
    "cd0c13ec5f974417c4a77e3d645d197360814fdc222a908846eeb814de5e0bdb"
    "e0323a0a4906245cc2d3ac629195e479e7c8376d8dd54ea96c56f4ea657aae08"
    "ba78252e1ca6b4c6e8dd741f4bbd8b8a703eb5664803f60e613557b986c11d9e"
    "e1f8981169d98e949b1e87e9ce5528df8ca1890dbfe6426841992d0fb054bb16"
)
_RCON = (0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36)


def _xtime(a: int) -> int:
    a <<= 1
    return (a ^ 0x1B) & 0xFF if a & 0x100 else a


def _expand_key(key: bytes) -> list[list[int]]:
    words = [list(key[i:i + 4]) for i in range(0, 16, 4)]
    for i in range(4, 44):
        word = list(words[i - 1])
        if i % 4 == 0:
            word = [_SBOX[b] for b in word[1:] + word[:1]]
            word[0] ^= _RCON[i // 4 - 1]
        words.append([a ^ b for a, b in zip(words[i - 4], word)])
    return [[b for word in words[r * 4:r * 4 + 4] for b in word] for r in range(11)]


def _encrypt_block(block: bytes, round_keys: list[list[int]]) -> bytes:
    s = [b ^ k for b, k in zip(block, round_keys[0])]
    for rnd in range(1, 11):
        s = [_SBOX[b] for b in s]
        # ShiftRows on the column-major state (byte r + 4c is row r, column c).
        s = [s[(i + 4 * (i % 4)) % 16] for i in range(16)]
        if rnd < 10:
            mixed = []
            for c in range(4):
                a = s[4 * c:4 * c + 4]
                t = a[0] ^ a[1] ^ a[2] ^ a[3]
                mixed += [a[i] ^ t ^ _xtime(a[i] ^ a[(i + 1) % 4]) for i in range(4)]
            s = mixed
        s = [b ^ k for b, k in zip(s, round_keys[rnd])]
    return bytes(s)


def _pad(data: bytes) -> bytes:
    pad = 16 - len(data) % 16
    return data + bytes([pad]) * pad


def aes_cbc_encrypt(data: bytes, key: bytes, iv: bytes) -> bytes:
    data = _pad(data)
    round_keys = _expand_key(key)
    out = bytearray()
    prev = iv
    for i in range(0, len(data), 16):
        prev = _encrypt_block(bytes(a ^ b for a, b in zip(data[i:i + 16], prev)), round_keys)
        out += prev
    return bytes(out)


def aes_ecb_encrypt(data: bytes, key: bytes) -> bytes:
    data = _pad(data)
    round_keys = _expand_key(key)
    return b"".join(_encrypt_block(data[i:i + 16], round_keys) for i in range(0, len(data), 16))


# ---------------------------------------------------------------- weapi

_PRESET_KEY = b"0CoJUm6Qyw8W8jud"
_IV = b"0102030405060708"
_RSA_E = 0x010001
_RSA_N = int(
    "00e0b509f6259df8642dbc35662901477df22677ec152b5ff68ace615bb7b725152b3ab17a876aea8a5aa76d2e417629ec4ee341f56135fccf"
    "695280104e0312ecbda92557c93870114af6c9d05c4f7f0c3685b7a46bee255932575cce10b424d813cfe4875d3e82047b97ddef52741d546b"
    "8e289dc6935b3ece0462db0a22b8e7", 16)
_BASE62 = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def weapi_encrypt(payload: dict[str, Any]) -> dict[str, str]:
    text = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    secret = "".join(random.choice(_BASE62) for _ in range(16)).encode("ascii")
    first = base64.b64encode(aes_cbc_encrypt(text, _PRESET_KEY, _IV))
    params = base64.b64encode(aes_cbc_encrypt(first, secret, _IV)).decode("ascii")
    enc = pow(int(secret[::-1].hex(), 16), _RSA_E, _RSA_N)
    return {"params": params, "encSecKey": format(enc, "x").zfill(256)}


# ---------------------------------------------------------------- eapi (the desktop client's)

_EAPI_KEY = b"e82ckenh8dichen8"


def eapi_encrypt(path: str, payload: dict[str, Any]) -> dict[str, str]:
    """path: the /api/... path the body is signed for."""
    text = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.md5(f"nobody{path}use{text}md5forencrypt".encode()).hexdigest()
    message = f"{path}-36cd479b6b5-{text}-36cd479b6b5-{digest}".encode()
    return {"params": aes_ecb_encrypt(message, _EAPI_KEY).hex().upper()}


VERIFY_NEEDED = -462  # risk control: 「请完成验证操作」, data carries the challenge (verify_slider / verify_qr)


class NeteaseError(Exception):
    """session: which cookies the refused request went with ("weapi", "pc", "mobile" or "qr:<key>"),
    as a verification only counts for those."""

    def __init__(self, message: str, code: int | None = None, data: Any = None, session: str = "weapi") -> None:
        super().__init__(message)
        self.code = code
        self.data = data
        self.session = session

    @property
    def challenge(self) -> dict[str, Any] | None:
        """The -462 challenge (verifyId, verifyType, verifyToken, params), if that is what this is."""
        if self.code == VERIFY_NEEDED and isinstance(self.data, dict) and self.data.get("verifyToken"):
            return self.data
        return None


def _fail(result: dict[str, Any], default: str, session: str = "weapi") -> NeteaseError:
    code = result.get("code")
    message = result.get("message") or result.get("msg") or default
    if code == 301:
        message = "需要登录网易云账号"
    return NeteaseError(f"网易云：{message}（{code}）", code, result.get("data"), session)


def _artists(song: dict[str, Any]) -> str:
    return " / ".join(a.get("name", "") for a in song.get("ar") or song.get("artists") or [] if a.get("name"))


def track_of(song: dict[str, Any]) -> dict[str, Any]:
    """The fields the player and the web UI need, from any of the API's song shapes."""
    album = song.get("al") or song.get("album") or {}
    privilege = song.get("privilege") or {}
    return {
        "id": int(song["id"]),
        "name": song.get("name", ""),
        "artist": _artists(song),
        "album": album.get("name", ""),
        "cover": album.get("picUrl", ""),
        "duration_s": round((song.get("dt") or song.get("duration") or 0) / 1000, 1),
        "vip": song.get("fee") == 1 or privilege.get("fee") == 1,
    }


class NeteaseClient:
    """Stateless apart from the cookies (MUSIC_U, __csrf, ...) handed in and updated by logins."""

    def __init__(self, cookies: dict[str, str] | None = None) -> None:
        self.cookies: dict[str, str] = dict(cookies or {})
        self.cookies.setdefault("NMTID", "00O" + "".join(random.choice(_BASE62) for _ in range(29)))
        self.cookies.setdefault("_ntes_nuid", os.urandom(16).hex())
        # One device for every login: a new one each time looks like a farm of clients.
        self.cookies.setdefault("deviceId", os.urandom(26).hex().upper())
        # Web QR logins under way: key -> (the browser cookies they started with, chain id).
        self._web_qr: dict[str, tuple[dict[str, str], str]] = {}
        # Phones whose SMS login got the web page's slider: their next try goes there directly.
        self._sms_web: set[str] = set()

    @property
    def logged_in(self) -> bool:
        return bool(self.cookies.get("MUSIC_U"))

    def _cookie_header(self) -> str:
        base = {**CLIENT, "__remember_me": "true"}
        return "; ".join(f"{k}={v}" for k, v in {**base, **self.cookies}.items() if v)

    def request(self, path: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        """POST to /weapi/{path} and return the JSON; Set-Cookie updates the cookies."""
        payload = {**(data or {}), "csrf_token": self.cookies.get("__csrf", "")}
        url = f"{BASE}/weapi/{path.lstrip('/')}"
        if self.cookies.get("__csrf"):
            url += f"?csrf_token={self.cookies['__csrf']}"
        return self._post(url, weapi_encrypt(payload), {
            "User-Agent": USER_AGENT,
            "Referer": BASE + "/",
            "Origin": BASE,
            "Cookie": self._cookie_header(),
        })

    def _eapi_header(self, mobile: bool) -> dict[str, str]:
        client = MOBILE if mobile else CLIENT
        header = {
            "osver": client["osver"], "deviceId": self.cookies["deviceId"], "os": client["os"],
            "appver": client["appver"], "versioncode": "140", "mobilename": "",
            "buildver": client.get("buildver") or str(int(time.time())), "resolution": "1920x1080",
            "__csrf": self.cookies.get("__csrf", ""), "channel": client["channel"],
            "requestId": f"{int(time.time() * 1000)}_{random.randrange(1000):04d}",
        }
        for name in ("MUSIC_U", "MUSIC_A") + (() if mobile else ("NMTID",)):
            if self.cookies.get(name):
                header[name] = self.cookies[name]
        return header

    @staticmethod
    def _eapi_cookie(header: dict[str, str]) -> str:
        return "; ".join(f"{urllib.parse.quote(k)}={urllib.parse.quote(str(v))}" for k, v in header.items())

    def eapi(self, path: str, data: dict[str, Any] | None = None, mobile: bool = False) -> dict[str, Any]:
        """POST /api/{path} the way the desktop client (or with mobile, the iPhone app) does."""
        api_path = "/api/" + path.lstrip("/")
        client = MOBILE if mobile else CLIENT
        device = self.cookies["deviceId"]
        header = self._eapi_header(mobile)
        cookie = self._eapi_cookie(header)
        if mobile:
            payload = {**(data or {}), "deviceId": device, "header": {}, "e_r": False}
            headers = {"User-Agent": MOBILE_USER_AGENT, "Cookie": cookie, "x-os": client["os"],
                       "x-osver": client["osver"], "x-appver": client["appver"], "x-buildver": client["buildver"],
                       "x-deviceid": device, "x-sdeviceid": device}
            if self.cookies.get("MUSIC_U"):
                headers["x-music-u"] = self.cookies["MUSIC_U"]
            url = f"{MOBILE_BASE}/eapi/{path.lstrip('/')}"
        else:
            payload = {**(data or {}), "header": header, "e_r": False}
            headers = {"User-Agent": EAPI_USER_AGENT, "Cookie": cookie}
            url = f"{EAPI_BASE}/eapi/{path.lstrip('/')}"
        return self._post(url, eapi_encrypt(api_path, payload), headers)

    def web_qr(self, path: str, data: dict[str, Any], cookies: dict[str, str],
               headers: dict[str, str] | None = None) -> dict[str, Any]:
        """POST /weapi/{path} as the web page's scan login does: only the browser's own cookies,
        which the answer's Set-Cookie updates like a browser would."""
        cookie = "; ".join(f"{k}={v}" for k, v in cookies.items())
        return self._post(f"{BASE}/weapi/{path.lstrip('/')}", weapi_encrypt({**data, "csrf_token": ""}),
                          {**WEB_QR_HEADERS, **(headers or {}), "User-Agent": WEB_QR_USER_AGENT, "Cookie": cookie},
                          jar=cookies)

    def _post(self, url: str, form: dict[str, str], headers: dict[str, str],
              jar: dict[str, str] | None = None) -> dict[str, Any]:
        """Set-Cookie goes to self.cookies (and jar, when given)."""
        body = urllib.parse.urlencode(form).encode("ascii")
        req = urllib.request.Request(url, data=body, method="POST",
                                     headers={**headers, "Content-Type": "application/x-www-form-urlencoded"})
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
                raw = resp.read()
                for header in resp.headers.get_all("Set-Cookie") or []:
                    name, _, rest = header.partition("=")
                    value = rest.split(";", 1)[0]
                    if name.strip() and value and "Max-Age=0" not in header:
                        self.cookies[name.strip()] = value
                        if jar is not None:
                            jar[name.strip()] = value
        except urllib.error.HTTPError as exc:
            try:  # risk control and friends answer JSON with an error status
                raw = exc.read()
                result = json.loads(raw.decode("utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                raise NeteaseError(f"网易云返回 HTTP {exc.code}", exc.code) from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise NeteaseError(f"连不上网易云：{getattr(exc, 'reason', exc)}") from None
        else:
            try:
                result = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise NeteaseError("网易云返回了无法解析的内容") from None
        if isinstance(result, dict) and isinstance(result.get("code"), str) and result["code"].lstrip("-").isdigit():
            result["code"] = int(result["code"])
        return result

    def _ok(self, path: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        result = self.request(path, data)
        if result.get("code") not in (200, None):
            raise _fail(result, "请求失败")
        return result

    # ---- music ----------------------------------------------------------------------------

    def search(self, keywords: str, limit: int = 20) -> list[dict[str, Any]]:
        # cloudsearch/get/web now answers 50000005 to the web client; the older search/get still
        # works but its song shape lacks covers and fees, hence the detail lookup.
        result = self._ok("search/get", {"s": keywords, "type": 1, "limit": limit, "offset": 0})
        found = (result.get("result") or {}).get("songs") or []
        ids = [int(s["id"]) for s in found if s.get("id")]
        if not ids:
            return []
        try:
            detailed = {t["id"]: t for t in self.songs(ids)}
        except NeteaseError:
            detailed = {}
        return [detailed.get(int(s["id"])) or track_of(s) for s in found if s.get("id")]

    def song_urls(self, ids: list[int], quality: str = "exhigh") -> dict[int, dict[str, Any]]:
        """{id: {"url", "br", "trial"}}; url is None when a song cannot be played (no copyright, or
        VIP-only without a VIP login). URLs expire after ~20 minutes."""
        if not ids:
            return {}
        level = quality if quality in QUALITIES else "exhigh"
        result = self._ok("song/enhance/player/url/v1",
                          {"ids": json.dumps([int(x) for x in ids]), "level": level, "encodeType": "mp3"})
        out: dict[int, dict[str, Any]] = {}
        for item in result.get("data") or []:
            out[int(item["id"])] = {"url": item.get("url"), "br": item.get("br") or 0,
                                    "trial": bool(item.get("freeTrialInfo")), "type": item.get("type") or "mp3"}
        return out

    def song_url(self, song_id: int, quality: str = "exhigh") -> dict[str, Any]:
        return self.song_urls([song_id], quality).get(int(song_id)) or {"url": None, "br": 0, "trial": False}

    def songs(self, ids: list[int]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for i in range(0, len(ids), 200):
            chunk = ids[i:i + 200]
            result = self._ok("v3/song/detail", {"c": json.dumps([{"id": int(x)} for x in chunk])})
            out += [track_of(s) for s in result.get("songs") or []]
        return out

    def playlist(self, playlist_id: int, limit: int = 100) -> list[dict[str, Any]]:
        result = self._ok("v6/playlist/detail", {"id": int(playlist_id), "n": 100000, "s": 8})
        info = result.get("playlist") or {}
        ids = [t["id"] for t in info.get("trackIds") or []][:limit]
        if not ids:
            return [track_of(s) for s in (info.get("tracks") or [])[:limit]]
        return self.songs(ids)

    def hot(self, limit: int = 50) -> list[dict[str, Any]]:
        return self.playlist(HOT_PLAYLIST, limit)

    def daily(self) -> list[dict[str, Any]]:
        result = self._ok("v3/discovery/recommend/songs", {})
        return [track_of(s) for s in (result.get("data") or {}).get("dailySongs") or []]

    def liked(self, limit: int = 200) -> list[dict[str, Any]]:
        """The user's 「我喜欢的音乐」 (their first playlist)."""
        uid = self.account().get("user_id")
        if not uid:
            raise NeteaseError("需要登录网易云账号", 301)
        result = self._ok("user/playlist", {"uid": uid, "limit": 1, "offset": 0, "includeVideo": True})
        lists = result.get("playlist") or []
        if not lists:
            return []
        return self.playlist(lists[0]["id"], limit)

    # ---- account --------------------------------------------------------------------------

    def account(self) -> dict[str, Any]:
        if not self.logged_in:
            return {"logged_in": False}
        result = self.request("w/nuser/account/get", {})
        profile = result.get("profile") or {}
        if result.get("code") != 200 or not profile:
            return {"logged_in": False, "expired": True}
        return {
            "logged_in": True,
            "user_id": profile.get("userId"),
            "nickname": profile.get("nickname", ""),
            "avatar": profile.get("avatarUrl", ""),
            "vip": bool(profile.get("vipType")) or bool((result.get("account") or {}).get("vipType")),
        }

    # Two QR flows, as risk control lets one or the other through: "web", the web page's scan
    # login (weapi, type 1, a browser's cookies and a login chain id), and "pc", the desktop
    # client's (eapi, type 3).
    QR_FLOWS = ("web", "pc")

    def qr_key(self, flow: str = "web") -> str:
        if flow == "web":
            nuid = os.urandom(16).hex()
            token = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-_"
            jar = {
                "JSESSIONID-WYYY": "".join(random.choice(token) for _ in range(190)),
                "_iuqxldmzr_": "33",
                "_ntes_nnid": f"{nuid},{int(time.time() * 1000)}",
                "_ntes_nuid": nuid,
                "NMTID": "00" + "".join(random.choice(token) for _ in range(39)),
                "WEVNSM": "1.0.0",
                "WNMCID": "".join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(6)) + f".{int(time.time() * 1000)}.01.0",
            }
            result = self.web_qr("login/qrcode/unikey", {"type": 1, "noCheckToken": True}, jar)
        else:
            result = self.eapi("login/qrcode/unikey", {"type": 3})
        if result.get("code") != 200 or not result.get("unikey"):
            raise _fail(result, "拿不到二维码")
        key = result["unikey"]
        if flow == "web":
            chain = f"v1_unknown-{random.randrange(1_000_000)}_web_login_{int(time.time() * 1000)}"
            if len(self._web_qr) > 20:
                self._web_qr.clear()
            self._web_qr[key] = (jar, chain)
        return key

    def qr_url(self, key: str) -> str:
        if key in self._web_qr:
            chain = urllib.parse.quote(self._web_qr[key][1], safe="")
            return f"{BASE}/st/platform/scanlogin?codekey={key}&chainId={chain}&hdw_device=web&hdw_appid=web&hitExp=1"
        return f"{BASE}/login?codekey={key}"

    def qr_check(self, key: str) -> dict[str, Any]:
        """code 800 expired, 801 waiting for a scan, 802 scanned (confirm in the app), 803 logged in;
        -462 with "challenge" (and its "session"): risk control wants it verified first; 8821 (web
        flow): it wants the browser's captcha, which only the web page can show."""
        web = self._web_qr.get(key)
        if web is not None:
            jar, chain = web
            result = self.web_qr("login/qrcode/client/login",
                                 {"key": key, "type": 1, "noCheckToken": True, "ydDeviceToken": ""}, jar,
                                 {"X-loginMethod": "QrCode", "x-login-chain-id": chain})
        else:
            result = self.eapi("login/qrcode/client/login", {"key": key, "type": 3})
        code = result.get("code")
        if code in (800, 803):
            self._web_qr.pop(key, None)
        state = {"code": code, "message": result.get("message", ""), "nickname": result.get("nickname", ""),
                 "avatar": result.get("avatarUrl", ""), "flow": "web" if web is not None else "pc"}
        error = _fail(result, "", f"qr:{key}" if web is not None else "pc")
        if error.challenge:
            state["challenge"] = error.challenge
            state["session"] = error.session
        return state

    def sms_send(self, phone: str, country: str = "86") -> None:
        result = self.eapi("sms/captcha/sent", {"ctcode": country, "cellphone": phone, **MOBILE_LOGIN_PAGE}, mobile=True)
        if result.get("code") == 200:
            return
        if result.get("code") == 400:  # the number itself is wrong
            raise _fail(result, "发送失败", "mobile")
        # The desktop client's, which risk control sometimes lets through instead.
        try:
            self._ok("sms/captcha/sent", {"cellphone": phone, "ctcode": country, "secrete": "music_middleuser_pclogin"})
        except NeteaseError as exc:
            if exc.challenge:
                raise
            raise _fail(result, "发送失败", "mobile") from None

    def sms_login(self, phone: str, captcha: str, country: str = "86") -> dict[str, Any]:
        """The app's login; refused by risk control (8810 and the like), the web page's, whose
        -462 a slider clears. Once that one asked for the slider, it is the one tried again."""
        risky = (RISKY_NETWORK, SECURE_CAPTCHA, VERIFY_NEEDED, -460)
        app: dict[str, Any] | None = None
        result: dict[str, Any] | None = None
        session = "mobile"
        if phone not in self._sms_web:
            app = self.eapi("login/cellphone", {
                "type": "1", "https": "true", "phone": phone, "countrycode": country, "captcha": captcha,
                "remember": "true", "rememberLogin": "true", **MOBILE_LOGIN_PAGE,
            }, mobile=True)
            if app.get("code") not in risky:
                result = app
        if result is None:
            web = self.request("login/cellphone", {"phone": phone, "countrycode": country, "captcha": captcha,
                                                   "rememberLogin": "true"})
            slider = _fail(web, "").challenge is not None
            if slider:
                self._sms_web.add(phone)
            # The web page's answer, unless it is plain risk control too and the app's came first.
            if app is None or slider or web.get("code") not in risky:
                result, session = web, "weapi"
            else:
                result = app
        if result.get("code") != 200:
            raise _fail(result, "登录失败", session)
        self._sms_web.discard(phone)
        if not self.logged_in and result.get("token"):  # the app's answer carries MUSIC_U as "token"
            self.cookies["MUSIC_U"] = str(result["token"])
        if not self.logged_in:
            raise NeteaseError("网易云说登录成功，但没有给登录凭证（MUSIC_U）")
        return self.account()

    # ---- risk-control verification (the -462 challenge) -------------------------------------

    def verify_slider(self, challenge: dict[str, Any], validate: str, session: str) -> bool:
        """Hands NetEase the validate of the NECaptcha slider solved in a browser (verifyType 40 / 41),
        with the cookies of the request that got the challenge. True: cleared, retry the request."""
        params = challenge.get("params") or {}
        if not isinstance(params, str):
            params = json.dumps(params, separators=(",", ":"), ensure_ascii=False)
        verify_type = int(challenge.get("verifyType") or 40)
        jar = None
        if session.startswith("qr:") and session[3:] in self._web_qr:
            jar = self._web_qr[session[3:]][0]
            cookie, agent = "; ".join(f"{k}={v}" for k, v in jar.items()), WEB_QR_USER_AGENT
        elif session == "mobile":
            cookie, agent = self._eapi_cookie(self._eapi_header(True)), MOBILE_USER_AGENT
        elif session == "pc":
            cookie, agent = self._eapi_cookie(self._eapi_header(False)), PC_USER_AGENT
        else:
            cookie, agent = self._cookie_header(), PC_USER_AGENT
        referer = VERIFY_BASE + "/encrypt-pages?" + urllib.parse.urlencode({
            "params": params, "verifyType": verify_type, "verifyId": challenge.get("verifyId"),
            "verifyToken": challenge.get("verifyToken"),
        })
        result = self._post(VERIFY_BASE + "/weapi/frontrisk/verify/check", weapi_encrypt({
            "token": challenge.get("verifyToken"), "verifyType": verify_type,
            "captchaId": SLIDER_CAPTCHA_IDS.get(verify_type, SLIDER_CAPTCHA_IDS[40]),
            "validate": validate, "params": params,
        }), {"Origin": VERIFY_BASE, "Referer": referer, "Cookie": cookie, "User-Agent": agent}, jar=jar)
        # Its code is "200" either way: only data true means it passed.
        return result.get("data") is True

    def verify_qr(self, challenge: dict[str, Any]) -> dict[str, str]:
        """A QR code for the NetEase app that clears the challenge: {"qr_code", "url"}."""
        params = challenge.get("params") or {}
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except json.JSONDecodeError:
                params = {}
        extra = json.dumps({"event_id": params.get("event_id"), "sign": params.get("sign")}, separators=(",", ":"))
        result = self._ok("frontrisk/verify/getqrcode", {
            "verifyConfigId": challenge.get("verifyId"), "verifyType": challenge.get("verifyType"),
            "token": challenge.get("verifyToken"), "params": extra, "size": 150,
        })
        qr_code = (result.get("data") or {}).get("qrCode")
        if not qr_code:
            raise NeteaseError("网易云没给安全验证二维码")
        url = "https://st.music.163.com/encrypt-pages?" + urllib.parse.urlencode({
            "qrCode": qr_code, "verifyToken": challenge.get("verifyToken"), "verifyId": challenge.get("verifyId"),
            "verifyType": challenge.get("verifyType"), "params": extra,
        })
        return {"qr_code": qr_code, "url": url}

    def verify_status(self, qr_code: str) -> int:
        """qrCodeStatus: 0 waiting, 10 scanned, 20 verified, 21 expired."""
        result = self._ok("frontrisk/verify/qrcodestatus", {"qrCode": qr_code})
        return int((result.get("data") or {}).get("qrCodeStatus") or 0)

    def use_cookie(self, text: str) -> None:
        """A cookie copied from a logged-in browser: the whole Cookie header, or just MUSIC_U's value."""
        text = (text or "").strip()
        found = dict(re.findall(r"([A-Za-z_][\w.-]*)=([^;\s]+)", text))
        if not found and re.fullmatch(r"[0-9A-Za-z+/=_-]{40,}", text):
            found = {"MUSIC_U": text}
        if not found.get("MUSIC_U"):
            raise NeteaseError("没找到 MUSIC_U：复制浏览器里 music.163.com 的 Cookie，或只填 MUSIC_U 的值")
        self.cookies.update({k: v for k, v in found.items() if k in ("MUSIC_U", "__csrf", "MUSIC_A_T", "MUSIC_R_T", "NMTID")})

    def logout(self) -> None:
        if self.logged_in:
            try:
                self.request("logout", {})
            except NeteaseError:
                pass
        for name in ("MUSIC_U", "__csrf", "MUSIC_A_T", "MUSIC_R_T"):
            self.cookies.pop(name, None)


def wait_qr(client: NeteaseClient, key: str, timeout_s: float = 120.0) -> dict[str, Any]:
    """Poll a QR login until it ends (for testing from a terminal)."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        state = client.qr_check(key)
        if state["code"] in (800, 803):
            return state
        time.sleep(2.0)
    return {"code": 800, "message": "超时"}
