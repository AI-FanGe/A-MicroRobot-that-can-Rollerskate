# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

"""Volcengine (Doubao) bidirectional streaming TTS, V3 WebSocket binary protocol.

One connection, one session per reply: StartSession, a TaskRequest per text delta as the LLM
streams it (the server splits sentences itself), FinishSession, then 24 kHz s16le PCM comes
back in TTSResponse frames until SessionFinished.

Auth: the API key of the new Volcengine speech console (X-Api-Key), or "APPID:ACCESS_TOKEN"
from the old console.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import struct
import uuid
from collections.abc import Callable
from dataclasses import dataclass

from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import WebSocketException

URL = "wss://openspeech.bytedance.com/api/v3/tts/bidirection"
API_KEY_ENV = "VOLC_TTS_API_KEY"
SAMPLE_RATE = 24000

START_CONNECTION, FINISH_CONNECTION = 1, 2
CONNECTION_STARTED, CONNECTION_FAILED, CONNECTION_FINISHED = 50, 51, 52
START_SESSION, CANCEL_SESSION, FINISH_SESSION = 100, 101, 102
SESSION_STARTED, SESSION_CANCELED, SESSION_FINISHED, SESSION_FAILED = 150, 151, 152, 153
TASK_REQUEST = 200
TTS_SENTENCE_START, TTS_SENTENCE_END, TTS_RESPONSE = 350, 351, 352

FULL_CLIENT_REQUEST, FULL_SERVER_RESPONSE, AUDIO_ONLY_RESPONSE, ERROR_INFORMATION = 0b0001, 0b1001, 0b1011, 0b1111
FLAG_EVENT = 0b0100
JSON, GZIP = 0b0001, 0b0001
# Events whose frames carry no connection/session id.
NO_ID_EVENTS = frozenset({START_CONNECTION, FINISH_CONNECTION})

SESSION_START_TIMEOUT_S = 10.0
SESSION_FINISH_TIMEOUT_S = 60.0


class VolcError(ConnectionError):
    pass


@dataclass
class Frame:
    kind: int
    event: int | None
    id: str
    payload: bytes
    code: int = 0

    def json(self) -> dict:
        try:
            return json.loads(self.payload or b"{}")
        except ValueError:
            return {}

    def message(self) -> str:
        data = self.json()
        text = data.get("message") or data.get("error") or self.payload.decode(errors="replace")
        code = data.get("status_code") or self.code
        return f"{text} ({code})" if code else text


def encode(event: int, session_id: str | None, payload: dict | None = None) -> bytes:
    body = json.dumps(payload or {}, ensure_ascii=False).encode()
    out = bytearray([0x11, (FULL_CLIENT_REQUEST << 4) | FLAG_EVENT, JSON << 4, 0])
    out += struct.pack(">i", event)
    if event not in NO_ID_EVENTS:
        sid = (session_id or "").encode()
        out += struct.pack(">I", len(sid)) + sid
    out += struct.pack(">I", len(body)) + body
    return bytes(out)


def _sized(data: bytes, pos: int) -> tuple[bytes, int] | None:
    if pos + 4 > len(data):
        return None
    size = struct.unpack_from(">I", data, pos)[0]
    end = pos + 4 + size
    return (data[pos + 4 : end], end) if end <= len(data) else None


def decode(data: bytes) -> Frame:
    pos = (data[0] & 0x0F) * 4
    kind, flags = data[1] >> 4, data[1] & 0x0F
    compressed = (data[2] & 0x0F) == GZIP
    code, event, ident = 0, None, ""
    if kind == ERROR_INFORMATION:
        code = struct.unpack_from(">I", data, pos)[0]
        pos += 4
    if flags & FLAG_EVENT:
        event = struct.unpack_from(">i", data, pos)[0]
        pos += 4
    payload = b""
    # An id field precedes the payload except on the connection requests; the docs are not
    # consistent about ConnectionFailed, so take whichever layout accounts for every byte.
    with_id = _sized(data, pos) if event not in NO_ID_EVENTS else None
    rest = _sized(data, with_id[1]) if with_id else None
    if with_id and rest and rest[1] == len(data):
        ident, payload = with_id[0].decode(errors="replace"), rest[0]
    else:
        plain = _sized(data, pos)
        payload = plain[0] if plain else data[pos:]
    if compressed and payload:
        payload = gzip.decompress(payload)
    return Frame(kind, event, ident, payload, code)


def auth_headers(key: str, resource: str) -> dict[str, str]:
    headers = {"X-Api-Resource-Id": resource, "X-Api-Connect-Id": str(uuid.uuid4())}
    app_id, sep, token = key.partition(":")
    if sep and app_id.isdigit():
        headers |= {"X-Api-App-Id": app_id, "X-Api-Access-Key": token}
    else:
        headers["X-Api-Key"] = key
    return headers


def session_params(voice: str, speech_rate: float) -> dict:
    audio = {"format": "pcm", "sample_rate": SAMPLE_RATE}
    rate = round((speech_rate - 1.0) * 100)
    if rate:
        audio["speech_rate"] = max(-50, min(100, rate))
    return {"speaker": voice, "audio_params": audio}


def request(event: int, params: dict, **extra: object) -> dict:
    return {"user": {"uid": "xiaoxiaobai"}, "event": event, "namespace": "BidirectionalTTS", "req_params": params | extra}


async def open_connection(key: str, resource: str) -> ClientConnection:
    ws = await connect(URL, additional_headers=auth_headers(key, resource), max_size=None, open_timeout=10, close_timeout=1)
    try:
        await ws.send(encode(START_CONNECTION, None))
        frame = await _next_frame(ws, SESSION_START_TIMEOUT_S)
        if frame.event != CONNECTION_STARTED:
            raise VolcError(f"火山引擎拒绝连接：{frame.message()}")
    except BaseException:
        await ws.close()
        raise
    return ws


async def _next_frame(ws: ClientConnection, timeout: float) -> Frame:
    while True:
        raw = await asyncio.wait_for(ws.recv(), timeout)
        if isinstance(raw, str):
            raise VolcError(f"火山引擎：{raw[:200]}")
        frame = decode(raw)
        if frame.kind == ERROR_INFORMATION:
            raise VolcError(f"火山引擎错误：{frame.message()}")
        return frame


async def synthesize(key: str, resource: str, voice: str, text: str, speech_rate: float = 1.0, timeout: float = 30.0) -> bytes:
    """Speak one text, return 24 kHz s16le mono PCM."""
    params = session_params(voice, speech_rate)
    sid = uuid.uuid4().hex
    audio = bytearray()
    ws = await open_connection(key, resource)
    try:
        await ws.send(encode(START_SESSION, sid, request(START_SESSION, params)))
        while True:
            frame = await _next_frame(ws, SESSION_START_TIMEOUT_S)
            if frame.event in (SESSION_FAILED, SESSION_CANCELED, SESSION_FINISHED):
                raise VolcError(f"火山引擎合成失败：{frame.message()}")
            if frame.event == SESSION_STARTED:
                break
        await ws.send(encode(TASK_REQUEST, sid, request(TASK_REQUEST, params, text=text)))
        await ws.send(encode(FINISH_SESSION, sid))
        while True:
            frame = await _next_frame(ws, timeout)
            if frame.kind == AUDIO_ONLY_RESPONSE or frame.event == TTS_RESPONSE:
                audio += frame.payload
            elif frame.event in (SESSION_FAILED, SESSION_CANCELED):
                raise VolcError(f"火山引擎合成失败：{frame.message()}")
            elif frame.event == SESSION_FINISHED:
                break
        await ws.send(encode(FINISH_CONNECTION, None))
        try:
            await _next_frame(ws, 1.0)
        except (WebSocketException, VolcError, asyncio.TimeoutError):
            pass
    finally:
        await ws.close()
    return bytes(audio)


class VolcTts:
    """Same interface as voice_chat.Tts: feed() streaming text, finish() per reply, reset() on barge-in."""

    def __init__(
        self, key: str, resource: str, voice: str, speech_rate: float,
        on_audio: Callable[[bytes], None], log: Callable[[str], None] = print,
    ) -> None:
        self._key, self._resource = key, resource
        self._params = session_params(voice, speech_rate)
        self._on_audio = on_audio
        self._log = log
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._ws: ClientConnection | None = None
        self._sender: asyncio.Task | None = None
        self._session: str | None = None
        self._started = asyncio.Event()
        self._finished = asyncio.Event()
        self._failure = ""

    async def run(self) -> None:
        backoff = 0.5
        while True:
            opened = asyncio.get_running_loop().time()
            try:
                async with await open_connection(self._key, self._resource) as ws:
                    self._ws = ws
                    self._sender = asyncio.create_task(self._send_loop(ws))
                    try:
                        await self._receive_loop(ws)
                    finally:
                        self._sender.cancel()
            except (OSError, WebSocketException, ConnectionError, asyncio.TimeoutError) as exc:
                self._log(f"[tts] 火山引擎: {exc}; reconnecting")
            self._ws, self._session = None, None
            if asyncio.get_running_loop().time() - opened > 5.0:
                backoff = 0.5
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 10.0)

    async def feed(self, text: str) -> None:
        if text:
            self._queue.put_nowait(text)

    async def finish(self) -> None:
        self._queue.put_nowait(None)

    def reset(self) -> None:
        """Barge-in: drop the reply being spoken; a fresh connection opens right away."""
        while not self._queue.empty():
            self._queue.get_nowait()
        if self._sender is not None:
            self._sender.cancel()
        ws, self._ws, self._session = self._ws, None, None
        if ws is not None:
            asyncio.get_running_loop().create_task(ws.close())

    async def _send_loop(self, ws: ClientConnection) -> None:
        skipping = False
        try:
            while True:
                text = await self._queue.get()
                if text is None:
                    skipping = False
                    if self._session is not None:
                        await ws.send(encode(FINISH_SESSION, self._session))
                        # One session at a time per connection: the next reply waits for this one's audio.
                        await asyncio.wait_for(self._finished.wait(), SESSION_FINISH_TIMEOUT_S)
                        self._session = None
                    continue
                if skipping:
                    continue
                if self._session is None and not await self._start_session(ws):
                    skipping = True
                    continue
                await ws.send(encode(TASK_REQUEST, self._session, request(TASK_REQUEST, self._params, text=text)))
        except (WebSocketException, asyncio.TimeoutError) as exc:
            self._log(f"[tts] 火山引擎: {type(exc).__name__} {exc}; reconnecting")
            await ws.close()

    async def _start_session(self, ws: ClientConnection) -> bool:
        sid = uuid.uuid4().hex
        self._started.clear()
        self._finished.clear()
        self._failure = ""
        self._session = sid
        await ws.send(encode(START_SESSION, sid, request(START_SESSION, self._params)))
        await asyncio.wait_for(self._started.wait(), SESSION_START_TIMEOUT_S)
        if self._failure:
            self._session = None
            return False
        return True

    async def _receive_loop(self, ws: ClientConnection) -> None:
        async for raw in ws:
            if self._ws is not ws:
                return
            if isinstance(raw, str):
                self._log(f"[tts] 火山引擎: {raw[:200]}")
                continue
            frame = decode(raw)
            if frame.kind == ERROR_INFORMATION:
                raise VolcError(frame.message())
            if frame.kind == AUDIO_ONLY_RESPONSE or frame.event == TTS_RESPONSE:
                if frame.payload:
                    self._on_audio(frame.payload)
            elif frame.event == SESSION_STARTED:
                self._started.set()
            elif frame.event in (SESSION_FINISHED, SESSION_CANCELED, SESSION_FAILED):
                if frame.event == SESSION_FAILED:
                    self._failure = f"合成失败：{frame.message()}"
                    self._log(f"[tts] 火山引擎{self._failure}")
                self._started.set()
                self._finished.set()
            elif frame.event == CONNECTION_FINISHED:
                return
