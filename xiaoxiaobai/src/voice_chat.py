# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

"""Real-time voice chat with Qwen through the robot's own microphone, speaker and camera.

Two routes, both fully streamed:

--route esp32 (default): the ESP32 talks to Qwen over its own Wi-Fi
    ESP32 PDM mic + camera -> TLS WebSocket -> Qwen-Omni-Realtime (+ Qwen3-TTS-Realtime)
        -> reply PCM -> head board -> amplifier
    This process only hands the ESP32 the session config (API key included, kept in its RAM)
    over the UART, keeps the session alive, and prints what the ESP32 reports.

--route pi: everything goes through the Pi (use it when the ESP32's Wi-Fi is poor)
    ESP32 PDM mic -> UART -> display loop (voice_bridge.py) -> this process
        -> Qwen-Omni-Realtime (server VAD; one camera frame when the user asks for a look)
        -> reply PCM -> display loop -> UART -> head board -> amplifier
    The reply is sent to the ESP32 only ~PLAYBACK_LEAD_S ahead of playback.

Voices the Omni model has (Tina, Momo, ...) come straight from it; any other voice (Mochi
by default) is spoken by Qwen3-TTS-Realtime, fed with the Omni model's streaming text.
Volcengine Doubao voices (zh_female_vv_uranus_bigtts, ...) are spoken by volc_tts.py on the Pi,
so they always use --route pi; their API key comes from $VOLC_TTS_API_KEY.

Needs the display loop (`make run` or the xiaoxiaobai-display service) on the UART backend,
and the API key in $DASHSCOPE_API_KEY or ~/.config/xiaoxiaobai/dashscope_api_key.

A JSON line {"prompt": text} on stdin (from the API) has the model answer `text` aloud as soon
as the user and the robot are quiet, e.g. to ask the user a question a PC task needs answered.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import re
import struct
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from urllib.parse import urlencode, urlsplit

import numpy as np
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import InvalidStatus, WebSocketException

import esp32_link as esp
from camera_stream import camera_port
from esp32_link import (
    CAMERA_SIZES,
    MAX_PAYLOAD,
    MIC_SAMPLE_RATE,
    MSG_MIC,
    MSG_MIC_CHUNK,
    MSG_PCM,
    MSG_PCM_FLUSH,
    MSG_VOICE_CONFIG,
    MSG_VOICE_CONTEXT,
    MSG_VOICE_EVENT,
    MSG_VOICE_STOP,
    SPEAKER_SAMPLE_RATE,
    VOICE_CONFIG_SLICE,
    StreamParser,
    encode_packet,
)
from api.knowledge import KnowledgeBase
from api.memo import voice_context as memo_voice_context
from api.memory import RECALL_HINT, ConversationStore, MemoryStore, recall_text
from expressions import read_expression_name, speech_mood, write_expression_name
from voice_bridge import voice_socket_path
from voice_catalog import is_volc, volc_resource
from volc_tts import API_KEY_ENV as VOLC_KEY_ENV
from volc_tts import VolcTts, synthesize

API_KEY_ENV = "DASHSCOPE_API_KEY"
API_KEY_FILE = Path.home() / ".config" / "xiaoxiaobai" / "dashscope_api_key"
URL_ENV = "XIAOXIAOBAI_QWEN_URL"
# Qianwen AI platform (sk-ws-... keys). Alibaba Cloud Model Studio keys use
# wss://dashscope.aliyuncs.com/api-ws/v1/realtime instead.
DEFAULT_URL = "wss://maas.qianwenaiapi.com/api-ws/v1/realtime"
OMNI_MODEL = "qwen3.8-omni-flash-realtime"
TTS_MODEL = "qwen3-tts-flash-realtime"
DEFAULT_VOICE = "Mochi"
# System voices of qwen3.8-omni-flash-realtime (platform.qianwenai.com omni voice list).
OMNI_VOICES = frozenset(
    "Tina Cindy Sunnybobi Raymond Ethan Zane Katerina Ryan Mia Cici Serena Maia Evan Qiao Momo Wil Angel Joyner Gold "
    "Jennifer Aiden Mione Sunny Dylan Eric Peter Marcus Li Rocky Kiki Sohee Eliška Alek Arda Dolce Lenn Sonrisa "
    "Bodega Andre Rizky Roya Hana Jakub Griet Marina Siiri Ingrid Sigga Bea Chloe Emilien Harvey longanlingxin".split()
    + ["Liora Mira", "Theo Calm", "Li Cassian", "Joseph Chen", "Ono Anna", "Radio Gol"]
)
DEFAULT_INSTRUCTIONS = (
    "你是小小白，一个约 30 厘米高、会走路的开源桌面人形机器人。你用头上的摄像头看世界，用麦克风听人说话。"
    "用简短、自然、口语化的中文回答，通常一两句话；不要使用 Markdown、列表、表情符号或括号注释。"
    "被问到看见了什么时，根据最新的画面回答；这一轮没有收到画面就说没看清，不要编造。"
)

# Knowledge base (the chat's, api/knowledge.py), searched with what the user said once its
# transcript is in. The passages go into the session's instructions and, if the reply has not
# started yet, it is made again with them (about half a second later than without).
KB_PASSAGES = 3
KB_PASSAGE_CHARS = 400
RECALL_VOICE_CHARS = 700
KB_NOTE = (
    "\n\n[知识库资料]（和用户刚才说的话有关：涉及这些内容时以资料为准，用口语讲出来，"
    "不要念标题或编号；资料里没有的照常回答）\n"
)
# Spoken filler, left out of the search so that only content words pick passages.
KB_FILLER = re.compile("|".join(sorted((
    "请问", "请", "帮我", "告诉我", "告诉", "一下", "一个", "这个", "那个", "这些", "那些", "这里", "那里", "哪里", "哪儿",
    "什么", "哪些", "哪个", "怎么样", "怎么", "怎样", "为什么", "多少", "是不是", "有没有", "能不能", "可不可以", "可以", "知道", "现在",
    "今天", "时候", "一点", "东西", "问题", "你们", "我们", "他们", "她们", "它们", "你好",
    "你", "我", "他", "她", "它", "的", "了", "吗", "呢", "吧", "啊", "呀", "哦", "嗯", "么", "是", "谁", "哪", "几",
), key=len, reverse=True)))

# With --search the model searches the web itself (session enable_search) when it judges it needs to.
SEARCH_NOTE = (
    "\n\n你可以联网搜索。用户说“联网搜一下”“上网查查”“搜一搜”之类时一定先搜索再回答；"
    "问到新闻、天气、价格、比分、日期、最新消息等实时或你不确定的事实时也主动搜索，平常闲聊不要搜。"
    "用户说“搜一下这个”“再查查他”时，结合前面的对话确定要搜什么。"
    "搜到后只说用户问的那一个重点，一句话，最多两句，三十字左右：不罗列多款多项，不报日期、来源和网址，"
    "不加建议和反问；用户想知道更多会接着问。"
)

PCM_PACKET_BYTES = MAX_PAYLOAD - 32  # 10 ms at 24 kHz
PLAYBACK_LEAD_S = 0.12
# Microphone stays muted this long after the speaker goes quiet (no echo cancellation).
ECHO_TAIL_S = 0.3
MIC_SILENT_WARN_S = 3.0
# Same list as kVisionWords in the firmware: one frame, only when the live transcript asks for a look.
VISION_WORDS = (
    "看", "瞧", "眼前", "面前", "前面", "这是什么", "这是啥", "这个是", "那是什么", "那是啥", "什么东西",
    "颜色", "拍", "照片", "图片", "画面", "认识", "长什么样", "长啥样", "样子", "镜头", "摄像头", "手里",
    "手上", "拿着", "几根", "几个", "多少个", "穿", "识别", "写的什么", "写了什么", "上面写", "念一下", "读一下",
)
CAPTURE_TIMEOUT_S = 6.0
TTS_SENTENCE_END = frozenset("。！？!?；;…\n")
TTS_CLAUSE_END = frozenset("，,、：:")
# The first clause of a reply goes to the TTS on its own to start speaking sooner.
TTS_FIRST_CLAUSE_CHARS = 6

# Direct route: config layout version and size limit (kVoiceConfigVersion / kVoiceMaxConfigBytes).
DIRECT_CONFIG_VERSION = 1
DIRECT_CONFIG_MAX_BYTES = 8192
# kVoiceMaxContextBytes: largest knowledge-base session.update the ESP32 takes.
DIRECT_CONTEXT_MAX_BYTES = 16384
# Without any report from the ESP32 this long after the config, its firmware predates the direct route.
DIRECT_FIRST_EVENT_S = 5.0
DIRECT_CONFIG_RESEND_S = 2.0
# A prompt waits at most this long for the user and the robot to be quiet.
PROMPT_WAIT_S = 20.0
# The ESP32 keeps only the newest context event: the prompt's two events go out this far apart.
DIRECT_PROMPT_GAP_S = 0.8


def log(message: str) -> None:
    print(message, flush=True)


def load_api_key() -> str | None:
    key = os.environ.get(API_KEY_ENV, "").strip()
    if key:
        return key
    try:
        return API_KEY_FILE.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def event(kind: str, **fields: object) -> str:
    return json.dumps({"event_id": f"event_{uuid.uuid4().hex}", "type": kind, **fields}, ensure_ascii=False)


def user_text(text: str) -> str:
    return event("conversation.item.create", item={"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]})


async def wait_quiet(busy: Callable[[], bool]) -> None:
    deadline = time.monotonic() + PROMPT_WAIT_S
    while busy() and time.monotonic() < deadline:
        await asyncio.sleep(0.2)


async def run_commands(prompt: Callable[[str], Awaitable[None]]) -> None:
    """{"prompt": text} lines from stdin; anything else (typing in a terminal) is ignored."""
    loop = asyncio.get_running_loop()
    lines: asyncio.Queue[str] = asyncio.Queue()

    def pump() -> None:
        # A daemon thread: a blocking read must not hold up the exit.
        for line in sys.stdin:
            loop.call_soon_threadsafe(lines.put_nowait, line)

    if sys.stdin is None or sys.stdin.isatty():
        return
    threading.Thread(target=pump, name="voice-commands", daemon=True).start()
    while True:
        line = await lines.get()
        try:
            command = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = command.get("prompt") if isinstance(command, dict) else None
        if isinstance(text, str) and text.strip():
            await prompt(text.strip())


async def open_realtime(url: str, model: str, key: str) -> ClientConnection:
    return await connect(
        f"{url}?model={model}",
        additional_headers={"Authorization": f"Bearer {key}"},
        compression=None,
        max_size=None,
        open_timeout=10,
        close_timeout=1,
    )


async def configure(ws: ClientConnection, session: dict) -> dict:
    """session.update, then wait for session.updated; raises on an error event."""
    await ws.send(event("session.update", session=session))
    async for raw in ws:
        message = json.loads(raw)
        if message["type"] == "session.updated":
            return message["session"]
        if message["type"] == "error":
            raise ConnectionError(f"session.update rejected: {message.get('error')}")
    raise ConnectionError("connection closed during session.update")


def omni_session(args: argparse.Namespace, extra: str = "") -> dict:
    """The Omni session; `extra` goes at the end of the instructions (knowledge-base passages)."""
    native = args.voice in OMNI_VOICES
    session: dict = {
        "modalities": ["text", "audio"] if native else ["text"],
        "instructions": args.instructions + (SEARCH_NOTE if args.search else "") + extra,
        "audio": {"input": {"format": {"type": "pcm", "sample_rate": MIC_SAMPLE_RATE}}},
        "turn_detection": {
            "type": args.vad,
            "threshold": args.vad_threshold,
            "prefix_padding_ms": 300,
            "silence_duration_ms": args.silence_ms,
        },
    }
    if native:
        session["voice"] = args.voice
        session["audio"]["output"] = {"voice": args.voice, "format": {"type": "pcm", "sample_rate": SPEAKER_SAMPLE_RATE}}
    # Transcription is what decides a look is needed, so a camera session always asks for it.
    if args.show_user_text or args.kb_top_k > 0 or args.memo or args.recall or not args.no_camera:
        session["input_audio_transcription"] = {"model": "qwen3-asr-flash-realtime"}
    if args.search:
        session["enable_search"] = True
    return session


def encode_direct_config(args: argparse.Namespace, key: str) -> bytes:
    """The ESP32's session config (parseVoiceConfig in firmware/xiao_face/xiao_face.ino).

    The ESP32 sends both session.update events verbatim. Its TTS session uses commit mode:
    it commits every sentence itself, so each commit maps to exactly one response, which is
    how it drops the rest of an interrupted reply without reconnecting.
    """
    url = urlsplit(args.url)
    tls = url.scheme == "wss"

    def path_for(model: str) -> str:
        query = f"{url.query}&" if url.query else ""
        return f"{url.path or '/'}?{query}{urlencode({'model': model})}"

    native = args.voice in OMNI_VOICES
    tts_session = {"voice": args.voice, "mode": "commit", "response_format": "pcm", "sample_rate": SPEAKER_SAMPLE_RATE}
    if args.speech_rate != 1.0:
        tts_session["speech_rate"] = args.speech_rate
    flags = (1 if tls else 0) | (2 if args.full_duplex else 0) | (0 if args.no_camera else 4)
    header = struct.pack(
        "<BBHBBBB",
        DIRECT_CONFIG_VERSION,
        flags,
        url.port or (443 if tls else 80),
        max(0, min(100, args.volume)),
        max(0, min(255, round(args.mic_gain * 16))),
        CAMERA_SIZES[args.camera_size][0],
        max(6, min(63, args.camera_quality)),
    )
    fields = [
        url.hostname or "",
        path_for(args.model),
        "" if native else path_for(args.tts_model),
        f"Authorization: Bearer {key}\r\n",
        event("session.update", session=omni_session(args)),
        "" if native else event("session.update", session=tts_session),
    ]
    out = bytearray(header)
    for field in fields:
        data = field.encode()
        out += struct.pack("<H", len(data)) + data
    if len(out) > DIRECT_CONFIG_MAX_BYTES:
        raise SystemExit(f"Voice config is {len(out)} bytes, the ESP32 takes at most {DIRECT_CONFIG_MAX_BYTES}: shorten --instructions.")
    return bytes(out)


def describe_error(exc: BaseException) -> str:
    if isinstance(exc, InvalidStatus):
        status = exc.response.status_code
        hint = " (check the API key and that the endpoint matches it)" if status in (401, 403) else ""
        return f"HTTP {status}{hint}"
    return f"{type(exc).__name__}: {exc}"


def context_label(extra: str) -> str:
    """What a turn's extra instructions hold, for the log."""
    parts = [f"{extra.count('《')} 段相关资料"] if KB_NOTE.strip() in extra else []
    if "[记忆检索]（" in extra:
        parts.append("以前的记忆")
    if "[备忘录" in extra:
        parts.append("备忘录")
    elif "[现在时间]" in extra:
        parts.append("当前时间")
    return "、".join(parts)


class Knowledge:
    """What the user just said brings into the instructions: knowledge-base passages about it,
    with --memo the time and the memos of the day asked about, and with --recall what the
    memories and old conversations hold when asked about the past (empty when none applies)."""

    def __init__(self, top_k: int, memo: bool = False, recall: bool = False) -> None:
        self._top_k = min(top_k, KB_PASSAGES)
        self._memo = memo
        self._recall = recall
        # This session's own dialogue the model already has.
        self._since = time.time()
        self._kb: KnowledgeBase | None = None
        self._version = -1.0
        self._in_session = False

    def update(self, heard: str) -> str | None:
        """Passages for `heard` to put into the session, "" to take the previous ones out, None
        when the session needs no change."""
        passages = self.passages(heard)
        if self._memo and heard.strip():
            passages += memo_voice_context(heard)
        if self._recall and RECALL_HINT.search(heard):
            passages += "\n" + recall_text(MemoryStore(), ConversationStore(), heard, self._since, RECALL_VOICE_CHARS)
        if not passages and not self._in_session:
            return None
        self._in_session = bool(passages)
        return passages

    def passages(self, heard: str) -> str:
        if self._top_k <= 0 or not heard.strip():
            return ""
        version = KnowledgeBase.version()
        if self._kb is None or version != self._version:
            self._kb, self._version = KnowledgeBase(), version
        hits = self._kb.search(KB_FILLER.sub(" ", heard), self._top_k)
        if not hits:
            return ""
        return KB_NOTE + "\n".join(f"《{h['title']}》{h['text'][:KB_PASSAGE_CHARS]}" for h in hits)


class Speaker:
    """Paces 24 kHz PCM to the ESP32 just ahead of playback, so a barge-in only drops ~PLAYBACK_LEAD_S."""

    def __init__(self, writer: asyncio.StreamWriter, volume: int) -> None:
        self._writer = writer
        self._gain = max(0, min(100, volume)) / 100.0
        self._buf = bytearray()
        self._carry = b""
        self._wake = asyncio.Event()
        # When everything sent so far will have finished playing (monotonic time).
        self.play_end = 0.0
        self.on_first_audio: Callable[[], None] | None = None

    @property
    def busy(self) -> bool:
        return bool(self._buf) or time.monotonic() < self.play_end

    def feed(self, pcm: bytes) -> None:
        data = self._carry + pcm
        cut = len(data) - len(data) % 2
        self._carry = data[cut:]
        if not cut:
            return
        if self.on_first_audio is not None:
            self.on_first_audio()
            self.on_first_audio = None
        samples = np.frombuffer(data[:cut], dtype="<i2")
        if self._gain != 1.0:
            samples = (samples.astype(np.float32) * self._gain).astype("<i2")
        self._buf += samples.tobytes()
        self._wake.set()

    def flush(self) -> None:
        self._buf.clear()
        self._carry = b""
        self.play_end = 0.0
        self._writer.write(encode_packet(MSG_PCM_FLUSH, b""))

    async def run(self) -> None:
        while True:
            if not self._buf:
                self._wake.clear()
                await self._wake.wait()
                continue
            now = time.monotonic()
            # Idle or starved: playback (re)starts as soon as the next packet lands.
            self.play_end = max(self.play_end, now)
            ahead = self.play_end - now
            if ahead > PLAYBACK_LEAD_S:
                await asyncio.sleep(ahead - PLAYBACK_LEAD_S)
                continue
            while self._buf and self.play_end - now <= PLAYBACK_LEAD_S:
                chunk = bytes(self._buf[:PCM_PACKET_BYTES])
                del self._buf[:PCM_PACKET_BYTES]
                self._writer.write(encode_packet(MSG_PCM, chunk))
                self.play_end += len(chunk) / (2 * SPEAKER_SAMPLE_RATE)
            await self._writer.drain()


def needs_look(text: str) -> bool:
    return any(word in text for word in VISION_WORDS)


class Camera:
    """One JPEG from the display loop, taken when a turn needs it (camera_stream.py)."""

    def __init__(self, size: str, quality: int, interval_ms: int, rotate: int) -> None:
        del interval_ms  # the one-shot capture does not keep a stream open
        self._size = size
        self._quality = quality
        self._rotate = rotate

    async def grab(self) -> bytes | None:
        url = (
            f"http://127.0.0.1:{camera_port()}/snapshot.jpg"
            f"?size={self._size}&quality={self._quality}&fresh=1"
        )

        def fetch() -> bytes | None:
            try:
                with urllib.request.urlopen(url, timeout=CAPTURE_TIMEOUT_S) as resp:
                    return resp.read() if resp.status == 200 else None
            except (OSError, urllib.error.URLError, TimeoutError):
                return None

        jpeg = await asyncio.to_thread(fetch)
        if jpeg is None or not self._rotate:
            return jpeg
        return await asyncio.to_thread(self._rotated, jpeg)

    def _rotated(self, jpeg: bytes) -> bytes:
        from PIL import Image

        out = io.BytesIO()
        Image.open(io.BytesIO(jpeg)).rotate(-self._rotate, expand=True).save(out, "JPEG", quality=85)
        return out.getvalue()


class Tts:
    """Persistent Qwen3-TTS-Realtime session fed with streaming text; reset() drops what is pending."""

    def __init__(self, url: str, key: str, model: str, voice: str, speech_rate: float, on_audio: Callable[[bytes], None]) -> None:
        self._url, self._key, self._model = url, key, model
        self._session = {"voice": voice, "mode": "server_commit", "response_format": "pcm", "sample_rate": SPEAKER_SAMPLE_RATE}
        if speech_rate != 1.0:
            self._session["speech_rate"] = speech_rate
        self._on_audio = on_audio
        self._ws: ClientConnection | None = None
        self._pending: list[str] = []
        self._pending_finish = False
        self._uncommitted = False
        self._clause_chars = 0
        self._first_clause = True

    async def run(self) -> None:
        backoff = 0.5
        while True:
            failed = False
            try:
                async with await open_realtime(self._url, self._model, self._key) as ws:
                    await configure(ws, self._session)
                    self._ws, backoff = ws, 0.5
                    pending, self._pending = self._pending, []
                    for text in pending:
                        await self._send_text(ws, text)
                    if self._pending_finish:
                        self._pending_finish = False
                        await self.finish()
                    async for raw in ws:
                        if self._ws is not ws:
                            break
                        message = json.loads(raw)
                        if message["type"] == "response.audio.delta":
                            self._on_audio(base64.b64decode(message["delta"]))
                        elif message["type"] == "error":
                            log(f"[tts] {message.get('error')}")
            except (OSError, WebSocketException, ConnectionError, asyncio.TimeoutError) as exc:
                failed = True
                log(f"[tts] {describe_error(exc)}; reconnecting")
            self._ws = None
            if failed:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 10.0)

    async def feed(self, text: str) -> None:
        if self._ws is None:
            self._pending.append(text)
            return
        try:
            await self._send_text(self._ws, text)
        except WebSocketException:
            pass

    async def finish(self) -> None:
        """End of a reply: synthesize whatever is still buffered."""
        if self._ws is None:
            self._pending_finish = bool(self._pending)
            return
        if self._uncommitted:
            try:
                await self._commit(self._ws)
            except WebSocketException:
                pass
        self._first_clause = True

    def reset(self) -> None:
        """Barge-in: drop the reply being spoken; a fresh session connects right away."""
        ws, self._ws = self._ws, None
        self._pending.clear()
        self._pending_finish = False
        self._uncommitted = False
        self._clause_chars = 0
        self._first_clause = True
        if ws is not None:
            asyncio.get_running_loop().create_task(ws.close())

    async def _send_text(self, ws: ClientConnection, text: str) -> None:
        await ws.send(event("input_text_buffer.append", text=text))
        self._uncommitted = True
        self._clause_chars += len(text)
        if any(ch in TTS_SENTENCE_END for ch in text) or (
            self._first_clause and self._clause_chars >= TTS_FIRST_CLAUSE_CHARS and any(ch in TTS_CLAUSE_END for ch in text)
        ):
            await self._commit(ws)

    async def _commit(self, ws: ClientConnection) -> None:
        await ws.send(event("input_text_buffer.commit"))
        self._uncommitted = False
        self._clause_chars = 0
        self._first_clause = False


class Face:
    """Listening / thinking / talking expressions on the LCD, restored on exit."""

    def __init__(self, enabled: bool) -> None:
        self._enabled = enabled
        self._base = read_expression_name() if enabled else ""
        self._current = self._base

    def show(self, name: str | None) -> None:
        name = name or self._base
        if self._enabled and name != self._current:
            try:
                write_expression_name(name)
                self._current = name
            except OSError:
                pass

    def restore(self) -> None:
        self.show(None)


class VoiceChat:
    def __init__(self, args: argparse.Namespace, key: str, speaker: Speaker, camera: Camera | None, face: Face) -> None:
        self._args = args
        self._key = key
        self._speaker = speaker
        self._camera = camera
        self._face = face
        self._native = args.voice in OMNI_VOICES
        self._tts: Tts | VolcTts | None = None
        if is_volc(args.voice):
            self._tts = VolcTts(args.volc_key, volc_resource(args.voice, args.volc_resource), args.voice, args.speech_rate, speaker.feed, log)
        elif not self._native:
            self._tts = Tts(args.url, key, args.tts_model, args.voice, args.speech_rate, speaker.feed)
        self._ws: ClientConnection | None = None
        self._user_speaking = False
        self._awaiting_reply = False
        self._response_id: str | None = None
        self._cancelled: set[str] = set()
        self._audio_sent = False
        self._want_shot = False
        self._shot_sent = False
        self._grabbing = False
        self._shot_gen = 0
        self._mic_chunks = 0
        self._mic_dc = 0.0
        self._printing_reply = False
        self._heard = ""
        self._reply = ""
        self._knowledge = Knowledge(args.kb_top_k, args.memo, args.recall)
        self._redo_on_created = False

    def tasks(self, reader: asyncio.StreamReader) -> list:
        tasks = [self._run_omni(), self._run_mic(reader), self._run_face(), self._warn_silent_mic(), run_commands(self.prompt)]
        if self._tts is not None:
            tasks.append(self._tts.run())
        if self._camera is not None:
            tasks.append(self._run_vision())
        return tasks

    async def _run_omni(self) -> None:
        backoff = 1.0
        while True:
            try:
                async with await open_realtime(self._args.url, self._args.model, self._key) as ws:
                    await configure(ws, omni_session(self._args))
                    engine = "Omni" if self._native else "Volcengine TTS" if is_volc(self._args.voice) else self._args.tts_model
                    log(f"[ready] {self._args.model}, voice {self._args.voice} via {engine}; talk to the robot (Ctrl+C to quit)")
                    self._ws, backoff = ws, 1.0
                    async for raw in ws:
                        await self._on_event(json.loads(raw))
                log("[omni] session closed by the server; reconnecting")
            except (OSError, WebSocketException, ConnectionError, asyncio.TimeoutError) as exc:
                log(f"[omni] {describe_error(exc)}; reconnecting in {backoff:.0f} s")
            self._ws = None
            self._audio_sent = False
            self._user_speaking = self._awaiting_reply = False
            self._response_id = None
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)

    async def _on_event(self, message: dict) -> None:
        kind = message["type"]
        if kind == "input_audio_buffer.speech_started":
            self._user_speaking = True
            self._heard = self._reply = ""
            self._want_shot = self._shot_sent = False
            self._shot_gen += 1
            self._redo_on_created = False
            if self._speaker.busy or self._response_id is not None:
                await self._barge_in()
        elif kind == "input_audio_buffer.speech_stopped":
            self._user_speaking = False
            self._awaiting_reply = True
            stopped = time.monotonic()

            def first_audio() -> None:
                if self._args.verbose:
                    log(f"\n[latency] {1000 * (time.monotonic() - stopped):.0f} ms from end of speech to first reply audio")

            self._speaker.on_first_audio = first_audio
        elif kind == "response.created":
            self._response_id = message.get("response", {}).get("id")
            if self._redo_on_created and self._ws is not None:
                self._redo_on_created = False
                await self._answer_again(self._ws)
        elif kind in ("response.audio_transcript.delta", "response.text.delta"):
            if message.get("response_id") in self._cancelled:
                return
            delta = message.get("delta", "")
            self._reply += delta
            if not self._printing_reply:
                print("机器人: ", end="", flush=True)
                self._printing_reply = True
            print(delta, end="", flush=True)
            if self._tts is not None:
                await self._tts.feed(delta)
        elif kind == "response.audio.delta":
            if self._native and message.get("response_id") not in self._cancelled:
                self._speaker.feed(base64.b64decode(message["delta"]))
        elif kind == "response.done":
            response_id = message.get("response", {}).get("id")
            self._cancelled.discard(response_id)
            if response_id == self._response_id:
                self._response_id = None
                self._awaiting_reply = False
                if self._tts is not None:
                    await self._tts.finish()
            if self._printing_reply:
                print(flush=True)
                self._printing_reply = False
        elif (
            kind.startswith("conversation.item.input_audio_transcription.")
            and not kind.endswith(".completed")
            and self._user_speaking
        ):
            delta = message.get("delta") or ""
            if delta and len(self._heard) < 1024:
                self._heard += delta
            if self._camera is not None and not self._want_shot and not self._shot_sent:
                texts = (self._heard, message.get("text") or "", message.get("stash") or "")
                if any(needs_look(text) for text in texts):
                    self._want_shot = True
                    log("[camera] 这句话要看东西，拍一张")
        elif kind == "conversation.item.input_audio_transcription.completed":
            self._heard = message.get("transcript", "")
            log(f"你: {self._heard}")
            if self._camera is not None and needs_look(self._heard) and not self._shot_sent:
                log("[camera] 说完才听出要看东西，这一轮没带图（把“看看”放在句子前面更稳）")
            await self._use_knowledge(self._heard)
        elif kind == "error":
            log(f"[omni] {message.get('error')}")

    async def _use_knowledge(self, heard: str) -> None:
        """Knowledge-base passages for what was just said into the instructions, and the reply
        made again with them unless it has started."""
        passages = await asyncio.to_thread(self._knowledge.update, heard)
        ws = self._ws
        if passages is None or ws is None:
            return
        session = omni_session(self._args, passages)
        try:
            await ws.send(event("session.update", session=session))
            if not passages:
                return
            if not self._awaiting_reply or self._reply or self._speaker.busy:
                log(f"[knowledge] 找到{context_label(passages)}（回答已开始，留给接下来的问题）")
                return
            log(f"[knowledge] 找到{context_label(passages)}，按资料重新回答")
            if self._response_id is None:
                self._redo_on_created = True
            else:
                await self._answer_again(ws)
        except WebSocketException:
            pass

    async def prompt(self, text: str) -> None:
        await wait_quiet(lambda: self._user_speaking or self._awaiting_reply or self._response_id is not None or self._speaker.busy)
        ws = self._ws
        if ws is None:
            log("[prompt] 千问还没连上，这句没说")
            return
        log(f"[prompt] 请千问说：{text[:40]}")
        try:
            await ws.send(user_text(text))
            await ws.send(event("response.create"))
        except WebSocketException:
            pass

    async def _answer_again(self, ws: ClientConnection) -> None:
        if self._response_id is not None:
            self._cancelled.add(self._response_id)
            self._response_id = None
        await ws.send(event("response.cancel"))
        await ws.send(event("response.create"))

    async def _barge_in(self) -> None:
        self._speaker.flush()
        self._speaker.on_first_audio = None
        if self._tts is not None:
            self._tts.reset()
        if self._response_id is not None:
            self._cancelled.add(self._response_id)
            self._response_id = None
            if self._ws is not None:
                try:
                    await self._ws.send(event("response.cancel"))
                except WebSocketException:
                    pass
        if self._printing_reply:
            print(" [打断]", flush=True)
            self._printing_reply = False

    async def _run_mic(self, reader: asyncio.StreamReader) -> None:
        parser = StreamParser()
        gain = self._args.mic_gain
        while True:
            data = await reader.read(65536)
            if not data:
                raise ConnectionError("voice bridge closed: the display loop stopped")
            for kind, payload in parser.feed(data):
                if kind != MSG_MIC_CHUNK:
                    continue
                self._mic_chunks += 1
                ws = self._ws
                if ws is None:
                    continue
                if not self._args.full_duplex and time.monotonic() < self._speaker.play_end + ECHO_TAIL_S:
                    continue
                # The PDM microphone sits on a DC offset; follow it and take it out.
                samples = np.frombuffer(payload[: len(payload) // 2 * 2], dtype="<i2").astype(np.float32)
                self._mic_dc += 0.1 * (float(samples.mean()) - self._mic_dc) if samples.size else 0.0
                samples = (samples - self._mic_dc) * gain
                payload = np.clip(samples, -32768, 32767).astype("<i2").tobytes()
                try:
                    await ws.send(event("input_audio_buffer.append", audio=base64.b64encode(payload).decode()))
                    self._audio_sent = True
                except WebSocketException:
                    pass

    async def _run_vision(self) -> None:
        while True:
            await asyncio.sleep(0.05)
            # One frame per turn, and only after the live transcript asks for a look.
            # The model wants audio in the buffer before the image, and the image before the turn ends.
            if self._grabbing or self._shot_sent or not self._want_shot:
                continue
            if not self._user_speaking or not self._audio_sent or self._ws is None or self._camera is None:
                continue
            gen = self._shot_gen
            self._grabbing = True
            try:
                jpeg = await self._camera.grab()
            finally:
                self._grabbing = False
            ws = self._ws
            if gen != self._shot_gen or jpeg is None or not self._user_speaking or ws is None:
                if gen == self._shot_gen and not self._user_speaking:
                    self._want_shot = False
                    log("[camera] 你说完时照片还没拍好，这一轮没带图")
                continue
            self._shot_sent = True
            self._want_shot = False
            try:
                await ws.send(event("input_image_buffer.append", image=base64.b64encode(jpeg).decode()))
                log("[camera] 已把这一帧发给千问")
            except WebSocketException:
                pass

    async def _run_face(self) -> None:
        try:
            while True:
                if self._user_speaking:
                    self._face.show("listening")
                elif self._speaker.busy:
                    self._face.show(speech_mood(self._reply, self._heard))
                elif self._awaiting_reply:
                    self._face.show("thinking")
                else:
                    self._face.show(None)
                await asyncio.sleep(0.1)
        finally:
            self._face.restore()

    async def _warn_silent_mic(self) -> None:
        await asyncio.sleep(MIC_SILENT_WARN_S)
        if self._mic_chunks == 0:
            log(
                "[mic] no microphone audio from the ESP32: flash the current firmware (make firmware-upload) "
                "and check the display loop log for a microphone error"
            )


class DirectChat:
    """ESP32 route: hands the ESP32 its session config and follows what it reports."""

    def __init__(self, args: argparse.Namespace, config: bytes, writer: asyncio.StreamWriter, face: Face) -> None:
        self._args = args
        self._config = config
        self._writer = writer
        self._face = face
        self._events = 0
        self._last_config = 0.0
        self._user_speaking = False
        self._awaiting_since: float | None = None
        self._playing = False
        self._heard = ""
        self._reply = ""
        self._printing_reply = False
        self._hinted = False
        self._mic_chunks = 0
        self._mic_square = 0.0
        self._mic_samples = 0
        self._mic_peak = 0
        self._mic_dc = 0.0
        self._knowledge = Knowledge(args.kb_top_k, args.memo, args.recall)

    def send_config(self) -> None:
        total = len(self._config)
        for offset in range(0, total, VOICE_CONFIG_SLICE):
            chunk = self._config[offset : offset + VOICE_CONFIG_SLICE]
            self._writer.write(encode_packet(MSG_VOICE_CONFIG, struct.pack("<HH", offset, total) + chunk))
        self._last_config = time.monotonic()

    def tasks(self, reader: asyncio.StreamReader) -> list:
        return [self._run_events(reader), self._run_face(), self._warn_no_events(), self._report_mic(), self._measure_network(),
                run_commands(self.prompt)]

    async def _run_events(self, reader: asyncio.StreamReader) -> None:
        self.send_config()
        if self._args.debug:
            # A copy of the microphone also comes to the Pi, only to show its level.
            self._writer.write(encode_packet(MSG_MIC, b"\x01"))
        await self._writer.drain()
        parser = StreamParser()
        while True:
            data = await reader.read(65536)
            if not data:
                raise ConnectionError("voice bridge closed: the display loop stopped")
            for kind, payload in parser.feed(data):
                if kind == MSG_VOICE_EVENT and payload:
                    self._events += 1
                    self._on_event(payload[0], payload[1:].decode(errors="replace"))
                elif kind == MSG_MIC_CHUNK and payload:
                    samples = np.frombuffer(payload[: len(payload) // 2 * 2], dtype="<i2").astype(np.float64)
                    self._mic_dc = float(samples.mean())
                    samples = (samples - self._mic_dc) * self._args.mic_gain
                    self._mic_chunks += 1
                    self._mic_square += float(np.sum(samples * samples))
                    self._mic_samples += samples.size
                    self._mic_peak = max(self._mic_peak, int(min(32767, np.max(np.abs(samples)))))
            await self._writer.drain()

    async def _report_mic(self) -> None:
        """Microphone level as heard on the Pi, so a dead or too quiet microphone shows up."""
        if not self._args.debug:
            return
        while True:
            await asyncio.sleep(3.0)
            chunks, samples = self._mic_chunks, self._mic_samples
            rms = (self._mic_square / samples) ** 0.5 if samples else 0.0
            peak = self._mic_peak
            self._mic_chunks = self._mic_samples = self._mic_peak = 0
            self._mic_square = 0.0
            if chunks == 0:
                log("[mic] 3 秒内树莓派没收到任何麦克风数据：ESP32 的麦克风没在工作（make run 开头看有没有 mic FAILED）")
                continue
            bar = "#" * min(30, int(rms / 300))
            log(f"[mic] 音量 rms {rms:5.0f} 峰值 {peak:5d}（满量程 32767，已乘增益 {self._args.mic_gain:g}，直流偏置 {self._mic_dc:.0f}）{bar}")

    def _end_reply_line(self, suffix: str = "") -> None:
        if self._printing_reply:
            print(suffix, flush=True)
            self._printing_reply = False

    def _on_event(self, kind: int, text: str) -> None:
        if kind == esp.VOICE_EVENT_STATUS:
            self._end_reply_line()
            if text.startswith("ready"):
                engine = "Omni" if self._args.voice in OMNI_VOICES else self._args.tts_model
                log(f"[ready] ESP32 direct, {self._args.model}, voice {self._args.voice} via {engine}; {text}")
                log("对着脸上的麦克风说完一句再停一秒。听到了会先打印 [heard]（Ctrl+C 结束）")
            else:
                log(f"[esp32] {text}")
        elif kind == esp.VOICE_EVENT_ERROR:
            self._end_reply_line()
            log(f"[esp32] {text}")
            if not self._hinted:
                log("[esp32] if the ESP32's Wi-Fi is the problem, use the Pi route instead: --route pi")
                self._hinted = True
        elif kind == esp.VOICE_EVENT_SPEECH_STARTED:
            self._user_speaking = True
            self._heard = self._reply = ""
            self._awaiting_since = None
            self._end_reply_line()
            log("[heard] 听到你说话了")
        elif kind == esp.VOICE_EVENT_SPEECH_STOPPED:
            self._user_speaking = False
            self._awaiting_since = time.monotonic()
            log("[heard] 说完了，正在想")
        elif kind == esp.VOICE_EVENT_REPLY_TEXT:
            self._reply += text
            if not self._printing_reply:
                print("机器人: ", end="", flush=True)
                self._printing_reply = True
            print(text, end="", flush=True)
        elif kind == esp.VOICE_EVENT_REPLY_DONE:
            self._end_reply_line()
        elif kind == esp.VOICE_EVENT_USER_TEXT:
            self._end_reply_line()
            self._heard = text
            log(f"你: {text}")
            self._send_knowledge(text)
        elif kind == esp.VOICE_EVENT_PLAYING:
            self._playing = text == "1"
            if self._playing:
                self._awaiting_since = None
        elif kind == esp.VOICE_EVENT_LATENCY:
            if self._args.debug:
                self._end_reply_line()
                log(f"[latency] {text} ms from end of speech to first reply audio (measured on the ESP32)")
        elif kind == esp.VOICE_EVENT_INTERRUPTED:
            self._awaiting_since = None
            self._end_reply_line(" [打断]")
        elif kind == esp.VOICE_EVENT_DEBUG and text.startswith("timing "):
            self._end_reply_line()
            self._log_timing(text)
        elif kind == esp.VOICE_EVENT_DEBUG and text.startswith("vision: "):
            self._end_reply_line()
            if text.startswith('vision: taking a picture for "'):
                log(f"[camera] 听到“{text[30:].rstrip(chr(34))}”，拍一张照片给千问")
            elif "not ready" in text:
                log("[camera] 你说完时照片还没拍好，这一轮没带图")
            else:
                log("[camera] 说完才听出要看东西，这一轮没带图（把“看看”放在句子前面更稳）")
        elif kind == esp.VOICE_EVENT_DEBUG:
            if self._args.debug:
                self._end_reply_line()
                log(f"[esp32] {text}")
        elif kind == esp.VOICE_EVENT_NEED_CONFIG:
            if time.monotonic() - self._last_config > DIRECT_CONFIG_RESEND_S:
                log("[esp32] restarted; sending the voice config again")
                self.send_config()

    def _send_knowledge(self, heard: str) -> None:
        """Knowledge-base passages for what was just said to the ESP32, which puts them into the
        session and has the reply made again with them unless it has started."""
        passages = self._knowledge.update(heard)
        if passages is None:
            return
        if not self._send_context(event("session.update", session=omni_session(self._args, passages)), bool(passages)):
            log("[knowledge] 资料太长，这一轮不带")
            return
        if passages:
            log(f"[knowledge] 找到{context_label(passages)}，交给 ESP32（还没开口就按资料重新回答）")

    def _send_context(self, message: str, redo: bool = False) -> bool:
        """One event for the ESP32 to send into its Qwen session as is; with `redo` the reply to what
        the user just said is made again unless it has started. False when it is too long."""
        data = bytes([1 if redo else 0]) + message.encode()
        if len(data) > DIRECT_CONTEXT_MAX_BYTES:
            return False
        for offset in range(0, len(data), VOICE_CONFIG_SLICE):
            chunk = data[offset : offset + VOICE_CONFIG_SLICE]
            self._writer.write(encode_packet(MSG_VOICE_CONTEXT, struct.pack("<HH", offset, len(data)) + chunk))
        return True

    async def prompt(self, text: str) -> None:
        await wait_quiet(lambda: self._user_speaking or self._playing or self._printing_reply or self._awaiting_since is not None)
        log(f"[prompt] 请千问说：{text[:40]}")
        self._send_context(user_text(text[:4000]))
        await self._writer.drain()
        await asyncio.sleep(DIRECT_PROMPT_GAP_S)
        self._send_context(event("response.create"))
        await self._writer.drain()

    def _log_timing(self, text: str) -> None:
        """Where the wait between the end of the user's speech and the reply went (ESP32 clock)."""
        values = dict(item.split("=", 1) for item in text.split()[1:] if "=" in item)
        ms = {name: int(value) for name, value in values.items() if value.lstrip("-").isdigit()}
        steps = [
            ("transcript", "识别出你说的话"),
            ("created", "千问开始生成回复"),
            ("text", "千问吐出第一个字（模型思考）"),
            ("commit", "凑够第一句，送去合成语音"),
            ("tts", "语音合成服务开始合成"),
            ("audio", "第一段声音到达 ESP32，开始播放"),
        ]
        present = sorted(((ms[name], label) for name, label in steps if ms.get(name, -1) >= 0), key=lambda step: step[0])
        log(f"[timing] 你停下后，千问先等 {self._args.silence_ms} ms 静音才算你说完；之后：")
        previous = 0
        for at, label in present:
            log(f"[timing]   {at:5d} ms（这一步 +{at - previous:4d}）{label}")
            previous = at
        if ms.get("images", 0):
            log(
                f"[timing]   说话时发了 {ms['images']} 张照片，共 {ms.get('image_bytes', 0) // 1024} KB，"
                f"发送占了 ESP32 {ms.get('image_send_ms', 0)} ms"
            )
        backlog, dropped = ms.get("mic_backlog_ms", 0), ms.get("mic_dropped", 0)
        if backlog >= 200 or dropped:
            log(
                f"[timing]   ESP32 上传跟不上：麦克风声音最多积压 {backlog} ms 才发出去，丢了 {dropped * 20} ms 声音"
                "（Wi-Fi 上行慢；千问那边听到的比你说的晚这么多）"
            )
        total = ms.get("audio", 0) + self._args.silence_ms
        log(f"[timing]   合计：你停下到机器人出声 ≈ {total} ms")

    async def _measure_network(self) -> None:
        """Round trip to the Qwen host from the Pi (same Wi-Fi as the ESP32), as a baseline."""
        url = urlsplit(self._args.url)
        host, port = url.hostname or "", url.port or (443 if url.scheme == "wss" else 80)
        times = []
        for _ in range(3):
            start = time.monotonic()
            try:
                _, writer = await asyncio.wait_for(asyncio.open_connection(host, port), 5.0)
            except (OSError, TimeoutError):
                continue
            times.append((time.monotonic() - start) * 1000)
            writer.close()
            await asyncio.sleep(0.2)
        if times:
            log(f"[net] 树莓派到 {host} 的网络往返约 {min(times):.0f} ms（3 次取最小，ESP32 在同一个 Wi-Fi 上）")
        else:
            log(f"[net] 树莓派连不上 {host}:{port}")

    async def _run_face(self) -> None:
        try:
            while True:
                if self._awaiting_since is not None and time.monotonic() - self._awaiting_since > 10.0:
                    self._awaiting_since = None
                if self._user_speaking:
                    self._face.show("listening")
                elif self._playing:
                    self._face.show(speech_mood(self._reply, self._heard))
                elif self._awaiting_since is not None:
                    self._face.show("thinking")
                else:
                    self._face.show(None)
                await asyncio.sleep(0.1)
        finally:
            self._face.restore()

    async def _warn_no_events(self) -> None:
        await asyncio.sleep(DIRECT_FIRST_EVENT_S)
        if self._events == 0:
            log(
                "[esp32] no report from the ESP32: flash the current firmware (make firmware-upload) "
                "and restart the display loop, or use --route pi"
            )


async def open_bridge() -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    path = voice_socket_path()
    try:
        return await asyncio.open_unix_connection(str(path))
    except OSError as exc:
        raise SystemExit(
            f"Cannot reach the voice bridge at {path} ({exc.strerror}): start `make run` or the "
            "xiaoxiaobai-display service first (UART backend, current code)."
        ) from None


async def run_direct(args: argparse.Namespace, key: str) -> None:
    config = encode_direct_config(args, key)
    if args.camera_rotate:
        log("[camera] --camera-rotate only applies to --route pi; the ESP32 sends its frames as they are")
    reader, writer = await open_bridge()
    chat = DirectChat(args, config, writer, Face(not args.no_face))
    try:
        await asyncio.gather(*chat.tasks(reader))
    finally:
        writer.write(encode_packet(MSG_VOICE_STOP, b""))
        writer.close()


async def run_chat(args: argparse.Namespace, key: str) -> None:
    reader, writer = await open_bridge()
    # The bridge keeps the ESP32 microphone streaming to us until we disconnect.
    writer.write(encode_packet(MSG_MIC, b"\x01"))
    speaker = Speaker(writer, args.volume)
    camera = None if args.no_camera else Camera(args.camera_size, args.camera_quality, args.camera_interval_ms, args.camera_rotate)
    chat = VoiceChat(args, key, speaker, camera, Face(not args.no_face))
    tasks = [speaker.run(), *chat.tasks(reader)]
    try:
        await asyncio.gather(*tasks)
    finally:
        writer.close()


async def run_check(args: argparse.Namespace, key: str) -> None:
    """Connectivity and latency check without the robot."""
    native = args.voice in OMNI_VOICES
    async with await open_realtime(args.url, args.model, key) as ws:
        session = {"modalities": ["text", "audio"] if native else ["text"], "turn_detection": None, "instructions": args.instructions}
        if native:
            session["audio"] = {"output": {"voice": args.voice, "format": {"type": "pcm", "sample_rate": SPEAKER_SAMPLE_RATE}}}
        await configure(ws, session)
        log(f"[check] {args.model}: session ok")
        start = time.monotonic()
        await ws.send(user_text("你好，用一句话介绍你自己。"))
        await ws.send(event("response.create"))
        first_text = first_audio = None
        text = ""
        async for raw in ws:
            message = json.loads(raw)
            kind = message["type"]
            if kind in ("response.text.delta", "response.audio_transcript.delta"):
                first_text = first_text or time.monotonic() - start
                text += message.get("delta", "")
            elif kind == "response.audio.delta":
                first_audio = first_audio or time.monotonic() - start
            elif kind == "error":
                raise ConnectionError(message.get("error"))
            elif kind == "response.done":
                break
        log(f"[check] first text after {1000 * (first_text or 0):.0f} ms: {text}")
        if native:
            log(f"[check] first audio ({args.voice}) after {1000 * (first_audio or 0):.0f} ms")
            return
    if is_volc(args.voice):
        resource = volc_resource(args.voice, args.volc_resource)
        start = time.monotonic()
        pcm = await synthesize(args.volc_key, resource, args.voice, text or "你好，我是小小白。", args.speech_rate)
        log(f"[check] Volcengine {resource} voice {args.voice}: {1000 * (time.monotonic() - start):.0f} ms, {len(pcm) / (2 * SPEAKER_SAMPLE_RATE):.1f} s of speech")
        return
    async with await open_realtime(args.url, args.tts_model, key) as ws:
        await configure(ws, {"voice": args.voice, "mode": "server_commit", "response_format": "pcm", "sample_rate": SPEAKER_SAMPLE_RATE})
        start = time.monotonic()
        await ws.send(event("input_text_buffer.append", text=text or "你好，我是小机器人。"))
        await ws.send(event("input_text_buffer.commit"))
        first_audio, total = None, 0
        async for raw in ws:
            message = json.loads(raw)
            if message["type"] == "response.audio.delta":
                first_audio = first_audio or time.monotonic() - start
                total += len(base64.b64decode(message["delta"]))
            elif message["type"] == "error":
                raise ConnectionError(message.get("error"))
            elif message["type"] == "response.done":
                break
        log(f"[check] {args.tts_model} voice {args.voice}: first audio after {1000 * (first_audio or 0):.0f} ms, {total / (2 * SPEAKER_SAMPLE_RATE):.1f} s of speech")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--route",
        default="esp32",
        choices=("esp32", "pi"),
        help="esp32: the ESP32 talks to Qwen over its own Wi-Fi (default); pi: relay everything through the Pi",
    )
    parser.add_argument("--voice", default=DEFAULT_VOICE, help="Omni voice, Qwen3-TTS voice or Volcengine voice ID (default %(default)s)")
    parser.add_argument("--model", default=OMNI_MODEL, help="Omni realtime model (default %(default)s)")
    parser.add_argument("--tts-model", default=TTS_MODEL, help="TTS model for non-Omni voices (default %(default)s)")
    parser.add_argument("--volc-resource", default="", help="Volcengine X-Api-Resource-Id (default: from the voice ID, e.g. seed-tts-2.0)")
    parser.add_argument("--url", default=os.environ.get(URL_ENV) or DEFAULT_URL, help="realtime WebSocket endpoint (default %(default)s)")
    parser.add_argument("--instructions", default=DEFAULT_INSTRUCTIONS, help="system prompt")
    parser.add_argument("--vad", default="server_vad", choices=("server_vad", "semantic_vad", "semantic_vad_ec"))
    parser.add_argument("--vad-threshold", type=float, default=0.2, help="-1..1, higher ignores more noise (default %(default)s)")
    parser.add_argument("--silence-ms", type=int, default=500, help="silence that ends your turn, 200-6000 (default %(default)s)")
    parser.add_argument("--debug", action="store_true", help="microphone levels and every event the ESP32 reports")
    parser.add_argument("--volume", type=int, default=100, help="speaker volume 0-100 (default %(default)s)")
    parser.add_argument("--mic-gain", type=float, default=4.0, help="microphone gain (default %(default)s)")
    parser.add_argument("--speech-rate", type=float, default=1.0, help="TTS voices only, 0.5-2.0 (default %(default)s)")
    parser.add_argument("--full-duplex", action="store_true", help="keep the microphone open while the robot talks (interruptible, but it may hear itself)")
    parser.add_argument("--no-camera", action="store_true", help="voice only")
    # Every pixel is image tokens the model reads before it answers: small frames reply sooner.
    parser.add_argument("--camera-size", default="qvga", choices=("qqvga", "qvga", "hvga", "vga"))
    parser.add_argument("--camera-quality", type=int, default=12, help="JPEG quality 6-63, lower is better (default %(default)s)")
    parser.add_argument("--camera-interval-ms", type=int, default=500, help="ignored; the camera no longer stays on between frames")
    parser.add_argument("--camera-rotate", type=int, default=0, choices=(0, 90, 180, 270), help="--route pi: clockwise rotation of the picture sent to the model")
    parser.add_argument("--no-face", action="store_true", help="do not change the LCD expression")
    parser.add_argument("--show-user-text", action="store_true", help="also transcribe what you say")
    parser.add_argument("--search", action="store_true", help="let the model search the web")
    parser.add_argument("--kb-top-k", type=int, default=0, help="knowledge-base passages per question, 0 off (default %(default)s)")
    parser.add_argument("--memo", action="store_true", help="answer from the memos (api/memo.py) when asked about a day, and tell the time")
    parser.add_argument("--recall", action="store_true", help="search the memories and old conversations (api/memory.py) when asked about the past")
    parser.add_argument("--verbose", action="store_true", help="print the reply latency of every turn")
    parser.add_argument("--check", action="store_true", help="test the API key, models and voice without the robot, then exit")
    args = parser.parse_args()

    key = load_api_key()
    if key is None:
        raise SystemExit(f"No API key: set ${API_KEY_ENV} or write it to {API_KEY_FILE} (chmod 600).")
    if urlsplit(args.url).scheme not in ("ws", "wss"):
        raise SystemExit(f"--url must be a ws:// or wss:// endpoint (got {args.url})")
    args.volc_key = os.environ.get(VOLC_KEY_ENV, "").strip()
    if is_volc(args.voice):
        if not args.volc_key:
            raise SystemExit(f"Voice {args.voice} is a Volcengine voice: set ${VOLC_KEY_ENV} to its API key.")
        if args.route == "esp32" and not args.check:
            log("[voice] Volcengine voices are spoken by the Pi: using --route pi")
            args.route = "pi"
    if args.check:
        runner = run_check(args, key)
    elif args.route == "esp32":
        runner = run_direct(args, key)
    else:
        runner = run_chat(args, key)
    try:
        asyncio.run(runner)
    except KeyboardInterrupt:
        pass
    except (OSError, WebSocketException, ConnectionError) as exc:
        print(f"\n{describe_error(exc)}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
