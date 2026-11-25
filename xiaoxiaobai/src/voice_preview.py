# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

"""One-shot speech in any of the robot's voices (settings page preview), as 24 kHz s16le PCM."""

from __future__ import annotations

import base64
import json

from voice_catalog import engine_of, is_volc, volc_resource

PREVIEW_INSTRUCTIONS = "你是语音试听助手。把用户发来的文字原样朗读出来，不要增删任何内容，也不要回答问题。"


async def _realtime_pcm(ws, timeout: float) -> bytes:
    import asyncio

    audio = bytearray()
    while True:
        message = json.loads(await asyncio.wait_for(ws.recv(), timeout))
        kind = message["type"]
        if kind == "response.audio.delta":
            audio += base64.b64decode(message["delta"])
        elif kind == "error":
            raise ConnectionError(f"千问：{(message.get('error') or {}).get('message') or message.get('error')}")
        elif kind == "response.done":
            return bytes(audio)


async def speak(
    voice: str, text: str, *, speech_rate: float = 1.0, qwen_key: str = "", qwen_url: str = "",
    volc_key: str = "", volc_resource_id: str = "", timeout: float = 30.0,
) -> bytes:
    if is_volc(voice):
        from volc_tts import synthesize

        if not volc_key:
            raise PermissionError("还没有填写火山引擎 API Key")
        return await synthesize(volc_key, volc_resource(voice, volc_resource_id), voice, text, speech_rate, timeout)

    from voice_chat import DEFAULT_URL, OMNI_MODEL, OMNI_VOICES, SPEAKER_SAMPLE_RATE, TTS_MODEL, configure, event, open_realtime

    if not qwen_key:
        raise PermissionError("还没有配置千问（DashScope）API Key")
    url = qwen_url or DEFAULT_URL
    if voice in OMNI_VOICES or engine_of(voice) == "omni":
        async with await open_realtime(url, OMNI_MODEL, qwen_key) as ws:
            output = {"voice": voice, "format": {"type": "pcm", "sample_rate": SPEAKER_SAMPLE_RATE}}
            await configure(ws, {"modalities": ["text", "audio"], "voice": voice, "turn_detection": None,
                                 "instructions": PREVIEW_INSTRUCTIONS, "audio": {"output": output}})
            item = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}
            await ws.send(event("conversation.item.create", item=item))
            await ws.send(event("response.create"))
            return await _realtime_pcm(ws, timeout)
    session = {"voice": voice, "mode": "server_commit", "response_format": "pcm", "sample_rate": SPEAKER_SAMPLE_RATE}
    if speech_rate != 1.0:
        session["speech_rate"] = speech_rate
    async with await open_realtime(url, TTS_MODEL, qwen_key) as ws:
        await configure(ws, session)
        await ws.send(event("input_text_buffer.append", text=text))
        await ws.send(event("input_text_buffer.commit"))
        return await _realtime_pcm(ws, timeout)
