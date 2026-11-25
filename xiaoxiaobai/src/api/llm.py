# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

"""OpenAI-compatible chat completions (streaming, tool calls) for the providers people actually use."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
import uuid
from typing import Any, Iterator

from api.protocol import ApiError, CommonState, Reason
from api.store import is_masked, mask_secret

PRESETS: list[dict[str, Any]] = [
    {"id": "qianwen", "name": "千问平台（sk-ws- Key）", "base_url": "https://maas.qianwenaiapi.com/compatible-mode/v1",
     "models": ["qwen-plus", "qwen-turbo", "qwen-max"], "key_url": "https://platform.qianwenai.com"},
    {"id": "dashscope", "name": "通义千问（阿里云百炼）", "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
     "models": ["qwen-plus", "qwen-turbo", "qwen-max", "qwen3-235b-a22b"], "key_url": "https://bailian.console.aliyun.com/?apiKey=1"},
    {"id": "deepseek", "name": "DeepSeek", "base_url": "https://api.deepseek.com/v1",
     "models": ["deepseek-chat", "deepseek-reasoner"], "key_url": "https://platform.deepseek.com/api_keys"},
    {"id": "moonshot", "name": "Kimi（月之暗面）", "base_url": "https://api.moonshot.cn/v1",
     "models": ["kimi-latest", "moonshot-v1-8k", "moonshot-v1-32k"], "key_url": "https://platform.moonshot.cn/console/api-keys"},
    {"id": "zhipu", "name": "智谱 GLM", "base_url": "https://open.bigmodel.cn/api/paas/v4",
     "models": ["glm-4-flash", "glm-4-plus", "glm-4-air"], "key_url": "https://open.bigmodel.cn/usercenter/apikeys"},
    {"id": "doubao", "name": "豆包（火山方舟）", "base_url": "https://ark.cn-beijing.volces.com/api/v3",
     "models": [], "key_url": "https://console.volcengine.com/ark", "model_hint": "填写方舟控制台里的推理接入点 ID（ep-…）或模型名"},
    {"id": "siliconflow", "name": "硅基流动 SiliconFlow", "base_url": "https://api.siliconflow.cn/v1",
     "models": ["Qwen/Qwen2.5-7B-Instruct", "deepseek-ai/DeepSeek-V3"], "key_url": "https://cloud.siliconflow.cn/account/ak"},
    {"id": "openai", "name": "OpenAI", "base_url": "https://api.openai.com/v1",
     "models": ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o"], "key_url": "https://platform.openai.com/api-keys"},
    {"id": "anthropic", "name": "Anthropic Claude", "base_url": "https://api.anthropic.com/v1",
     "models": ["claude-sonnet-4-5", "claude-haiku-4-5"], "key_url": "https://console.anthropic.com/settings/keys"},
    {"id": "gemini", "name": "Google Gemini", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
     "models": ["gemini-2.5-flash", "gemini-2.5-pro"], "key_url": "https://aistudio.google.com/apikey"},
    {"id": "openrouter", "name": "OpenRouter", "base_url": "https://openrouter.ai/api/v1",
     "models": ["openai/gpt-4o-mini", "deepseek/deepseek-chat"], "key_url": "https://openrouter.ai/keys"},
    {"id": "ollama", "name": "Ollama（本地）", "base_url": "http://127.0.0.1:11434/v1",
     "models": ["qwen2.5:3b", "qwen2.5:7b"], "key_url": "", "no_key": True},
    {"id": "custom", "name": "自定义（OpenAI 兼容）", "base_url": "", "models": [], "key_url": ""},
]
PRESET_BY_ID = {p["id"]: p for p in PRESETS}
CONNECT_TIMEOUT_S = 30

# Qwen's built-in web search (enable_search): the model searches when it judges it needs to, or
# always with forced_search. Billed per search by the platform.
QWEN_PRESETS = ("qianwen", "dashscope")
QWEN_HOSTS = ("dashscope", "qianwenai", "aliyuncs")
SEARCH_STRATEGY = "turbo"
WEB_SEARCH_MAX_TOKENS = 400
# (base_url, model) that rejected enable_search; they get the web_search tool instead.
_NO_NATIVE_SEARCH: set[tuple[str, str]] = set()


class LlmError(Exception):
    pass


class SearchUnsupported(LlmError):
    """The provider rejected enable_search before replying (nothing was streamed)."""


def public_provider(provider: dict[str, Any]) -> dict[str, Any]:
    return {**provider, "api_key": mask_secret(provider.get("api_key", "")), "has_key": bool(provider.get("api_key"))}


def provider_for_qwen_key(key: str) -> dict[str, Any]:
    """Text chat on the same Qwen key as voice: sk-ws- keys belong to the Qianwen platform, others to DashScope."""
    preset = "qianwen" if key.startswith("sk-ws-") else "dashscope"
    return normalize_provider({"preset": preset, "name": f"{PRESET_BY_ID[preset]['name']}（与语音共用）", "api_key": key})


def normalize_provider(values: dict[str, Any], existing: dict[str, Any] | None = None) -> dict[str, Any]:
    preset = PRESET_BY_ID.get(values.get("preset") or (existing or {}).get("preset") or "custom", PRESET_BY_ID["custom"])
    out = dict(existing or {"id": f"llm_{uuid.uuid4().hex[:8]}", "preset": preset["id"]})
    for key in ("name", "base_url", "model", "preset"):
        if key in values and values[key] is not None:
            out[key] = str(values[key]).strip()
    if "api_key" in values and not is_masked(values["api_key"]):
        out["api_key"] = str(values["api_key"] or "").strip()
    if "enabled" in values:
        out["enabled"] = bool(values["enabled"])
    out.setdefault("name", preset["name"])
    out.setdefault("base_url", preset["base_url"])
    out.setdefault("model", (preset["models"] or [""])[0])
    out.setdefault("api_key", "")
    out.setdefault("enabled", True)
    out["base_url"] = out["base_url"].rstrip("/")
    if not out["base_url"].startswith(("http://", "https://")):
        raise ApiError("Base URL 需要以 http:// 或 https:// 开头", CommonState.INVALID)
    return out


def _request(provider: dict[str, Any], path: str, body: dict[str, Any] | None, timeout: float) -> Any:
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream" if body and body.get("stream") else "application/json"}
    if provider.get("api_key"):
        headers["Authorization"] = f"Bearer {provider['api_key']}"
    if provider.get("preset") == "openrouter":
        headers["X-Title"] = "XiaoXiaoBai"
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(provider["base_url"] + path, data=data, headers=headers, method="POST" if data else "GET")
    try:
        return urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:600]
        try:
            parsed = json.loads(detail)
            err = parsed.get("error", parsed)
            detail = err.get("message", detail) if isinstance(err, dict) else str(err)
        except (json.JSONDecodeError, AttributeError):
            pass
        raise LlmError(f"HTTP {exc.code}：{detail}") from None
    except urllib.error.URLError as exc:
        raise LlmError(f"连接失败：{exc.reason}") from None
    except TimeoutError:
        raise LlmError("连接超时") from None


def native_search(provider: dict[str, Any] | None) -> bool:
    """Whether the provider is a Qwen model with built-in web search (not known to reject it)."""
    if not provider or not provider.get("model"):
        return False
    base, model = provider.get("base_url", ""), provider["model"]
    qwen_host = provider.get("preset") in QWEN_PRESETS or any(h in base for h in QWEN_HOSTS)
    return qwen_host and model.lower().startswith("qwen") and (base, model) not in _NO_NATIVE_SEARCH


def stream_chat(
    provider: dict[str, Any],
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    search: str | None = None,
) -> Iterator[tuple[str, Any]]:
    """Yields ("delta", text), ("reasoning", text), then ("done", {"tool_calls", "finish_reason", "usage"}).

    search: None, "auto" (the model decides) or "force", for providers with native_search(); raises
    SearchUnsupported, before yielding anything, if the provider rejects it.
    """
    body: dict[str, Any] = {
        "model": provider["model"],
        "messages": messages,
        "stream": True,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools:
        body["tools"] = tools
    if search:
        body["enable_search"] = True
        body["search_options"] = {"search_strategy": SEARCH_STRATEGY, "forced_search": search == "force"}
    try:
        resp = _request(provider, "/chat/completions", body, CONNECT_TIMEOUT_S)
    except LlmError as exc:
        if search and str(exc).startswith("HTTP 400"):
            _NO_NATIVE_SEARCH.add((provider.get("base_url", ""), provider["model"]))
            raise SearchUnsupported(str(exc)) from None
        if tools and str(exc).startswith("HTTP 400") and "tool" in str(exc).lower():
            yield ("notice", "该模型不支持工具调用，已关闭机器人控制工具")
            yield from stream_chat(provider, messages, tools=None, temperature=temperature, max_tokens=max_tokens)
            return
        raise
    calls: dict[int, dict[str, Any]] = {}
    finish, usage = None, None
    with resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if chunk.get("error"):
                raise LlmError(str(chunk["error"].get("message", chunk["error"])))
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("reasoning_content"):
                    yield ("reasoning", delta["reasoning_content"])
                if delta.get("content"):
                    yield ("delta", delta["content"])
                for call in delta.get("tool_calls") or []:
                    slot = calls.setdefault(call.get("index", len(calls)), {"id": "", "name": "", "arguments": ""})
                    slot["id"] = call.get("id") or slot["id"]
                    fn = call.get("function") or {}
                    slot["name"] += fn.get("name") or ""
                    slot["arguments"] += fn.get("arguments") or ""
                finish = choice.get("finish_reason") or finish
    tool_calls = [
        {"id": c["id"] or f"call_{i}", "name": c["name"], "arguments": c["arguments"]}
        for i, c in sorted(calls.items())
        if c["name"]
    ]
    yield ("done", {"tool_calls": tool_calls, "finish_reason": finish, "usage": usage})


def complete(
    provider: dict[str, Any], messages: list[dict[str, Any]], max_tokens: int = 512, temperature: float = 0.3, search: str | None = None,
) -> str:
    text = []
    for kind, value in stream_chat(provider, messages, temperature=temperature, max_tokens=max_tokens, search=search):
        if kind == "delta":
            text.append(value)
    return "".join(text)


def web_search(provider: dict[str, Any], query: str, context: list[dict[str, Any]] | None = None) -> str:
    """Search the web through a native_search() provider; returns the facts found, briefly."""
    lines = [f"{'用户' if m['role'] == 'user' else '助手'}：{m['content'][:200]}" for m in (context or [])[-4:]]
    prompt = (
        f"联网搜索并回答：{query}\n"
        + (f"（前面的对话，用来理解指代：\n" + "\n".join(lines) + "）\n" if lines else "")
        + f"当前时间 {time.strftime('%Y-%m-%d %H:%M')}。只列出搜到的关键事实和数字，注明时间，不超过 200 字。"
    )
    return complete(provider, [{"role": "user", "content": prompt}], max_tokens=WEB_SEARCH_MAX_TOKENS, temperature=0.2, search="force").strip()


def list_models(provider: dict[str, Any]) -> list[str]:
    with _request(provider, "/models", None, 15) as resp:
        data = json.loads(resp.read().decode("utf-8", "replace"))
    items = data.get("data", data.get("models", [])) if isinstance(data, dict) else data
    names = [m.get("id") or m.get("name") for m in items if isinstance(m, dict)]
    return sorted({n.removeprefix("models/") for n in names if n})


def test_provider(provider: dict[str, Any]) -> dict[str, Any]:
    if not provider.get("model"):
        raise ApiError("请先填写模型名", CommonState.INVALID)
    started = time.monotonic()
    first = None
    text = []
    try:
        for kind, value in stream_chat(provider, [{"role": "user", "content": "用一句话打个招呼。"}], max_tokens=48):
            if kind == "delta":
                first = first or time.monotonic()
                text.append(value)
    except LlmError as exc:
        raise ApiError(str(exc), CommonState.FAILURE, Reason.UPSTREAM, 502) from None
    return {
        "reply": "".join(text).strip(),
        "first_token_ms": round(1000 * ((first or time.monotonic()) - started)),
        "total_ms": round(1000 * (time.monotonic() - started)),
    }
