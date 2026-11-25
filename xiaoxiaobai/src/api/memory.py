# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

"""Long-term memories about the user and the conversation history (text chat and voice)."""

from __future__ import annotations

import math
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from api.knowledge import tokenize
from api.protocol import ApiError, CommonState, Reason
from api.store import DATA_DIR, read_json, write_json_atomic

MAX_MEMORIES = 500
MAX_MESSAGES = 400
# Old conversations searched when asked about the past, newest first.
SEARCH_CONVERSATIONS = 80
SEARCH_MESSAGES = 8000
EXCERPT_CHARS = 120

# Asking about the past ("还记得…吗", "上次说的…"): memories and old conversations are searched for it.
RECALL_HINT = re.compile(
    r"记得|记不记得|记住了|忘了|忘记|上次|上回|之前|以前|那天|那次|前几天|前阵子|说过|聊过|讲过|提过|告诉过|问过|跟你说|"
    r"我叫什么|我的名字|我是谁|我喜欢什么|我的生日|你知道我"
)
# Left out of the search, so that what is asked about picks the matches.
RECALL_FILLER = re.compile("|".join(sorted((
    "还记得", "记不记得", "记得", "记住了", "忘了", "忘记", "上次", "上回", "之前", "以前", "那天", "那次", "前几天", "前阵子",
    "说过", "聊过", "讲过", "提过", "告诉过", "问过", "跟你说", "跟你", "和你", "你知道", "知道", "请问", "告诉我", "一下",
    "什么", "哪个", "哪些", "那个", "这个", "怎么", "是不是", "有没有", "时候", "东西", "我们", "你们", "来着",
    "你", "我", "的", "了", "吗", "呢", "吧", "啊", "呀", "哦", "嗯", "么", "是", "过", "说",
), key=len, reverse=True)))


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def recall_terms(query: str) -> list[str]:
    """The words of `query` a recall search looks for: no filler, each once."""
    return list(dict.fromkeys(tokenize(RECALL_FILLER.sub(" ", query))))


def _day(ts: float) -> str:
    t = time.localtime(ts)
    return f"{t.tm_mon}月{t.tm_mday}日"


class MemoryStore:
    def __init__(self, path: Path = DATA_DIR / "memory.json") -> None:
        self._path = path
        self._lock = threading.Lock()
        self._items: list[dict[str, Any]] = read_json(path, [])

    def _save(self) -> None:
        write_json_atomic(self._path, self._items)

    def list(self, query: str = "") -> list[dict[str, Any]]:
        with self._lock:
            items = list(self._items)
        if query:
            needle = query.lower()
            items = [m for m in items if needle in m["text"].lower() or needle in " ".join(m.get("tags", [])).lower()]
        return sorted(items, key=lambda m: (not m.get("pinned"), -m["updated"]))

    def add(self, text: str, source: str = "manual", tags: list[str] | None = None, pinned: bool = False) -> dict[str, Any]:
        text = (text or "").strip()
        if not text:
            raise ApiError("记忆内容不能为空", CommonState.INVALID)
        with self._lock:
            for item in self._items:
                if item["text"] == text:
                    item["updated"] = time.time()
                    self._save()
                    return dict(item)
            now = time.time()
            item = {
                "id": new_id("mem"),
                "text": text[:500],
                "tags": [str(t)[:20] for t in (tags or [])][:8],
                "source": source,
                "pinned": bool(pinned),
                "created": now,
                "updated": now,
            }
            self._items.append(item)
            if len(self._items) > MAX_MEMORIES:
                self._items.sort(key=lambda m: (m.get("pinned", False), m["updated"]))
                del self._items[0]
            self._save()
            return dict(item)

    def update(self, memory_id: str, values: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            for item in self._items:
                if item["id"] == memory_id:
                    if "text" in values:
                        text = str(values["text"]).strip()
                        if not text:
                            raise ApiError("记忆内容不能为空", CommonState.INVALID)
                        item["text"] = text[:500]
                    if "tags" in values:
                        item["tags"] = [str(t)[:20] for t in values["tags"] or []][:8]
                    if "pinned" in values:
                        item["pinned"] = bool(values["pinned"])
                    item["updated"] = time.time()
                    self._save()
                    return dict(item)
        raise ApiError("没有这条记忆", CommonState.INVALID, Reason.NOT_FOUND, 404)

    def delete(self, memory_id: str) -> None:
        with self._lock:
            before = len(self._items)
            self._items = [m for m in self._items if m["id"] != memory_id]
            if len(self._items) == before:
                raise ApiError("没有这条记忆", CommonState.INVALID, Reason.NOT_FOUND, 404)
            self._save()

    def clear(self) -> int:
        with self._lock:
            count = len(self._items)
            self._items = []
            self._save()
            return count

    def for_prompt(self, query: str = "", limit_chars: int = 1200) -> list[dict[str, Any]]:
        """Pinned memories first, then the ones sharing the most words with the query, then the newest."""
        words = set(tokenize(query))
        with self._lock:
            items = list(self._items)

        def rank(m: dict[str, Any]) -> tuple:
            overlap = len(words & set(tokenize(m["text"]))) if words else 0
            return (not m.get("pinned"), -overlap, -m["updated"])

        picked, used = [], 0
        for item in sorted(items, key=rank):
            if used + len(item["text"]) > limit_chars:
                continue
            picked.append(item)
            used += len(item["text"]) + 4
        return picked

    def search(self, query: str, limit: int = 6) -> list[dict[str, Any]]:
        """The memories holding the most words of `query`."""
        terms = recall_terms(query)
        if not terms:
            return []
        with self._lock:
            items = list(self._items)
        scored = [(sum(t in m["text"].lower() for t in terms), m) for m in items]
        scored = [(s, m) for s, m in scored if s]
        return [m for _, m in sorted(scored, key=lambda sm: (-sm[0], -sm[1]["updated"]))[:limit]]


class ConversationStore:
    def __init__(self, root: Path = DATA_DIR / "conversations") -> None:
        self._root = root
        self._lock = threading.Lock()
        self._index_path = root / "index.json"
        self._index: dict[str, dict[str, Any]] = read_json(self._index_path, {})

    def _file(self, conv_id: str) -> Path:
        if not conv_id.replace("_", "").isalnum():
            raise ApiError("会话 ID 无效", CommonState.INVALID)
        return self._root / f"{conv_id}.json"

    def _save_index(self) -> None:
        write_json_atomic(self._index_path, self._index)

    def list(self, channel: str = "") -> list[dict[str, Any]]:
        with self._lock:
            items = [dict(v) for v in self._index.values() if not channel or v["channel"] == channel]
        return sorted(items, key=lambda c: -c["updated"])

    def create(self, channel: str = "chat", title: str = "") -> dict[str, Any]:
        now = time.time()
        conv = {
            "id": new_id("conv"),
            "title": title or ("新对话" if channel == "chat" else time.strftime("语音对话 %m-%d %H:%M")),
            "channel": channel,
            "created": now,
            "updated": now,
            "count": 0,
        }
        with self._lock:
            self._index[conv["id"]] = conv
            write_json_atomic(self._file(conv["id"]), [])
            self._save_index()
        return dict(conv)

    def get(self, conv_id: str) -> dict[str, Any]:
        with self._lock:
            meta = self._index.get(conv_id)
            if meta is None:
                raise ApiError("没有这个会话", CommonState.INVALID, Reason.NOT_FOUND, 404)
            return {**meta, "messages": read_json(self._file(conv_id), [])}

    def exists(self, conv_id: str) -> bool:
        with self._lock:
            return conv_id in self._index

    def append(self, conv_id: str, role: str, content: str, **meta: Any) -> dict[str, Any]:
        message = {"role": role, "content": content, "ts": time.time(), **({"meta": meta} if meta else {})}
        with self._lock:
            info = self._index.get(conv_id)
            if info is None:
                raise ApiError("没有这个会话", CommonState.INVALID, Reason.NOT_FOUND, 404)
            path = self._file(conv_id)
            messages = read_json(path, [])
            messages.append(message)
            write_json_atomic(path, messages[-MAX_MESSAGES:])
            info["updated"] = message["ts"]
            info["count"] = len(messages)
            if role == "user" and info["title"] == "新对话":
                info["title"] = content.strip().replace("\n", " ")[:24] or "新对话"
            self._save_index()
        return message

    def history(self, conv_id: str, turns: int) -> list[dict[str, str]]:
        messages = self.get(conv_id)["messages"]
        out = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] in ("user", "assistant") and m["content"]]
        return out[-2 * turns :] if turns > 0 else []

    def window_start(self, conv_id: str, turns: int) -> float:
        """When the part of a conversation history() hands the model begins (now if none)."""
        messages = [m for m in self.get(conv_id)["messages"] if m["role"] in ("user", "assistant") and m["content"]]
        window = messages[-2 * turns :] if turns > 0 else []
        return window[0]["ts"] if window else time.time()

    def search(self, query: str, limit: int = 4, before: float | None = None) -> list[dict[str, Any]]:
        """Past exchanges (a question and its answer) about `query`, best first; only messages
        from before `before`, so that what the model already has is left out."""
        terms = recall_terms(query)
        if not terms:
            return []
        with self._lock:
            convs = sorted(self._index.values(), key=lambda c: -c["updated"])[:SEARCH_CONVERSATIONS]
            threads = []
            for conv in convs:
                messages = [m for m in read_json(self._file(conv["id"]), [])
                            if m["role"] in ("user", "assistant") and m["content"] and (before is None or m["ts"] < before)]
                if messages:
                    threads.append((dict(conv), messages))
        corpus = [(t, i) for t, (_, messages) in enumerate(threads) for i in range(len(messages))][-SEARCH_MESSAGES:]
        if not corpus:
            return []
        texts = {(t, i): threads[t][1][i]["content"].lower() for t, i in corpus}
        df = {term: sum(term in text for text in texts.values()) for term in terms}
        idf = {term: math.log(1 + len(corpus) / n) for term, n in df.items() if n}
        need = min(2, len(terms))
        scored = []
        for key, text in texts.items():
            found = [term for term in idf if term in text]
            if len(found) >= need:
                scored.append((sum(idf[term] for term in found), threads[key[0]][1][key[1]]["ts"], key))
        out, seen = [], set()
        for _, _, (t, i) in sorted(scored, reverse=True):
            conv, messages = threads[t]
            start = i - 1 if messages[i]["role"] == "assistant" and i > 0 and messages[i - 1]["role"] == "user" else i
            end = start + 2 if start + 1 < len(messages) and messages[start]["role"] == "user" and messages[start + 1]["role"] == "assistant" else start + 1
            if (t, start) in seen:
                continue
            seen.add((t, start))
            out.append({
                "conversation": conv["title"],
                "channel": conv["channel"],
                "ts": messages[start]["ts"],
                "messages": [{"role": m["role"], "content": m["content"][:EXCERPT_CHARS]} for m in messages[start:end]],
            })
            if len(out) >= limit:
                break
        return out


RECALL_NOTE = "\n[记忆检索]（用户在问以前的事，这是从长期记忆和以前的聊天记录里搜到的：据此回答，记录里没有的不要编）\n"
RECALL_NONE = ("\n[记忆检索] 已经翻过以前的聊天记录，没搜到直接相关的：先看已有的记忆和对话里有没有答案，"
               "都没有就如实说没找到、请用户提醒一下，不要编，也不要说自己没有记忆或者记不住。\n")


def recall_text(memory: MemoryStore, conversations: ConversationStore, query: str, before: float | None = None,
                limit_chars: int = 900) -> str:
    """What the memories and old conversations hold about `query`, for a prompt."""
    lines = [f"- {m['text']}" for m in memory.search(query)]
    for hit in conversations.search(query, before=before):
        said = "；".join(f"{'用户' if m['role'] == 'user' else '你'}：{m['content']}" for m in hit["messages"])
        lines.append(f"- {_day(hit['ts'])}{'语音' if hit['channel'] == 'voice' else '聊天'}里 {said}")
    if not lines:
        return RECALL_NONE
    return RECALL_NOTE + "\n".join(lines)[:limit_chars]

    def rename(self, conv_id: str, title: str) -> dict[str, Any]:
        with self._lock:
            info = self._index.get(conv_id)
            if info is None:
                raise ApiError("没有这个会话", CommonState.INVALID, Reason.NOT_FOUND, 404)
            info["title"] = (title or "").strip()[:60] or info["title"]
            self._save_index()
            return dict(info)

    def delete(self, conv_id: str) -> None:
        with self._lock:
            if self._index.pop(conv_id, None) is None:
                raise ApiError("没有这个会话", CommonState.INVALID, Reason.NOT_FOUND, 404)
            self._file(conv_id).unlink(missing_ok=True)
            self._save_index()
