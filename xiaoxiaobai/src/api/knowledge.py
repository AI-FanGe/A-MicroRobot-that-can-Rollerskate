# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 AI-FanGe

"""Local knowledge base: plain-text documents split into chunks and ranked with BM25.

Chinese has no spaces, so text is indexed as character bigrams (plus ASCII words). That is
enough for a few hundred documents on a Raspberry Pi with no embedding model.
"""

from __future__ import annotations

import math
import re
import threading
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from api.protocol import ApiError, CommonState, Reason
from api.store import DATA_DIR, read_json, write_json_atomic

CHUNK_CHARS = 420
CHUNK_OVERLAP = 60
MAX_DOC_CHARS = 400_000
K1, B = 1.4, 0.75

_WORD = re.compile(r"[a-z0-9_]+|[\u3400-\u9fff\uf900-\ufaff]+")


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for match in _WORD.finditer(text.lower()):
        part = match.group()
        if part[0].isascii():
            tokens.append(part)
        elif len(part) == 1:
            tokens.append(part)
        else:
            tokens.extend(part[i : i + 2] for i in range(len(part) - 1))
    return tokens


def split_chunks(text: str) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text.replace("\r\n", "\n")) if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        while len(para) > CHUNK_CHARS:
            sentence_end = max(para.rfind(sep, 0, CHUNK_CHARS) for sep in "。！？.!?\n；;")
            hard_cut = sentence_end <= CHUNK_CHARS // 2
            cut = CHUNK_CHARS if hard_cut else sentence_end + 1
            if current:
                chunks.append(current)
                current = ""
            chunks.append(para[:cut].strip())
            para = para[cut - CHUNK_OVERLAP if hard_cut else cut :].strip()
        if current and len(current) + len(para) + 1 > CHUNK_CHARS:
            chunks.append(current)
            current = ""
        current = f"{current}\n{para}" if current else para
    if current:
        chunks.append(current)
    return chunks


class KnowledgeBase:
    def __init__(self, root: Path = DATA_DIR / "knowledge") -> None:
        self._root = root
        self._lock = threading.Lock()
        self._index_path = root / "index.json"
        self._docs: dict[str, dict[str, Any]] = read_json(self._index_path, {})
        self._chunks: list[tuple[str, int, str, Counter, int]] = []
        self._df: Counter = Counter()
        self._avg_len = 1.0
        self._rebuild()

    def _doc_path(self, doc_id: str) -> Path:
        if not doc_id.replace("_", "").isalnum():
            raise ApiError("文档 ID 无效", CommonState.INVALID)
        return self._root / f"{doc_id}.txt"

    def _rebuild(self) -> None:
        chunks: list[tuple[str, int, str, Counter, int]] = []
        df: Counter = Counter()
        for doc_id, doc in self._docs.items():
            if not doc.get("enabled", True):
                continue
            try:
                text = self._doc_path(doc_id).read_text(encoding="utf-8")
            except OSError:
                continue
            for index, chunk in enumerate(split_chunks(f"{doc['title']}\n\n{text}")):
                counts = Counter(tokenize(chunk))
                df.update(counts.keys())
                chunks.append((doc_id, index, chunk, counts, sum(counts.values())))
        self._chunks, self._df = chunks, df
        self._avg_len = (sum(c[4] for c in chunks) / len(chunks)) if chunks else 1.0

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return sorted((dict(d) for d in self._docs.values()), key=lambda d: -d["created"])

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {"documents": len(self._docs), "chunks": len(self._chunks), "chars": sum(d["chars"] for d in self._docs.values())}

    def get(self, doc_id: str) -> dict[str, Any]:
        with self._lock:
            doc = self._docs.get(doc_id)
            if doc is None:
                raise ApiError("没有这个文档", CommonState.INVALID, Reason.NOT_FOUND, 404)
            text = self._doc_path(doc_id).read_text(encoding="utf-8")
        return {**doc, "text": text}

    def add(self, title: str, text: str, source: str = "manual") -> dict[str, Any]:
        text = (text or "").strip()
        if not text:
            raise ApiError("文档内容为空", CommonState.INVALID)
        if len(text) > MAX_DOC_CHARS:
            raise ApiError(f"文档太长（{len(text)} 字），上限 {MAX_DOC_CHARS} 字", CommonState.INVALID)
        doc_id = f"doc_{uuid.uuid4().hex[:10]}"
        doc = {
            "id": doc_id,
            "title": (title or "").strip()[:80] or text.splitlines()[0][:40],
            "source": source,
            "chars": len(text),
            "chunks": len(split_chunks(text)),
            "enabled": True,
            "created": time.time(),
        }
        with self._lock:
            self._root.mkdir(parents=True, exist_ok=True)
            self._doc_path(doc_id).write_text(text, encoding="utf-8")
            self._docs[doc_id] = doc
            write_json_atomic(self._index_path, self._docs)
            self._rebuild()
        return dict(doc)

    def update(self, doc_id: str, values: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            doc = self._docs.get(doc_id)
            if doc is None:
                raise ApiError("没有这个文档", CommonState.INVALID, Reason.NOT_FOUND, 404)
            if "title" in values:
                doc["title"] = str(values["title"]).strip()[:80] or doc["title"]
            if "enabled" in values:
                doc["enabled"] = bool(values["enabled"])
            if "text" in values:
                text = str(values["text"]).strip()
                if not text:
                    raise ApiError("文档内容为空", CommonState.INVALID)
                self._doc_path(doc_id).write_text(text, encoding="utf-8")
                doc["chars"], doc["chunks"] = len(text), len(split_chunks(text))
            write_json_atomic(self._index_path, self._docs)
            self._rebuild()
            return dict(doc)

    def delete(self, doc_id: str) -> None:
        with self._lock:
            if self._docs.pop(doc_id, None) is None:
                raise ApiError("没有这个文档", CommonState.INVALID, Reason.NOT_FOUND, 404)
            self._doc_path(doc_id).unlink(missing_ok=True)
            write_json_atomic(self._index_path, self._docs)
            self._rebuild()

    def search(self, query: str, top_k: int = 4) -> list[dict[str, Any]]:
        terms = set(tokenize(query))
        if not terms:
            return []
        with self._lock:
            chunks, df, avg, docs = self._chunks, self._df, self._avg_len, self._docs
            n = len(chunks)
            scored = []
            for doc_id, index, text, counts, length in chunks:
                score = 0.0
                for term in terms:
                    tf = counts.get(term)
                    if not tf:
                        continue
                    idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
                    score += idf * tf * (K1 + 1) / (tf + K1 * (1 - B + B * length / avg))
                if score > 0:
                    scored.append((score, doc_id, index, text))
            scored.sort(key=lambda s: -s[0])
            return [
                {"doc_id": d, "title": docs[d]["title"], "chunk": i, "text": t, "score": round(s, 3)}
                for s, d, i, t in scored[: max(1, min(int(top_k), 20))]
            ]

    @staticmethod
    def version(root: Path = DATA_DIR / "knowledge") -> float:
        """Changes whenever documents are added, edited, enabled or removed (by any process)."""
        try:
            return (root / "index.json").stat().st_mtime
        except OSError:
            return 0.0
