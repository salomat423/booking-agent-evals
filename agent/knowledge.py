"""Retrieval over the business knowledge base.

Production used Sentence Transformers + ChromaDB. For a small, auditable demo this uses
TF-IDF with cosine similarity in pure Python: no model download, deterministic results,
and the same interface (`search(query, k)`), so a vector store can be swapped in later.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .config import DATA_DIR

_TOKEN = re.compile(r"[a-zа-яё0-9]+", re.IGNORECASE)
_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are", "be", "by",
    "with", "we", "you", "your", "our", "it", "at", "from", "as", "do", "does", "can", "i",
    "my", "me", "how", "what", "much", "any", "there", "this", "that", "not", "no",
}


def tokenize(text: str) -> list[str]:
    return [t for t in (m.lower() for m in _TOKEN.findall(text)) if t not in _STOP]


@dataclass
class Chunk:
    title: str
    text: str


@dataclass
class Hit:
    title: str
    text: str
    score: float


class KnowledgeBase:
    def __init__(self, chunks: list[Chunk]):
        if not chunks:
            raise ValueError("knowledge base is empty")
        self.chunks = chunks
        docs = [Counter(tokenize(c.title + " " + c.text)) for c in chunks]
        df = Counter(term for d in docs for term in d)
        n = len(docs)
        self.idf = {t: math.log((1 + n) / (1 + f)) + 1 for t, f in df.items()}
        self.vectors = [self._weigh(d) for d in docs]

    @classmethod
    def from_markdown(cls, path: Path | None = None) -> "KnowledgeBase":
        path = path or DATA_DIR / "knowledge_base.md"
        chunks: list[Chunk] = []
        title, lines = None, []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("## "):
                if title:
                    chunks.append(Chunk(title, "\n".join(lines).strip()))
                title, lines = line[3:].strip(), []
            elif title:
                lines.append(line)
        if title:
            chunks.append(Chunk(title, "\n".join(lines).strip()))
        return cls(chunks)

    def _weigh(self, counts: Counter) -> dict[str, float]:
        vec = {t: c * self.idf.get(t, 0.0) for t, c in counts.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def search(self, query: str, k: int = 2) -> list[Hit]:
        q = self._weigh(Counter(tokenize(query)))
        scored = []
        for chunk, vec in zip(self.chunks, self.vectors):
            score = sum(w * vec.get(t, 0.0) for t, w in q.items())
            scored.append(Hit(chunk.title, chunk.text, round(score, 4)))
        scored.sort(key=lambda h: h.score, reverse=True)
        return [h for h in scored[:k] if h.score > 0]
