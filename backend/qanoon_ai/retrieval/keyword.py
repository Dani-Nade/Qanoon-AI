"""Local keyword retriever used until the hybrid index is built."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

from qanoon_ai.retrieval.models import EvidenceChunk

TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u0600-\u06ff]+")
STOPWORDS = {
    "the",
    "is",
    "are",
    "a",
    "an",
    "of",
    "for",
    "in",
    "on",
    "to",
    "and",
    "or",
    "what",
    "does",
    "under",
    "law",
    "pakistan",
    "pakistani",
}


def tokenize(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


class KeywordRetriever:
    def __init__(self, chunks_file: Path) -> None:
        self.chunks_file = chunks_file
        self._records: list[dict] | None = None
        self._doc_freq: Counter[str] | None = None

    @property
    def ready(self) -> bool:
        return self.chunks_file.exists()

    def _load(self) -> None:
        if self._records is not None:
            return

        records: list[dict] = []
        doc_freq: Counter[str] = Counter()
        if self.chunks_file.exists():
            with self.chunks_file.open(encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)
                    tokens = tokenize(
                        " ".join(
                            [
                                record.get("title", ""),
                                record.get("source_file", ""),
                                record.get("text", ""),
                            ]
                        )
                    )
                    record["_tokens"] = Counter(tokens)
                    records.append(record)
                    doc_freq.update(set(tokens))

        self._records = records
        self._doc_freq = doc_freq

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        jurisdiction: str | None = None,
    ) -> list[EvidenceChunk]:
        if not self.ready:
            return []

        self._load()
        assert self._records is not None
        assert self._doc_freq is not None

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        query_counts = Counter(query_tokens)
        total_docs = max(len(self._records), 1)
        scored: list[EvidenceChunk] = []

        for record in self._records:
            if jurisdiction and record.get("jurisdiction") not in {jurisdiction, None, ""}:
                continue
            token_counts: Counter[str] = record["_tokens"]
            score = 0.0
            for token, query_count in query_counts.items():
                tf = token_counts.get(token, 0)
                if not tf:
                    continue
                idf = math.log(1 + (total_docs / (1 + self._doc_freq[token])))
                score += (1 + math.log(tf)) * idf * query_count

            title = record.get("title", "")
            source_file = record.get("source_file", "")
            lowered_query = query.lower()
            if lowered_query and lowered_query in title.lower():
                score += 8.0
            if lowered_query and lowered_query in source_file.lower():
                score += 5.0

            if score <= 0:
                continue

            scored.append(
                EvidenceChunk(
                    chunk_id=record.get("chunk_id", ""),
                    source_file=source_file,
                    title=title,
                    year=record.get("year"),
                    text=record.get("text", ""),
                    score=round(float(score), 4),
                    source_id=record.get("source_id"),
                    source_url=record.get("source_url"),
                    jurisdiction=record.get("jurisdiction"),
                    document_type=record.get("document_type"),
                    legal_status=record.get("legal_status"),
                    page_start=record.get("page_start"),
                    page_end=record.get("page_end"),
                    section_ref=record.get("section_ref"),
                )
            )

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]
