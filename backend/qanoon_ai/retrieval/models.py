"""Retrieval domain models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class EvidenceChunk:
    chunk_id: str
    source_file: str
    title: str
    year: str | None
    text: str
    score: float
    source_id: str | None = None
    source_url: str | None = None
    jurisdiction: str | None = None
    document_type: str | None = None
    legal_status: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    section_ref: str | None = None


class EvidenceRetriever(Protocol):
    def search(self, query: str, *, top_k: int = 5, jurisdiction: str | None = None,
               semantic_query: str | None = None) -> list[EvidenceChunk]: ...
