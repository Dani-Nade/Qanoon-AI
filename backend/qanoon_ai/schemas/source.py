"""Source and citation schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SourceRecord(BaseModel):
    source_id: str
    title: str
    local_path: str
    year: str | None = None
    jurisdiction: str | None = None
    province: str | None = None
    court: str | None = None
    document_type: str | None = None
    source_url: str | None = None
    language: str = "english"
    amended_upto: str | None = None
    is_repealed: bool | None = None
    official_status: str = "unknown"
    file_sha256: str | None = None
    file_size_bytes: int | None = None


class Citation(BaseModel):
    title: str
    source_file: str
    source_id: str | None = None
    source_url: str | None = None
    chunk_id: str | None = None
    year: str | None = None
    jurisdiction: str | None = None
    document_type: str | None = None
    legal_status: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    section_ref: str | None = None
    score: float | None = None
    excerpt: str = Field(default="")
