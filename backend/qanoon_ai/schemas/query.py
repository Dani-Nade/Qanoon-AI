"""Query request and response schemas."""

from __future__ import annotations
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from qanoon_ai.schemas.source import Citation


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    language: Literal["auto", "english", "urdu", "roman_urdu"] = "auto"
    jurisdiction: str | None = Field(default=None)
    max_sources: int = Field(default=5, ge=1, le=20)
    require_citations: bool = True


class LanguageInfo(BaseModel):
    language: str
    script: str
    confidence: float


class QueryResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    question: str
    answer: str
    language: LanguageInfo
    citations: list[Citation] = Field(default_factory=list)
    retrieval_ready: bool
    model_ready: bool
    answer_mode: str = "insufficient_evidence"
    model_name: str | None = None
    warnings: list[str] = Field(default_factory=list)
    disclaimer: str = (
        "Qanoon AI provides general legal information only. "
        "Consult a licensed lawyer for professional legal advice."
    )
