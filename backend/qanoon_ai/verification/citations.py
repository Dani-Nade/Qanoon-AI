"""Citation verification primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
import re

from qanoon_ai.schemas.source import Citation

CITATION_PATTERN = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def citation_numbers(text: str) -> list[int]:
    return [int(number) for match in CITATION_PATTERN.finditer(text)
            for number in match.group(1).split(",")]


def strip_citations(text: str) -> str:
    return re.sub(r"\s*" + CITATION_PATTERN.pattern, "", text)


@dataclass(frozen=True)
class CitationVerification:
    ok: bool
    warnings: list[str] = field(default_factory=list)


def verify_citations(citations: list[Citation]) -> CitationVerification:
    warnings: list[str] = []
    if not citations:
        warnings.append("No citations were retrieved for this answer.")
        return CitationVerification(ok=False, warnings=warnings)

    for citation in citations:
        if not citation.source_file:
            warnings.append("A citation is missing its source file.")
        if not citation.excerpt:
            warnings.append(f"Citation {citation.source_file} has no excerpt.")

    return CitationVerification(ok=not warnings, warnings=warnings)
