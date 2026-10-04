"""Official legal source catalog loading and filtering."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OfficialSource:
    """A configured official source for legal documents."""

    source_id: str
    name: str
    jurisdiction: str
    homepage: str
    seed_urls: tuple[str, ...]
    allowed_domains: tuple[str, ...]
    document_scope: tuple[str, ...]
    notes: str | None = None
    seed_method: str = "GET"
    seed_form_data: tuple[tuple[str, str], ...] = ()
    document_url_patterns: tuple[str, ...] = ()
    crawl_url_patterns: tuple[str, ...] = ()
    url_rewrites: tuple[tuple[str, str], ...] = ()
    verify_tls: bool = True
    filename_strategy: str = "source_filename"


@dataclass(frozen=True)
class SourceCatalog:
    """A versioned set of official sources."""

    version: str
    sources: tuple[OfficialSource, ...]

    def select(
        self,
        *,
        jurisdictions: set[str] | None = None,
        source_ids: set[str] | None = None,
    ) -> list[OfficialSource]:
        selected = list(self.sources)
        if jurisdictions:
            selected = [source for source in selected if source.jurisdiction in jurisdictions]
        if source_ids:
            selected = [source for source in selected if source.source_id in source_ids]
        return selected


def load_source_catalog(path: Path) -> SourceCatalog:
    """Load the official source catalog from JSON."""

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    sources = []
    for item in payload.get("sources", []):
        sources.append(
            OfficialSource(
                source_id=item["source_id"],
                name=item["name"],
                jurisdiction=item["jurisdiction"],
                homepage=item["homepage"],
                seed_urls=tuple(item.get("seed_urls", [])),
                allowed_domains=tuple(item.get("allowed_domains", [])),
                document_scope=tuple(item.get("document_scope", [])),
                notes=item.get("notes"),
                seed_method=item.get("seed_method", "GET").upper(),
                seed_form_data=tuple(item.get("seed_form_data", {}).items()),
                document_url_patterns=tuple(item.get("document_url_patterns", [])),
                crawl_url_patterns=tuple(item.get("crawl_url_patterns", [])),
                url_rewrites=tuple(
                    (rewrite["from"], rewrite["to"])
                    for rewrite in item.get("url_rewrites", [])
                ),
                verify_tls=item.get("verify_tls", True),
                filename_strategy=item.get("filename_strategy", "source_filename"),
            )
        )
    return SourceCatalog(version=payload.get("version", "unknown"), sources=tuple(sources))
