"""PDF inventory and source registry generation."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class PdfInventoryRecord:
    source_id: str
    title: str
    local_path: str
    year: str | None
    document_type: str | None
    jurisdiction: str | None
    language: str
    official_status: str
    file_size_bytes: int
    file_sha256: str | None


def _slug(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:120] or "source"


def _guess_document_type(title: str) -> str | None:
    lowered = title.lower()
    for doc_type in ["act", "ordinance", "rules", "regulations", "order", "code"]:
        if doc_type in lowered:
            return doc_type
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_pdfs(pdfs_dir: Path, *, hash_files: bool = False) -> list[PdfInventoryRecord]:
    records: list[PdfInventoryRecord] = []
    base_dir = pdfs_dir.parent
    for pdf_path in sorted(pdfs_dir.rglob("*.pdf")):
        title = pdf_path.stem
        year = pdf_path.parent.name if pdf_path.parent.name.isdigit() else None
        source_id = f"local_{year or 'unknown'}_{_slug(title)}"
        try:
            local_path = str(pdf_path.relative_to(base_dir))
        except ValueError:
            local_path = str(pdf_path)
        records.append(
            PdfInventoryRecord(
                source_id=source_id,
                title=title,
                local_path=local_path,
                year=year,
                document_type=_guess_document_type(title),
                jurisdiction="unknown",
                language="english",
                official_status="unknown",
                file_size_bytes=pdf_path.stat().st_size,
                file_sha256=_sha256(pdf_path) if hash_files else None,
            )
        )
    return records


def write_jsonl(records: list[PdfInventoryRecord], out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
