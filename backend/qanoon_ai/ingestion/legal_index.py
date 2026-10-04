"""Incremental PDF extraction and legal-structure-aware indexing."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import re
import sqlite3
import unicodedata
from collections.abc import Callable, Iterable
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pypdf import PdfReader

LOGGER = logging.getLogger(__name__)
SCHEMA_VERSION = "1"
PIPELINE_VERSION = "2026.09.14.2"
DEFAULT_TARGET_WORDS = 550
DEFAULT_MAX_WORDS = 850
MIN_TEXT_CHARS = 80

PAGE_NUMBER_RE = re.compile(r"^\s*(?:page\s+)?\d+(?:\s+of\s+\d+)?\s*$", re.IGNORECASE)
LEGAL_HEADING_RE = re.compile(
    r"^\s*(?:(section|article|rule|order|chapter|part|schedule|appendix)\s+"
    r"(\d+[A-Z]?(?:[-.]\d+[A-Z]?)*|[IVXLCDM]+|[A-Z])\b|([0-9]{1,3}(?:-?[A-Z])?)\.)\s*(.*)$",
    re.IGNORECASE,
)
OPAQUE_TITLE_RE = re.compile(r"^(?:administrator)?[0-9a-f]{20,}$", re.IGNORECASE)
DOCUMENT_TITLE_RE = re.compile(
    r"\b(?:ACT|CODE|CONSTITUTION|ORDINANCE|ORDER|REGULATIONS?|RULES?)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PageText:
    page_number: int
    text: str


@dataclass(frozen=True)
class LegalChunk:
    chunk_index: int
    page_start: int
    page_end: int
    section_ref: str | None
    heading: str | None
    text: str
    word_count: int


@dataclass(frozen=True)
class IndexBuildSummary:
    selected_documents: int
    indexed_documents: int
    skipped_unchanged: int
    needs_ocr: int
    failed_documents: int
    removed_stale_documents: int
    indexed_chunks: int
    index_file: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_extracted_text(text: str) -> str:
    """Normalize extraction artifacts without removing legal structure."""

    text = unicodedata.normalize("NFC", text or "").replace("\x00", "")
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    lines = []
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = re.sub(r"[\t ]+", " ", raw_line).strip()
        if PAGE_NUMBER_RE.fullmatch(line):
            continue
        lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_pages(pdf_path: Path) -> list[PageText]:
    """Extract text page by page so every chunk can retain a page citation."""

    reader = PdfReader(pdf_path, strict=False)
    if reader.is_encrypted:
        reader.decrypt("")
    pages: list[PageText] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # a damaged page should not discard the document
            LOGGER.warning("Page extraction failed path=%s page=%s error=%s", pdf_path, page_number, exc)
            text = ""
        pages.append(PageText(page_number=page_number, text=clean_extracted_text(text)))
    return pages


def extract_pdf_pages_with_ocr(
    pdf_path: Path,
    *,
    enable_ocr: bool = True,
    ocr_languages: str = "eng+urd",
) -> list[PageText]:
    """Extract text first and OCR only pages that have no useful text layer."""

    pages = extract_pdf_pages(pdf_path)
    missing = [page.page_number for page in pages if len(page.text.strip()) < 30]
    if not enable_ocr or not missing:
        return pages
    try:
        import fitz
        import pytesseract
        from PIL import Image
    except ImportError:
        LOGGER.warning("OCR dependencies are unavailable for %s", pdf_path)
        return pages

    document = fitz.open(pdf_path)
    replacements = {page.page_number: page for page in pages}
    try:
        for page_number in missing:
            pdf_page = document.load_page(page_number - 1)
            pixmap = pdf_page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
            image = Image.open(io.BytesIO(pixmap.tobytes("png")))
            text = pytesseract.image_to_string(image, lang=ocr_languages)
            replacements[page_number] = PageText(
                page_number=page_number,
                text=clean_extracted_text(text),
            )
    finally:
        document.close()
    return [replacements[number] for number in sorted(replacements)]


def _display_title(record: dict[str, Any], pages: list[PageText]) -> str:
    manifest_title = str(record["title"]).strip()
    if not OPAQUE_TITLE_RE.fullmatch(manifest_title) or not pages:
        return manifest_title
    for line in pages[0].text.splitlines()[:40]:
        candidate = re.sub(r"\s+", " ", line).strip(" _-\t")
        if 8 <= len(candidate) <= 220 and DOCUMENT_TITLE_RE.search(candidate):
            return candidate
    return manifest_title


def _heading(line: str) -> tuple[str | None, str | None]:
    line = re.sub(r"^(?:\s*\d+\[)+\s*(?=\d{1,3}(?:-?[A-Z])?\.)", "", line)
    match = LEGAL_HEADING_RE.match(line)
    if not match:
        return None, None
    kind, identifier, numeric_identifier, remainder = match.groups()
    if kind:
        section_ref = f"{kind.title()} {identifier}"
    else:
        section_ref = f"Section {numeric_identifier}"
    heading = " ".join(part for part in (section_ref, remainder.strip()) if part)
    return section_ref, heading[:300]


def _page_units(page: PageText) -> list[tuple[int, str | None, str | None, str]]:
    units: list[tuple[int, str | None, str | None, str]] = []
    current: list[str] = []
    current_ref: str | None = None
    current_heading: str | None = None

    def flush() -> None:
        nonlocal current
        text = "\n".join(current).strip()
        if text:
            units.append((page.page_number, current_ref, current_heading, text))
        current = []

    for line in page.text.splitlines():
        line = line.strip()
        if not line:
            if current and current[-1] != "":
                current.append("")
            continue
        section_ref, heading = _heading(line)
        if section_ref:
            flush()
            current_ref = section_ref
            current_heading = heading
        current.append(line)
    flush()
    return units


def chunk_legal_document(
    pages: Iterable[PageText],
    *,
    target_words: int = DEFAULT_TARGET_WORDS,
    max_words: int = DEFAULT_MAX_WORDS,
) -> list[LegalChunk]:
    """Build page-aware chunks, preferring legal section boundaries."""

    if target_words < 50 or max_words < target_words:
        raise ValueError("Chunk limits must satisfy 50 <= target_words <= max_words")

    units = []
    previous_ref = previous_heading = None
    for page in pages:
        for number, ref, heading, text in _page_units(page):
            if ref:
                previous_ref, previous_heading = ref, heading
            units.append((number, ref or previous_ref, heading or previous_heading, text))
    chunks: list[LegalChunk] = []
    current: list[tuple[int, str | None, str | None, str]] = []
    current_words = 0

    def emit(items: list[tuple[int, str | None, str | None, str]]) -> None:
        if not items:
            return
        text = "\n\n".join(item[3] for item in items).strip()
        if not text:
            return
        section_ref = next((item[1] for item in items if item[1]), None)
        heading = next((item[2] for item in items if item[2]), None)
        chunks.append(
            LegalChunk(
                chunk_index=len(chunks),
                page_start=min(item[0] for item in items),
                page_end=max(item[0] for item in items),
                section_ref=section_ref,
                heading=heading,
                text=text,
                word_count=len(text.split()),
            )
        )

    for unit in units:
        unit_words = unit[3].split()
        if len(unit_words) > max_words:
            emit(current)
            current = []
            current_words = 0
            start = 0
            while start < len(unit_words):
                end = min(start + max_words, len(unit_words))
                piece = (unit[0], unit[1], unit[2], " ".join(unit_words[start:end]))
                emit([piece])
                if end == len(unit_words):
                    break
                start = max(end - 80, start + 1)
            continue

        current_ref = next((item[1] for item in current if item[1]), None)
        starts_new_section = unit[1] is not None and unit[1] != current_ref
        if current and (
            starts_new_section
            or current_words + len(unit_words) > target_words
        ):
            emit(current)
            current = []
            current_words = 0
        current.append(unit)
        current_words += len(unit_words)
    emit(current)
    return chunks


def _connect(index_file: Path) -> sqlite3.Connection:
    index_file.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(index_file)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA busy_timeout = 5000")
    return connection


def initialize_index(index_file: Path) -> None:
    with closing(_connect(index_file)) as connection, connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS index_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                file_sha256 TEXT NOT NULL UNIQUE,
                title TEXT NOT NULL,
                manifest_title TEXT,
                local_path TEXT NOT NULL UNIQUE,
                source_id TEXT NOT NULL,
                source_url TEXT,
                authority_type TEXT,
                jurisdiction TEXT,
                year INTEGER,
                document_type TEXT,
                legal_status TEXT,
                language TEXT,
                page_count INTEGER NOT NULL,
                text_chars INTEGER NOT NULL,
                extraction_status TEXT NOT NULL,
                extraction_error TEXT,
                pipeline_version TEXT NOT NULL DEFAULT 'legacy',
                indexed_at_utc TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                page_start INTEGER NOT NULL,
                page_end INTEGER NOT NULL,
                section_ref TEXT,
                heading TEXT,
                text TEXT NOT NULL,
                word_count INTEGER NOT NULL,
                UNIQUE(document_id, chunk_index)
            );
            CREATE INDEX IF NOT EXISTS idx_documents_jurisdiction ON documents(jurisdiction);
            CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_id UNINDEXED,
                title,
                jurisdiction,
                document_type,
                section_ref,
                text,
                tokenize = 'unicode61 remove_diacritics 2'
            );
            """
        )
        connection.execute(
            "INSERT OR REPLACE INTO index_metadata(key, value) VALUES('schema_version', ?)",
            (SCHEMA_VERSION,),
        )
        columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(documents)")
        }
        if "pipeline_version" not in columns:
            connection.execute(
                "ALTER TABLE documents ADD COLUMN pipeline_version TEXT NOT NULL DEFAULT 'legacy'"
            )
        if "manifest_title" not in columns:
            connection.execute("ALTER TABLE documents ADD COLUMN manifest_title TEXT")


def read_canonical_manifest(manifest_file: Path) -> list[dict[str, Any]]:
    if not manifest_file.exists():
        raise FileNotFoundError(f"Canonical manifest not found: {manifest_file}")
    records: list[dict[str, Any]] = []
    with manifest_file.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            required = {"document_id", "local_path", "file_sha256", "title", "source_id"}
            missing = required - record.keys()
            if missing:
                raise ValueError(f"Manifest line {line_number} is missing {sorted(missing)}")
            records.append(record)
    return records


def _delete_document(connection: sqlite3.Connection, document_id: str) -> None:
    # chunk_id is UNINDEXED in FTS5: deleting one id at a time scans the entire
    # corpus once per chunk. Delete a document's entries in a single scan.
    connection.execute(
        "DELETE FROM chunks_fts WHERE chunk_id IN (SELECT chunk_id FROM chunks WHERE document_id = ?)",
        (document_id,),
    )
    connection.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))


def _store_document(
    connection: sqlite3.Connection,
    record: dict[str, Any],
    pages: list[PageText],
    chunks: list[LegalChunk],
    *,
    extraction_status: str,
    extraction_error: str | None = None,
) -> None:
    display_title = _display_title(record, pages)
    conflicting_ids = {
        row["document_id"]
        for row in connection.execute(
            "SELECT document_id FROM documents WHERE document_id = ? OR local_path = ? OR file_sha256 = ?",
            (record["document_id"], record["local_path"], record["file_sha256"]),
        )
    }
    for document_id in conflicting_ids:
        _delete_document(connection, document_id)
    connection.execute(
        """
        INSERT INTO documents(
            document_id, file_sha256, title, manifest_title, local_path, source_id, source_url,
            authority_type, jurisdiction, year, document_type, legal_status,
            language, page_count, text_chars, extraction_status, extraction_error,
            pipeline_version, indexed_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record["document_id"], record["file_sha256"], display_title, record["title"],
            record["local_path"], record["source_id"], record.get("source_url"),
            record.get("authority_type"), record.get("jurisdiction"), record.get("year"),
            record.get("document_type"), record.get("legal_status"), record.get("language"),
            len(pages), sum(len(page.text) for page in pages), extraction_status,
            extraction_error, PIPELINE_VERSION, utc_now(),
        ),
    )
    for chunk in chunks:
        identity = f"{record['file_sha256']}:{chunk.chunk_index}:{chunk.text}".encode("utf-8")
        chunk_id = f"chunk:{hashlib.sha256(identity).hexdigest()[:24]}"
        connection.execute(
            """
            INSERT INTO chunks(
                chunk_id, document_id, chunk_index, page_start, page_end,
                section_ref, heading, text, word_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chunk_id, record["document_id"], chunk.chunk_index, chunk.page_start,
                chunk.page_end, chunk.section_ref, chunk.heading, chunk.text, chunk.word_count,
            ),
        )
        connection.execute(
            "INSERT INTO chunks_fts(chunk_id, title, jurisdiction, document_type, section_ref, text) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                chunk_id, display_title, record.get("jurisdiction") or "",
                record.get("document_type") or "", chunk.section_ref or "", chunk.text,
            ),
        )


def _extract_and_chunk(pdf_path: str, extractor: Callable[[Path], list[PageText]]) -> tuple:
    """Worker-safe extraction: (pages, chunks, extraction_status, error message or None)."""
    try:
        pages = extractor(Path(pdf_path))
        if sum(len(page.text.strip()) for page in pages) < MIN_TEXT_CHARS:
            return pages, [], "needs_ocr", None
        chunks = chunk_legal_document(pages)
        return pages, chunks, "indexed" if chunks else "needs_ocr", None
    except Exception as exc:
        return [], [], "failed", str(exc)


def build_legal_index(
    *,
    datasets_dir: Path,
    manifest_file: Path,
    index_file: Path,
    source_ids: set[str] | None = None,
    title_contains: str | None = None,
    limit: int | None = None,
    force: bool = False,
    prune: bool = False,
    extractor: Callable[[Path], list[PageText]] = extract_pdf_pages,
    workers: int = 1,
) -> IndexBuildSummary:
    """Incrementally index selected canonical records into SQLite FTS5.

    With workers > 1, PDF extraction and chunking run in parallel processes (the CPU-bound
    part) while this process does all SQLite writes, since SQLite allows one writer.
    """

    initialize_index(index_file)
    all_records = read_canonical_manifest(manifest_file)
    records = all_records
    if source_ids:
        records = [record for record in records if record["source_id"] in source_ids]
    if title_contains:
        needle = title_contains.casefold()
        records = [record for record in records if needle in record["title"].casefold()]
    records = sorted(records, key=lambda item: item["local_path"])
    if limit is not None:
        records = records[:limit]

    if prune and (source_ids or title_contains or limit is not None):
        raise ValueError("Pruning is only allowed for an unfiltered full-manifest build")

    indexed = skipped = needs_ocr = failed = removed = indexed_chunks = 0
    manifest_ids = {record["document_id"] for record in all_records}
    with closing(_connect(index_file)) as connection, connection:
        pending: list[dict[str, Any]] = []
        for record in records:
            existing = connection.execute(
                "SELECT file_sha256, pipeline_version, extraction_status FROM documents WHERE document_id = ?",
                (record["document_id"],),
            ).fetchone()
            if (
                existing
                and existing["file_sha256"] == record["file_sha256"]
                and existing["pipeline_version"] == PIPELINE_VERSION
                and existing["extraction_status"] == "indexed"
                and not force
            ):
                skipped += 1
                continue
            if not (datasets_dir / record["local_path"]).is_file():
                LOGGER.error("Manifest path does not exist: %s", datasets_dir / record["local_path"])
                failed += 1
                continue
            pending.append(record)

        def store(record: dict[str, Any], result: tuple) -> None:
            nonlocal indexed, needs_ocr, failed, indexed_chunks
            pages, chunks, extraction_status, error = result
            if error is not None:
                LOGGER.error("Failed to index %s: %s", record["local_path"], error)
                with connection:
                    _store_document(connection, record, [], [], extraction_status="failed", extraction_error=error[:1000])
                failed += 1
                return
            with connection:
                _store_document(connection, record, pages, chunks, extraction_status=extraction_status)
            indexed += 1
            indexed_chunks += len(chunks)
            needs_ocr += extraction_status == "needs_ocr"

        if workers <= 1:
            for record in pending:
                store(record, _extract_and_chunk(str(datasets_dir / record["local_path"]), extractor))
        else:
            from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

            with ProcessPoolExecutor(max_workers=workers) as pool:
                queue = iter(pending)
                running: dict[Any, dict[str, Any]] = {}
                done_count = 0
                while True:
                    # Keep a bounded number of extractions in flight so memory stays flat.
                    while len(running) < workers * 3:
                        record = next(queue, None)
                        if record is None:
                            break
                        running[pool.submit(_extract_and_chunk, str(datasets_dir / record["local_path"]), extractor)] = record
                    if not running:
                        break
                    finished, _ = wait(running, return_when=FIRST_COMPLETED)
                    for future in finished:
                        record = running.pop(future)
                        try:
                            result = future.result()
                        except Exception as exc:  # a crashed worker process
                            result = ([], [], "failed", str(exc))
                        store(record, result)
                        done_count += 1
                        if done_count % 500 == 0:
                            LOGGER.info("Indexed %d/%d documents", done_count, len(pending))

        if prune:
            stale = [
                row["document_id"]
                for row in connection.execute("SELECT document_id FROM documents")
                if row["document_id"] not in manifest_ids
            ]
            with connection:
                for document_id in stale:
                    _delete_document(connection, document_id)
            removed = len(stale)

        with connection:
            metadata = {
                "schema_version": SCHEMA_VERSION,
                "pipeline_version": PIPELINE_VERSION,
                "canonical_manifest": str(manifest_file.resolve()),
                "last_build_at_utc": utc_now(),
            }
            connection.executemany(
                "INSERT OR REPLACE INTO index_metadata(key, value) VALUES(?, ?)",
                metadata.items(),
            )

    return IndexBuildSummary(
        selected_documents=len(records),
        indexed_documents=indexed,
        skipped_unchanged=skipped,
        needs_ocr=needs_ocr,
        failed_documents=failed,
        removed_stale_documents=removed,
        indexed_chunks=indexed_chunks,
        index_file=str(index_file),
    )


def index_status(index_file: Path) -> dict[str, Any]:
    if not index_file.exists():
        return {"ready": False, "index_file": str(index_file), "documents": 0, "chunks": 0}
    try:
        with closing(_connect(index_file)) as connection, connection:
            status_counts = {
                row["extraction_status"]: row["count"]
                for row in connection.execute(
                    "SELECT extraction_status, COUNT(*) AS count FROM documents GROUP BY extraction_status"
                )
            }
            documents = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            chunks = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            metadata = dict(connection.execute("SELECT key, value FROM index_metadata"))
        return {
            "ready": chunks > 0,
            "index_file": str(index_file),
            "documents": documents,
            "chunks": chunks,
            "extraction_status": status_counts,
            "metadata": metadata,
        }
    except sqlite3.DatabaseError as exc:
        return {"ready": False, "index_file": str(index_file), "error": str(exc)}
