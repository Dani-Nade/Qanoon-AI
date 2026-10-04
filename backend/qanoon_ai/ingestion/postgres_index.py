"""Production canonical-corpus builder for PostgreSQL and pgvector."""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
import numpy as np
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qanoon_ai.ingestion.legal_index import (
    MIN_TEXT_CHARS,
    PIPELINE_VERSION,
    PageText,
    chunk_legal_document,
    extract_pdf_pages_with_ocr,
    read_canonical_manifest,
)

LOGGER = logging.getLogger(__name__)
POSTGRES_SCHEMA_VERSION = "2"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ProductionBuildSummary:
    run_id: str
    manifest_documents: int
    selected_documents: int
    indexed_documents: int
    skipped_unchanged: int
    needs_ocr: int
    failed_documents: int
    removed_stale_documents: int
    indexed_chunks: int
    database_documents: int
    database_chunks: int
    duplicate_hashes: int
    missing_manifest_files: int
    missing_embeddings: int
    orphan_chunks: int
    indexed_documents_without_chunks: int
    document_count_mismatch: int
    embedding_model: str
    pipeline_version: str
    started_at_utc: str
    finished_at_utc: str
    report_file: str


class DenseEmbedder:
    """Lazy multilingual embedding model shared across the complete build."""

    def __init__(self, model_name: str, batch_size: int, revision: str = "") -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.revision = revision
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, revision=self.revision or None)
        return self._model

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.astype("float32").tolist()


@dataclass(frozen=True)
class ExtractedDocument:
    record: dict[str, Any]
    pages: list[PageText]
    chunks: list[Any]
    extraction_status: str


def _extract_record(
    record: dict[str, Any],
    *,
    datasets_dir: Path,
    enable_ocr: bool,
) -> ExtractedDocument:
    pdf_path = datasets_dir / record["local_path"]
    pages = extract_pdf_pages_with_ocr(pdf_path, enable_ocr=enable_ocr)
    text_chars = sum(len(page.text.strip()) for page in pages)
    if text_chars < MIN_TEXT_CHARS:
        return ExtractedDocument(record, pages, [], "needs_ocr")
    chunks = chunk_legal_document(pages)
    return ExtractedDocument(
        record,
        pages,
        chunks,
        "indexed" if chunks else "needs_ocr",
    )


def _iter_extracted(
    records: list[dict[str, Any]],
    *,
    datasets_dir: Path,
    enable_ocr: bool,
    workers: int,
):
    """Bound extraction concurrency so memory use stays stable on a full corpus."""

    if workers <= 1:
        for record in records:
            try:
                yield record, _extract_record(
                    record,
                    datasets_dir=datasets_dir,
                    enable_ocr=enable_ocr,
                ), None
            except Exception as exc:
                yield record, None, exc
        return

    iterator = iter(records)
    pending: dict[Future, dict[str, Any]] = {}
    max_pending = max(workers * 2, workers)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="qanoon-extract") as executor:
        for _ in range(max_pending):
            try:
                record = next(iterator)
            except StopIteration:
                break
            future = executor.submit(
                _extract_record,
                record,
                datasets_dir=datasets_dir,
                enable_ocr=enable_ocr,
            )
            pending[future] = record

        while pending:
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                record = pending.pop(future)
                try:
                    yield record, future.result(), None
                except Exception as exc:
                    yield record, None, exc
                try:
                    next_record = next(iterator)
                except StopIteration:
                    continue
                next_future = executor.submit(
                    _extract_record,
                    next_record,
                    datasets_dir=datasets_dir,
                    enable_ocr=enable_ocr,
                )
                pending[next_future] = next_record


def _connect(database_url: str, *, register_vectors: bool = True):
    import psycopg

    connection = psycopg.connect(database_url, autocommit=True, connect_timeout=5)
    if register_vectors:
        from pgvector.psycopg import register_vector

        register_vector(connection)
    return connection


def initialize_production_schema(database_url: str, embedding_dimension: int) -> None:
    if embedding_dimension < 1:
        raise ValueError("Embedding dimension must be positive")
    connection = _connect(database_url, register_vectors=False)
    try:
        with connection.transaction():
            connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
            connection.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
        from pgvector.psycopg import register_vector

        register_vector(connection)
        with connection.transaction():
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS system_metadata (
                    key text PRIMARY KEY,
                    value text NOT NULL,
                    updated_at timestamptz NOT NULL DEFAULT now()
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS legal_documents (
                    document_id text PRIMARY KEY,
                    file_sha256 char(64) NOT NULL UNIQUE,
                    title text NOT NULL,
                    manifest_title text NOT NULL,
                    local_path text NOT NULL UNIQUE,
                    source_id text NOT NULL,
                    source_url text,
                    authority_type text,
                    jurisdiction text,
                    year integer,
                    document_type text,
                    legal_status text,
                    language text,
                    page_count integer NOT NULL DEFAULT 0,
                    text_chars bigint NOT NULL DEFAULT 0,
                    extraction_status text NOT NULL,
                    extraction_error text,
                    pipeline_version text NOT NULL,
                    indexed_at timestamptz NOT NULL DEFAULT now()
                )
                """
            )
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS legal_chunks (
                    chunk_id text PRIMARY KEY,
                    document_id text NOT NULL REFERENCES legal_documents(document_id) ON DELETE CASCADE,
                    chunk_index integer NOT NULL,
                    title text NOT NULL,
                    page_start integer NOT NULL,
                    page_end integer NOT NULL,
                    section_ref text,
                    heading text,
                    text text NOT NULL,
                    word_count integer NOT NULL,
                    embedding vector({int(embedding_dimension)}),
                    search_vector tsvector GENERATED ALWAYS AS (
                        setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
                        setweight(to_tsvector('english', coalesce(section_ref, '')), 'A') ||
                        setweight(to_tsvector('english', coalesce(heading, '')), 'B') ||
                        setweight(to_tsvector('english', coalesce(text, '')), 'C')
                    ) STORED,
                    UNIQUE(document_id, chunk_index)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS build_runs (
                    run_id uuid PRIMARY KEY,
                    status text NOT NULL,
                    manifest_path text NOT NULL,
                    pipeline_version text NOT NULL,
                    embedding_model text NOT NULL,
                    summary jsonb NOT NULL DEFAULT '{}'::jsonb,
                    started_at timestamptz NOT NULL,
                    finished_at timestamptz
                )
                """
            )
            # Existing rows remain unknown until rebuilt; never label old vectors
            # as if they had been generated by the currently configured model.
            connection.execute("ALTER TABLE legal_documents ADD COLUMN IF NOT EXISTS embedding_model text")
            connection.execute("ALTER TABLE legal_documents ADD COLUMN IF NOT EXISTS embedding_revision text")
            connection.execute("ALTER TABLE legal_documents ADD COLUMN IF NOT EXISTS embedding_dimension integer")
            vector_type = connection.execute(
                "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
                "WHERE attrelid = 'legal_chunks'::regclass AND attname = 'embedding'"
            ).fetchone()[0]
            if vector_type != f"vector({embedding_dimension})":
                raise ValueError(
                    f"Existing index uses {vector_type}; configured dimension is {embedding_dimension}. "
                    "Build into a separate database when changing vector dimensions."
                )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS build_failures (
                    run_id uuid NOT NULL REFERENCES build_runs(run_id) ON DELETE CASCADE,
                    document_id text NOT NULL,
                    local_path text NOT NULL,
                    error text NOT NULL,
                    created_at timestamptz NOT NULL DEFAULT now(),
                    PRIMARY KEY(run_id, document_id)
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_legal_documents_jurisdiction "
                "ON legal_documents(jurisdiction)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_legal_documents_source ON legal_documents(source_id)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_legal_documents_type ON legal_documents(document_type)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_legal_chunks_document ON legal_chunks(document_id)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_legal_chunks_search ON legal_chunks USING gin(search_vector)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_legal_chunks_title_trgm "
                "ON legal_chunks USING gin(title gin_trgm_ops)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_legal_chunks_embedding "
                "ON legal_chunks USING hnsw (embedding vector_cosine_ops)"
            )
            connection.execute(
                """
                INSERT INTO system_metadata(key, value)
                VALUES ('postgres_schema_version', %s)
                ON CONFLICT (key) DO UPDATE
                SET value = excluded.value, updated_at = now()
                """,
                (POSTGRES_SCHEMA_VERSION,),
            )
    finally:
        connection.close()


def _display_title(record: dict[str, Any], pages: list[PageText]) -> str:
    manifest_title = str(record["title"]).strip()
    if not manifest_title.lower().startswith("administrator") or not pages:
        return manifest_title
    legal_markers = (" act", " code", " constitution", " ordinance", " order", " rules")
    for line in pages[0].text.splitlines()[:40]:
        candidate = " ".join(line.split()).strip(" _-")
        lowered = f" {candidate.lower()}"
        if 8 <= len(candidate) <= 220 and any(marker in lowered for marker in legal_markers):
            return candidate
    return manifest_title


def _validate_manifest(records: list[dict[str, Any]], datasets_dir: Path) -> dict[str, int]:
    document_ids = [record["document_id"] for record in records]
    hashes = [record["file_sha256"] for record in records]
    paths = [record["local_path"] for record in records]
    missing = sum(not (datasets_dir / path).is_file() for path in paths)
    duplicate_ids = len(document_ids) - len(set(document_ids))
    duplicate_hashes = len(hashes) - len(set(hashes))
    duplicate_paths = len(paths) - len(set(paths))
    if duplicate_ids or duplicate_hashes or duplicate_paths or missing:
        raise ValueError(
            "Canonical manifest validation failed: "
            f"duplicate_ids={duplicate_ids} duplicate_hashes={duplicate_hashes} "
            f"duplicate_paths={duplicate_paths} missing_files={missing}"
        )
    return {"duplicate_hashes": duplicate_hashes, "missing_files": missing}


def _existing_documents(connection) -> dict[str, tuple]:
    rows = connection.execute(
        "SELECT document_id, file_sha256, pipeline_version, extraction_status, "
        "embedding_model, embedding_revision, embedding_dimension FROM legal_documents"
    ).fetchall()
    return {str(row[0]): tuple(row[1:]) for row in rows}


def _document_unchanged(prior: tuple | None, record: dict[str, Any],
                        model: str, revision: str, dimension: int) -> bool:
    return prior == (record["file_sha256"], PIPELINE_VERSION, "indexed", model, revision, dimension)


def _store_document(
    connection,
    record: dict[str, Any],
    pages: list[PageText],
    chunks,
    embeddings: list[list[float]],
    *,
    extraction_status: str,
    extraction_error: str | None = None,
    embedding_model: str = "",
    embedding_revision: str = "",
    embedding_dimension: int | None = None,
) -> None:
    title = _display_title(record, pages)
    conflicts = connection.execute(
        """
        SELECT document_id FROM legal_documents
        WHERE document_id = %s OR local_path = %s OR file_sha256 = %s
        """,
        (record["document_id"], record["local_path"], record["file_sha256"]),
    ).fetchall()
    for conflict in conflicts:
        connection.execute("DELETE FROM legal_documents WHERE document_id = %s", (conflict[0],))
    connection.execute(
        """
        INSERT INTO legal_documents(
            document_id, file_sha256, title, manifest_title, local_path, source_id,
            source_url, authority_type, jurisdiction, year, document_type,
            legal_status, language, page_count, text_chars, extraction_status,
            extraction_error, pipeline_version, embedding_model, embedding_revision,
            embedding_dimension, indexed_at
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s, now()
        )
        """,
        (
            record["document_id"], record["file_sha256"], title, record["title"],
            record["local_path"], record["source_id"], record.get("source_url"),
            record.get("authority_type"), record.get("jurisdiction"), record.get("year"),
            record.get("document_type"), record.get("legal_status"), record.get("language"),
            len(pages), sum(len(page.text) for page in pages), extraction_status,
            extraction_error, PIPELINE_VERSION, embedding_model, embedding_revision, embedding_dimension,
        ),
    )
    rows = []
    for chunk, embedding in zip(chunks, embeddings, strict=True):
        identity = f"{record['file_sha256']}:{chunk.chunk_index}:{chunk.text}".encode("utf-8")
        chunk_id = f"chunk:{hashlib.sha256(identity).hexdigest()[:24]}"
        rows.append(
            (
                chunk_id, record["document_id"], chunk.chunk_index, title,
                chunk.page_start, chunk.page_end, chunk.section_ref, chunk.heading,
                chunk.text, chunk.word_count, np.asarray(embedding, dtype=np.float32),
            )
        )
    if rows:
        with connection.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO legal_chunks(
                    chunk_id, document_id, chunk_index, title, page_start, page_end,
                    section_ref, heading, text, word_count, embedding
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                rows,
            )


def _write_report(report_file: Path, payload: dict[str, Any]) -> None:
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def build_production_index(
    *,
    database_url: str,
    datasets_dir: Path,
    manifest_file: Path,
    report_file: Path,
    embedding_model: str,
    embedding_dimension: int,
    embedding_batch_size: int,
    extraction_workers: int,
    enable_ocr: bool,
    embedding_revision: str = "",
    source_ids: set[str] | None = None,
    limit: int | None = None,
    force: bool = False,
    prune: bool = True,
) -> ProductionBuildSummary:
    """Validate, incrementally index, verify, and report one canonical build."""

    started_at = _utc_now()
    run_id = str(uuid.uuid4())
    records = read_canonical_manifest(manifest_file)
    validation = _validate_manifest(records, datasets_dir)
    selected = records
    if source_ids:
        selected = [record for record in selected if record["source_id"] in source_ids]
    selected = sorted(selected, key=lambda record: record["local_path"])
    if limit is not None:
        selected = selected[:limit]
    if prune and (source_ids or limit is not None):
        raise ValueError("Pruning requires an unfiltered full-manifest build")

    initialize_production_schema(database_url, embedding_dimension)
    embedder = DenseEmbedder(embedding_model, embedding_batch_size, embedding_revision)
    connection = _connect(database_url)
    indexed = skipped = needs_ocr = failed = removed = indexed_chunks = 0
    try:
        connection.execute(
            """
            INSERT INTO build_runs(
                run_id, status, manifest_path, pipeline_version,
                embedding_model, started_at
            ) VALUES (%s, 'running', %s, %s, %s, %s)
            """,
            (run_id, str(manifest_file.resolve()), PIPELINE_VERSION, embedding_model, started_at),
        )
        existing = _existing_documents(connection)
        to_process: list[dict[str, Any]] = []
        for record in selected:
            prior = existing.get(record["document_id"])
            if not force and _document_unchanged(prior, record, embedding_model, embedding_revision, embedding_dimension):
                skipped += 1
                continue
            to_process.append(record)

        for record, extracted, extraction_error in _iter_extracted(
            to_process,
            datasets_dir=datasets_dir,
            enable_ocr=enable_ocr,
            workers=max(1, extraction_workers),
        ):
            pdf_path = datasets_dir / record["local_path"]
            try:
                if extraction_error is not None:
                    raise extraction_error
                assert extracted is not None
                pages = extracted.pages
                chunks = extracted.chunks
                status = extracted.extraction_status
                if status == "needs_ocr":
                    needs_ocr += 1
                if chunks:
                    passages = [
                        "\n".join(
                            part
                            for part in (
                                record["title"], chunk.section_ref, chunk.heading, chunk.text
                            )
                            if part
                        )
                        for chunk in chunks
                    ]
                    embeddings = embedder.encode(passages)
                    if embeddings and len(embeddings[0]) != embedding_dimension:
                        raise ValueError(
                            f"Embedding dimension {len(embeddings[0])} does not match "
                            f"configured dimension {embedding_dimension}"
                        )
                else:
                    embeddings = []
                with connection.transaction():
                    _store_document(
                        connection,
                        record,
                        pages,
                        chunks,
                        embeddings,
                        extraction_status=status,
                        embedding_model=embedding_model,
                        embedding_revision=embedding_revision,
                        embedding_dimension=embedding_dimension,
                    )
                indexed += 1
                indexed_chunks += len(chunks)
            except Exception as exc:
                LOGGER.exception("Production indexing failed for %s", pdf_path)
                connection.rollback()
                with connection.transaction():
                    connection.execute(
                        """
                        INSERT INTO build_failures(run_id, document_id, local_path, error)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (run_id, document_id) DO UPDATE
                        SET error = excluded.error, created_at = now()
                        """,
                        (
                            run_id,
                            record["document_id"],
                            record["local_path"],
                            str(exc)[:2000],
                        ),
                    )
                failed += 1

        if prune:
            manifest_ids = {record["document_id"] for record in records}
            database_ids = {
                str(row[0]) for row in connection.execute("SELECT document_id FROM legal_documents")
            }
            stale = database_ids - manifest_ids
            with connection.transaction():
                for document_id in stale:
                    connection.execute(
                        "DELETE FROM legal_documents WHERE document_id = %s", (document_id,)
                    )
            removed = len(stale)

        database_documents = connection.execute(
            "SELECT count(*) FROM legal_documents"
        ).fetchone()[0]
        database_chunks = connection.execute("SELECT count(*) FROM legal_chunks").fetchone()[0]
        duplicate_hashes = connection.execute(
            """
            SELECT count(*) FROM (
                SELECT file_sha256 FROM legal_documents
                GROUP BY file_sha256 HAVING count(*) > 1
            ) duplicate_groups
            """
        ).fetchone()[0]
        missing_embeddings = connection.execute(
            "SELECT count(*) FROM legal_chunks WHERE embedding IS NULL"
        ).fetchone()[0]
        orphan_chunks = connection.execute(
            """
            SELECT count(*) FROM legal_chunks c
            LEFT JOIN legal_documents d ON d.document_id = c.document_id
            WHERE d.document_id IS NULL
            """
        ).fetchone()[0]
        indexed_without_chunks = connection.execute(
            """
            SELECT count(*) FROM legal_documents d
            WHERE d.extraction_status = 'indexed'
              AND NOT EXISTS (
                  SELECT 1 FROM legal_chunks c WHERE c.document_id = d.document_id
              )
            """
        ).fetchone()[0]
        count_mismatch = int(prune and database_documents != len(records))
        finished_at = _utc_now()
        summary = ProductionBuildSummary(
            run_id=run_id,
            manifest_documents=len(records),
            selected_documents=len(selected),
            indexed_documents=indexed,
            skipped_unchanged=skipped,
            needs_ocr=needs_ocr,
            failed_documents=failed,
            removed_stale_documents=removed,
            indexed_chunks=indexed_chunks,
            database_documents=database_documents,
            database_chunks=database_chunks,
            duplicate_hashes=duplicate_hashes,
            missing_manifest_files=validation["missing_files"],
            missing_embeddings=missing_embeddings,
            orphan_chunks=orphan_chunks,
            indexed_documents_without_chunks=indexed_without_chunks,
            document_count_mismatch=count_mismatch,
            embedding_model=embedding_model,
            pipeline_version=PIPELINE_VERSION,
            started_at_utc=started_at,
            finished_at_utc=finished_at,
            report_file=str(report_file),
        )
        payload = asdict(summary)
        _write_report(report_file, payload)
        quality_errors = any(
            (
                failed,
                duplicate_hashes,
                missing_embeddings,
                orphan_chunks,
                indexed_without_chunks,
                count_mismatch,
            )
        )
        with connection.transaction():
            connection.execute(
                """
                UPDATE build_runs
                SET status = %s, summary = %s::jsonb, finished_at = %s
                WHERE run_id = %s
                """,
                (
                    "completed" if not quality_errors else "completed_with_errors",
                    json.dumps(payload), finished_at, run_id,
                ),
            )
            if not quality_errors:
                connection.execute(
                    """
                    INSERT INTO system_metadata(key, value)
                    VALUES ('active_index_run_id', %s)
                    ON CONFLICT (key) DO UPDATE
                    SET value = excluded.value, updated_at = now()
                    """,
                    (run_id,),
                )
        return summary
    except Exception:
        connection.rollback()
        finished_at = _utc_now()
        try:
            with connection.transaction():
                connection.execute(
                    """
                    UPDATE build_runs SET status = 'failed', finished_at = %s
                    WHERE run_id = %s
                    """,
                    (finished_at, run_id),
                )
        except Exception:
            LOGGER.exception("Unable to mark build %s as failed", run_id)
        raise
    finally:
        connection.close()


def production_index_status(database_url: str, *, embedding_model: str | None = None,
                            embedding_revision: str = "", embedding_dimension: int = 1024) -> dict[str, Any]:
    try:
        connection = _connect(database_url)
        try:
            documents = connection.execute("SELECT count(*) FROM legal_documents").fetchone()[0]
            chunks = connection.execute("SELECT count(*) FROM legal_chunks").fetchone()[0]
            compatible_chunks = chunks
            if embedding_model is not None:
                compatible_chunks = connection.execute(
                    "SELECT count(*) FROM legal_chunks c JOIN legal_documents d USING(document_id) "
                    "WHERE d.embedding_model = %s AND d.embedding_revision = %s AND d.embedding_dimension = %s",
                    (embedding_model, embedding_revision, embedding_dimension),
                ).fetchone()[0]
            states = dict(
                connection.execute(
                    "SELECT extraction_status, count(*) FROM legal_documents GROUP BY extraction_status"
                ).fetchall()
            )
            active_run = connection.execute(
                "SELECT value FROM system_metadata WHERE key = 'active_index_run_id'"
            ).fetchone()
            return {
                "ready": compatible_chunks > 0,
                "compatible_chunks": compatible_chunks,
                "documents": documents,
                "chunks": chunks,
                "extraction_status": states,
                "active_index_run_id": active_run[0] if active_run else None,
            }
        finally:
            connection.close()
    except Exception as exc:
        return {"ready": False, "documents": 0, "chunks": 0, "error": str(exc)}
