"""Opt-in SQL tests: set QANOON_TEST_DATABASE_URL to a disposable pgvector database.

Each test owns a unique schema; no project corpus or model downloads are used.
"""

import json
import os
import uuid
from unittest.mock import Mock

import pytest

from qanoon_ai.ingestion import postgres_index as pg
from qanoon_ai.ingestion.legal_index import PageText, chunk_legal_document
from qanoon_ai.retrieval.hybrid import PostgresHybridRetriever


@pytest.fixture
def database():
    base = os.getenv("QANOON_TEST_DATABASE_URL")
    if not base:
        pytest.skip("Set QANOON_TEST_DATABASE_URL for isolated PostgreSQL tests")
    psycopg = pytest.importorskip("psycopg")
    from psycopg import sql
    from psycopg.conninfo import make_conninfo

    schema = "qanoon_test_" + uuid.uuid4().hex
    with psycopg.connect(base, autocommit=True) as connection:
        connection.execute("CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public")
        connection.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm WITH SCHEMA public")
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    try:
        yield make_conninfo(base, options=f"-c search_path={schema},public")
    finally:
        with psycopg.connect(base, autocommit=True) as connection:
            connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


@pytest.fixture
def build_options(database, tmp_path, monkeypatch):
    (tmp_path / "example.pdf").write_bytes(b"%PDF-1.7\nfixture")
    record = dict(document_id="example", file_sha256="a" * 64, title="Example Penal Code",
                  local_path="example.pdf", source_id="example", jurisdiction="federal", year=2024)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(pg, "_extract_record", extract_fixture)
    monkeypatch.setattr(pg.DenseEmbedder, "encode", lambda self, texts: [[1., 0., 0.] for _ in texts])
    return dict(database_url=database, datasets_dir=tmp_path, manifest_file=manifest,
                report_file=tmp_path / "report.json", embedding_model="fixture-model",
                embedding_revision="revision-1", embedding_dimension=3, embedding_batch_size=2,
                extraction_workers=1, enable_ocr=True)


def extract_fixture(record, **kwargs):
    pages = [PageText(1, "1. Theft. Whoever commits theft shall be punished according to this provision. " * 3)]
    return pg.ExtractedDocument(record, pages, chunk_legal_document(pages), "indexed")


def retriever(database, model="fixture-model", revision="revision-1"):
    result = PostgresHybridRetriever(database_url=database, embedding_model=model,
                                     embedding_revision=revision, embedding_dimension=3,
                                     reranker_model="fixture-reranker")
    result._reranker = Mock(predict=lambda pairs, **kwargs: [5.] * len(pairs))
    return result


def test_build_and_search_execute_real_postgres_sql(database, build_options):
    summary = pg.build_production_index(**build_options)
    assert summary.failed_documents == 0
    assert summary.database_documents == 1 and summary.database_chunks > 0
    assert pg.build_production_index(**build_options).skipped_unchanged == 1
    search = retriever(database)
    assert search.search("theft", semantic_query="punishment for theft")[0].source_id == "example"
    assert search.search("theft", jurisdiction="federal")
    assert search.search("theft", jurisdiction="punjab") == []


def test_model_revision_and_legacy_metadata_require_reembedding(database, build_options):
    pg.build_production_index(**build_options)
    changed = {**build_options, "embedding_model": "new-model"}
    assert pg.build_production_index(**changed).indexed_documents == 1
    assert not retriever(database).ready
    assert retriever(database, model="new-model").ready
    revised = {**changed, "embedding_revision": "revision-2"}
    assert pg.build_production_index(**revised).indexed_documents == 1
    assert not retriever(database, model="new-model").ready
    assert retriever(database, model="new-model", revision="revision-2").search("theft")
    import psycopg
    with psycopg.connect(database, autocommit=True) as connection:
        for column in ("embedding_model", "embedding_revision", "embedding_dimension"):
            connection.execute(f"ALTER TABLE legal_documents DROP COLUMN {column}")
    assert pg.build_production_index(**revised).indexed_documents == 1
    with pytest.raises(ValueError, match="separate database"):
        pg.build_production_index(**{**revised, "embedding_dimension": 4})
    assert retriever(database, model="new-model", revision="revision-2").ready


def test_ocr_records_are_retried(database, build_options, monkeypatch):
    monkeypatch.setattr(pg, "_extract_record", lambda record, **kwargs: pg.ExtractedDocument(record, [], [], "needs_ocr"))
    assert pg.build_production_index(**build_options).needs_ocr == 1
    assert not retriever(database).ready
    monkeypatch.setattr(pg, "_extract_record", extract_fixture)
    recovered = pg.build_production_index(**build_options)
    assert recovered.indexed_documents == 1 and recovered.skipped_unchanged == 0
    assert retriever(database).search("theft")
