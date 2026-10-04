"""Production adapter contracts, without loading models or requiring a database."""

from dataclasses import replace
from unittest.mock import Mock, PropertyMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from qanoon_ai.ingestion.legal_index import LegalChunk, PageText, PIPELINE_VERSION
from qanoon_ai.ingestion.postgres_index import _document_unchanged, _store_document
from qanoon_ai.retrieval.hybrid import PostgresHybridRetriever
from qanoon_ai.retrieval.sqlite import SQLiteFTSRetriever
from qanoon_ai.retrieval.models import EvidenceChunk


class RecordingConnection:
    """Only the connection/cursor methods that Psycopg actually exposes."""

    def __init__(self):
        self.statements = []
        self.batch = []
        self.cursor_closed = False

    def execute(self, sql, params):
        assert sql.count("%s") == len(params)
        self.statements.append((sql, params))
        return Mock(fetchall=lambda: [])

    def cursor(self):
        owner = self

        class Cursor:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                owner.cursor_closed = True

            def executemany(self, sql, rows):
                for row in rows:
                    assert sql.count("%s") == len(row)
                owner.batch.extend(rows)

        return Cursor()


def test_document_insert_uses_cursor_and_vector_adapter_values():
    connection = RecordingConnection()
    record = dict(document_id="doc", file_sha256="a" * 64, title="Example",
                  local_path="example.pdf", source_id="source")
    _store_document(
        connection, record, [PageText(1, "A legal provision.")],
        [LegalChunk(0, 1, 1, "Section 1", "Heading", "A legal provision.", 3)],
        [[1.0, 0.0, 0.0]], extraction_status="indexed",
        embedding_model="model", embedding_revision="revision", embedding_dimension=3,
    )
    assert len(connection.batch) == 1 and connection.cursor_closed
    assert isinstance(connection.batch[0][-1], np.ndarray)
    np.testing.assert_array_equal(connection.batch[0][-1], [1, 0, 0])
    assert connection.statements[-1][1][-3:] == ("model", "revision", 3)


@pytest.mark.parametrize("jurisdiction", [None, "kp"])
def test_search_parameters_bind_vectors_and_filter_embedding_identity(jurisdiction):
    retriever = PostgresHybridRetriever(database_url="unused", embedding_model="model",
                                       reranker_model="unused", embedding_revision="revision",
                                       embedding_dimension=3)
    connection = RecordingConnection()
    retriever._dense_search(connection, [1., 0., 0.], candidate_limit=5, jurisdiction=jurisdiction)
    sql, params = connection.statements[-1]
    assert isinstance(params[0], np.ndarray) and isinstance(params[-2], np.ndarray)
    assert params[1:4] == ["model", "revision", 3]
    assert "d.embedding_model = %s" in sql
    if jurisdiction:
        assert params[4] == ["khyber_pakhtunkhwa", "courts_khyber_pakhtunkhwa"]
    retriever._lexical_search(connection, "theft", candidate_limit=5, jurisdiction=jurisdiction)
    sql, params = connection.statements[-1]
    assert params[:5] == ["theft"] * 5
    assert params[5:8] == ["model", "revision", 3]
    assert "d.embedding_model = %s" in sql


@pytest.mark.parametrize("changed", [
    {"status": "needs_ocr"}, {"status": "failed"}, {"model": "other"},
    {"revision": "other"}, {"dimension": 4}, {"model": None}, {"pipeline": "old"},
])
def test_incremental_build_retries_incomplete_or_incompatible_documents(changed):
    values = dict(hash="a" * 64, pipeline=PIPELINE_VERSION, status="indexed",
                  model="model", revision="revision", dimension=3)
    record = {"file_sha256": values["hash"]}
    assert _document_unchanged(tuple(values.values()), record, "model", "revision", 3)
    values.update(changed)
    assert not _document_unchanged(tuple(values.values()), record, "model", "revision", 3)


@pytest.mark.parametrize("mode,postgres_ready,selected", [
    ("postgres", False, "postgres"), ("auto", True, "postgres"),
    ("auto", False, "sqlite"), ("sqlite", True, "sqlite"),
])
def test_chat_route_uses_the_configured_shared_retriever(monkeypatch, mode, postgres_ready, selected):
    import json
    from qanoon_ai.api.app import app
    from qanoon_ai.api.routes.chat import chat_service
    from qanoon_ai.api.routes.query import answer_service

    chunk = EvidenceChunk("one", "example.pdf", "Example", None, "A complete provision.", 1.)
    pg_search = Mock(return_value=[chunk])
    sqlite_search = Mock(return_value=[chunk])
    llm = Mock()
    llm.complete.return_value = json.dumps(dict(language="english", needs_law=True, queries=["theft"]))
    llm.stream.return_value = iter([("content", "Answer [1].")])
    monkeypatch.setattr(answer_service, "settings", replace(answer_service.settings, retrieval_mode=mode))
    monkeypatch.setattr(answer_service.postgres_retriever, "search", pg_search)
    monkeypatch.setattr(answer_service.sqlite_retriever, "search", sqlite_search)
    monkeypatch.setattr(chat_service, "llm", llm)
    with patch.object(PostgresHybridRetriever, "ready", new_callable=PropertyMock, return_value=postgres_ready), \
         patch.object(SQLiteFTSRetriever, "ready", new_callable=PropertyMock, return_value=True):
        response = TestClient(app).post("/chat", json={"messages": [{"role": "user", "content": "theft"}]})
    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.splitlines()]
    assert events[0]["citations"][0]["chunk_id"] == "one"
    assert events[-1]["type"] == "done"
    used, unused = (pg_search, sqlite_search) if selected == "postgres" else (sqlite_search, pg_search)
    used.assert_called_once()
    assert used.call_args.kwargs["semantic_query"] == "theft"
    unused.assert_not_called()
