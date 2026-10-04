from __future__ import annotations

import json
import sqlite3

import pytest

from qanoon_ai.ingestion.legal_index import (
    PageText,
    build_legal_index,
    chunk_legal_document,
    index_status,
)
from qanoon_ai.retrieval.sqlite import SQLiteFTSRetriever


def test_legal_chunker_preserves_section_and_page_metadata():
    pages = [
        PageText(
            page_number=3,
            text=(
                "Section 1. Short title and commencement.\n"
                "This Act may be called the Example Act.\n\n"
                "Section 2. Punishment for theft.\n"
                "A person convicted of theft may be punished according to law."
            ),
        )
    ]

    chunks = chunk_legal_document(pages, target_words=50, max_words=100)

    assert chunks
    assert chunks[0].page_start == 3
    assert chunks[0].page_end == 3
    assert chunks[0].section_ref == "Section 1"
    assert "Section 2" not in chunks[0].text
    assert chunks[1].section_ref == "Section 2"
    assert "Punishment for theft" in chunks[1].text


def test_sections_keep_cross_page_qualifications_and_do_not_absorb_previous_section():
    pages = [
        PageText(1, "3. Who may testify. All persons may testify subject to the following conditions."),
        PageText(2, "Provided that a person must understand the questions.\n4. Other rule. Another complete rule."),
    ]
    chunks = chunk_legal_document(pages)
    assert chunks[0].section_ref == "Section 3"
    assert chunks[0].page_end == 2
    assert "Provided that" in chunks[0].text
    assert "Other rule" not in chunks[0].text
    assert chunks[1].section_ref == "Section 4"


def test_incremental_index_build_and_filtered_retrieval(tmp_path):
    datasets_dir = tmp_path / "datasets"
    pdf_path = datasets_dir / "raw" / "federal" / "example" / "example.pdf"
    pdf_path.parent.mkdir(parents=True)
    pdf_path.write_bytes(b"%PDF-1.7\ntest")
    manifest_dir = datasets_dir / "manifests"
    manifest_dir.mkdir(parents=True)
    manifest_file = manifest_dir / "canonical_manifest.jsonl"
    manifest_file.write_text(
        json.dumps(
            {
                "document_id": "example:sha:path",
                "title": "Example Penal Act, 2024",
                "local_path": "raw/federal/example/example.pdf",
                "source_id": "example",
                "source_url": "https://example.gov.pk/example.pdf",
                "authority_type": "legislation",
                "jurisdiction": "federal",
                "year": 2024,
                "document_type": "act",
                "legal_status": "current",
                "language": "english",
                "file_sha256": "a" * 64,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    index_file = tmp_path / "qanoon.sqlite3"

    def extractor(_path):
        return [
            PageText(
                7,
                "Section 9. Punishment for theft. "
                "Whoever commits theft may be punished after conviction by a court. "
                "This provision applies subject to the remaining provisions of this Act.",
            )
        ]

    first = build_legal_index(
        datasets_dir=datasets_dir,
        manifest_file=manifest_file,
        index_file=index_file,
        extractor=extractor,
    )
    second = build_legal_index(
        datasets_dir=datasets_dir,
        manifest_file=manifest_file,
        index_file=index_file,
        extractor=extractor,
    )

    assert first.indexed_documents == 1
    assert first.indexed_chunks == 1
    assert second.skipped_unchanged == 1
    assert index_status(index_file)["ready"]

    retriever = SQLiteFTSRetriever(index_file)
    results = retriever.search("punishment theft", jurisdiction="federal", top_k=3)
    assert len(results) == 1
    assert results[0].source_id == "example"
    assert results[0].section_ref == "Section 9"
    assert results[0].page_start == 7
    assert not retriever.search("punishment theft", jurisdiction="sindh", top_k=3)


def test_index_records_documents_that_need_ocr(tmp_path):
    datasets_dir = tmp_path / "datasets"
    pdf_path = datasets_dir / "raw" / "sindh" / "example" / "scan.pdf"
    pdf_path.parent.mkdir(parents=True)
    pdf_path.write_bytes(b"%PDF-1.7\nscan")
    manifest_dir = datasets_dir / "manifests"
    manifest_dir.mkdir(parents=True)
    manifest_file = manifest_dir / "canonical_manifest.jsonl"
    manifest_file.write_text(
        json.dumps(
            {
                "document_id": "scan:sha:path",
                "title": "Scanned Act",
                "local_path": "raw/sindh/example/scan.pdf",
                "source_id": "example",
                "jurisdiction": "sindh",
                "file_sha256": "b" * 64,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = build_legal_index(
        datasets_dir=datasets_dir,
        manifest_file=manifest_file,
        index_file=tmp_path / "scan.sqlite3",
        extractor=lambda _path: [PageText(1, "")],
    )

    assert result.indexed_documents == 1
    assert result.needs_ocr == 1
    assert result.indexed_chunks == 0


@pytest.mark.parametrize("initial_status", ["failed", "needs_ocr"])
def test_incomplete_documents_are_retried_without_force(tmp_path, initial_status):
    pdf = tmp_path / "example.pdf"
    pdf.write_bytes(b"%PDF-1.7\nexample")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(dict(document_id="example", file_sha256="a" * 64,
                                       title="Example", local_path=pdf.name, source_id="example")), encoding="utf-8")
    def first_extractor(path):
        if initial_status == "failed":
            raise RuntimeError("Temporary extraction failure")
        return [PageText(1, "")]
    options = dict(datasets_dir=tmp_path, manifest_file=manifest, index_file=tmp_path / "index.sqlite3")
    build_legal_index(**options, extractor=first_extractor)
    recovered = build_legal_index(**options, extractor=lambda path: [PageText(1, "1. A person shall comply with this provision. " * 5)])
    assert recovered.skipped_unchanged == 0 and recovered.indexed_chunks > 0
    assert index_status(options["index_file"])["extraction_status"] == {"indexed": 1}


def test_build_and_search_close_connections_even_when_references_remain(tmp_path, monkeypatch):
    connections = []
    real_connect = sqlite3.connect
    def connect(*args, **kwargs):
        connection = real_connect(*args, **kwargs)
        connections.append(connection)
        return connection
    monkeypatch.setattr(sqlite3, "connect", connect)
    pdf = tmp_path / "example.pdf"
    pdf.write_bytes(b"%PDF-1.7\nexample")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(dict(document_id="example", file_sha256="a" * 64,
                                       title="Penal Code", local_path=pdf.name, source_id="example")), encoding="utf-8")
    index = tmp_path / "index.sqlite3"
    build_legal_index(datasets_dir=tmp_path, manifest_file=manifest, index_file=index,
                      extractor=lambda path: [PageText(1, "1. Theft. Whoever commits theft shall be punished. " * 10)])
    assert index_status(index)["ready"]
    assert SQLiteFTSRetriever(index).search("theft")
    assert connections
    for connection in connections:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")
