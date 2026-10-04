from __future__ import annotations

import json

from qanoon_ai.ingestion.corpus_audit import audit_corpus
from qanoon_ai.ingestion.source_catalog import SourceCatalog


def test_audit_preserves_raw_and_builds_canonical_manifest(tmp_path):
    datasets_dir = tmp_path / "datasets"
    source_dir = datasets_dir / "raw" / "test_jurisdiction" / "test_source"
    source_dir.mkdir(parents=True)
    (source_dir / "2024 Test Act.pdf").write_bytes(b"%PDF-1.7\nlegal")
    (source_dir / "2024 Test Act copy.pdf").write_bytes(b"%PDF-1.7\nlegal")
    (source_dir / "bad.pdf").write_bytes(b"<html>blocked</html>")
    manifest_dir = datasets_dir / "manifests"
    manifest_dir.mkdir(parents=True)
    download_manifest = manifest_dir / "download_manifest.jsonl"
    download_manifest.write_text(
        json.dumps(
            {
                "source_id": "test_source",
                "document_url": "https://example.gov/2024-test-act.pdf",
                "local_path": "raw/test_jurisdiction/test_source/2024 Test Act.pdf",
                "status": "downloaded",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = audit_corpus(
        datasets_dir=datasets_dir,
        catalog=SourceCatalog(version="test", sources=()),
        download_manifest=download_manifest,
    )

    assert summary["raw_pdf_files"] == 3
    assert summary["curated_records"] == 2
    assert summary["canonical_records"] == 1
    assert summary["quarantined_records"] == 1
    assert (manifest_dir / "canonical_manifest.jsonl").read_text(encoding="utf-8").count("\n") == 1
