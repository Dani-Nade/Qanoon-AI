"""Index downloaded federal/Punjab statutes plus the official Penal Code."""
import hashlib
import json
import logging
import re
import sqlite3
from pathlib import Path
import sys
from dataclasses import asdict
from contextlib import closing

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parent
sys.path.insert(0, str(BACKEND))
import requests
import pymupdf as fitz
from qanoon_ai.ingestion.legal_index import PageText, build_legal_index, clean_extracted_text


def extract(path):
    with fitz.open(path) as document:
        return [PageText(i + 1, clean_extracted_text(page.get_text(sort=True))) for i, page in enumerate(document)]


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    manifest = ROOT / "datasets/manifests/canonical_manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    url = "https://pakistancode.gov.pk/pdffiles/administratoracbdabf2b79c8956d6ea4804fcceb92d.pdf"
    path = ROOT / "datasets/raw/federal/federal_pakistan_code/pakistan_penal_code_1860.pdf"
    if not path.exists():
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        if not response.content.startswith(b"%PDF"):
            raise ValueError("Official Penal Code response was not a PDF")
        path.write_bytes(response.content)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if not any(row["file_sha256"] == digest for row in rows):
        record = {
            "document_id": f"federal_pakistan_code:{digest[:20]}:penal_code_1860",
            "title": "THE PAKISTAN PENAL CODE, 1860",
            "local_path": path.relative_to(ROOT / "datasets").as_posix(),
            "source_id": "federal_pakistan_code", "source_url": url,
            "file_sha256": digest, "jurisdiction": "federal", "year": 1860,
            "document_type": "code", "legal_status": "unverified_currentness",
            "authority_type": "legislation", "language": "english",
        }
        with manifest.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        rows.append(record)
    selected = [row for row in rows if row["source_id"] in {"federal_pakistan_code", "pakistan_code_archive", "punjab_code"}]
    # Record the actual build selection without changing other canonical records.
    selection = ROOT / "data/reports/answer-corpus-manifest.jsonl"
    selection.parent.mkdir(parents=True, exist_ok=True)
    selection.write_text("\n".join(json.dumps(row) for row in selected) + "\n", encoding="utf-8")
    print(f"Building {len(selected)} statute PDFs", flush=True)
    counts = {"processed": 0}

    def progress_extract(path):
        counts["processed"] += 1
        if counts["processed"] % 25 == 0:
            print(f"Extracted {counts['processed']}/{len(selected)}", flush=True)
        return extract(path)

    with closing(sqlite3.connect(ROOT / "data/index/qanoon_fts.sqlite3")) as connection:
        titles = dict(connection.execute("SELECT document_id, title FROM documents"))
    core = [row for row in selected if re.search(r"penal.?code|criminal.?procedure|family.?courts|shahadat|civil.?servants.?act", titles.get(row["document_id"], row["title"]), re.I)]
    core_manifest = ROOT / "data/reports/core-corpus-manifest.jsonl"
    core_manifest.write_text("\n".join(json.dumps(row) for row in core) + "\n", encoding="utf-8")
    build_legal_index(
        datasets_dir=ROOT / "datasets", manifest_file=core_manifest,
        index_file=ROOT / "data/index/qanoon_fts.sqlite3", extractor=progress_extract,
    )
    print(f"Core statutes ready ({len(core)} PDFs)", flush=True)
    result = build_legal_index(
        datasets_dir=ROOT / "datasets", manifest_file=selection,
        index_file=ROOT / "data/index/qanoon_fts.sqlite3", extractor=progress_extract,
    )
    report = asdict(result)
    (ROOT / "data/reports/answer-corpus-build.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
