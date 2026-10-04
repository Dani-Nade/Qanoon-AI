"""Build deterministic corpus, curation, quarantine, and canonical manifests."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qanoon_ai.ingestion.source_catalog import SourceCatalog


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl_rows(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def _source_url_index(download_manifest: Path) -> tuple[dict[str, str], dict[str, set[str]]]:
    by_path: dict[str, str] = {}
    discovered: dict[str, set[str]] = defaultdict(set)
    for record in _jsonl_rows(download_manifest) or ():
        source_id = record.get("source_id")
        document_url = record.get("document_url")
        if source_id and document_url and record.get("status") != "discovery_error":
            discovered[source_id].add(document_url)
        local_path = record.get("local_path")
        if local_path and document_url:
            by_path[local_path.replace("\\", "/")] = document_url
    return by_path, discovered


def _prior_quarantine(path: Path) -> dict[str, str]:
    return {
        record["local_path"].replace("\\", "/"): record["reason"]
        for record in (_jsonl_rows(path) or ())
        if record.get("local_path") and record.get("reason")
    }


def _content_exclusions(datasets_dir: Path) -> tuple[dict[str, str], dict[str, str]]:
    path = datasets_dir / "registry" / "content_exclusions.json"
    if not path.exists():
        return {}, {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_path: dict[str, str] = {}
    by_hash: dict[str, str] = {}
    for exclusion in payload.get("exclusions", []):
        reason = exclusion["reason"]
        if exclusion.get("local_path"):
            by_path[exclusion["local_path"].replace("\\", "/")] = reason
        if exclusion.get("file_sha256"):
            by_hash[exclusion["file_sha256"]] = reason
    return by_path, by_hash


def _document_type(source_id: str, title: str) -> str:
    if "judgment" in source_id:
        return "judgment"
    lowered = title.lower()
    candidates = (
        ("constitution", "constitution"),
        ("rules", "rules"),
        ("regulation", "regulations"),
        ("ordinance", "ordinance"),
        ("notification", "notification"),
        ("order", "order"),
        ("act", "act"),
        ("code", "code"),
    )
    return next((kind for token, kind in candidates if token in lowered), "unknown")


def _year(path: Path, title: str) -> int | None:
    for part in reversed(path.parts[:-1]):
        if re.fullmatch(r"(?:18|19|20)\d{2}", part):
            return int(part)
    match = re.search(r"(?<!\d)((?:18|19|20)\d{2})(?!\d)", title)
    return int(match.group(1)) if match else None


def _authority_type(source_id: str) -> str:
    if "judgment" in source_id:
        return "judiciary"
    if "court" in source_id and "rule" in source_id:
        return "court_rules"
    return "legislation"


def _canonical_rank(record: dict[str, Any]) -> tuple[int, int, int, str]:
    source_id = record["source_id"]
    return (
        1 if record.get("source_url") else 0,
        0 if "archive" in source_id else 1,
        1 if record["legal_status"] != "repealed" else 0,
        record["local_path"],
    )


def audit_corpus(
    *,
    datasets_dir: Path,
    catalog: SourceCatalog,
    download_manifest: Path,
) -> dict[str, Any]:
    raw_dir = datasets_dir / "raw"
    manifest_dir = datasets_dir / "manifests"
    corpus_path = manifest_dir / "corpus_manifest.jsonl"
    curated_path = manifest_dir / "curated_manifest.jsonl"
    quarantine_path = manifest_dir / "quarantine_manifest.jsonl"
    canonical_path = manifest_dir / "canonical_manifest.jsonl"

    prior_reasons = _prior_quarantine(quarantine_path)
    excluded_paths, excluded_hashes = _content_exclusions(datasets_dir)
    source_urls, discovered_urls = _source_url_index(download_manifest)
    records: list[dict[str, Any]] = []

    for pdf_path in sorted(raw_dir.rglob("*.pdf")):
        relative = pdf_path.relative_to(datasets_dir)
        relative_posix = relative.as_posix()
        parts = relative.parts
        jurisdiction = parts[1] if len(parts) > 2 else "unknown"
        source_id = parts[2] if len(parts) > 3 else "unknown"
        title = re.sub(r"^[0-9a-f]{12}__", "", pdf_path.stem, flags=re.IGNORECASE)
        with pdf_path.open("rb") as stream:
            header = stream.read(1024)
        if header.startswith(b"%PDF-"):
            validation_status = "valid_pdf_signature"
        elif b"%PDF-" in header:
            validation_status = "nonstandard_pdf_preamble"
        else:
            validation_status = "invalid_pdf_signature"
        digest = _sha256(pdf_path)
        year = _year(relative, title)
        path_fingerprint = hashlib.sha1(relative_posix.encode("utf-8")).hexdigest()[:12]
        records.append(
            {
                "document_id": f"{source_id}:{digest[:20]}:{path_fingerprint}",
                "title": title,
                "local_path": relative_posix,
                "source_id": source_id,
                "source_url": source_urls.get(relative_posix),
                "authority_type": _authority_type(source_id),
                "jurisdiction": jurisdiction,
                "year": year,
                "document_type": _document_type(source_id, title),
                "legal_status": "repealed" if "repeal" in title.lower() else "unverified_currentness",
                "language": "english_or_unknown",
                "file_size_bytes": pdf_path.stat().st_size,
                "file_sha256": digest,
                "file_validation_status": validation_status,
                "text_status": "not_assessed",
                "quality_flags": [],
                "exact_duplicate_count": 1,
                "duplicate_group": None,
            }
        )

    hash_counts = Counter(record["file_sha256"] for record in records)
    for record in records:
        duplicate_count = hash_counts[record["file_sha256"]]
        flags: list[str] = []
        if duplicate_count > 1:
            flags.append("exact_content_duplicate")
        if record["legal_status"] == "repealed":
            flags.append("repealed_or_historical")
        if record["file_validation_status"] == "invalid_pdf_signature":
            flags.append("invalid_pdf_signature")
        elif record["file_validation_status"] == "nonstandard_pdf_preamble":
            flags.append("nonstandard_pdf_preamble")
        record["quality_flags"] = flags
        record["exact_duplicate_count"] = duplicate_count
        record["duplicate_group"] = (
            f"sha256:{record['file_sha256'][:20]}" if duplicate_count > 1 else None
        )

    quarantine: list[dict[str, Any]] = []
    curated: list[dict[str, Any]] = []
    for record in records:
        reason = excluded_paths.get(record["local_path"])
        reason = excluded_hashes.get(record["file_sha256"], reason)
        reason = prior_reasons.get(record["local_path"], reason)
        if record["file_validation_status"] == "invalid_pdf_signature":
            reason = "invalid_pdf_signature"
        elif record["file_validation_status"] == "nonstandard_pdf_preamble":
            reason = "nonstandard_pdf_preamble"
        if reason:
            if reason not in {"invalid_pdf_signature", "nonstandard_pdf_preamble"}:
                record["quality_flags"].append("content_exclusion")
            quarantine.append(
                {
                    "document_id": record["document_id"],
                    "local_path": record["local_path"],
                    "reason": reason,
                    "action": "exclude_from_curated_retrieval_keep_raw",
                }
            )
            continue
        curated.append({**record, "curation_status": "accepted_for_content_processing"})

    canonical_by_hash: dict[str, dict[str, Any]] = {}
    for record in curated:
        digest = record["file_sha256"]
        current = canonical_by_hash.get(digest)
        if current is None or _canonical_rank(record) > _canonical_rank(current):
            canonical_by_hash[digest] = record
    canonical = [
        {**record, "canonical_status": "canonical_content_record"}
        for record in sorted(canonical_by_hash.values(), key=lambda item: item["local_path"])
    ]

    _write_jsonl(corpus_path, records)
    _write_jsonl(curated_path, curated)
    _write_jsonl(quarantine_path, quarantine)
    _write_jsonl(canonical_path, canonical)

    by_source = Counter(record["source_id"] for record in records)
    curated_by_source = Counter(record["source_id"] for record in curated)
    source_names = {source.source_id: source.name for source in catalog.sources}
    jurisdictions = {source.source_id: source.jurisdiction for source in catalog.sources}
    local_catalog_path = datasets_dir / "registry" / "local_collections.json"
    if local_catalog_path.exists():
        local_catalog = json.loads(local_catalog_path.read_text(encoding="utf-8"))
        for collection in local_catalog.get("collections", []):
            source_names[collection["collection_id"]] = collection["display_name"]
            jurisdictions[collection["collection_id"]] = collection["jurisdiction"]
    all_source_ids = sorted(set(by_source) | set(source_names))
    source_results = [
        {
            "source_id": source_id,
            "name": source_names.get(source_id, source_id),
            "jurisdiction": jurisdictions.get(
                source_id,
                next(
                    (record["jurisdiction"] for record in records if record["source_id"] == source_id),
                    "unknown",
                ),
            ),
            "physical_pdf_files": by_source[source_id],
            "accepted_records": curated_by_source[source_id],
            "unique_manifest_document_urls": len(discovered_urls.get(source_id, set())),
        }
        for source_id in all_source_ids
    ]

    generated_at = datetime.now(timezone.utc).isoformat()
    integrity = {
        "generated_at_utc": generated_at,
        "raw_pdf_files": len(records),
        "curated_records": len(curated),
        "canonical_records": len(canonical),
        "quarantined_records": len(quarantine),
        "unique_content_hashes": len(hash_counts),
        "duplicate_file_copies": len(records) - len(hash_counts),
        "total_bytes": sum(record["file_size_bytes"] for record in records),
        "by_source": dict(sorted(by_source.items())),
        "by_jurisdiction": dict(sorted(Counter(record["jurisdiction"] for record in records).items())),
        "by_document_type": dict(sorted(Counter(record["document_type"] for record in records).items())),
        "by_file_validation_status": dict(
            sorted(Counter(record["file_validation_status"] for record in records).items())
        ),
        "quarantine_reasons": dict(
            sorted(Counter(record["reason"] for record in quarantine).items())
        ),
        "scope_note": "Raw files are preserved. Canonical records deduplicate exact content; OCR and legal currentness remain unverified.",
    }
    coverage = {
        "generated_at_utc": generated_at,
        "raw_pdf_files": len(records),
        "curated_records": len(curated),
        "canonical_records": len(canonical),
        "quarantined_records": len(quarantine),
        "configured_official_sources": len(catalog.sources),
        "sources_with_physical_files": sum(1 for result in source_results if result["physical_pdf_files"]),
        "source_results": source_results,
        "known_coverage_gaps": [
            "Supreme Court corpus blocked by the official site's HTTP 403 policy",
            "complete Sindh and Islamabad High Court judgment corpora",
            "AJK Legislative Assembly index blocked by HTTP 508 redirect loops",
            "complete federal and provincial Gazette archives",
            "complete tribunal, regulator, and special-court decision corpora",
            "law-by-law amendment, commencement, repeal, and applicability certification",
            "full-text OCR and bilingual Urdu-English normalization",
        ],
        "quality_note": "Accepted and canonical records are structurally usable, not lawyer-certified as complete or currently applicable.",
    }
    (manifest_dir / "integrity_summary.json").write_text(
        json.dumps(integrity, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (manifest_dir / "coverage_report.json").write_text(
        json.dumps(coverage, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return integrity
