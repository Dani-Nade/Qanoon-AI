"""Qanoon AI command line utilities."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from qanoon_ai.core.config import settings
from qanoon_ai.ingestion.corpus_audit import audit_corpus
from qanoon_ai.ingestion.downloader import download_sources
from qanoon_ai.ingestion.inventory import scan_pdfs, write_jsonl
from qanoon_ai.ingestion.legal_index import build_legal_index, index_status
from qanoon_ai.ingestion.postgres_index import (
    build_production_index,
    production_index_status,
)
from qanoon_ai.ingestion.source_catalog import load_source_catalog


def _inventory(args: argparse.Namespace) -> None:
    pdfs_dir = Path(args.pdfs_dir) if args.pdfs_dir else settings.pdfs_dir
    out_file = Path(args.out) if args.out else settings.source_registry_file
    records = scan_pdfs(pdfs_dir, hash_files=args.hash_files)
    write_jsonl(records, out_file)
    print(f"Wrote {len(records)} source records to {out_file}")


def _download_sources(args: argparse.Namespace) -> None:
    catalog_path = Path(args.catalog) if args.catalog else settings.official_source_catalog_file
    catalog = load_source_catalog(catalog_path)
    jurisdictions = set(args.jurisdiction or [])
    source_ids = set(args.source or [])
    selected_sources = catalog.select(
        jurisdictions=jurisdictions or None,
        source_ids=source_ids or None,
    )

    if args.list:
        for source in catalog.sources:
            print(f"{source.source_id}\t{source.jurisdiction}\t{source.name}")
        return

    if not selected_sources:
        raise SystemExit("No official sources matched the requested filters.")

    limit = None if args.limit == 0 else args.limit
    summaries = download_sources(
        selected_sources,
        raw_dir=Path(args.out_dir) if args.out_dir else settings.datasets_raw_dir,
        manifest_file=Path(args.manifest) if args.manifest else settings.download_manifest_file,
        limit=limit,
        dry_run=args.dry_run,
        force=args.force,
        max_depth=args.max_depth,
        max_pages=args.max_pages,
        timeout=args.timeout,
        sleep_seconds=args.sleep,
        workers=args.workers,
    )

    for summary in summaries:
        print(
            f"{summary.source_id}: discovered={summary.discovered} "
            f"downloaded={summary.downloaded} skipped={summary.skipped_existing} "
            f"dry_run={summary.dry_run} errors={summary.errors}"
        )


def _audit_datasets(_args: argparse.Namespace) -> None:
    catalog = load_source_catalog(settings.official_source_catalog_file)
    summary = audit_corpus(
        datasets_dir=settings.datasets_dir,
        catalog=catalog,
        download_manifest=settings.download_manifest_file,
    )
    print(
        f"Audited {summary['raw_pdf_files']} raw PDFs: "
        f"curated={summary['curated_records']} "
        f"canonical={summary['canonical_records']} "
        f"quarantined={summary['quarantined_records']}"
    )


def _build_index(args: argparse.Namespace) -> None:
    limit = None if args.limit == 0 else args.limit
    summary = build_legal_index(
        datasets_dir=settings.datasets_dir,
        manifest_file=Path(args.manifest) if args.manifest else settings.canonical_manifest_file,
        index_file=Path(args.index_file) if args.index_file else settings.search_index_file,
        source_ids=set(args.source or []) or None,
        title_contains=args.title_contains,
        limit=limit,
        force=args.force,
        prune=args.prune,
        workers=args.workers,
    )
    print(json.dumps(asdict(summary), indent=2))


def _index_status(args: argparse.Namespace) -> None:
    index_file = Path(args.index_file) if args.index_file else settings.search_index_file
    print(json.dumps(index_status(index_file), indent=2))


def _build_system(args: argparse.Namespace) -> None:
    limit = None if args.limit == 0 else args.limit
    filtered = bool(args.source or limit is not None)
    summary = build_production_index(
        database_url=settings.database_url,
        datasets_dir=settings.datasets_dir,
        manifest_file=settings.canonical_manifest_file,
        report_file=settings.build_report_file,
        embedding_model=settings.embedding_model,
        embedding_revision=settings.embedding_revision,
        embedding_dimension=settings.embedding_dimension,
        embedding_batch_size=settings.indexing_batch_size,
        extraction_workers=settings.indexing_workers,
        enable_ocr=settings.enable_ocr and not args.no_ocr,
        source_ids=set(args.source or []) or None,
        limit=limit,
        force=args.force,
        prune=not filtered and not args.no_prune,
    )
    print(json.dumps(asdict(summary), indent=2))
    if any(
        (
            summary.failed_documents,
            summary.duplicate_hashes,
            summary.missing_manifest_files,
            summary.missing_embeddings,
            summary.orphan_chunks,
            summary.indexed_documents_without_chunks,
            summary.document_count_mismatch,
        )
    ):
        raise SystemExit(2)


def _build_vectors(args: argparse.Namespace) -> None:
    from qanoon_ai.retrieval.vectors import build_vectors, resolve_model_path

    summary = build_vectors(
        index_file=settings.search_index_file,
        vector_dir=settings.vector_dir,
        model_path=resolve_model_path(settings.model_cache_dir, settings.embedding_model, settings.embedding_revision),
        model_id=settings.embedding_model,
        batch_size=args.batch_size,
        limit=None if args.limit == 0 else args.limit,
    )
    print(json.dumps(summary, indent=2))


def _system_status(_args: argparse.Namespace) -> None:
    print(json.dumps(production_index_status(
        settings.database_url, embedding_model=settings.embedding_model,
        embedding_revision=settings.embedding_revision, embedding_dimension=settings.embedding_dimension,
    ), indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qanoon", description="Qanoon AI utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory = subparsers.add_parser("inventory", help="Generate PDF source registry")
    inventory.add_argument("--pdfs-dir", default=None, help="PDF directory to scan")
    inventory.add_argument("--out", default=None, help="Output JSONL file")
    inventory.add_argument(
        "--hash-files",
        action="store_true",
        help="Calculate SHA-256 hashes for every PDF",
    )
    inventory.set_defaults(func=_inventory)

    download = subparsers.add_parser(
        "download-sources",
        help="Download official legal datasets into datasets/raw with a manifest",
    )
    download.add_argument("--catalog", default=None, help="Official source catalog JSON")
    download.add_argument("--jurisdiction", action="append", help="Filter by jurisdiction")
    download.add_argument("--source", action="append", help="Filter by source_id")
    download.add_argument("--out-dir", default=None, help="Raw dataset output directory")
    download.add_argument("--manifest", default=None, help="JSONL manifest output file")
    download.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Maximum documents per source; use 0 for no limit",
    )
    download.add_argument("--max-depth", type=int, default=1, help="HTML crawl depth")
    download.add_argument("--max-pages", type=int, default=50, help="Max HTML pages per source")
    download.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout seconds")
    download.add_argument("--sleep", type=float, default=0.3, help="Delay between downloads")
    download.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent document downloads per source",
    )
    download.add_argument("--dry-run", action="store_true", help="Discover and manifest only")
    download.add_argument("--force", action="store_true", help="Overwrite existing files")
    download.add_argument("--list", action="store_true", help="List catalog sources and exit")
    download.set_defaults(func=_download_sources)

    audit = subparsers.add_parser(
        "audit-datasets",
        help="Rebuild corpus, curation, quarantine, canonical, and coverage manifests",
    )
    audit.set_defaults(func=_audit_datasets)

    build_index = subparsers.add_parser(
        "build-index",
        help="Incrementally extract and index canonical PDFs into SQLite FTS5",
    )
    build_index.add_argument("--manifest", default=None, help="Canonical JSONL manifest")
    build_index.add_argument("--index-file", default=None, help="SQLite index output path")
    build_index.add_argument("--source", action="append", help="Filter by source_id")
    build_index.add_argument("--title-contains", default=None, help="Filter by title text")
    build_index.add_argument(
        "--limit", type=int, default=0, help="Maximum selected documents; 0 means unlimited"
    )
    build_index.add_argument("--force", action="store_true", help="Re-extract unchanged files")
    build_index.add_argument("--workers", type=int, default=1, help="Parallel PDF extraction processes")
    build_index.add_argument(
        "--prune",
        action="store_true",
        help="Remove stale records; only valid for an unfiltered full build",
    )
    build_index.set_defaults(func=_build_index)

    status = subparsers.add_parser("index-status", help="Show legal search index status")
    status.add_argument("--index-file", default=None, help="SQLite index path")
    status.set_defaults(func=_index_status)

    system_build = subparsers.add_parser(
        "build-system",
        help="Validate and incrementally build the production PostgreSQL hybrid index",
    )
    system_build.add_argument("--source", action="append", help="Optional source_id filter")
    system_build.add_argument(
        "--limit", type=int, default=0, help="Maximum selected documents; 0 means full corpus"
    )
    system_build.add_argument("--force", action="store_true", help="Reprocess unchanged PDFs")
    system_build.add_argument("--no-ocr", action="store_true", help="Disable OCR fallback")
    system_build.add_argument(
        "--no-prune", action="store_true", help="Keep database documents absent from the manifest"
    )
    system_build.set_defaults(func=_build_system)

    vectors = subparsers.add_parser(
        "build-vectors",
        help="Embed SQLite index chunks with the local embedding model for meaning-based search (incremental)",
    )
    vectors.add_argument("--batch-size", type=int, default=64, help="Chunks per embedding batch")
    vectors.add_argument("--limit", type=int, default=0, help="Maximum new chunks to embed; 0 means all")
    vectors.set_defaults(func=_build_vectors)

    system_status = subparsers.add_parser(
        "system-status", help="Show the production PostgreSQL index status"
    )
    system_status.set_defaults(func=_system_status)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
