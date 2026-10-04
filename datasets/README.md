# Qanoon AI Datasets

This folder is the root-level legal data acquisition area. Keep it outside
`backend/` because the dataset is a shared product asset, not application
source code.

## Layout

```text
datasets/
  registry/    Official source catalog and source-specific acquisition metadata
  raw/         Downloaded official files, grouped by jurisdiction/source
  manifests/   JSONL provenance logs for every discovery/download attempt
```

Retrieval and training should consume `manifests/canonical_manifest.jsonl`. The
other manifests serve separate audit purposes:

- `corpus_manifest.jsonl`: every raw PDF, including duplicates and quarantined files
- `curated_manifest.jsonl`: structurally accepted legal documents
- `canonical_manifest.jsonl`: one preferred record per exact content hash
- `quarantine_manifest.jsonl`: excluded records retained in raw storage

The original root `pdfs/` collection is preserved and mirrored as:

```text
datasets/raw/federal/pakistan_code_archive/by_enactment_year/<YYYY>/
```

This is a federal Pakistan Code archive, not a claim of complete Pakistan-wide
coverage. Its files retain `legacy_source_unverified` provenance until they are
matched to an official source URL and checked for current legal status. See
`registry/local_collections.json` for the machine-readable collection record.

The existing `pdfs/` folder remains untouched for application compatibility and
recovery. New ingestion should read the dataset archive and its manifests rather
than treating the root folder as an independently verified source. Do not delete
or reclassify unverified documents until the manifest proves source,
jurisdiction, hash, and document type.

## Download Commands

List configured official sources:

```powershell
cd backend
python -m qanoon_ai.cli download-sources --list
```

Dry-run discovery for one source:

```powershell
cd backend
python -m qanoon_ai.cli download-sources --source sindh_high_court_laws --dry-run --limit 10
```

Download a safe batch:

```powershell
cd backend
python -m qanoon_ai.cli download-sources --source sindh_high_court_laws --limit 10
```

For full ingestion, run source-by-source with rate limits and review the manifest
after each run. Use `--limit 0` only when you are ready for a complete source
download.

Rebuild all corpus, curation, quarantine, canonical, integrity, and coverage
manifests after an acquisition run:

```powershell
cd backend
python -m qanoon_ai.cli audit-datasets
```

See `../docs/DATASET_ACQUISITION.md` for the source coverage plan and adapter
notes.
