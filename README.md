# Qanoon AI

Qanoon AI is a compact monorepo for a source-grounded Pakistani legal assistant.

## Structure

```text
backend/   Python API, ingestion, retrieval, model adapters, tests
frontend/  Next.js user app
data/      Generated search indexes, registries, and training data
datasets/  Official dataset acquisition, raw downloads, manifests
pdfs/      Preserved compatibility copy of the original PDF collection
docs/      Architecture and implementation plans
```

The original Pakistan Code collection is canonically organized for ingestion at
`datasets/raw/federal/pakistan_code_archive/by_enactment_year/<YYYY>/`.

The fine-tuned LLM remains part of the target system, but legal sources are the truth layer.

```text
question -> language normalization -> legal retrieval -> relevance gate -> citations -> answer
```

The production retrieval layer uses PostgreSQL full-text search, `pg_trgm`,
`pgvector`, BGE-M3 multilingual embeddings, reciprocal-rank fusion, and a
multilingual cross-encoder reranker. SQLite FTS5 remains a development fallback.

## One-Line Build

From the repository root:

```powershell
.\qanoon.ps1 build
```

That command starts PostgreSQL, builds the backend image, validates
`datasets/manifests/canonical_manifest.jsonl`, incrementally extracts/OCRs
changed PDFs, creates legal-structure chunks, generates embeddings, publishes
the hybrid index, verifies counts and duplicates, and writes
`data/reports/system-build-latest.json`.

`datasets/` is the only legal input. Existing files under `data/training/` and
`data/registry/` are not consumed by the production builder.

## Backend

```bash
cd backend
uvicorn api:app --reload --port 8000
```

Generate a local PDF source registry:

```bash
cd backend
python -m qanoon_ai.cli inventory --out ../data/registry/sources.jsonl
```

Build or incrementally update the legal search index from the canonical dataset:

```powershell
cd backend
python -m qanoon_ai.cli build-index --limit 25
python -m qanoon_ai.cli index-status
```

The commands above are the lightweight SQLite development path. The normal
production path is `qanoon.ps1 build`.

Run an unfiltered corpus build and prune records no longer present in the
canonical manifest:

```powershell
python -m qanoon_ai.cli build-index --prune
```

Filtered builds never prune existing indexed documents. Unchanged PDF hashes
and unchanged pipeline versions are skipped automatically.

## Runner Commands

```powershell
.\qanoon.ps1 setup
.\qanoon.ps1 build
.\qanoon.ps1 build --limit 25
.\qanoon.ps1 build --source pakistan_code_archive
.\qanoon.ps1 status
.\qanoon.ps1 api
.\qanoon.ps1 logs
.\qanoon.ps1 stop
```

Copy `infra/.env.example` to `infra/.env` before a shared or production
deployment and replace the development passwords.

List and download official dataset sources:

```powershell
cd backend
python -m qanoon_ai.cli download-sources --list
python -m qanoon_ai.cli download-sources --source sindh_high_court_laws --limit 10
```

Run tests:

```powershell
cd backend
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest -q
```

## Frontend

```bash
cd frontend
npm install
npm run dev
```

Set `NEXT_PUBLIC_API_BASE_URL` if the backend is not running on `http://localhost:8000`.

See `docs/ENTERPRISE_GRADE_PLAN.md` for the full architecture plan.
