# Qanoon AI Backend

This folder owns the Python backend for Qanoon AI.

## Layout

```text
api.py              Uvicorn compatibility entrypoint
qanoon_ai/          Application package
pipeline/           Legacy extraction/chunking/embedding pipeline
training/           Model fine-tuning scripts
tests/              Backend tests
evals/              Evaluation fixtures and runners
requirements.txt    Backend Python dependencies
```

The backend reads shared legal artifacts from the repository root:

```text
../pdfs
../data
../datasets
../data/index
```

The canonical extraction and legal chunking primitives live in
`qanoon_ai/ingestion/legal_index.py`. The production builder in
`qanoon_ai/ingestion/postgres_index.py` publishes PostgreSQL full-text and
pgvector indexes. The API uses dense plus lexical retrieval, reciprocal-rank
fusion, and cross-encoder reranking through `qanoon_ai/retrieval/hybrid.py`.
SQLite remains a local fallback.

## Commands

Run the API:

```bash
uvicorn api:app --reload --port 8000
```

Generate the source registry:

```bash
python -m qanoon_ai.cli inventory --out ../data/registry/sources.jsonl
```

List official dataset sources and download a limited batch:

```powershell
python -m qanoon_ai.cli download-sources --list
python -m qanoon_ai.cli download-sources --source sindh_high_court_laws --limit 10
```

Run tests on PowerShell:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest -q
```

To also run the real PostgreSQL regression tests, set
`QANOON_TEST_DATABASE_URL` to a disposable PostgreSQL/pgvector database before
running pytest. These tests create and remove their own unique schemas and use
small deterministic embeddings; they do not download models or read the corpus.

Build a controlled batch and inspect it:

```powershell
python -m qanoon_ai.cli build-index --source federal_pakistan_code --limit 25
python -m qanoon_ai.cli index-status
```

Build the full canonical corpus:

```powershell
python -m qanoon_ai.cli build-index --prune
```

The direct production equivalents, when PostgreSQL is already available, are:

```powershell
python -m qanoon_ai.cli build-system
python -m qanoon_ai.cli system-status
```

For the normal Docker workflow, run `.\qanoon.ps1 build` from the repository
root instead.

Generated indexes live under `../data/index/` and are intentionally ignored by
Git. If no persistent or legacy index exists, `/query` refuses to guess.
