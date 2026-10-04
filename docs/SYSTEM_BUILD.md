# Qanoon AI System Build

## Data Contract

The builder has one legal input:

```text
datasets/manifests/canonical_manifest.jsonl
```

Every manifest path must exist under `datasets/`, and document IDs, local
paths, and SHA-256 hashes must be unique. The builder does not scan or train
from `data/training/qa_pairs.jsonl`, `data/registry/sources.jsonl`, or `pdfs/`.

`data/` is generated output and cache space. PostgreSQL is the authoritative
search index.

## First Setup

Install Docker Desktop, then create the local environment file:

```powershell
Copy-Item infra\.env.example infra\.env
```

Change the passwords in `infra/.env`, then build the backend image and start
PostgreSQL:

```powershell
.\qanoon.ps1 setup
```

## Full Build

The complete incremental production build is one command:

```powershell
.\qanoon.ps1 build
```

It performs:

1. PostgreSQL/pgvector startup and health wait.
2. Canonical manifest validation.
3. Hash and pipeline-version change detection.
4. Bounded concurrent PDF extraction.
5. Page-level English/Urdu OCR fallback where text layers are missing.
6. Legal section/article/rule-aware chunking.
7. BGE-M3 normalized multilingual embeddings.
8. PostgreSQL weighted full-text and HNSW vector publication.
9. Stale-document pruning on unfiltered builds.
10. Duplicate, missing-file, failed-document, and database-count verification.
11. An audited build row plus `data/reports/system-build-latest.json`.

The command exits nonzero if any document fails, any canonical file is missing,
or duplicate hashes reach the production database.

## Controlled Build

Use a small batch while testing infrastructure:

```powershell
.\qanoon.ps1 build --limit 25
```

Build one source incrementally:

```powershell
.\qanoon.ps1 build --source pakistan_code_archive
```

Filtered builds never prune documents from other sources.

The index records the embedding model, revision, and dimension per document.
After upgrading an older index, run a full build once to populate this metadata
and rebuild its vectors. Changing a model or revision reprocesses affected
documents automatically; retrieval excludes incompatible vectors during a build.
When changing `QANOON_EMBEDDING_MODEL`, also set its matching
`QANOON_EMBEDDING_REVISION` (or leave the revision empty for the model default).
Changing vector dimensions requires a separate database, preserving the existing
index until the new one is ready.

Force reprocessing or disable OCR temporarily:

```powershell
.\qanoon.ps1 build --force
.\qanoon.ps1 build --limit 25 --no-ocr
```

## Operation

Chat requires Ollama running on the host, with the configured model installed:

```powershell
ollama pull qwen3.5:9b
```

Start Ollama before starting the API. Docker connects to it through
`http://host.docker.internal:11434`, not the API container's loopback address.
If Ollama listens only on loopback, configure its `OLLAMA_HOST` to listen on an
interface reachable from Docker (for example `0.0.0.0:11434`) and restrict access
to trusted hosts with your firewall. For a separate model server, set
`QANOON_OLLAMA_URL` and `QANOON_CHAT_MODEL` in `infra/.env`.
The `/health` response reports `ollama_reachable` and `chat_model_ready`.

```powershell
.\qanoon.ps1 status
.\qanoon.ps1 api
.\qanoon.ps1 logs
.\qanoon.ps1 stop
```

The API is available at `http://127.0.0.1:8000` after `qanoon.ps1 api`.

## Direct Backend Commands

These are useful only when Python dependencies and PostgreSQL are already
configured outside Docker:

```powershell
cd backend
python -m pip install -r requirements-dev.txt
python -m qanoon_ai.cli build-system
python -m qanoon_ai.cli system-status
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

## Retrieval Runtime

Each question uses:

```text
multilingual query
  -> normalized query variants
  -> BGE-M3 query embedding
  -> pgvector dense candidates
  -> PostgreSQL weighted lexical/trigram candidates
  -> reciprocal-rank fusion
  -> BGE multilingual cross-encoder reranking
  -> relevance threshold
  -> jurisdiction filter
  -> source and page citations
  -> fail-closed answer service
```

The fine-tuned LLM will be connected after this index passes retrieval and
citation evaluations. Voice remains a later input/output layer and will use the
same multilingual legal engine.
