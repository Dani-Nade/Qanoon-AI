# Qanoon AI Enterprise-Grade Plan

## Goal

Build Qanoon AI as a source-grounded Pakistani legal assistant for text and voice, supporting English, Urdu, and Roman Urdu queries while using official legal sources as the truth layer.

Important principle:

```text
Legal source database = truth
Fine-tuned LLM = reasoning, explanation, language, formatting, and citation discipline
Voice assistant = input/output layer
```

The target is not a model that memorizes all law. The target is a system that retrieves current legal sources, reasons over them, cites them, and refuses to guess when sources are missing.

## Current Repo State

The project now has a working source-grounded retrieval foundation, but the
complete advanced RAG and model-serving layers are not finished yet.

Implemented foundation as of 2026-08-09:

- Canonical dataset manifest with 18,260 unique PDFs.
- Incremental, pipeline-versioned PDF extraction from the canonical manifest.
- Page-aware and legal-section-aware chunks with stable identifiers.
- Production PostgreSQL/pgvector builder with weighted full-text search,
  trigram matching, HNSW vector search, and build-run audit records.
- Persistent SQLite FTS5 retrieval as the local fallback.
- Query normalization for English, Urdu, and Roman Urdu legal terms.
- Dense and lexical candidate retrieval, reciprocal-rank fusion,
  cross-encoder reranking, term-coverage gating, and safe refusal.
- Citations carrying source, jurisdiction, document type, section, page, and legal status.
- API/UI integration and focused automated tests.
- A verified local SQLite starter index; the complete 18,260-document
  PostgreSQL index still needs to be executed using `.\qanoon.ps1 build`.

Remaining major work:

- OCR workers for image-only documents.
- Run and evaluate the complete production hybrid index on all 18,260 documents.
- Currentness/amendment verification and lawyer review workflows.
- A production LLM adapter and citation-grounded answer generation.
- Evaluation gates, authentication, audit logging, observability, and voice.
- Replacement of the legacy pipeline and demo fine-tuning scripts after parity checks.

## Target System Architecture

```text
Text or voice question
  -> Speech-to-text, if voice
  -> Language detection: English, Urdu, Roman Urdu
  -> Legal intent detection
  -> Jurisdiction detection: federal, province, court, department
  -> Query rewrite into legal search queries
  -> Hybrid retrieval over official sources
  -> Reranking and source filtering
  -> Evidence pack construction
  -> Fine-tuned legal LLM answer generation
  -> Citation verification
  -> Safety and uncertainty check
  -> Final answer in user's language
  -> Text-to-speech, if voice response requested
```

## Core Components

### 1. Source Registry

Create a source-of-truth registry for every legal document.

Recommended table or JSONL fields:

```json
{
  "source_id": "pk_federal_act_1860_ppc_current",
  "title": "Pakistan Penal Code, 1860",
  "jurisdiction": "federal",
  "province": null,
  "court": null,
  "document_type": "act",
  "source_url": "official URL",
  "local_path": "pdfs/...",
  "language": "english",
  "enacted_date": "1860-10-06",
  "amended_upto": "2026-08-08",
  "is_repealed": false,
  "repealed_by": null,
  "official_status": "official",
  "file_sha256": "hash",
  "ingested_at": "timestamp",
  "version": "2026-08-08"
}
```

Why this matters:

- Prevents duplicate or outdated law from being treated as current.
- Allows "current law only" filtering.
- Allows province/court-specific answers.
- Makes citations auditable.

### 2. Ingestion Pipeline

Replace the basic pipeline with a production ingestion workflow.

Pipeline:

```text
Discover source
  -> Download or import
  -> Hash file
  -> OCR/extract text
  -> Clean text
  -> Detect document structure
  -> Split by sections/articles/rules/orders
  -> Deduplicate
  -> Classify jurisdiction and document type
  -> Store source metadata
  -> Store source text
  -> Generate search chunks
  -> Embed chunks
  -> Publish index version
```

Required improvements:

- Use file hashes to avoid reprocessing unchanged files.
- Split by legal structure, not only word count.
- Preserve section numbers, rules, schedules, provisos, explanations, footnotes, and amendment notes.
- Detect repealed and amended documents.
- Keep raw PDF, extracted text, cleaned text, chunk text, and metadata separately.

Recommended storage:

- Raw files: `object storage` such as S3 or MinIO.
- Metadata and app data: PostgreSQL.
- Search index: Qdrant or PostgreSQL with pgvector plus full-text search.

### 3. Advanced Legal RAG

Use advanced retrieval, not simple vector search.

Retrieval stages:

```text
User query
  -> language normalization
  -> legal query rewrite
  -> dense vector retrieval
  -> sparse vector retrieval
  -> BM25/full-text retrieval
  -> metadata filters
  -> cross-encoder reranking
  -> neighboring section expansion
  -> evidence pack
```

Filters:

- Jurisdiction: federal, Punjab, Sindh, KP, Balochistan, ICT.
- Court: Supreme Court, High Court, district courts, special courts.
- Document type: Act, Ordinance, Rules, Regulations, Order, Judgment.
- Status: current, amended, repealed, historical.
- Date range.

Recommended retrieval stack:

- Option A: Qdrant for dense plus sparse hybrid retrieval.
- Option B: PostgreSQL with pgvector plus full-text search.
- Optional: OpenSearch or Elasticsearch for large-scale BM25 and filters.

Reranking:

- Use a cross-encoder reranker for top 50-100 retrieved chunks.
- Rerank based on legal relevance, exact section match, jurisdiction match, and current-law status.

Citation behavior:

- Every answer should cite source title, section/rule/article, page if available, and source version.
- If no source supports the answer, the app should say it does not have enough source material.

### 4. Fine-Tuned Qanoon LLM

Keep a fine-tuned model, but use it for legal behavior, not as the only legal memory.

Fine-tuning goals:

- Answer in a lawyer-like but user-friendly style.
- Use citations correctly.
- Explain English law sources in Urdu and Roman Urdu.
- Refuse unsupported answers.
- Distinguish federal, provincial, and court-specific law.
- Warn about repealed/amended law.
- Ask clarifying questions when jurisdiction is missing.

Recommended training stages:

```text
Base instruct model
  -> SFT on high-quality source-grounded Q&A
  -> DPO/preference tuning on good vs bad legal answers
  -> Safety/refusal tuning
  -> Multilingual style tuning
  -> Evaluation before release
```

Training data format:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are Qanoon AI, a Pakistani legal assistant. Answer only from provided sources and cite them."
    },
    {
      "role": "user",
      "content": "Pakistan mein theft ki saza kya hai?"
    },
    {
      "role": "assistant",
      "content": "Pakistan Penal Code ke mutabiq theft se related saza section ... mein di gayi hai. [Pakistan Penal Code, 1860, Section ...]"
    }
  ],
  "sources": [
    {
      "source_id": "pk_federal_ppc_1860",
      "section": "...",
      "text": "..."
    }
  ],
  "language": "roman_urdu",
  "jurisdiction": "federal"
}
```

Training script requirements:

- Use TRL `SFTTrainer`.
- Use PEFT/QLoRA.
- Use assistant-only loss masking.
- Use a longer context window, ideally 2048 to 8192 tokens depending on model and GPU.
- Split train/eval by document or legal area, not random chunks.
- Track dataset version and model version.
- Save model cards and eval reports.

Recommended serving:

- For local/self-hosted models: vLLM with an OpenAI-compatible API.
- For highest quality: use a hosted frontier model behind the same internal LLM interface.

### 5. Multilingual Layer

Support:

- English
- Urdu
- Roman Urdu

Since most legal sources are English, use this flow:

```text
Urdu/Roman Urdu query
  -> detect language
  -> translate/rewrite into English legal search queries
  -> retrieve English legal sources
  -> generate answer in original user language
  -> cite English source
```

Examples:

```text
Question: What is punishment for theft in Pakistan?
Answer language: English
```

```text
Question: Pakistan mein chori ki saza kya hai?
Answer language: Roman Urdu or Urdu, based on app setting
```

```text
Question: پاکستان میں چوری کی سزا کیا ہے؟
Answer language: Urdu
```

Important:

- Retrieval should happen against normalized English legal terms.
- Final answer should follow user language preference.
- Citations can remain in English source titles.

### 6. Voice Assistant

Voice should sit on top of the legal engine.

Voice flow:

```text
User speech
  -> speech-to-text
  -> Qanoon legal engine
  -> answer text
  -> text-to-speech
  -> spoken response
```

Voice features:

- English, Urdu, and Roman Urdu speech support.
- Streaming partial response.
- Interruptible voice conversation.
- "Show source" or "read source" voice command.
- Conversation memory scoped to the current matter.
- Clear disclaimer in first response or settings.

Do not add voice before the legal engine is reliable. Otherwise the app will only speak weak answers confidently.

### 7. Citation Verifier

Add a second verification step before final answer.

Verifier checks:

- Does every legal claim have support in retrieved source text?
- Are cited sections actually present?
- Is the law current or repealed?
- Is the jurisdiction correct?
- Did the model introduce unsupported punishment, deadline, court name, or procedure?

If verification fails:

- Remove unsupported claim.
- Ask for clarification.
- Or return an uncertainty message.

### 8. Evaluation System

Create an eval suite before trusting the app.

Core evals:

- Retrieval recall: did the correct law section appear in top results?
- Citation accuracy: do citations match source text?
- Faithfulness: are answer claims supported by retrieved text?
- Jurisdiction accuracy: did it choose correct federal/province/court?
- Language accuracy: did it answer in requested language?
- Repealed-law handling: did it warn correctly?
- Refusal quality: did it refuse when sources were missing?

Datasets:

- English legal Q&A.
- Urdu legal Q&A.
- Roman Urdu legal Q&A.
- Court rules questions.
- Provincial law questions.
- Negative tests where answer is not in database.
- Conflicting-law tests.
- Repealed-law tests.

Release rule:

```text
No model or index version ships unless it passes the eval threshold.
```

### 9. Product Features

Premium user-facing features:

- Chat with citations.
- Same-language answer.
- Voice assistant.
- Ask follow-up questions.
- "Current law only" toggle.
- Province/court selector.
- Source viewer with highlighted legal text.
- Drafting assistant for notices, applications, affidavits, petitions, replies.
- Uploaded document analysis.
- Legal deadline calculator.
- Matter folders.
- Saved chats and citations.
- Export answer as PDF or Word.
- Lawyer escalation button.

Admin/lawyer features:

- Source upload dashboard.
- Source metadata editor.
- Duplicate detection dashboard.
- Repealed/amended status review.
- Eval dashboard.
- Bad-answer review queue.
- Human feedback labeling.
- Model/index version history.

### 10. Security and Enterprise Controls

Required:

- Authentication.
- Role-based access control.
- Tenant isolation for law firms and businesses.
- Encryption at rest and in transit.
- Audit logs for every answer, source, model version, and retrieval result.
- Rate limits.
- Abuse detection.
- Data retention policy.
- PII handling and redaction for uploaded documents.
- Admin approval for production source changes.

Enterprise deployment options:

- SaaS cloud.
- Private cloud.
- On-prem deployment for sensitive clients.

### 11. Observability

Track every request end to end:

- User language.
- Detected legal area.
- Detected jurisdiction.
- Retrieval query variants.
- Retrieved chunks.
- Reranked chunks.
- Final citations.
- Model used.
- Index version used.
- Latency.
- Token usage.
- Verifier result.
- User feedback.

Recommended stack:

- OpenTelemetry for traces and logs.
- Sentry for application errors.
- Prometheus/Grafana for metrics.
- Structured logs for legal answer audits.

### 12. Data Update Workflow

Adding new laws or rules should usually re-index data, not retrain the model.

Recommended command flow:

```bash
qanoon ingest --source ./new_pdfs --jurisdiction federal --type act
qanoon validate --dataset latest
qanoon index --changed-only
qanoon eval --suite legal-core
qanoon publish-index --version 2026-08-08
```

Retrain only when behavior needs improvement:

```bash
qanoon train-sft --dataset qanoon_sft_v12
qanoon train-dpo --preferences qanoon_pref_v5
qanoon eval-model --model qanoon-legal-v3
qanoon publish-model --model qanoon-legal-v3
```

Decision rule:

```text
New source data -> ingest and re-index.
Bad answer style or reasoning -> fine-tune.
New language behavior -> fine-tune.
New domain behavior -> curate examples and fine-tune.
```

## Recommended Tech Stack

Backend:

- FastAPI
- Pydantic
- SQLAlchemy or SQLModel
- Celery, Dramatiq, Temporal, Prefect, or Dagster for background jobs

Frontend:

- Next.js
- TypeScript
- Tailwind or shadcn/ui

Storage:

- PostgreSQL for app data and metadata
- S3 or MinIO for raw PDFs and extracted artifacts
- Qdrant for hybrid vector search, or PostgreSQL with pgvector plus full-text search

Model training:

- Hugging Face Transformers
- TRL
- PEFT/QLoRA
- Accelerate
- Weights and Biases or MLflow for experiment tracking

Model serving:

- vLLM for self-hosted open models
- Hosted LLM provider behind an internal model gateway if quality is more important than local hosting

Evaluation:

- Ragas
- Custom legal evals
- Human lawyer review set

Voice:

- Speech-to-text provider
- Text-to-speech provider
- Optional realtime voice API for low-latency assistant behavior

Observability:

- OpenTelemetry
- Sentry
- Prometheus/Grafana

## Implementation Phases

### Phase 1: Stabilize Current Repo

- Fix dependencies.
- Add README and setup instructions.
- Add source registry schema.
- Add scripts for file hashing and PDF inventory.
- Rebuild extraction and chunking.
- Create an initial searchable index.
- Replace direct LLM answers with source-grounded answers.

### Phase 2: Build Advanced RAG

- Add hybrid search.
- Add metadata filters.
- Add legal query rewrite.
- Add reranking.
- Add section-neighbor expansion.
- Add citation formatting.
- Add citation verifier.
- Add "not enough source material" behavior.

### Phase 3: Multilingual Support

- Add language detection.
- Add Urdu/Roman Urdu query normalization.
- Add answer-language control.
- Add multilingual eval questions.
- Add multilingual prompt templates.

### Phase 4: Training Upgrade

- Replace current `training/train.py`.
- Build curated source-grounded SFT dataset.
- Add assistant-only loss.
- Add document-level eval split.
- Train LoRA/QLoRA adapter.
- Add DPO/preference tuning later.
- Add model eval report.

### Phase 5: Enterprise Product Features

- Add user auth.
- Add matter folders.
- Add source viewer.
- Add drafting assistant.
- Add admin source dashboard.
- Add feedback and bad-answer review queue.
- Add audit logs.

### Phase 6: Voice Assistant

- Add speech-to-text.
- Add text-to-speech.
- Add streaming conversation.
- Add voice interruption.
- Add "show/read citation" voice commands.

### Phase 7: Production Hardening

- Add CI checks.
- Add API tests.
- Add eval gates.
- Add monitoring.
- Add backup and restore.
- Add security review.
- Add deployment pipeline.

## Immediate Changes Needed In This Repo

1. Replace `requirements.txt` with complete dependencies.
2. Add `docs/` with architecture, setup, and data policy.
3. Add `source_registry` data model.
4. Add PDF inventory and hashing script.
5. Replace chunking with legal section-aware chunking.
6. Add RAG service module.
7. Replace `api.py` with a RAG-backed API.
8. Add language detection and query normalization.
9. Add citation verifier.
10. Replace training script with modern TRL/PEFT pipeline.
11. Add eval dataset and eval runner.
12. Add tests for API, retrieval, citations, and language behavior.

## Reference Technologies

- Qdrant hybrid search: https://qdrant.tech/documentation/search/hybrid-queries/
- pgvector: https://github.com/pgvector/pgvector
- vLLM serving: https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/
- TRL SFTTrainer: https://huggingface.co/docs/trl/en/sft_trainer
- TRL DPOTrainer: https://huggingface.co/docs/trl/en/dpo_trainer
- PEFT LoRA: https://huggingface.co/docs/peft/en/package_reference/lora
- LangGraph: https://docs.langchain.com/oss/python/langgraph/overview
- Ragas metrics: https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/
- OpenTelemetry: https://opentelemetry.io/docs/
