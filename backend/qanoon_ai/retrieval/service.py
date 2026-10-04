"""Source-grounded answer service."""

from __future__ import annotations

import logging

from qanoon_ai.core.config import Settings, settings
from qanoon_ai.llm.client import DisabledLLMClient, LocalLLMClient
from qanoon_ai.llm.answer import LocalAnswerClient
from qanoon_ai.language.detection import (
    build_retrieval_queries,
    detect_language,
    not_enough_sources_message,
)
from qanoon_ai.retrieval.keyword import KeywordRetriever
from qanoon_ai.retrieval.hybrid import PostgresHybridRetriever
from qanoon_ai.retrieval.sqlite import SQLiteFTSRetriever
from qanoon_ai.retrieval.vectors import VectorSearcher
from qanoon_ai.retrieval.models import EvidenceChunk
from qanoon_ai.schemas.query import LanguageInfo, QueryRequest, QueryResponse
from qanoon_ai.schemas.source import Citation
from qanoon_ai.verification.citations import verify_citations

LOGGER = logging.getLogger(__name__)


def _excerpt(text: str) -> str:
    # Keep qualifications and cross-page continuations available to the model.
    return " ".join(text.split())


class LegalAnswerService:
    """Coordinates language detection, retrieval, and safe answer shaping."""

    def __init__(self, app_settings: Settings = settings) -> None:
        self.settings = app_settings
        self.postgres_retriever = PostgresHybridRetriever(
            database_url=app_settings.database_url,
            embedding_model=app_settings.embedding_model,
            reranker_model=app_settings.reranker_model,
            reranker_min_score=app_settings.reranker_min_score,
            embedding_batch_size=app_settings.indexing_batch_size,
            embedding_revision=app_settings.embedding_revision,
            embedding_dimension=app_settings.embedding_dimension,
        )
        vectors = VectorSearcher(
            app_settings.search_index_file, app_settings.vector_dir, app_settings.model_cache_dir,
            app_settings.embedding_model, app_settings.embedding_revision,
        ) if app_settings.vector_search else None
        self.sqlite_retriever = SQLiteFTSRetriever(app_settings.search_index_file, vectors)
        self.keyword_retriever = KeywordRetriever(app_settings.chunks_file)
        self.llm = LocalAnswerClient(app_settings.root_dir) if app_settings.llm_mode in {"local", "auto"} else (
            LocalLLMClient(
                app_settings.trained_model_dir,
                app_settings.root_dir / "data/cache/huggingface/hub",
            )
            if app_settings.llm_mode == "quotation"
            else DisabledLLMClient()
        )

    @property
    def retriever(self) -> PostgresHybridRetriever | SQLiteFTSRetriever | KeywordRetriever:
        if self.settings.retrieval_mode == "postgres":
            return self.postgres_retriever
        if self.settings.retrieval_mode == "auto" and self.postgres_retriever.ready:
            return self.postgres_retriever
        if self.sqlite_retriever.ready:
            return self.sqlite_retriever
        return self.keyword_retriever

    @property
    def retrieval_ready(self) -> bool:
        return self.retriever.ready

    @property
    def model_ready(self) -> bool:
        return self.llm.ready

    def search(self, query: str, *, top_k: int = 5, jurisdiction: str | None = None,
               semantic_query: str | None = None) -> list[EvidenceChunk]:
        """Resolve the configured backend per request, sharing its loaded models."""
        retriever = self.retriever
        options = {"semantic_query": semantic_query} if isinstance(
            retriever, (SQLiteFTSRetriever, PostgresHybridRetriever)
        ) else {}
        return retriever.search(query, top_k=top_k, jurisdiction=jurisdiction, **options)

    def answer(self, request: QueryRequest) -> QueryResponse:
        detected = detect_language(request.question, request.language)
        retriever = self.retriever
        queries = build_retrieval_queries(request.question, detected)
        if isinstance(retriever, (SQLiteFTSRetriever, PostgresHybridRetriever)):
            # Keyword search uses the normalized query; meaning-based search uses the question as asked.
            queries = queries or [""]

        merged: dict[str, Citation] = {}
        retrieval_failed = False
        for query in queries:
            try:
                options = {"semantic_query": request.question} if isinstance(retriever, (SQLiteFTSRetriever, PostgresHybridRetriever)) else {}
                chunks = retriever.search(
                    query,
                    top_k=max(request.max_sources, self.settings.retrieval_top_k),
                    jurisdiction=request.jurisdiction,
                    **options,
                )
            except Exception:
                LOGGER.exception("Legal retrieval failed")
                retrieval_failed = True
                continue
            for chunk in chunks:
                existing = merged.get(chunk.chunk_id)
                if existing and existing.score is not None and existing.score >= chunk.score:
                    continue
                merged[chunk.chunk_id] = Citation(
                    title=chunk.title,
                    source_file=chunk.source_file,
                    source_id=chunk.source_id,
                    source_url=chunk.source_url,
                    chunk_id=chunk.chunk_id,
                    year=chunk.year,
                    jurisdiction=chunk.jurisdiction,
                    document_type=chunk.document_type,
                    legal_status=chunk.legal_status,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    section_ref=chunk.section_ref,
                    score=chunk.score,
                    excerpt=_excerpt(chunk.text),
                )

        citations = sorted(
            merged.values(),
            key=lambda item: item.score or 0.0,
            reverse=True,
        )[: request.max_sources]

        warnings: list[str] = []
        if not self.retrieval_ready:
            warnings.append("Legal chunk index is not built yet.")
        if retrieval_failed:
            warnings.append("Legal retrieval was unavailable; no unsupported answer was generated.")
        model_ready = self.model_ready
        if not model_ready:
            warnings.append("The local answer model is unavailable; retrieved sources remain accessible.")
        if request.jurisdiction and not citations:
            warnings.append("No indexed source matched the selected jurisdiction.")

        verification = verify_citations(citations)
        if not verification.ok:
            warnings.extend(verification.warnings)

        answer_mode = "insufficient_evidence"
        model_name = None
        if not citations:
            answer = not_enough_sources_message(detected.language)
        else:
            answer_mode = "source_extracts"
            answer = self._build_grounded_draft(detected.language, citations)
            if model_ready and verification.ok:
                try:
                    options = {"language": detected.language, "jurisdiction": request.jurisdiction} if isinstance(self.llm, LocalAnswerClient) else {}
                    generated = self.llm.answer(
                        request.question,
                        [(f"{citation.title or citation.source_file}; jurisdiction: {citation.jurisdiction}; {citation.section_ref or ''}; pages {citation.page_start}-{citation.page_end}", citation.excerpt) for citation in citations],
                        **options,
                    )
                    model_name = generated.model
                    if getattr(generated, "supported", True):
                        answer = generated.text
                        answer_mode = "grounded_answer" if isinstance(self.llm, LocalAnswerClient) else "quotation"
                        if getattr(generated, "language", None) == "english" and detected.language != "english":
                            warnings.append("A verified answer in the requested language could not be produced, so the answer is shown in English.")
                        dropped = getattr(generated, "dropped", 0)
                        if dropped == 1:
                            warnings.append("1 statement could not be verified against the sources and was left out.")
                        elif dropped:
                            warnings.append(f"{dropped} statements could not be verified against the sources and were left out.")
                    else:
                        answer = not_enough_sources_message(detected.language)
                        answer_mode = "insufficient_evidence"
                except Exception:
                    LOGGER.exception("Local generation failed; returning retrieved extracts")
                    warnings.append("Model output could not be verified; showing retrieved source extracts.")

        return QueryResponse(
            question=request.question,
            answer=answer,
            language=LanguageInfo(
                language=detected.language,
                script=detected.script,
                confidence=detected.confidence,
            ),
            citations=citations,
            retrieval_ready=self.retrieval_ready,
            model_ready=model_ready,
            answer_mode=answer_mode,
            model_name=model_name,
            warnings=warnings,
        )

    def _build_grounded_draft(self, language: str, citations: list[Citation]) -> str:
        if language == "urdu":
            intro = "جواب کی تصدیق نہیں ہو سکی۔ متعلقہ قانونی ذرائع کے اصل اقتباسات درج ذیل ہیں:"
        elif language == "roman_urdu":
            intro = (
                "Jawab ki tasdeeq nahi ho saki. Mutaliqa qanooni zaraye ke asal iqtibas neeche hain:"
            )
        else:
            intro = (
                "A generated answer could not be verified. These are the retrieved source extracts:"
            )

        lines = [intro]
        for idx, citation in enumerate(citations, start=1):
            title = citation.title or citation.source_file
            location: list[str] = []
            if citation.section_ref:
                location.append(citation.section_ref)
            if citation.page_start:
                page = f"page {citation.page_start}"
                if citation.page_end and citation.page_end != citation.page_start:
                    page = f"pages {citation.page_start}-{citation.page_end}"
                location.append(page)
            label = f" ({', '.join(location)})" if location else ""
            lines.append(f"{idx}. {title}{label}: {citation.excerpt}")
        return "\n\n".join(lines)
