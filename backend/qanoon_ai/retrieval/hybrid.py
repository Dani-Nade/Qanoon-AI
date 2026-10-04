"""PostgreSQL hybrid retrieval with pgvector, lexical search, and reranking."""

from __future__ import annotations

import math
import numpy as np
from collections.abc import Iterable
from typing import Any

from qanoon_ai.ingestion.postgres_index import DenseEmbedder, production_index_status
from qanoon_ai.retrieval.models import EvidenceChunk
from qanoon_ai.retrieval.sqlite import JURISDICTION_ALIASES


def _connect_dict(database_url: str):
    import psycopg
    from pgvector.psycopg import register_vector
    from psycopg.rows import dict_row

    connection = psycopg.connect(
        database_url,
        autocommit=True,
        connect_timeout=3,
        row_factory=dict_row,
    )
    register_vector(connection)
    return connection


def _jurisdiction_clause(jurisdiction: str | None) -> tuple[str, list[str]]:
    if not jurisdiction:
        return "", []
    values = list(
        JURISDICTION_ALIASES.get(jurisdiction.casefold(), (jurisdiction.casefold(),))
    )
    return " AND d.jurisdiction = ANY(%s)", values


def _rrf(
    result_sets: Iterable[tuple[float, list[dict[str, Any]]]],
    *,
    constant: int = 60,
) -> list[tuple[float, dict[str, Any]]]:
    scores: dict[str, float] = {}
    records: dict[str, dict[str, Any]] = {}
    for weight, rows in result_sets:
        for rank, row in enumerate(rows, start=1):
            chunk_id = row["chunk_id"]
            records[chunk_id] = row
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (constant + rank)
    return sorted(
        ((score, records[chunk_id]) for chunk_id, score in scores.items()),
        key=lambda item: item[0],
        reverse=True,
    )


class PostgresHybridRetriever:
    """Dense + lexical retrieval followed by a multilingual cross-encoder."""

    def __init__(
        self,
        *,
        database_url: str,
        embedding_model: str,
        reranker_model: str,
        reranker_min_score: float = 0.35,
        embedding_batch_size: int = 32,
        embedding_revision: str = "",
        embedding_dimension: int = 1024,
    ) -> None:
        self.database_url = database_url
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        self.reranker_min_score = reranker_min_score
        self.embedding_batch_size = embedding_batch_size
        self.embedding_revision = embedding_revision
        self.embedding_dimension = embedding_dimension
        self._embedder: DenseEmbedder | None = None
        self._reranker = None

    @property
    def ready(self) -> bool:
        return bool(production_index_status(
            self.database_url, embedding_model=self.embedding_model,
            embedding_revision=self.embedding_revision, embedding_dimension=self.embedding_dimension,
        ).get("ready"))

    @property
    def embedder(self) -> DenseEmbedder:
        if self._embedder is None:
            self._embedder = DenseEmbedder(self.embedding_model, self.embedding_batch_size, self.embedding_revision)
        return self._embedder

    @property
    def reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(self.reranker_model)
        return self._reranker

    def _dense_search(
        self,
        connection,
        query_vector: list[float],
        *,
        candidate_limit: int,
        jurisdiction: str | None,
    ) -> list[dict[str, Any]]:
        jurisdiction_sql, jurisdiction_values = _jurisdiction_clause(jurisdiction)
        # pgvector registers NumPy arrays as vectors; Python lists are SQL arrays.
        query_vector = np.asarray(query_vector, dtype=np.float32)
        parameters: list[Any] = [query_vector, self.embedding_model, self.embedding_revision, self.embedding_dimension]
        if jurisdiction_values:
            parameters.append(jurisdiction_values)
        parameters.append(candidate_limit)
        return connection.execute(
            f"""
            SELECT
                c.chunk_id, c.text, c.page_start, c.page_end, c.section_ref,
                c.heading, d.title, d.local_path, d.source_id, d.source_url,
                d.jurisdiction, d.year, d.document_type, d.legal_status,
                1 - (c.embedding <=> %s) AS retrieval_score
            FROM legal_chunks c
            JOIN legal_documents d ON d.document_id = c.document_id
            WHERE c.embedding IS NOT NULL
              AND d.embedding_model = %s AND d.embedding_revision = %s AND d.embedding_dimension = %s
              {jurisdiction_sql}
            ORDER BY c.embedding <=> %s
            LIMIT %s
            """,
            [*parameters[:-1], query_vector, parameters[-1]],
        ).fetchall()

    def _lexical_search(
        self,
        connection,
        query: str,
        *,
        candidate_limit: int,
        jurisdiction: str | None,
    ) -> list[dict[str, Any]]:
        jurisdiction_sql, jurisdiction_values = _jurisdiction_clause(jurisdiction)
        return connection.execute(
            f"""
            WITH query AS (
                SELECT websearch_to_tsquery('english', %s) AS terms
            )
            SELECT
                c.chunk_id, c.text, c.page_start, c.page_end, c.section_ref,
                c.heading, d.title, d.local_path, d.source_id, d.source_url,
                d.jurisdiction, d.year, d.document_type, d.legal_status,
                ts_rank_cd(c.search_vector, query.terms, 32)
                    + similarity(c.title, %s)
                    + similarity(coalesce(c.heading, ''), %s) AS retrieval_score
            FROM legal_chunks c
            JOIN legal_documents d ON d.document_id = c.document_id
            CROSS JOIN query
            WHERE (
                c.search_vector @@ query.terms
                OR similarity(c.title, %s) > 0.18
                OR similarity(coalesce(c.heading, ''), %s) > 0.18
            ) AND d.embedding_model = %s AND d.embedding_revision = %s AND d.embedding_dimension = %s
            {jurisdiction_sql}
            ORDER BY retrieval_score DESC
            LIMIT %s
            """,
            [
                query,
                query,
                query,
                query,
                query,
                self.embedding_model,
                self.embedding_revision,
                self.embedding_dimension,
                *([jurisdiction_values] if jurisdiction_values else []),
                candidate_limit,
            ],
        ).fetchall()

    def _rerank(
        self,
        query: str,
        fused: list[tuple[float, dict[str, Any]]],
        *,
        top_k: int,
    ) -> list[tuple[float, dict[str, Any]]]:
        shortlist = fused[: max(top_k * 8, 40)]
        if not shortlist:
            return []
        passages = [
            "\n".join(
                part
                for part in (
                    row["title"], row.get("section_ref"), row.get("heading"), row["text"]
                )
                if part
            )
            for _, row in shortlist
        ]
        logits = self.reranker.predict(
            [(query, passage) for passage in passages],
            batch_size=min(self.embedding_batch_size, 16),
            show_progress_bar=False,
        )
        reranked: list[tuple[float, dict[str, Any]]] = []
        for (fused_score, row), raw_score in zip(shortlist, logits, strict=True):
            bounded = max(-30.0, min(30.0, float(raw_score)))
            relevance = 1.0 / (1.0 + math.exp(-bounded))
            final_score = (relevance * 0.92) + (min(fused_score * 30.0, 1.0) * 0.08)
            if relevance >= self.reranker_min_score:
                reranked.append((final_score, row))
        reranked.sort(key=lambda item: item[0], reverse=True)
        return reranked[:top_k]

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        jurisdiction: str | None = None,
        semantic_query: str | None = None,
    ) -> list[EvidenceChunk]:
        semantic_query = semantic_query or query
        if not semantic_query.strip() or not self.ready:
            return []
        query_vector = self.embedder.encode([semantic_query])[0]
        candidate_limit = max(top_k * 20, 100)
        connection = _connect_dict(self.database_url)
        try:
            dense = self._dense_search(
                connection,
                query_vector,
                candidate_limit=candidate_limit,
                jurisdiction=jurisdiction,
            )
            lexical = self._lexical_search(
                connection,
                query,
                candidate_limit=candidate_limit,
                jurisdiction=jurisdiction,
            )
        finally:
            connection.close()
        fused = _rrf(((1.0, dense), (1.2, lexical)))
        reranked = self._rerank(semantic_query, fused, top_k=top_k)
        return [
            EvidenceChunk(
                chunk_id=row["chunk_id"],
                source_file=row["local_path"],
                source_id=row["source_id"],
                source_url=row["source_url"],
                title=row["title"],
                jurisdiction=row["jurisdiction"],
                year=str(row["year"]) if row["year"] is not None else None,
                document_type=row["document_type"],
                legal_status=row["legal_status"],
                page_start=row["page_start"],
                page_end=row["page_end"],
                section_ref=row["section_ref"],
                text=row["text"],
                score=round(score, 6),
            )
            for score, row in reranked
        ]
