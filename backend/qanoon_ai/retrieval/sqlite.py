"""Persistent SQLite FTS5 retriever for the local legal index."""

from __future__ import annotations

from functools import lru_cache
from contextlib import closing
import math
import re
import sqlite3
from pathlib import Path

from typing import TYPE_CHECKING

from qanoon_ai.ingestion.legal_index import index_status
from qanoon_ai.retrieval.models import EvidenceChunk
from qanoon_ai.language.detection import normalize_legal_query

if TYPE_CHECKING:
    from qanoon_ai.retrieval.vectors import VectorSearcher

# Prefer the operative law over schedules in unrelated statutes which merely
# mention it. These are retrieval concepts, not canned legal answers.
TOPICS = {
    "theft": ({"theft", "steal", "stealing", "stolen", "thief"}, ("penal code",)),
    "bail": ({"bail", "bailable", "nonbailable"}, ("criminal procedure",)),
    "maintenance": ({"maintenance", "alimony"}, ("family courts",)),
    "witness": ({"witness", "witnesses", "testify", "testimony", "competency"}, ("shahadat",)),
    "civil_servant": ({"servant", "servants"}, ("civil servants",)),
}


def _topic_keys(groups):
    tokens = set().union(*groups) if groups else set()
    return [key for key, (terms, _) in TOPICS.items() if terms & tokens]


def _is_navigation(text):
    headings = re.findall(r"(?m)^\s*\d{1,3}[A-Z]?\.\s", text)
    operative = re.findall(r"\b(?:shall|whoever|provided|means|unless|may|entitled)\b", text, re.I)
    return ("CONTENTS" in text[:350].upper() and len(operative) < 2) or (len(headings) >= 4 and len(operative) < 2) or (len(text.split()) < 45 and not re.search(r"\b(?:shall|whoever|provided|means|unless|entitled|is|are)\b", text, re.I))

MINIMUM_WEIGHTED_COVERAGE = 0.6

QUERY_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u0600-\u06ff]+")
QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "be",
    "for",
    "hai",
    "hain",
    "is",
    "ka",
    "ke",
    "ki",
    "kon",
    "kaun",
    "liye",
    "may",
    "mein",
    "of",
    "the",
    "to",
    "under",
    "what",
    "who",
}
QUERY_EXPANSIONS = {
    "banne": {"person", "persons"},
    "gawah": {"witness", "testify", "testimony"},
    "shaheed": {"witness", "testify"},
    "testify": {"witness", "testimony"},
    "testimony": {"witness", "testify"},
    "witness": {"testify", "testimony"},
    "competent": {"competency", "capable"},
    "chori": {"theft"},
    "saza": {"punishment", "penalty", "imprisonment"},
    "zamanat": {"bail"},
    # Everyday wording -> the wording used in the statutes.
    "bounce": {"dishonoured", "dishonour"},
    "fir": {"cognizable"},
    "motorcycle": {"motor", "cycle", "vehicle"},
    "self": {"private"},
    "defense": {"defence"},
    "murder": {"qatl"},
    "threaten": {"intimidation", "threat"},
    "threat": {"intimidation"},
    "custody": {"detain", "detained", "detention"},
    "compromise": {"compound", "compounded"},
    "divorce": {"talaq", "dissolution"},
    "second": {"another", "polygamy"},
    "nikah": {"marriage"},
    "buy": {"receive", "purchase", "retain"},
    "limit": {"limitation"},
    "license": {"licence"},
    "tenancy": {"tenant", "landlord"},
    "report": {"forward"},
    "kill": {"qatl", "death"},
    "crime": {"criminal", "offence"},
    "people": {"persons"},
    "register": {"writing", "entered", "registered"},
    "constitutional": {"constitution"},
    "marry": {"marriage", "contract"},
    "decide": {"dispose", "disposed"},
    "time": {"period"},
    "own": {"vest", "vested"},
    "owns": {"vest", "vested"},
    "girl": {"female"},
    "adult": {"minor"},
    "boy": {"male"},
    "below": {"under"},
    "defamation": {"defamatory"},
    "suit": {"action"},
    "valid": {"contracts"},
    "enter": {"contract"},
}
JURISDICTION_ALIASES = {
    "kp": ("khyber_pakhtunkhwa", "courts_khyber_pakhtunkhwa"),
    "khyber_pakhtunkhwa": ("khyber_pakhtunkhwa", "courts_khyber_pakhtunkhwa"),
    "punjab": ("punjab", "courts_punjab"),
    "sindh": ("sindh", "courts_sindh"),
    "balochistan": ("balochistan", "courts_balochistan"),
    "federal": ("federal", "courts_federal"),
    "ict": ("federal_ict", "courts_federal_ict"),
}


@lru_cache(maxsize=200_000)
def stem(token: str) -> str:
    """Light English suffix stripping so cheating/cheats/cheated and punished/punishment meet."""
    if len(token) <= 3 or not token.isascii():
        return token
    # (suffix, replacement, minimum length of what remains)
    for suffix, replacement, keep in (("ments", "", 4), ("ment", "", 4), ("ings", "", 3), ("ing", "", 3),
                                      ("able", "", 3), ("ies", "y", 2), ("ied", "y", 2), ("edly", "", 3),
                                      ("ed", "", 3), ("ly", "", 5), ("es", "", 3), ("s", "", 3)):
        if not token.endswith(suffix) or token.endswith("ss") or len(token) - len(suffix) < keep:
            continue
        if suffix == "es" and not token[:-2].endswith(("s", "x", "z", "h")):
            continue  # "gives" -> "give", handled by the "s" rule
        token = token[: -len(suffix)] + replacement
        break
    if len(token) > 4 and token[-1] == token[-2] and token[-1] not in "lsz":
        token = token[:-1]  # committed -> commit
    if len(token) > 3 and token.endswith("e"):
        token = token[:-1]  # give/gives -> giv, notice/noticed -> notic
    return token


def _stems(tokens) -> set[str]:
    return {stem(token) for token in tokens}


# Keyed by stem so "bounces" and "threatening" reach the "bounce" and "threaten" entries.
EXPANSIONS_BY_STEM: dict[str, set[str]] = {}
for _word, _terms in QUERY_EXPANSIONS.items():
    EXPANSIONS_BY_STEM.setdefault(stem(_word), set()).update(_terms)


def _query_groups(text: str) -> list[set[str]]:
    seen: set[str] = set()
    groups: list[set[str]] = []
    for token in QUERY_TOKEN_RE.findall(normalize_legal_query(text)):
        normalized = token.casefold()
        if len(normalized) < 2 or normalized in QUERY_STOPWORDS or normalized in seen:
            continue
        seen.add(normalized)
        equivalents = set(EXPANSIONS_BY_STEM.get(stem(normalized), set()))
        for terms, _ in TOPICS.values():
            if normalized in terms:
                equivalents.update(terms)
        if any(normalized in existing for existing in groups):
            continue
        groups.append(_stems({normalized, *equivalents}))
        if len(groups) == 16:
            break
    return groups


def _fts_term(token: str) -> str:
    escaped = token.replace('"', '""')
    # Stems are matched as prefixes so "punish" finds punished, punishment, punishable.
    return f'"{escaped}"*' if token.isascii() and len(token) >= 3 else f'"{escaped}"'


def _fts_query(groups: list[set[str]]) -> str:
    return " OR ".join(_fts_term(token) for token in sorted({token for group in groups for token in group}))


@lru_cache(maxsize=20_000)
def _document_frequency(index_file: str, index_mtime: float, term: str | None) -> int:
    with closing(sqlite3.connect(f"file:{index_file}?mode=ro", uri=True)) as connection:
        if term is None:
            return connection.execute("SELECT count(*) FROM chunks").fetchone()[0]
        return connection.execute("SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH ?", (_fts_term(term),)).fetchone()[0]


def _group_weights(index_file: Path, groups: list[set[str]]) -> list[float]:
    """Inverse document frequency per group: rare legal terms matter more than "give" or "happens"."""
    mtime = index_file.stat().st_mtime
    total = max(_document_frequency(str(index_file), mtime, None), 1)
    weights = []
    for group in groups:
        frequency = max(_document_frequency(str(index_file), mtime, term) for term in group)
        # A word no passage contains cannot tell passages apart; it must not veto the rest.
        weights.append(math.log((total - frequency + 0.5) / (frequency + 0.5) + 1.0) if frequency else 0.0)
    return weights


def _rerank_score(row: sqlite3.Row, groups: list[set[str]], weights: list[float]) -> tuple[float, int, float]:
    title_tokens = _stems(token.casefold() for token in QUERY_TOKEN_RE.findall(row["title"] or ""))
    heading_tokens = _stems(token.casefold() for token in QUERY_TOKEN_RE.findall(row["heading"] or ""))
    body_sequence = [
        stem(token.casefold())
        for token in QUERY_TOKEN_RE.findall(
            " ".join((row["section_ref"] or "", row["heading"] or "", row["text"] or ""))
        )
    ]
    all_tokens = title_tokens | set(body_sequence)
    matched = [bool(group & all_tokens) for group in groups]
    matched_groups = sum(matched)
    title_matches = sum(bool(group & title_tokens) for group in groups)
    heading_matches = sum(weight for group, weight in zip(groups, weights) if group & heading_tokens)
    proximity_matches = 0
    for left, right in zip(groups, groups[1:]):
        left_positions = [i for i, token in enumerate(body_sequence) if token in left]
        right_positions = [i for i, token in enumerate(body_sequence) if token in right]
        if any(0 < right_pos - left_pos <= 5 for left_pos in left_positions for right_pos in right_positions):
            proximity_matches += 1
    coverage = sum(weight for hit, weight in zip(matched, weights) if hit) / max(sum(weights), 1e-9)
    bm25_score = max(0.0, -float(row["rank"]))
    score = (
        bm25_score
        + (coverage * 30.0)
        + (title_matches * 3.0)
        + (heading_matches * 2.0)
        + (proximity_matches * 6.0)
    )
    title = re.sub(r"[^a-z0-9]+", " ", (row["title"] or "").lower())
    for key in _topic_keys(groups):
        terms, primary_titles = TOPICS[key]
        if any(pattern in title for pattern in primary_titles):
            score += 80
        if terms & heading_tokens:
            score += 30
    if (row["section_ref"] or "").lower().startswith(("schedule", "chapter", "part")):
        score -= 15
    # Appeal-stage bail is not the starting point for a general bail question.
    if "bail" in _topic_keys(groups) and "appeal" in heading_tokens and not any("appeal" in group for group in groups):
        score -= 30
    return score, matched_groups, coverage


ROW_COLUMNS = """
    c.chunk_id, c.document_id, c.chunk_index, c.text, c.page_start, c.page_end, c.section_ref, c.heading,
    d.title, d.local_path, d.source_id, d.source_url, d.jurisdiction,
    d.year, d.document_type, d.legal_status, d.authority_type
"""
# Of the passages returned, this share is reserved for statutes; the rest for judgments and
# court rules. Without it, the ~13k judgments that quote a section crowd out the section itself.
LEGISLATION_SHARE = 5 / 6
MAX_EXPANDED_CHARS = 12000
# Reciprocal rank fusion constant, and the similarity a passage found only by
# meaning (no keyword match) needs before it may be shown as evidence.
RRF_K = 60
VECTOR_CANDIDATES = 50
# Chosen on the eval set: 0.60 kept every expected section in the top 5 while 0.55 let noise displace two.
VECTOR_MIN_SIMILARITY = 0.60


class SQLiteFTSRetriever:
    def __init__(self, index_file: Path, vectors: "VectorSearcher | None" = None) -> None:
        self.index_file = index_file
        self.vectors = vectors

    @property
    def ready(self) -> bool:
        return bool(index_status(self.index_file).get("ready"))

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        jurisdiction: str | None = None,
        semantic_query: str | None = None,
    ) -> list[EvidenceChunk]:
        if not self.ready:
            return []
        jurisdictions = (
            JURISDICTION_ALIASES.get(jurisdiction.casefold(), (jurisdiction.casefold(),)) if jurisdiction else None
        )
        ranked = self._keyword_ranked(query, top_k, jurisdictions)
        if self.vectors is not None and self.vectors.available and (semantic_query or query).strip():
            ranked = self._fuse_vectors(ranked, semantic_query or query, jurisdictions)
        return self._assemble(ranked, top_k)

    def _keyword_ranked(self, query: str, top_k: int, jurisdictions: tuple[str, ...] | None) -> list[tuple[float, sqlite3.Row]]:
        groups = _query_groups(query)
        match_query = _fts_query(groups)
        if not match_query:
            return []

        weights = _group_weights(self.index_file, groups)
        match_queries = [match_query]
        if len(groups) >= 2:
            # Any-term matching lets long, wordy statutes crowd out the right section,
            # so also fetch passages containing the question's rarest terms together.
            rarest = sorted(range(len(groups)), key=lambda i: weights[i], reverse=True)[:3]
            match_queries.append(" AND ".join(
                "(" + " OR ".join(_fts_term(token) for token in sorted(groups[i])) + ")" for i in rarest
            ))

        connection = sqlite3.connect(f"file:{self.index_file}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        clauses = ["chunks_fts MATCH ?"]
        filters: list[object] = []
        if jurisdictions:
            placeholders = ",".join("?" for _ in jurisdictions)
            clauses.append(f"d.jurisdiction IN ({placeholders})")
            filters.extend(jurisdictions)
        candidate_limit = max(top_k * 50, 400)

        sql = f"""
            SELECT {ROW_COLUMNS},
                bm25(chunks_fts, 0.0, 5.0, 2.0, 2.0, 4.0, 1.0) AS rank
            FROM chunks_fts
            JOIN chunks c ON c.chunk_id = chunks_fts.chunk_id
            JOIN documents d ON d.document_id = c.document_id
            WHERE {' AND '.join(clauses)}
            ORDER BY rank
            LIMIT ?
        """
        rows_by_chunk: dict[str, sqlite3.Row] = {}
        try:
            for query_text in match_queries:
                for row in connection.execute(sql, [query_text, *filters, candidate_limit]).fetchall():
                    rows_by_chunk.setdefault(row["chunk_id"], row)
        finally:
            connection.close()
        rows = list(rows_by_chunk.values())

        topics = _topic_keys(groups)
        ranked: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            if _is_navigation(row["text"]):
                continue
            body_tokens = {token.casefold() for token in QUERY_TOKEN_RE.findall(row["text"] or "")}
            # Generic words like "legal process" can never satisfy a bail query.
            if topics and not any(TOPICS[key][0] & body_tokens for key in topics):
                continue
            score, matched_groups, coverage = _rerank_score(row, groups, weights)
            # Coverage is weighted by rarity, so missing "happens" or "someone" costs little,
            # while a passage without the rare legal term of the question is dropped.
            if matched_groups and coverage >= (0.35 if topics else MINIMUM_WEIGHTED_COVERAGE):
                ranked.append((score, row))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked

    def _fuse_vectors(self, ranked, semantic_query: str, jurisdictions: tuple[str, ...] | None):
        """Merge keyword and meaning-based rankings with reciprocal rank fusion."""
        hits = [
            (chunk_id, similarity)
            for chunk_id, similarity in self.vectors.search(semantic_query, top_k=VECTOR_CANDIDATES, jurisdictions=jurisdictions)
            if similarity >= VECTOR_MIN_SIMILARITY
        ]
        if not hits:
            return ranked
        rows = {row["chunk_id"]: row for _, row in ranked}
        missing = [chunk_id for chunk_id, _ in hits if chunk_id not in rows]
        if missing:
            placeholders = ",".join("?" for _ in missing)
            with closing(sqlite3.connect(f"file:{self.index_file}?mode=ro", uri=True)) as connection:
                connection.row_factory = sqlite3.Row
                for row in connection.execute(
                    f"SELECT {ROW_COLUMNS}, 0.0 AS rank FROM chunks c JOIN documents d ON d.document_id = c.document_id "
                    f"WHERE c.chunk_id IN ({placeholders})", missing,
                ):
                    if not _is_navigation(row["text"]):
                        rows[row["chunk_id"]] = row
        keyword_rank = {row["chunk_id"]: position for position, (_, row) in enumerate(ranked)}
        vector_rank = {chunk_id: position for position, (chunk_id, _) in enumerate(hits)}
        fused = []
        for chunk_id, row in rows.items():
            score = sum(1.0 / (RRF_K + ranks[chunk_id]) for ranks in (keyword_rank, vector_rank) if chunk_id in ranks)
            fused.append((score * 1000.0, row))
        fused.sort(key=lambda item: item[0], reverse=True)
        return fused

    def _assemble(self, ranked, top_k: int) -> list[EvidenceChunk]:
        # Avoid filling the context with multiple fragments of one section.
        diverse = []
        seen_sections = set()
        for score, row in ranked:
            normalized_title = re.sub(r"[^a-z0-9]+", " ", (row["title"] or "").lower())
            canonical_title = next((pattern for _, patterns in TOPICS.values() for pattern in patterns if pattern in normalized_title and "amendment" not in normalized_title), row["local_path"])
            section = (row["jurisdiction"], canonical_title, row["section_ref"])
            if row["section_ref"] and section in seen_sections:
                continue
            diverse.append((score, row))
            seen_sections.add(section)

        # Statutes first, then how courts applied them; either type fills unused slots.
        statutes = [item for item in diverse if (item[1]["authority_type"] or "legislation") == "legislation"]
        others = [item for item in diverse if (item[1]["authority_type"] or "legislation") != "legislation"]
        statute_slots = max(1, round(top_k * LEGISLATION_SHARE))
        chosen_statutes = statutes[:statute_slots]
        chosen_others = others[: top_k - len(chosen_statutes)]
        chosen_statutes += statutes[len(chosen_statutes): top_k - len(chosen_statutes) - len(chosen_others)]
        selected = chosen_statutes + chosen_others
        # Scores follow this order so callers that re-sort by score keep it.
        selected = [(float(len(selected) - position), row) for position, (_, row) in enumerate(selected)]

        # Reassemble contiguous fragments of a provision, including qualifications
        # on the next page. A separate contents entry must never be joined to it.
        # Judgments are not split by real sections, so their fragments are not merged.
        expanded = []
        with closing(sqlite3.connect(f"file:{self.index_file}?mode=ro", uri=True)) as connection:
            connection.row_factory = sqlite3.Row
            for score, source in selected:
                row = dict(source)
                if row["section_ref"] and (row["authority_type"] or "legislation") == "legislation":
                    siblings = connection.execute("SELECT chunk_index, text, page_start, page_end FROM chunks WHERE document_id = ? AND section_ref = ? ORDER BY chunk_index", (row["document_id"], row["section_ref"])).fetchall()
                    indices = {row["chunk_index"]}
                    available = {item["chunk_index"]: item for item in siblings}
                    for direction in (-1, 1):
                        index = row["chunk_index"] + direction
                        while index in available:
                            indices.add(index)
                            index += direction
                    parts = [available[index] for index in sorted(indices)]
                    while len(parts) > 1 and sum(len(part["text"]) for part in parts) > MAX_EXPANDED_CHARS:
                        # Trim the fragment farthest from the matched one.
                        first, last = parts[0]["chunk_index"], parts[-1]["chunk_index"]
                        parts = parts[1:] if row["chunk_index"] - first > last - row["chunk_index"] else parts[:-1]
                    row["text"] = "\n\n".join(part["text"] for part in parts)
                    row["page_start"] = min(part["page_start"] for part in parts)
                    row["page_end"] = max(part["page_end"] for part in parts)
                expanded.append((score, row))

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
                section_ref=(row["section_ref"].replace("Section", "Article", 1) if row["section_ref"] and "shahadat" in row["title"].lower() else row["section_ref"]),
                text=row["text"],
                score=round(score, 6),
            )
            for score, row in expanded
        ]
