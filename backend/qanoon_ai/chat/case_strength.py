"""Case strength estimation: compares a user's facts against similar court judgments."""

from __future__ import annotations

from collections.abc import Iterator
import json
import logging

from qanoon_ai.core.config import Settings, settings
from qanoon_ai.language.detection import detect_language
from qanoon_ai.llm.ollama import GenerationLimitReached, OllamaClient
from qanoon_ai.retrieval.models import EvidenceChunk, EvidenceRetriever
from qanoon_ai.retrieval.service import LegalAnswerService
from qanoon_ai.schemas.source import Citation
from qanoon_ai.verification.citations import citation_numbers

LOGGER = logging.getLogger(__name__)

LANGUAGE_NAMES = {
    "english": "English",
    "urdu": "Urdu, written in Urdu script",
    "roman_urdu": "Roman Urdu (Urdu written in English letters, never Urdu script)",
}

SYSTEM_PROMPT = """You are Qanoon AI, estimating how strong a user's legal case looks by comparing it against \
reported Pakistani court judgments with broadly similar facts.

Grounding rules:
- Use ONLY the numbered precedent judgments given below. Do not use outside knowledge of case law.
- Cite a precedent inline like [1] immediately after any statement drawn from it. Only cite numbers that exist.
- Never state or imply a guaranteed outcome. Courts decide on the specific facts and evidence of each case; \
precedents show how courts have reasoned in similar situations, not what will happen here.
- If the precedents are too few, too old, or too different in facts to support a judgement, say so plainly \
instead of guessing.

How to answer, in {language}:
1. Start with one line, exactly: "Case strength: Low" or "Case strength: Medium" or "Case strength: High" or \
"Case strength: Not enough information".
2. In 2-4 short paragraphs, explain why: which facts help the user's position, which hurt it, and how the cited \
precedents reasoned about similar facts.
3. End with exactly this sentence, translated naturally into {language}: "This is a general comparison with past \
cases, not a prediction of your result or legal advice -- speak to a lawyer about your specific case."
"""

PLAN_PROMPT = """You plan a search for Pakistani court judgments with facts similar to a user's legal situation.
Read the user's case description, written in English, Urdu script, or Roman Urdu, possibly with informal spelling.
Return ONLY a JSON object:
{"language": "english" or "urdu" or "roman_urdu",
 "queries": [2 to 3 short English search queries describing the FACTS and legal issue, not the user's wording]}
Focus each query on a distinct angle a court judgment might share: the type of dispute, the specific act alleged,
and any aggravating or mitigating circumstance mentioned.

Message: "meri biwi ne mujh par jhootay ilzaam laga kar khula mang liya"
{"language": "roman_urdu", "queries": ["wife seeking khula on false allegations against husband", "husband defence against khula without proof", "family court khula proceedings evidentiary standard"]}"""


def _source_label(number: int, chunk: EvidenceChunk) -> str:
    parts = [chunk.title, (chunk.jurisdiction or "").replace("courts_", "").replace("_", " ") or None, chunk.year]
    return f"[{number}] " + " | ".join(p for p in parts if p)


def _is_judgment(chunk: EvidenceChunk) -> bool:
    return (chunk.document_type or "").casefold() == "judgment" or (chunk.jurisdiction or "").startswith("courts_")


class CaseStrengthService:
    def __init__(self, app_settings: Settings = settings, retriever: EvidenceRetriever | None = None,
                 llm: OllamaClient | None = None):
        self.settings = app_settings
        self.retriever = retriever if retriever is not None else LegalAnswerService(app_settings)
        self.llm = llm or OllamaClient(app_settings.ollama_url, app_settings.chat_model,
                                       num_ctx=app_settings.chat_num_ctx, think=app_settings.chat_think,
                                       max_tokens=app_settings.chat_max_tokens)

    def plan(self, facts: str) -> dict:
        fallback = {"language": detect_language(facts, "auto").language, "queries": [facts]}
        try:
            raw = self.llm.complete(
                [{"role": "system", "content": PLAN_PROMPT}, {"role": "user", "content": facts[:2000]}],
                max_tokens=250, json_output=True,
            )
            plan = json.loads(raw)
            queries = [q.strip() for q in plan.get("queries", []) if isinstance(q, str) and q.strip()][:3]
            language = plan.get("language") if plan.get("language") in LANGUAGE_NAMES else fallback["language"]
            if detect_language(facts, "auto").language == "urdu":
                language = "urdu"
            return {"language": language, "queries": queries or [facts]}
        except Exception:
            LOGGER.exception("Case strength search planning failed; searching with the raw facts")
            return fallback

    def retrieve_precedents(self, queries: list[str], jurisdiction: str | None) -> list[EvidenceChunk]:
        seen: dict[str, EvidenceChunk] = {}
        for query in queries:
            for chunk in self.retriever.search(query, top_k=20, jurisdiction=jurisdiction, semantic_query=query):
                if not _is_judgment(chunk):
                    continue
                if chunk.chunk_id not in seen or (seen[chunk.chunk_id].score or 0) < (chunk.score or 0):
                    seen[chunk.chunk_id] = chunk
        ranked = sorted(seen.values(), key=lambda c: c.score or 0, reverse=True)
        return ranked[: self.settings.chat_max_sources]

    def build_messages(self, facts: str, precedents: list[EvidenceChunk], language: str) -> list[dict]:
        system = SYSTEM_PROMPT.format(language=LANGUAGE_NAMES.get(language, "English"))
        if precedents:
            block = "\n\n".join(f"{_source_label(i, c)}\n{' '.join(c.text.split())}" for i, c in enumerate(precedents, 1))
            user = f"Precedent judgments:\n{block}\n\nUser's case facts: {facts}"
        else:
            user = f"Precedent judgments: none were found with similar facts.\n\nUser's case facts: {facts}"
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def stream(self, facts: str, jurisdiction: str | None = None) -> Iterator[dict]:
        """Events: {"type": "sources"}, {"type": "thinking"}, {"type": "token"}, {"type": "done"} or {"type": "error"}."""
        plan = self.plan(facts)
        language = plan["language"]
        try:
            precedents = self.retrieve_precedents(plan["queries"], jurisdiction)
        except Exception:
            LOGGER.exception("Precedent retrieval failed")
            precedents = []

        yield {
            "type": "sources", "language": language, "search_query": "; ".join(plan["queries"]),
            "citations": [
                Citation(title=c.title, source_file=c.source_file, source_id=c.source_id, source_url=c.source_url,
                         chunk_id=c.chunk_id, year=c.year, jurisdiction=c.jurisdiction, document_type=c.document_type,
                         legal_status=c.legal_status, page_start=c.page_start, page_end=c.page_end,
                         section_ref=c.section_ref, score=c.score, excerpt=" ".join(c.text.split())).model_dump()
                for c in precedents
            ],
        }
        if not precedents:
            message = ("Case strength: Not enough information\n\nI couldn't find Pakistani court judgments with "
                        "similar facts in the indexed corpus, so I can't compare your case to precedent. This is "
                        "not a judgement on your case's merits -- please consult a lawyer.")
            yield {"type": "token", "text": message}
            yield {"type": "done", "warnings": ["No matching precedents were found."]}
            return

        prompt = self.build_messages(facts, precedents, language)
        answer: list[str] = []
        try:
            try:
                for kind, text in self.llm.stream(prompt):
                    if kind == "thinking":
                        yield {"type": "thinking", "text": text}
                    else:
                        answer.append(text)
                        yield {"type": "token", "text": text}
            except GenerationLimitReached:
                LOGGER.warning("Thinking hit the token limit; answering without thinking")
                for _, text in self.llm.stream(prompt, think=False):
                    answer.append(text)
                    yield {"type": "token", "text": text}
        except Exception as error:
            LOGGER.exception("Case strength generation failed")
            yield {"type": "error", "message": f"The language model is unavailable: {error}"}
            return

        cited = citation_numbers("".join(answer))
        warnings = []
        if any(n < 1 or n > len(precedents) for n in cited):
            warnings.append("The answer refers to a precedent number that does not exist.")
        yield {"type": "done", "warnings": warnings}
