"""Conversational legal assistant: plan searches, retrieve legal sources, stream a cited answer."""

from __future__ import annotations

from collections.abc import Iterator
import json
import logging
import re

from qanoon_ai.core.config import Settings, settings
from qanoon_ai.language.detection import build_retrieval_queries, detect_language
from qanoon_ai.llm.answer import translation_problems
from qanoon_ai.llm.ollama import GenerationLimitReached, OllamaClient
from qanoon_ai.retrieval.models import EvidenceChunk, EvidenceRetriever
from qanoon_ai.retrieval.service import LegalAnswerService
from qanoon_ai.schemas.source import Citation
from qanoon_ai.verification.quantities import unsupported_quantities
from qanoon_ai.verification.citations import citation_numbers, strip_citations

LOGGER = logging.getLogger(__name__)

HISTORY_MESSAGES = 8

LANGUAGE_NAMES = {
    "english": "English",
    "urdu": "Urdu, written in Urdu script",
    "roman_urdu": "Roman Urdu (Urdu written in English letters, never Urdu script)",
}

NOT_COVERED = {
    "english": "I couldn't find this in the Pakistani law documents I have.",
    "urdu": "مجھے یہ بات میرے پاس موجود پاکستانی قانونی دستاویزات میں نہیں ملی۔",
    "roman_urdu": "Mujhe yeh baat mere paas maujood Pakistani qanooni dastavezat mein nahi mili.",
}

SYSTEM_PROMPT = """You are Qanoon AI, a friendly, knowledgeable assistant for Pakistani law.
Talk with the user naturally, like a helpful expert explaining the law to a non-lawyer.

Grounding rules:
- For any legal question, use ONLY the numbered legal sources in the latest message. Do not use outside knowledge of the law.
- Each source is marked STATUTE or COURT JUDGMENT. State the legal rule from STATUTE sources. Use a COURT JUDGMENT only to show how a court applied or interpreted a statute, and say it is a court decision. Never present something found only in a judgment as the law of Pakistan: judgments often quote other or foreign laws.
- Cite sources inline like [1] or [2] immediately after the statement they support. Every paragraph or bullet with legal content needs a citation. Only cite numbers that exist.
- Never invent sections, penalties, deadlines, procedures, authorities, examples or advice that the sources do not state.
- If the sources do not answer the question, begin your reply with exactly: "{not_covered}" and then briefly say what the sources do cover, if anything relevant.
- Greetings, thanks or small talk need no sources: reply briefly and invite a legal question.

How to answer:
- Reply in {language}.
- Start with a direct answer in one or two sentences.
- Then explain in depth in plain language, using short paragraphs, headings or bullet points where they help: what the law says, the conditions, exceptions, who decides, deadlines and penalties exactly as written.
- If the user asks how to do something or what the procedure is, include a section "What you need to do" with numbered steps in order: who to go to, what to submit or send, to whom, and each deadline. Every step needs a citation. Then say what happens next and the consequences of not following the procedure, if the sources state them.
- If the right answer depends on the user's situation (for example which tax, which province, or the type of case), give the general answer first and finish with one short question asking for that detail.
- When a provision sets out alternatives such as clauses (a) and (b), list each alternative separately with all of its own conditions, for example who it applies to, which offences, and its own time limit. Never move a number from one alternative to another.
- A maximum penalty is "up to", not a fixed term; "may" is not "shall".
- Name the law and section, and say when a rule applies only to a particular province or territory. Court judgments show how courts applied the law; say so when citing one.
- Practical next steps only if the sources state them.
- In Roman Urdu use proper legal words: jurm (crime), saza (punishment), qaid (imprisonment), jurmana (fine), zamanat (bail), adalat (court). In Urdu: جرم، سزا، قید، جرمانہ، ضمانت، عدالت.
- End legal answers with one short line saying this is general legal information, not legal advice."""

PLAN_PROMPT = """You plan searches over a collection of Pakistani laws and court judgments for a legal assistant.
Read the conversation and focus on the user's LATEST message; use earlier messages only to understand what it refers to.
The user may write in English, Urdu script, or Roman Urdu with informal spelling.
First work out what the user actually wants to know, even from informal words, spelling mistakes or mixed languages.
Return ONLY a JSON object:
{"language": "english" or "urdu" or "roman_urdu" (the language of the latest message),
 "needs_law": true or false (false only for greetings, thanks or small talk),
 "vague": true or false (true only if you cannot tell which legal matter the user means),
 "queries": [2 to 4 short English search queries]}
Each query should target one specific provision a complete answer needs, using the legal terms a statute would use:
for a procedure, the steps (who to apply or give notice to, what to file), the deadlines and when it takes effect, and the
consequences of not following it; for an offence, the definition and the punishment. Do not answer the question.

The examples below only show the format. Never copy their topics; plan for the user's own message.
Message: "bhai mera landlord bina notice ke ghar khali karwa raha he kya kr skta hu"
{"language": "roman_urdu", "needs_law": true, "vague": false, "queries": ["grounds for eviction of tenant by landlord", "notice required before eviction of tenant", "application to rent controller for eviction"]}
Message: "ضمانت قبل از گرفتاری کیسے ملتی ہے"
{"language": "urdu", "needs_law": true, "vague": false, "queries": ["pre-arrest bail application", "court power to grant bail before arrest", "conditions for grant of bail in non-bailable offence"]}
Message: "someone cheated me online and took my money what can i do"
{"language": "english", "needs_law": true, "vague": false, "queries": ["punishment for cheating and dishonestly inducing delivery of property", "registration of first information report cognizable offence", "electronic fraud offence"]}
Message: "mera case kya banega"
{"language": "roman_urdu", "needs_law": true, "vague": true, "queries": ["legal procedure in criminal and civil cases"]}"""

LANGUAGES = set(LANGUAGE_NAMES)


def _round_robin(result_lists: list[list[EvidenceChunk]], limit: int) -> list[EvidenceChunk]:
    """Take each query's best remaining passage in turn, so every part of the question gets a source."""
    chosen: dict[str, EvidenceChunk] = {}
    depth = 0
    while len(chosen) < limit and any(depth < len(results) for results in result_lists):
        for results in result_lists:
            if depth < len(results) and results[depth].chunk_id not in chosen and len(chosen) < limit:
                chosen[results[depth].chunk_id] = results[depth]
        depth += 1
    return list(chosen.values())


def _strip_citations(text: str) -> str:
    return strip_citations(text)


def source_kind(chunk: EvidenceChunk) -> str:
    if (chunk.document_type or "").casefold() == "judgment" or (chunk.jurisdiction or "").startswith("courts_"):
        return "COURT JUDGMENT"
    return "STATUTE"


def _source_label(number: int, chunk: EvidenceChunk) -> str:
    kind = source_kind(chunk)
    # Judgment chunks carry paragraph numbers, not real sections, so no section is shown for them.
    section = chunk.section_ref if kind == "STATUTE" else None
    parts = [kind, chunk.title, section, (chunk.jurisdiction or "").replace("courts_", "").replace("_", " ")]
    return f"[{number}] " + " | ".join(part for part in parts if part)


class ChatService:
    def __init__(self, app_settings: Settings = settings, retriever: EvidenceRetriever | None = None, llm: OllamaClient | None = None):
        self.settings = app_settings
        self.retriever = retriever if retriever is not None else LegalAnswerService(app_settings)
        self.llm = llm or OllamaClient(app_settings.ollama_url, app_settings.chat_model,
                                       num_ctx=app_settings.chat_num_ctx, think=app_settings.chat_think,
                                       max_tokens=app_settings.chat_max_tokens)

    def plan(self, messages: list[dict]) -> dict:
        """Language, whether law is needed, and English search queries for the latest message."""
        question = messages[-1]["content"]
        fallback = {"language": detect_language(question, "auto").language, "needs_law": True, "vague": False, "queries": [question]}
        transcript = "\n".join(f"{m['role']}: {_strip_citations(m['content'])[:1200]}" for m in messages[-HISTORY_MESSAGES:])
        try:
            raw = self.llm.complete(
                [{"role": "system", "content": PLAN_PROMPT}, {"role": "user", "content": transcript}],
                max_tokens=300, json_output=True,
            )
            plan = json.loads(raw)
            queries = [q.strip() for q in plan.get("queries", []) if isinstance(q, str) and q.strip()][:4]
            language = plan.get("language") if plan.get("language") in LANGUAGES else fallback["language"]
            # Urdu script is unambiguous; trust the character check over the model for it.
            if detect_language(question, "auto").language == "urdu":
                language = "urdu"
            return {"language": language, "needs_law": plan.get("needs_law") is not False,
                    "vague": plan.get("vague") is True, "queries": queries or [question]}
        except Exception:
            LOGGER.exception("Search planning failed; searching with the latest message")
            return fallback

    def retrieve(self, queries: list[str], jurisdiction: str | None) -> list[EvidenceChunk]:
        result_lists = []
        for query in queries:
            detected = detect_language(query, "auto")
            merged: dict[str, EvidenceChunk] = {}
            for keyword_query in build_retrieval_queries(query, detected) or [""]:
                for chunk in self.retriever.search(keyword_query, top_k=self.settings.chat_max_sources,
                                                   jurisdiction=jurisdiction, semantic_query=query):
                    if chunk.chunk_id not in merged or (merged[chunk.chunk_id].score or 0) < (chunk.score or 0):
                        merged[chunk.chunk_id] = chunk
            result_lists.append(sorted(merged.values(), key=lambda chunk: chunk.score or 0, reverse=True))
        return _round_robin(result_lists, self.settings.chat_max_sources)

    def build_messages(self, messages: list[dict], sources: list[EvidenceChunk], language: str, vague: bool = False) -> list[dict]:
        system = SYSTEM_PROMPT.format(language=LANGUAGE_NAMES.get(language, "English"), not_covered=NOT_COVERED[language])
        history = [{"role": m["role"], "content": _strip_citations(m["content"])} for m in messages[-HISTORY_MESSAGES - 1:-1]]
        if sources:
            block = "\n\n".join(f"{_source_label(i, c)}\n{' '.join(c.text.split())}" for i, c in enumerate(sources, 1))
            latest = f"Legal sources:\n{block}\n\nUser's message: {messages[-1]['content']}"
        else:
            latest = f"Legal sources: none were found for this message.\n\nUser's message: {messages[-1]['content']}"
        if vague:
            latest += ("\n\nNote: it is unclear which legal matter the user means. Do not guess: briefly ask one "
                       "clarifying question (for example what happened, and whether it is a criminal, family, "
                       "property or tax matter), in the user's language.")
        return [{"role": "system", "content": system}, *history, {"role": "user", "content": latest}]

    def fit_context(self, messages: list[dict], sources: list[EvidenceChunk], language: str,
                    vague: bool = False) -> tuple[list[dict], list[EvidenceChunk], list[str]]:
        """Fit complete passages; never let the server silently truncate a provision.

        Without the model's tokenizer, UTF-8 bytes give a conservative token bound
        for Ollama's byte-backed tokenizers. Reserve output and template overhead.
        """
        budget = self.settings.chat_num_ctx - self.settings.chat_max_tokens - 512
        history = messages[-HISTORY_MESSAGES - 1:]
        selected = list(sources)
        while True:
            prompt = self.build_messages(history, selected, language, vague)
            size = sum(len(m["content"].encode("utf-8")) + 64 for m in prompt)
            if size <= budget:
                break
            if len(history) > 1:
                history = history[min(2, len(history) - 1):]
            elif selected:
                selected.pop()
            else:
                raise ValueError("The question exceeds the model context budget; please shorten it.")
        notes = []
        if len(selected) < len(sources):
            notes.append("Some retrieved sources could not fit in the model context. Only complete passages were used.")
        return prompt, selected, notes

    def check_answer(self, answer: str, sources: list[EvidenceChunk], language: str) -> list[str]:
        """Warnings only: the answer is never altered."""
        warnings: list[str] = []
        if answer.strip().startswith(NOT_COVERED[language][:25]):
            return warnings
        cited = citation_numbers(answer)
        if any(n < 1 or n > len(sources) for n in cited):
            warnings.append("The answer refers to a source number that does not exist.")
        if sources and not cited and len(answer) > 300:
            warnings.append("This answer does not cite the legal sources; verify it against the sources panel.")
        problems: list[str] = []
        for paragraph in re.split(r"\n\s*\n", answer):
            refs = {n for n in citation_numbers(paragraph) if 1 <= n <= len(sources)}
            if not refs:
                continue
            evidence = " ".join(f"{sources[n - 1].title} {sources[n - 1].section_ref or ''} {sources[n - 1].text}" for n in refs)
            # List numbering ("1.", "2)") is formatting, not a quantity from the law.
            claim = re.sub(r"(?m)^\s*\d+[.)]\s+", "", _strip_citations(paragraph))
            problems += [value for value in unsupported_quantities(claim, evidence) if value not in problems]
            problems += [p for p in translation_problems(paragraph, evidence) if p not in problems]
        if problems:
            warnings.append("Please double-check against the cited sources: " + "; ".join(problems[:5]) + ".")
        return warnings

    def stream(self, messages: list[dict], jurisdiction: str | None = None) -> Iterator[dict]:
        """Events: {"type": "sources"}, {"type": "token"}..., {"type": "done"}, or {"type": "error"}."""
        plan = self.plan(messages)
        language = plan["language"]
        sources: list[EvidenceChunk] = []
        if plan["needs_law"] and not plan.get("vague"):
            try:
                sources = self.retrieve(plan["queries"], jurisdiction)
            except Exception:
                LOGGER.exception("Retrieval failed")
        retrieved_sources = bool(sources)
        try:
            prompt, sources, notes = self.fit_context(messages, sources, language, vague=plan.get("vague", False))
        except ValueError as error:
            yield {"type": "error", "message": str(error)}
            return
        yield {
            "type": "sources", "language": language, "search_query": "; ".join(plan["queries"]),
            "citations": [
                Citation(title=c.title, source_file=c.source_file, source_id=c.source_id, source_url=c.source_url,
                         chunk_id=c.chunk_id, year=c.year, jurisdiction=c.jurisdiction, document_type=c.document_type,
                         legal_status=c.legal_status, page_start=c.page_start, page_end=c.page_end,
                         section_ref=c.section_ref, score=c.score, excerpt=" ".join(c.text.split())).model_dump()
                for c in sources
            ],
        }
        answer: list[str] = []
        if retrieved_sources and not sources:
            yield {"type": "token", "text": NOT_COVERED[language]}
            yield {"type": "done", "warnings": notes}
            return
        try:
            try:
                for kind, text in self.llm.stream(prompt):
                    if kind == "thinking":
                        yield {"type": "thinking", "text": text}
                    else:
                        answer.append(text)
                        yield {"type": "token", "text": text}
            except GenerationLimitReached:
                # Reasoning used the whole budget without answering: answer directly instead of hanging.
                LOGGER.warning("Thinking hit the token limit; answering without thinking")
                notes.append("The detailed reasoning step ran too long, so this answer was written in fast mode; check it carefully against the sources.")
                for _, text in self.llm.stream(prompt, think=False):
                    answer.append(text)
                    yield {"type": "token", "text": text}
        except Exception as error:
            LOGGER.exception("Chat generation failed")
            yield {"type": "error", "message": f"The language model is unavailable: {error}"}
            return
        yield {"type": "done", "warnings": notes + self.check_answer("".join(answer), sources, language)}
