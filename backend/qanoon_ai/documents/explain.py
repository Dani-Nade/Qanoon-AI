"""Explain an uploaded document: what it is, what it says, the law behind it, and what to do.

The explanation is built from the reviewed page text (including the user's corrections).
Handwritten and uncertain values are marked as unreliable in the prompt and must not be
stated as fact. Sections the document cites are looked up in the legal index so the law
is explained from statutes, with the same citation format as chat.
"""

from __future__ import annotations

from collections.abc import Iterator
import logging
import re

from qanoon_ai.chat.service import LANGUAGE_NAMES, ChatService, _source_label
from qanoon_ai.llm.ollama import GenerationLimitReached
from qanoon_ai.retrieval.models import EvidenceChunk
from qanoon_ai.schemas.source import Citation
from qanoon_ai.verification.quantities import unsupported_quantities

LOGGER = logging.getLogger(__name__)

DOCUMENT_CHARS = 14000
# Case numbers, dates and similar references: digits possibly joined by letters or separators.
REFERENCE_RE = re.compile(r"\d[\dA-Za-z./\-]*\d|\d")
YEAR_RE = re.compile(r"(19|20)\d\d")
CITATION_RE = re.compile(r"\[\d+\](?:\s*,\s*\[\d+\])*")


def _compact(text: str) -> str:
    return "".join(ch for ch in text.casefold() if ch.isalnum())

# Statute abbreviations used in Pakistani court papers and FIRs, keyed without spaces or dots.
LAW_NAMES = {
    "ppc": "Pakistan Penal Code", "تپ": "Pakistan Penal Code",
    "crpc": "Code of Criminal Procedure", "ضف": "Code of Criminal Procedure",
    "cpc": "Code of Civil Procedure", "qso": "Qanun-e-Shahadat Order", "mflo": "Muslim Family Laws Ordinance",
    "ata": "Anti-Terrorism Act", "cnsa": "Control of Narcotic Substances Act",
}
SECTION_RE = re.compile(
    r"(?:u/s|under\s+sections?|sections?|s\.|زیر\s+دفعہ|دفعہ)\s*([0-9][0-9A-Za-z/,\-\s]{0,30}?)\s*"
    r"(ppc|p\.p\.c|cr\.?\s?p\.?\s?c|c\.p\.c|cpc|qso|mflo|ata|cnsa|ت\s?پ|ض\s?ف)\b",
    re.IGNORECASE,
)

EXPLAIN_PROMPT = """You are Qanoon AI. A person has uploaded a Pakistani legal document and wants to understand it.
Explain it clearly and correctly for a non-lawyer, in {language}.

You are given:
1. The document text, read from the pages. Values marked [HANDWRITTEN, NOT RELIABLE: ...] or
   [UNCERTAIN: ...] were not read reliably. Never state them as facts; list them under "Could not be read".
2. Numbered legal sources from the law index, marked STATUTE or COURT JUDGMENT.

Rules:
- Describe the document only from its text. Cite the page like (page 1). Do not invent parties, dates,
  numbers, courts, orders or events that the text does not contain.
- Explain the law only from the numbered STATUTE sources, citing them like [1]. If a cited section is not in
  the sources, say the text of that section was not available instead of explaining it from memory.
- Explain only sources that apply to THIS document: the sections it cites, and the procedure it is part of.
  Skip sources about other laws or other offences (e.g. narcotics or prohibition laws for an attempted-murder
  bail petition); do not mention skipped sources at all.
- Keep numbers, dates and case numbers exactly as written in the document. Urdu number words keep their value:
  تیس = 30 (thirty), تین = 3, دس = 10, نوے = 90, ساٹھ = 60; "تیس دنوں" is thirty days, never three days.
- Handwritten or uncertain values go ONLY under "Could not be read", never under "Key details" or anywhere else.
- Read legal wording precisely: "roped/implicated by the complainant" means named as an accused (not that the
  person admits involvement); "may" is not "shall"; a maximum penalty is "up to".
- Next steps only if the document or the sources state them. No guessing about the outcome of the case.
- Legal words in Roman Urdu: jurm (crime/offence), muqadma (case), saza (punishment), qaid (imprisonment),
  jurmana (fine), zamanat (bail), adalat (court), mulzim (accused), mudai (complainant), fareeq (party, فریق),
  degree (decree, ڈگری), faisla (judgment), muqarrar karna (appoint, تقرر), razamandi (consent, رضامندی),
  dastakhat (signature), gawah (witness), tasfiya / samjhota (settlement), qabil-e-rawani (compoundable).
  Never write "jinnat", "fric", "dekhi", "taweez" or "izzat" for these. If unsure of a Roman Urdu word, use the
  English legal word instead of guessing. In Urdu: جرم، مقدمہ، سزا، قید، جرمانہ، ضمانت، عدالت، ملزم، مدعی، فریق، ڈگری۔

Use these six headings in order, written in {language} (not in English unless the language is English):
## What this document is
## Key details
(a short list: parties, court or police station, case/FIR/petition numbers, important dates, sections cited; with page)
## What it says
## The law behind it
## What this means for you
## Could not be read
(every handwritten or uncertain value, or "Nothing" if there are none)
End with one line saying this is general legal information, not legal advice."""


def document_text(pages: list[dict]) -> tuple[str, list[str]]:
    """Page text for the prompt, with unreliable values marked; also returns those values."""
    unreliable: list[str] = []
    parts = []
    for page in pages:
        lines = [f"--- Page {page['number']} ({page.get('document_type') or 'document'}, {page.get('language') or 'unknown language'}) ---"]
        for block in page["blocks"]:
            text = " ".join(block["text"].split())
            if not text:
                continue
            if block.get("handwritten") and not block.get("edited"):
                lines.append(f"[HANDWRITTEN, NOT RELIABLE: {text}]")
                # Numbers are what an explanation would repeat ("383-B/2016"), so track those;
                # a bare year or one or two digits also appear in printed dates, so they are not tracked alone.
                references = [r for r in REFERENCE_RE.findall(text) if len(r) >= 3 and not YEAR_RE.fullmatch(r)]
                unreliable += references or [text]
                continue
            for value in block.get("uncertain", []):
                if value and value in text:
                    text = text.replace(value, f"[UNCERTAIN: {value}]")
                    unreliable.append(value)
            lines.append(text)
        parts.append("\n".join(lines))
    return "\n\n".join(parts), unreliable


def cited_sections(text: str) -> list[str]:
    """Search queries for sections the document cites, e.g. 'section 497 Code of Criminal Procedure'."""
    queries: list[str] = []
    for numbers, law in SECTION_RE.findall(text):
        name = LAW_NAMES.get(re.sub(r"[\s.]", "", law.casefold()))
        if not name:
            continue
        for number in re.findall(r"\d+[A-Za-z]?", numbers):
            query = f"section {number} {name}"
            if query not in queries:
                queries.append(query)
    return queries[:6]


class DocumentExplainer:
    def __init__(self, chat: ChatService):
        self.chat = chat

    def _sources(self, text: str) -> list[EvidenceChunk]:
        queries = cited_sections(text)
        # Also let the planner name the laws a document like this depends on.
        summary = text[:3000]
        plan = self.chat.plan([{"role": "user", "content": f"Which laws apply to this legal document?\n{summary}"}])
        # When the document names its sections, planner queries only fill in the procedure around them.
        limit = min(8, len(queries) + 2) if queries else 8
        for query in plan.get("queries", []):
            if query not in queries and len(queries) < limit:
                queries.append(query)
        return self.chat.retrieve(queries, None) if queries else []

    def check(self, explanation: str, text: str, sources: list[EvidenceChunk], unreliable: list[str]) -> list[str]:
        warnings = []
        evidence = text + " " + " ".join(f"{s.title} {s.section_ref or ''} {s.text}" for s in sources)
        # Citation markers ([7]) and list numbering (1.) are formatting, not quantities.
        body = re.sub(r"(?m)^\s*\d+[.)]\s+", "", CITATION_RE.sub(" ", explanation))
        numbers = unsupported_quantities(body, evidence)
        if numbers:
            warnings.append("Please check these numbers against the document: " + ", ".join(numbers[:6]) + ".")
        # Unreliable values may appear only in the "could not be read" part (the last heading).
        headings = [m.start() for m in re.finditer(r"(?m)^##\s", explanation)]
        stated = explanation[: headings[-1]] if len(headings) >= 2 else explanation
        compact_stated = _compact(stated)
        leaked = list(dict.fromkeys(value for value in unreliable if len(_compact(value)) >= 3 and _compact(value) in compact_stated))
        if leaked:
            warnings.append("The explanation mentions values that were not read reliably: " + "; ".join(leaked[:4]) + ".")
        return warnings

    def stream(self, pages: list[dict], language: str = "english") -> Iterator[dict]:
        language = language if language in LANGUAGE_NAMES else "english"
        text, unreliable = document_text(pages)
        notes = []
        if len(text) > DOCUMENT_CHARS:
            text = text[:DOCUMENT_CHARS]
            notes.append("The document is long; only the first part was used for this explanation.")
        try:
            sources = self._sources(text)
        except Exception:
            LOGGER.exception("Retrieving laws for a document failed")
            sources = []
            notes.append("The law index could not be searched; the legal explanation may be incomplete.")

        block = "\n\n".join(f"{_source_label(i, c)}\n{' '.join(c.text.split())[:3500]}" for i, c in enumerate(sources, 1))
        prompt = [
            {"role": "system", "content": EXPLAIN_PROMPT.format(language=LANGUAGE_NAMES[language])},
            {"role": "user", "content": f"Document text:\n{text}\n\nLegal sources:\n{block or 'none found'}"},
        ]
        yield {"type": "sources", "language": language, "search_query": "", "citations": [
            Citation(title=c.title, source_file=c.source_file, source_id=c.source_id, source_url=c.source_url,
                     chunk_id=c.chunk_id, year=c.year, jurisdiction=c.jurisdiction, document_type=c.document_type,
                     legal_status=c.legal_status, page_start=c.page_start, page_end=c.page_end,
                     section_ref=c.section_ref, score=c.score, excerpt=" ".join(c.text.split())).model_dump()
            for c in sources
        ]}
        answer: list[str] = []
        try:
            try:
                for kind, piece in self.chat.llm.stream(prompt):
                    if kind == "thinking":
                        yield {"type": "thinking", "text": piece}
                    else:
                        answer.append(piece)
                        yield {"type": "token", "text": piece}
            except GenerationLimitReached:
                notes.append("The detailed reasoning step ran too long, so this explanation was written in fast mode; check it carefully.")
                for _, piece in self.chat.llm.stream(prompt, think=False):
                    answer.append(piece)
                    yield {"type": "token", "text": piece}
        except Exception as error:
            LOGGER.exception("Document explanation failed")
            yield {"type": "error", "message": f"The language model is unavailable: {error}"}
            return
        yield {"type": "done", "warnings": notes + self.check("".join(answer), text, sources, unreliable)}
