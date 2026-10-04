import json

from qanoon_ai.documents.explain import DocumentExplainer, cited_sections, document_text
from qanoon_ai.retrieval.models import EvidenceChunk

S497 = EvidenceChunk(chunk_id="crpc-497", text="497. When bail may be taken in case of non-bailable offence.", source_file="crpc.pdf",
                     title="THE CODE OF CRIMINAL PROCEDURE, 1898", year="1898", section_ref="Section 497", jurisdiction="federal", score=1.0)


def _page(blocks):
    return {"number": 1, "document_type": "bail petition", "language": "english", "blocks": blocks}


def _block(text, handwritten=False, uncertain=(), edited=False):
    return {"text": text, "handwritten": handwritten, "uncertain": list(uncertain), "edited": edited}


def test_sections_cited_in_court_papers_become_law_queries():
    text = ("FIR NO. 313 DATED 11-05-2016 CHARGE U/S 324/34 PPC P.S LAKKI\n"
            "BAIL PETITION U/S 497 CR.P.C FOR THE RELEASE OF ACCUSED\n"
            "مقدمہ علت نمبر 245/2026 زیر دفعہ 489-F ت پ")
    queries = cited_sections(text)
    assert "section 324 Pakistan Penal Code" in queries
    assert "section 34 Pakistan Penal Code" in queries
    assert "section 497 Code of Criminal Procedure" in queries
    assert "section 489F Pakistan Penal Code" in queries or "section 489 Pakistan Penal Code" in queries


def test_unreliable_values_are_marked_and_corrected_blocks_are_trusted():
    pages = [_page([
        _block("BAIL PETITION NO. 383-B/2016.", handwritten=True),
        _block("FIR No. 313 dated 11-05-2016", uncertain=["11-05-2016"]),
        _block("Date of decision 27.2.2009", handwritten=True, edited=True),
    ])]
    text, unreliable = document_text(pages)
    assert "[HANDWRITTEN, NOT RELIABLE: BAIL PETITION NO. 383-B/2016.]" in text
    assert "[UNCERTAIN: 11-05-2016]" in text
    assert "Date of decision 27.2.2009" in text and "27.2.2009" not in unreliable
    assert set(unreliable) == {"383-B/2016", "11-05-2016"}


class FakeLLM:
    def __init__(self, answer):
        self.answer = answer
        self.prompts = []

    def stream(self, messages, think=None, **kwargs):
        self.prompts.append(messages)
        yield ("content", self.answer)


class FakeChat:
    def __init__(self, answer):
        self.llm = FakeLLM(answer)
        self.queries = None

    def plan(self, messages):
        return {"queries": ["bail in non-bailable offence"]}

    def retrieve(self, queries, jurisdiction):
        self.queries = queries
        return [S497]


def test_explanation_streams_with_law_sources_and_checks_leaks_and_numbers():
    answer = ("## What this document is\nA bail petition (page 1).\n\n## Key details\n- Petition No. 383-B/2016\n- FIR No. 313 of 12-05-2016\n\n"
              "## The law behind it\nBail under section 497 [1].\n\n## Could not be read\n- Petition number")
    chat = FakeChat(answer)
    pages = [_page([_block("BAIL PETITION NO. 383-B/2016.", handwritten=True),
                    _block("FIR NO. 313 DATED 11-05-2016 BAIL PETITION U/S 497 CR.P.C")])]
    events = list(DocumentExplainer(chat).stream(pages, "english"))
    assert [e["type"] for e in events] == ["sources", "token", "done"]
    assert events[0]["citations"][0]["section_ref"] == "Section 497"
    assert chat.queries[0] == "section 497 Code of Criminal Procedure"
    prompt = chat.llm.prompts[0][1]["content"]
    assert "[HANDWRITTEN, NOT RELIABLE" in prompt and "[1] STATUTE | THE CODE OF CRIMINAL PROCEDURE, 1898 | Section 497" in prompt
    warnings = " ".join(events[-1]["warnings"])
    assert "not read reliably" in warnings and "383-B/2016" in warnings
    assert "12-05-2016" in warnings or "12" in warnings


def test_urdu_thirty_days_written_as_three_days_is_flagged_and_citations_are_not_numbers():
    chat = FakeChat("")
    explainer = DocumentExplainer(chat)
    text = "(۲) ذیلی دفعہ (۱) کے تحت تقرر کردہ غیر جانبدار تیس دنوں کے اندر جرم کے تصفیہ کو سہل بنانے کی کوشش کریگا۔"
    warnings = explainer.check("Neutral ko 3 din ke andar koshish karni hogi [7].", text, [], [])
    assert warnings and "3 day" in warnings[0] and "7" not in warnings[0]
    assert explainer.check("Neutral ko 30 din ke andar koshish karni hogi [7].", text, [], []) == []


def test_a_printed_year_shared_with_a_handwritten_number_is_not_a_leak():
    pages = [_page([_block("1st - 586 B/2016", handwritten=True), _block("FIR NO. 313 DATED 11-05-2016")])]
    text, unreliable = document_text(pages)
    assert "2016" not in unreliable and "586" in unreliable
    explanation = "## Key details\n- FIR 313 dated 11-05-2016\n\n## Could not be read\n- 586 B/2016"
    assert DocumentExplainer(FakeChat("")).check(explanation, text, [], unreliable) == []


def test_unknown_language_falls_back_to_english():
    chat = FakeChat("## What this document is\nA notice (page 1).")
    events = list(DocumentExplainer(chat).stream([_page([_block("Notice")])], "klingon"))
    assert events[0]["language"] == "english"
    assert "in English" in chat.llm.prompts[0][0]["content"]
