from dataclasses import replace
import json

from qanoon_ai.chat.service import NOT_COVERED, ChatService, _round_robin
from qanoon_ai.core.config import settings
from qanoon_ai.retrieval.models import EvidenceChunk

THEFT = EvidenceChunk(
    chunk_id="ppc-379", text="379. Punishment for theft. Whoever commits theft shall be punished with imprisonment of either description for a term which may extend to three years, or with fine or with both.",
    source_file="ppc.pdf", title="THE PAKISTAN PENAL CODE", year="1860", section_ref="Section 379", jurisdiction="federal", score=1.0,
)


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = []

    def search(self, query, *, top_k, jurisdiction=None, semantic_query=None):
        self.calls.append((query, semantic_query, jurisdiction))
        return self.chunks


class FakeLLM:
    def __init__(self, answer, queries=("punishment for theft of a car",), language="english", needs_law=True):
        self.answer = answer
        self.plan = {"language": language, "needs_law": needs_law, "queries": list(queries)}
        self.streamed = []
        self.planned = []

    def complete(self, messages, **kwargs):
        self.planned.append(messages)
        return json.dumps(self.plan)

    def stream(self, messages, **kwargs):
        self.streamed.append(messages)
        yield from [("content", self.answer[: len(self.answer) // 2]), ("content", self.answer[len(self.answer) // 2:])]

    def status(self):
        return {"reachable": True, "model_available": True}


def _service(answer, chunks=(THEFT,), **plan):
    retriever, llm = FakeRetriever(list(chunks)), FakeLLM(answer, **plan)
    return ChatService(replace(settings, vector_search=False), retriever=retriever, llm=llm), retriever, llm


def _chunk(chunk_id):
    return EvidenceChunk(chunk_id=chunk_id, text="text", source_file=f"{chunk_id}.pdf", title=chunk_id, year=None, score=1.0)


def test_stream_sends_sources_then_tokens_then_done():
    service, retriever, llm = _service("Theft is punishable with up to three years in prison, a fine, or both [1].")
    events = list(service.stream([{"role": "user", "content": "What is the punishment for theft?"}]))
    assert [e["type"] for e in events] == ["sources", "token", "token", "done"]
    assert events[0]["citations"][0]["section_ref"] == "Section 379"
    assert "".join(e["text"] for e in events if e["type"] == "token").endswith("[1].")
    assert events[-1]["warnings"] == []
    prompt = llm.streamed[0][-1]["content"]
    assert "[1] STATUTE | THE PAKISTAN PENAL CODE | Section 379" in prompt and "three years" in prompt


def test_judgments_are_labelled_and_their_paragraph_numbers_are_not_shown_as_sections():
    judgment = EvidenceChunk(chunk_id="lhc", text="138. Dishonour of cheque ...", source_file="2023LHC2938.pdf", title="2023LHC2938",
                             year="2023", section_ref="Section 138", jurisdiction="courts_punjab", document_type="judgment", score=0.5)
    service, _, llm = _service("Answer [1].", chunks=(THEFT, judgment))
    list(service.stream([{"role": "user", "content": "cheque bounce"}]))
    prompt = llm.streamed[0][-1]["content"]
    assert "[2] COURT JUDGMENT | 2023LHC2938 | punjab" in prompt and "Section 138" not in prompt.split("[2]")[1].splitlines()[0]


def test_follow_up_is_rewritten_for_search_and_history_is_sent_without_old_citation_numbers():
    service, retriever, llm = _service("A car theft is dealt with under another section [1].")
    messages = [
        {"role": "user", "content": "What is the punishment for theft?"},
        {"role": "assistant", "content": "Up to three years [1]."},
        {"role": "user", "content": "And what if it's a car?"},
    ]
    events = list(service.stream(messages))
    assert events[0]["search_query"] == "punishment for theft of a car"
    assert retriever.calls[0][1] == "punishment for theft of a car"
    history = llm.streamed[0][1:-1]
    assert history == [{"role": "user", "content": "What is the punishment for theft?"}, {"role": "assistant", "content": "Up to three years."}]


def test_answer_language_follows_the_planned_language():
    service, _, llm = _service("Chori ki saza teen saal tak qaid hai [1].", language="roman_urdu")
    list(service.stream([{"role": "user", "content": "agr koi chori kre tou kya hota he"}]))
    assert "Roman Urdu" in llm.streamed[0][0]["content"]


def test_urdu_script_is_detected_even_if_the_plan_says_otherwise():
    service, _, llm = _service("جواب [1]", language="english")
    events = list(service.stream([{"role": "user", "content": "چوری کی سزا کیا ہے؟"}]))
    assert events[0]["language"] == "urdu"


def test_informal_question_is_searched_with_every_planned_query():
    queries = ["procedure for pronouncing talaq", "notice of talaq to chairman Union Council", "when talaq becomes effective"]
    service, retriever, _ = _service("Answer [1].", queries=queries, language="roman_urdu")
    events = list(service.stream([{"role": "user", "content": "agr mene kisi ko talaq deni ho tou kese dounga?"}]))
    assert [call[1] for call in retriever.calls] == queries
    assert events[0]["search_query"] == "; ".join(queries)


def test_invalid_plan_falls_back_to_the_message_itself():
    service, retriever, llm = _service("Answer [1].")
    llm.complete = lambda messages, **kwargs: "not json"
    events = list(service.stream([{"role": "user", "content": "What is the punishment for theft?"}]))
    assert retriever.calls[0][1] == "What is the punishment for theft?" and events[-1]["type"] == "done"


def test_small_talk_skips_retrieval():
    service, retriever, _ = _service("Hello!", needs_law=False)
    events = list(service.stream([{"role": "user", "content": "hi"}]))
    assert retriever.calls == [] and events[0]["citations"] == []


def test_round_robin_gives_each_query_a_source_before_seconds():
    a, b, c = [_chunk("a1"), _chunk("a2"), _chunk("a3")], [_chunk("b1"), _chunk("a1")], [_chunk("c1")]
    assert [x.chunk_id for x in _round_robin([a, b, c], 5)] == ["a1", "b1", "c1", "a2", "a3"]


def test_checks_warn_without_altering_the_answer():
    service, _, _ = _service("")
    answer = "Theft is punishable with up to seven years [1]. See also [4]."
    warnings = service.check_answer(answer, [THEFT], "english")
    assert any("does not exist" in w for w in warnings)
    assert any("7 year" in w for w in warnings)
    assert service.check_answer("Up to three years, or a fine, or both [1].", [THEFT], "english") == []
    assert service.check_answer(NOT_COVERED["english"] + " The sources cover theft.", [THEFT], "english") == []
    steps = "1. Report the theft [1].\n2. The penalty is up to three years [1].\n3. A fine may also apply [1]."
    assert service.check_answer(steps, [THEFT], "english") == []


def test_thinking_is_announced_and_a_runaway_reasoning_step_falls_back_to_fast_mode():
    from qanoon_ai.llm.ollama import GenerationLimitReached

    class ThinkingLLM(FakeLLM):
        def stream(self, messages, think=None, **kwargs):
            self.streamed.append(think)
            if think is None:
                yield ("thinking", "")
                raise GenerationLimitReached("limit")
            yield ("content", "Up to three years, or a fine, or both [1].")

    llm = ThinkingLLM("")
    service = ChatService(replace(settings, vector_search=False), retriever=FakeRetriever([THEFT]), llm=llm)
    events = list(service.stream([{"role": "user", "content": "What is the punishment for theft?"}]))
    assert [e["type"] for e in events] == ["sources", "status", "token", "done"]
    assert llm.streamed == [None, False]
    assert any("fast mode" in w for w in events[-1]["warnings"])


def test_no_sources_still_answers_conversationally():
    service, _, llm = _service("Hello! Ask me a question about Pakistani law.", chunks=())
    events = list(service.stream([{"role": "user", "content": "what about my case"}]))
    assert events[0]["citations"] == [] and events[-1]["type"] == "done"
    assert "none were found" in llm.streamed[0][-1]["content"]


def test_source_tail_reaches_the_model_and_matches_the_citation_panel():
    source = replace(THEFT, text="The operative provision. " * 170 + "Provided that this exception applies.")
    service, _, llm = _service("Answer [1].", chunks=(source,))
    events = list(service.stream([{"role": "user", "content": "Explain the exception"}]))
    excerpt = events[0]["citations"][0]["excerpt"]
    assert len(excerpt) > 3500
    assert excerpt in llm.streamed[0][-1]["content"]
    assert excerpt.endswith("Provided that this exception applies.")


def test_context_budget_omits_whole_sources_and_keeps_numbering_consistent():
    first = replace(THEFT, text="First provision. " * 200 + "FIRST_EXCEPTION")
    second = replace(THEFT, chunk_id="second", text="Second provision. " * 10000 + "SECOND_EXCEPTION")
    service, _, llm = _service("Answer [1].", chunks=(first, second))
    events = list(service.stream([{"role": "user", "content": "Explain"}]))
    assert [c["chunk_id"] for c in events[0]["citations"]] == [first.chunk_id]
    prompt = llm.streamed[0]
    assert first.text in prompt[-1]["content"]
    assert "Second provision" not in prompt[-1]["content"]
    assert events[-1]["warnings"]
    size = sum(len(m["content"].encode("utf-8")) + 64 for m in prompt)
    assert size + service.settings.chat_max_tokens + 512 <= service.settings.chat_num_ctx


def test_oversized_source_abstains_instead_of_generating_from_its_prefix():
    source = replace(THEFT, text="Long provision. " * 10000)
    service, _, llm = _service("Should never be generated", chunks=(source,))
    events = list(service.stream([{"role": "user", "content": "Explain"}]))
    assert events[0]["citations"] == []
    assert events[1]["text"] == NOT_COVERED["english"]
    assert llm.streamed == []


def test_grouped_citations_check_numbers_and_support():
    service, _, _ = _service("")
    warnings = service.check_answer("Up to seven years [1, 99].", [THEFT], "english")
    assert any("does not exist" in warning for warning in warnings)
    assert any("7 year" in warning for warning in warnings)
    assert service.check_answer("Up to three years [1, 2].", [THEFT, THEFT], "english") == []
