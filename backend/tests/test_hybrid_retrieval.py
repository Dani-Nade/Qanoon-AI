from qanoon_ai.retrieval.hybrid import _jurisdiction_clause, _rrf


def test_rrf_rewards_chunks_returned_by_both_retrievers():
    dense = [{"chunk_id": "dense-only"}, {"chunk_id": "shared"}]
    lexical = [{"chunk_id": "shared"}, {"chunk_id": "lexical-only"}]

    ranked = _rrf(((1.0, dense), (1.2, lexical)))

    assert ranked[0][1]["chunk_id"] == "shared"
    assert {row["chunk_id"] for _, row in ranked} == {
        "shared",
        "dense-only",
        "lexical-only",
    }


def test_jurisdiction_alias_includes_laws_and_courts():
    clause, values = _jurisdiction_clause("kp")

    assert clause
    assert values == ["khyber_pakhtunkhwa", "courts_khyber_pakhtunkhwa"]


def test_empty_jurisdiction_does_not_add_filter():
    clause, values = _jurisdiction_clause(None)

    assert clause == ""
    assert values == []
