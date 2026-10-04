from qanoon_ai.schemas.source import Citation
from qanoon_ai.verification.citations import verify_citations


def test_verifier_rejects_missing_citations():
    result = verify_citations([])

    assert not result.ok


def test_verifier_accepts_basic_citation():
    result = verify_citations(
        [
            Citation(
                title="Example Act",
                source_file="example.pdf",
                chunk_id="example_0",
                excerpt="Section 1. Example text.",
            )
        ]
    )

    assert result.ok
