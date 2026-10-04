from qanoon_ai.language.detection import build_retrieval_queries, detect_language


def test_detects_urdu_script():
    detected = detect_language("پاکستان میں چوری کی سزا کیا ہے؟")

    assert detected.language == "urdu"
    assert detected.script == "arabic"


def test_detects_roman_urdu():
    detected = detect_language("Pakistan mein chori ki saza kya hai?")

    assert detected.language == "roman_urdu"


def test_builds_english_retrieval_variant_for_roman_urdu():
    detected = detect_language("Pakistan mein chori ki saza kya hai?")

    variants = build_retrieval_queries("Pakistan mein chori ki saza kya hai?", detected)

    assert any("theft" in variant.lower() for variant in variants)
    assert any("punishment" in variant.lower() for variant in variants)
