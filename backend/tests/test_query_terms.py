import pytest

from qanoon_ai.language.detection import normalize_legal_query
from qanoon_ai.retrieval.sqlite import _query_groups, stem


@pytest.mark.parametrize("words", [
    ("cheating", "cheats", "cheated", "cheat"),
    ("punishment", "punished", "punishable", "punish"),
    ("gives", "give", "giving"),
    ("dishonoured", "dishonour"),
    ("committed", "commit"),
    ("married", "marry"),
])
def test_word_forms_share_a_stem(words):
    assert len({stem(word) for word in words}) == 1


@pytest.mark.parametrize("word", ["family", "passes", "age", "fir"])
def test_stemming_does_not_mangle_short_or_special_words(word):
    assert stem(word) in {"family", "pass", "age", "fir"}


def test_everyday_terms_expand_to_statute_wording():
    groups = _query_groups("What happens if a cheque bounces?")
    assert any({"dishonour", "bounc"} <= group for group in groups)
    assert not any("happen" in group for group in groups)


def test_talaq_and_nikah_keep_the_statutory_word():
    assert {"talaq", "divorce"} <= set(normalize_legal_query("Talaq kitne din baad muassar hoti hai?").split())
    assert {"talaq", "divorce"} <= set(normalize_legal_query("طلاق کتنے دن بعد مؤثر ہوتی ہے؟").split())


def test_urdu_terms_replace_whole_words_only():
    assert "days" not in normalize_legal_query("دنیا")
    assert "days" in normalize_legal_query("کتنے دن")


@pytest.mark.parametrize("text", [
    "agr mene kisi ko talaq deni ho tou kese dounga?",
    "bhai mjhe zmanat kese milegi",
    "kia police bina warrant giraftar kr skti he",
])
def test_informal_roman_urdu_is_detected_and_normalized(text):
    from qanoon_ai.language.detection import detect_language
    assert detect_language(text).language == "roman_urdu"
    normalized = normalize_legal_query(text).split()
    assert not {"agr", "mene", "kese", "tou", "dounga", "mjhe", "kr", "skti"} & set(normalized)


def test_english_with_short_words_is_not_mistaken_for_roman_urdu():
    from qanoon_ai.language.detection import detect_language
    assert detect_language("What happens if he gives me a cheque that bounces?").language == "english"


def test_informal_talaq_question_keeps_the_legal_term():
    assert "talaq" in normalize_legal_query("agr mene kisi ko tlaq deni ho tou kese dounga?").split()


def test_province_names_are_not_search_terms():
    assert "punjab" not in normalize_legal_query("What is the punishment for marrying a child in Punjab?")
