import copy
import json
from pathlib import Path

import pytest

from qanoon_ai.llm.answer import validate_answer, validate_points, parse_object
from qanoon_ai.language.detection import normalize_legal_query, detect_language
from qanoon_ai.ingestion.legal_index import PageText, chunk_legal_document
from qanoon_ai.retrieval.sqlite import _is_navigation
from qanoon_ai.core.config import _env_int


SOURCE = 'Whoever commits theft shall be punished with imprisonment which may extend to three years, or with fine or with both.'
VALID = {'supported': True, 'points': [{'text': 'The maximum prison term is three years; a fine or both are also possible.', 'citations': [1], 'evidence': [{'source': 1, 'quote': SOURCE}]}]}


def test_attributed_answer_and_abstention():
    assert validate_answer(VALID, [('Act', SOURCE)], 'english').endswith('[1]')
    assert validate_answer({'supported': False, 'points': []}, [], 'english') == ''


@pytest.mark.parametrize('mutation', ['quote', 'citation', 'no_evidence', 'malformed_point', 'malformed_evidence'])
def test_rejects_fabricated_or_malformed_evidence(mutation):
    value = copy.deepcopy(VALID)
    point = value['points'][0]
    if mutation == 'quote':
        point['evidence'][0]['quote'] = 'The mandatory penalty is twenty years in prison.'
    elif mutation == 'citation':
        point['citations'] = [2]
    elif mutation == 'no_evidence':
        point['evidence'] = []
    elif mutation == 'malformed_point':
        value['points'] = ['claim']
    else:
        point['evidence'] = ['passage']
    with pytest.raises(ValueError):
        validate_answer(value, [('Act', SOURCE)], 'english')


def _answer(quote, source_number=1, citations=None):
    return {'supported': True, 'points': [{'text': 'An explanation of the rule.', 'citations': citations or [source_number],
                                           'evidence': [{'source': source_number, 'quote': quote}]}]}


# Real mismatches between model quotes and Pakistan Code PDF text.
@pytest.mark.parametrize('source, quote', [
    ('(2) The plaint shall contain all 8[material] facts relating to the dispute',
     'The plaint shall contain all material facts relating to the dispute'),
    ('he has been guilty of 2[an offence punishable with death or 3[imprisonment for life or imprisonment for ten years]]:',
     'guilty of an offence punishable with death or imprisonment for life or imprisonment for ten years'),
    ('shall not be excessive ; and the High Court or Court of Session may, in any case, direct',
     'The High Court or Court of Session may, in any case, direct'),
    ('in respect of which 2* * * criminal breach of trust has been committed',
     'in respect of which criminal breach of trust has been committed'),
    ('Theft of a car or other motor vehicles.__ Whoever commits theft of a car',
     'Theft of a car or other motor vehicles. Whoever commits theft of a car'),
    ('(b) prescribe the annual increase in the maintenance. (3) If the Family Court does not prescribe the annual increase',
     'prescribe the annual increase in the maintenance. If the Family Court does not prescribe the annual increase'),
])
def test_quotes_match_despite_editorial_markers_and_punctuation(source, quote):
    assert validate_answer(_answer(quote), [('Act', source)], 'english').endswith('[1]')


@pytest.mark.parametrize('quote', [
    'shall be punished with imprisonment which may extend to one year',       # changed word
    'shall be punished with fine which may extend to three years',            # changed word
    'imprisonment which may extend to three years shall be punished with',    # reordered
    'Whoever commits theft shall be punished with imprisonment for three years',  # paraphrase
    'commits theft shall be punished with imprisonment which may extend to three years, or with fine or with both. Always.',
])
def test_changed_reordered_or_extended_quotes_are_still_rejected(quote):
    with pytest.raises(ValueError):
        validate_answer(_answer(quote), [('Act', SOURCE)], 'english')


def test_plain_numbers_still_count_even_though_clause_labels_are_ignored():
    source = 'shall be punished with imprisonment for a term which may extend to 3 years (a) or with fine'
    with pytest.raises(ValueError):
        validate_answer(_answer('punished with imprisonment for a term which may extend to 7 years'), [('Act', source)], 'english')
    assert validate_answer(_answer('imprisonment for a term which may extend to 3 years or with fine'), [('Act', source)], 'english')


def test_quote_must_match_whole_words():
    with pytest.raises(ValueError):
        validate_answer(_answer('hoever commits theft shall be punished with imprisonment'), [('Act', SOURCE)], 'english')


def test_verbatim_quote_under_wrong_source_number_is_relabelled():
    other = 'Every suit before a Family Court shall be instituted by the presentation of a plaint.'
    value = _answer(SOURCE, source_number=1)
    rendered = validate_answer(value, [('Other Act', other), ('Penal Code', SOURCE)], 'english')
    assert rendered.endswith('[2]')
    assert value['points'][0]['citations'] == [2]
    assert value['points'][0]['evidence'][0]['source'] == 2


def test_relabelling_keeps_other_valid_evidence_for_the_same_citation():
    other = 'Every suit before a Family Court shall be instituted by the presentation of a plaint.'
    value = _answer(other, citations=[1])
    value['points'][0]['evidence'].append({'source': 1, 'quote': SOURCE})
    assert validate_answer(value, [('Family Courts Act', other), ('Penal Code', SOURCE)], 'english').endswith('[1] [2]')


def test_citation_without_any_evidence_is_still_rejected():
    other = 'Every suit before a Family Court shall be instituted by the presentation of a plaint.'
    with pytest.raises(ValueError):
        validate_answer(_answer(SOURCE, citations=[1, 2]), [('Penal Code', SOURCE), ('Family Courts Act', other)], 'english')


INTERIM = 'At any stage of proceedings in a suit for maintenance, the Family Court may pass an interim order for maintenance, whereunder the payment shall be made by the fourteenth of each month.'


def _point(text, quote, source=1):
    return {'text': text, 'citations': [source], 'evidence': [{'source': source, 'quote': quote}]}


@pytest.mark.parametrize('text, language, reason', [
    ('چوری کی سزا ایک سال تک قید ہے، اور یہ قانون کے مطابق ہے', 'urdu', '1 year'),
    ('چوری کی سزا تین سال تک قید یا فیس یا دونوں ہے، اور یہ قانون ہے', 'urdu', 'fee'),
    ('Agar defendant 14 din mein maintenance nahi deta to court kar sakta hai', 'roman_urdu', '14 day'),
])
def test_changed_numbers_and_mistranslations_are_rejected(text, language, reason):
    source = INTERIM if 'maintenance' in text else SOURCE
    quote = INTERIM if source is INTERIM else SOURCE
    with pytest.raises(ValueError, match=reason):
        validate_answer({'supported': True, 'points': [_point(text, quote)]}, [('Act', source)], language)


def test_invalid_points_are_dropped_when_another_point_is_verified():
    value = {'supported': True, 'points': [
        _point('The maximum prison term is three years.', SOURCE),
        _point('The maximum prison term is ten years.', SOURCE),
    ]}
    valid, errors = validate_points(value, [('Act', SOURCE)])
    assert [number for number, *_ in valid] == [1]
    assert errors and errors[0].startswith('point 2 ') and '10 year' in errors[0]
    assert validate_answer(value, [('Act', SOURCE)], 'english') == 'The maximum prison term is three years. [1]'


def test_points_are_labelled_when_federal_and_provincial_sources_are_mixed():
    punjab = 'the Family Court shall, on the date of the first appearance of the defendant, fix interim monthly maintenance'
    sources = [('Family Courts Act; jurisdiction: federal; Section 17A', INTERIM),
               ('Family Courts Act; jurisdiction: punjab; Section 17-A', punjab)]
    value = {'supported': True, 'points': [
        _point('The court may pass an interim maintenance order payable by the fourteenth of each month.', INTERIM, 1),
        _point('The court fixes interim monthly maintenance at the first appearance of the defendant.', punjab, 2),
    ]}
    answer = validate_answer(value, sources, 'english')
    assert '(Federal) [1]' in answer and '(Punjab) [2]' in answer
    single = validate_answer({'supported': True, 'points': [value['points'][0]]}, sources[:1], 'english')
    assert '(Federal)' not in single


def test_answer_client_keeps_points_that_pass_the_model_check_and_counts_dropped():
    from unittest.mock import patch
    from qanoon_ai.llm.answer import LocalAnswerClient

    raw = json.dumps({'supported': True, 'points': [
        _point('The maximum prison term is three years.', SOURCE),
        _point('A court may also impose a fine.', SOURCE),
    ], 'follow_up': ''})
    client = LocalAnswerClient(Path('unused'))
    verdicts = iter(['', 'was judged unsupported: adds nothing'])
    with patch.object(LocalAnswerClient, 'ready', True), \
            patch.object(client, '_generate', return_value=raw), \
            patch.object(client, '_check_point', side_effect=lambda *args: next(verdicts)):
        result = client.answer('What is the punishment for theft?', [('Act; jurisdiction: federal', SOURCE)])
    assert result.supported and result.dropped == 1
    assert result.text == 'The maximum prison term is three years. [1]'


def test_answer_client_retries_with_specific_feedback_when_no_point_survives():
    from unittest.mock import patch
    from qanoon_ai.llm.answer import LocalAnswerClient

    bad = json.dumps({'supported': True, 'points': [_point('The maximum prison term is ten years.', SOURCE)]})
    good = json.dumps({'supported': True, 'points': [_point('The maximum prison term is three years.', SOURCE)]})
    client = LocalAnswerClient(Path('unused'))
    calls = []

    def generate(messages, max_new_tokens=1400):
        calls.append(messages[-1]['content'])
        return bad if len(calls) == 1 else good

    with patch.object(LocalAnswerClient, 'ready', True), patch.object(client, '_generate', side_effect=generate), \
            patch.object(client, '_check_point', return_value=''):
        result = client.answer('What is the punishment for theft?', [('Act', SOURCE)])
    assert result.text == 'The maximum prison term is three years. [1]'
    assert 'point 1 states 10 year' in calls[1] and 'supported=false only if none' in calls[1]


def test_roman_urdu_falls_back_to_a_verified_english_answer():
    from unittest.mock import patch
    from qanoon_ai.llm.answer import LocalAnswerClient

    urdu_script = json.dumps({'supported': True, 'points': [_point('چوری کی سزا تین سال تک قید یا جرمانہ یا دونوں ہے اور یہ قانون ہے', SOURCE)]})
    english = json.dumps({'supported': True, 'points': [_point('The maximum prison term is three years.', SOURCE)]})
    client = LocalAnswerClient(Path('unused'))
    prompts = []

    def generate(messages, max_new_tokens=1400):
        prompts.append(messages[1]['content'])
        return english if 'in English.' in messages[1]['content'] else urdu_script

    with patch.object(LocalAnswerClient, 'ready', True), patch.object(client, '_generate', side_effect=generate), \
            patch.object(client, '_check_point', return_value=''):
        result = client.answer('Chori ki saza kya hai?', [('Act', SOURCE)], language='roman_urdu')
    assert result.language == 'english' and result.text == 'The maximum prison term is three years. [1]'
    assert 'Roman Urdu: Urdu words spelled with English letters' in prompts[0]


@pytest.mark.parametrize('language', ['urdu', 'roman_urdu'])
def test_english_is_rejected_when_another_language_was_requested(language):
    with pytest.raises(ValueError):
        validate_answer(VALID, [('Act', SOURCE)], language)


def test_equivalent_questions_share_retrieval_and_detect_output_language():
    questions = ['What is the punishment for theft in Pakistan?', 'Pakistan mein chori ki saza kya hai?', 'پاکستان میں چوری کی سزا کیا ہے؟']
    terms = [set(normalize_legal_query(question).split()) for question in questions]
    assert terms[0] == terms[1] == terms[2] == {'theft', 'punishment'}
    assert [detect_language(q).language for q in questions] == ['english', 'roman_urdu', 'urdu']


def test_nested_amendment_markers_and_hyphenated_sections():
    chunks = chunk_legal_document([PageText(1, '17. General rule. The rule shall apply.\n5[ 6[17-A. Suit for maintenance. The court shall fix interim maintenance.\n7[17-B. Commission. The court may issue a commission.')])
    assert [c.section_ref for c in chunks] == ['Section 17', 'Section 17-A', 'Section 17-B']
    assert 'interim maintenance' not in chunks[0].text


def test_contents_entry_does_not_replace_operative_bail_rule():
    assert _is_navigation('497. When bail may be taken in case of non-bailable offence')
    assert not _is_navigation('496. In what cases bail to be taken. When any person is detained and is prepared to give bail, he shall be released on bail.')


def test_integer_environment_settings(monkeypatch):
    monkeypatch.setenv('QANOON_ANSWER_MAX_SOURCES', '7')
    assert _env_int('QANOON_ANSWER_MAX_SOURCES', 5) == 7
    monkeypatch.setenv('QANOON_ANSWER_MAX_SOURCES', 'invalid')
    assert _env_int('QANOON_ANSWER_MAX_SOURCES', 5) == 5
