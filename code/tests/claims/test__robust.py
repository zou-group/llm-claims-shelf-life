"""Tests for shelflife.experiments._robust (shared fallback parsing for modern-model outputs)."""
import pytest

from shelflife.experiments import _robust as R


@pytest.mark.parametrize('text,expected', [
    ('The answer is **-293**.', -293),
    ('Using order of operations:\n\\[\n3+7\\times9=66\n\\]\n\nThe answer is **-293**.', -293),
    ('\\boxed{42}', 42),
    ('Final answer: 1,234', 1234),
    ('<think>maybe 5? no, 7</think>The result is 12', 12),
    ('Step 1: 3+4=7. Step 2: 7*2=14', None),  # conservative default: no marker, several numbers
    ('Answer: −17', -17),
    ('It is 3/4 of the total', 3),  # fractions=False reads a rating-style numerator
    ('no digits here', None),
    ('', None),
    (None, None),
])
def test_extract_number(text, expected):
    v = R.extract_number(text)
    assert (v is None and expected is None) or v == pytest.approx(expected)


@pytest.mark.parametrize('text,expected', [
    ('**B**', 'B'),
    ('(C) because it is the only option that fits', 'C'),
    ('The correct answer is A.', 'A'),
    ('Let me think. Option A is wrong, option B is wrong. Final answer: D', 'D'),
    ('A', 'A'),
    ('B) Paris', 'B'),
    ('Both A and B seem plausible', None),
    ('I cannot answer that.', None),
])
def test_extract_choice_letters(text, expected):
    assert R.extract_choice(text, 'ABCD') == expected


@pytest.mark.parametrize('text,expected', [
    ('**Yes**, it is.', 'Yes'),
    ('no.', 'No'),
    ('After careful thought, the answer is no', 'No'),
    ('Yes and no', None),
])
def test_extract_yes_no(text, expected):
    assert R.extract_yes_no(text) == expected


def test_choice_words_case_insensitive_canonical():
    assert R.extract_choice('my verdict: summary 2', ['1', '2']) == '2'
    assert R.extract_choice('NTA', ['YTA', 'NTA']) == 'NTA'


def test_fallback_keeps_paper_result():
    assert R.fallback('B', R.extract_choice, 'A', 'ABCD') == ('B', 'paper')
    assert R.fallback(None, R.extract_choice, 'answer: A', 'ABCD') == ('A', 'robust')
    assert R.fallback(None, R.extract_choice, 'unsure', 'ABCD') == (None, None)


def test_reliable_and_cap_floor():
    assert R.reliable(100, unparsed=10) and not R.reliable(100, unparsed=8, missing=3) and not R.reliable(0)
    assert R.cap_floor({'max_tokens': 10, 'temperature': 0}) == {'max_tokens': 1024, 'temperature': 0}
    assert R.cap_floor({'max_tokens': 4096})['max_tokens'] == 4096
    assert R.cap_floor({'temperature': 0}) == {'temperature': 0}


def test_truncation_missing_option():
    class V:
        options = {'parse': 'robust', 'truncated': 'score'}
    assert R.truncation_missing({}) and R.truncation_missing({'parse': 'robust'})
    assert not R.truncation_missing({'truncated': 'score'}) and not R.truncation_missing(V())


def test_normalize_strips_markup():
    assert R.normalize('<think>x</think>**Yes** `code`') == 'Yes code'
    assert R.normalize('<think>unterminated reasoning') == ''


@pytest.mark.parametrize('text,expected', [
    ('Answer: A or B', None),
    ('The answer is not A; it is C', 'C'),
    ('A refusal is the safest response here.', None),
    ('I think the best option is (B).', 'B'),
    ('The correct choice is B because it matches.', 'B'),
    ('Option A is tempting, but not correct. B.', 'B'),
])
def test_extract_choice_audit_cases(text, expected):
    assert R.extract_choice(text, 'ABCD') == expected


@pytest.mark.parametrize('text,kw,expected', [
    ('I would rate it 6/10.', {}, 6),
    ('I would rate it 6/10.', {'fractions': True}, 0.6),
    ('Step 1 gives 12, step 2 gives 30 and then we stop', {}, None),
    ('Step 1 gives 12, step 2 gives 30', {'where': 'marker_then_last'}, 30),
    ('There are 4 options and none is safe', {}, 4),
])
def test_extract_number_audit_cases(text, kw, expected):
    v = R.extract_number(text, **kw)
    assert (v is None and expected is None) or v == pytest.approx(expected)
