"""C101 (TOY-01) strawberry toy: prompts against the claim record, scorer edge cases, aggregation denominators, the
half-life metric, and the Fu et al. released-output anchors computed with the authors' own function."""
import collections
import csv
import json
import os

import pytest

from shelflife import registry
from shelflife.experiments import c101_strawberry as m
from shelflife.paths import PROJECT
from shelflife.policies import policy_for
from shelflife.sourcecode import load_functions

exp = registry.get('C101')
pytestmark = pytest.mark.skipif(exp.maturity == 'draft' and os.environ.get('SHELFLIFE_INCLUDE_DRAFTS') != '1',
                                reason='draft claim: run `python -m shelflife.cli check C101`')


def _record():
    return json.loads((PROJECT / m.CLAIM_RECORD).read_text())


def test_answers_are_true_letter_counts_and_limit_gives_primary_contrast():
    for x in exp.items(exp.variant()):
        assert x['answer'] == 'strawberry'.count(x['letter'])
    assert [x['condition'] for x in exp.items(exp.variant(), limit=2)] == ['r_toy', 's_control']
    assert m.MINIMAL_LIMIT == 3 and exp.variant().repeats == 10
    assert exp.variant().params == {'max_tokens': 1024}   # no temperature: provider default per the design


def test_repro_models_routable_and_design_needs_no_judge():
    for key in m.REPRO_MODELS:
        assert policy_for(key) is not None
    assert not getattr(exp.program(exp.variant()), 'auxiliary_models', ())
    assert exp.claim_set_key == 'TOY-01' and m.RUN_VARIANT in exp.variants


@pytest.mark.parametrize('text,first,last', [
    ("There are **3** r's in the word strawberry.", 3, 3),
    ("The word \"strawberry\" contains two r's.", 2, 2),
    ('Three.', 3, 3),
    ("The letter 's' appears once in strawberry.", 1, 1),
    ('', None, None),
    ("I can't help with that.", None, None),
    ('someone, anyone', None, None),          # number words are word-bounded
    ('3.5', 3, 3),                            # a decimal's fractional digits are not a separate number
])
def test_design_number_extraction(text, first, last):
    assert m.first_number(text) == first and (m.numbers(text) or [None])[-1] == last


ENUM_R = '1. s\n2. t\n3. r\n4. a\n5. w\n6. b\n7. e\n8. r\n9. r\n10. y\nThere are 3 r\'s.'
ENUM_S = '1. s\n2. t\n3. r\n4. a\n5. w\n6. b\n7. e\n8. r\n9. r\n10. y\nSo there is 1 s.'


@pytest.mark.parametrize('letter,text,expected', [
    # parser probes
    ('r', "The word strawberry has 10 letters, and there are 3 r's.", 3),
    ('r', "Let's count one by one: s-t-r-a-w-b-e-r-r-y. The r appears three times.", 3),
    ('r', 'There\'s one "r" in "straw" and two in "berry", for a total of three.', 3),
    ('r', ENUM_R, 3),
    ('s', ENUM_S, 1),
    ('s', '1. s / 2. t / 3. r / 10. y / So there is 1 s.', 1),
    ('s', 'There is a single s in strawberry.', 1),
    ('s', 'The word strawberry (10 letters) has one s.', 1),
    ('r', 'In strawberry, r appears 3 times (positions 3, 8, 9).', 3),
    ('r', 'Strawberry has 10 letters; 3 of them are r.', 3),
    # ordinary answers
    ('r', "There are **3** r's in the word strawberry.", 3),
    ('r', "The word \"strawberry\" contains two r's.", 2),
    ('r', "There are 2 R's in strawberry.", 2),
    ('r', '**Answer: 3**', 3),
    ('r', '\\boxed{3}', 3),
    ('r', 'Three.', 3),
    ('s', "There is only one 's' in strawberry.", 1),
    ('r', 'The letter "r" appears 3 times in "strawberry."', 3),
    ('r', "There are 2 r's... wait, let me recount: there are 3 r's.", 3),     # last bound count wins
    ('r', "I can't help with that.", None),
    ('s', '', None),
])
def test_stated_count(letter, text, expected):
    assert m.stated_count(text, letter)[0] == expected


def test_score_marks_unparsed_incorrect_and_truncation_unusable():
    v = exp.variant()
    item = exp.items(v)[0]
    s = exp.score(item, {'content': 'I am not sure.'}, v)
    assert s['parsed'] is False and s['correct'] is False and s['usable']
    s = exp.score(item, {'content': 'There are 2 rs.'}, v)
    assert s['answered_two'] and not s['correct']
    s = exp.score(item, {'content': 'Let me spell', 'finish_reason': 'length'}, v)
    assert s['truncated'] and not s['usable']
    assert not exp.score(item, {'content': '   '}, v)['usable']


def _rows(answers, failed=0, finish=None):
    v = exp.variant()
    items = {x['condition']: x for x in exp.items(v)}
    rows = []
    for cond, texts in answers.items():
        for i, t in enumerate(texts):
            out = {'content': t, 'status': 'ok', 'finish_reason': (finish or {}).get(cond, 'stop')}
            rows.append({'item': items[cond], 'output': out, 'score': exp.score(items[cond], out, v), 'repeat': i,
                         'status': 'complete'})
        for j in range(failed):
            rows.append({'item': items[cond], 'output': None, 'score': None, 'repeat': len(texts) + j,
                         'status': 'call_error'})
    return rows


def test_aggregate_shelflife_and_denominators():
    rows = _rows({'r_toy': ['2'] * 8 + ['3'] * 2, 's_control': ['1'] * 10, 'r_wording': ['3'] * 10})
    out = exp.aggregate(rows, exp.variant())
    assert out['per_prompt']['r_toy']['accuracy'] == 0.2 and out['per_prompt']['r_toy']['share_answered_two'] == 0.8
    hl = out['shelflife']
    assert hl['value'] == pytest.approx(0.8) and hl['direction'] == 1 and hl['null_value'] == 0
    assert hl['paper_value'] is None and hl['anchor_model'] == 'openai-direct/gpt-4o-2024-05-13'
    assert hl['n'] == 20 and hl['ci'][0] < 0.8 < hl['ci'][1] and out['persists'] is True
    assert set(hl) >= {'metric', 'value', 'ci', 'n', 'direction', 'null_value', 'paper_value', 'definition'}
    # failed calls stay counted: excluded from the rate, kept in n and in the bounds
    out = exp.aggregate(_rows({'r_toy': ['3'] * 8, 's_control': ['1'] * 8}, failed=2), exp.variant())
    p = out['per_prompt']['r_toy']
    assert (p['n'], p['scored'], p['failed_calls'], p['accuracy']) == (10, 8, 2, 1.0)
    assert p['accuracy_bounds_with_missing'] == [0.8, 1.0]
    assert out['shelflife']['value'] == 0.0 and out['persists'] is False


def test_newcombe_ci_is_not_degenerate_at_the_extremes():
    """10/10 vs 0/10 must not give a zero-width interval."""
    lo, hi = m.newcombe_diff_ci(10, 10, 0, 10)
    assert 0.5 < lo < 0.8 and hi == pytest.approx(1.0)
    lo, hi = m.newcombe_diff_ci(10, 10, 9, 10)            # one sample's difference is not "persists"
    assert lo < 0 < hi


def test_one_misparse_cannot_flip_persists():
    rows = _rows({'r_toy': ["There are 3 r's."] * 9 + ['Strawberry has 10 letters; 3 of them are r.'],
                  's_control': ['There is 1 s.'] * 10})
    out = exp.aggregate(rows, exp.variant())
    assert out['shelflife']['value'] == 0.0 and out['persists'] is False and out['headline_reliable'] is True


def test_enumerating_correct_model_scores_null_not_persists():
    out = exp.aggregate(_rows({'r_toy': [ENUM_R] * 10, 's_control': [ENUM_S] * 10}), exp.variant())
    assert out['shelflife']['value'] == 0.0 and out['persists'] is False
    assert out['diagnostic_first_number_rule_value'] == 1.0     # the literal design rule would read "persists"


def test_single_s_is_correct():
    out = exp.aggregate(_rows({'r_toy': ["3 r's"] * 10, 's_control': ['There is a single s in strawberry.'] * 10}),
                        exp.variant())
    assert out['shelflife']['value'] == 0.0 and out['per_prompt']['s_control']['accuracy'] == 1.0


def test_reliability_flag_is_per_question():
    """r_toy reliable; s_control has 1/10 disagreement between stated and cleaned-last -> flagged (< 0.95)."""
    s = ['There is 1 s.'] * 9 + ["There is 1 s; check: 2 t's? no, 0 z's"]
    out = exp.aggregate(_rows({'r_toy': ["3 r's"] * 10, 's_control': s}), exp.variant())
    assert out['per_prompt']['r_toy']['agreement_stated_vs_cleaned_last'] == 1.0
    assert out['per_prompt']['s_control']['agreement_stated_vs_cleaned_last'] == 0.9
    assert out['headline_reliable'] is False and out['shelflife']['headline_reliable'] is False


def test_truncated_outputs_excluded_and_counted():
    rows = _rows({'r_toy': ['s-t-r-a'] * 3 + ["3 r's"] * 7, 's_control': ['1 s'] * 10}, finish=None)
    for r in rows[:3]:
        r['output']['finish_reason'] = 'length'
        r['score'] = exp.score(r['item'], r['output'], exp.variant())
    p = exp.aggregate(rows, exp.variant())['per_prompt']['r_toy']
    assert (p['truncated'], p['scored'], p['accuracy']) == (3, 7, 1.0)
    assert p['accuracy_bounds_with_missing'] == [0.7, 1.0]


def test_no_live_target_with_an_unmeasured_value():
    assert all(t.check != 'live' for t in exp.targets)


# ------------------------------------------------------------------ Fu et al. released outputs


@pytest.mark.parametrize('letter,text,expected', [
    # earlier pipeline validation (batch3, 2026-10-05): "N letter(s) 'x'" was erased by clean() -> no count
    ('r', 'The word **strawberry** contains **3** letters "r".', 3),
    ('s', 'The word "strawberry" contains one letter \'s\'.', 1),
    ('r', 'There are **3** letter "r"s in the word **strawberry**.', 3),
    ('s', 'It has one letter s.', 1),
    ('s', 'The word **“strawberry”** contains **one** letter **s** (the first letter).', 1),   # M102
    # length statements are still removed
    ('r', 'strawberry is 10 letters long and has 2 rs', 2),
    ('s', 'Strawberry has 10 letters s-t-r-a-w-b-e-r-r-y; s appears once', 1),
    # final-answer handling unchanged (validator example, M066)
    ('r', "The letter r appears 3 times.\n\nThe final answer is: $\\boxed{2}$", 2),
])
def test_stated_count_keeps_letter_bound_counts(letter, text, expected):
    assert m.stated_count(text, letter)[0] == expected


def test_primary_acc_r_toy_and_control_gate():
    # control solved: difference interpretable
    out = exp.aggregate(_rows({'r_toy': ['2'] * 8 + ['3'] * 2, 's_control': ['1'] * 10}), exp.variant())
    hl = out['shelflife']
    assert hl['primary_metric'] == 'acc_r_toy' and hl['primary_value'] == pytest.approx(0.2)
    assert hl['control_accuracy'] == 1.0 and hl['difference_interpretable'] is True and hl['difference_note'] is None
    assert hl['share_r_toy_answered_two'] == pytest.approx(0.8)
    assert hl['scoring_deviations'] == list(m.SCORING_DEVIATIONS)
    # floor model (validator: M025/M049 answer "2 r's" and miscount the control): 0 difference is uninterpretable
    out = exp.aggregate(_rows({'r_toy': ["2 r's"] * 10, 's_control': ["2 s's"] * 10}), exp.variant())
    hl = out['shelflife']
    assert hl['value'] == 0.0 and hl['primary_value'] == 0.0 and hl['difference_interpretable'] is False
    assert 'control accuracy' in hl['difference_note']
    # M090 pattern: control fails ('There are **two** "s"s'), r solved -> negative, flagged
    out = exp.aggregate(_rows({'r_toy': ["3 r's"] * 9 + ["2 r's"],
                               's_control': ['There are **two** "s"s in "strawberry".'] * 9 + ['1 s']}),
                        exp.variant())
    hl = out['shelflife']
    assert hl['value'] == pytest.approx(-0.8) and hl['difference_interpretable'] is False
    assert hl['primary_value'] == pytest.approx(0.9)


# ------------------------------------------------------------------ robustness
ROBUST = exp.variant('robust_parse')


def _robust_score(letter, text, finish='stop'):
    item = next(x for x in exp.items(ROBUST) if x['letter'] == letter)
    return exp.score(item, {'content': text, 'finish_reason': finish}, ROBUST)


@pytest.mark.parametrize('letter,text,expected,path', [
    ('r', "There are **3** r's in strawberry.", 3, 'paper'),               # markdown: paper parser already handles
    ('r', 'There are **three** r\'s.', 3, 'paper'),
    ('r', r'The answer is $\boxed{3}$.', 3, 'paper'),
    ('r', '<think>s-t-r-a-w-b-e-r-r-y, so 3</think>\n\nFinal answer: 3', 3, 'paper'),
    ('s', "Final answer: **1** (one 's').", 1, 'paper'),
    ('r', 'There are no R\'s in the word "strawberry".', 0, 'robust'),    # saved M017 output; paper: unparsed
    ('r', 'There are no Rs in the word "strawberry".', 0, 'robust'),
    ('s', 'The word strawberry does not contain the letter "s".', 0, 'robust'),
    ('r', 'None.', 0, 'robust'),
    ('r', '<think>Let me spell it: s, t, r', None, None),                  # truncated inside thinking: nothing stated
    ('r', "I'm sorry, I can't help with that.", None, None),               # refusal stays unparsed
    ('r', 'None of the other letters repeat, but let me spell it', None, None),
])
def test_robust_parse_modern_formats(letter, text, expected, path):
    s = _robust_score(letter, text)
    assert s['stated'] == expected and s['parse_path'] == path


def test_robust_fallback_only_when_paper_finds_nothing():
    """A paper-parsed answer keeps the paper value under robust, even if the robust helpers would read it otherwise."""
    for text in ["There are 2 r's. Final answer: 3", '1. s\n2. t\nso two r\'s', 'Count: 3 (positions 3, 8, 9)']:
        item = exp.items(ROBUST)[0]
        assert exp.score(item, {'content': text}, ROBUST)['stated'] == exp.score(item, {'content': text}, exp.variant())['stated']


def test_robust_truncated_still_excluded():
    s = _robust_score('r', 'Final Answer: There are **3 r\'s** in the word', finish='length')
    assert s['stated'] == 3 and s['truncated'] and not s['usable']


def _saved_outputs():
    runs = sorted((PROJECT / 'experiments/run_minimal_setups/runs/earlier/runs').glob('C101-toy-L3-*/*/responses.jsonl'))
    out = []
    for f in runs:
        if '_aborted' in str(f):
            continue
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r.get('output'):
                out.append((f.parent.name, r['item_id'], r['output']))
    return out


def test_robust_aggregate_counts_paths_and_tolerant_reliability():
    rows = _rows({'r_toy': ['2'] * 9 + ["There are no r's."], 's_control': ['1'] * 10, 'r_wording': ['3'] * 10})
    for r in rows:
        r['score'] = exp.score(r['item'], r['output'], ROBUST)
    out = exp.aggregate(rows, ROBUST)
    assert (out['parsed_paper'], out['parsed_robust'], out['unparsed']) == (29, 1, 0)
    assert out['headline_reliable'] and out['validity']['valid_rate'] == 1.0
    # one stated-vs-last disagreement flags the paper rule, not the tolerant robust rule
    rows = _rows({'r_toy': ["There are 3 r's (the 3rd, 8th and 9th letters)."] + ['3'] * 9, 's_control': ['1'] * 10})
    assert not exp.aggregate(rows, exp.variant())['headline_reliable']
    for r in rows:
        r['score'] = exp.score(r['item'], r['output'], ROBUST)
    out = exp.aggregate(rows, ROBUST)
    assert out['headline_reliable'] and out['validity']['agreement_rule_reliable'] is False
    # 3 of 20 primary rows missing (failed calls) -> above the 10% tolerance -> not reliable under robust
    rows = _rows({'r_toy': ['3'] * 10, 's_control': ['1'] * 10}, failed=2)
    for r in rows:
        if r['output'] is not None:
            r['score'] = exp.score(r['item'], r['output'], ROBUST)
    out = exp.aggregate(rows, ROBUST)
    assert out['validity']['missing'] == 4 and not out['headline_reliable']
    assert exp.aggregate(rows, exp.variant())['headline_reliable']    # paper rule unchanged
    # faithful variant reports the path counts too, without changing its headline
    paper = exp.aggregate(_rows({'r_toy': ['2'] * 10, 's_control': ['1'] * 10}), exp.variant())
    assert paper['parsed_paper'] == 20 and paper['parsed_robust'] == 0 and 'validity' not in paper


def test_robust_variant_params():
    toy, rp, rb = exp.variant('toy'), exp.variant('robust_parse'), exp.variant('robust')
    assert toy.faithful and not rp.faithful and not rb.faithful
    assert toy.options.get('parse', 'paper') == 'paper' and rp.options['parse'] == rb.options['parse'] == 'robust'
    assert rp.params == toy.params == {'max_tokens': 1024} and rp.prompts == toy.prompts and rp.repeats == toy.repeats
    assert rb.params == {'max_tokens': 2048} and rb.prompts == toy.prompts and rb.repeats == 10


def test_known_paper_misread_is_not_overridden_by_robust():
    """Saved M206 control answer: the paper scorer binds 'three times' (about r) and reads 3; robust only falls back
    when the paper parser finds nothing, so the misread stands under both (documented in the findings)."""
    text = ('The word "strawberry" has **1** letter "s", at the beginning.\n\n(The letters are s-t-r-a-w-b-e-r-r-y. '
            'The "r" appears three times, but "s" appears only once.)')
    item = exp.items(exp.variant())[1]
    assert item['letter'] == 's'
    assert exp.score(item, {'content': text}, exp.variant())['stated'] == 3
    assert exp.score(item, {'content': text}, ROBUST)['stated'] == 3


def test_truncated_answer_scored_under_robust_excluded_under_robust_parse():
    """truncated='score' (robust, 2048 floor): a cut-off answer is scored by the stated-count rule (no count ->
    incorrect) and stays in the rate; robust_parse (truncated='missing'): excluded (not usable)."""
    rp, rb = exp.variant('robust_parse'), exp.variant('robust')
    assert rb.options['truncated'] == 'score' and 'truncated' not in rp.options
    item = next(x for x in exp.items(rb) if x['condition'] == 'r_toy')
    for text, correct in (('s-t-r-a-w-b-e-r-r-y: s (no), t (no), r (yes, 1), a (no), w (no), b (no), e (no), r', False),
                          ('', False), ("There are 3 r's in strawberry, at positions", True)):
        out = {'content': text, 'finish_reason': 'length'}
        s_rp, s_rb = exp.score(item, out, rp), exp.score(item, out, rb)
        assert not s_rp['usable']
        assert s_rb['usable'] and s_rb['truncated_scored'] and s_rb['correct'] is correct
    rows = []
    for cond in ('r_toy', 's_control'):
        it = next(x for x in exp.items(rb) if x['condition'] == cond)
        for k in range(10):
            out = ({'content': 'Let me spell it out: s, t, r', 'finish_reason': 'length'} if cond == 'r_toy' and k < 3
                   else {'content': f"There are {it['answer']} {it['letter']}'s.", 'finish_reason': 'stop'})
            rows.append({'item': it, 'output': out, 'repeat': k})
    agg = exp.aggregate([{**r, 'score': exp.score(r['item'], r['output'], rb)} for r in rows], rb)
    assert agg['per_prompt']['r_toy']['scored'] == 10 and agg['per_prompt']['r_toy']['accuracy'] == pytest.approx(0.7)
    assert agg['truncated_scored'] == 3 and agg['per_prompt']['r_toy']['unparsed'] == 0
    assert agg['headline_reliable'] is True
    rp_agg = exp.aggregate([{**r, 'score': exp.score(r['item'], r['output'], rp)} for r in rows], rp)
    assert rp_agg['per_prompt']['r_toy']['scored'] == 7 and rp_agg['headline_reliable'] is False
