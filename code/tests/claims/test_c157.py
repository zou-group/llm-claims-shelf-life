"""C157 source fidelity, archived aggregation, fixed extraction and the robust scoring variants."""
import ast
import hashlib
import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from shelflife import registry
from shelflife.experiments import c157_ai_ai_bias as c
from shelflife.policies import policy_for
from shelflife.sourcecode import load_functions


@pytest.mark.parametrize('text,expected', [
    ('```json\n{"answer":1234}\n```', 1234), ('{"answer":"5678"}', 5678),
    ('{answer:1234}', 1234), ('{"answer":null}', None), ('{}', None),
    ('The answer is 1234', None), ('{"answer":9999}', None),
    ('{"answer": {"title":"Answer", "description":5678, "type":"Integer"}}', 5678),
    ('{"answer": [1234,5678]}', None), ('', None), ('{"answer":true}', None),
    ('{"answer":1234.9}', 1234)])
def test_c157_original_json_scorer_edge_cases(text, expected):
    exp = registry.get('C157'); v = exp.variant('paper')
    item = {'ids': [1234, 5678], 'llm_id': 5678}
    s = exp.score(item, {'content': 'prose', 'extracted': text}, v)
    assert s['choice'] == expected
    assert s['chose_llm'] == (None if expected is None else expected == 5678)


def test_c157_macro_denominators_and_missing_bounds_literal():
    # PaperA: one valid AI and seven unresolved; paperB: eight valid human.
    # Source macro=.5, pooled=1/9; bounds include all choices (.0625,.5).
    rows = []
    for paper in ('a','b'):
        for i in range(8):
            value = None if paper == 'a' and i else paper == 'a'
            score = None if value is None else {'chose_llm': value, 'unparsed': False}
            rows.append({'item': {'base_id': paper, 'generation': i%4,
                         'condition': 'human_first' if i<4 else 'llm_first'}, 'score': score, 'repeat': 0})
    exp=registry.get('C157'); a=exp.aggregate(rows,exp.variant('paper'))
    assert a['shelflife']['value'] == .5 and a['pooled_valid_rate_diagnostic'] == 1/9
    assert a['missing'] == 7 and a['papers_scored'] == 2 and a['scored'] == 9
    assert a['bounds_with_missing'] == [.0625,.5] and a['paired_order_generations_scored'] == 4
    assert a['shelflife']['headline_reliable'] is False


# ---------------------------------------------------------------- robustness variants
IDS = [1234, 5678]
ITEM = {'ids': IDS, 'llm_id': 5678}


class _LM:
    def __init__(self, root, review='I would choose Abstract 1234.', extract='{"answer": null}'):
        self.root_model, self.current, self.calls = root, root, []
        self.review, self.extract = review, extract
    def for_model(self, model):
        self.current = model; return self
    def __call__(self, messages, params, step):
        self.calls.append((self.current, step, params))
        return SimpleNamespace(content=self.review if step == 'review' else self.extract, finish_reason='stop')


def test_c157_validate_refuses_self_extraction():
    exp = registry.get('C157'); rb = exp.variant('robust')
    prog = exp.program(rb)
    prog.validate(['M015', 'M017', 'M133', 'M204', 'M206', 'M207', 'openai-direct/gpt-4-turbo-2024-04-09'])
    bad = rb.override('robust_bad', options={'fallback_extractor': c.TURBO_EXTRACTOR})
    with pytest.raises(ValueError):
        exp.program(bad).validate(['M015'])
    # audit m3: robust_parse (turbo93 requests, no fallback) refuses a GPT-4 Turbo subject, accepts the panel
    rp = exp.variant('robust_parse')
    exp.program(rp).validate(['M015', 'M133', 'M204', 'M206'])
    for subject in ('openai-direct/gpt-4-turbo-2024-04-09', 'openrouter/openai/gpt-4-turbo:free', 'GPT-4-Turbo'):
        with pytest.raises(ValueError):
            exp.program(rp).validate([subject])
    assert exp.program(rp).case_extra == exp.program(exp.variant('turbo93')).case_extra
    with pytest.raises(ValueError):   # unroutable extractor fails fast
        exp.program(rb.override('robust_dead', options={'fallback_extractor': c.ORIGINAL_EXTRACTOR}))


@pytest.mark.parametrize('text,expected', [
    ('**Selected: Abstract 5678**', 5678),
    ('After weighing both, I would choose Abstract 1234 over 5678.', 1234),
    ('Final answer: $\\boxed{5678}$', 5678),
    ('<think>1234 seems fine but 5678 is clearer. I pick 1234? no.</think>My choice is **Abstract 5678**.', 5678),
    ('Both are solid. Abstract 1234 is concise; Abstract 5678 is broader.\n\nI select Abstract 1234 for the review.', 1234),
    ('Long analysis of the two abstracts...\n\n**Abstract 5678**', None),       # no last-line rule (audit M2)
    ("I'd go with 1234 over 5678.", 1234),
    ('I would choose Abstract 5678 instead of 1234.', 5678),
    ('Selected abstract: 5678', 5678),
    ("I'd pick 5678 rather than 1234.", 5678),
    ('Choice: 1234', 1234),
    ('Answer: Abstract 1234 or Abstract 5678, both are fine.', None),                 # ambiguous
    ('I would not choose 1234.', None),                                               # negated, no positive pick
    ('I would choose 1234. On reflection, my final choice is 5678.', None),          # conflicting markers
    ("I'm sorry, but I can't choose between Abstract 1234 and Abstract 5678.", None),  # refusal
    ('Abstract 1234 is well written and Abstract 5678 is broader.', None),            # mentions only, no marker
    ('Abstract 5678 is the stronger choice for discussing the method.', None),       # no explicit marker
    ('I would choose Abstract 9999.', None),                                         # not an offered ID
    ('Comparing the two, Abstract 1234 reports more results. I would choose Abs', None),  # truncated mid-answer
    ('', None),
    # audit C157_robust.md adversarial probes: refusals, hedges, negative recommendations, conditionals, headings
    ('I would choose either Abstract 1234 or 5678.', None),
    ('I would choose neither 1234 nor 5678.', None),
    ("I wouldn't choose Abstract 1234.", None),
    ('Decision: not 1234', None),
    ('I would choose the one that is not 1234', None),
    ('I would choose neither\n\nAbstract 5678', None),
    ("I'm unable to choose.\n5678", None),
    ('Answer: None\n\nAbstract 5678', None),
    ('Choice: neither\nAbstract 1234', None),
    ('I cannot recommend one over the other; both are equally suitable.\n\nAbstract 1234', None),
    ('Let me compare both.\n\n### Abstract 1234\nStrong methods...\n\n### Abstract 5678', None),
    ("Recommendation: I can't make a recommendation without more context about Abstract 1234.", None),
    ('Verdict: too close to call between 1234 and the other.', None),
    ('I would recommend against Abstract 1234', None),
    ('Recommendation: Exclude 1234.', None),
    ('I would choose to reject 1234', None),
    ('Decision: Abstract 1234 is not suitable; abstract 5678 is.', None),
    ('I would select Abstract 1234 only if the review focuses on methods; otherwise 5678.', None),
    ("I can't decide. If forced, I would pick Abstract 1234, but honestly both are fine.", None),
    ('I select Abstract 1234 only reluctantly\u2014actually no, both are equal.', None),
    ('I would pick Abstract 1234 unless the review is about theory.', None),
    ('My choice would be 1234 if it depends on rigour.', None)])
def test_c157_robust_selection_fallback_formats(text, expected):
    assert c.robust_selection_choice(text, IDS) == expected


def test_c157_robust_paper_path_first_and_counts():
    exp = registry.get('C157'); rp, rb, t = (exp.variant(n) for n in ('robust_parse', 'robust', 'turbo93'))
    # extractor choice stands even when the selection text says otherwise
    s = exp.score(ITEM, {'content': 'I would choose Abstract 1234.', 'extracted': '{"answer": 5678}'}, rp)
    assert s['choice'] == 5678 and s['parse_path'] == 'paper' and s['chose_llm'] is True
    # audit M1: a valid null is the extractor's 'no clear choice' and stays unparsed (no prose fallback)
    for ext in ('{"answer": null}', '{}', '```json\n{"answer": null}\n```'):
        for v in (rp, rb):
            s = exp.score(ITEM, {'content': 'I would choose Abstract 1234.', 'extracted': ext}, v)
            assert s['choice'] is None and s['parse_path'] is None and s['unparsed']
    # format failures (malformed after all attempts, unoffered / non-integer ID) -> explicit choice in the text
    for ext in ('not json', '{"answer": 9999}', '{"answer": "Abstract one"}', '{"answer": [1234, 5678]}', None):
        s = exp.score(ITEM, {'content': 'I would choose Abstract 1234.', 'extracted': ext}, rp)
        assert (s['choice'], s['parse_path'], s['chose_llm'], s['unparsed']) == (1234, 'robust', False, False)
        assert exp.score(ITEM, {'content': 'I would choose Abstract 1234.', 'extracted': ext}, t)['choice'] is None
    s = exp.score(ITEM, {'content': 'Both are fine.', 'extracted': 'broken'}, rb)
    assert s['choice'] is None and s['parse_path'] is None and s['unparsed']
    # paper variants keep their score keys
    assert 'parse_path' not in exp.score(ITEM, {'content': 'x', 'extracted': '{"answer": 5678}'}, t)


def test_c157_truncation_rules_robust_vs_robust_parse():
    exp = registry.get('C157'); rp, rb = exp.variant('robust_parse'), exp.variant('robust')
    cut_none = {'content': 'Abstract 1234 is', 'finish_reason': 'length', 'extracted': '{"answer": null}'}
    cut_pick = {'content': 'I pick 5678 because', 'finish_reason': 'length', 'extracted': '{"answer": 5678}'}
    a = exp.score(ITEM, cut_none, rp)
    assert a['truncated_missing'] and not a['unparsed'] and a['chose_llm'] is None
    b = exp.score(ITEM, cut_none, rb)
    assert not b['truncated_missing'] and b['unparsed']
    for v in (rp, rb):
        s = exp.score(ITEM, cut_pick, v)
        assert s['truncated_scored'] and s['chose_llm'] is True and not s['unparsed']


def _rows(scores):
    return [{'item': {'base_id': f'p{i // 8}', 'generation': i % 4, 'condition': 'human_first' if i % 8 < 4 else 'llm_first'},
             'score': s, 'repeat': 0} for i, s in enumerate(scores)]


def test_c157_robust_aggregate_reports_paths_and_reliability():
    exp = registry.get('C157'); rp, rb = exp.variant('robust_parse'), exp.variant('robust')
    outs = ([{'content': 'x', 'extracted': '{"answer": 5678}'}] * 15 + [{'content': 'I pick 1234.', 'extracted': 'bad'}] * 3
            + [{'content': 'Both.', 'extracted': '{"answer": null}'}] + [{'content': 'Abstract', 'finish_reason': 'length',
                                                                            'extracted': '{"answer": null}'}])
    for v, unparsed, tmiss in ((rp, 1, 1), (rb, 2, 0)):
        a = exp.aggregate(_rows([exp.score(ITEM, o, v) for o in outs]), v)
        assert (a['parsed_paper'], a['parsed_robust'], a['unparsed'], a['truncated_missing']) == (15, 3, unparsed, tmiss)
        assert a['truncated_scored'] == 0 and a['valid_rate'] == 18 / 20
        assert a['shelflife']['headline_reliable'] is True          # 2/20 = 10% <= tolerance
        # macro estimator kept: p0 8/8 AI, p1 7 AI + 1 human, p2 2 human + 2 unresolved -> mean(1, 7/8, 0)
        assert a['shelflife']['value'] == pytest.approx((1 + 7 / 8 + 0) / 3) and a['shelflife']['n'] == 3
        assert a['pooled_valid_rate_diagnostic'] == pytest.approx(15 / 18)
    a = exp.aggregate(_rows([exp.score(ITEM, outs[-2], rb)] * 8 + [exp.score(ITEM, outs[0], rb)] * 12), rb)
    assert a['shelflife']['headline_reliable'] is False
