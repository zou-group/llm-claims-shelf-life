"""Independent author-source and archived-output checks for C129."""
import ast
import json
from pathlib import Path
import pytest
from shelflife import registry
from shelflife.experiments.c129_medlitspin import ROOT, FILES, author, archive_rows, parse_score


def test_c129_verbatim_prompts_and_counts():
    exp = registry.get('C129')
    for protocol, filename in FILES.items():
        tree = ast.parse((ROOT / 'code' / filename).read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Evaluator')
        values = {n.targets[0].id: ast.literal_eval(n.value) for n in cls.body if isinstance(n, ast.Assign)
                  and n.targets[0].id in ('BASE_PROMPT', 'QUESTIONS')}
        variant = exp.variant('paper_' + protocol)
        items = exp.items(variant)
        assert len(items) == 300
        assert exp.items(variant, limit=0) == []
        assert len({i['id'] for i in items}) == 300
        for item in items:
            expected = values['BASE_PROMPT'].format(ABSTRACT=item['abstract'], QUESTION=values['QUESTIONS'][item['question']])
            assert exp.program(variant).preview(item) == [{'role': 'user', 'content': expected}]
        assert len(exp.items(exp.variant('benefit_' + protocol))) == 60


@pytest.mark.parametrize('text, expected', [('7', 7), (' 7.5\n', 7.5), ('**7**', 7), ('-1', 1),
    ('2024 results; answer 7', None), ('11 then 7', None), ('I cannot assess', None), ('', None),
    ('{"detection":"yes","interpretation":4}', 4)])
def test_c129_literal_parser_cases(text, expected):
    assert parse_score(text) == expected
    assert parse_score(text, 'joint') == expected


def test_c129_paired_missing_denominator():
    exp = registry.get('C129'); variant = exp.variant('benefit_baseline')
    rows = []
    for pid, condition, content in [('a','no_spin','2'), ('a','spin','8'),
                                     ('b','no_spin','4'), ('b','spin','11')]:
        item = {'id': pid+condition, 'base_id':pid,'condition':condition,'question':'benefit'}
        rows.append({'item':item,'score':exp.score(item,{'content':content},variant),'repeat':0})
    missing = {'item':{'id':'cspin','base_id':'c','condition':'spin','question':'benefit'},'score':None}
    rows.append(missing)
    result = exp.aggregate(rows, variant)
    assert result['benefit']['pairs_intended'] == 3
    assert result['benefit']['pairs_scored'] == 1
    assert result['benefit']['mean_difference'] == 6
    assert result['benefit']['bounds_with_missing'] == [-14/3, 26/3]
    assert result['missing'] == result['unparsed'] == 1
    assert result['headline_reliable'] is False


# ---------------------------------------------------------------- robustness
from shelflife.experiments import c129_medlitspin as c129


@pytest.mark.parametrize('text, expected, path', [
    ('7', 7, 'paper'), ('**8**', 8, 'paper'), ('-1', 1, 'paper'),            # released quirks stay
    ('11 then 7', None, None),                                      # no 'last number' guess
    ('In 2024 the trial ... Final answer: **6**', 6, 'robust'),
    ('In 2024 I would rate it 7 out of 10', 7, 'robust'), ('In 2025, 6/10.', 6, 'robust'),
    ('In 2025: Score: 7. Rating: 8', None, None),                   # two different ratings: ambiguous
    ('2025 trial; on a 0-10 scale my rating is 8', 8, 'robust'),
    ('Of the 120 patients enrolled, benefit is likely. Rating: \\boxed{7}', 7, 'robust'),
    ('<think>2025 data, maybe</think> 2026: I would rate it 12', None, None),      # out of range stays unparsed
    ('I cannot assess this abstract.', None, None), ('', None, None), (None, None, None)])
def test_c129_robust_fallback_modern_formats(text, expected, path):
    assert c129.parse_with_path(text, 'baseline', 'robust') == (expected, path)


def test_c129_robust_variants_params():
    exp = registry.get('C129')
    base, nolp = exp.variant('benefit_baseline'), exp.variant('benefit_baseline_nologprobs')
    rpn, rb = exp.variant('robust_parse_nologprobs'), exp.variant('robust')
    assert base.params == {'temperature': 0, 'max_tokens': 50, 'seed': 42, 'logprobs': True, 'top_logprobs': 20}
    assert nolp.params == {'temperature': 0, 'max_tokens': 50}
    assert rpn.params == nolp.params and rpn.prompts == nolp.prompts and rpn.options['parse'] == 'robust'
    assert rb.params == {'temperature': 0, 'max_tokens': 1024} and rb.options['parse'] == 'robust'
    assert rb.options['min_cap_reasoning'] == c129.ROBUST_REASONING_CAP == 16000
    item = exp.items(rb)[0]
    assert exp.program(rb).case(item)['min_cap_reasoning'] == 16000
    assert exp.program(rb).case_extra == {'min_cap_reasoning': 16000} and exp.program(nolp).case_extra == {}
    assert rb.options['questions'] == ('benefit',) and exp.items(rb) == exp.items(nolp)
    assert 'parse' not in base.options and not rb.faithful


def test_c129_robust_aggregate_counts_and_tolerance():
    exp = registry.get('C129'); v = exp.variant('robust')
    rows = []
    for i in range(10):
        for cond, content in (('no_spin', '2'), ('spin', 'Rating: **8** out of 10' if i else '')):
            item = {'id': f'{i}{cond}', 'base_id': str(i), 'condition': cond, 'question': 'benefit'}
            rows.append({'item': item, 'score': exp.score(item, {'content': content}, v), 'repeat': 0})
    res = exp.aggregate(rows, v)
    assert (res['parsed_paper'], res['parsed_robust'], res['unparsed']) == (19, 0, 1)
    assert res['benefit']['mean_difference'] == 6 and res['headline_reliable'] is True
    assert exp.aggregate(rows, exp.variant('benefit_baseline_nologprobs'))['headline_reliable'] is False


