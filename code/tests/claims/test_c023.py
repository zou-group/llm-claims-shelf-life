"""C023 diffuse distributions: prompts against the released prompt files, scorer against the frozen score.py on every
saved output, and rescoring parity with initial-reproduction summaries and the first-pass results.json."""
import json
import math

import pytest

from shelflife import registry
from shelflife.experiments import c023_random_numbers as m
from shelflife.paths import INITIAL, HELPERS
from shelflife.replay import legacy_records
from shelflife.sourcecode import load_functions

exp = registry.get('C023')
PRIOR_SCORE = INITIAL / 'reproductions/C023/score.py'
PANEL_SCORE = HELPERS / 'C023/score.py'


def prior_score():
    return load_functions(PRIOR_SCORE, ['score'])['score']


def _item(condition):
    k = m.K[condition]
    return {'expected': {'min': 1, 'max': k, 'targets': [f'{{{i}}}' for i in range(1, k + 1)]}}


# ------------------------------------------------------------------------------------------- prompts
def test_prompts_are_the_released_requests_for_every_item():
    for name, v in exp.variants.items():
        prog = exp.program(v)
        items = exp.items(v)
        assert len(items) == m.DRAWS * len(v.options['conditions'])
        assert len({x['id'] for x in items}) == len(items)
        for cond in v.options['conditions']:
            src = json.loads((m.REPO / m.CONDITIONS[cond]).read_text())
            msgs = {json.dumps(prog.build(x)) for x in items if x['condition'] == cond}
            assert msgs == {json.dumps([{'role': 'user', 'content': src['request']}])}, name


def test_rng0_prompt_text_and_targets():
    assert exp.variant().prompts['rng-0'].render() == (
        "Generate a random number between 1 and 10. Output only the number between two curly braces, like this: "
        "{number}. Don't output code.")
    assert m._targets('rng-0') == [f'{{{i}}}' for i in range(1, 11)]


EDGE = ['', '   ', '{7}', ' {7}\n', '{07}', '{ 7 }', '{+7}', '7', '{0}', '{11}', '{-3}', '{7.0}', '{7} {8}',
        'I choose {7}', '```\n{7}\n```', '{number}', '{"number": 7}', 'seven', '**{7}**', '{١}', '{7} ',
        "I'm sorry, but I can't generate random numbers", None, 7]


def test_scorer_strictness():
    assert m.parse_number(' {7}\n', 'rng-0')['strict_valid']
    for text in ('{07}', '{ 7 }', '7', 'I choose {7}', '**{7}**', '{11}'):
        assert not m.parse_number(text, 'rng-0')['strict_valid'], text
    assert m.parse_number('{07}', 'rng-0')['semantic_valid'] and not m.parse_number('{11}', 'rng-0')['semantic_valid']


def test_name_parser():
    assert m.parse_name('{Avery}')['value'] == 'Avery'
    assert m.parse_name(" {O'Brien} ")['value'] == "O'Brien"
    for text in ('Avery', '{Avery Smith}', 'My pick: {Avery}', '{}', '', None, '{Avery}{Mia}'):
        assert not m.parse_name(text)['strict_valid'], text


def test_metrics_formulas():
    d = m.distribution(list(range(1, 11)) * 100, 10)
    assert math.isclose(d['entropy_nats'], math.log(10)) and d['coverage'] == 10 and d['tv_to_uniform'] < 1e-12
    d = m.distribution([7] * 999, 10)
    assert d['entropy_nats'] == 0 and d['coverage'] == 1 and math.isclose(d['tv_to_uniform'], 0.9)
    assert d['kl_uniform_to_emp_infinite'] and d['mode_value'] == 7 and d['mode_share'] == 1
    d = m.distribution([5, 5, 8, 8, 1], 10)  # tie: lowest value is the mode, as in the panel analysis
    assert d['mode_value'] == 5
    names = m.distribution(['Avery', 'Mia', 'Avery', 'Liam'], None)
    assert names['coverage'] == 3 and names['mode_value'] == 'Avery'
    assert math.isclose(names['entropy_nats'], -(0.5 * math.log(0.5) + 2 * 0.25 * math.log(0.25)))


def test_aggregate_counts_unknowns_separately():
    v = exp.variant()
    items = exp.items(v, limit=5)
    outs = [{'content': '{7}', 'status': 'ok'}, {'content': 'I choose {3}', 'status': 'ok'},
            {'content': '', 'status': 'empty', 'finish_reason': 'stop'},
            {'content': '{7', 'status': 'truncated', 'finish_reason': 'length'}, None]
    rows = [{'item': x, 'output': o, 'score': None if o is None else exp.score(x, o, v)} for x, o in zip(items, outs)]
    agg = exp.aggregate(rows, v)['rng-0']
    assert (agg['n_valid'], agg['observed_invalid'], agg['unknown']) == (1, 2, 2)
    # paper Eq. 3-4: all 3 observed generations are outcomes (the empty one included); unknowns are not generations
    assert agg['n_generations'] == 3
    assert agg['metrics']['counts'] == {'': 1, 'I choose {3}': 1, '{7}': 1}
    assert agg['valid_only_metrics']['counts'] == {7: 1}


# ------------------------------------------------------------- initial-reproduction rescoring parity
def _check_initial_summary(summary_file, runs):
    """Groups of the initial-reproduction analyze.py summary: attempted, observed, unknown, strict counts, entropy."""
    summary = json.loads((INITIAL / 'reproductions/C023' / summary_file).read_text())
    checked = 0
    for g in summary['groups']:
        run = next(r for r in runs if any(rec['model'] == g['model'] for rec in legacy_records(r)))
        agg = exp.rescore_prior(run, model=g['model'])[g['condition']]
        assert agg['attempted'] == g['attempted']
        assert agg['n_valid'] + agg['observed_invalid'] == g['observed_completed']
        assert agg['unknown'] == g['unknown']
        assert agg['observed_invalid'] == g['invalid_outputs']
        counts = [agg['valid_only_metrics']['counts'].get(i, 0) for i in range(1, len(g['strict_distribution']['counts']) + 1)]
        assert counts == g['strict_distribution']['counts']
        assert math.isclose(agg['valid_only_metrics']['entropy_nats'], g['strict_distribution']['entropy_nats'],
                            abs_tol=1e-12)
        assert agg['valid_only_metrics']['coverage'] == g['strict_distribution']['coverage']
        checked += 1
    return checked


def test_headline_metric_is_paper_eq3_4_over_all_generations():
    """Paper §2.2 Eq. 3-4 (v2 p.3): Entropy(p~) sums over y in V* (all token sequences) of the empirical distribution of
    the generations; Coverage-N = |{y_i}|. Invalid generations are outcomes and coverage is not capped at k. The
    earlier runs' strict-valid-conditional metric is only the labelled diagnostic ``valid_only_metrics``."""
    v = exp.variant()
    texts = ['{7}', ' {7}\n', '{7}', '{3}', '7', 'I choose {3}', '{11}', '{ 7 }', '{1}', '{2}', '{4}', '{5}', '{6}',
             '{8}', '{9}', '{10}']
    items = exp.items(v, limit=len(texts))
    rows = [{'item': x, 'output': {'content': t, 'status': 'ok', 'finish_reason': 'stop'}} for x, t in zip(items, texts)]
    for r in rows:
        r['score'] = exp.score(r['item'], r['output'], v)
    agg = exp.aggregate(rows, v)['rng-0']
    assert agg['metrics']['counts']['{7}'] == 3      # outer whitespace stripped (HF batch_decode is near-canonical)
    assert agg['metrics']['coverage'] == 14 > 10       # uncapped: '7', 'I choose {3}', '{11}', '{ 7 }' are outcomes
    assert agg['valid_only_metrics']['coverage'] == 10
    counts = list(agg['metrics']['counts'].values())
    assert math.isclose(agg['metrics']['entropy_nats'], m.entropy(counts))
    assert agg['metric_definition'].startswith('paper Eq. 3-4')
    assert 'diagnostic' in agg['valid_only_metric_definition']
    assert 'Eq. 3-4' in ' '.join(v.deviations) and 'does not state how invalid' not in ' '.join(v.deviations)


def test_baby_names_paper_metric_counts_multi_token_names():
    """Audit minor 3: '{Mary Ann}' is a generation the paper counts; our provisional parser rejects it."""
    v = exp.variant('baby_names')
    items = exp.items(v, limit=3)
    outs = ['{Mary Ann}', '{Avery}', 'Avery']
    rows = [{'item': x, 'output': {'content': t, 'status': 'ok'}, 'score': exp.score(x, {'content': t}, v)}
            for x, t in zip(items, outs)]
    agg = exp.aggregate(rows, v)['baby-names']
    assert agg['metrics']['coverage'] == 3 and agg['valid_only_metrics']['coverage'] == 1

