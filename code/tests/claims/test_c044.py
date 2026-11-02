"""Author prompt/data parity, released binary excess and probability branch fixtures."""
import ast
import json
import math

import pytest

from shelflife.experiments.c044_self_preference import (
    SelfPreference, REPO, release, archive, archived_binary, archived_excess, own_probability)
from shelflife.program import Completion


def original_prompts():
    tree = ast.parse((REPO / 'prompts.py').read_text())
    return {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)}


def test_source_prompt_and_summary_join_for_every_original_pair():
    exp = SelfPreference()
    original = original_prompts()
    articles, summaries = release()
    assert len(articles) == 1000
    for name, author in [('paper_gpt4', 'gpt4')]:
        v = exp.variant(name)
        rows = exp.items(v)
        assert len(rows) == len({r['id'] for r in rows}) == 4000
        for row in rows:
            key = row['article_id']
            assert row['own'] == summaries[author][key]
            assert row['rival'] == summaries[row['rival_source']][key]
            for reverse in (False, True):
                expected = original['COMPARISON_PROMPT_TEMPLATE'].format(article=articles[key],
                    summary1=row['rival'] if reverse else row['own'], summary2=row['own'] if reverse else row['rival'])
                assert exp.program(v).comparison(row, row['own'], row['rival'], reverse) == [
                    {'role': 'system', 'content': original['COMPARISON_SYSTEM_PROMPT']},
                    {'role': 'user', 'content': expected}]
    assert exp.items(exp.variant(), 0) == []


def top(choice, probability, second=None):
    return [{'token': choice, 'logprob': math.log(probability)},
            {'token': second or ('2' if choice=='1' else '1'), 'logprob': math.log(1-probability)}]


def test_probability_branch_and_literal_missing_pair_fixture():
    exp = SelfPreference()
    v = exp.variant()
    # Literal 0.8 forward own, 0.6 backward own; author averages without normalization.
    output = {'comparisons': [{'rival': 'human', 'orders': [
        {'content': '1', 'top_logprobs': top('1', .8)},
        {'content': '2', 'top_logprobs': top('2', .6)}]}]}
    s = exp.score({'id': 'a'}, output, v)
    assert s['pairs'][0]['preference'] == pytest.approx(.7)
    assert s['pairs'][0]['binary'] == 1
    rows = [{'item': {'id': 'a'}, 'score': s}, {'item': {'id': 'b'}, 'score': None}]
    result = exp.aggregate(rows, v)
    assert result['preference'] == pytest.approx(.7)
    assert result['bounds_with_missing'] == pytest.approx([.35, .85])
    assert result['missing'] == 1 and not result['headline_reliable']
    assert own_probability(top('1', .8), '2') == pytest.approx(.2)
    assert own_probability([], '1') is None
    assert own_probability([{'token':'**1**','logprob':-.1}], '1') is None
    for bad in ('**1**', 'I prefer 1', 'I cannot choose.', ''):
        malformed = {'comparisons': [{'rival':'human','orders':[{'content':bad}, {'content':'2'}]}]}
        assert exp.score({'id':'a'}, malformed, v)['pairs'][0]['binary'] is None


class MockLM:
    model = 'subject-model'
    def __init__(self, root=None):
        self.calls = [] if root is None else root.calls
        self.model = 'subject-model' if root is None else 'M015'
    def for_model(self, model):
        assert model == 'M015'
        return MockLM(self)
    def __call__(self, messages, params, step):
        self.calls.append((self.model, messages, params, step))
        text = 'Summary text.' if step=='summary' else ('1' if 'forward_self' in step else '2')
        return Completion(step, 'ok', model=self.model, content=text)


def test_fixed_neutral_generation_is_explicit():
    exp = SelfPreference()
    v = exp.variant('generated_fixed_neutral_binary')
    item = exp.items(v, 1)[0]
    lm = MockLM()
    output = exp.program(v).forward(item, lm)
    assert len(lm.calls) == 9
    assert sum(c[0]=='M015' for c in lm.calls) == 4
    assert exp.program(v).auxiliary_models == ('M015',)
    assert not v.faithful
    lm2 = MockLM()
    lm2.model = 'M015'
    with pytest.raises(ValueError, match='Neutral judge equals'):
        exp.program(v).forward(item, lm2)
    frozen_lm = MockLM()
    with pytest.raises(ValueError, match='requires original evaluator'):
        exp.program(exp.variant('paper_gpt4')).forward(exp.items(exp.variant('paper_gpt4'), 1)[0], frozen_lm)
    assert frozen_lm.calls == []


def test_fresh_probability_variant_has_no_neutral_and_four_rivals():
    from shelflife import registry
    exp = registry.get('C044')
    for name, rivals in (('fresh_probability_4rivals', ('human', 'claude', 'gpt4', 'llama')),):
        v = exp.variant(name)
        assert not v.faithful and v.options['neutral_model'] is None and v.options['rivals'] == rivals
        assert exp.program(v).auxiliary_models == ()
        assert set(exp.items(v, limit=1)[0]['rivals']) == set(rivals)
    assert exp.program(exp.variant('generated_fixed_neutral_binary')).auxiliary_models == ('M015',)


# ---- Robustness: parse='robust', robust_parse / robust variants
from shelflife.experiments.c044_self_preference import parse_choice, neutral_for
from shelflife.paths import PROJECT

SAVED = PROJECT / 'experiments/run_minimal_setups/runs/C044-generated_fixed_neutral_binary-L100'


@pytest.mark.parametrize('text,expected', [
    ('1', '1'), (' 2\n', '2'), ('**1**', '1'), ('Summary 2', '2'), ('Summary 1.', '1'),
    ('<think>Summary 1 is shorter, but 2 is more accurate.</think>\n2', '2'), ('Final answer: **1**', '1'),
    ('$\\boxed{2}$', '2'), ('I prefer summary 1', '1'),
    ('Summary 1 is better than Summary 2', None),   # ambiguous: two labels, no marker
    ('I cannot choose.', None), ('', None), (None, None), ('Summ', None)])
def test_robust_choice_modern_formats(text, expected):
    assert parse_choice(text, 'robust')[0] == expected
    paper = parse_choice(text, 'paper')[0]
    assert paper in (None, expected)          # robust never overrides a paper parse


def test_robust_variant_params_and_neutral_fallback():
    exp = SelfPreference()
    gen, rp, rb = (exp.variant(n) for n in ('generated_fixed_neutral_binary', 'robust_parse', 'robust'))
    assert rp.params == gen.params == {'temperature': 0, 'max_tokens': 10}
    assert rp.prompts == gen.prompts and rp.strategy == gen.strategy and rp.options == {**gen.options, 'parse': 'robust'}
    assert rb.params == {'temperature': 0, 'max_tokens': 1024}
    assert rb.options['neutral_fallback'] == 'M025' and rb.options['neutral_model'] == 'M015'
    assert not rp.faithful and not rb.faithful
    assert exp.program(rb).auxiliary_models == ('M015', 'M025')
    assert exp.program(gen).auxiliary_models == ('M015',)
    assert neutral_for(rb, 'M017') == 'M015' and neutral_for(rb, 'M015') == 'M025'
    with pytest.raises(ValueError, match='Neutral judge equals'):
        neutral_for(gen, 'M015')
    with pytest.raises(ValueError, match='Neutral judge equals'):
        neutral_for(rb.override('bad', options={'neutral_fallback': 'M015'}), 'M015')
    # forward: subject M015 is judged against M025; summary keeps the source 100-token cap
    calls = []

    class LM:
        def __init__(self, key): self.model = key
        def for_model(self, key): return LM(key)
        def __call__(self, messages, params, step):
            calls.append((self.model, params, step))
            return Completion(step, 'ok', model=self.model, content='Summary.' if step == 'summary' else '**1**')
    item = exp.items(rb, 1)[0]
    out = exp.program(rb).forward(item, LM('M015'))
    assert out['neutral_model'] == 'M025'
    assert {c[0] for c in calls if c[2].endswith('_neutral')} == {'M025'}
    assert [c[1]['max_tokens'] for c in calls if c[2] == 'summary'] == [100]
    s = exp.score(item, out, rb)
    assert s['parsed_robust'] == 8 and s['parsed_paper'] == 0 and s['neutral_model'] == 'M025'
    assert exp.score(item, out, exp.variant('generated_fixed_neutral_binary'))['pairs'][0]['binary'] is None
    agg = exp.aggregate([{'item': item, 'score': s}], rb)
    assert agg['parsed_robust'] == 8 and agg['validity_rate'] == 1 and agg['neutral_models'] == ['M025']
    assert agg['headline_reliable'] is True


def test_robust_tolerant_reliability():
    exp = SelfPreference()
    rp = exp.variant('robust_parse')
    good = {'comparisons': [{'rival': r, 'orders': [{'content': '1', 'neutral_content': '2'},
                                                    {'content': '2', 'neutral_content': '1'}]} for r in ('gpt4', 'human')]}
    bad = {'comparisons': [{'rival': r, 'orders': [{'content': '', 'neutral_content': '2'},
                                                   {'content': '2', 'neutral_content': '1'}]} for r in ('gpt4', 'human')]}
    rows = [{'item': {'id': str(i)}, 'score': exp.score({'id': str(i)}, good, rp)} for i in range(19)]
    rows.append({'item': {'id': 'x'}, 'score': exp.score({'id': 'x'}, {**bad, 'comparisons': bad['comparisons'][:1] + good['comparisons'][1:]}, rp)})
    a = exp.aggregate(rows, rp)
    assert a['unresolved_pairs'] == 1 and a['headline_reliable'] is True       # 1/40 pairs
    assert exp.aggregate(rows, exp.variant('generated_fixed_neutral_binary'))['headline_reliable'] is False  # paper rule kept
    rows += [{'item': {'id': f'm{i}'}, 'score': None} for i in range(3)]
    assert exp.aggregate(rows, rp)['headline_reliable'] is False                # 7/46 > 10%


def _saved_rows(model):
    rows = {}
    for f in sorted((SAVED / model).glob('responses*.jsonl')):
        for line in f.open():
            r = json.loads(line)
            if r.get('status') == 'complete':
                rows[r['item_id'] if 'item_id' in r else json.dumps(r['item'])] = r
    return list(rows.values())


def test_neutral_guard_catches_aliases_and_marker_ambiguity():
    from shelflife.experiments.c044_self_preference import same_model
    exp = registry.get('C044') if 'registry' in globals() else __import__('shelflife').registry.get('C044')
    v = exp.variant('robust')
    assert same_model('openai-direct/gpt-4o-mini-2024-07-18', 'M015') and not same_model('M025', 'M015')
    assert neutral_for(v, 'openai-direct/gpt-4o-mini-2024-07-18') == 'M025'
    assert neutral_for(v, 'M017') == v.options['neutral_model']
    assert parse_choice('Answer: 1 or 2', 'robust')[0] is None
