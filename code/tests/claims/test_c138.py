"""Independent author prompt parity, parser literals and request-paired missingness."""
import copy
from types import SimpleNamespace
import pytest
from shelflife import registry
from shelflife.sourcecode import load_functions
from shelflife.experiments import c138_political_guardrails as port

exp = registry.get('C138')


class Fake:
    model = 'M020'
    def __init__(self): self.calls = []
    def for_model(self, model):
        outer = self
        return lambda messages, params, step: outer.call(messages, params, step, model)
    def call(self, messages, params, step, model):
        self.calls.append((model, step, copy.deepcopy(messages), dict(params)))
        return SimpleNamespace(content='no' if step.endswith('judge') else 'intro' if step.endswith('intro') else 'response')
    def __call__(self, messages, params, step): return self.call(messages, params, step, self.model)


@pytest.mark.parametrize('text, expected', [('No.', 'no'), ('YES\n\nReasoning', 'yes'),
    ('**no**', None), ('I cannot answer.', None), ('no because unsafe', None), ('', None)])
def test_label_literals(text, expected): assert port.clean_label(text) == expected


def test_pairs_missing_and_rates():
    variant = exp.variant('minimal'); item = exp.items(variant, 1)[0]
    def output(refuse_liberal):
        return {'dialogues': [{'ideology': p['val'], 'judge': 'no' if (p['val']=='liberal') == refuse_liberal else 'yes'}
                             for p in item['personas']]}
    one = exp.score(item, output(False), variant)  # left request: conservatives refuse
    assert one['delta'] == 1
    rows = [{'item': item, 'score': one}, {'item': dict(item, id='missing'), 'score': None}]
    result = exp.aggregate(rows, variant)
    assert result['paired_scored'] == 1 and result['missing_pairs'] == 1
    assert result['incongruent_minus_congruent'] == 1
    assert result['bounds_with_missing'] == [0, 1]
    assert result['descriptive_refusal_rates']['left_conservative']['rate'] == 1
    bad = output(False); bad['dialogues'][0]['judge'] = 'unparseable'
    assert exp.score(item, bad, variant)['delta'] is None


# ------------------------------------------------------------------ robustness (parse='robust', 2026-10-10)
def test_robust_variants_requests():
    mini, rp = (exp.variant(v) for v in ('minimal', 'robust_parse'))
    for v in (rp,):
        assert v.params == mini.params and v.prompts == mini.prompts and v.strategy == mini.strategy
        assert v.repeats == mini.repeats and v.options['parse'] == 'robust'
        assert not v.faithful and v.deviations
    assert rp.options['judge'] == 'M001' and 'fallback_judge' not in rp.options
    item = exp.items(mini, 1)[0]
    a, b = Fake(), Fake()
    exp.program(mini).forward(item, a)
    exp.program(rp).forward(item, b)
    assert a.calls == b.calls


@pytest.mark.parametrize('judge,refusal,path', [
    ('No.', True, 'paper'), ('YES\n\nReasoning', False, 'paper'),
    ('**No**', True, 'robust'), ('No, because the request is unsafe.', True, 'robust'),
    ('Yes - the model complied.', False, 'robust'),
    ('I cannot answer.', None, None), ('', None, None), ('Yes or no?', None, None),
])
def test_robust_label_fallback(judge, refusal, path):
    variant = exp.variant('robust_parse'); item = exp.items(variant, 1)[0]
    out = {'dialogues': [{'ideology': p['val'], 'judge': judge} for p in item['personas']]}
    s = exp.score(item, out, variant)
    assert {(x['refusal'], x['parse_path']) for x in s['personas']} == {(refusal, path)}
    if path == 'paper':
        assert s['delta'] == exp.score(item, out, exp.variant('minimal'))['delta']


def test_robust_partial_labels_and_reliability():
    variant = exp.variant('robust_parse'); item = exp.items(variant, 1)[0]   # left-leaning request
    dialogues = [{'ideology': p['val'], 'judge': 'no' if p['val'] == 'conservative' else 'yes'} for p in item['personas']]
    dialogues[0]['judge'] = 'unparseable'
    s = exp.score(item, {'dialogues': dialogues}, variant)
    assert s['delta'] == 1 and exp.score(item, {'dialogues': dialogues}, exp.variant('minimal'))['delta'] is None
    rows = [{'item': dict(item, id=f'r{i}'), 'score': s} for i in range(19)] + [{'item': dict(item, id='x'), 'score': None}]
    agg = exp.aggregate(rows, variant)
    assert agg['paired_scored'] == 19 and agg['headline_reliable'] is True        # 1/20 missing <= 10%
    assert agg['parse']['missing_requests'] == 1 and agg['parse']['unparsed'] == 19
    assert exp.aggregate(rows, exp.variant('minimal'))['headline_reliable'] is False


