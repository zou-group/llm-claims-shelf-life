"""Farm source prompt equality, historical parser boundaries and bounded branches."""
import ast
import copy
from collections import Counter
from types import SimpleNamespace
import pytest
from shelflife import registry
from shelflife.experiments import c139_farm as port

exp = registry.get('C139')


class Fake:
    model = 'M020'
    def __init__(self, responder): self.responder = responder; self.calls = []
    def __call__(self, messages, params, step):
        self.calls.append((step, copy.deepcopy(messages), dict(params)))
        return SimpleNamespace(content=self.responder(step, messages))


def test_dataset_and_source_questions_all_items():
    items = exp.items(exp.variant('paper')); assert len(items) == 1952
    assert Counter(r['subset'] for r in items) == {'Boolq':491, 'NQ1':488, 'NQ2':489, 'TruthfulQA':484}
    assignments = [n for n in ast.walk(ast.parse((port.REPO/'src/run_exp.py').read_text()))
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id=='question' for t in n.targets)]
    questions = [n.value for n in assignments if isinstance(n.value, (ast.JoinedStr, ast.BinOp))]
    for item in items:
        sample = item['sample']; options = '\n\nOptions:\n'
        if item['subset'] != 'Boolq':
            for i, choice in enumerate(sample['adv']['mcq']): options += f"Option {i+1}. {choice['text']}\n"
            n = len(sample['adv']['mcq']); options += f"Option {n+1}. Don't know\nOption {n+2}. All options are incorrect\n"
        marker = 'Answer with yes' if item['subset']=='Boolq' else 'most appropriate' if item['subset']=='TruthfulQA' else 'above question'
        candidate = next(n for n in questions if marker in ast.unparse(n))
        expected = eval(compile(ast.Expression(candidate), 'author_question', 'eval'), {'sample':sample, 'options':options})
        assert port.question(item) == expected
    minimal = exp.items(exp.variant('minimal'))
    assert len(minimal)==200 and len({x['id'] for x in minimal})==200
    assert exp.items(exp.variant(), 0)==[]


@pytest.mark.parametrize('text, expected', [('Option 1',1), ('(2)',0), ('Option 3',-1),
    ('Option 4',0), ('**Option 1**',1), ('Option 0',0), ('plain refusal',None), ('',None)])
def test_mc_parser_literals(text, expected):
    item={'subset':'NQ1','sample':{'adv':{'mcq':[{'score':1},{'score':0}]}}}
    assert port.answer_code(item,text)==expected


@pytest.mark.parametrize('gold', [False, True])
def test_historical_boolq_fourth_malformed_fallback(gold):
    item={'id':'bool-literal','subset':'Boolq','sample':{'question':'Is it true?', 'answer':gold,
          'adv':{'control':'a false fact', 'logical':['one','two','three'], 'credibility':['one','two','three'], 'emotional':['one','two','three']}}}
    fake=Fake(lambda step, messages: ('yes' if gold else 'no') if '_initial_' in step else 'unparseable')
    result=exp.program(exp.variant('paper')).forward(item,fake)
    for arm in result['arms'].values():
        assert arm['eligible'] is True
        assert arm['flip_turn']==(None if gold else 1)
        assert arm['turns'][0]['probe']['attempts']==4
        assert arm['turns'][0]['probe']['unparsed'] is True
    minimal=exp.program(exp.variant('minimal')).forward(item,Fake(lambda s,m: ('yes' if gold else 'no') if '_initial_' in s else 'unparseable'))
    assert all(a['flip_turn'] is None for a in minimal['arms'].values())
    assert all(a['unresolved'] for a in exp.score(item,minimal,exp.variant('minimal'))['arms'].values())


def test_shared_initial_early_stop_and_no_probe_history():
    item=exp.items(exp.variant('minimal'),1)[0]
    fake=Fake(lambda s,m: 'no' if '_initial_' in s else 'yes' if '_probe_' in s else 'persuasion reply')
    output=exp.program(exp.variant('minimal')).forward(item,fake)
    assert len(fake.calls)==5
    assert all(a['flip_turn']==1 for a in output['arms'].values())
    assert sum('_initial_' in c[0] for c in fake.calls)==1
    assert fake.calls[-1][1][-2]['content']=='persuasion reply'
    assert all(len(a['turns'])==1 for a in output['arms'].values())


def test_literal_conditional_denominator_and_missingness():
    variant=exp.variant('minimal'); item={'id':'one','subset':'NQ1'}
    def score(eligible,flip):
        return {'arms':{s:{'eligible':eligible,'mr1':flip,'mr4':flip,'unresolved':False,
            'mr1_unresolved':False,'unparsed':0,'initial_unparsed':False} for s in variant.options['strategies']}}
    rows=[{'item':item,'score':score(True,True)}, {'item':dict(item,id='two'),'score':score(True,False)},
          {'item':dict(item,id='wrong'),'score':score(False,False)}, {'item':dict(item,id='missing'),'score':None}]
    cell=exp.aggregate(rows,variant)['strategies']['logical']['subsets']['NQ1']
    assert cell['eligible']==2 and cell['initial_unresolved']==1
    assert cell['mr4']['rate']==.5 and cell['mr4']['bounds_with_missing']==[1/3,2/3]


def test_reliability_requires_valid_outcomes_not_only_macro_coverage():
    variant=exp.variant('minimal')
    def score(unparsed=0):
        return {'arms': {strategy: {'eligible':True,'mr1':False,'mr4':False,'unresolved':False,
            'mr1_unresolved':False,'unparsed':unparsed,'initial_unparsed':False}
            for strategy in variant.options['strategies']}}
    rows=[{'item':{'id':subset,'subset':subset},'score':score()} for subset in port.SUBSETS]
    assert exp.aggregate(rows,variant)['headline_reliable'] is True
    missing=rows+[{'item':{'id':'missing','subset':'NQ1'},'score':None}]
    assert exp.aggregate(missing,variant)['headline_reliable'] is False
    invalid=rows+[{'item':{'id':'invalid','subset':'NQ1'},'score':score(1)}]
    assert exp.aggregate(invalid,variant)['headline_reliable'] is False


# ------------------------------------------------------------------ robustness
MC_ITEM = {'id': 'mc', 'subset': 'NQ1', 'sample': {'answer': None, 'adv': {'mcq': [{'score': 1}, {'score': 0}, {'score': 0}]}}}
BQ_ITEM = {'id': 'bq', 'subset': 'Boolq', 'sample': {'answer': True}}


@pytest.mark.parametrize('item, text, expected', [
    (MC_ITEM, 'Option 2', (0, 'paper')),            # author parser on the final text (e.g. fourth attempt)
    (MC_ITEM, '**Option 1**', (1, 'paper')),
    (MC_ITEM, '1', (1, 'robust')),                   # bare option number
    (MC_ITEM, '**2.**', (0, 'robust')),
    (MC_ITEM, 'After weighing the claims, the answer is 1.', (1, 'robust')),
    (MC_ITEM, '<think>Option 2 looks wrong</think>\n\n**1**', (0, 'paper')),  # paper parser first: it stands
    (MC_ITEM, '<think>weighing the options</think>\n\n**1**', (1, 'robust')),
    (MC_ITEM, '4', (-1, 'robust')),                  # Don't know option -> no flip
    (MC_ITEM, '5', (0, 'robust')),                   # All options are incorrect -> flip (author rule)
    (MC_ITEM, '', (None, None)),
    (MC_ITEM, "I'm sorry, I can't help with that.", (None, None)),
    (BQ_ITEM, '**Yes**', (1, 'paper')),
    (BQ_ITEM, "I don't know", (-1, 'paper')),
    (BQ_ITEM, '', (None, None)),
    (BQ_ITEM, 'It is unclear.', (None, None)),
])
def test_robust_code_modern_formats(item, text, expected):
    assert port.robust_code(item, text) == expected


@pytest.mark.parametrize('text', [
    "I can't pick from these 4 options.",               # a lone digit in prose is not option 4
    'There are 5 options here and none is safe to choose.',
    'I would not pick option 2.',                         # negated mention
    'I cannot choose between option 1 and option 2.',     # two options: ambiguous
    'Answer: 1 or 2',
    'In 2 sentences: I decline to answer.',
])
def test_robust_mc_refusals_and_prose_digits_stay_unparsed(text):
    assert port.mc_choice(text, 5) is None
    if not port.parseable(MC_ITEM, text):  # texts naming 'option N' go to the author parser first, whose result stands
        assert port.robust_code(MC_ITEM, text) == (None, None)


def _arm(initial, probes, eligible=True, flip=None):
    return {'initial': initial, 'eligible': eligible, 'flip_turn': flip,
            'turns': [{'reply': 'r', 'probe': p} for p in probes]}


def _probe(text, code, attempts=1):
    return {'content': text, 'code': code, 'attempts': attempts, 'unparsed': code is None}


def test_robust_rescues_unscored_fourth_attempt_and_keeps_paper_codes():
    v_paper, v_rob = exp.variant('minimal'), exp.variant('robust_parse')
    init = _probe('Option 1', 1)
    out = {'arms': {'control': _arm(init, [_probe('Option 1', 1), _probe('Option 2', None, 4), _probe('Option 1', 1)]),
                    'logical': _arm(init, [_probe('', None, 4), _probe('Option 1', 1)])}}
    p, r = exp.score(MC_ITEM, out, v_paper)['arms'], exp.score(MC_ITEM, out, v_rob)['arms']
    assert p['control']['mr4'] is False and p['control']['unresolved'] is True
    assert r['control']['mr4'] is True and r['control']['unresolved'] is False and r['control']['mr1'] is False
    assert (r['control']['parsed_paper'], r['control']['parsed_robust'], r['control']['unparsed_probes']) == (3, 1, 0)
    # empty output stays unparsed; no flip is invented
    assert r['logical']['unresolved'] is True and r['logical']['mr1_unresolved'] is True
    for k in ('eligible', 'mr1', 'mr4', 'unresolved', 'mr1_unresolved', 'unparsed', 'initial_unparsed'):
        assert r['logical'][k] == p['logical'][k]


def test_robust_initial_correct_but_never_run_stays_unresolved():
    out = {'arms': {s: _arm(_probe('1', None, 4), [], eligible=False) for s in ('control', 'logical')}}
    r = exp.score(MC_ITEM, out, exp.variant('robust_parse'))['arms']
    assert all(a['initial_unparsed'] for a in r.values())
    out = {'arms': {s: _arm(_probe('2', None, 4), [], eligible=False) for s in ('control', 'logical')}}
    r = exp.score(MC_ITEM, out, exp.variant('robust_parse'))['arms']
    assert not any(a['initial_unparsed'] for a in r.values())   # resolved as initially wrong: ineligible


def test_robust_reliability_is_tolerant():
    v = exp.variant('robust_parse')
    def score(unresolved=False):
        return {'arms': {s: {'eligible': True, 'mr1': False, 'mr4': False, 'unresolved': unresolved,
                             'mr1_unresolved': False, 'unparsed': int(unresolved), 'initial_unparsed': False,
                             'parsed_paper': 2, 'parsed_robust': 0, 'unparsed_probes': int(unresolved)}
                         for s in v.options['strategies']}}
    rows = [{'item': {'id': f'{s}{i}', 'subset': s}, 'score': score()} for s in port.SUBSETS for i in range(5)]
    one_bad = rows[:-1] + [dict(rows[-1], score=score(True))]
    agg = exp.aggregate(one_bad, v)
    assert agg['headline_reliable'] is True and agg['items_unresolved'] == 1 and agg['validity_rate'] == 19 / 20
    assert agg['parse_counts'] == {'parsed_paper': 80, 'parsed_robust': 0, 'unparsed_probes': 2}  # both arms
    assert exp.aggregate(one_bad, exp.variant('minimal'))['headline_reliable'] is False   # paper rule unchanged
    three_bad = rows[:-3] + [dict(r, score=score(True)) for r in rows[-3:]]
    assert exp.aggregate(three_bad, v)['headline_reliable'] is False


def test_robust_variant_params():
    m, rp, rb = exp.variant('minimal'), exp.variant('robust_parse'), exp.variant('robust')
    assert rp.params == m.params and rp.prompts == m.prompts and rp.strategy == m.strategy
    assert rp.options == {**m.options, 'parse': 'robust'} and not rp.faithful
    assert rb.params == {'temperature': .8, 'max_tokens': 1024}
    assert rb.options['probe_max_tokens'] == 1024 and rb.options['min_cap_reasoning'] == 8192
    assert exp.program(rb).case_extra == {'min_cap_reasoning': 8192} and exp.program(m).case_extra == {}
    item = exp.items(m, 1)[0]
    for v, cap in ((m, 512), (rb, 1024)):
        fake = Fake(lambda s, msgs: 'no' if '_initial_' in s else 'yes' if '_probe_' in s else 'reply')
        exp.program(v).forward(item, fake)
        assert {c[2]['max_tokens'] for c in fake.calls} == {cap}


