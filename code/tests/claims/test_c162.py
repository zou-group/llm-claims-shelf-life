"""Independent source prompts, released scorer execution, denominators and paper cells."""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
from shelflife.experiments import c162_temporal_grounding as m
from shelflife.sourcecode import load_functions
from shelflife.planning import plan_program

exp=m.TemporalGrounding()

def prep_module():
    spec=importlib.util.spec_from_file_location('c162_prepare',m.ROOT/'prepare.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def item():
    return {'id':'literal','domain':'news','document':'test','original':{'answer':'yes'},'negative':{'answer':'no'}}

@pytest.mark.parametrize('a,b,correct,inconsistent,has_answer,unparsed', [
    ('Yes.','No.',True,False,True,False),('No.','Yes.',False,False,True,False),
    ('Yes.','Yes.',False,True,True,False),('No.','No.',False,True,True,False),
    ('TRUE','false',True,False,True,False),('','',False,False,False,True),
    ('Unknown','Unclear',False,False,True,True),
    ('**Yes**','No.',True,False,True,True),
    ('Yes, then no.','No, then yes.',True,False,True,True),
    ('I do not know.','No.',False,True,True,True),
    ('','No.',False,True,True,True),
])
def test_literal_substring_quirks_are_preserved(a,b,correct,inconsistent,has_answer,unparsed):
    score=exp.score(item(),{'original':{'content':a},'negative':{'content':b}},exp.variant())
    assert [score[k] for k in ('correct','inconsistent','has_answer','unparsed')]==[correct,inconsistent,has_answer,unparsed]

def test_pair_missingness_and_conditional_denominator():
    def row(i,a=None,b=None):
        x={**item(),'id':str(i)}
        out={'original':{'content':a},'negative':{'content':b}} if a is not None else None
        return {'item':x,'output':out,'score':exp.score(x,out,exp.variant()) if out is not None else None}
    metric=exp.aggregate([row(0,'Yes','Yes'),row(1,'Yes','No'),row(2,'',''),row(3)],exp.variant())
    assert metric['inconsistency']==.5 and metric['scored']==2 and metric['n']==4
    assert metric['paired_accuracy']['rate']==1/3 and metric['paired_accuracy']['unscored']==1
    assert metric['no_answer']==1 and metric['missing_pairs']==1
    assert metric['bounds_with_missing']==[.25,.75]
    assert metric['headline_reliable'] is False

def test_limits_order_and_pairs_balance():
    v=exp.variant(); items=exp.items(v)
    assert len(items)==224 and exp.items(v,0)==[]
    assert exp.items(v,12)==items[:12]
    assert len({x['id'] for x in items})==224
    assert {d:sum(x['domain']==d for x in items) for d in m.DOMAINS}=={'blogs':38,'news':76,'wikipedia':110}
    assert all(x['original']['answer']!=x['negative']['answer'] for x in items)
    # Each complete pair always supplies one yes and one no gold label.
    assert sum(x[d]['answer']=='yes' for x in items[:12] for d in ('original','negative'))==12
    assert {x['domain'] for x in items[:12]}==set(m.DOMAINS)

def test_pair_plan_uses_two_fresh_independent_requests_and_no_judge():
    v=exp.variant(); first=exp.items(v,1)
    plan=plan_program(exp.program(v),first,['openai-direct/gpt-4-0613'],policy='originals_v1',responder=lambda r:'Yes.')
    assert plan['totals']['calls']==2 and not exp.program(v).auxiliary_models
    assert {c['step'] for c in plan['calls']}=={'original','negative'}
    for c in plan['calls']:
        assert [m['role'] for m in c['case']['messages']]==['system','user']


def test_zero_shot_panel_cap_option():
    assert exp.variant('zero_shot_panel').options['panel_cap']


# ---- robustness variants ----
from shelflife.experiments import _robust
from shelflife.paths import PROJECT

SAVED_RUN = PROJECT / 'experiments/run_minimal_setups/runs/added-c/runs/C162-panel-two-full'


def test_robust_requests_byte_identical_to_zero_shot_panel_and_no_top_k():
    items = exp.items(exp.variant('zero_shot_panel'), 2)
    models = ['M015', 'M133', 'M204', 'M206']
    plans = {v: plan_program(exp.program(exp.variant(v)), items, models, policy='panel_v2', responder=lambda r: 'Yes.')
             for v in ('zero_shot_panel', 'robust_parse', 'robust')}
    dump = lambda p: [json.dumps(c, sort_keys=True, default=str) for c in p['calls']]
    assert dump(plans['robust']) == dump(plans['zero_shot_panel']) == dump(plans['robust_parse'])
    for c in plans['robust']['calls']:
        assert 'top_k' not in c['payload'] and 'logprobs' not in c['payload'] and 'seed' not in c['payload']
        cap = c['payload'].get('max_tokens') or c['payload'].get('max_completion_tokens')
        assert cap == (1024 if c['model'] == 'M015' else 16384)


def pair(a, b, fa='stop', fb='stop'):
    return {'original': {'content': a, 'finish_reason': fa}, 'negative': {'content': b, 'finish_reason': fb}}


@pytest.mark.parametrize('a,b,paths,correct,inconsistent,unparsed', [
    ('**Yes**', '**No**', ['paper', 'paper'], True, False, False),                       # bold
    ('Final answer: Yes', 'Final answer: Yes', ['paper', 'paper'], False, True, False),  # answer marker
    ('<think>Is it before? It is.</think>\nYes', 'No.', ['paper', 'paper'], True, False, False),
    ('Y**es**', 'N**o**', ['robust', 'robust'], True, False, False),                     # markup splits the word
    ('Y**es**', 'Y**es**', ['robust', 'robust'], False, True, False),
    ("I'm unable to determine that.", 'No.', [None, 'paper'], False, True, True),         # refusal: unparsed, paper score
    ('', '', [None, None], False, False, True),                                         # no-answer pair
])
def test_robust_parse_modern_formats(a, b, paths, correct, inconsistent, unparsed):
    v = exp.variant('robust_parse')
    s = exp.score(item(), pair(a, b), v)
    assert s['parse_paths'] == paths
    assert (s['robust_correct'], s['robust_inconsistent'], s['robust_unparsed']) == (correct, inconsistent, unparsed)
    if 'robust' not in paths:   # no fallback used: identical to the paper rule
        assert (s['robust_correct'], s['robust_inconsistent']) == (s['correct'], s['inconsistent'])


def test_robust_truncation_rules():
    out = pair('The event happened', 'No.', fa='length')
    rp, rb = exp.score(item(), out, exp.variant('robust_parse')), exp.score(item(), out, exp.variant('robust'))
    assert rp['robust_missing'] and not rp['truncated_scored'] and not rp['robust_unparsed']
    assert not rb['robust_missing'] and rb['truncated_scored'] and not rb['robust_unparsed']
    assert rb['robust_inconsistent'] is True and rb['robust_correct'] is False   # cut, no answer -> wrong
    both_cut = exp.score(item(), pair('', '', 'length', 'length'), exp.variant('robust'))
    assert both_cut['robust_unparsed'] and not both_cut['robust_has_answer']   # paper rule gives no value
    assert not both_cut['truncated_scored']   # audit m2: unparsed only, not also truncated_scored


def test_robust_aggregate_counts_and_reliability():
    def rows(outs, v):
        res = []
        for i, o in enumerate(outs):
            x = {**item(), 'id': str(i)}
            res.append({'item': x, 'output': o, 'score': exp.score(x, o, v) if o else None})
        return res
    outs = [pair('Yes', 'No')] * 17 + [pair('Yes', 'Yes'), pair('Y**es**', 'No'), pair('', ''),
                                       pair('Yes', 'it', fb='length'), None]
    rp = exp.aggregate(rows(outs, exp.variant('robust_parse')), exp.variant('robust_parse'))
    assert (rp['parsed_paper'], rp['parsed_robust'], rp['unparsed_responses']) == (37, 1, 2)   # cut pair is missing
    assert (rp['unparsed'], rp['missing'], rp['missing_pairs'], rp['missing_truncated'], rp['truncated_scored']) == (1, 2, 1, 1, 0)
    assert rp['n'] == 22 and rp['scored'] == 19 and rp['k'] == 1
    assert rp['paired_accuracy']['scored'] == 20 and rp['paired_accuracy']['k'] == 18
    assert rp['headline_reliable'] is False and rp['validity_rate'] == 19 / 22   # 3/22 > 10%
    rb = exp.aggregate(rows(outs, exp.variant('robust')), exp.variant('robust'))
    assert (rb['parsed_paper'], rb['parsed_robust'], rb['unparsed_responses']) == (38, 1, 3)   # cut 'it' has no answer
    assert (rb['unparsed'], rb['missing'], rb['truncated_scored']) == (1, 1, 1)
    assert rb['scored'] == 20 and rb['k'] == 2 and rb['headline_reliable'] is True   # 2/22 <= 10%
    assert rb['shelflife']['value'] == .1 and rb['shelflife']['headline_reliable'] is True
    amb = exp.aggregate(rows([pair('Yes, I know', 'No')] * 3 + [pair('Yes', 'No')] * 17, exp.variant('robust')),
                        exp.variant('robust'))
    assert amb['ambiguous_substring'] == 3 and amb['unparsed'] == 0 and amb['headline_reliable'] is False


@pytest.mark.parametrize('text,flag', [
    ("I don't know.", True), ('The article does not specify.', True), ('Cannot be determined.', True),
    ('It is not mentioned in the text.', True), ('Unknown', True), ('There is no information about that.', True),
    ('No.', False), ('No, it was not mentioned first.', False), ('Yes.', False), ('Yes, I know.', False),
    ('Not before.', False), ('', False),
])
def test_refusal_like_no_flag(text, flag):
    assert m.refusal_like_no(text) is flag


def test_refusal_like_no_counts_in_guard_but_score_stays_paper():
    v = exp.variant('robust')
    s = exp.score(item(), pair("I don't know.", 'No.'), v)
    assert s['refusal_like_no'] and (s['robust_correct'], s['robust_inconsistent']) == (s['correct'], s['inconsistent'])
    def rows(outs):
        return [{'item': {**item(), 'id': str(i)}, 'output': o, 'score': exp.score({**item(), 'id': str(i)}, o, v)}
                for i, o in enumerate(outs)]
    agg = exp.aggregate(rows([pair('Yes', 'Cannot be determined.')] * 2 + [pair('Yes, I know', 'No')] * 1
                             + [pair('Yes', 'No')] * 17), v)
    assert agg['refusal_like_no_DIAGNOSTIC'] == 2 and agg['ambiguous_substring'] == 1
    assert agg['suspect_substring_pairs'] == 3 and agg['unparsed'] == 0 and agg['headline_reliable'] is False
    ok = exp.aggregate(rows([pair('Yes', 'Cannot be determined.')] * 2 + [pair('Yes', 'No')] * 18), v)
    assert ok['suspect_substring_pairs'] == 2 and ok['headline_reliable'] is True
