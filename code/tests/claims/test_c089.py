import json
from types import SimpleNamespace
import pytest
from shelflife.experiments.c089_lost_conversation import C089, REPO, PROMPTS, SUPPORT, evaluate, task
from shelflife.policies import policy_for

def test_genuine_dataset_and_order():
    e=C089(); v=e.variant(); data=e.items(v)
    original=json.loads((REPO/'data/sharded_instructions_600.json').read_text())
    assert len(data)==203 and len({d['id'] for d in data})==203
    assert len([d for d in data if d['task']=='math'])==103
    assert len([d for d in data if d['task']=='code'])==100
    assert all(d['shards'] for d in data)
    assert {d['id'] for d in data}=={d['task_id'] for d in original if d['task'] in ('math','code')}
    assert e.items(v,0)==[]

def test_all_source_prompts():
    assert PROMPTS
    for p in PROMPTS.values(): assert p.text==p.source.file.read_text()
    e=C089();v=e.variant()
    for item in e.items(v):
        request=e.program(v).preview(item)
        if item['task']=='math': assert request[1]['content']==(REPO/'prompts/math/math_full_prompt.txt').read_text().replace('[[QUESTION]]',item['question'])
        else: assert request[0]['content']==(REPO/'prompts/lcb/lcb_system_prompt.txt').read_text()

@pytest.mark.parametrize('answer,expected',[('42',1),('The answer is 42.',1),('**42**',1),('43',0),('',0),('I refuse.',0),('42 then 43',0)])
def test_math_literal_scoring(answer,expected):
    item={'task':'math','answer':'reasoning #### 42'}
    assert evaluate(item,answer)['score']==expected

def test_fixed_support_model():
    e=C089(); assert e.program(e.variant()).auxiliary_models==(SUPPORT,)
    assert policy_for(SUPPORT)

def test_code_parser_source_fixture():
    item={'task':'code'}
    code='def solution(a):\n    return a + 1'
    assert task(item).extract_answer('Explanation\n```python\n'+code+'\n```')==code
    assert task(item).extract_answer('I cannot provide a solution.')==''

def test_code_scoring_assets_are_available():
    # The scorer is launched by module name, so ordinary registry imports do not
    # detect a missing subprocess worker or its upstream evaluator snapshot.
    import importlib.util
    assert importlib.util.find_spec('shelflife.experiments._c089_eval_worker') is not None
    assert (REPO/'tasks/code/eval_code.py').is_file()

def test_code_preflight_blocks_before_model_calls(monkeypatch):
    import shelflife.experiments.c089_lost_conversation as module
    monkeypatch.setattr(module.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=1,stderr='sandbox_apply: Operation not permitted'))
    item=next(x for x in C089().items(C089().variant()) if x['task']=='code')
    def forbidden(*args,**kwargs):pytest.fail('Model called before protected-runtime preflight')
    with pytest.raises(RuntimeError,match='before model calls'):C089().program(C089().variant()).forward(item,forbidden)

def test_adaptive_first_turn_correct_stops_early():
    e=C089();v=e.variant('paper_math')
    item={'id':'fixture','task':'math','question':'What is six times seven?','answer':'#### 42','shards':[{'shard_id':0,'shard':'I need a multiplication.'},{'shard_id':1,'shard':'Six times seven.'}]}
    calls=[]
    class LM:
        def for_model(self,model): assert model==SUPPORT;return self
        def __call__(self,messages,params,step):
            calls.append((step,messages,params))
            text=json.dumps({'response_type':'answer_attempt'}) if step.startswith('classify') else json.dumps({'answer':'42'}) if 'extract' in step else '42'
            return SimpleNamespace(content=text)
    output=e.program(v).forward(item,LM())
    assert [x[0] for x in calls]==['full','full_extract_0','sharded_0','classify_0','extract_0_0']
    assert e.score(item,output,v)['full'] and e.score(item,output,v)['sharded']
    assert len(output['revealed'])==1

def test_missing_paired_fixture():
    e=C089(); v=e.variant(); rows=[{'item':{'id':'a','task':'math'},'score':{'full':True,'sharded':False,'difference':-1,'unparsed':False,'capped':False}},{'item':{'id':'b','task':'code'},'score':None}]
    result=e.aggregate(rows,v)
    assert result['difference']==-1 and result['missing']==1 and result['bounds_with_missing']==[-1,0]

def test_empty_extractions_and_evaluator_errors_are_unparsed():
    e=C089();v=e.variant();item={'task':'math','answer':'#### 42'}
    out={'full_answer':'','full_evaluation':evaluate(item,''),
         'sharded_evaluations':[{'answer':'','evaluation':evaluate(item,''),'turn':0}],'capped':False}
    score=e.score(item,out,v)
    assert score['full'] is False and score['sharded'] is False and score['difference']==0
    assert score['unparsed'] is True and score['no_answer_attempt'] is False
    assert score['full_extraction_failed'] is True and score['sharded_extraction_failures']==1
    assert score['full_evaluator_error'] is True and score['sharded_evaluator_errors']==1
    result=e.aggregate([{'item':{'id':'a','task':'math'},'score':score}],v)
    assert result['missing']==0 and result['unparsed']==1
    assert result['full_extraction_failed']==result['sharded_extraction_failures']==1

def test_valid_wrong_answer_is_parsed_and_no_attempt_is_separate():
    e=C089();v=e.variant();item={'task':'math','answer':'#### 42'}
    out={'full_answer':'43','full_evaluation':evaluate(item,'43'),
         'sharded_evaluations':[{'answer':'43','evaluation':evaluate(item,'43'),'turn':0}],'capped':False}
    score=e.score(item,out,v)
    assert score['unparsed'] is False and score['full'] is False and score['sharded'] is False
    out['sharded_evaluations']=[];score=e.score(item,out,v)
    assert score['no_answer_attempt'] is True and score['unparsed'] is True
    assert score['sharded_extraction_failures']==score['sharded_evaluator_errors']==0

def test_extraction_retry_exhaustion_and_malformed_support():
    e=C089();program=e.program(e.variant());item={'task':'math'}
    trace=[{'role':'user','content':'What is six times seven?'},{'role':'assistant','content':'I refuse to answer.'}];steps=[]
    class LM:
        def for_model(self,model):assert model==SUPPORT;return self
        def __call__(self,messages,params,step):
            steps.append(step);return SimpleNamespace(content='{"answer":"42"}')
    assert program.extract(item,trace,LM(),'fixture_extract')==''
    assert steps==['fixture_extract_0','fixture_extract_1','fixture_extract_2']
    class BadLM(LM):
        def __call__(self,*args,**kwargs):return SimpleNamespace(content='not json')
    with pytest.raises(json.JSONDecodeError):program.extract(item,trace,BadLM(),'bad')


# ------------------------------------------------------------------ robustness (parse='robust', 2026-10-10)
from shelflife.experiments import c089_lost_conversation as module


def test_robust_variants_requests():
    e=C089(); pm,rp,rb=(e.variant(n) for n in ('paper_math','robust_parse','robust'))
    for v in (rp,rb):
        assert v.params==pm.params and v.prompts==pm.prompts and v.strategy==pm.strategy and v.repeats==pm.repeats
        assert v.options['tasks']==('math',) and v.options['parse']=='robust' and not v.faithful and v.deviations
    assert e.variant('paper_math').faithful and e.items(rp)==e.items(pm)


@pytest.mark.parametrize('response,candidates,expected', [
    ('\\[\n18\\,000 - 12\\,000 = 6\\,000\\text{ gal}\n\\]\n\n\\boxed{6\\,000\\ \\text{gallons}}', ['6,000'], '6,000'),
    ('Total: \\boxed{\\$45,\\!000}', ['$45,000'], '$45,000'),
    ('**Final answer: 374 miles**', ['375'], '374'),              # candidate absent: final number of the response
    ('The answer is **-293**.', [None], '-293'),
    ('<think>maybe 12</think>\nSo she has 15 apples.', [], '15'),
    ('I cannot help with that.', ['42'], None),
    ('', ['42'], None),
])
def test_robust_math_answer_modern_formats(response,candidates,expected):
    assert module.robust_math_answer(response,candidates)==expected


def _out(item, full_response, full_answer, attempts):
    trace=[{'role':'system','content':'s'}]
    for i,(resp,ans) in enumerate(attempts):
        trace+=[{'role':'user','content':f'u{i}'},{'role':'assistant','content':resp}]
    return {'full_answer':full_answer,'full_evaluation':evaluate(item,full_answer),'full_response':full_response,
            'sharded_evaluations':[{'answer':a,'evaluation':evaluate(item,a),'turn':i} for i,(_,a) in enumerate(attempts)],
            'trace':trace,'capped':False,'extractor_candidates':{'full_extract':['6,000']*3,
                                                                 **{f'extract_{i}':[a or '6,000'] for i,(_,a) in enumerate(attempts)}}}


def test_robust_score_recovers_latex_and_keeps_paper_answers():
    e=C089(); rp=e.variant('robust_parse'); pm=e.variant('paper_math'); item={'task':'math','answer':'#### 6000'}
    latex='... \\boxed{6\\,000\\ \\text{gallons}}'
    out=_out(item,latex,'',[(latex,'')])                     # paper: both extractions failed -> both wrong
    p,r=e.score(item,out,pm),e.score(item,out,rp)
    assert (p['full'],p['sharded'],p['unparsed'])==(False,False,True)
    assert (r['r_full'],r['r_sharded'],r['r_unparsed'],r['r_paths'])==(True,True,False,['robust','robust'])
    out=_out(item,'6000','6000',[('5000','5000'),('6000','6000')])   # paper parses everywhere: same verdicts
    p,r=e.score(item,out,pm),e.score(item,out,rp)
    assert (r['r_full'],r['r_sharded'],r['r_difference'])==(p['full'],p['sharded'],p['difference'])==(True,True,0)
    out=_out(item,'I refuse.','',[('no idea','')])           # nothing recoverable: unparsed (missing)
    assert e.score(item,out,rp)['r_unparsed'] is True
    out=_out(item,'6000','6000',[])                           # no answer attempt: behaviour, not a parse failure
    r=e.score(item,out,rp); assert (r['r_sharded'],r['r_unparsed'],r['r_difference'])==(False,False,-1)


def test_robust_aggregate_headline_keys_and_reliability():
    e=C089(); rp=e.variant('robust_parse'); item={'id':'a','task':'math','answer':'#### 6000'}
    good=_out(item,'6000','6000',[('6000','6000')]); bad=_out(item,'x','',[('y','')])
    rows=[{'item':dict(item,id=f'i{k}'),'score':e.score(item,good if k else bad,rp)} for k in range(10)]+[{'item':dict(item,id='m'),'score':None}]
    agg=e.aggregate(rows,rp)
    assert agg['scored']==9 and agg['difference']==0 and agg['difference_paper_parse']==-0 and agg['scored_paper_parse']==10
    assert agg['parse']['unparsed_conversations']==1 and agg['parse']['missing']==1
    assert agg['headline_reliable'] is (2/11<=0.10)
    assert 'parse' not in e.aggregate(rows,e.variant('paper_math'))


def test_record_output_rebuilds_full_response_and_candidates():
    e=C089()
    rec={'output':{'full_answer':'','trace':[]},'trace':[
        {'step':'full','content':'... 6\\,000'},{'step':'full_extract_0','content':'{"answer": "6,000"}'},
        {'step':'full_extract_1','content':'{"answer": null}'},{'step':'sharded_0','content':'r'},
        {'step':'classify_0','content':'{"response_type": "answer_attempt"}'},{'step':'extract_0_0','content':'{"answer": "7"}'}]}
    out=e.record_output(rec)
    assert out['full_response']=='... 6\\,000'
    assert out['extractor_candidates']=={'full_extract':['6,000',None],'extract_0':['7']}


def test_forward_saves_the_full_reply_not_the_last_sharded_reply():
    """full_response must be the FULL-condition reply even after the sharded loop ran."""
    e=C089();v=e.variant('robust')
    item={'id':'fixture','task':'math','question':'q','answer':'#### 42','shards':[{'shard_id':0,'shard':'a'},{'shard_id':1,'shard':'b'}]}
    class LM:
        def for_model(self,model):return self
        def __call__(self,messages,params,step):
            if step=='full': text='FULL reply: \\boxed{4\\,2}'
            elif step.startswith('sharded'): text=f'SHARDED reply {step}: 41'
            elif step.startswith('classify'): text=json.dumps({'response_type':'answer_attempt'})
            elif step.startswith('user'): text=json.dumps({'response':'b','shard_id':1})
            else: text=json.dumps({'answer':'42'})
            return SimpleNamespace(content=text)
    out=e.program(v).forward(item,LM())
    assert out['full_response']=='FULL reply: \\boxed{4\\,2}'
    assert out['extractor_candidates']['full_extract']==['42','42','42']
    r=e.score(item,out,v)
    assert r['r_full'] is True and r['r_paths'][0]=='robust' and r['full'] is False   # paper rejects '42' vs '4\,2'
