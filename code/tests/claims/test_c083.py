import json
import statistics
import pytest
from shelflife.experiments.c083_political_orientation import PoliticalOrientation,ROOT,wrappers,rule_parse,axis_scores,axis_bounds,pct_display
from shelflife.program import Completion

def test_counts_original_wrappers_and_question_text():
    e=PoliticalOrientation();r=e.items(e.variant())
    assert [len(i['questions']) for i in r]==[62,70]
    assert [len(a) for a in wrappers()]==[18,28]
    original=[json.loads(s) for s in (ROOT/'experiment_1/eightValuesPoliticalTest/gpt-4/gpt-4_results_on_eightValuesPoliticalTest_trial_1.jsonl').read_text().splitlines()]
    assert [q['text'] for q in r[1]['questions']]==[q['question'] for q in original]
    assert all(e.variant().prompts['question'].render(prefix=next(p for p in wrappers()[0] if q['question_expanded'].startswith(p+'\n\n')),
        question=q['question'],choices='\n'.join(q['allowed_answers']),suffix=next(s for s in wrappers()[1] if q['question_expanded'].endswith('\n\n\n'+s)))==q['question_expanded'] for q in original)
    assert e.items(e.variant(),limit=0)==[]


class Fake:
    repeat=0
    def __init__(self):self.calls=[]
    def for_model(self,key):return self
    def __call__(self,messages,params,step):
        self.calls.append(step)
        return Completion(step=step,status='ok',content='unknown')

def test_bounded_retry_mapper_no_random_fill():
    e=PoliticalOrientation();v=e.variant();item=e.items(v)[0];small={**item,'questions':item['questions'][:2]};lm=Fake()
    out=e.program(v).forward(small,lm)
    assert len(lm.calls)==8 and len(set(lm.calls))==8
    assert all(r['choice_index'] is None for r in out['responses'])
    assert rule_parse('Strongly disagree',['Strongly disagree','Disagree'])==0
    assert rule_parse('I disagree',['Strongly disagree','Disagree']) is None
    assert e.score(item,out,v)['axes'] is None

def test_primary_unit_retakes_and_analytic_random_baseline():
    e=PoliticalOrientation();v=e.variant();item=e.items(v)[0]
    rows=[{'item':item,'repeat':i,'score':{'axes':{'econ':-3,'social':-4},'unparsed':0}} for i in range(5)]
    rows.append({'item':item,'repeat':5,'score':None})
    a=e.aggregate(rows,v)
    assert a['shelflife']['n']==5 and a['shelflife']['missing']==1
    assert a['pct_economic_mean']==-3
    assert a['analytic_random_pct']['econ']==pytest.approx(.03625)
    extreme=axis_bounds(item,[])['econ']
    assert a['bounds_with_missing']==pytest.approx([(-15+extreme[0])/6,(-15+extreme[1])/6])
    assert a['shelflife']['headline_reliable'] is False
    assert pct_display(9.99)==10 and pct_display(-10.01)==-10

def test_rounded_partial_8values_bounds_contain_possible_completion():
    e=PoliticalOrientation();item=e.items(e.variant())[1]
    answers=[1 for q in item['questions']]
    j=next(i for i,q in enumerate(item['questions']) if q['weights']['econ']==0)
    partial=list(answers);partial[j]=None
    b=axis_bounds(item,partial)['econ']
    for choice in range(5):
        completed=list(answers);completed[j]=choice
        score=axis_scores(item,completed)['econ']
        assert b[0]<=score<=b[1]


# ---- Robustness: parse='robust', robust_parse / robust variants
import glob
from pathlib import Path
from shelflife.experiments.c083_political_orientation import robust_choice,resolve_answers,axis_scores_imputed
from shelflife.paths import PROJECT

AGREE=['Strongly disagree','Disagree','Agree','Strongly agree']
SAVED=PROJECT/'experiments/run_minimal_setups/runs/C083-two_instrument_10-L1'

@pytest.mark.parametrize('text,expected',[
    ('**Agree**','Agree'),('Agree.','Agree'),('Answer: Strongly agree','Strongly agree'),
    ('Final answer: **Strongly disagree**\n','Strongly disagree'),('<think>maybe agree</think>\n\nDisagree','Disagree'),
    ('$\\boxed{\\text{Agree}}$','Agree'),('"Disagree"','Disagree'),
    # refusals and prose mentions stay unparsed (all saved the panel extract_choice hits were of this kind)
    ('I do not agree with this',None),('This is a question where reasonable people disagree.',None),
    ("I'm not going to pick an option here. Abortion is contested and reasonable people disagree.",None),
    ('* **Arguments for "Agree" or "Strongly agree":** many argue',None),('None of the above.',None),
    ('Strongly ag',None),('',None)])
def test_robust_choice_modern_formats(text,expected):
    got=robust_choice(text,AGREE)
    assert (AGREE[got] if got is not None else None)==expected

def test_robust_variants_params():
    e=PoliticalOrientation();two=e.variant('two_instrument_10');rp=e.variant('robust_parse');rb=e.variant('robust')
    assert rp.params==two.params and rp.prompts==two.prompts and rp.strategy==two.strategy and rp.repeats==two.repeats
    assert rp.options=={'attempts':10,'parse':'robust'} and not rp.faithful
    assert rb.params=={'temperature':.7,'max_tokens':1024} and rb.options=={**rp.options,'mapper_fallback':'M025'} and not rb.faithful
    assert two.params=={'temperature':.7,'max_tokens':100} and 'parse' not in two.options

def test_imputation_equals_paper_axes_when_complete_and_expected_random_when_missing():
    e=PoliticalOrientation();pct,eight=e.items(e.variant())
    for item in (pct,eight):
        full=[1]*len(item['questions'])
        assert axis_scores_imputed(item,full)==axis_scores(item,full)
    # a fully unanswered PCT retake scores the analytic random baseline (rounded for display)
    q=pct['questions'];none=[None]*62
    assert axis_scores_imputed(pct,none)['econ']==pct_display(sum(statistics.mean(x['weights']['econ']) for x in q)/8+.38)
    assert axis_scores_imputed(eight,[None]*70)=={k:50 for k in ('econ','dipl','govt','scty')}

def _saved_rows(model):
    rows={}
    for f in sorted(glob.glob(str(SAVED/model/'responses*.jsonl'))):
        for line in open(f):
            r=json.loads(line)
            if r.get('status')=='complete':rows[r['repeat']]=r
    return list(rows.values())


def test_refused_question_rule():
    from shelflife.experiments.c083_political_orientation import refused_question
    q={'allowed':['Agree','Disagree']}
    unk={'attempts':[{'content':'As an AI I cannot choose.','mapper_content':'UNKNOWN'}]*3}
    assert refused_question(q,unk,None) and not refused_question(q,unk,0)
    assert not refused_question(q,{'attempts':[{'content':'x','mapper_content':'Agreeish'}]},None)
    assert not refused_question(q,{'attempts':[]},None)


def test_mapper_never_equals_subject_under_robust():
    from shelflife.experiments.c083_political_orientation import mapper_for, same_model, JUDGE
    e=PoliticalOrientation();rb=e.variant('robust');two=e.variant('two_instrument_10')
    assert mapper_for(rb,'M015')=='M025' and mapper_for(rb,'openai-direct/gpt-4o-mini-2024-07-18')=='M025'
    assert mapper_for(rb,'M017')==JUDGE and mapper_for(two,'M015')==JUDGE   # paper variants unchanged
    assert e.program(rb).auxiliary_models==(JUDGE,'M025') and e.program(two).auxiliary_models==(JUDGE,)
    from shelflife.program import Completion
    calls=[]
    class LM:
        model='M015';repeat=0
        def __call__(self,messages,params,step):
            calls.append(('M015',step));return Completion(step,'ok',content='I remain neutral.')
        def for_model(self,key):
            def f(messages,params,step):
                calls.append((key,step));return Completion(step,'ok',content='UNKNOWN')
            return f
    item=e.items(rb)[0];item={**item,'questions':item['questions'][:1]}
    e.program(rb).forward(item,LM())
    assert {k for k,s in calls if s.startswith('map')}=={'M025'}
