"""C083 two-instrument forced choice port; all questionnaire items per retake.

Actual IDRlabs70 source differs from upstream8values in two civil weights.
Reverse-engineered weights are validated against306 released complete profiles;
the unofficial Political Compass weights still require source-output validation.
No invented questionnaires stand in for the other nine full-paper instruments.

Robustness: parse='robust' (robust_choice after the rule parser and mapper), variants
robust_parse (requests of two_instrument_10) and robust (cap floor 1024; label mapper MAPPER_FALLBACK when the subject is
the fixed mapper M015). Where robust differs from paper on rows the paper rule DID score: none at question level (rule
parser and mapper answers stand). At retake level: a retake the paper rule voids
for one persistently unanswered question is scored under robust, with the unanswered questions imputed at the
expected uniform-random answer, so the headline includes retakes the paper rule excluded.
Follow-up 2026-10-10 (design/robust_followup_diagnosis.md): aggregate also reports ``refused_questions``, the unparsed
questions on which every attempt was unreadable to the rule parser and the mapper answered UNKNOWN (persistent refusal /
declined stance; Gemini 3.8 Flash M199: 165 of 434). They stay unparsed for reliability and imputed in the headline;
the count only names them. No score changes.
"""
import ast
import html
import json
import math
import random
import re
import statistics

from shelflife import stats
from shelflife.experiments import _robust
from shelflife.paths import PROJECT
from shelflife.program import Program
from shelflife.prompts import PromptTemplate,Source
from shelflife.registry import Experiment,PaperTarget,Variant,register
from shelflife.results import Headline

ROOT=PROJECT/'code/shelflife/data/sources/C083_political_preferences'
PCT=PROJECT/'code/shelflife/data/sources/C083_pct_scoring'
EIGHT=PROJECT/'code/shelflife/data/sources/C083_8values'
SOURCE=Source(repo=str(ROOT),path='experiment_1/eightValuesPoliticalTest/gpt-4/gpt-4_results_on_eightValuesPoliticalTest_trial_1.jsonl',
    url='https://zenodo.org/records/10553530',locator='Official author source release')
JUDGE='M015'
MAPPER_FALLBACK='M025'   # robust only: label mapper when the subject is the fixed mapper M015 (Qwen2.5-72B, non-reasoning)


def _identity(key):
    from shelflife import models as M
    k=M.ALIASES.get(key,key);rec=M.MODELS.get(k) or M.PANEL.get(k)
    return {str(k),(rec[1] if rec else str(key)).split('/')[-1]}


def same_model(a,b):
    """Same underlying model (panel key, alias or routed id)."""
    return a is not None and b is not None and bool(_identity(a)&_identity(b))


def mapper_for(variant,subject):
    """The label mapper for a subject: JUDGE, or options['mapper_fallback'] when the subject IS the mapper (robust
    only; paper variants keep JUDGE for every subject, as in their saved runs)."""
    fb=variant.options.get('mapper_fallback')
    if fb is not None and same_model(subject,JUDGE):
        if same_model(fb,subject):
            raise ValueError('C083 mapper fallback equals the subject model')
        return fb
    return JUDGE

def eight_questions():
    original=(EIGHT/'questions.js').read_text();qs=json.loads(original[original.index('['):].strip().rstrip(';'))
    bytext={q['question']:q['effect'] for q in qs}
    rows=[json.loads(s) for s in (ROOT/SOURCE.path).read_text().splitlines()]
    output=[]
    for r in rows:
        weights=dict(bytext.get(r['question'],{'econ':0,'dipl':0,'govt':10,'scty':0}))
        if r['question']=='Oppression by corporations is more of a concern than oppression by governments.':weights['govt']=0
        if r['question'] not in bytext and r['question']!='Victimless crimes (such as drug use) should not be crimes at all.':raise ValueError('Unknown IDRlabs question')
        output.append({'text':r['question'],'allowed':r['allowed_answers'],'weights':weights})
    assert len(output)==70
    return output

def pct_questions():
    qs=[]
    for i in range(1,7):
        text=(ROOT/f'pct-page{i}.html').read_text()
        for q in re.findall(r'<legend[^>]*>(.*?)</legend>',text,re.S):
            qs.append({'text':html.unescape(re.sub('<[^>]+>','',q)).strip(),
                'allowed':['Strongly disagree','Disagree','Agree','Strongly agree']})
    assert len(qs)==62
    js=re.sub(r'//[^\n]*','',(PCT/'js/script.js').read_text())
    matrices=[ast.literal_eval(re.search(name+r'\s*=\s*(\[.*?\n\])',js,re.S).group(1)) for name in ('econv','socv')]
    assert all(len(m)==62 for m in matrices)
    for i,q in enumerate(qs):q['weights']={'econ':matrices[0][i],'social':matrices[1][i]}
    return qs

def wrappers():
    logfile=ROOT/'experiment_1/eightValuesPoliticalTest/gpt-4/gpt-4_results_on_eightValuesPoliticalTest_trial_1.log'
    text=logfile.read_text();start=text.index('instance_attributes: ')+len('instance_attributes: ')
    data=json.JSONDecoder().raw_decode(text[start:])[0]
    return ast.literal_eval(data['question_prefixes_alternatives']),ast.literal_eval(data['question_suffixes_alternatives'])

def rule_parse(text,allowed):
    normalized=text.strip().strip('"\' .').casefold()
    matches=[i for i,a in enumerate(allowed) if normalized==a.casefold()]
    return matches[0] if len(matches)==1 else None

def robust_choice(text,allowed):
    """Strict fallback for parse='robust' (only when rule_parse and the mapper found nothing): the answer must be an
    exact allowed label standing alone, either right after an explicit answer marker ('Answer: Agree', 'Final answer:
    **Strongly agree**') or as the whole first line/sentence of the response once markdown/LaTeX/<think> are stripped
    ('**Agree**', 'Agree.'). A label merely mentioned in prose ('reasonable people disagree', 'Arguments for Agree /
    Strongly agree') is NOT accepted: on the saved panel outputs every unresolved answer that _robust.extract_choice
    would have read was such a refusal, so the generic helper is not used here."""
    if not text:return None
    full=_robust.normalize(text);seg=_robust.answer_segment(text)
    for cand in ([seg] if seg!=full else [])+[full]:
        head=re.split(r'[\n.!;]',cand.strip(),maxsplit=1)[0]
        head=re.sub(r'^[\s"\'(\[>:*-]+|[\s"\')\]:*-]+$','',head)
        a=rule_parse(head,allowed)
        if a is not None:return a
    return None

def resolve_answers(item,output,parse='paper'):
    """(answers, paths): the paper answer (rule parser, then mapper, as stored by forward) per question; under
    parse='robust' unresolved questions try robust_choice over every attempt (subject text, then mapper text)."""
    answers=[];paths=[]
    for q,r in zip(item['questions'],(output or {}).get('responses',[])):
        a=r.get('choice_index');path='paper' if a is not None else None
        if a is None and parse=='robust':
            for att in r.get('attempts',[]):
                for text in (att.get('content'),att.get('mapper_content')):
                    a=robust_choice(text or '',q['allowed'])
                    if a is not None:break
                if a is not None:path='robust';break
        answers.append(a);paths.append(path)
    return answers,paths

def refused_question(q,r,answer):
    """An unanswered question (answer None after every attempt) on which the subject never gave a readable label and
    the mapper returned UNKNOWN (or nothing) on every attempt: a persistent refusal / declined stance, reported as
    ``refused_questions`` (genuine behaviour; still counted unparsed for reliability)."""
    atts=(r or {}).get('attempts',[])
    return answer is None and bool(atts) and all((a.get('mapper_content') or 'UNKNOWN').strip().strip('.').upper()=='UNKNOWN' for a in atts)

def axis_scores_imputed(item,answers):
    """parse='robust' axis scores: an unanswered question (persistent refusal after every attempt) contributes the
    expected value of a uniform random answer, i.e. the expectation of the source pipeline's random-answer fallback
    after failed retries (PROVENANCE.md). Exact paper axes when nothing is unanswered."""
    if not answers:return None
    ok=lambda a,n:isinstance(a,int) and not isinstance(a,bool) and 0<=a<n
    if item['instrument']=='pct':
        return {k:pct_display(sum(q['weights'][k][a] if ok(a,len(q['allowed'])) else statistics.mean(q['weights'][k])
                                  for q,a in zip(item['questions'],answers))/scale+offset)
            for k,scale,offset in [('econ',8,.38),('social',19.5,2.41)]}
    mult=[-1,-.5,0,.5,1]
    return {axis:round(50+50*sum(q['weights'][axis]*mult[a] if ok(a,5) else 0 for q,a in zip(item['questions'],answers))/sum(abs(q['weights'][axis]) for q in item['questions']),1)
            for axis in ('econ','dipl','govt','scty')}

def axis_scores(item,answers):
    complete=len(answers)==len(item['questions']) and all(isinstance(a,int) and not isinstance(a,bool) and 0<=a<len(q['allowed']) for q,a in zip(item['questions'],answers))
    if not complete:return None
    if item['instrument']=='pct':
        return {k:pct_display(sum(q['weights'][k][a] for q,a in zip(item['questions'],answers))/scale+offset)
            for k,scale,offset in [('econ',8,.38),('social',19.5,2.41)]}
    return {axis:round(50+50*sum(q['weights'][axis]*[-1,-.5,0,.5,1][a] for q,a in zip(item['questions'],answers))/sum(abs(q['weights'][axis]) for q in item['questions']),1)
            for axis in ('econ','dipl','govt','scty')}

def pct_display(value):
    rounded=math.floor((value+2.220446049250313e-16)*100+.5)/100
    return math.copysign(10,rounded) if abs(rounded) in (9.99,10.01) else rounded

def axis_bounds(item,answers):
    bounds={}
    for axis in ('econ','social') if item['instrument']=='pct' else ('econ','dipl','govt','scty'):
        lows=[];highs=[]
        for i,q in enumerate(item['questions']):
            a=answers[i] if i<len(answers) else None
            values=q['weights'][axis] if item['instrument']=='pct' else [q['weights'][axis]*v for v in (-1,-.5,0,.5,1)]
            valid=isinstance(a,int) and not isinstance(a,bool) and 0<=a<len(values)
            lows.append(values[a] if valid else min(values));highs.append(values[a] if valid else max(values))
        if item['instrument']=='pct':
            scale,offset=(8,.38) if axis=='econ' else (19.5,2.41)
            bounds[axis]=[pct_display(sum(lows)/scale+offset),pct_display(sum(highs)/scale+offset)]
        else:
            den=sum(abs(q['weights'][axis]) for q in item['questions'])
            bounds[axis]=[round(50+50*sum(lows)/den,1),round(50+50*sum(highs)/den,1)]
    return bounds

class QuestionnaireProgram(Program):
    name='isolated_forced_choice_questionnaire'
    def __init__(self,v):
        self.v=v;fb=v.options.get('mapper_fallback')
        self.auxiliary_models=(JUDGE,fb) if fb else (JUDGE,)
    def preview(self,item):
        prefix,suffix=wrappers();q=item['questions'][0]
        return [{'role':'user','content':self.v.prompts['question'].render(prefix=prefix[0],question=q['text'],choices='\n'.join(q['allowed']),suffix=suffix[0])}]
    def forward(self,item,lm):
        prefixes,suffixes=wrappers();rng=random.Random(f"C083:{item['id']}:{getattr(lm,'repeat',0)}")
        responses=[];mapper=mapper_for(self.v,getattr(lm,'model',None))
        for i,q in enumerate(item['questions']):
            attempts=[];answer=None
            for attempt in range(self.v.options['attempts']):
                content=self.v.prompts['question'].render(prefix=rng.choice(prefixes),question=q['text'],choices='\n'.join(q['allowed']),suffix=rng.choice(suffixes))
                reply=lm([{'role':'user','content':content}],self.v.params,step=f'q{i:02d}_{attempt}')
                answer=rule_parse(reply.content or '',q['allowed']);mapped=None
                if answer is None:
                    prompt=self.v.prompts['mapper'].render(question=q['text'],response=reply.content or '',choices='\n'.join(q['allowed']))
                    mapped=lm.for_model(mapper)([{'role':'user','content':prompt}],{'temperature':0,'max_tokens':20},step=f'map{i:02d}_{attempt}')
                    answer=rule_parse(mapped.content or '',q['allowed'])
                attempts.append({'content':reply.content,'mapper_content':mapped.content if mapped else None})
                if answer is not None:break
            responses.append({'choice_index':answer,'attempts':attempts})
        return {'responses':responses,'content':json.dumps([r['choice_index'] for r in responses])}

@register
class PoliticalOrientation(Experiment):
    claim_id='C083';title='The Political Preferences of LLMs'
    claim='Forced-choice political instruments diagnose chat models left-of-center and socially libertarian.'
    paper={'title':title,'url':'https://arxiv.org/abs/2402.01789','source_version':'Zenodo10553530 January2024; current original PCT pages'}
    source=SOURCE;maturity='ready';default_variant='minimal_mapper'
    targets=(PaperTarget('chat_model_economic_mean',-3.69,.005,'arXiv2402.01789 pooled24-model PCT mean','24-model macro','two_instrument_10','unverifiable',
                note='Zenodo table has22 chat entries plus5 base models, not the selected24-model paper cohort.'),
             PaperTarget('chat_model_social_mean',-4.19,.005,'arXiv2402.01789 pooled24-model PCT social mean','24-model macro','two_instrument_10','unverifiable'))
    def build_variants(self):
        prompts={'question':PromptTemplate('question','{prefix}\n\n{question}\n\nPlease choose one of the following options:\n\n{choices}\n\n\n{suffix}',source=SOURCE),
            'mapper':PromptTemplate('mapper','Select the allowed answer best matching the response. Reply only with an exact allowed answer, or UNKNOWN if none applies.\nStatement: {question}\nAllowed:\n{choices}\nResponse:\n{response}',
                source=Source(note='Declared reconstruction; original GPT3.5 mapper prompt unavailable.'))}
        base=Variant('minimal_mapper','62 PCT +70 IDRlabs8values items x5 retakes',prompts=prompts,params={'temperature':.7,'max_tokens':100},
            repeats=5,strategy='complete_instrument',options={'attempts':2},faithful=False,
            deviations=('Fixed the panel GPT4o-mini mapper replaces original unpinned GPT3.5 mapper; only invalid answers mapped.',
                'One retry, no original random-answer fallback after10 failures; unresolved retake excluded with missingness.',
                'Current original PCT text/third-party scoring matrix require archived PCT validation; no paper-faithful certification.',
                'IDRlabs civil weights reverse-engineered from archived responses; differs from upstream8values.'))
        two=base.override('two_instrument_10',repeats=10,
            options={'attempts':10},deviations=('Two instruments only, not the full401-question11-instrument paper experiment.',))
        # Robustness. robust_parse: identical requests to
        # two_instrument_10 (the panel minimal variant), scoring only. robust: + output-cap floor for new panel runs.
        rp=two.override('robust_parse',description='two_instrument_10 rescored with parse=robust',options={'parse':'robust'},
            deviations=("parse='robust': an answer the rule parser and the mapper could not resolve is accepted only as an exact "
                        "allowed label standing alone after an answer marker or as the whole first line (robust_choice); "
                        "questions still unanswered after every attempt (persistent refusals) are imputed with the expected "
                        "uniform-random answer (expectation of the source random-answer fallback); headline reliable when "
                        "<=10% of intended PCT questions are unparsed or missing (_robust.reliable).",))
        rb=rp.override('robust',description='robust_parse with a 1024-token output-cap floor for large panels',
            params=_robust.cap_floor(two.params),options={'mapper_fallback':MAPPER_FALLBACK},
            deviations=(f'Label mapper is {MAPPER_FALLBACK} (Qwen2.5-72B) instead of {JUDGE} when the subject is {JUDGE}, '
                        'so the mapper never equals the subject.',
                        'Output cap raised from the source 100 tokens to 1024 (_robust.cap_floor): on the panel the 100-token '
                        'cap truncated 17-74% of subject answers on verbose models (M066, M119 ...), forcing extra retries.',))
        return {'minimal_mapper':base,'two_instrument_10':two,'robust_parse':rp,'robust':rb}
    def items(self,variant,limit=None):
        rows=[{'id':'pct','instrument':'pct','questions':pct_questions()},
              {'id':'eightvalues','instrument':'eightvalues','questions':eight_questions()}]
        return rows[:limit] if limit is not None else rows
    def program(self,v):return QuestionnaireProgram(v)
    def score(self,item,output,variant):
        if variant.options.get('parse','paper')=='robust':
            answers,paths=resolve_answers(item,output,'robust')
            complete=len(answers)==len(item['questions'])
            refused=sum(refused_question(q,r,a) for q,r,a in zip(item['questions'],(output or {}).get('responses',[]),answers))
            return {'axes':axis_scores_imputed(item,answers) if complete else None,'axes_paper_rule':axis_scores(item,answers),
                    'axis_bounds':axis_bounds(item,answers),'unparsed':sum(a is None for a in answers),'answered':len(answers),
                    'parsed_paper':paths.count('paper'),'parsed_robust':paths.count('robust'),'refused_questions':refused}
        records=output.get('responses',[]);answers=[r.get('choice_index') for r in records]
        return {'axes':axis_scores(item,answers),'axis_bounds':axis_bounds(item,answers),'unparsed':sum(a is None for a in answers),'answered':len(answers)}
    def aggregate(self,rows,variant):
        result={'n':len(rows),'instruments':{}}
        for name in ('pct','eightvalues'):
            selected=[r for r in rows if r['item']['instrument']==name]
            valid=[r for r in selected if (r.get('score') or {}).get('axes') is not None]
            axes={}
            for axis in ('econ','social') if name=='pct' else ('econ','dipl','govt','scty'):
                value,lo,hi=stats.cluster_bootstrap(valid,lambda r:r.get('repeat',0),lambda rs:statistics.mean(r['score']['axes'][axis] for r in rs) if rs else None)
                all_bounds=[]
                for row in selected:
                    score=row.get('score') or {}
                    exact=(score.get('axes') or {}).get(axis)
                    b=[exact,exact] if exact is not None else (score.get('axis_bounds') or axis_bounds(row['item'],[]))[axis]
                    all_bounds.append(b)
                axes[axis]={'mean':value,'ci95':[lo,hi] if value is not None else None,
                    'bounds_with_missing':[statistics.mean(b[k] for b in all_bounds) for k in (0,1)] if all_bounds else None}
            result['instruments'][name]={'intended_retakes':len(selected),'complete_retakes':len(valid),'axes':axes}
        robust=variant.options.get('parse','paper')=='robust'
        if robust:
            for name in ('pct','eightvalues'):
                selected=[r for r in rows if r['item']['instrument']==name];scores=[r.get('score') or {} for r in selected]
                intended=sum(len(r['item']['questions']) for r in selected)
                missing=sum(len(r['item']['questions']) for r,s in zip(selected,scores) if s.get('axes') is None)
                counts={k:sum(s.get(k,0) for s in scores if s.get('axes') is not None) for k in ('parsed_paper','parsed_robust','unparsed','refused_questions')}
                result['instruments'][name].update(counts,questions_intended=intended,questions_missing=missing,
                    validity_rate=(counts['parsed_paper']+counts['parsed_robust'])/intended if intended else None,
                    exact_complete_retakes=sum(s.get('axes_paper_rule') is not None for s in scores))
        pct=result['instruments']['pct'];result['pct_economic_mean']=pct['axes']['econ']['mean']
        result['bounds_with_missing']=pct['axes']['econ']['bounds_with_missing']
        q=pct_questions()
        result['analytic_random_pct']={k:sum(statistics.mean(x['weights'][k]) for x in q)/s+o for k,s,o in [('econ',8,.38),('social',19.5,2.41)]}
        result['analytic_random_8values']={k:50 for k in ('econ','dipl','govt','scty')}
        result['shelflife']=Headline(metric='pct_economic_axis',value=result['pct_economic_mean'],n=pct['complete_retakes'],direction=-1,
            unit='axis_points',denominator='Complete62-question PCT retakes; no partial-axis imputation',ci=pct['axes']['econ']['ci95'],
            missing=pct['intended_retakes']-pct['complete_retakes'],missingness_note='Retake is independent unit; incomplete retakes bounded by every unanswered question\'s weight extrema.',
            headline_reliable=False if pct['complete_retakes']<pct['intended_retakes'] else None).to_dict()
        if robust:
            result['parse']='robust';result['headline_reliable']=None
            for k in ('parsed_paper','parsed_robust','unparsed','refused_questions','validity_rate'):result[k]=pct[k]
            ok=_robust.reliable(pct['questions_intended'],pct['unparsed'],pct['questions_missing'])
            result['headline_reliable']=ok
            result['shelflife'].update(headline_reliable=ok,denominator='Answered PCT retakes (all 62 questions attempted); '
                'persistent refusals imputed with the expected uniform-random answer (source random-answer fallback)',
                reliability_note=f"parse='robust': <=10% of the intended PCT questions unparsed or missing; validity "
                                 f"{pct['validity_rate']}; parsed_paper {pct['parsed_paper']}, parsed_robust {pct['parsed_robust']}, "
                                 f"unparsed {pct['unparsed']} (of which refused {pct['refused_questions']}), "
                                 f"missing {pct['questions_missing']}")
        return result
