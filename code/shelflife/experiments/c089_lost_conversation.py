"""C089: FULL vs SHARDED Math/Code using the released adaptive simulator.

Fixed gpt-4o-mini support, original extractive retries and early-correct stopping.
Author code unit tests run only inside a deny-network, deny-write sandbox.
Safety turn cap of 40 is an explicit deviation from the unbounded source loop.

Robustness variants (non-faithful):
  robust_parse  paper_math requests, parse='robust'. The authors' extraction (gpt-4o-mini JSON extractor, answer kept
                only if it is a substring of the response, 3 tries) stands whenever it returns an answer. Only when it
                returns '' (on the panel mostly LaTeX thin spaces: extractor "6,000" vs response "6\\,000"), the
                extractor's own candidates are re-checked against the response with formatting removed
                (_robust.normalize; commas, spaces, $, braces, backslashes), else _robust.extract_number of the
                response with where='marker_then_last' (opt-in: a GSM8K answer is the final number of a worked
                solution); the candidate is scored by the authors' GSM8K evaluator. SHARDED success = any attempt
                correct (identical to the last attempt under the paper rule, which stops at the first correct one).
                A conversation whose FULL answer, or whose deciding SHARDED attempts, still yield no number is
                unparsed (missing). Reliable iff <= 10% unparsed or missing. Math only (the code arm extracts code
                without an LLM).
  robust        = robust_parse (FULL/SHARDED caps 1000 rarely bind: <= 4% truncation except M049, whose T=1
                multi-turn replies degenerate into gibberish, which no parser or cap can fix).
Outputs now also save ``full_response`` and ``extractor_candidates`` (requests unchanged); for earlier runs
``record_output`` rebuilds both from the runner trace.
"""
import ast
import base64
import json
import os
import re
import pickle
from pathlib import Path
import subprocess
import sys
import tempfile
import zlib
from . import _robust
from .. import stats
from ..program import Program
from ..prompts import PromptTemplate, Source
from ..registry import Experiment, Variant, PaperTarget, register
from ..sourcecode import load_functions

REPO=Path(__file__).resolve().parents[1]/'data/sources/C089_lost_in_conversation'
COMMIT='c865793fe34a929d316119b0451d01bd9183bcfd'
SUPPORT='M015'  # existing direct route for exact gpt-4o-mini-2024-07-18 snapshot

def template(name,path):
    return PromptTemplate.from_source(name,Source(repo=str(REPO),path=path,commit=COMMIT,url='https://github.com/microsoft/lost_in_conversation'),kind='file',style='literal')

PROMPTS={p:template(p,f'prompts/{p}.txt') for p in ('user_agent','system_turn_categorization','system_answer_extraction_gen')}
for p in ('math/math_full_prompt','math/math_system_prompt','lcb/lcb_full_prompt','lcb/lcb_system_prompt','humaneval/humaneval_full_prompt'):
    PROMPTS[p]=template(p,f'prompts/{p}.txt')
TASK_MATH=load_functions(REPO/'tasks/math/task_math.py',['TaskMath'],extra_globals={'Task':object})['TaskMath']
TASK_CODE=load_functions(REPO/'tasks/code/task_code.py',['TaskCode'],extra_globals={'Task':object,'ast':ast,'base64':base64,'pickle':pickle,'zlib':zlib})['TaskCode']
EXTRACT_CONVERSATION=load_functions(REPO/'utils.py',['extract_conversation'])['extract_conversation']

def task(item):
    obj=object.__new__(TASK_MATH if item['task']=='math' else TASK_CODE)
    if item['task']=='math': obj.fully_specified_prompt=PROMPTS['math/math_full_prompt'].text
    else:
        obj.fully_specified_prompt_lcb=PROMPTS['lcb/lcb_full_prompt'].text
        obj.fully_specified_prompt_humaneval=PROMPTS['humaneval/humaneval_full_prompt'].text
    return obj

def evaluate(item,text):
    if item['task']=='math': return TASK_MATH.evaluator_function(task(item),text,item)
    if not Path('/usr/bin/sandbox-exec').exists():
        raise RuntimeError('C089 code scoring requires macOS sandbox-exec; refusing unsandboxed execution')
    # Author evaluator can execute arbitrary subject Python. Restrict its reads,
    # writes, network, and environment; only code/ and OS libraries are readable.
    with tempfile.TemporaryDirectory(prefix='c089-scorer-') as tmp:
        profile='(version 1)(deny default)(allow process*)(allow sysctl-read)(allow mach-lookup)(allow file-read* (subpath "/System") (subpath "/usr") (subpath "/opt/homebrew") (subpath "'+sys.base_prefix+'") (subpath "'+str(REPO.parents[3])+'") (subpath "'+tmp+'"))(allow file-write* (subpath "'+tmp+'") (literal "/dev/null"))'
        command=['/usr/bin/sandbox-exec','-p',profile,sys.executable,'-m','shelflife.experiments._c089_eval_worker',str(REPO)]
        env={'PATH':os.defpath,'PYTHONPATH':str(Path(__file__).resolve().parents[2]),'PYTHONDONTWRITEBYTECODE':'1','TMPDIR':tmp}
        try:
            result=subprocess.run(command,input=json.dumps({'item':item,'text':text}),capture_output=True,text=True,timeout=30,env=env,cwd=tmp)
        except subprocess.TimeoutExpired:
            return {'score':0,'is_correct':False,'timeout':True}
        marker=[s for s in result.stdout.splitlines() if s.startswith('C089_RESULT ')]
        if result.returncode or not marker:
            raise RuntimeError('C089 sandboxed author scorer unavailable: '+result.stderr[-1000:])
        return json.loads(marker[-1].split(' ',1)[1])['evaluation']

def populate(text,**values):
    for k,v in values.items(): text=text.replace('[['+k+']]',str(v))
    return text

class LostConversation(Program):
    name='full_and_adaptive_sharded'
    auxiliary_models=(SUPPORT,)
    def __init__(self,variant): self.variant=variant
    def preview(self,item):
        system=PROMPTS['math/math_system_prompt' if item['task']=='math' else 'lcb/lcb_system_prompt'].text
        return [{'role':'system','content':system},{'role':'user','content':task(item).populate_fully_specific_prompt(item)}]
    def extract(self,item,trace,lm,step,log=None):
        response=[m['content'] for m in trace if m['role']=='assistant'][-1]
        if item['task']=='code': return task(item).extract_answer(response)
        prompt=populate(PROMPTS['system_answer_extraction_gen'].text,ASSISTANT_RESPONSE=EXTRACT_CONVERSATION(trace,to_str=True,only_last_turn=True),ANSWER_DESCRIPTION=task(item).get_answer_description())
        for attempt in range(3):
            answer=json.loads(lm.for_model(SUPPORT)([{'role':'user','content':prompt}],{'temperature':0,'response_format':{'type':'json_object'}},step=f'{step}_{attempt}').content)['answer']
            if log is not None: log.setdefault(step,[]).append(answer)
            if answer is not None and answer in response: return answer
        return ''
    def forward(self,item,lm):
        # Preflight execution isolation before any paid subject/support call.
        if item['task']=='code':
            if not Path('/usr/bin/sandbox-exec').exists(): raise RuntimeError('C089 Code requires sandbox-exec')
            probe=subprocess.run(['/usr/bin/sandbox-exec','-p','(version 1)(deny default)(allow process*)','/usr/bin/true'],capture_output=True,text=True,env={'PATH':os.defpath},timeout=10)
            if probe.returncode: raise RuntimeError('C089 Code sandbox preflight failed before model calls: '+probe.stderr.strip())
        full=self.preview(item)
        full_reply=lm(full,self.variant.params,step='full')   # kept apart: `response` is reused by the sharded loop
        full+=[{'role':'assistant','content':full_reply.content}]
        candidates={}
        full_answer=self.extract(item,full,lm,'full_extract',candidates)
        full_eval=evaluate(item,full_answer)
        trace=[full[0]]; revealed=[]; evaluations=[]
        for turn in range(self.variant.options['max_turns']):
            remaining=[s for s in item['shards'] if s['shard_id'] not in revealed]
            if not remaining: break
            if turn==0: user=item['shards'][0]['shard']; shard_id=item['shards'][0]['shard_id']
            else:
                shards=item['shards'][1:]
                prompt=populate(PROMPTS['user_agent'].text,CONVERSATION_SO_FAR=EXTRACT_CONVERSATION(trace,to_str=True,skip_system=True),SHARDS_REVEALED=json.dumps([s for s in shards if s['shard_id'] in revealed]),SHARDS_NOT_REVEALED=json.dumps([s for s in shards if s['shard_id'] not in revealed]))
                data=json.loads(lm.for_model(SUPPORT)([{'role':'user','content':prompt}],{'temperature':1,'response_format':{'type':'json_object'}},step=f'user_{turn}').content)
                user,shard_id=data['response'],data['shard_id']
            if shard_id!=-1: revealed.append(shard_id)
            trace.append({'role':'user','content':user})
            response=lm(trace,self.variant.params,step=f'sharded_{turn}')
            trace.append({'role':'assistant','content':response.content})
            classify=populate(PROMPTS['system_turn_categorization'].text,CONVERSATION_SO_FAR=EXTRACT_CONVERSATION(trace,to_str=True,only_last_turn=True),INITIAL_SHARD=item['shards'][0]['shard'],SHARDS=json.dumps(item['shards'][1:]),ANSWER_DESCRIPTION=task(item).get_answer_description())
            category=json.loads(lm.for_model(SUPPORT)([{'role':'user','content':classify}],{'temperature':0,'response_format':{'type':'json_object'}},step=f'classify_{turn}').content)
            if category['response_type']=='answer_attempt':
                answer=self.extract(item,trace,lm,f'extract_{turn}',candidates)
                evaluation=evaluate(item,answer); evaluations.append({'answer':answer,'evaluation':evaluation,'turn':turn})
                if evaluation.get('score')==1: break
        return {'full_answer':full_answer,'full_evaluation':full_eval,'sharded_evaluations':evaluations,'trace':trace,'revealed':revealed,'capped':len(trace)//2>=self.variant.options['max_turns'],
                'full_response':full_reply.content,'extractor_candidates':candidates}
    @staticmethod
    def extract_record(trace,evaluation): return trace[-1]['content']

_STEP = re.compile(r'^(full_extract|extract_\d+)_(\d+)$')


def _key(text):
    """Digits-and-symbols key for the robust substring check: _robust.normalize, then commas, whitespace, $, braces
    and backslashes removed ('6\\,000' and '6,000' -> '6000'; '\\$45,\\!000' -> '45000')."""
    return re.sub(r'[\s,$\\{}]', '', _robust.normalize(text or ''))


def robust_math_answer(response, candidates):
    """Fallback answer when the authors' extraction returned '': the first extractor candidate found in the response
    once formatting is removed, else the response's final number (_robust.extract_number). None if neither."""
    rk = _key(response)
    for c in candidates or ():
        if isinstance(c, (str, int, float)) and _key(str(c)) and _key(str(c)) in rk:
            return str(c)
    # GSM8K answers are by construction the final number of a worked solution: the last-number fallback is opted in
    # (where='marker_then_last'); fractions=False (a '6/10' is read as 6; GSM8K golds are integers or decimals).
    n = _robust.extract_number(response or '', where='marker_then_last')
    if n is None:
        return None
    return str(int(n)) if float(n).is_integer() else str(n)


def _robust_eval(item, paper_answer, paper_eval, response, candidates):
    """(correct, path) for one extraction under parse='robust'; path None = unparsed."""
    if (paper_answer or '').strip():
        return paper_eval.get('score') == 1, 'paper'
    answer = robust_math_answer(response, candidates)
    if answer is None:
        return False, None
    ev = TASK_MATH.evaluator_function(task(item), answer, item)
    if ev.get('error'):
        return False, None
    return ev.get('score') == 1, 'robust'


@register
class C089(Experiment):
    claim_id='C089'
    title='LLMs Get Lost In Multi-Turn Conversation'
    claim='Task success decreases when instructions are revealed piece by piece over multiple turns rather than fully specified.'
    paper={'title':title,'url':'https://arxiv.org/abs/2505.06120','first_date':'2025'}
    source=Source(repo=str(REPO),path='simulator_sharded.py',commit=COMMIT,
                  url='https://github.com/microsoft/lost_in_conversation',locator='run_simulation')
    maturity='ready'
    default_variant='paper_math_code'
    targets=(PaperTarget('math_released_instructions',103,0,'Released sharded_instructions_600.json at pinned code commit','dataset','paper_math_code','released_data',compute=lambda exp:sum(x['task']=='math' for x in exp.items(exp.variant()))),PaperTarget('code_released_instructions',100,0,'Released sharded_instructions_600.json at pinned code commit','dataset','paper_math_code','released_data',compute=lambda exp:sum(x['task']=='code' for x in exp.items(exp.variant()))),PaperTarget('GPT4o_math_FULL',.938,.0005,'arXiv 2505.06120 Table 1','gpt-4o-2024-11-20','paper_math_code','live'),PaperTarget('GPT4o_math_SHARDED',.679,.0005,'arXiv 2505.06120 Table 1','gpt-4o-2024-11-20','paper_math_code','live'))
    def build_variants(self):
        base=Variant('paper_math_code','Released Math and Code, FULL and adaptive SHARDED',prompts=PROMPTS,params={'temperature':1,'max_tokens':1000},strategy='full_and_adaptive_sharded',options={'per_task':None,'max_turns':40},repeats=10,deviations=('40-turn safety bound replaces unbounded author simulator; report capped episodes.', 'Non-reasoning author caps used; original reasoning models require separate 16000 FULL / 10000 SHARDED caps.'))
        paper_math=base.override('paper_math',options={'tasks':('math',)})
        robust_parse=paper_math.override('robust_parse',description='paper_math requests, parse=robust (authors\' extraction first; formatting-insensitive re-check of the extractor\'s candidates, else the final number, only when it returned nothing)',
            options={'parse':'robust'},
            deviations=('scoring: when the authors\' extraction returns \'\' (its answer is not a substring of the response, e.g. "6,000" vs LaTeX "6\\,000"), the extractor\'s candidates are re-checked with formatting removed, else the response\'s final number is used; both scored by the authors\' evaluator. SHARDED success = any attempt correct. Conversations with no usable FULL answer, or no usable deciding SHARDED attempt, are unparsed (missing)',
                        'reliability under parse=robust: <= 10% of conversations unparsed or missing (_robust.reliable)'))
        robust=robust_parse.override('robust',description='= robust_parse (no request change needed: caps rarely bind)')
        return {'paper_math_code':base,'paper_math':paper_math,
                'robust_parse':robust_parse,'robust':robust}
    def items(self,variant,limit=None):
        data=json.loads((REPO/'data/sharded_instructions_600.json').read_text())
        groups=[[dict(x,id=x['task_id']) for x in data if x['task']==t] for t in variant.options.get('tasks',('math','code'))]
        rows=[]
        for i in range(max(map(len,groups))):
            rows.extend(g[i] for g in groups if i<len(g))
        return rows[:limit] if limit is not None else rows
    def program(self,variant): return LostConversation(variant)
    def record_output(self,record):
        """Runner rows saved before 2026-10-10 lack full_response / extractor_candidates: rebuild both from the
        runner trace (the subject 'full' call and the support extractor calls)."""
        out=super().record_output(record)
        if not isinstance(out,dict) or 'trace' not in record or 'full_response' in out: return out
        out=dict(out); cands={}
        for t in record.get('trace') or ():
            step=t.get('step') or ''
            if step=='full': out['full_response']=t.get('content') or ''
            m=_STEP.match(step)
            if m and t.get('content'):
                try: cands.setdefault(m.group(1),[]).append(json.loads(t['content']).get('answer'))
                except (ValueError,AttributeError): pass
        out['extractor_candidates']=cands
        return out
    def robust_fields(self,item,output):
        """parse='robust' (math): FULL and SHARDED success with the extraction fallback; see the module docstring."""
        cands=output.get('extractor_candidates') or {}
        full,fpath=_robust_eval(item,output.get('full_answer'),output['full_evaluation'],output.get('full_response'),cands.get('full_extract'))
        assistant=[m['content'] for m in output['trace'] if m['role']=='assistant']
        paths=[]; sharded=False
        for a in output['sharded_evaluations']:
            t=a.get('turn')
            response=assistant[t] if t is not None and t<len(assistant) else ''
            ok,path=_robust_eval(item,a.get('answer'),a['evaluation'],response,cands.get(f'extract_{t}'))
            paths.append(path)
            if ok: sharded=True; break
        sharded_unparsed=not sharded and any(p is None for p in paths)
        return {'r_full':full,'r_sharded':sharded,'r_difference':int(sharded)-int(full),
                'r_unparsed':fpath is None or sharded_unparsed,'r_paths':[fpath]+paths}
    def score(self,item,output,variant):
        robust=variant is not None and variant.options.get('parse','paper')=='robust' and item['task']=='math'
        extra=self.robust_fields(item,output) if robust else {}
        attempts=output['sharded_evaluations']; sharded=bool(attempts and attempts[-1]['evaluation'].get('score')==1)
        full=output['full_evaluation'].get('score')==1
        full_empty=not (output.get('full_answer') or '').strip()
        sharded_empty=sum(not (a.get('answer') or '').strip() for a in attempts)
        full_error=bool(output['full_evaluation'].get('error'))
        sharded_errors=sum(bool(a['evaluation'].get('error')) for a in attempts)
        return {'full':full,'sharded':sharded,'difference':int(sharded)-int(full),
                'unparsed':not attempts or full_empty or bool(sharded_empty) or full_error or bool(sharded_errors),
                'no_answer_attempt':not attempts,'full_extraction_failed':full_empty,
                'sharded_extraction_failures':sharded_empty,'full_evaluator_error':full_error,
                'sharded_evaluator_errors':sharded_errors,'capped':output['capped'],**extra}
    def aggregate(self,rows,variant):
        usable=[r for r in rows if r.get('score') is not None]; n=len(rows); missing=n-len(usable)
        s=sum(r['score']['difference'] for r in usable)
        out={'n':n,'scored':len(usable),'missing':missing,'unparsed':sum(r['score']['unparsed'] for r in usable),'capped':sum(r['score']['capped'] for r in usable),'difference':s/len(usable) if usable else None,'bounds_with_missing':[(s-missing)/n,(s+missing)/n] if n else [None,None]}
        out['ci95']=list(stats.cluster_bootstrap(usable,lambda r:r['item']['id'],lambda rs:sum(r['score']['difference'] for r in rs)/len(rs))[1:]) if usable else None
        for key in ('no_answer_attempt','full_extraction_failed','sharded_extraction_failures',
                    'full_evaluator_error','sharded_evaluator_errors'):
            out[key]=sum(r['score'].get(key,0) for r in usable)
        for condition in ('full','sharded'): out[condition]=stats.rate_summary(rows,lambda r:r['score'][condition])
        out['by_task']={t:{c:stats.rate_summary([r for r in rows if r['item']['task']==t],lambda r:r['score'][c]) for c in ('full','sharded')} for t in ('math','code')}
        if variant.options.get('parse','paper')=='robust': self.robust_aggregate(out,rows,usable)
        return out
    @staticmethod
    def robust_aggregate(out,rows,usable):
        """parse='robust': headline keys (difference, ci95, scored, full, sharded) recomputed from the fallback
        scores over parsed conversations; the paper-parse values move to *_paper_parse."""
        rs=[r for r in usable if 'r_unparsed' in r['score']]
        parsed=[r for r in rs if not r['score']['r_unparsed']]
        paths=[p for r in rs for p in r['score']['r_paths']]
        n=len(rows); unparsed=len(rs)-len(parsed); missing=n-len(usable)
        for k in ('difference','ci95','scored','full','sharded','bounds_with_missing'):
            out[f'{k}_paper_parse']=out[k]
        s=sum(r['score']['r_difference'] for r in parsed)
        out['difference']=s/len(parsed) if parsed else None
        out['ci95']=list(stats.cluster_bootstrap(parsed,lambda r:r['item']['id'],lambda xs:sum(r['score']['r_difference'] for r in xs)/len(xs))[1:]) if parsed else None
        out['scored']=len(parsed)
        out['bounds_with_missing']=[(s-unparsed-missing)/n,(s+unparsed+missing)/n] if n else [None,None]
        ok=lambda r:r.get('score') is not None and 'r_unparsed' in r['score'] and not r['score']['r_unparsed']
        out['full']=stats.rate_summary(rows,lambda r:r['score']['r_full'],usable=ok)
        out['sharded']=stats.rate_summary(rows,lambda r:r['score']['r_sharded'],usable=ok)
        out['parse']={'parsed_paper':paths.count('paper'),'parsed_robust':paths.count('robust'),'unparsed':paths.count(None),
                      'unparsed_conversations':unparsed,'missing':missing}
        out['parsed_paper'],out['parsed_robust']=out['parse']['parsed_paper'],out['parse']['parsed_robust']
        out['validity_rate']=len(parsed)/n if n else None
        out['headline_reliable']=_robust.reliable(n,unparsed,missing)
