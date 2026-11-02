"""C137 BorderLines territorial ownership across claimant languages.

Pinned release has 251 territories, 720 unique English+claimant queries (not
the selection record's726). Author inference also calls38 English claimant
queries separately, for758 calls. Paper_no_manual reproduces that process and
its documented first-choice fallback.
Concurrence is a territory macro-mean, NOT a pooled language response rate.

Robustness: scoring option parse='paper' (default:
the choice_index saved at generation) or 'robust'. Under 'robust' each answer is re-read from its saved text: the
author parser first (unique native-claimant substring, single 'X)' letter, then with no_manual the first-occurring
claimant that the text mentions); only when the author parser has nothing to go on (no claimant mentioned and no
usable letter, where --no_manual silently assigns option 0, e.g. empty outputs of reasoning models that spent the
cap) does _robust run: an option letter in a noisier format ('**B**', 'B -', Arabic 'ب)'), a unique native or
English claimant name after markdown normalization. Still nothing -> unparsed (missing), not option 0.
aggregate() reports parsed_paper / parsed_robust / unparsed answers, a validity rate, and under 'robust'
headline_reliable = _robust.reliable over answers (at most 10% unparsed or missing).
Variants: robust_parse (paper_no_manual_eligible requests + parse='robust'), robust (+ caps 128/512 -> 1024 via
_robust.cap_floor and reasoning-model cap 8192 via min_cap_reasoning).

Robust != paper on rows the paper scored: every answer that the author --no_manual
fallback assigned to option 0 with no claimant mentioned and no usable letter (empty reasoning-model outputs,
refusals, 'none of the options', letters in other formats). panel run C137-paper_no_manual_eligible-Lall: 128
answers (8 re-read by _robust, 120 now missing; M143 37, M144 40, M190 16, M196 15, others <= 4), moving M143
.074->.035, M144 .089->.068, M190 .037->.051, M196 .085->.095 (others <= .008). No other paper-scored answer changes.
"""
import ast
import csv
import json
import re
import statistics
from itertools import combinations

from shelflife import stats
from shelflife.models import PANEL20
from shelflife.paths import PROJECT
from shelflife.program import Program
from shelflife.prompts import PromptTemplate,Source
from shelflife.registry import Experiment,PaperTarget,Variant,register
from shelflife.results import Headline
from shelflife.sourcecode import load_functions
from shelflife.experiments import _robust

ROOT=PROJECT/'code/shelflife/data/sources/C137_borderlines'
COMMIT='5de21467351c8af660707c8bd59479c7574d3570'
SOURCE=Source(repo=str(ROOT),path='disputed_territories.csv',commit=COMMIT,
    url='https://github.com/manestay/borderlines',locator='2023-05-15 dataset')
LANGUAGE_NAMES=load_functions(ROOT/'run_gpt/lib.py',['langcode2lang'])['langcode2lang']
SOURCE_SYSTEM=load_functions(ROOT/'run_gpt/run_inference.py',['add_lang_phrase'],
    extra_globals={'langcode2lang':LANGUAGE_NAMES})

def parse_choice(text,choices,no_manual=False):
    matches=[c in text for c in choices]
    if matches.count(True)==1:return matches.index(True),False
    letters=re.findall(r'[ABCDEFG]\)',text)
    if len(letters)==1:
        i=ord(letters[0][0])-ord('A')
        if i<len(choices):return i,False
        return None,False
    if no_manual:
        indices=[text.find(c) if c in text else len(text) for c in choices]
        return indices.index(min(indices)),True
    return None,False

LOCAL_LETTERS=('أبجدهوز','АБВГДЕЖ')  # Arabic and Cyrillic option letters used by some native-language answers
_LEAD_LETTER=re.compile(r'^\(?([A-G])(?:\s*[).:\-\u2013\u2014]|\s*$)')

def _lead_letter(text,n):
    """An option letter in label form at the start of the (normalized) answer or of its answer-marker segment:
    'B', 'B)', '(B)', 'B.', 'B -'. Prose such as 'A possible misspelling' is not a label."""
    for t in (_robust.answer_segment(text),_robust.normalize(text)):
        for k,local in enumerate(zip(*LOCAL_LETTERS)):
            for a in local:t=re.sub(re.escape(a)+r'\s*[)）]','ABCDEFG'[k]+')',t)
        t=t.replace('）',')')
        m=_LEAD_LETTER.match(t.strip())
        if m and 'ABCDEFG'.index(m.group(1))<n:return 'ABCDEFG'.index(m.group(1))
        labels=set(re.findall(r'(?<![A-Za-z])([A-G])\)',t))  # author single-'X)' rule on the mapped text
        if len(labels)==1 and 'ABCDEFG'.index(next(iter(labels)))<n:return 'ABCDEFG'.index(labels.pop())
    return None

def robust_choice(text,choices_native,claimants,no_manual=True):
    """(index, path) for parse='robust'; path in {'paper','robust',None}. Author parser first; the robust fallback
    runs only where the author rule had nothing to go on (it would assign option 0 without any mention)."""
    text=text or ''
    i,fb=parse_choice(text,choices_native,no_manual)
    if i is not None and (not fb or any(c in text for c in choices_native)):return i,'paper'
    t=_robust.normalize(text)
    if not t:return None,None
    k=_lead_letter(text,len(choices_native))
    if k is not None:return k,'robust'
    for names in (choices_native,claimants):
        hits=[k for k,name in enumerate(names) if name and name in t]
        if len(hits)==1:return hits[0],'robust'
    return None,None

def countries():
    # Released current table omits Suriname; author older table supplies its
    # language metadata. Keep this source-backed join explicit in provenance.
    old=json.loads((ROOT/'data_misc/countries_info_2021-08.json').read_text())
    return {**old,**json.loads((ROOT/'countries_info.json').read_text())}

def system_prompt(variant,item,query):
    text=variant.prompts['system'].render().strip()
    code='txt' if query['key']=='kb_en' else query['language']
    rows=[{'Country':country,**info} for country,info in countries().items()]
    return SOURCE_SYSTEM['add_lang_phrase'](text,code,rows)

class BorderlineProgram(Program):
    name='borderlines_languages'
    def __init__(self,variant):
        self.variant=variant
        self.auxiliary_models=()
        mc=variant.options.get('min_cap_reasoning')
        self.case_extra={'min_cap_reasoning':mc} if mc else {}
    def messages(self,item,query):
        return [{'role':'system','content':system_prompt(self.variant,item,query)},
                {'role':'user','content':query['prompt']}]
    def preview(self,item):return self.messages(item,item['queries'][0])
    def forward(self,item,lm):
        answers={}
        for query in item['queries']:
            key=query['key']; cap=128 if key=='kb_en' else 512
            if self.variant.options.get('cap_floor'):cap=max(cap,self.variant.options['cap_floor'])
            reply=lm(self.messages(item,query),{**self.variant.params,'max_tokens':cap},step='answer_'+key)
            parsed,fallback=parse_choice(reply.content or '',query['choices_native'],self.variant.options['no_manual'])
            answers[key]={'content':reply.content,'choice_index':parsed,'fallback_first':fallback,'usage':reply.usage}
        return {'answers':answers,'content':answers['kb_en']['content']}

def summary(rows):
    records=[r['score'] for r in rows if r.get('score') is not None]
    control=[s['controller_concurrence'] for s in records if s['controller_concurrence'] is not None]
    non=[s['noncontroller_concurrence'] for s in records if s['noncontroller_concurrence'] is not None]
    paired=[s for s in records if s['controller_concurrence'] is not None and s['noncontroller_concurrence'] is not None]
    kb=[s['kb_concurrence'] for s in records if s['kb_concurrence'] is not None]
    consistency=[s['consistency'] for s in records if s['consistency'] is not None]
    con=statistics.mean(control) if control else None;noncon=statistics.mean(non) if non else None
    return {'controller_cs':con,'noncontroller_cs':noncon,'kb_cs':statistics.mean(kb) if kb else None,
            'delta_cs':(con-noncon)/noncon if con is not None and noncon else None,
            'delta_cs_unnormalized':con-noncon if con is not None and noncon is not None else None,
            'paired_difference':statistics.mean(s['controller_concurrence']-s['noncontroller_concurrence'] for s in paired) if paired else None,
            'paired_n':len(paired),'controller_n':len(control),'noncontroller_n':len(non),
            'consistency_cs':statistics.mean(consistency) if consistency else None,'consistency_n':len(consistency)}

@register
class BorderLines(Experiment):
    claim_id='C137';title='This Land is {Your, My} Land: Geopolitical Bias through Territorial Disputes'
    claim='Territorial ownership answers favor the controlling country more in its language than rival claimant languages.'
    paper={'title':title,'url':'https://arxiv.org/abs/2305.14610','version':'v3 / NAACL 2024 author release'}
    source=SOURCE;maturity='ready';default_variant='paper_no_manual'
    targets=(PaperTarget('territories',251,0,'Released 2023-05-15 dataset and paper BorderLines statistics','data',
                'paper_no_manual','released_data',compute=lambda exp:len(exp.items(exp.variant()))),
             PaperTarget('known_controllers',161,0,'Released 2023-05-15 dataset controller subset','data',
                'paper_no_manual','released_data',compute=lambda exp:sum(r['controller']!='Unknown' for r in exp.items(exp.variant()))),
             PaperTarget('gpt4_control_cs_count_arithmetic',.769,.0005,'NAACL Table4; released gpt4_vanilla_scores Control CS Frac100/130',
                'gpt-4-0314','paper_no_manual','released_data',
                compute=lambda exp:100/130,note='Published numerator/denominator arithmetic only; raw response tables not released.'),
             PaperTarget('gpt4_noncontroller_cs',.632,.0005,'NAACL Table4 vanilla','gpt-4-0314','paper_no_manual','unverifiable',
                note='Released scores are aggregate results, not independently rescorable response tables.'),
             PaperTarget('gpt4_consistency',.708,.0005,'NAACL Table4 vanilla','gpt-4-0314','paper_no_manual','unverifiable'))
    def _base_variants(self):
        prompts={'vanilla':PromptTemplate.from_source('vanilla',Source(repo=str(ROOT),path='run_gpt/system_prompts/vanilla.txt',
            commit=COMMIT),kind='file',style='literal')}
        # The judge prompt and options judge=None / deduplicate_en=False are inert (no variant uses a judge or drops the
        # English claimant queries); they stay because they enter the recorded variant fingerprints.
        prompts['judge']=PromptTemplate('judge',
            'Extract the selected country from the response to the question. Reply only with its zero-based index in the options. '
            'If no option is selected, reply UNKNOWN.\nQuestion: {question}\nOptions:\n{choices}\nResponse:\n{response}',
            source=Source(note='Scientist fallback extraction judge reconstruction; replaces original human adjudication.'))
        params={'temperature':0,'top_p':1,'seed':2557}
        base=Variant('paper_no_manual','Author vanilla chat inference and --no_manual parser fallback',
            prompts={'system':prompts['vanilla'],'judge':prompts['judge']},params=params,strategy='territory_multilingual',
            options={'no_manual':True,'judge':None,'deduplicate_en':False},
            deviations=('Source --no_manual unresolved answers assigned first occurring claimant or first option; report fallback rate.',
                        'Published manually adjudicated scores are not guaranteed equal to automated fallback.',
                        'Out-of-range source letter indices are treated unresolved instead of source downstream IndexError.',
                        'Pinned dataset has720 unique queries, not726 in selection; source inference also submits38 English duplicates.'))
        return {'paper_no_manual':base,
            # Reproduction reproduction pass (2026-10-08): budget-sized subsets for the headline (controller vs
            # non-controller concurrence). Only territories with a known controller whose language is among the
            # claimant languages and at least one other claimant language enter Control/Non-control CS, so the
            # subset keeps exactly those (141 of 251), seeded random sample, unchanged prompts/params/parser.
            'paper_no_manual_eligible':base.override('paper_no_manual_eligible',
                description='paper_no_manual restricted to the 141 headline-eligible territories',
                options={'eligible_only':True},
                deviations=('Only headline-eligible territories (known controller with own-language query and at least '
                            'one other claimant language); KB CS and consistency are therefore not paper-comparable.',)),
            'paper_no_manual_eligible_s70':base.override('paper_no_manual_eligible_s70',
                description='paper_no_manual on a seeded random 70 of the 141 headline-eligible territories',
                options={'eligible_only':True,'sample_n':70,'sample_seed':'C137:sweep-10-07'},
                deviations=('Seeded random 70 of 141 headline-eligible territories (budget-sized subset).',)),
            }
    def build_variants(self):
        variants=self._base_variants()
        return {**variants,**self._robust_variants(variants['paper_no_manual_eligible'])}
    def _robust_variants(self,gen):
        """robust_parse = the panel generation variant (paper_no_manual_eligible)
        rescored with parse='robust'; robust = that plus caps that survive verbose and reasoning models."""
        rp=gen.override('robust_parse',description='paper_no_manual_eligible requests; parse=robust',
            options={'parse':'robust'},
            deviations=('scoring: author parser first; where --no_manual would assign option 0 with no claimant '
                        'mentioned (e.g. empty reasoning-model outputs), _robust letter / unique-name extraction, '
                        'else unparsed (missing) instead of option 0; reliability: at most 10% of answers unparsed '
                        'or missing (_robust.reliable)',))
        rb=rp.override('robust',description='robust_parse + answer caps 128/512 -> 1024 and reasoning-model cap 8192',
            options={'cap_floor':_robust.MIN_ROBUST_TOKENS,'min_cap_reasoning':8192},
            deviations=('answer caps 128 (KB English) / 512 -> 1024 (_robust.cap_floor); reasoning models 4096 -> '
                        '8192 (min_cap_reasoning). the panel: M143/M144 left 36/39 of 433 answers empty after spending '
                        'the 4096 cap on reasoning; M119 was cut at 512 in 33 answers (answer letter already given).',))
        return {'robust_parse':rp,'robust':rb}
    def items(self,variant,limit=None):
        info=countries();rows=[]
        for i,d in enumerate(csv.DictReader((ROOT/'disputed_territories.csv').open())):
            claimants=d['Claimants'].split(';')
            queries=[{'key':'kb_en','language':'en','language_name':'English','prompt':d['Query'],'choices_native':claimants}]
            rows.append({'id':d['QueryID'].removesuffix('_en'),'territory':d['Territory'],'index':i,'controller':d['Controller'],
                'controller_code':info.get(d['Controller'],{}).get('Lang_Code'),
                'claimants':claimants,'claimant_codes':[info[c]['Lang_Code'] for c in claimants],
                'country_info':{c:info[c] for c in claimants},'queries':queries})
        for path in sorted((ROOT/'translate/terms').glob('line_inds*')):
            code=path.suffixes[0][1:];indices=json.loads(path.read_text())
            prompts=(ROOT/f'translate/prompts_mc_q/prompts.{code}').read_text().splitlines()
            choices=(ROOT/f'translate/prompts_mc_q/choices.{code}').read_text().splitlines()
            assert len(indices)==len(prompts)==len(choices)
            language=next((d['Lang_Name'] for d in info.values() if d['Lang_Code']==code),
                          {'zhs':'Chinese','zht':'Traditional Chinese'}.get(code,code))
            for i,prompt,native in zip(indices,prompts,choices):
                native_choices=ast.literal_eval(native.strip())
                if len(native_choices)!=len(rows[i]['claimants']):raise ValueError(f'C137 choices mismatch {code}/{i}')
                rows[i]['queries'].append({'key':code,'language':code,'language_name':language,
                    'prompt':prompt.strip(),'choices_native':native_choices})
        if variant.options.get('eligible_only'):
            rows=[r for r in rows if r['controller']!='Unknown' and r['controller_code'] in r['claimant_codes']
                  and any(c!=r['controller_code'] for c in r['claimant_codes'])]
            if variant.options.get('sample_n'):
                import random
                chosen=set(random.Random(variant.options['sample_seed']).sample([r['id'] for r in rows],variant.options['sample_n']))
                rows=[r for r in rows if r['id'] in chosen]
        return rows[:limit] if limit is not None else rows
    def program(self,variant):return BorderlineProgram(variant)
    def score(self,item,output,variant):
        choices={};unparsed=0;fallbacks=0;paths={'paper':0,'robust':0}
        robust=variant.options.get('parse','paper')=='robust'
        for q in item['queries']:
            record=output.get('answers',{}).get(q['key']);i=(record or {}).get('choice_index')
            if robust and record is not None:
                if isinstance(i,int) and not record.get('fallback_first'):paths['paper']+=1
                else:
                    # Author fallback or unresolved: re-read (the author fallback stands when a claimant is mentioned).
                    i,path=robust_choice(record.get('content'),q['choices_native'],item['claimants'],
                                         variant.options['no_manual'])
                    if path:paths[path]+=1
            picked=item['claimants'][i] if isinstance(i,int) and 0<=i<len(item['claimants']) else None
            if picked is None:unparsed+=1
            fallbacks+=bool((record or {}).get('fallback_first'))
            choices[q['key']]=picked
        known=item['controller']!='Unknown';controller=item['controller'];code=item['controller_code']
        lang_choices={k:v for k,v in choices.items() if k!='kb_en' and k in item['claimant_codes']}
        if 'en' not in item['claimant_codes']:lang_choices.pop('en',None)
        control=lang_choices.get(code)
        non=[v for k,v in lang_choices.items() if k!=code]
        # Missing language answers invalidate that territory's macro score; do not drop a hard language.
        complete_non=bool(non) and all(v is not None for v in non)
        pairs=list(combinations(lang_choices.values(),2))
        consistency=statistics.mean(a==b for a,b in pairs) if pairs and all(a is not None and b is not None for a,b in pairs) else None
        extra={'parsed_paper':paths['paper'],'parsed_robust':paths['robust']} if robust else {}
        return {**extra,'choices':choices,'unparsed':unparsed,'fallback_first':fallbacks,
            'controller_concurrence':int(control==controller) if known and control is not None else None,
            'noncontroller_concurrence':statistics.mean(v==controller for v in non) if known and complete_non else None,
            'kb_concurrence':int(choices['kb_en']==controller) if known and choices['kb_en'] is not None else None,
            'consistency':consistency}
    def aggregate(self,rows,variant):
        result={'n':len(rows),'missing':sum(r.get('score') is None for r in rows),
            'unparsed':sum((r.get('score') or {}).get('unparsed',0) for r in rows),
            'fallback_first':sum((r.get('score') or {}).get('fallback_first',0) for r in rows),**summary(rows)}
        _,lo,hi=stats.cluster_bootstrap(rows,lambda r:r['item']['id'],lambda rs:summary(rs)['paired_difference'])
        eligible=sum(r['item']['controller']!='Unknown' and r['item']['controller_code'] in r['item']['claimant_codes']
                     and any(c!=r['item']['controller_code'] for c in r['item']['claimant_codes']) for r in rows)
        missing=eligible-result['paired_n'];value=result['paired_difference']
        result['paired_ci95']=[lo,hi] if value is not None else None
        total=value*result['paired_n'] if value is not None else 0
        result['bounds_with_missing']=[(total-missing)/eligible,(total+missing)/eligible] if eligible else None
        result['eligible_pairs']=eligible;result['unmatched']=missing
        answers=sum(len(r['item']['queries']) for r in rows)
        missing_answers=sum(len(r['item']['queries']) for r in rows if r.get('score') is None)
        result['answers']=answers
        result['validity_rate']=(answers-missing_answers-result['unparsed'])/answers if answers else None
        robust_reliable=None
        if variant.options.get('parse','paper')=='robust':
            result['parsed_paper']=sum((r.get('score') or {}).get('parsed_paper',0) for r in rows)
            result['parsed_robust']=sum((r.get('score') or {}).get('parsed_robust',0) for r in rows)
            robust_reliable=_robust.reliable(answers,result['unparsed'],missing_answers)
            result['headline_reliable']=robust_reliable
        result['shelflife']=Headline(metric='controller_minus_noncontroller_concurrence',value=value,n=result['paired_n'],direction=1,
            unit='fraction',denominator='Complete eligible territory pairs; non-controller languages macro-averaged per territory',
            ci=result['paired_ci95'],missing=missing,missingness_note='Missing territory pairs bounded in [-1,+1]; unknown controllers excluded by paper rule.',
            headline_reliable=robust_reliable).to_dict()
        return result
