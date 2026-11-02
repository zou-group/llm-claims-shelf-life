"""Table 14 reconstruction on all twenty released papers; no paid API calls."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import random
from archive_anchor import load, recommendation, COMMIT

HERE=Path(__file__).resolve().parent
SEED=20260930
PROTOCOL='table14_reconstructed_v1'

CORRECTNESS='''4: All of the claims and statements are well-supported and correct.
3: Some of the paper's claims have minor issues. A few statements are not well-supported, or require small changes to be made correct.
2: Several of the paper's claims are incorrect or not well-supported.
1: The main claims of the paper are incorrect or not at all supported by theory or empirical results.'''
NOVELTY='''4: The contributions are significant, and do not exist in prior works.
3: The contributions are significant and somewhat new. Aspects of the contributions exist in prior work.
2: The contributions are only marginally significant or novel.
1: The contributions are neither significant nor novel.'''
RECOMMENDATION='''10: strong accept, should be highlighted at the conference
8: accept, good paper
6: marginally above the acceptance threshold
5: marginally below the acceptance threshold
3: reject, not good enough
1: strong reject'''
CONFIDENCE='''5: You are absolutely certain about your assessment and very familiar with the relevant literature.
4: You are confident in your assessment, but not absolutely certain.
3: You are fairly confident in your assessment. Some parts of the submission or related work may be unfamiliar; details were not carefully checked.
2: You are willing to defend your assessment, but central parts of the submission or related work may be unfamiliar; details were not carefully checked.
1: Your assessment is an educated guess; the submission is outside your area or was difficult to understand.'''


def sha(text):return hashlib.sha256(text.encode()).hexdigest()


def review_text(review):
    text='\n'.join(s['segment_text'] for s in review['review'])
    return text+'\n'+'\n'.join(f'{k}: {v.strip()}' for k,v in review['score'].items())


def select_examples(human,target_titles):
    rng=random.Random(SEED);result=[]
    for decision in ('accepted','rejected'):
        eligible=[(i,r) for i,r in enumerate(human) if r['decision']==decision and r['title'].strip() not in target_titles]
        i,row=rng.choice(eligible);keys=[k for k,v in row.items() if isinstance(v,dict) and recommendation(v['score']) is not None];key=rng.choice(keys)
        result.append({'source_index':i,'title':row['title'],'decision':decision,'review_key':key,'text':review_text(row[key])})
    return result


def build_cases():
    rows=load('ReviewCritique_LLM.jsonl');human=load('ReviewCritique.jsonl');by_title={r['title'].strip():r for r in human}
    assert collections.Counter(r['decision'] for r in rows)=={'accepted':10,'rejected':10}
    examples=select_examples(human,{r['title'].strip() for r in rows})
    substitutions={'review_guidelines':(HERE/'reviewer_guidelines_reconstructed.txt').read_text().rstrip(),
                   'Explanation on different correctness scores':CORRECTNESS,
                   'Explanation on different Technical Novelty and Significance scores':NOVELTY,
                   'Explanation on different Empirical Novelty and Significance scores':NOVELTY+'\nNA: Not applicable.',
                   'Explanation on different recommendation scores':RECOMMENDATION,
                   'Explanation on different confidence scores':CONFIDENCE,
                   'review_example_1':examples[0]['text'],'review_example_2':examples[1]['text']}
    template=(HERE/'table14_template.txt').read_text().rstrip()
    # Substitute only template placeholders, never curly braces in paper text.
    import re
    prompt=re.sub(r'\{([^{}]+)\}',lambda m:substitutions[m[1]],template)
    cases=[]
    for i,row in enumerate(rows):
        paired=by_title[row['title'].strip()];assert paired['body_text']==row['body_text']
        content=prompt+'\n\n'+row['title']+'\n'+row['body_text']
        meta={'claim_id':'C014','protocol':PROTOCOL,'paper_id':f'p{i:02d}','paper_cluster_id':f'C014:p{i:02d}',
              'decision':row['decision'],'decision_is_quality_gold':False,'expert_deficiency_gold_available_for_new_review':False,
              'title':row['title'],'body_sha256':sha(row['body_text']),'body_chars':len(row['body_text']),
              'source_commit':COMMIT,'template_sha256':sha(template),'prompt_sha256':sha(content),
              'example_seed':SEED,'examples':[{k:v for k,v in e.items() if k!='text'}|{'text_sha256':sha(e['text'])} for e in examples],
              'archive_llm_recommendations':{k:recommendation(v['score']) for k,v in row.items() if isinstance(v,dict)},
              'archive_human_recommendations':[recommendation(v['score']) for v in paired.values() if isinstance(v,dict)],
              'fidelity_limits':['Unreleased original guideline substitution, example choices and score definitions; reconstructed explicitly.',
                                'Paper names ICLR2024 format, but Table14 fields match older correctness/novelty form.',
                                'Single user message and no system are reproduction choices; original roles unspecified.',
                                'Confidence anchors paraphrased, not exact original substitution. Completion cap4096 is a reproduction choice; original sampling/cap unspecified.']}
        cases.append({'id':f'C014:p{i:02d}:{PROTOCOL}','messages':[{'role':'user','content':content}],
                      'expected':None,'condition':PROTOCOL+':'+row['decision'],'metadata':meta,'params':{'max_tokens':4096}})
    return cases


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=HERE/'cases.jsonl');ap.add_argument('--pilot',action='store_true');args=ap.parse_args();cases=build_cases()
    if args.pilot:
        cases=[next(c for c in cases if c['metadata']['decision']==d) for d in ('accepted','rejected')]
    with args.output.open('x') as f:
        for case in cases:f.write(json.dumps(case,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({'cases':len(cases),'input_chars':sum(len(c['messages'][0]['content']) for c in cases),'protocol':PROTOCOL,'fidelity':'explicit reconstruction; not exact source prompt replay'}))
