"""Replay released scores and expert labels without treating acceptance as quality."""
import collections
import hashlib
import json
from pathlib import Path
import re
import statistics

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent.parent/'claims/C014_llm-reviews-nondiscriminating/code/ReviewCritique'
COMMIT='629932bcfa31879b1df1e011e3cd7188aa3b9b46'


def load(name):return [json.loads(l) for l in (SOURCE/'data'/name).read_text().splitlines()]


def recommendation(fields):
    found=[]
    for key,value in fields.items():
        if key.lower() in ('recommendation','rating'):
            m=re.match(r'\s*(\d+(?:\.\d+)?)\b',str(value))
            if m:found.append(float(m[1]))
        else:
            m=re.fullmatch(r'(?i)Recommendation\s+(\d+(?:\.\d+)?)',key)
            if m:found.append(float(m[1]))
    if len(found)!=1 or not 1<=found[0]<=10:return None
    return found[0]


def summarize(rows,keys=None):
    groups=collections.defaultdict(list);counts=collections.Counter();missing=[];by_model=collections.defaultdict(lambda:collections.defaultdict(list));papers=[]
    for i,row in enumerate(rows):
        scores=[]
        for key,value in row.items():
            if not isinstance(value,dict) or (keys is not None and key not in keys):continue
            counts['reviews']+=1;ss=value['review'];counts['segments']+=len(ss)
            deficient=[s for s in ss if s['reliability']=='No'];counts['deficient_segments']+=len(deficient);counts['reviews_with_deficiency']+=bool(deficient)
            score=recommendation(value['score'])
            if score is None:missing.append({'paper_index':i,'title':row['title'],'review_key':key})
            else:groups[row['decision']].append(score);by_model[key][row['decision']].append(score);scores.append(score)
        papers.append({'title':row['title'],'decision':row['decision'],'mean_score':statistics.mean(scores) if scores else None,'review_scores':scores})
    return {'counts':dict(counts),'segment_deficiency_fraction':counts['deficient_segments']/counts['segments'],
            'review_deficiency_fraction':counts['reviews_with_deficiency']/counts['reviews'],
            'review_weighted_decision_means':{k:{'n':len(v),'mean':statistics.mean(v)} for k,v in groups.items()},
            'paper_weighted_decision_means':{d:statistics.mean(p['mean_score'] for p in papers if p['decision']==d and p['mean_score'] is not None) for d in ('accepted','rejected')},
            'model_decision_means':{k:{d:statistics.mean(v) for d,v in ds.items()} for k,ds in by_model.items()},'missing_scores':missing,'papers':papers}


def main():
    human=load('ReviewCritique.jsonl');llm=load('ReviewCritique_LLM.jsonl');by_title={r['title'].strip():r for r in human}
    matched=[by_title[r['title'].strip()] for r in llm]
    assert all(a['body_text']==b['body_text'] and a['decision']==b['decision'] for a,b in zip(llm,matched))
    output={'source_commit':COMMIT,'source_files':{n:hashlib.sha256((SOURCE/'data'/n).read_bytes()).hexdigest() for n in ('ReviewCritique.jsonl','ReviewCritique_LLM.jsonl')},
            'llm20':summarize(llm),'human100':summarize(human),'human_matched20':summarize(matched),
            'paper_targets':{'llm_score_accepted':7.43,'llm_score_rejected':7.47,'human_score_accepted':6.41,'human_score_rejected':4.81,'human_deficient_segments':713,'llm_deficient_segments':225,'llm_segments':1611},
            'interpretation':'Expert labels belong to archived segments only. Decision association is not objective paper-quality accuracy. Released means/counts differ from reported values.'}
    with (HERE/'archive_anchor_stats.json').open('x') as f:json.dump(output,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:v['review_weighted_decision_means'] for k,v in output.items() if isinstance(v,dict) and 'review_weighted_decision_means' in v},indent=2))


if __name__=='__main__':main()
