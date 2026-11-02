#!/usr/bin/env python3
"""Score option IDs, keep paper-compatible extraction separate from robust extraction."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import re
import statistics

LETTERS = 'ABCD'

def norm(text):
    return ' '.join(text.strip().split())

def parse_letter(text):
    """Explicit final markers take precedence; reject unmarked explanation guesses."""
    text = text.strip()
    if re.search(r'final\s+answer\s*(?:is\s*)?[:=]?\s*[\(\[]?[ABCD][\)\]]?\s*(?:or|/)\s*[\(\[]?[ABCD]',text,re.I):
        return None
    # Explicit terminal answer formatting takes precedence over references to
    # earlier answers in the reasoning. This never consults the gold/content.
    final = re.findall(r'final\s+answer\s*(?:is\s*)?[:=]?\s*[*_]*[\(\[]?([ABCD])[\)\]]?(?![a-zA-Z])', text, re.I)
    if final:
        values=[x.upper() for x in final]
        return values[-1] if len(set(values))==1 else None
    terminal_bold = re.search(r'(?:^|\n)\s*\*\*(?:option\s+)?([ABCD])(?:[.:)]\s+[^*\n]+)?\*\*[.]?\s*$',text,re.I)
    if terminal_bold:
        return terminal_bold.group(1).upper()
    terminal = re.search(r'(?:^|\n)\s*[*_]*[\(\[]?([ABCD])[\)\]]?[*_]*[.]?\s*$', text, re.I)
    if terminal:
        return terminal.group(1).upper()
    if re.search(r'(?:answer|option)\s*(?:is\s*)?[:=]?\s*[\(\[]?[ABCD][\)\]]?\s*(?:or|/)\s*[\(\[]?[ABCD]',text,re.I):
        return None
    marked = re.findall(r'(?:final\s+answer|answer|correct\s+(?:answer|option)|option\s+chosen)\s*(?:is\s*)?[:=]?\s*[*_]*(?:option\s+)?[\(\[]?([ABCD])[\)\]]?(?![a-zA-Z])', text, re.I)
    if marked:
        # Multiple different explicit answers are ambiguous, unless a FINAL marker exists.
        final = re.findall(r'final\s+answer\s*(?:is\s*)?[:=]?\s*[*_]*[\(\[]?([ABCD])[\)\]]?(?![a-zA-Z])', text, re.I)
        values = [x.upper() for x in (final or marked)]
        return values[-1] if len(set(values)) == 1 else None
    m = re.match(r'^(?:[*_]+)?(?:option\s+)?[\(\[]?([ABCD])[\)\]]?(?=$|[\s.:,*_])', text, re.I)
    return m.group(1).upper() if m else None

def score(text, item):
    text = text if isinstance(text, str) else ''
    meta = item['metadata']
    if item['condition'] == 'noid':
        matches = [i for i,o in enumerate(meta['displayed_options']) if norm(text) == norm(o)]
        strict = meta['displayed_labels'][matches[0]] if len(matches) == 1 else None
        # The original accuracy test was startswith(gold option), but prediction
        # extraction for RStd used exact option equality. Report both honestly.
        strict_correct = norm(text).startswith(norm(meta['displayed_options'][meta['gold_position']]))
        robust = strict
        if robust is None:
            cleaned = norm(text).strip('"\'').removesuffix('.')
            matches = [i for i,o in enumerate(meta['displayed_options']) if cleaned == norm(o).removesuffix('.')]
            if len(matches) == 1:
                robust = meta['displayed_labels'][matches[0]]
    else:
        # Exact original startswith uppercase ID rule; spaces normalized first.
        strict = norm(text)[:1] if norm(text)[:1] in LETTERS and norm(text) else None
        strict_correct = strict == item['expected']
        robust = parse_letter(text)
    label_index = meta['displayed_labels'].index(robust) if robust in meta['displayed_labels'] else None
    return dict(prediction=robust, strict_prediction=strict,
                correct=robust == item['expected'], strict_correct=strict_correct,
                parsed=robust is not None, strict_parsed=strict is not None,
                parser_disagreement=robust != strict,
                predicted_position=label_index,
                predicted_content_index=meta['permutation'][label_index] if label_index is not None else None)

def summarize(rows, parser='robust'):
    n = len(rows)
    counts = Counter()
    gold_counts = Counter()
    correct_counts = Counter()
    positions = Counter()
    failures = Counter()
    for row in rows:
        item = row['case']
        sc = score(row.get('content',''),item)
        good_status = acceptable_status(row)
        failures[row.get('status','ok')] += 1
        pred = sc['prediction' if parser == 'robust' else 'strict_prediction'] if good_status else None
        gold_counts[item['expected']] += 1
        if pred is not None:
            counts[pred] += 1
            positions[item['metadata']['displayed_labels'].index(pred)] += 1
        if good_status and sc['correct' if parser=='robust' else 'strict_correct']:
            correct_counts[item['expected']] += 1
    recalls = {l:100*correct_counts[l]/gold_counts[l] if gold_counts[l] else None for l in LETTERS}
    rstd = statistics.pstdev(recalls.values()) if all(v is not None for v in recalls.values()) else None
    parsed = sum(counts.values())
    chi_square = sum((counts[l]-parsed/4)**2/(parsed/4) for l in LETTERS) if parsed else None
    return dict(n=n, accuracy_pct=100*sum(correct_counts.values())/n if n else None,
                recall_pct=recalls, rstd_pp=rstd, gold_counts=dict(gold_counts),
                prediction_counts={l:counts[l] for l in LETTERS}, prediction_position_counts=dict(positions),
                invalid_or_failed=n-parsed, statuses=dict(failures),
                prediction_uniformity_chi_square=chi_square,
                chi_square_interpretable=all(gold_counts[l] == n/4 for l in LETTERS),
                parser_disagreements=sum(score(x.get('content',''),x['case'])['parser_disagreement'] for x in rows))

def acceptable_status(row):
    status = row.get('status','ok')
    if status == 'ok':
        return True
    # At the paper's 2-token cap, a complete C. can have finish_reason=length.
    # Accept only a complete standalone ID; exclude cut-off explanations.
    return status == 'truncated' and row['case']['condition'] != 'noid' and bool(
        re.fullmatch(r'\s*[*_]*[\(\[]?[ABCD][\)\]]?[*_]*[.]?\s*',row.get('content',''),re.I))

def permutation_debias(observed, permutations):
    """Released debias_utils.simple/full: normalize rows, content mean, geometric ID prior."""
    if len(observed) != len(permutations) or not observed:
        raise ValueError('Probability matrix and permutations mismatch')
    rows = [[x/(sum(row)+1e-10) for x in row] for row in observed]
    if any(sum(row) <= 0 for row in observed):
        raise ValueError('No parsed samples in a permutation; do not infer a prior')
    gathered = [[] for _ in LETTERS]
    for row, perm in zip(rows, permutations):
        for j, original in enumerate(perm):
            gathered[original].append(row[j])
    debiased = [statistics.mean(v) for v in gathered]
    logits = [statistics.mean(math.log(row[j]+1e-10) for row in rows) for j in range(4)]
    exp = [math.exp(x-max(logits)) for x in logits]
    prior = [x/(sum(exp)+1e-10) for x in exp]
    return rows, debiased, prior

def permutation_metrics(records):
    groups = defaultdict(list)
    for row in records:
        if row['case']['condition'] in ('cyclic','full'):
            groups[(row.get('model','unknown'),row.get('repeat',0),row['case']['metadata']['question_id'],row['case']['condition'])].append(row)
    output = []
    for key, rows in groups.items():
        rows.sort(key=lambda r:r['case']['metadata']['permutation_index'])
        needed = 4 if key[-1]=='cyclic' else 24
        if len(rows) != needed or any(r.get('status','ok')!='ok' for r in rows):
            output.append(dict(group=key, status='incomplete_permutations'))
            continue
        observed = []
        for row in rows:
            choices = row.get('response',{}).get('choices',[])
            texts = [c.get('message',{}).get('content','') or '' for c in choices] if choices else [row.get('content','')]
            pred = [score(t,row['case'])['strict_prediction'] for t in texts]
            observed.append([pred.count(l) for l in LETTERS])
        try:
            normalized, debiased, prior = permutation_debias(observed,[r['case']['metadata']['permutation'] for r in rows])
        except ValueError as e:
            output.append(dict(group=key,status=str(e)))
            continue
        prediction = max(range(4),key=lambda i:debiased[i])
        output.append(dict(group=key,status='ok', observed=observed, normalized=normalized,
                           debiased=debiased, prior=prior, prediction=LETTERS[prediction],
                           correct=prediction==rows[0]['case']['metadata']['gold_content_index'],
                           target_samples_per_permutation=rows[0]['case']['params'].get('n',100),
                           observed_samples_per_permutation=[sum(x) for x in observed]))
    return output

def pride_replay(observations, ratio=.05, iterations=5, seed='results_mmlu/0s_ichat/mmlu_cyclic'):
    """Released PriDe: shuffle, prefix cyclic mean, average geometric priors,
    divide remaining original ID probabilities by prior; labels only score.
    observations are dict(observed:4x4 counts, gold:original_ID).
    """
    data=list(observations)
    prefix_n=int(len(data)*ratio)
    if prefix_n<1 or prefix_n>=len(data):
        raise ValueError('PriDe needs at least one calibration and one held-out item')
    rng=random.Random(seed.encode('utf-8'))
    perms=[list(range(k,4))+list(range(k)) for k in range(4)]
    reports=[]
    for iteration in range(iterations):
        rng.shuffle(data)
        prefix=data[:prefix_n]
        prior_vectors=[]
        predictions=[]
        labels=[]
        for entry in prefix:
            _,debiased,prior=permutation_debias(entry['observed'],perms)
            prior_vectors.append(prior)
            predictions.append(LETTERS[max(range(4),key=lambda i:debiased[i])])
            labels.append(entry['gold'])
        prior=[statistics.mean(v[j] for v in prior_vectors) for j in range(4)]
        for entry in data[prefix_n:]:
            observed=entry['observed'][0]
            logits=[math.log(observed[j]+1e-10)-math.log(prior[j]+1e-10) for j in range(4)]
            predictions.append(LETTERS[max(range(4),key=lambda i:logits[i])])
            labels.append(entry['gold'])
        recall=[100*sum(p==g==l for p,g in zip(predictions,labels))/labels.count(l) if labels.count(l) else None for l in LETTERS]
        reports.append(dict(iteration=iteration,prior=prior,
                            accuracy_pct=100*sum(p==g for p,g in zip(predictions,labels))/len(labels),
                            rstd_pp=statistics.pstdev(recall) if all(x is not None for x in recall) else None))
    return dict(n=len(data),prefix_n=prefix_n,ratio=ratio,iterations=reports,
                accuracy_pct=statistics.mean(x['accuracy_pct'] for x in reports),
                seed=seed,caveat='Probability observations must retain all samples per permutation; n=1 majority votes do not reproduce PriDe.')

def paired_spread(rows, bootstrap=1000):
    by_question = defaultdict(dict)
    for row in rows:
        if row['case']['condition'].startswith('move_'):
            by_question[row['case']['metadata']['question_id']][row['case']['condition']] = int(
                acceptable_status(row) and score(row.get('content',''),row['case'])['correct'])
    complete = [v for v in by_question.values() if len(v)==4]
    if not complete:
        return dict(paired_questions=0, spread_pp=None, bootstrap_95ci_pp=None)
    keys = ['move_'+l for l in LETTERS]
    def spread(sample):
        means = [100*statistics.mean(v[k] for v in sample) for k in keys]
        return max(means)-min(means)
    rng = random.Random(123)
    draws = sorted(spread(rng.choices(complete,k=len(complete))) for _ in range(bootstrap))
    return dict(paired_questions=len(complete),spread_pp=spread(complete),
                bootstrap_95ci_pp=[draws[int(.025*bootstrap)],draws[min(bootstrap-1,int(.975*bootstrap))]],
                caveat='Paired item bootstrap of max-minus-min; finite-sample spread is upward biased. This is not evidence of absence.')

def load_jsonl(path):
    return [json.loads(line) for line in path.open() if line.strip()]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases',type=Path,required=True)
    p.add_argument('--responses',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args = p.parse_args()
    cases = {c['id']:c for c in load_jsonl(args.cases)}
    records = load_jsonl(args.responses)
    for row in records:
        if 'case' not in row:
            row['case'] = cases[row.get('case_id',row.get('id'))]
        if 'content' not in row:
            row['content'] = row.get('text',row.get('output',''))
            if 'response' in row:
                row['content'] = row['response'].get('choices',[{}])[0].get('message',{}).get('content','') or ''
    groups = defaultdict(list)
    models = defaultdict(list)
    for row in records:
        key = (row.get('model','unknown'),row.get('repeat',0),row['case']['condition'])
        groups[key].append(row)
        models[key[:2]].append(row)
    result = dict(claim='C073',conditions=[dict(model=k[0],repeat=k[1],condition=k[2],
                                               robust=summarize(v), strict=summarize(v,'strict')) for k,v in sorted(groups.items())],
                  paired_answer_moving=[dict(model=k[0],repeat=k[1],**paired_spread(v)) for k,v in sorted(models.items())],
                  permutation_debiasing=permutation_metrics(records),
                  interpretation='Small runs detect implementation failures and large effects. Absence requires expanded samples, ablations, confidence bounds and an original-model check.')
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(out=str(args.out),records=len(records),groups=len(groups))))

if __name__=='__main__':
    main()
