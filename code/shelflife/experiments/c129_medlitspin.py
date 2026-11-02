"""C129: spun/unspun oncology abstracts, Yun et al. arXiv 2502.07963v2.

Verbatim released prompts and scorer; 30 trial pairs, five separate questions.
The released scorer searches the FIRST unsigned decimal, then range-checks it.
Thus '-1' becomes 1 and '2024 ... 7' fails; these quirks remain the headline.
Both members of a trial are excluded for an invalid result for that question.
Joint prompting is a single JSON request, NOT two turns. Only API-compatible
GPT chat layouts are ported; local-weight chat templates are not claimed equal.

Robustness. Scoring option ``parse``: 'paper' (default,
the released first-unsigned-decimal rule) or 'robust': the released rule first; only when it returns nothing (no
number, or the first number is out of 0-10, e.g. "In 2024 ... I'd rate it 7 out of 10"), the local ``robust_score``:
(1) _robust.extract_number with its conservative default (number after the last answer marker, else the sole number;
'6/10' -> 6); else (2) one rating stated as '\\boxed{N}', 'rating/score: N', 'rate it N' or 'N out of 10' / 'N/10' (if
several such statements give different values, ambiguous -> None). Values outside 0-10 are rejected. No "last number
anywhere" guess. Where robust differs from paper on rows the paper parser DID score: none (the released result always
stands, including its quirks such as '-1' -> 1 and '0 to 10 ... 7' -> 0). Reliability under
'robust': at most 10% of rows unparsed or missing (_robust.reliable). Variants: ``robust_parse_nologprobs``
(benefit_baseline_nologprobs requests), for offline rescoring of the panel runs; ``robust`` (benefit_baseline_nologprobs requests: no logprobs/top_logprobs/seed, which 11
the panel routes reject; robust parse; max_tokens floor 1024 via _robust.cap_floor for every model, and the policy
option min_cap_reasoning=16000 (case_extra) so only reasoning models get 16000: M144 spent the policy's default
4096-token reasoning cap on 29 of 60 benefit questions and returned no text).
"""
import dataclasses
import csv
import functools
import json
import random
import re
from collections import defaultdict
from statistics import mean

from .. import stats
from ..paths import DATA
from ..program import SingleTurn
from ..prompts import PromptTemplate, Source
from ..registry import Experiment, PaperTarget, Variant, register
from ..sourcecode import load_functions
from . import _robust

ROBUST_REASONING_CAP = 16000   # policy min_cap_reasoning for the robust variant (reasoning models only)

ROOT = DATA / 'sources/C129_MedLitSpin'
COMMIT = '4ecc829da6733eca6ed196536c4f103a151052f3'
FILES = {'baseline': 'run_spin_interpretation_evaluation.py',
         'joint': 'run_spin_combined_detection_interpretion_evaluation.py'}


def source(protocol):
    return Source(repo=str(ROOT), path='code/' + FILES[protocol], commit=COMMIT,
                  url='https://github.com/hyesunyun/MedLitSpin', locator='Evaluator.BASE_PROMPT')


@functools.lru_cache(None)
def author(protocol):
    # Pure source loading omits imports of SDKs, torch, argparse and module body.
    cls = load_functions(ROOT / 'code' / FILES[protocol], ['Evaluator'])['Evaluator']
    return cls.__new__(cls)


def parse_score(text, protocol='baseline'):
    cleaned = author(protocol)._Evaluator__clean_text(text)
    return float(cleaned) if cleaned else None


_RATING_STATEMENTS = (
    re.compile(r'\\boxed\s*\{\s*(\d+(?:\.\d+)?)\s*(?:/\s*10)?\s*\}'),
    re.compile(r'\b(?:rating|score)\s*(?:is|of|would\s+be|:)\s*:?\s*(\d+(?:\.\d+)?)\b(?!\s*(?:-|to)\s*\d)', re.I),
    re.compile(r'\brate\s+(?:it|this|the\s+\w+)?\s*(?:as\s+|at\s+|a\s+)?(\d+(?:\.\d+)?)\b(?!\s*(?:-|to)\s*\d)', re.I),
    re.compile(r'(?<![\d.])(\d+(?:\.\d+)?)\s*(?:/|out\s+of)\s*10\b', re.I),
)


def robust_score(text):
    """Fallback for parse='robust' (only when parse_score returns None): a 0-10 rating. Conservative
    _robust.extract_number first (marker, else the sole number); else one explicit rating statement."""
    if not text:
        return None
    v = _robust.extract_number(text)
    if v is None:
        raw = str(text)
        t = _robust.normalize(raw)
        found = {float(m.group(1)) for rx in _RATING_STATEMENTS[:1] for m in rx.finditer(raw)}
        found |= {float(m.group(1)) for rx in _RATING_STATEMENTS[1:] for m in rx.finditer(t)}
        v = found.pop() if len(found) == 1 else None
    return float(v) if v is not None and 0.0 <= v <= 10.0 else None


def parse_with_path(text, protocol='baseline', parse='paper'):
    """(rating, path): path in {'paper', 'robust', None}."""
    paper = parse_score(text or '', protocol)
    if parse != 'robust':
        return paper, ('paper' if paper is not None else None)
    return _robust.fallback(paper, robust_score, text)


@functools.lru_cache(None)
def abstract_rows():
    with (ROOT / 'data/spin_and_no_spin_abstracts.csv').open(encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    random.Random(42).shuffle(rows)  # source selection order
    assert len(rows) == 60 and len({r['PMID'] for r in rows}) == 30
    return rows


def archive_rows(exp, protocol='baseline', model='gpt4o'):
    suffix = 'interpretation' if protocol == 'baseline' else 'combined_detection_interpretation'
    path = ROOT / f'code/eval_outputs/{model}/{model}_{suffix}_outputs.json'
    archive = json.loads(path.read_text())
    assert len(archive) == 60
    variant = exp.variant('paper_' + protocol)
    rows = []
    for record in archive:
        for question in author(protocol).QUESTIONS:
            item = {'id': f'{record["PMID"]}:{record["abstract_type"]}:{question}',
                    'base_id': record['PMID'], 'condition': record['abstract_type'], 'question': question}
            # Older baseline archive contains cleaned answers only. Joint archive
            # additionally releases raw completions; prefer them whenever present.
            content = record.get(question + '_raw_answer', record[question + '_answer'])
            output = {'content': content}
            rows.append({'item': item, 'output': output, 'score': exp.score(item, output, variant), 'repeat': 0})
    return rows


def archived_target(exp, protocol='baseline', model='gpt4o'):
    return exp.aggregate(archive_rows(exp, protocol, model), exp.variant('paper_' + protocol))['benefit']['mean_difference']


def pooled_target(exp, protocol):
    models = sorted(p.name for p in (ROOT / 'code/eval_outputs').iterdir() if p.is_dir())
    assert len(models) == 22
    return mean(archived_target(exp, protocol, model) for model in models)


@register
class MedLitSpin(Experiment):
    claim_id = 'C129'
    maturity = 'ready'
    title = 'Clinical trial abstract spin inflates benefit ratings'
    claim = 'Spun abstracts produce higher benefit ratings; joint detection/interpretation reduces the gap.'
    paper = {'title': 'Caught in the Web of Words: Do LLMs Fall for Spin in Medical Literature?',
             'url': 'https://arxiv.org/abs/2502.07963v2', 'version': 'v2'}
    source = source('baseline')
    default_variant = 'paper_baseline'
    targets = (
        PaperTarget('baseline_22model_benefit_difference', 3.38, .005, 'arXiv v2 Section 4', '22 models',
                    'paper_baseline', 'archived_outputs', compute=lambda e: pooled_target(e, 'baseline'),
                    note='Two-decimal reporting; invalid trial pairs excluded per question by released scorer.'),
        PaperTarget('joint_22model_benefit_difference', 1.14, .005, 'arXiv v2 Section 4', '22 models',
                    'paper_joint', 'archived_outputs', compute=lambda e: pooled_target(e, 'joint'),
                    note='Two-decimal reporting.'),
        PaperTarget('gpt4o_benefit_difference', 3.1333333333333333, 1e-12,
                    'Released gpt4o_mean_differences_metrics.json, benefit_answer_avg', 'gpt-4o-2024-08-06',
                    'paper_baseline', 'archived_outputs', compute=archived_target),
        PaperTarget('abstracts', 60, 0, 'arXiv v2 Section 2.1', 'dataset', 'paper_baseline', 'released_data',
                    compute=lambda e: len(abstract_rows())),
    )

    def build_variants(self):
        variants = {}
        for protocol in FILES:
            p = PromptTemplate.from_source('user', source(protocol), kind='py_var', var='BASE_PROMPT')
            v = Variant('paper_' + protocol, f'B.2/B.3 {protocol}, five questions, released API prompt',
                        prompts={'user': p}, params={'temperature': 0, 'max_tokens': 50, 'seed': 42,
                                                    'logprobs': True, 'top_logprobs': 20},
                        options={'protocol': protocol, 'questions': tuple(author(protocol).QUESTIONS)},
                        deviations=('GPT API one-user-message branch; local-weight templates unported.',))
            variants[v.name] = v
            variants['benefit_' + protocol] = v.override('benefit_' + protocol,
                options={'questions': ('benefit',)}, description='Primary benefit outcome only (same requests/scorer)')
            if protocol != 'baseline':
                continue
            # Black-box adaptation: several routes reject logprobs/seed (OpenRouter 404/400). The released scorer
            # parses the text answer only, so dropping them leaves scoring unchanged.
            nolp = v.override('benefit_' + protocol + '_nologprobs', options={'questions': ('benefit',)},
                              params={'seed': None}, description='benefit_' + protocol + ' without logprobs, '
                              'top_logprobs and seed (text scorer unchanged)',
                              deviations=('logprobs/top_logprobs/seed not sent: unsupported on several the panel '
                                          'routes; the scorer reads the text answer only',))
            variants[nolp.name] = dataclasses.replace(nolp, params={k: x for k, x in v.params.items()
                                                                    if k not in ('seed', 'logprobs', 'top_logprobs')})
        robust_dev = ('Scoring: released first-decimal rule first; only when it returns nothing, _robust.extract_number '
                      '(answer marker, else last number) within 0-10. Reliability: <= 10% rows unparsed or missing.',)
        variants['robust_parse_nologprobs'] = variants['benefit_baseline_nologprobs'].override(
            'robust_parse_nologprobs', options={'parse': 'robust'}, deviations=robust_dev,
            description='benefit_baseline_nologprobs requests (panel runs) with the robust fallback parse')
        variants['robust'] = variants['robust_parse_nologprobs'].override(
            'robust', params=_robust.cap_floor(variants['robust_parse_nologprobs'].params),
            options={'min_cap_reasoning': ROBUST_REASONING_CAP},
            description='benefit question, no logprobs/seed, robust parse, cap 1024 for every model and '
                        f'{ROBUST_REASONING_CAP} for reasoning models (policy min_cap_reasoning)',
            deviations=('max_tokens 1024 (paper 50) for every model via _robust.cap_floor; reasoning models get '
                        f'min_cap_reasoning {ROBUST_REASONING_CAP} through the policy (its default 4096 was exhausted '
                        'on hidden reasoning by M144 in 29/60 rows)',))
        return variants

    def items(self, variant, limit=None):
        # The source shuffles abstract rows, but pairs are interleaved for usable
        # pilot prefixes. Prompts/data unchanged; request order has no chat state.
        by_trial = defaultdict(dict)
        order = []
        for row in abstract_rows():
            if row['PMID'] not in by_trial:
                order.append(row['PMID'])
            by_trial[row['PMID']][row['abstract_type']] = row
        rows = []
        for pmid in order:
            assert set(by_trial[pmid]) == {'spin', 'no_spin'}
            for q in variant.options['questions']:
                for condition in ('no_spin', 'spin'):
                    row = by_trial[pmid][condition]
                    rows.append({'id': f'{pmid}:{condition}:{q}', 'base_id': pmid,
                                 'condition': condition, 'question': q, 'abstract': row['abstract']})
        return rows[:limit] if limit is not None else rows

    def program(self, variant):
        questions = author(variant.options['protocol']).QUESTIONS
        mc = variant.options.get('min_cap_reasoning')
        return SingleTurn(lambda item: [{'role': 'user', 'content': variant.prompts['user'].render(
            ABSTRACT=item['abstract'], QUESTION=questions[item['question']])}], variant.params,
            case_extra={'min_cap_reasoning': mc} if mc else None)

    def score(self, item, output, variant):
        text = output.get('content') or ''
        parse = variant.options.get('parse', 'paper')
        score, path = parse_with_path(text, variant.options['protocol'], parse)
        # A strict-output validity diagnostic does not change historical scoring.
        strict = bool(re.fullmatch(r'(?:10(?:\.0+)?|[0-9](?:\.\d+)?)', text.strip()))
        out = {'rating': score, 'unparsed': score is None, 'strict_numeric': strict}
        if parse == 'robust':
            out['path'] = path
        return out

    def aggregate(self, rows, variant):
        result = {'n': len(rows), 'missing': sum(r.get('score') is None for r in rows),
                  'unparsed': sum(bool((r.get('score') or {}).get('unparsed')) for r in rows)}
        for question in variant.options['questions']:
            selected = [r for r in rows if r['item']['question'] == question]
            groups = defaultdict(lambda: defaultdict(list))
            for r in selected:
                key = (r['item']['base_id'], r.get('repeat', 0))
                groups[key][r['item']['condition']].append((r.get('score') or {}).get('rating'))
            paired = []
            for (base_id, repeat), conditions in groups.items():
                if all(len(conditions[c]) == 1 and conditions[c][0] is not None for c in ('no_spin', 'spin')):
                    paired.append({'base_id': base_id, 'difference': conditions['spin'][0] - conditions['no_spin'][0]})
            n, m = len(groups), len(paired)
            point = mean(p['difference'] for p in paired) if paired else None
            ci = list(stats.cluster_bootstrap(paired, lambda r: r['base_id'],
                      lambda rs: mean(r['difference'] for r in rs))[1:]) if paired else None
            unresolved = n - m
            bounds = [(sum(p['difference'] for p in paired) - 10 * unresolved) / n,
                      (sum(p['difference'] for p in paired) + 10 * unresolved) / n] if n else None
            result[question] = {'pairs_intended': n, 'pairs_scored': m, 'unmatched_or_invalid_pairs': unresolved,
                                'mean_difference': point, 'ci95': ci, 'bounds_with_missing': bounds,
                                'denominator': 'Trial pairs with two valid ratings for this question'}
        result['headline_reliable'] = result['missing'] == 0 and result['unparsed'] == 0
        if variant.options.get('parse') == 'robust':
            scores = [r['score'] for r in rows if r.get('score')]
            result.update(parse='robust', parsed_paper=sum(x.get('path') == 'paper' for x in scores),
                          parsed_robust=sum(x.get('path') == 'robust' for x in scores),
                          validity_rate=(len(rows) - result['missing'] - result['unparsed']) / len(rows) if rows else None,
                          headline_reliable=_robust.reliable(len(rows), result['unparsed'], result['missing']))
        return result
