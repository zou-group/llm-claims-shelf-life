"""C046 — AlpacaEval length bias under verbosity prompting (Dubois et al. 2024, "Length-Controlled AlpacaEval").

Claim: prompting gpt-4-1106-preview to be concise / standard / verbose moves its AlpacaEval 2.0 win rate against
itself from 22.9% to 50.0% to 64.3% under the weighted GPT-4-Turbo annotator; length control (LC) shrinks this to
41.9 / 50.0 / 51.6, the normalized std across verbosity prompts from 25% to 10%, and raises Spearman correlation
with Chatbot Arena from 0.94 to 0.98 (Section 4.1-4.2, Figures 1 and 3).

Official pipeline (tatsu-lab/alpaca_eval @ cd543a14), verified in tests/claims/test_c046.py:
  annotator  evaluators_configs/weighted_alpaca_eval_gpt4_turbo/configs.yaml: template alpaca_eval_clf.txt filled by
             utils.make_prompts (sequential first-occurrence replacement) -> prompt_to_chatml (system + user);
             gpt-4-1106-preview, temperature 1, max_tokens 1, logprobs, top_logprobs 5 (+ top_p 1.0 added by the
             decoder helper); output order switched by random.Random('is_switched_outputs' + instruction + '0');
             completion_parsers.logprob_parser(numerator 'm', denominators ['m','M'], is_binarize False) gives
             preference = 2 - P(m)/(P(m)+P(M)), mapped back with 3 - x when switched; identical outputs get 1.5
             without a call. Win rate = mean(preference - 1) * 100 (metrics/helpers.AbsoluteScoringRule).
  LC         metrics/glm_winrate.predict_winrate: mean_i logistic(dw_notgamed + dw_difficulty * difficulty_i) with
             released weights and instruction_difficulty.csv (HF tatsu-lab/alpaca_eval @ 2edc6fa).
             For new judgments ``aggregate`` REFITS length_controlled_v1 per condition (``_c046_lc``: the official
             get_length_controlled_winrate re-implemented in numpy/scipy, sklearn/patsy unavailable; L1 logistic
             regression with GroupKFold-CV over C, regularised to the released df_gamed rows). Validated: refitting
             the released annotations recovers the released weights and leaderboard LC values (median |diff| 0.004 pp
             over 209 models; gpt4_1106_preview concise/verbose 41.90 / 51.59 vs 41.90 / 51.58).

Variants
  weighted_frozen       default, faithful. The annotator protocol on the released gpt4_1106_preview concise/verbose
                        outputs vs the released baseline outputs (805 instructions, seeded single order). Retesting it
                        means swapping the judge model; the generations stay frozen (as all prior runs did).
  weighted_frozen_sample100  weighted_frozen on the 100-instruction nested subset (the binary_paired subset).
  binary_paired         initial-reproduction adaptation for modern judges: same messages, both orders, temperature 0,
                        max_tokens 64, a text verdict parsed as a single 'm'/'M' token (100-instruction nested
                        stratified sample, seed 20260930).

Scoring
  weighted: the official logprob_parser applied to the first emitted token's top_logprobs (log_prob_index=0).
    Deviations: the official parser uses log_prob_index='batch', which for a multi-token completion returns one value
    per token and breaks the batch_size-1 annotator; we read the first token and flag multi-token completions.
    Missing logprobs / no sentinel in the top 5 (official NaN or assertion) are unparsed, never dropped. A first
    emitted token that is not 'm'/'M' (reasoning models: '</think>', ' m') is flagged ``aligned=False``; aggregate
    reports both the official-arithmetic rate and the aligned-only rate.
  binary: ``strict`` = initial-reproduction score.py (stripped text must be exactly 'm' or 'M'; identical outputs 0.5).
Metrics: candidate (gamed) win rate per verbosity condition with instruction-cluster bootstrap CIs and missingness
bounds; verbose-minus-concise matched difference. Binary aggregation follows reproductions/C046/analyze.py.

Robustness. panel runs binary_paired, where the model
under test is the judge (the weighted logprob protocol and its fixed GPT-4-Turbo annotator are untouched). Failures:
Claude Opus 4.x (M206) writes an explanation and puts the identifier alone on the last line (155 of 600 judgments
unparsed by the strict exact-token rule); Llama-4-Maverick (M066) starts "To determine the best model identifier ..."
and is cut off by the 64-token cap before any identifier (249 of 600).
  Scoring option ``parse``: 'paper' (default, ``strict``) or 'robust'. Under 'robust' the strict rule runs first and,
  only when it returns nothing, ``parse_binary_robust``: on ``_robust.normalize``d text, the last content line is an
  identifier (optionally "Model"/"Answer:"/"Final answer:" + m/M, quotes or bold), or ends with a verdict clause
  naming one identifier ("The best answer is M.", "Therefore, the best model identifier is: m"; added 2026-10-10,
  M066 robust run: 191 of 192 unparsed verdicts had this form); else the first content line is one
  and every identifier-only line agrees. Identifiers are
  case-sensitive (m and M are different models). ``_aggregate_binary`` adds parsed_paper / parsed_robust / unparsed /
  failed and ``headline_reliable`` = ``_robust.reliable`` over the judgments sent.
  Documented exception (rows the strict rule scored that robust treats differently): none; the fallback runs only
  when the strict rule returns nothing.
  robust        binary_paired with parse='robust' + max_tokens floor 1024 (64 cut off Llama-4-Maverick before the
                verdict).
"""
import ast
import collections
import copy
import functools
import json
import logging
import math
import random
import re
import statistics
import typing
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as _st
from scipy.special import logsumexp

from ..paths import INITIAL, source_repo
from ..program import SingleTurn
from ..prompts import PromptTemplate, Source
from ..registry import Experiment, PaperTarget, PriorRun, Variant, register
from ..replay import legacy_records
from ..sourcecode import load_functions
from .. import stats
from . import _robust

REPO = source_repo('C046', 'alpaca_eval')
COMMIT = 'cd543a149df89434d8a54582c0151c0b945c3d20'
SRC = REPO / 'src/alpaca_eval'
REPRO = INITIAL / 'reproductions/C046'
DIFFICULTY_CSV = REPRO / 'instruction_difficulty.csv'   # HF tatsu-lab/alpaca_eval @ 2edc6fa (download_manifest.json)
BASELINE = 'gpt4_1106_preview'
CONDITIONS = ('concise', 'verbose')
SEED = 20260930
SENTINELS = ('m', 'M')
_TYPING = {k: getattr(typing, k) for k in ('Any', 'Callable', 'Optional', 'Sequence', 'Union', 'Literal')}

_URL = 'https://github.com/tatsu-lab/alpaca_eval'
JUDGE_PATH = 'src/alpaca_eval/evaluators_configs/alpaca_eval_clf_gpt4_turbo/alpaca_eval_clf.txt'
JUDGE = PromptTemplate.from_source(
    'judge', Source(repo=str(REPO), path=JUDGE_PATH, commit=COMMIT, url=_URL), style='replace',
    tokens={'{instruction}': 'instruction', '{output_1}': 'output_1', '{output_2}': 'output_2'}, kind='file',
    note='ChatML text; filled by utils.make_prompts and split by utils.prompt_to_chatml (official functions)')

# weighted_alpaca_eval_gpt4_turbo/configs.yaml completions_kwargs + decoders/openai._openai_completion_helper top_p.
ANNOTATOR_PARAMS = {'temperature': 1, 'max_tokens': 1, 'logprobs': True, 'top_logprobs': 5, 'top_p': 1.0}


# ------------------------------------------------------------------ official code
@functools.lru_cache(maxsize=None)
def official():
    """The authors' functions, compiled from the clone without running module bodies."""
    g = {'pd': pd, 'np': np, 'copy': copy, 'Counter': Counter, 're': re, 'random': random, 'logging': logging,
         'logsumexp': logsumexp, **_TYPING}
    out = {}
    out.update(load_functions(SRC / 'utils.py', ['make_prompts', 'prompt_to_chatml', '_string_to_dict',
                                                 'random_seeded_choice', 'validate_alpacaeval_preference'],
                              extra_globals=dict(g)))
    out.update(load_functions(SRC / 'completion_parsers.py', ['logprob_parser'], extra_globals=dict(g)))
    out.update(load_functions(SRC / 'metrics/glm_winrate.py', ['_logistic'],
                              extra_globals=dict(g)))
    import abc
    import numbers
    from dataclasses import dataclass
    import numpy.typing as npt
    h = load_functions(SRC / 'metrics/helpers.py', ['BaseScoringRule', 'AbsoluteScoringRule', '_random_mode'],
                       extra_globals={**g, 'abc': abc, 'dataclass': dataclass, 'Number': numbers.Number, 'npt': npt,
                                      'validate_alpacaeval_preference': out['validate_alpacaeval_preference']})
    out['AbsoluteScoringRule'] = h['AbsoluteScoringRule']
    return out


def chatml_messages(template_text, fields):
    """utils.make_prompts (single row) then utils.prompt_to_chatml, exactly as the annotator/decoder do."""
    o = official()
    prompts, _ = o['make_prompts'](pd.DataFrame([fields]), template_text)
    return o['prompt_to_chatml'](prompts[0])


def is_switched(instruction, seed=0):
    """RandomSwitchTwoColumnsProcessor: seed = 'is_switched_outputs' + instruction + str(annotator seed 0)."""
    return official()['random_seeded_choice']('is_switched_outputs' + instruction + str(seed), [False, True])


def official_logprob_preference(completion):
    """completion_parsers.logprob_parser with the weighted_alpaca_eval_gpt4_turbo kwargs (batch mode, as configured)."""
    return official()['logprob_parser'](completion, numerator_token='m', denominator_tokens=['m', 'M'],
                                        is_binarize=False)


def official_winrate(preferences):
    """metrics.winrate.get_winrate's win_rate (AbsoluteScoringRule().describe_head2head)."""
    return official()['AbsoluteScoringRule']().describe_head2head(list(preferences))['win_rate']


# ------------------------------------------------------------------ data
def _results(model):
    return json.loads((REPO / f'results/{model}/weighted_alpaca_eval_gpt4_turbo/annotations.json').read_text())


@functools.lru_cache(maxsize=None)
def annotations(condition):
    rows = _results(f'{BASELINE}_{condition}')
    assert len(rows) == 805 and len({r['instruction'] for r in rows}) == 805
    assert all(r['generator_1'] == BASELINE and r['generator_2'] == f'{BASELINE}_{condition}' for r in rows)
    return rows


@functools.lru_cache(maxsize=None)
def baseline_outputs():
    return json.loads((REPO / f'results/{BASELINE}/model_outputs.json').read_text())


@functools.lru_cache(maxsize=None)
def selected_indices(sample=None):
    """initial-reproduction prepare.py: nested proportional dataset stratification, shuffle seed str(SEED)+dataset."""
    rows = annotations('concise')
    groups = collections.defaultdict(list)
    for i, row in enumerate(rows):
        groups[row['dataset']].append(i)
    for name, indices in groups.items():
        random.Random(str(SEED) + name).shuffle(indices)
    ordered, used = [], collections.Counter()
    for step in range(len(rows)):
        eligible = [g for g in groups if used[g] < len(groups[g])]
        name = max(eligible, key=lambda g: ((step + 1) * len(groups[g]) / len(rows) - used[g], g))
        ordered.append(groups[name][used[name]])
        used[name] += 1
    return tuple(sorted(ordered[:sample])) if sample and sample < 805 else tuple(range(805))


@functools.lru_cache(maxsize=None)
def instruction_difficulty():
    df = pd.read_csv(DIFFICULTY_CSV, index_col=0)
    return df['instruction_difficulty']


def leaderboard():
    return pd.read_csv(SRC / 'leaderboards/data_AlpacaEval_2/weighted_alpaca_eval_gpt4_turbo_leaderboard.csv',
                       index_col=0)


def glm_weights():
    return pd.read_csv(SRC / 'metrics/weights/weighted_alpaca_eval_gpt4_turbo/length_controlled_v1/'
                             'baseline_gpt4_1106_preview.csv', index_col=0)


def predict_lc_winrate(model, baseline=BASELINE):
    """glm_winrate.predict_winrate (length_controlled_v1) with the released weights, in percent."""
    w = glm_weights()
    d = w.loc[model] - w.loc[baseline]
    p = official()['_logistic'](d['not_gamed_baseline.astype(float)'] + d['instruction_difficulty'] * instruction_difficulty())
    return float(p.mean()) * 100


def _parse_raw(raw):
    return raw if isinstance(raw, dict) or raw is None else ast.literal_eval(raw)


def archived_rescored_winrate(model):
    """Re-parse a released annotations.json from its raw logprobs (official parser + seeded switch); win rate %."""
    prefs = []
    for r in _results(model):
        if r['output_1'] == r['output_2']:
            prefs.append(1.5)
            continue
        raw = _parse_raw(r['raw_completion'])
        p = official_logprob_preference(raw)[0]
        prefs.append(3 - p if is_switched(r['instruction']) else p)
    return official_winrate(prefs)


def gameability(metric):
    """notebooks/notebook_helpers.report: mean of verbosity and conciseness relative-std gameability (percent)."""
    lb = leaderboard()
    gamed = [i for i in lb.index if (i + '_verbose') in lb.index and (i + '_concise') in lb.index]
    diff_models = [i for i in lb.index if '_verbose' not in i and i + '_concise' not in i]
    weight = lb.loc[diff_models, 'win_rate'].std() / lb.loc[diff_models, metric].std()
    rel = lambda names: lb.loc[names, metric].std() / lb.loc[names, metric].mean()
    verbosity = np.mean([rel([g, g + '_verbose']) for g in gamed]) * weight * 100
    conciseness = np.mean([rel([g, g + '_concise']) for g in gamed]) * weight * 100
    return {'verbosity': float(verbosity), 'conciseness': float(conciseness),
            'gameability': float((verbosity + conciseness) / 2), 'gamed_models': gamed}


def arena_spearman(metric):
    """notebook_helpers.make_lb_arena + print_correlations: Spearman of the leaderboard metric with Arena Elo (Feb 2 2024)."""
    lb = leaderboard()
    b = pd.read_csv(REPO / 'notebooks/benchmarks.csv', index_col=0)
    arena = b.dropna(subset='LC AlpacaEval 2.0').dropna(subset='Arena Elo\n[Feb 2, 2024]')['Arena Elo\n[Feb 2, 2024]'].to_dict()
    names = [k for k in arena if k in lb.index]
    return float(_st.spearmanr(lb.loc[names, metric], [arena[k] for k in names]).statistic), len(names)


# ------------------------------------------------------------------ prior (frozen) binary scorers
def prior_strict(text, identical):
    """reproductions/C046/score.py score() for binary_paired: returns (parsed slot index 0/1, 'tie', or None, status)."""
    token = text.strip() if isinstance(text, str) else ''
    valid = token in SENTINELS
    if identical:
        return 'tie', 'identical_output_shortcut'
    if valid:
        return (0 if token == 'm' else 1), 'exact_identifier'
    return None, 'empty' if not token else 'invalid_identifier'


_IDENT_LINE = re.compile(r'^(?i:(?:the\s+)?(?:final\s+)?(?:answer|verdict|choice|output|decision|preferred(?:\s+model)?|'
                         r'best(?:\s+model)?|model\s+identifier|identifier)\s*(?:is)?\s*[:\-]?\s*)?'
                         r'(?i:model\s+)?["\'(\[]?([mM])["\')\]]?\s*\.?$')


# Last line that ENDS with a verdict clause naming one identifier ("The best answer is M.", "Therefore, the best model
# identifier is: m", "Best Model Identifier: M"); words case-insensitive, identifier case-sensitive. 'is not m' and
# 'is m or M' do not match (the identifier must follow 'is'/':' directly and end the line). Added 2026-10-10 after
# Llama-4-Maverick (M066) put 191 of 192 robust-run verdicts in this form (design/robust_followup_diagnosis.md).
_VERDICT_CLAUSE = re.compile(r'(?i:\b(?:best\s+(?:answer|output|model(?:\s+identifier)?)|'
                             r'model\s+identifier(?:\s+of\s+the\s+best\s+model)?)\s*(?:is)?\s*:?\s*)'
                             r'["\'(\[]?([mM])["\')\]]?\s*\.?$')


def parse_binary_robust(text):
    """Fallback for parse='robust' (strict returned nothing): slot index 0/1 for 'm'/'M', or None."""
    lines = [l.strip() for l in _robust.normalize(text or '').splitlines() if l.strip()]
    if not lines:
        return None
    last = _IDENT_LINE.match(lines[-1]) or _VERDICT_CLAUSE.search(lines[-1])
    if last:
        return 0 if last.group(1) == 'm' else 1
    idents = {m.group(1) for m in map(_IDENT_LINE.match, lines) if m}
    if _IDENT_LINE.match(lines[0]) and len(idents) == 1:
        return 0 if idents.pop() == 'm' else 1
    return None


# ------------------------------------------------------------------ programs
def _raw_response(lm, completion):
    """Full provider response of a call (Completion drops logprobs; the ledger's raw record keeps them)."""
    path = Path(lm.client.run_dir) / 'raw' / f'{completion.cache_key}.json'
    return json.loads(path.read_text()).get('response') or {}


def _first_choice_logprobs(response):
    choices = (response or {}).get('choices') or []
    return choices[0].get('logprobs') if choices else None


class FrozenJudge(SingleTurn):
    """One annotator call per (instruction, condition, order); identical outputs are not sent (official 1.5)."""

    def __init__(self, build, params, want_logprobs):
        super().__init__(build, params)
        self.want_logprobs = want_logprobs

    def forward(self, item, lm):
        if item['identical_outputs']:
            return {'content': None, 'status': 'identical_output_shortcut', 'logprobs': None}
        c = lm(self.build(item), self.params_for(item), step='main')
        out = {'content': c.content, 'finish_reason': c.finish_reason, 'status': c.status}
        if self.want_logprobs:
            out['logprobs'] = _first_choice_logprobs(_raw_response(lm, c))
        return out


# ---------------------------------------------------------------------------- earlier pipeline runs
# experiments/run_minimal_setups/runs/earlier/runs/<claim>-<variant>-L<N>[-<tag>]/<model>/: one shelflife.runner directory per model.
# Registered: every directory that sent at least one request (complete or partial); tests/test_prior_runs.py
# replays each one against this config. Excluded: aborted, smoke-test and superseded directories, and directories
# that sent nothing. Listed explicitly (written 2026-10-05 from the run directories; nothing is globbed at import).

_R2 = 'experiments/run_minimal_setups/runs/earlier/runs'
_NOTHING_SENT = ('no request reached a provider: every call is not_submitted (circuit breaker open, or stale ledger '
                 'rows after a package-hash change); the directory only records the attempts')


def _unsent(*runs):
    """excluded_runs entries for per-model directories that sent nothing: (run, space-separated models)."""
    return {f'{_R2}/{run}/{m}': _NOTHING_SENT for run, models in runs for m in models.split()}


# ------------------------------------------------------------------ experiment
@register
class C046AlpacaEvalLength(Experiment):
    claim_id = 'C046'
    title = 'AlpacaEval length bias under verbosity prompting (LC AlpacaEval)'
    claim = ('Prompting gpt-4-1106-preview to be concise/verbose swings its AlpacaEval 2.0 win rate against itself '
             'from 22.9% to 64.3% under the weighted GPT-4-Turbo annotator; length control reduces this to 41.9-51.6% '
             'and raises the Spearman correlation with Chatbot Arena from 0.94 to 0.98.')
    paper = {'title': 'Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators',
             'authors': 'Dubois, Galambosi, Liang, Hashimoto', 'year': 2024, 'url': 'https://arxiv.org/abs/2404.04475'}
    source = Source(repo=str(REPO), path='src/alpaca_eval/evaluators_configs/weighted_alpaca_eval_gpt4_turbo/configs.yaml',
                    commit=COMMIT, url=_URL)
    default_variant = 'weighted_frozen'

    def build_variants(self):
        frozen_dev = ('judges frozen released generations (gpt4_1106_preview concise/verbose, Jan 2024); only the '
                      'annotator call is re-run',
                      'official logprob_parser applied to the first emitted token only (log_prob_index=0 instead of '
                      "'batch'); multi-token completions are flagged",
                      'original annotator gpt-4-1106-preview may be retired; any other judge is a proxy')
        weighted = Variant(
            'weighted_frozen', 'weighted_alpaca_eval_gpt4_turbo annotator on released verbosity outputs, seeded order',
            prompts={'judge': JUDGE}, params=dict(ANNOTATOR_PARAMS), strategy='frozen_output_judge',
            options={'protocol': 'source_weighted', 'sample': 805, 'parser': 'official_logprob'},
            deviations=frozen_dev)
        binary = Variant(
            'binary_paired', 'initial-reproduction modern-judge adaptation: both orders, temperature 0, text verdict',
            prompts={'judge': JUDGE}, params={'temperature': 0, 'max_tokens': 64}, strategy='paired_order_binary_judge',
            options={'protocol': 'binary_paired', 'sample': 100, 'parser': 'strict'}, repeats=3, faithful=False,
            deviations=('binary m/M text verdict instead of the normalized first-token probability',
                        'both display orders instead of the seeded single order',
                        'temperature 0 and max_tokens 64 instead of 1 / 1 with logprobs',
                        '100-instruction nested stratified subset (seed 20260930)'),
            parent='weighted_frozen')
        return {
            'weighted_frozen': weighted,
            'binary_paired': binary,
            'robust': binary.override(
                'robust', options={'parse': 'robust'}, params=_robust.cap_floor(binary.params),
                description='robust_parse with a 1,024-token cap floor for panel runs',
                deviations=('parse=robust (see robust_parse)', 'max_tokens 64 -> 1024 (a reasoning preamble was cut '
                            'off before the identifier)')),
            # reproduction pass: the paper's weighted annotator protocol on the same 100-instruction nested
            # subset as binary_paired, so a served substitute judge (gpt-4-turbo-2024-04-09) can be compared item by
            # item with the released gpt-4-1106-preview annotations at ~1/8 of the full cost.
            'weighted_frozen_sample100': weighted.override(
                'weighted_frozen_sample100', description='weighted_frozen on the 100-instruction nested subset (seed 20260930)',
                options={'sample': 100},
                deviations=('100-instruction nested stratified subset (seed 20260930), the binary_paired subset',)),
        }

    # ------------------------------------------------------------ items
    def items(self, variant, limit=None):
        protocol = variant.options['protocol']
        selected = set(selected_indices(variant.options['sample']))
        rows = []
        for condition in CONDITIONS:
            for i, r in enumerate(annotations(condition)):
                if i not in selected:
                    continue
                identical = r['output_1'] == r['output_2']
                if protocol == 'source_weighted':
                    orders = ['candidate_first' if is_switched(r['instruction']) else 'reference_first']
                else:
                    orders = ['candidate_first', 'reference_first']
                for order in orders:
                    cand_first = order == 'candidate_first'
                    rows.append({
                        'id': f'C046:i{i:03d}:{condition}:{protocol}:{order}', 'instruction_index': i,
                        'cluster_id': f'C046:i{i:03d}', 'pair_id': f'C046:i{i:03d}:{condition}',
                        'condition': condition, 'dataset': r['dataset'], 'order': order,
                        'instruction': r['instruction'],
                        'output_1': r['output_2'] if cand_first else r['output_1'],
                        'output_2': r['output_1'] if cand_first else r['output_2'],
                        'slot_identities': ['candidate', 'reference'] if cand_first else ['reference', 'candidate'],
                        'candidate_chars': len(r['output_2']), 'reference_chars': len(r['output_1']),
                        'identical_outputs': identical,
                        'archived_candidate_preference': float(r['preference']) - 1})
        return rows[:limit] if limit is not None else rows

    def program(self, variant):
        protocol = variant.options['protocol']
        template = variant.prompts['judge'].text

        def build(item):
            return chatml_messages(template, {'instruction': item['instruction'], 'output_1': item['output_1'],
                                              'output_2': item['output_2']})
        return FrozenJudge(build, variant.params, want_logprobs=protocol == 'source_weighted')

    # ------------------------------------------------------------ scoring
    @staticmethod
    def weighted_preference(logprobs):
        """(first-slot preference in [1,2] or None, status, aligned, n_tokens) via the official parser on token 0."""
        content = (logprobs or {}).get('content') or []
        if not content or 'top_logprobs' not in content[0]:
            return None, 'missing_logprobs', False, len(content)
        p = official()['logprob_parser']({'logprobs': {'content': content}}, numerator_token='m',
                                         denominator_tokens=['m', 'M'], is_binarize=False, log_prob_index=0)[0]
        if p is None or (isinstance(p, float) and math.isnan(p)):
            return None, 'missing_sentinel_logprobs', False, len(content)
        present = {t['token'] for t in content[0]['top_logprobs']} & set(SENTINELS)
        status = 'normalized_logprobs' if len(present) == 2 else 'normalized_logprobs_one_sentinel_censored'
        return float(p), status, content[0].get('token') in SENTINELS, len(content)

    def score(self, item, output, variant):
        protocol = variant.options['protocol']
        identical = item['identical_outputs']
        if protocol == 'source_weighted':
            if identical:
                return {'candidate_preference': 0.5, 'parsed': 'tie', 'parse_status': 'identical_output_shortcut',
                        'aligned': True}
            pref, status, aligned, n = self.weighted_preference(output.get('logprobs'))
            if pref is None:
                return {'candidate_preference': None, 'parsed': None, 'parse_status': status, 'aligned': False}
            first_slot_prob = 2 - pref                  # P(m) / (P(m) + P(M))
            cand = first_slot_prob if item['slot_identities'][0] == 'candidate' else 1 - first_slot_prob
            return {'candidate_preference': cand, 'first_slot_probability': first_slot_prob,
                    'parsed': 'tie' if cand == 0.5 else ('candidate' if cand > 0.5 else 'reference'),
                    'parse_status': status, 'aligned': aligned, 'n_tokens': n}
        slot, status = prior_strict(output.get('content') or '', identical)
        robust = variant.options.get('parse', 'paper') == 'robust'
        path = 'paper' if slot is not None else None
        if robust and slot is None:
            slot = parse_binary_robust(output.get('content'))
            if slot is not None:
                status, path = 'robust_identifier_line', 'robust'
        extra = {'parse_path': path} if robust else {}
        if slot is None:
            return {'candidate_preference': None, 'parsed': None, 'parse_status': status, **extra}
        if slot == 'tie':
            return {'candidate_preference': 0.5, 'parsed': 'tie', 'parse_status': status, **extra}
        who = item['slot_identities'][slot]
        return {'candidate_preference': float(who == 'candidate'), 'parsed': who, 'parse_status': status, **extra}

    # ------------------------------------------------------------ aggregation
    def _padded(self, rows, variant):
        """All intended (item, repeat) rows; missing ones get score None (never silently dropped)."""
        items = self.items(variant)
        repeats = max([variant.repeats] + [r['repeat'] + 1 for r in rows])
        seen = {(r['item']['id'], r['repeat']): r for r in rows}
        return [seen.get((x['id'], k)) or {'item': x, 'repeat': k, 'score': None, 'output': None}
                for k in range(repeats) for x in items], repeats

    def aggregate(self, rows, variant):
        if variant.options['protocol'] == 'binary_paired':
            return self._aggregate_binary(rows, variant)
        padded, _ = self._padded(rows, variant)
        out = {'metric': 'weighted win rate of the verbosity-prompted candidate vs baseline (%)',
               'standard_condition_win_rate': 50.0}
        rates = {}
        for condition in CONDITIONS:
            sel = [r for r in padded if r['item']['condition'] == condition]
            prefs = [(r['score'] or {}).get('candidate_preference') for r in sel]
            known = [p for p in prefs if p is not None]
            aligned = [(r['score'] or {}).get('candidate_preference') for r in sel
                       if (r['score'] or {}).get('aligned') and (r['score'] or {}).get('candidate_preference') is not None]
            n, k = len(sel), len(known)
            data = [r for r in sel if (r['score'] or {}).get('candidate_preference') is not None]
            point, lo, hi = stats.cluster_bootstrap(
                data, lambda r: r['item']['cluster_id'],
                lambda rs: 100 * statistics.mean(r['score']['candidate_preference'] for r in rs) if rs else float('nan'),
                n_boot=2000, seed=SEED)
            rates[condition] = 100 * statistics.mean(known) if known else None
            out[condition] = {
                'intended': n, 'parsed': k, 'win_rate': rates[condition], 'win_rate_ci95_instruction_bootstrap': [lo, hi],
                'win_rate_missingness_bounds': [100 * sum(known) / n, 100 * (sum(known) + n - k) / n] if n else None,
                'aligned_parsed': len(aligned), 'win_rate_aligned_only': 100 * statistics.mean(aligned) if aligned else None,
                'aligned_share_of_parsed': len(aligned) / k if k else None,
                # official arithmetic counts judgments whose first token is not m/M (e.g. '</think>'); flag when that
                # is more than 5% of parsed judgments so the aligned-only rate is read instead (audit minor 4)
                'win_rate_flag': ('misaligned_first_tokens_over_5pct_use_win_rate_aligned_only'
                                  if k and len(aligned) / k < 0.95 else None),
                'parse_status_counts': dict(Counter((r['score'] or {}).get('parse_status', 'missing') for r in sel))}
            out[condition].update(self.length_controlled(sel, variant))
        if rates.get('concise') is not None and rates.get('verbose') is not None:
            out['verbose_minus_concise_pp'] = rates['verbose'] - rates['concise']
        lc = {c: (out[c].get('length_controlled_win_rate')) for c in CONDITIONS}
        lc['standard'] = 50.0     # baseline vs itself: is_baseline branch of get_length_controlled_winrate
        out['length_controlled_win_rate'] = lc
        if lc.get('concise') is not None and lc.get('verbose') is not None:
            out['lc_verbose_minus_concise_pp'] = lc['verbose'] - lc['concise']
        return out

    @staticmethod
    def length_controlled(sel, variant):
        """Refit length_controlled_v1 (official get_length_controlled_winrate) on one condition's rows.

        Official conventions: output_1 = baseline, output_2 = candidate, preference = 1 + candidate preference;
        rows without a preference are excluded from the fit but included in the prediction mean (as officially)."""
        from . import _c046_lc
        if not any((r['score'] or {}).get('candidate_preference') is not None for r in sel):
            return {'length_controlled_win_rate': None, 'lc_status': 'no_parsed_preferences'}
        sel = sorted(sel, key=lambda r: r['item']['instruction_index'])
        diff = instruction_difficulty()
        pref, lb, lm_ = [], [], []
        for r in sel:
            x, sc = r['item'], r['score'] or {}
            cp = sc.get('candidate_preference')
            pref.append(float('nan') if cp is None else 1 + cp)
            lb.append(x['reference_chars'])
            lm_.append(x['candidate_chars'])
        fitted = [(p, a, b) for p, a, b in zip(pref, lb, lm_) if p == p and b == b]
        if len(fitted) < 2 or len({a - b for _, a, b in fitted}) < 2:
            # std_delta_len needs >= 2 distinct length differences (pandas std ddof=1); the official code would
            # produce NaN features and fail, so report the LC as undetermined instead
            return {'length_controlled_win_rate': None, 'lc_status': f'insufficient_rows_for_glm ({len(fitted)} fitted)',
                    'lc_rows_without_preference': sum(p != p for p in pref)}
        fit = _c046_lc.fit_length_controlled(pref, lb, lm_, [diff.iloc[r['item']['instruction_index']] for r in sel])
        return {'length_controlled_win_rate': fit['length_controlled_winrate'], 'lc_glm_weights': fit['weights'],
                'lc_glm_C': fit['C'], 'lc_rows_without_preference': sum(p != p for p in pref),
                'lc_status': 'refit_length_controlled_v1' + ('' if len(sel) == 805 else f'_on_subset_{len(sel)}'),
                'lc_note': 'regularised to the released df_gamed rows (gpt-4-1106-preview annotations), as the '
                           'official code does for any annotator'}

    def _aggregate_binary(self, rows, variant):
        """reproductions/C046/analyze.py summarize(): order-balanced pairs, matched verbose-concise differences."""
        padded, repeats = self._padded(rows, variant)
        groups = collections.defaultdict(dict)
        for r in padded:
            groups[(r['item']['pair_id'], r['repeat'])][r['item']['order']] = r
        pairs = []
        for (pair_id, repeat), by_order in sorted(groups.items()):
            rs = list(by_order.values())
            x = rs[0]['item']
            valid = len(rs) == 2 and all((r['score'] or {}).get('candidate_preference') is not None for r in rs)
            pairs.append({'pair_id': pair_id, 'repeat': repeat, 'cluster_id': x['cluster_id'], 'condition': x['condition'],
                          'pref': sum(r['score']['candidate_preference'] for r in rs) / 2 if valid else None,
                          'conflict': rs[0]['score']['parsed'] != rs[1]['score']['parsed'] if valid else None,
                          'candidate_chars': x['candidate_chars'], 'reference_chars': x['reference_chars'],
                          'identical': x['identical_outputs']})
        out = {'metric': 'binary order-balanced candidate preference (adaptation, not the weighted win rate)',
               'repeats': repeats}
        for condition in CONDITIONS:
            sel = [p for p in pairs if p['condition'] == condition]
            known = [p for p in sel if p['pref'] is not None]
            n, k, total = len(sel), len(known), sum(p['pref'] for p in known)
            longer = [(p['pref'] if p['candidate_chars'] > p['reference_chars'] else 1 - p['pref'])
                      for p in known if abs(p['candidate_chars'] - p['reference_chars']) > 30]
            _, lo, hi = stats.cluster_bootstrap(known, lambda p: p['cluster_id'],
                                                lambda ps: statistics.mean(p['pref'] for p in ps) if ps else float('nan'),
                                                n_boot=2000, seed=SEED)
            out[condition] = {'intended_pair_repeats': n, 'complete_pair_repeats': k,
                              'candidate_win_fraction_complete': total / k if k else None,
                              'candidate_win_fraction_missingness_bounds': [total / n, (total + n - k) / n] if n else None,
                              'order_conflict_rate_complete': sum(p['conflict'] for p in known) / k if k else None,
                              'longer_answer_preference_abs_delta_gt30': statistics.mean(longer) if longer else None,
                              'eligible_longer_comparisons': len(longer), 'ci95_instruction_bootstrap': [lo, hi],
                              'analytic_identical_pair_repeats': sum(p['identical'] for p in sel)}
        matched = collections.defaultdict(dict)
        for p in pairs:
            matched[(p['cluster_id'], p['repeat'])][p['condition']] = p['pref']
        diffs = [{'cluster_id': c, 'd': v['verbose'] - v['concise']} for (c, _), v in matched.items()
                 if v.get('concise') is not None and v.get('verbose') is not None]
        n, k, total = len(matched), len(diffs), sum(d['d'] for d in diffs)
        _, lo, hi = stats.cluster_bootstrap(diffs, lambda d: d['cluster_id'],
                                            lambda ds: statistics.mean(d['d'] for d in ds) if ds else float('nan'),
                                            n_boot=2000, seed=SEED)
        out.update({'matched_intended_instruction_repeats': n, 'matched_complete_instruction_repeats': k,
                    'verbose_minus_concise_complete': total / k if k else None,
                    'difference_missingness_bounds': [(total - (n - k)) / n, (total + (n - k)) / n] if n else None,
                    'difference_ci95_instruction_bootstrap': [lo, hi], '_pairs': pairs, '_diffs': diffs})
        if variant.options.get('parse', 'paper') == 'robust':
            # judgments of the repeats this run actually made (a --repeats 1 run is not 2/3 missing)
            observed = max([r['repeat'] + 1 for r in rows] or [variant.repeats])
            sent = [r for r in padded if not r['item']['identical_outputs'] and r['repeat'] < observed]
            paths = Counter((r['score'] or {}).get('parse_path') for r in sent if r['score'] is not None)
            failed = sum(r['score'] is None for r in sent)
            unparsed = paths.get(None, 0)
            out.update(parse='robust', judgments_sent=len(sent), parsed_paper=paths.get('paper', 0),
                       parsed_robust=paths.get('robust', 0), unparsed=unparsed, failed_or_missing=failed,
                       validity_rate=(len(sent) - unparsed - failed) / len(sent) if sent else None,
                       headline_reliable=_robust.reliable(len(sent), unparsed, failed))
        return out

    # ------------------------------------------------------------ prior runs and archive
    def rescore_prior(self, run, variant_name, model=None):
        variant = self.variant(variant_name)
        items = {x['id']: x for x in self.items(variant)}
        rows = []
        for rec in legacy_records(run):
            if model is not None and (rec.get('record_id') or rec.get('model')) != model:
                continue
            if rec.get('status') not in {'ok', 'truncated', 'empty'}:
                continue
            item = items[rec['case_id']]
            output = {'content': rec.get('content') or '',
                      'logprobs': _first_choice_logprobs(rec.get('response'))}
            rows.append({'item': item, 'output': output, 'repeat': rec.get('repeat', 0),
                         'score': self.score(item, output, variant)})
        return self.aggregate(rows, variant)

    def rescore_archive(self):
        """weighted_frozen applied to the released raw annotator completions (all 805 x 2)."""
        variant = self.variant('weighted_frozen')
        rows = []
        for item in self.items(variant):
            raw = annotations(item['condition'])[item['instruction_index']]['raw_completion']
            output = {'logprobs': None if raw is None else _parse_raw(raw).get('logprobs')}
            rows.append({'item': item, 'output': output, 'repeat': 0, 'score': self.score(item, output, variant)})
        return self.aggregate(rows, variant)

    _archive_cache = {}

    def archive(self):
        if 'v' not in self._archive_cache:
            self._archive_cache['v'] = self.rescore_archive()
        return self._archive_cache['v']

    _arena_note = ('Spearman over the 37 models of notebooks/benchmarks.csv with Arena Elo (Feb 2 2024) using the '
                   'leaderboard CSV at cd543a14 gives 0.9305 raw / 0.9746 LC; the authors\' notebook printed 0.936 / '
                   '0.977 on its in-notebook leaderboard (paper: 38 models). The leaderboard at this commit is a later '
                   'state than the paper\'s; the paper-time leaderboard is not released.')
    targets = (
        PaperTarget('gpt4_1106_concise_win_rate', 22.9, 0.05, 'Section 4.1 / Figure 3', 'gpt-4-1106-preview annotator',
                    'weighted_frozen', 'archived_outputs', compute=lambda e: e.archive()['concise']['win_rate'],
                    note='re-parsed from released raw top-5 logprobs; tolerance = rounding to 0.1'),
        PaperTarget('gpt4_1106_verbose_win_rate', 64.3, 0.05, 'Section 4.1 / Figure 3', 'gpt-4-1106-preview annotator',
                    'weighted_frozen', 'archived_outputs', compute=lambda e: e.archive()['verbose']['win_rate']),
        PaperTarget('gpt4_1106_lc_concise', 41.9, 0.05, 'Section 4.1 / Figure 3', 'GLM length_controlled_v1',
                    'weighted_frozen', 'archived_outputs',
                    compute=lambda e: predict_lc_winrate(f'{BASELINE}_concise'),
                    note='official predict_winrate with released GLM weights; not a refit (sklearn unavailable)'),
        PaperTarget('gpt4_1106_lc_verbose', 51.6, 0.05, 'Section 4.1 / Figure 3', 'GLM length_controlled_v1',
                    'weighted_frozen', 'archived_outputs', compute=lambda e: predict_lc_winrate(f'{BASELINE}_verbose')),
        PaperTarget('claude21_concise_win_rate', 9.2, 0.05, 'Figure 3 (bar value; text not machine-readable, from claim.json)',
                    'gpt-4-1106-preview annotator', 'weighted_frozen', 'archived_outputs',
                    compute=lambda e: archived_rescored_winrate('claude-2.1_concise')),
        PaperTarget('claude21_standard_win_rate', 15.7, 0.05, 'Figure 3 (claim.json)', 'gpt-4-1106-preview annotator',
                    'weighted_frozen', 'archived_outputs', compute=lambda e: archived_rescored_winrate('claude-2.1')),
        PaperTarget('claude21_verbose_win_rate', 24.4, 0.05, 'Figure 3 (claim.json)', 'gpt-4-1106-preview annotator',
                    'weighted_frozen', 'archived_outputs', compute=lambda e: archived_rescored_winrate('claude-2.1_verbose')),
        PaperTarget('gpt35_1106_concise_win_rate', 7.4, 0.05, 'Figure 3 (claim.json)', 'gpt-4-1106-preview annotator',
                    'weighted_frozen', 'archived_outputs',
                    compute=lambda e: archived_rescored_winrate('gpt-3.5-turbo-1106_concise')),
        PaperTarget('gpt35_1106_standard_win_rate', 9.2, 0.05, 'Figure 3 (claim.json)', 'gpt-4-1106-preview annotator',
                    'weighted_frozen', 'archived_outputs', compute=lambda e: archived_rescored_winrate('gpt-3.5-turbo-1106')),
        PaperTarget('gpt35_1106_verbose_win_rate', 12.8, 0.05, 'Figure 3 (claim.json)', 'gpt-4-1106-preview annotator',
                    'weighted_frozen', 'archived_outputs',
                    compute=lambda e: archived_rescored_winrate('gpt-3.5-turbo-1106_verbose')),
        PaperTarget('raw_gameability_pct', 25.0, 0.5, 'Section 4.1 (text 25%; notebook table 26%)', 'leaderboard',
                    'weighted_frozen', 'archived_outputs', compute=lambda e: gameability('win_rate')['gameability'],
                    note='notebook_helpers.report on the released leaderboard; tolerance = rounding to whole %',
                    known_discrepancy=('released leaderboard gives (21.35 + 29.76) / 2 = 25.56%, which rounds to 26; the '
                                       'paper text says 25% while the authors\' notebook table prints 26 for the same '
                                       'quantity. Raw gameability does not depend on leaderboard composition '
                                       '(metric_weight = 1 for win_rate), so this is a paper-text rounding/transcription '
                                       'difference, not a pipeline difference.')),
        PaperTarget('lc_gameability_pct', 10.0, 0.5, 'Section 4.1; notebook length_controlled.ipynb table', 'leaderboard',
                    'weighted_frozen', 'archived_outputs',
                    compute=lambda e: gameability('length_controlled_winrate')['gameability'],
                    known_discrepancy=('released leaderboard at cd543a14 gives (7.48 + 16.00) / 2 = 11.7%; the notebook '
                                       'printed 6.4 / 13.6 (= 10%) on its paper-time leaderboard. The gamed models\' LC '
                                       'values are unchanged; the metric_weight (ratio of across-model std of raw vs LC '
                                       'over all non-gamed models) depends on the leaderboard composition, which has '
                                       'grown since the paper.')),
        # LC refit by our numpy/scipy length_controlled_v1 (what aggregate reports for a new run) on the released
        # annotations: re-tests the LC half of the claim end to end, not only the released coefficients.
        PaperTarget('gpt4_1106_lc_concise_refit', 41.9, 0.05, 'Section 4.1 / Figure 3', 'GLM length_controlled_v1 (refit)',
                    'weighted_frozen', 'archived_outputs',
                    compute=lambda e: e.archive()['concise']['length_controlled_win_rate'],
                    note='aggregate() refit on our re-parse of the released raw logprobs (41.90); tolerance = rounding to 0.1'),
        PaperTarget('gpt4_1106_lc_verbose_refit', 51.6, 0.05, 'Section 4.1 / Figure 3', 'GLM length_controlled_v1 (refit)',
                    'weighted_frozen', 'archived_outputs',
                    compute=lambda e: e.archive()['verbose']['length_controlled_win_rate'],
                    note='aggregate() refit (51.59; released weights give 51.575)'),
        PaperTarget('arena_spearman_raw', 0.94, 0.005, 'Section 4.2 / Figure 1', 'leaderboard vs Arena Elo',
                    'weighted_frozen', 'archived_outputs', compute=lambda e: arena_spearman('win_rate')[0],
                    known_discrepancy=_arena_note),
        PaperTarget('arena_spearman_lc', 0.98, 0.005, 'Section 4.2 / Figure 1', 'leaderboard vs Arena Elo',
                    'weighted_frozen', 'archived_outputs',
                    compute=lambda e: arena_spearman('length_controlled_winrate')[0], known_discrepancy=_arena_note),
    )

    # Figure 3 cells for the other prompted models (arXiv 2404.04475v1 p.6, read from the figure image 2026-10-03):
    # raw = official parser re-run on the released annotations; LC = predict_winrate with the released weights.
    targets = targets + tuple(
        PaperTarget(f'fig3_{tag}_{cond}_{kind}', value, 0.05, 'Figure 3', 'gpt-4-1106-preview annotator', 'weighted_frozen',
                    'archived_outputs',
                    compute=(lambda n: lambda e: archived_rescored_winrate(n))(name) if kind == 'raw'
                    else (lambda n: lambda e: predict_lc_winrate(n))(name),
                    note='tolerance = rounding to 0.1' + ('; raw re-parsed from released raw logprobs' if kind == 'raw'
                                                          else '; released length_controlled_v1 weights'))
        for tag, model, raws, lcs in (
            ('mixtral', 'Mixtral-8x7B-Instruct-v0.1', (13.7, 18.3, 24.6), (23.0, 23.7, 23.2)),
            ('gpt4_0613', 'gpt4_0613', (9.4, 15.8, 23.2), (21.6, 30.2, 33.8)),
            ('claude21', 'claude-2.1', None, (18.2, 25.3, 30.3)),
            ('gpt35_1106', 'gpt-3.5-turbo-1106', None, (15.8, 19.3, 22.0)),
            ('alpaca7b', 'alpaca-7b', (2.0, 2.6, 2.9), (4.5, 5.9, 6.8)))
        for kind, values in (('raw', raws), ('lc', lcs)) if values is not None
        for cond, suffix, value in zip(('concise', 'standard', 'verbose'), ('_concise', '', '_verbose'), values)
        for name in (model + suffix,))


