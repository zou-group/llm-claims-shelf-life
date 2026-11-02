"""C086 — Lost in the middle (Liu et al. 2023, "Lost in the Middle: How Language Models Use Long Contexts").

Claim: multi-document QA accuracy is U-shaped in the position of the single answer-bearing passage (high at the start
and end, lowest in the middle); for GPT-3.5-Turbo-0613 with 20 documents the middle (index 9: 53.8%) falls below the
closed-book baseline (56.1%) (Table 6 / Table 1, arXiv 2307.03172v3).

Sources
  nelson-liu/lost-in-the-middle @ 29b8a6d0 (clone in initial-reproduction/claims/C086_*). Released: all position files
  (qa_data/{10,20,30}_total_documents/nq-open-*_gold_at_{p}.jsonl.gz, p zero-based), the oracle file, the prompt
  templates and the evaluator. NOT released: API inference scripts (so the API output cap and chat wrapping are unknown)
  and any model predictions (no archived_outputs target is possible).

Prompts
  ``src/lost_in_the_middle/prompts/qa.prompt`` and ``closedbook_qa.prompt`` (extracted verbatim, trailing newline
  stripped as ``get_qa_prompt`` does). Documents are formatted ``Document [i+1](Title: t) text`` joined by newlines,
  exactly as the authors' ``get_qa_prompt``; the oracle condition is ``get_qa_prompt`` on the single gold passage of
  nq-open-oracle; closed-book is ``get_closedbook_qa_prompt``. The tests byte-compare every rendered prompt against the
  authors' functions. The prompt is sent as one user message (paper: no system message/wrapper described).

Variants
  paper_qa20          default, faithful. All 2,655 questions x gold index {0,4,9,14,19} of 20 + oracle + closed-book;
                      temperature 0 (greedy), max_tokens 100 (the released open-model commands' --max-new-tokens; the
                      API cap is not released: documented deviation).
  paper_qa20_subset100  paper_qa20 on the initial-reproduction 100-family subset (seeded permutation, seed 20260930).
  robust_parse        paper_qa20 requests with parse='robust': the official first-line
                      scorer first; only when the first line is answerless (blank -- M017/M144 open with "\n\n" --,
                      a markdown title -- M119 opens 99% of answers with "# <title>" --, a lead-in ending in ':', a
                      reasoning tag) and wrong, the same best_subspan_em on the first answer-bearing line (scored as
                      returned even if the cap cut it, like the official first line). No such line is unknown.
                      Reliability: <= 10% unparsed or missing (_robust.reliable). Used to rescore saved the panel
                      outputs.
                      Robust != paper on rows the paper scored: (a) rows whose first line is answerless and wrong
                      (2,273 panel rows, e.g. M119 '# Title' openings, M017/M144 '\n\n' openings) are re-scored
                      on the answer-bearing line (can turn wrong -> correct, never correct -> wrong); (b) rows with no
                      answer-bearing line at all (scored wrong by the paper) become unknown.
  robust              robust_parse with max_tokens floored at 1024 (_robust.cap_floor) for new panel runs.

Scoring
  Official ``scripts/evaluate_qa_responses.py``: first line of the response (``split('\\n')[0].strip()``), then
  ``metrics.best_subspan_em`` (SQuAD normalisation with the ``regex`` module; normalised-gold substring containment).
  The authors' functions are loaded from the clone and used directly.

  Length-truncated generations: the authors' inference scripts cap generation at ``--max-new-tokens 100`` and
  ``evaluate_qa_responses.py`` scores whatever text was generated, whatever the stop reason (it never sees one). The
  faithful scorer therefore scores a truncated output (ledger status 'truncated', finish_reason length/max_tokens) on
  its first line exactly like a completed one; truncation is reported, not treated as missing (``score['truncated']``,
  ``score['first_line_complete']`` and the per-condition ``truncated`` / ``truncated_first_line_incomplete`` counts).

  Missing-output policy (project METHODOLOGY): an output is *unknown* (not wrong) when the call failed (status
  error/not_submitted or no output), the visible content is blank (e.g. a reasoning model that spent the whole cap on
  reasoning), finish_reason is content_filter, or finish_reason is 'error' (stream interrupted) without a correct first
  line. The official evaluator would score blank outputs as 0; ``accuracy_official_blank_wrong`` reports that
  convention too. Accuracies are reported complete-case with requested-denominator bounds.

  Scoring change after the 2026-10-03 audit: the port first treated every status other than 'ok' as unknown, so every
  length-truncated output (which the ledger marks status='truncated') became unknown even with a correct first line.
  The initial-reproduction analyze.py rule (``unknown_analyze_py``: status != 'ok', or length/error without a correct
  first line) is kept as a field; on all saved outputs it equals ``unknown`` (the only saved truncations are 6 blank GLM
  outputs at the 4096 cap), so rescoring parity is unchanged.
"""
import functools
import gzip
import json
import random
import re
import statistics
import typing
from collections import Counter, defaultdict

import regex

from . import _robust
from ..paths import source_repo
from ..program import SingleTurn
from ..prompts import PromptTemplate, Source
from ..registry import Experiment, PaperTarget, PriorRun, Variant, register
from ..sourcecode import load_functions
from .. import stats

REPO = source_repo('C086', 'lost-in-the-middle')
COMMIT = '29b8a6d042ce29abccee3db1a73171a107d7e6af'
URL = 'https://github.com/nelson-liu/lost-in-the-middle'
POSITIONS = {20: (0, 4, 9, 14, 19)}
# Middle position used for contrasts: index 9 is the paper's GPT-3.5-Turbo minimum (Table 6) and the claim's headline.
MIDDLE = {20: 9}
N_QUESTIONS = 2655
SUBSET_SEED = 20260930                 # initial-reproduction prepare.py SEED

QA_TEMPLATE = PromptTemplate.from_source(
    'user', Source(repo=str(REPO), path='src/lost_in_the_middle/prompts/qa.prompt', commit=COMMIT, url=URL,
                   note='read by get_qa_prompt with .rstrip("\\n")'),
    kind='lines', lines=(1, 6))
CLOSEDBOOK_TEMPLATE = PromptTemplate.from_source(
    'user_closedbook', Source(repo=str(REPO), path='src/lost_in_the_middle/prompts/closedbook_qa.prompt', commit=COMMIT,
                              url=URL, note='read by get_closedbook_qa_prompt with .rstrip("\\n")'),
    kind='lines', lines=(1, 2))


@functools.lru_cache(maxsize=None)
def official():
    """The authors' prompt builders and evaluator, loaded from the clone without importing the package."""
    prompting = load_functions(REPO / 'src/lost_in_the_middle/prompting.py', ['get_qa_prompt', 'get_closedbook_qa_prompt'],
                               extra_globals={'PROMPTS_ROOT': REPO / 'src/lost_in_the_middle/prompts', 'Document': object,
                                              'List': typing.List, 'Tuple': typing.Tuple})
    metrics = load_functions(REPO / 'src/lost_in_the_middle/metrics.py', ['normalize_answer', 'best_subspan_em'],
                             extra_globals={'regex': regex, 'List': typing.List})
    evaluate = load_functions(REPO / 'scripts/evaluate_qa_responses.py', ['get_metrics_for_example'],
                              extra_globals={'METRICS': [(metrics['best_subspan_em'], 'best_subspan_em')]})
    return {**prompting, **metrics, **evaluate}


def _read_gz(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


@functools.lru_cache(maxsize=None)
def position_rows(document_count, position):
    return _read_gz(REPO / f'qa_data/{document_count}_total_documents/'
                           f'nq-open-{document_count}_total_documents_gold_at_{position}.jsonl.gz')


@functools.lru_cache(maxsize=None)
def oracle_rows():
    return _read_gz(REPO / 'qa_data/nq-open-oracle.jsonl.gz')


@functools.lru_cache(maxsize=None)
def subset_indices(families, seed):
    """initial-reproduction prepare.py: Random(seed).shuffle(range(2655)), first `families` (nested subsets)."""
    indices = list(range(N_QUESTIONS))
    random.Random(seed).shuffle(indices)
    return tuple(indices[:families])


def format_documents(documents):
    return '\n'.join(f'Document [{i + 1}](Title: {d["title"]}) {d["text"]}' for i, d in enumerate(documents))


SCORABLE_STATUS = ('ok', 'truncated', 'empty')   # Completion.usable: the call returned an output
TRUNCATION_FINISH = ('length', 'max_tokens')      # ledger marks these status='truncated'


def first_line(text):
    """evaluate_qa_responses.get_metrics_for_example: everything up to the first newline, stripped."""
    return text.split('\n')[0].strip()


# ---------------------------------------------------------------- robustness (parse='robust', 2026-10-10)
_THINK_BLOCK = re.compile(r'<(think|thinking|reasoning)>.*?</\1>', re.S | re.I)
_OPEN_THINK = re.compile(r'<(?:think|thinking|reasoning)>.*\Z', re.S | re.I)


def answerless_line(line):
    """True for a line that cannot carry the answer: blank, a markdown heading ('# First Nobel Prize'), a lead-in that
    ends with ':' ('Based on the provided documents:', '**Answer:**'), a horizontal rule, or a reasoning tag."""
    s = line.strip()
    if not s or re.match(r'^#{1,6}(\s|$)', s) or re.fullmatch(r'[-*_=\s]{3,}', s):
        return True
    if re.fullmatch(r'</?(think|thinking|reasoning)>', s, re.I):
        return True
    plain = _robust.normalize(s).strip()
    return not plain or plain.endswith(':')


def robust_answer_line(text):
    """Fallback answer line (used only when the paper's first line is answerless): reasoning blocks removed, then the
    first line that is not answerless. Returns (line, cut) where cut is True when that line is the last line of the
    text (it may have been cut off by the output cap), or (None, False) if the text has no such line."""
    t = _OPEN_THINK.sub('', _THINK_BLOCK.sub('', text or ''))
    lines = t.split('\n')
    for i, line in enumerate(lines):
        if not answerless_line(line):
            return line.strip(), i == len(lines) - 1
    return None, False


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


@register
class C086LostInTheMiddle(Experiment):
    claim_id = 'C086'
    title = 'Lost in the middle (multi-document QA by gold position)'
    claim = ('Multi-document QA accuracy is U-shaped in the position of the answer-bearing passage; for '
             'GPT-3.5-Turbo-0613 with 20 documents, 75.8% at index 0, 53.8% at index 9, 63.2% at index 19, with the '
             'middle below closed-book (56.1%).')
    paper = {'title': 'Lost in the Middle: How Language Models Use Long Contexts', 'authors': 'Liu et al.',
             'year': 2023, 'url': 'https://arxiv.org/abs/2307.03172'}
    source = Source(repo=str(REPO), path='src/lost_in_the_middle/prompting.py', locator='get_qa_prompt', commit=COMMIT,
                    url=URL)
    default_variant = 'paper_qa20'

    def build_variants(self):
        dev = ('API inference code not released: output cap unknown; max_tokens=100 taken from the released '
               'open-model commands (--max-new-tokens 100)',
               'chat wrapping of API models not described; prompt sent as a single user message',
               'original snapshots (gpt-3.5-turbo-0613, claude-1.3) retired; any rerun uses a proxy/new model',
               'max_tokens=100 includes reasoning tokens on reasoning models, which then often return a blank visible '
               'answer (scored unknown); use a reasoning adapter such as initial_subset100_cap4096 for those models')
        base = Variant('paper_qa20', '20 documents, gold at 0/4/9/14/19 + oracle + closed-book, all 2,655 questions, greedy',
                       prompts={'user': QA_TEMPLATE, 'user_closedbook': CLOSEDBOOK_TEMPLATE},
                       params={'temperature': 0, 'max_tokens': 100},
                       options={'document_count': 20, 'families': None, 'seed': None, 'controls': True},
                       deviations=dev)
        variants = {
            'paper_qa20': base,
            'paper_qa20_subset100': base.override(
                'paper_qa20_subset100', description='paper_qa20 on the seeded 100-family subset of initial-reproduction',
                options={'families': 100, 'seed': SUBSET_SEED}),
        }
        # Robustness variants. robust_parse: the panel requests
        # (paper_qa20) with parse='robust', used to rescore saved outputs offline. robust: for new panel runs.
        variants['robust_parse'] = base.override(
            'robust_parse', description='paper_qa20 requests, parse=robust (official first-line scorer first; when the '
            'first line is answerless, the first answer-bearing line)', options={'parse': 'robust'},
            deviations=('scoring: the official first-line best_subspan_em first; only when the first line is '
                        'answerless (blank, markdown heading, lead-in ending in ":", reasoning tag) and wrong, the same '
                        'rule on the first answer-bearing line (reasoning blocks removed; scored as returned even if '
                        'cut by the cap). Rows with no answer-bearing line are unknown, not wrong',
                        'reliability under parse=robust: <= 10% of rows unparsed or missing (_robust.reliable)'))
        variants['robust'] = variants['robust_parse'].override(
            'robust', description='robust_parse + max_tokens floor 1024 so a verbose answer line is not cut off',
            params=_robust.cap_floor(base.params),
            deviations=('max_tokens 100 -> 1024 (_robust.cap_floor): verbose chat models (markdown title or lead-in '
                        'first, then the answer) were cut off at 100 tokens; the scorer still reads one line',))
        return variants

    # ------------------------------------------------------------------ data
    def items(self, variant, limit=None):
        n_docs = variant.options['document_count']
        families = variant.options['families']
        order = (subset_indices(families, variant.options['seed']) if families else tuple(range(N_QUESTIONS)))
        oracle = oracle_rows()
        rows = []
        for rank, i in enumerate(order):
            conditions = [(f'qa{n_docs}_gold{p}', p) for p in POSITIONS[n_docs]]
            if variant.options['controls']:
                conditions += [('oracle', 0), ('closedbook', None)]
            for condition, p in conditions:
                if limit is not None and len(rows) >= limit:
                    return rows
                if condition == 'closedbook':
                    src, docs = oracle[i], []
                elif condition == 'oracle':
                    src = oracle[i]
                    docs = src['ctxs']
                else:
                    src = position_rows(n_docs, p)[i]
                    docs = src['ctxs']
                rows.append({'id': f'C086/nq/{i:04d}/{condition}', 'pair_id': f'C086/nq/{i:04d}', 'source_index': i,
                             'sample_rank': rank, 'condition': condition, 'document_count': len(docs),
                             'gold_position': p, 'question': src['question'], 'answers': list(src['answers']),
                             'documents': [{'title': d['title'], 'text': d['text'], 'isgold': d.get('isgold'),
                                            'hasanswer': d.get('hasanswer')} for d in docs]})
        return rows

    def render(self, item, variant):
        if item['condition'] == 'closedbook':
            return variant.prompts['user_closedbook'].render(question=item['question'])
        return variant.prompts['user'].render(question=item['question'],
                                              search_results=format_documents(item['documents']))

    def program(self, variant):
        return SingleTurn(lambda item: [{'role': 'user', 'content': self.render(item, variant)}], variant.params)

    # --------------------------------------------------------------- scoring
    def score(self, item, output, variant=None):
        text = output.get('content') if isinstance(output.get('content'), str) else ''
        line = first_line(text)
        correct = bool(official()['best_subspan_em'](prediction=line, ground_truths=item['answers']))
        status, finish = output.get('status', 'ok'), output.get('finish_reason')
        blank = not text.strip()
        truncated = status == 'truncated' or finish in TRUNCATION_FINISH
        unknown = (status not in SCORABLE_STATUS or blank or finish == 'content_filter'
                   or (finish == 'error' and not correct))
        unknown_analyze_py = (status != 'ok' or blank or finish == 'content_filter'
                              or (finish in ('length', 'error') and not correct))
        out = {'correct': correct, 'first_line': line, 'unknown': unknown, 'blank': blank,
               'truncated': truncated, 'first_line_complete': '\n' in text or not truncated,
               'unknown_analyze_py': unknown_analyze_py, 'status': status, 'finish_reason': finish}
        if (variant is not None and variant.options.get('parse', 'paper') == 'robust'
                and status in SCORABLE_STATUS):
            out.update(self._robust_score(item, text, line, correct, truncated, unknown))
        return out

    @staticmethod
    def _robust_score(item, text, line, correct, truncated, unknown):
        """parse='robust': the paper's first line stands unless it is answerless and wrong; then the first
        answer-bearing line is scored with the same official rule. Returns the fields that change."""
        if unknown:                              # blank / content filter / interrupted: stays unknown
            return {'parse_path': None, 'unparsed': True}
        if correct or not answerless_line(line):
            return {'parse_path': 'paper', 'unparsed': False}
        alt, cut = robust_answer_line(text)
        if alt is None:
            return {'parse_path': None, 'unparsed': True, 'unknown': True}
        ok = bool(official()['best_subspan_em'](prediction=alt, ground_truths=item['answers']))
        # A fallback line cut off by the cap is scored as returned, exactly as the official scorer treats a cut first
        # line (dropping cut lines only when wrong would select rows on the outcome).
        return {'parse_path': 'robust', 'unparsed': False, 'correct': ok, 'robust_line': alt,
                'robust_line_cut': bool(cut and truncated)}

    def aggregate(self, rows, variant):
        """Per-condition accuracy (complete-case, requested-denominator bounds) and paired family contrasts.

        rows without a score (failed calls) are unknown. Families are the clusters.
        """
        n_docs = variant.options['document_count']
        values = {}            # (pair_id, condition, repeat) -> 1/0/None(unknown)
        truncation = defaultdict(Counter)   # condition -> truncated / truncated_first_line_incomplete
        official_values = {}   # official evaluator convention: any returned output scored, failed calls excluded
        for r in rows:
            s = r.get('score')
            key = (r['item']['pair_id'], r['item']['condition'], r.get('repeat', 0))
            if key in values:
                raise ValueError(f'duplicate trial {key}')
            values[key] = None if not s or s['unknown'] else int(s['correct'])
            if s and s['status'] in SCORABLE_STATUS:
                official_values[key] = int(s['correct'])
            if s and s.get('truncated'):
                truncation[key[1]]['truncated'] += 1
                if not s.get('first_line_complete', True):
                    truncation[key[1]]['truncated_first_line_incomplete'] += 1
        conditions = defaultdict(list)
        for key in values:
            conditions[key[1]].append(key)
        out = {'conditions': {}, 'contrasts': {}}
        for cond, keys in sorted(conditions.items()):
            known = [values[k] for k in keys if values[k] is not None]
            k, n = sum(known), len(keys)
            off = [official_values[k] for k in keys if k in official_values]
            out['conditions'][cond] = {
                'requested': n, 'correct': k, 'wrong': len(known) - k, 'unknown': n - len(known),
                'accuracy': k / len(known) if known else None,
                'requested_bounds': list(stats.bounds_with_missing(k, len(known) - k, n - len(known))),
                'wilson95': list(stats.wilson(k, len(known))) if known else None,
                'accuracy_official_blank_wrong': sum(off) / len(off) if off else None,
                'truncated': truncation[cond]['truncated'],
                'truncated_first_line_incomplete': truncation[cond]['truncated_first_line_incomplete']}
        mid = f'qa{n_docs}_gold{MIDDLE[n_docs]}'
        first, last = f'qa{n_docs}_gold0', f'qa{n_docs}_gold{POSITIONS[n_docs][-1]}'
        contrasts = {'first_minus_middle': {first: 1, mid: -1}, 'last_minus_middle': {last: 1, mid: -1},
                     'mean_endpoints_minus_middle': {first: .5, last: .5, mid: -1},
                     'middle_minus_closedbook': {mid: 1, 'closedbook': -1}}
        families = sorted({k[0] for k in values})
        repeats = sorted({k[2] for k in values})
        for name, weights in contrasts.items():
            if not all(c in conditions for c in weights):
                continue
            lows, highs, points = [], [], []
            for fam in families:
                per_rep = []
                for rep in repeats:
                    lo = hi = 0.0
                    for c, w in weights.items():
                        v = values.get((fam, c, rep))
                        if v is None:
                            lo += min(0, w)
                            hi += max(0, w)
                        else:
                            lo += w * v
                            hi += w * v
                    per_rep.append((lo, hi))
                lo, hi = statistics.mean(x[0] for x in per_rep), statistics.mean(x[1] for x in per_rep)
                lows.append(lo)
                highs.append(hi)
                if lo == hi:
                    points.append({'family': fam, 'value': lo})
            mean = lambda ps: statistics.mean(p['value'] for p in ps) if ps else float('nan')
            point, ci_lo, ci_hi = stats.cluster_bootstrap(points, lambda p: p['family'], mean, n_boot=2000, seed=20260930)
            out['contrasts'][name] = {'families': len(families), 'complete_families': len(points),
                                      'requested_bounds': [statistics.mean(lows), statistics.mean(highs)],
                                      'complete_case': point if points else None,
                                      'complete_case_ci95_family_bootstrap': [ci_lo, ci_hi]}
        if variant.options.get('parse', 'paper') == 'robust':
            scored = [r['score'] for r in rows if r.get('score')]
            n, missing = len(rows), len(rows) - len(scored)
            parse = {'parsed_paper': sum(s.get('parse_path') == 'paper' for s in scored),
                     'parsed_robust': sum(s.get('parse_path') == 'robust' for s in scored),
                     'unparsed': sum(bool(s.get('unparsed')) for s in scored), 'missing': missing}
            out['parse'] = parse
            out.update({k: parse[k] for k in ('parsed_paper', 'parsed_robust', 'unparsed')})
            out['validity_rate'] = (parse['parsed_paper'] + parse['parsed_robust']) / n if n else None
            out['headline_reliable'] = _robust.reliable(n, parse['unparsed'], missing)
        return out

    # ------------------------------------------------- targets and prior runs
    targets = ()  # filled below


_TOL = 0.10   # ~2 binomial SE for a 100-question subset vs the 2,655-question population (SE <= 0.05)
C086LostInTheMiddle.targets = (
    PaperTarget('claude13_qa20_gold9_accuracy', 0.568, _TOL, 'Table 6', 'claude-1.3', 'paper_qa20', 'unverifiable',
                note='claude-1.3 retired; predictions not released'),
)
