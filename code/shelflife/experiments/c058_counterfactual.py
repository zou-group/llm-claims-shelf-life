"""C058 — Counterfactual arithmetic (Wu et al. 2023, "Reasoning or Reciting?", NAACL 2024).

Claim: models that add two-digit numbers almost perfectly in base 10 do much worse in a counterfactual base (GPT-4
base 9: 23.4% without and 38.6% with zero-shot CoT, vs 100.0% / 98.2% in base 10; Table 18), while passing the
counterfactual comprehension check (successor in base 9: 90.0%).

Sources: ZhaofengWu/counterfactual-evaluation @ cc0122ad (clone in initial-reproduction/claims/C058_*).
  prompts   arithmetic/query.py::templatize (addition, cot True/False) and arithmetic/query_ccc.py::templatize (CCC)
  data      arithmetic/data/0shot/base{B}.txt (1,000 rows, seed 0, sampled with replacement, counterfactual rows
            rejection-sampled so the answer differs from the decimal reading), arithmetic/data/ccc/base{B}.txt (200)
  scorer    arithmetic/eval.py::parse_output + get_label (exact string match after .upper()),
            arithmetic/eval_ccc.py::parse_output (no .upper())
  outputs   arithmetic/model_interactions/{0shot/without_0cot,0shot/with_0cot,ccc}/{gpt4,gpt3.5,claude,palm2}_base{B}.tsv
            (query_utils prompt-keyed cache rows, 9 columns, query_utils.escape = unicode_escape). Row i corresponds
            to data row i (query_batch writes interactions in prompt order); tests check every prompt.

Variants
  paper_addition            default. Bases 9 and 10 x {plain, 0-CoT}, all 1,000 released rows per cell, temperature 0.
  paper_ccc                 successor comprehension check, bases 9 and 10, all 200 released rows.
  paper_addition_plain_s50  earlier pipeline option-only subset (faithful): plain only, 50 seeded rows per base 9/10,
                            interleaved so --limit stays balanced.

Scoring (``parser='official'``): the authors' parse_output, loaded from the clone. Outputs it marks "FAILED" (empty
or one of the authors' manually excluded hashes) and outputs on which it raises (it ends in ``assert False``; the
authors added hashes until their own outputs no longer raised) are unparsed and count as incorrect, exactly as in
eval.py, where FAILED != label and every row is in the denominator. Raising on a new output is a scoring deviation we
cannot avoid; such rows are reported separately (``unparseable``) and in missingness bounds.
Metric: accuracy per (base, CoT) cell over all rows, with an expression-cluster bootstrap CI (the released data
repeat expressions; base 9 has 900 unique expressions in 1,000 rows, base 10 932), and base10 - base9 gaps.

Robustness variants (non-faithful)
  * ``robust_parse``: paper_addition_plain_s50 requests unchanged, option parse='robust'. The official parse_output
    runs first; only when it returns FAILED or raises, a fallback reads the LAST ``\boxed{...}`` (subscript base,
    \text, spaces and a trailing '_9' removed) or, failing that, the first base-digit token after the last explicit
    answer marker (_robust.answer_segment); the token must consist of valid digits of the base. No marker -> no
    answer (a mid-reasoning number is never taken). Rows still without an answer, and truncated rows (finish_reason
    'length') whose fallback found nothing, are ``unparsed`` and count as missing (not as incorrect) in the robust
    accuracy; the paper-rule accuracy is kept as ``accuracy_paper_rule``. headline_reliable = _robust.reliable per
    cell (<= 10% unparsed + missing). parsed_paper / parsed_robust / unparsed and validity_rate per cell and overall.
    Where robust differs from paper on rows the paper scorer DID score: none per row (the fallback runs only where
    parse_output gave FAILED or raised; a truncated row it parsed keeps its answer). Rows the paper counted incorrect
    because parse_output FAILED are, when the fallback also finds nothing, missing instead of incorrect in the
    robust accuracy (``accuracy_paper_rule`` keeps the paper count).
  * ``robust``: robust_parse with max_tokens 2048 (case params carried none, so non-reasoning models got the
    policy's 1024 and M045 was cut off once) and min_cap_reasoning 16384 (M183 / M144 spent the 4096 policy cap on
    hidden reasoning: 24 / 9 of 100 outputs empty), plus option truncated='score' (_robust.truncation_missing): at
    these caps a row cut off before any answer (finish_reason length, even with empty content) failed the task, so it
    is INCORRECT, as the paper rule scores a FAILED parse, and stays in the robust accuracy; counted as
    ``truncated_scored``. A complete row with no answer stays unparsed (missing). Documented exception for robust:
    such cut-off rows are incorrect (paper), not missing. robust_parse keeps truncated='missing'.
"""
import functools
import hashlib
import re
import random
from collections import defaultdict

import numpy as np

from ..paths import source_repo
from ..program import SingleTurn
from ..prompts import PromptTemplate, Source
from ..registry import Experiment, PaperTarget, Variant, register
from ..sourcecode import load_functions
from .. import stats
from . import _robust

REPO = source_repo('C058', 'counterfactual-evaluation')
COMMIT = 'cc0122ad6a65ea80e2f0f223e79858406a0df326'
URL = 'https://github.com/ZhaofengWu/counterfactual-evaluation'
ARITH = REPO / 'arithmetic'
DIGITS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
SUBSET_SEED = 20260929   # initial-reproduction prepare.py; per-base seed is SUBSET_SEED + base
ARCHIVE_MODELS = {'gpt4': 'gpt-4-0314', 'gpt3.5': 'gpt-3.5-turbo-0301', 'claude': 'claude-v1.3',
                  'palm2': 'models/text-bison-001'}

_TOKENS = {'{base}': 'base', '{digits}': 'digits', '{expr}': 'expr'}
_START = 'You are a mathematician. Assuming that all numbers are in base-{base} where the digits are "{digits}", '


def _src(path, locator):
    return Source(repo=str(REPO), path=path, locator=locator, commit=COMMIT, url=URL)


# Transcribed from the f-strings in templatize (they are built in code, so not extractable by py_var); the claim tests
# check byte equality against the authors' templatize for every released row of every base.
ADD_PLAIN = PromptTemplate('user', _START + 'what is {expr}? End the response with the result in "\\boxed{result}".',
                           style='replace', tokens=_TOKENS,
                           source=_src('arithmetic/query.py', 'templatize (cot=False, n_shots=0)'))
ADD_COT = PromptTemplate('user', _START + 'what is {expr}? Let\'s think step by step, and end the response with the '
                                          'result in "\\boxed{result}".', style='replace', tokens=_TOKENS,
                         source=_src('arithmetic/query.py', 'templatize (cot=True, n_shots=0)'))
CCC = PromptTemplate('user', _START + 'what is the next number after {expr}? Do this by counting the few preceding '
                                      'numbers and completing the sequence. End the response with the result.',
                     style='replace', tokens=_TOKENS, source=_src('arithmetic/query_ccc.py', 'templatize'))

_quiet = {'print': lambda *a, **k: None}


@functools.lru_cache(maxsize=None)
def official():
    """The authors' prompt builders, label function, parsers and TSV (un)escape, loaded from the clone."""
    ev = load_functions(ARITH / 'eval.py', ['parse_output', 'get_label', 'unescape'],
                        extra_globals={'hashlib': hashlib, 'np': np, **_quiet})
    ccc = load_functions(ARITH / 'eval_ccc.py', ['parse_output'], extra_globals=_quiet)
    query = load_functions(ARITH / 'query.py', ['templatize', 'answer'], extra_globals={'get_label': ev['get_label']})
    query_ccc = load_functions(ARITH / 'query_ccc.py', ['templatize'])
    qu = load_functions(REPO / 'query_utils.py', ['escape', 'unescape'])
    return {'templatize': query['templatize'], 'templatize_ccc': query_ccc['templatize'],
            'get_label': ev['get_label'], 'parse_output': ev['parse_output'], 'parse_output_ccc': ccc['parse_output'],
            'eval_unescape': ev['unescape'], 'tsv_escape': qu['escape'], 'tsv_unescape': qu['unescape']}


def official_parse(text, task='addition'):
    """(prediction or None, status) under the authors' eval.py / eval_ccc.py.

    status: 'parsed' | 'failed' (parse_output returned FAILED: empty or manually excluded hash) |
            'unparseable' (parse_output raised; the original evaluation would have stopped here).

    The authors' I/O is mirrored: eval.py reads each TSV line with ``line.strip("\n ")`` and eval_ccc.py with
    ``line.strip()``; responses are written with ``unicode_escape`` (query_utils.escape), so a space is the only raw
    whitespace that can trail the response field and is the only thing those strips remove from it. Hence
    ``text.rstrip(' ')``: '101 ' parses as 101. A real trailing newline survives (escaped in the TSV), so '101\n'
    gives pred '101\n' (status parsed, scored incorrect), exactly as in the authors' pipeline.
    """
    fn = official()['parse_output_ccc' if task == 'ccc' else 'parse_output']
    text = text.rstrip(' ') if isinstance(text, str) else text
    try:
        pred = fn(text)
    except Exception:   # any raise from the authors' regex chain (assert False, IndexError, TypeError, ...)
        return None, 'unparseable'
    if task != 'ccc' and pred == 'FAILED':
        return None, 'failed'
    return (pred if task == 'ccc' else pred.upper()), 'parsed'


ROBUST_MAX_TOKENS = 2048
ROBUST_MIN_CAP_REASONING = 16384
_BOXED_ALL = re.compile(r'\\boxed\s*\{((?:[^{}]|\{[^{}]*\})*)\}')


def robust_answer(text, base):
    """Fallback answer for parse='robust' (used only when the official parser gives FAILED or raises): the last
    \\boxed{...} content, else the first token after the last explicit answer marker; subscript bases, \\text, spaces
    and markdown removed; returned (upper-cased) only if it is a valid base-``base`` numeral, else None."""
    if not text:
        return None
    digits = DIGITS[:base]

    def clean(s):
        s = re.sub(r'\\text\{([^{}]*)\}', r'\1', s)
        s = re.sub(r'_\{?\s*\(?\s*(?:base[- ]?)?\d+\s*\)?\s*\}?', '', s)      # 101_9, 101_{9}, 101_{(9)}
        s = re.sub(r'\(\s*base[- ]?\d+\s*\)|\bin\s+base[- ]?\d+', '', s, flags=re.I)
        s = re.sub(r'[\s*`$,]|\\[,!;]', '', s).replace('\u2212', '-').rstrip('.')
        s = re.sub(r'[\u2080-\u2089]+$', '', s)                       # 110₉ (subscript base)
        return s.upper()

    boxes = _BOXED_ALL.findall(str(text))
    cands = [clean(boxes[-1])] if boxes else []
    seg, t = _robust.answer_segment(text), _robust.normalize(text)
    if not boxes and seg != t:
        m = re.match(r'\s*([0-9A-Za-z]+)(?:\s*[_\u2080-\u2089(]|\b)', seg)
        if m:
            cands.append(clean(m.group(1)))
    for c in cands:
        if c and all(ch in digits for ch in c):
            return c
    return None


def label(expr, base, task='addition'):
    get_label = official()['get_label']
    return get_label(f'{expr}+1', base) if task == 'ccc' else get_label(expr, base)


def _data(task, base):
    folder = 'ccc' if task == 'ccc' else '0shot'
    return [line.strip() for line in (ARITH / 'data' / folder / f'base{base}.txt').read_text().splitlines()]


def _cell(task, base, cot):
    return f'base{base}_ccc' if task == 'ccc' else f'base{base}_{"cot" if cot else "plain"}'


# ---------------------------------------------------------------------------- earlier pipeline runs
# experiments/run_minimal_setups/runs/earlier/runs/<claim>-<variant>-L<N>[-<tag>]/<model>/: one shelflife.runner directory per model.
# Registered: every directory that sent at least one request (complete or partial); tests/test_prior_runs.py
# replays each one against this config. Excluded: directories that sent nothing. Listed explicitly (written
# 2026-10-05 from the run directories; nothing is globbed at import).

_R2 = 'experiments/run_minimal_setups/runs/earlier/runs'
_NOTHING_SENT = ('no request reached a provider: every call is not_submitted (circuit breaker open, or stale ledger '
                 'rows after a package-hash change); the directory only records the attempts')


def _unsent(*runs):
    """excluded_runs entries for per-model directories that sent nothing: (run, space-separated models)."""
    return {f'{_R2}/{run}/{m}': _NOTHING_SENT for run, models in runs for m in models.split()}


@register
class C058Counterfactual(Experiment):
    claim_id = 'C058'
    title = 'Counterfactual arithmetic: base 9 vs base 10 addition'
    claim = ('LLMs do much worse on counterfactual variants of tasks they solve by default even when they pass a '
             'comprehension check: GPT-4 two-digit addition is 100% in base 10 but 23.4% in base 9 (38.6% vs 98.2% '
             'with zero-shot CoT), with 90% on the base-9 successor check.')
    paper = {'title': 'Reasoning or Reciting? Exploring the Capabilities and Limitations of Language Models Through '
                      'Counterfactual Tasks', 'authors': 'Wu et al.', 'year': 2023,
             'url': 'https://arxiv.org/abs/2307.02477'}
    source = Source(repo=str(REPO), path='arithmetic', locator='query.py / eval.py', commit=COMMIT, url=URL)
    default_variant = 'paper_addition'

    def build_variants(self):
        deviations = (
            'original snapshots (gpt-4-0314, gpt-3.5-turbo-0301, claude-v1.3, text-bison-001) are retired',
            'original OpenAI calls sent no max_tokens (claude-v1.3 got 9000, PaLM the API default); case params '
            'carry only temperature 0 and the transport policy sets the output cap',
            'the authors\' prompt-keyed cache answered duplicate expressions once; each released row is queried here '
            '(identical at temperature 0 up to provider nondeterminism)',
            'official parse_output raises on formats the authors never saw; those rows are scored unparsed (incorrect)',
        )
        add = Variant('paper_addition', 'Table 18: two-digit addition, bases 9 and 10, plain and 0-CoT, 1,000 rows each',
                      prompts={'plain': ADD_PLAIN, 'cot': ADD_COT}, params={'temperature': 0},
                      options={'task': 'addition', 'bases': (9, 10), 'cots': (False, True), 'subset': None,
                               'parser': 'official'},
                      deviations=deviations)
        return {
            'paper_addition': add,
            'paper_ccc': Variant('paper_ccc', 'Successor comprehension check (CCC), bases 9 and 10, 200 rows each',
                                 prompts={'ccc': CCC}, params={'temperature': 0},
                                 options={'task': 'ccc', 'bases': (9, 10), 'cots': (False,), 'subset': None,
                                          'parser': 'official'},
                                 deviations=deviations),
            'paper_addition_plain_s50': (s50 := add.override(
                'paper_addition_plain_s50', description='earlier pipeline option-only subset of paper_addition: plain '
                '(no-CoT) prompts only, bases 9 and 10, 50 rows per base = the 50 lowest source indices of the '
                'initial-reproduction seeded 200-row sample (random.Random(20260929 + base)); rows interleaved '
                'base9, base10, base9, ... so that --limit 2k gives k rows per base. Prompts, params and strategy '
                'are paper_addition\'s, so it stays faithful. Headline: gap_base10_minus_base9_plain',
                options={'cots': (False,), 'subset': ('seeded', 200, 50), 'order': 'interleave_cells'})),
            'robust_parse': (rparse := s50.override(
                'robust_parse', description='paper_addition_plain_s50 requests, parse=robust (official parser first, '
                'boxed / answer-marker fallback; unparsed rows missing; tolerant reliability)',
                options={'parse': 'robust'},
                deviations=('scoring: official parse_output first; when it gives FAILED or raises, the last \\boxed{} '
                            'or the token after an explicit answer marker, if a valid base-B numeral; unparsed rows '
                            '(empty, cut off, no answer) are missing instead of incorrect; reliability = <=10% unparsed '
                            'or missing per cell',))),
            'robust': rparse.override(
                'robust', description='robust_parse + max_tokens 2048 and min_cap_reasoning 16384 so the boxed answer '
                'is reached on modern models', params={'max_tokens': ROBUST_MAX_TOKENS},
                options={'min_cap_reasoning': ROBUST_MIN_CAP_REASONING, 'truncated': 'score'},
                deviations=('output cap set to 2048 (paper sent none; policy default 1024) and reasoning-model cap '
                            '4096 -> 16384: reasoning models exhausted the cap before answering',
                            "truncated='score': a row cut off before any answer at these caps is incorrect (paper "
                            'rule), not unparsed')),
        }

    # ------------------------------------------------------------------ data
    def items(self, variant, limit=None):
        o = variant.options
        task = o['task']
        cells = []
        for base in o['bases']:
            data = _data(task, base)
            indices = list(range(len(data)))
            if o['subset'] is not None:
                _, n, head = o['subset']
                indices = sorted(random.Random(SUBSET_SEED + base).sample(range(len(data)), n))[:head]
            for cot in o['cots']:
                cell = _cell(task, base, cot)
                cells.append([{'id': f'C058_{cell}_{i:04d}', 'task': task, 'condition': cell, 'base': base,
                               'cot': cot, 'expr': data[i], 'source_index': i, 'expected': label(data[i], base, task),
                               'cluster': f'{base}:{data[i]}'} for i in indices])
        if o.get('order') == 'interleave_cells':
            # round-robin over cells (same row set as the blocked order), so a global --limit stays balanced
            if len({len(c) for c in cells}) != 1:
                raise ValueError('interleave_cells needs equal cell sizes')
            rows = [row for group in zip(*cells) for row in group]
        else:   # source order: cell blocks (base-major, then plain/CoT)
            rows = [row for c in cells for row in c]
        return rows[:limit] if limit is not None else rows

    def template(self, variant, item):
        return variant.prompts['ccc' if item['task'] == 'ccc' else ('cot' if item['cot'] else 'plain')]

    def program(self, variant):
        def build(item):
            content = self.template(variant, item).render(base=item['base'], digits=DIGITS[:item['base']],
                                                          expr=item['expr'])
            return [{'role': 'user', 'content': content}]
        extra = ({'min_cap_reasoning': variant.options['min_cap_reasoning']}
                 if variant.options.get('min_cap_reasoning') else None)
        return SingleTurn(build, variant.params, case_extra=extra)

    # --------------------------------------------------------------- scoring
    def score(self, item, output, variant):
        text = output.get('content') or ''
        pred, status = official_parse(text, item['task'])
        out = {'pred': pred, 'status': status, 'parsed': pred is not None, 'correct': pred == item['expected'],
               'expected': item['expected']}
        if variant.options.get('parse', 'paper') == 'robust' and item['task'] != 'ccc':
            path = 'paper' if pred is not None else None
            if pred is None:
                rp = robust_answer(text, item['base'])
                if rp is not None:
                    path = 'robust'
                    out.update(pred=rp, parsed=True, correct=rp == item['expected'])
            cut = output.get('finish_reason') == 'length' or output.get('status') == 'truncated'
            if path is None and cut and not _robust.truncation_missing(variant):
                # truncated='score' (robust, 2048 / 16384 caps): cut off before any answer = incorrect, as the paper
                # rule (parse_output FAILED) scores it, not unparsed
                path = 'paper'
                out['truncated_scored'] = True
            out.update(parse_path=path, unparsed=path is None, truncated=output.get('finish_reason') == 'length')
        return out

    def aggregate(self, rows, variant):
        cells = defaultdict(list)
        for r in rows:
            cells[r['item']['condition']].append(r)
        out = {}
        for cell, rs in sorted(cells.items()):
            scored = [r for r in rs if r.get('score')]
            k = sum(r['score']['correct'] for r in scored)
            n = len(scored)
            parsed = sum(r['score']['parsed'] for r in scored)
            acc = lambda xs: sum(x['score']['correct'] for x in xs) / len(xs) if xs else float('nan')
            _, lo, hi = stats.cluster_bootstrap(scored, lambda r: r['item']['cluster'], acc, n_boot=2000, seed=20261003)
            out[cell] = {'n_rows': len(rs), 'n_scored': n, 'missing_outputs': len(rs) - n, 'correct': k,
                         'accuracy': k / n if n else None, 'ci95_expression_bootstrap': [lo, hi],
                         'parsed': parsed, 'unparsed': n - parsed,
                         'unparseable_official': sum(r['score']['status'] == 'unparseable' for r in scored),
                         'accuracy_among_parsed': k / parsed if parsed else None,
                         'accuracy_bounds_unparsed': list(stats.bounds_with_missing(k, parsed - k, n - parsed)) if n else None,
                         # all requested rows (paper denominator): unparsed and missing outputs both unknown
                         'accuracy_bounds_all_rows': list(stats.bounds_with_missing(k, parsed - k, len(rs) - parsed))
                         if rs else None,
                         'unique_expressions': len({r['item']['cluster'] for r in rs})}
            if variant.options.get('parse', 'paper') == 'robust' and scored and 'parse_path' in scored[0]['score']:
                ok = [r for r in scored if not r['score']['unparsed']]
                kk = sum(r['score']['correct'] for r in ok)
                _, rlo, rhi = stats.cluster_bootstrap(ok, lambda r: r['item']['cluster'], acc, n_boot=2000,
                                                      seed=20261003) if ok else (None, None, None)
                unp = n - len(ok)
                out[cell].update({
                    'accuracy_paper_rule': out[cell]['accuracy'], 'ci95_paper_rule': out[cell]['ci95_expression_bootstrap'],
                    # robust headline: unparsed (empty / cut off / no answer) rows are missing, not incorrect
                    'accuracy': kk / len(ok) if ok else None, 'ci95_expression_bootstrap': [rlo, rhi],
                    'n_scored': len(ok), 'correct': kk,
                    'parsed_paper': sum(r['score']['parse_path'] == 'paper' for r in scored),
                    'parsed_robust': sum(r['score']['parse_path'] == 'robust' for r in scored),
                    'unparsed': unp, 'truncated': sum(r['score']['truncated'] for r in scored),
                    'truncated_scored': sum(bool(r['score'].get('truncated_scored')) for r in scored),
                    'validity_rate': len(ok) / len(rs) if rs else None,
                    'reliable': _robust.reliable(len(rs), unp, len(rs) - n)})
        if variant.options.get('parse', 'paper') == 'robust':
            cs = [c for c in out.values() if 'parsed_paper' in c]
            for k in ('parsed_paper', 'parsed_robust', 'unparsed', 'truncated_scored'):
                out[k] = sum(c[k] for c in cs)
            n_rows = sum(c['n_rows'] for c in cs)
            out['validity_rate'] = (out['parsed_paper'] + out['parsed_robust']) / n_rows if n_rows else None
            out['headline_reliable'] = bool(cs) and all(c['reliable'] for c in cs)
            out['headline_caveats'] = [f"{k}: {c['unparsed']} unparsed + {c['missing_outputs']} missing of "
                                       f"{c['n_rows']} > {_robust.ROBUST_TOLERANCE:.0%}"
                                       for k, c in sorted(out.items()) if isinstance(c, dict) and c.get('reliable') is False]
        for cot in ('plain', 'cot'):
            a, b = out.get(f'base9_{cot}'), out.get(f'base10_{cot}')
            if a and b and a['accuracy'] is not None and b['accuracy'] is not None:
                out[f'gap_base10_minus_base9_{cot}'] = b['accuracy'] - a['accuracy']
        return out

    # ------------------------------------------------------- archived outputs
    def archived_rows(self, archive_model, task, base, cot=False):
        """Released outputs of an original model scored with our scorer. Every TSV prompt must equal our rendering."""
        variant = self.variant('paper_ccc' if task == 'ccc' else 'paper_addition')
        items = [x for x in self.items(variant.override('_tmp', options={'bases': (base,)})) if x['cot'] == cot]
        sub = 'ccc' if task == 'ccc' else ('0shot/with_0cot' if cot else '0shot/without_0cot')
        path = ARITH / 'model_interactions' / sub / f'{archive_model}_base{base}.tsv'
        lines = path.read_text().split('\n')
        lines = lines[:-1] if lines[-1] == '' else lines
        assert len(lines) == len(items), (path, len(lines), len(items))
        unescape = official()['tsv_unescape']
        program = self.program(variant)
        rows = []
        for item, line in zip(items, lines):
            f = line.split('\t')
            assert len(f) == 9 and f[3] == ARCHIVE_MODELS[archive_model], (path, f[3:])
            assert f[4:9] == ['None', 'None', 'None', '0', '1'], f[4:9]   # no system/history/max_tokens; T=0, beams=1
            assert unescape(f[0]) == program.build(item)[0]['content'], (path, item['id'])
            output = {'content': unescape(f[1])}
            rows.append({'item': item, 'output': output, 'repeat': 0, 'score': self.score(item, output, variant)})
        return rows

    def archived_accuracy(self, archive_model, task, base, cot=False):
        rows = self.archived_rows(archive_model, task, base, cot)
        return sum(r['score']['correct'] for r in rows) / len(rows)

    # ------------------------------------------------------------- targets
    targets = tuple(
        PaperTarget(f'{m.replace(".", "")}_{_cell(t, b, c)}_accuracy', v, 0.0005, 'Table 18 (App. arithmetic)',
                    ARCHIVE_MODELS[m], 'paper_ccc' if t == 'ccc' else 'paper_addition', 'archived_outputs',
                    compute=(lambda m, t, b, c: lambda exp: exp.archived_accuracy(m, t, b, c))(m, t, b, c),
                    note='released model_interactions TSV rescored with eval.py; tolerance is the 0.1-point rounding')
        for m, t, b, c, v in (
            ('gpt4', 'addition', 9, False, 0.234), ('gpt4', 'addition', 10, False, 1.000),
            ('gpt4', 'addition', 9, True, 0.386), ('gpt4', 'addition', 10, True, 0.982),
            ('gpt3.5', 'addition', 9, False, 0.066), ('gpt3.5', 'addition', 10, False, 1.000),
            ('claude', 'addition', 9, False, 0.002), ('claude', 'addition', 10, False, 0.998),
            ('gpt4', 'ccc', 9, False, 0.900), ('gpt4', 'ccc', 10, False, 1.000),
        )
    )

