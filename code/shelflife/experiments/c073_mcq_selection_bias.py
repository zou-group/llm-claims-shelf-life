"""C073 — MCQ selection bias (Zheng et al. 2023, "Large Language Models Are Not Robust Multiple Choice Selectors").

Claim: LLMs prefer particular option IDs. Moving every gold answer to one letter swings gpt-3.5-turbo's 0-shot MMLU
accuracy from 67.2 to 74.2 (C) or 60.9 (D) (Table 1). The recall standard deviation (RStd) is 5.5 by default, 5.1 with
shuffled IDs and 1.0 with IDs removed. A debiasing instruction (6.1) and CoT (4.5) do not remove the bias (Table 2).
PriDe (5%) reduces RStd to 2.8 (Table 3). Only the gpt-3.5-turbo (API, generated first token) branch is ported. The
open-model logprob branch needs local weights.

Sources
  code     chujiezheng/LLM-MCQ-Bias @ 2ae2f40c (eval_ichat_utils.py prompts/settings, utils.py selection, move and
           load_results scorer).
  outputs  chujiezheng/LLM-MCQ-Bias_data @ 29fcbb48: the paper's gpt-3.5 0-shot MMLU outputs (default and move_a-d),
           snapshotted compactly in data/sources/C073_LLM-MCQ-Bias_data.

Prompt format. The repository HEAD strips the question and each option line (``.strip()``). The released outputs for
original and move_a-d were generated WITHOUT stripping, and in them one question reads 'Step l' where the CSV now has
'Step 1'. ``fmt='released'`` reproduces this and is byte-checked against every archived prompt.

Registered variants (this release): paper_answer_moving (default; Table 1), shuffled_answer_moving (seeded random
question order), shuffled_answer_moving_cap32 (non-faithful diagnostic) and robust.

Selection: per subject ``utils._index_samples`` (random.Random(123) shuffle, first 100, sorted), giving 5,700 requests.
At analysis time ``load_results`` excludes 214 questions whose options reference other options (BAD_OPTIONS /
REFER_OPTIONS). Items carry ``excluded`` and aggregate() drops them, as the paper does (footnote: about 3.2%).

Scoring
  official   (faithful) ``utils.load_results``: prediction = first character of the raw output (``sampled[0]``).
             Accuracy = prediction == gold over all non-excluded requests. RStd = population std (np.std) of the
             per-ID recalls x100 (debias_pride.py via sklearn recall).
  prior_v3   the earlier implementation's final robust parser (reproductions/C073/score.py: parse_letter,
             acceptable_status), loaded unchanged for rescoring parity. This is NOT the paper's scorer. It searches
             reasoning-model outputs for final-answer markers and rejects 2-token truncations that are not a lone ID.
  The official first-character rule is unusable for verbose modern outputs. For example, 'Answer: C' is scored as
  'A' and '**C**' as '*'. Modern-model runs should report both parsers. aggregate() always includes the prior_v3
  sensitivity.
  Headline validity. Every condition reports ``valid`` and ``valid_rate`` (share of non-excluded requests
  whose prediction is one of the displayed IDs), and aggregate() reports ``headline`` = {parser, min_valid_rate,
  reliable} plus ``headline_reliable``: False when any condition's valid rate is below ``HEADLINE_MIN_VALID_RATE``
  (0.90), with a ``headline_caveat`` pointing to ``prior_v3_sensitivity``. The headline itself stays the paper's
  metric.
  Bootstrap: the move-spread CI resamples questions only.
Robustness. Option ``parse='robust'`` keeps the variant's parser first and calls
  ``robust_choice`` (marker, opening ID, 'is:' lead-in, repeated option text, unique standalone ID; truncated outputs
  only the first three) only when it yields no ID. Under the official rule, a first letter that begins a word
  ('Answer: C' -> 'A', 'Based on' -> 'B') counts as no ID: on the panel cap-4096 runs these false parses inflated
  the move_A arm (M171 spread 6.1 -> 0.5). aggregate() then reports parsed_paper/parsed_robust/unparsed/missing and
  applies _robust.reliable per condition. Variant: robust (requests of shuffled_answer_moving with the cap floor
  1024, for new runs).
  Where robust differs from paper on rows the paper parser DID score: only
  the first-letter rule above; such rows are re-read by robust_choice (or stay unparsed). In robust_choice, an answer
  marker followed by two IDs ('Answer: A or B', 'B/D') is ambiguous -> None.
"""
import ast
import functools
import gzip
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict

import numpy as np

from ..paths import DATA, INITIAL, source_repo
from ..program import SingleTurn
from ..prompts import PromptTemplate, Source
from ..registry import Experiment, PaperTarget, Variant, register
from ..sourcecode import load_functions
from .. import stats
from . import _robust

REPO = source_repo('C073', 'LLM-MCQ-Bias')
CODE = REPO / 'code'
COMMIT = '2ae2f40c77006f7a00a4675bdfb30151c2406691'
ARCHIVE = DATA / 'sources' / 'C073_LLM-MCQ-Bias_data'
DATA_COMMIT = '29fcbb4807bc53a84ac26a64ddbad9526c940958'
# Earlier implementation's final scorer (behind EXPANDED_REVIEW.md), frozen.
PRIOR_SCORERS = {'prior_v3': INITIAL / 'reproductions/C073/score.py'}
LETTERS = 'ABCD'
MOVES = tuple(f'move_{l}' for l in LETTERS)
# archive directory per condition
ARCHIVE_DIRS = {'original': 'mmlu', **{f'move_{l}': f'mmlu_move_{l.lower()}' for l in LETTERS}}

_SRC = Source(repo=str(REPO), path='code/eval_ichat_utils.py', locator='prepare_eval', commit=COMMIT,
              url='https://github.com/chujiezheng/LLM-MCQ-Bias')
_ARCHIVE_SRC = Source(url=f'https://github.com/chujiezheng/LLM-MCQ-Bias_data/tree/{DATA_COMMIT}/results_mmlu/0s_ichat',
                      note='format of the released gpt-3.5 outputs; byte-checked against every archived prompt '
                           '(tests/claims/test_c073.py)')

SYSTEM = PromptTemplate('system', 'The following are multiple choice questions about {subject}. You should directly '
                        'answer the question by choosing the correct option.', source=_SRC,
                        note="sys_msg + ' You should directly answer ...' (non-CoT settings); subject underscores -> spaces")
USER_RELEASED = PromptTemplate('user', 'Question: {question}\nOptions:\n{option_lines}\nAnswer:', source=_ARCHIVE_SRC,
                               note="option_lines = '\\n'.join(f'{id}. {option}') unstripped ('{option}' for noid); "
                                    'question unstripped')
# ------------------------------------------------------------------------------------------- official code
class _Silent:
    info = error = warning = staticmethod(lambda *a, **k: None)


@functools.lru_cache(maxsize=None)
def official():
    """Authors' functions from utils.py (module body not executed)."""
    return load_functions(CODE / 'utils.py', ['move_answer', '_index_samples', '_norm', 'load_results'],
                          constants=['BAD_OPTIONS', 'REFER_OPTIONS'], extra_globals={'logger': _Silent(), 'os': os})


def _prepare_eval_node():
    tree = ast.parse((CODE / 'eval_ichat_utils.py').read_text())
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'prepare_eval')


def official_string_constants():
    """Every string literal in prepare_eval (for verifying the SYSTEM template)."""
    return {n.value for n in ast.walk(_prepare_eval_node()) if isinstance(n, ast.Constant) and isinstance(n.value, str)}


@functools.lru_cache(maxsize=None)
def prior_scorer(name='prior_v3'):
    """An earlier implementation's score.py (parse_letter/score, plus acceptable_status where it existed), frozen."""
    path = PRIOR_SCORERS[name]
    names = ['norm', 'parse_letter', 'score'] + (['acceptable_status'] if 'def acceptable_status' in path.read_text() else [])
    return load_functions(path, names, constants=['LETTERS'])


@functools.lru_cache(maxsize=None)
def released_errata():
    return json.loads((ARCHIVE / 'released_question_errata.json').read_text())


# ------------------------------------------------------------------------------------------------ data
@functools.lru_cache(maxsize=None)
def mmlu_tables():
    """{subject: [row dicts]} read exactly as prepare_eval_samples does (pandas, dtype=str, default NA handling)."""
    import pandas as pd
    out = {}
    subjects = sorted(f.split('_test.csv')[0] for f in os.listdir(CODE / 'data_mmlu/test') if '_test.csv' in f)
    for subject in subjects:
        with open(CODE / f'data_mmlu/test/{subject}_test.csv') as fh:
            df = pd.read_csv(fh, names=('Question', *LETTERS, 'Answer'), dtype=str)
        # str() mirrors the original: options are str(x[e]) and f'{answer}' (NaN -> 'nan', e.g. a literal 'None')
        out[subject] = [{k: str(v) for k, v in row.items()} for row in df.to_dict('records')]
    return out


@functools.lru_cache(maxsize=None)
def official_selection():
    """[(subject, idx)] in the original order: subjects sorted, _index_samples(…, 100) per subject."""
    idx_fn = official()['_index_samples']
    return tuple((s, i) for s, rows in mmlu_tables().items() for i, _ in idx_fn(list(range(len(rows))), 100))


def excluded(options):
    o = official()
    return (any(e in opt for e in o['BAD_OPTIONS'] for opt in options)
            or any(e in opt.lower() for e in o['REFER_OPTIONS'] for opt in options))


@functools.lru_cache(maxsize=None)
def shuffled_selection(seed=20261008):
    """Reproduction-reproduction pass subset order: the non-excluded official sample in a seeded random order, so that a
    ``--limit`` prefix is a random question subset spanning all subjects (conditions of a question stay adjacent)."""
    rows = mmlu_tables()
    pool = [(s, i) for s, i in official_selection() if not excluded([rows[s][i][l] for l in LETTERS])]
    random.Random(seed).shuffle(pool)
    return tuple(pool)


def make_item(subject, idx, condition):
    """One request for question (subject, idx) under a condition (original or move_A..D)."""
    row = dict(mmlu_tables()[subject][idx])
    original = [row[l] for l in LETTERS]
    gold = row['Answer']
    perm = list(range(4))                              # displayed position -> original content index
    labels = list(LETTERS)                             # displayed ID at each position
    if condition.startswith('move_'):
        moved = official()['move_answer'](dict(row), condition[-1])
        target, g = LETTERS.index(condition[-1]), LETTERS.index(gold)
        perm[target], perm[g] = perm[g], perm[target]
        assert [moved[l] for l in LETTERS] == [original[p] for p in perm] and moved['Answer'] == condition[-1]
    displayed = [original[p] for p in perm]
    gold_position = perm.index(LETTERS.index(gold))
    question = released_errata().get(f'{subject}:{idx}', row['Question'])
    return {'id': f'C073:{subject}:{idx}:{condition}:p0', 'question_id': f'{subject}:{idx}', 'subject': subject,
            'row_index': idx, 'condition': condition, 'fmt': 'released', 'question': question,
            'original_options': original, 'original_gold': gold, 'gold_content_index': LETTERS.index(gold),
            'permutation': perm, 'displayed_labels': labels, 'displayed_options': displayed,
            'gold_position': gold_position, 'expected': labels[gold_position], 'excluded': excluded(original)}


def user_prompt(item):
    """User message in the released (unstripped) format."""
    option_lines = '\n'.join(f'{i}. {a}' for i, a in zip(item['displayed_labels'], item['displayed_options']))
    return USER_RELEASED.render(question=item['question'], option_lines=option_lines)


def system_prompt(item, variant):
    subject = item['subject'].replace('_', ' ')
    return SYSTEM.render(subject=subject)


# ------------------------------------------------------------------------------------------------ scoring
def official_prediction(text, item):
    """Prediction as utils.load_results derives it (raw first character)."""
    return text[0] if text else None   # the original indexes sampled[0]; an empty output would raise there


# Robust fallback (parse='robust', 2026-10-10; the robust-variant conventions). The paper parser runs
# first; these helpers only see answers it could not read. Implemented locally (not _robust.extract_choice) because a
# single-letter ID collides with the English article 'A' ("A determination cannot be made ...") and because a
# truncated answer must not be read from an option merely mentioned in an unfinished explanation.
_ID = re.compile(r'(?<![A-Za-z0-9])\(?([ABCD])\)?(?![A-Za-z0-9])')
# 'A man', 'A determination': the article, not option A ('A or B', 'A is correct' stay option IDs)
_ARTICLE = re.compile(r' (?!(?:or|and|nor|vs|versus|is|was|are|were|seems|appears|would|could|might|may|must|should|'
                      r'through|to|until|because|since|but)\b)[a-z]')
# 'The first machine cycle ... is:\n\nA. An op-code fetch cycle', '... is likely: D. Inflation': the ID right after a
# lead-in ending in 'is:'
_LEAD_IN = re.compile(r'\b(?:is|are|be|likely|answer|option|choice)\s*:\s*\(?([ABCD])\)?(?:[.:)]|\s*$|\s*\n)', re.I)


def _ids(text, start_only=False):
    """Standalone option-ID tokens in order; an 'A' followed by a lowercase word is read as the article."""
    out = []
    for m in (_ID.finditer(text) if not start_only else filter(None, [_ID.match(text)])):
        if m.group(1) == 'A' and _ARTICLE.match(text, m.end()):
            continue
        out.append(m.group(1))
    return out


def standalone_first_id(text):
    """The first character as an option ID only when it stands alone (the paper's first-character rule minus its
    false parses of words: 'Answer: C' -> 'A', 'Based on' -> 'B', 'A determination' -> 'A'). None otherwise."""
    ids = _ids(text or '', start_only=True)
    return ids[0] if ids and (text or '')[0] == ids[0] else None


def _content_choice(answer, labels, options):
    """The ID whose displayed option text the answer repeats verbatim ('The final answer is \\boxed{6.0 eV}' ->
    the option '6.0 eV'), compared case-insensitively after whitespace/trailing-period normalization."""
    if not options:
        return None
    key = lambda x: ' '.join(str(x).split()).rstrip('.').strip().lower()
    a = key(answer)
    hits = [l for l, o in zip(labels, options) if a and key(o) == a]
    return hits[0] if len(hits) == 1 else None


def robust_choice(text, labels, truncated=False, options=None):
    """Option ID from a modern-format answer: the first ID right after the last explicit answer marker
    (_robust.answer_segment: 'Answer:', 'the answer is', 'Final answer'), else an ID opening the normalized response
    (markdown, LaTeX, headings, <think> blocks stripped by _robust.normalize), else an ID opening the line after a
    lead-in ending in 'is:', else (complete outputs only) the option whose text the answer (after the marker, or the
    whole response) repeats verbatim, else the unique ID mentioned as a standalone token. Returns None if ambiguous or absent."""
    ordered, labels = list(labels), set(labels)
    t = _robust.normalize(text)
    seg = _robust.answer_segment(text)
    if seg != t:
        clause = re.split(r'[.\n;]', seg[:80], 1)[0]
        alt = '|'.join(re.escape(x) for x in sorted(labels, key=len, reverse=True))
        if re.search(r'(?<![A-Za-z0-9])\(?(?:' + alt + r')\)?\s*(?:or|/|and|,)\s*\(?(?:' + alt + r')\)?(?![A-Za-z0-9])',
                     clause):
            return None                      # "Answer: A or B" names two options: ambiguous
        ids = _ids(seg[:80])
        if ids and ids[0] in labels:
            return ids[0]
        if not truncated and _content_choice(seg, ordered, options):
            return _content_choice(seg, ordered, options)
    ids = _ids(t, start_only=True)
    if ids and ids[0] in labels:
        return ids[0]
    m = _LEAD_IN.search(t)
    if m and m.group(1) in labels:
        return m.group(1)
    if truncated:
        return None
    if _content_choice(t, ordered, options):
        return _content_choice(t, ordered, options)
    found = set(_ids(t)) & labels
    return found.pop() if len(found) == 1 else None


def _legacy_case(item):
    return {'condition': item['condition'], 'expected': item['expected'],
            'metadata': {'displayed_options': item['displayed_options'], 'displayed_labels': item['displayed_labels'],
                         'gold_position': item['gold_position'], 'permutation': item['permutation']}}


def recall_std(predictions, labels):
    """debias_pride.py: recall per class 0..3 (sklearn classification_report) x100, population std (np.std).
    Classes absent from labels get recall 0 (sklearn's zero_division default)."""
    recalls = []
    for c in LETTERS:
        support = sum(l == c for l in labels)
        recalls.append(100 * sum(p == l == c for p, l in zip(predictions, labels)) / support if support else 0.0)
    return float(np.std(recalls)), recalls


# -------------------------------------------------------------------------------------------- archive
@functools.lru_cache(maxsize=None)
def archive(condition):
    """Released gpt-3.5 0-shot MMLU records for a condition (compact snapshot; see PROVENANCE.md)."""
    name = ARCHIVE_DIRS.get(condition, condition)
    with gzip.open(ARCHIVE / f'{name}.jsonl.gz', 'rt') as fh:
        return tuple(json.loads(line) for line in fh if line.strip())


def prompt_sha256(messages):
    return hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@register
class C073MCQSelectionBias(Experiment):
    claim_id = 'C073'
    title = 'MCQ option-ID selection bias (answer moving, RStd)'
    claim = ('LLMs prefer specific option IDs. Moving all gold answers to one letter swings gpt-3.5-turbo 0-shot MMLU '
             'accuracy from 67.2 to 74.2 (C) or 60.9 (D). RStd is 5.5 by default, 5.1 with shuffled IDs and 1.0 '
             'with IDs removed. A debiasing instruction (6.1) and CoT (4.5) do not remove it.')
    paper = {'title': 'Large Language Models Are Not Robust Multiple Choice Selectors', 'authors': 'Zheng et al.',
             'year': 2023, 'url': 'https://arxiv.org/abs/2309.03882'}
    source = _SRC
    default_variant = 'paper_answer_moving'

    def build_variants(self):
        base_dev = ('paper used gpt-3.5-turbo-0613 via the `gpt-3.5-turbo` alias (retired)',)
        id_params = {'temperature': 0.0, 'max_tokens': 2}
        prompts = {'system': SYSTEM, 'user': USER_RELEASED}
        common = {'selection': 'official', 'fmt': 'released', 'parser': 'official'}
        moving = Variant('paper_answer_moving', 'Table 1 answer-moving attack: original + gold moved to A/B/C/D, '
                         'released prompt format, temperature 0, max_tokens 2, 5,700 requests per condition '
                         '(214 excluded at analysis)', prompts=prompts, params=id_params,
                         options={**common, 'conditions': ('original',) + MOVES}, deviations=base_dev)
        v = {
            'paper_answer_moving': moving,
            'shuffled_answer_moving': moving.override(
                'shuffled_answer_moving', description='paper_answer_moving (prompts, params, scorer unchanged) over the '
                'non-excluded official sample in a seeded random question order (seed 20261008); run with --limit '
                '5*Q for a random Q-question subset across all subjects (reproduction reproduction pass)',
                options={'selection': 'shuffled'},
                deviations=('random question subset (prefix of a seeded shuffle of the 5,486 non-excluded official '
                            'questions) instead of the full 5,700-request sample',)),
            'shuffled_answer_moving_cap32': moving.override(
                'shuffled_answer_moving_cap32', description='NON-FAITHFUL diagnostic: shuffled_answer_moving with '
                'max_tokens 32 instead of 2, scored with the prior_v3 answer parser, for models that no longer start '
                'with the option ID (gpt-3.5-turbo-0613 as served in 2026)',
                params={'temperature': 0.0, 'max_tokens': 32}, options={'selection': 'shuffled', 'parser': 'prior_v3'},
                deviations=('random question subset (seeded shuffle prefix)', 'max_tokens 32 instead of 2',
                            'prior_v3 parser instead of the first-character rule')),
        }
        # Robustness variants. Same selection-bias
        # statistic (move spread, RStd, accuracy); only the answer reading (and, for `robust`, the output cap) changes.
        fallback = ("parse='robust': the paper parser first; where it yields no option ID, robust_choice() reads the ID "
                    "after an explicit answer marker, or opening the response after markdown/LaTeX/<think> are "
                    "stripped, or (complete outputs only) the unique standalone ID mentioned; tolerant reliability "
                    "(<= 10% unparsed + missing per condition)",)
        first_char = ("under the official first-character rule, a first letter that begins a word ('Answer: C' -> 'A', "
                      "'Based on' -> 'B', 'A determination' -> 'A') is treated as no ID and passed to the fallback",)
        v['robust'] = v['shuffled_answer_moving'].override(
            'robust', description="robust_parse with the output cap raised to the _robust.cap_floor (1024 tokens) so "
            'models that answer after a preamble are not cut off; for new panel runs',
            params=_robust.cap_floor(id_params), options={'parse': 'robust'},
            deviations=fallback + first_char + (
                f'max_tokens {_robust.MIN_ROBUST_TOKENS} instead of 2 (the paper relied on gpt-3.5 answering with the '
                'ID first; modern models open with a preamble, markdown or reasoning)',))
        return v

    # ------------------------------------------------------------------ data
    def items(self, variant, limit=None):
        sel = variant.options['selection']
        questions = {'official': official_selection, 'shuffled': shuffled_selection}[sel]()
        rows = []
        for subject, idx in questions:
            for condition in variant.options['conditions']:
                rows.append(make_item(subject, idx, condition))
            if limit is not None and len(rows) >= limit:
                return rows[:limit]
        return rows

    def program(self, variant):
        def build(item):
            return [{'role': 'system', 'content': system_prompt(item, variant)},
                    {'role': 'user', 'content': user_prompt(item)}]
        return SingleTurn(build, variant.params)

    # --------------------------------------------------------------- scoring
    def score(self, item, output, variant, parser=None):
        # parse='robust' applies only to the variant's own parser (an explicit ``parser`` argument, e.g. the
        # prior_v3 sensitivity, is always scored as that parser alone).
        robust = parser is None and variant.options.get('parse', 'paper') == 'robust'
        parser = parser or variant.options['parser']
        text = output.get('content') or ''
        if parser == 'official':
            pred = official_prediction(text, item)
            valid = pred in item['displayed_labels']
            if robust and valid and standalone_first_id(text) is None:
                pred, valid = None, False   # first letter of a word ('Answer: C', 'Based on'), not an option ID
        elif parser in PRIOR_SCORERS:
            scorer = prior_scorer(parser)
            sc = scorer['score'](text, _legacy_case(item))
            row = {'status': output.get('status', 'ok'), 'content': text, 'case': _legacy_case(item)}
            valid = sc['parsed'] and ('acceptable_status' not in scorer or scorer['acceptable_status'](row))
            pred = sc['prediction'] if valid else None
        else:
            raise ValueError(parser)
        path = 'paper' if valid else None
        if robust and not valid:
            truncated = output.get('status') == 'truncated' or output.get('finish_reason') == 'length'
            pred = robust_choice(text, item['displayed_labels'], truncated=truncated,
                                 options=item['displayed_options'])
            valid = pred is not None
            path = 'robust' if valid else None
        position = item['displayed_labels'].index(pred) if valid else None
        out = {'parser': parser, 'prediction': pred, 'valid': valid, 'correct': valid and pred == item['expected'],
               'predicted_content_index': item['permutation'][position] if valid else None}
        if robust:
            out['parse_path'] = path
        return out

    def _condition_metrics(self, rows):
        labels = [r['item']['expected'] for r in rows]
        preds = [(r['score'] or {}).get('prediction') if (r['score'] or {}).get('valid') else None for r in rows]
        n = len(rows)
        correct = sum(p == l for p, l in zip(preds, labels))
        valid = sum(p is not None for p in preds)
        out = {'n': n, 'valid': valid, 'valid_rate': valid / n if n else None,
               'accuracy_pct': 100 * correct / n if n else None,
               'prediction_counts': dict(Counter(p for p in preds if p is not None)), 'gold_counts': dict(Counter(labels))}
        if n and all(l in labels for l in LETTERS):
            out['rstd_pp'], out['recall_pct'] = recall_std(preds, labels)
        return out

    HEADLINE_MIN_VALID_RATE = 0.90

    def aggregate(self, rows, variant, parser=None):
        """Paper metrics over non-excluded items. Unscored rows (call failures) count as incorrect, never dropped."""
        kept = [r for r in rows if not r['item']['excluded']]
        out = {'requests': len(rows), 'excluded_items': len(rows) - len(kept)}
        if parser is not None:
            kept = [{**r, 'score': self.score(r['item'], r['output'], variant, parser) if r.get('output') else None}
                    for r in kept]
        by_cond = defaultdict(list)
        for r in kept:
            by_cond[r['item']['condition']].append(r)
        out['conditions'] = {c: self._condition_metrics(rs) for c, rs in sorted(by_cond.items())}
        rates = [c['valid_rate'] for c in out['conditions'].values() if c['valid_rate'] is not None]
        reliable = bool(rates) and min(rates) >= self.HEADLINE_MIN_VALID_RATE
        robust = parser is None and variant.options.get('parse', 'paper') == 'robust'
        if robust:
            # robust-variant conventions: parse-path counts, and the tolerant rule (<= 10% unparsed + missing) in every condition
            for c, rs in by_cond.items():
                paths = Counter((r['score'] or {}).get('parse_path') for r in rs if r.get('score') is not None)
                missing = sum(r.get('score') is None for r in rs)
                out['conditions'][c].update(parsed_paper=paths['paper'], parsed_robust=paths['robust'],
                                            unparsed=paths[None], missing=missing)
            conds = out['conditions'].values()
            out['parse_paths'] = {k: sum(c[k] for c in conds) for k in ('parsed_paper', 'parsed_robust', 'unparsed',
                                                                         'missing')}
            reliable = bool(rates) and all(_robust.reliable(c['n'], c['unparsed'], c['missing']) for c in conds)
        out['headline'] = {'parser': (parser or variant.options['parser']) + ('+robust' if robust else ''),
                           'min_valid_rate': min(rates) if rates else None, 'reliable': reliable}
        out['headline_reliable'] = reliable
        if not reliable:
            out['headline_caveat'] = (
                f"only {min(rates) if rates else 0:.1%} of requests in the worst condition yield a valid option ID under "
                f"the {out['headline']['parser']} parser (threshold {self.HEADLINE_MIN_VALID_RATE:.0%}); accuracy, RStd "
                "and spread are not interpretable" + ('; see prior_v3_sensitivity' if parser is None else ''))
        if all(m in by_cond for m in MOVES):
            out['answer_moving'] = self._moving_metrics(kept)
        if parser is None and variant.options['parser'] == 'official' and kept and kept[0].get('output') is not None:
            out['prior_v3_sensitivity'] = {c: self._condition_metrics(
                [{**r, 'score': self.score(r['item'], r['output'], variant, 'prior_v3') if r.get('output') else None}
                 for r in rs]) for c, rs in sorted(by_cond.items())}
        return out

    def _moving_metrics(self, rows):
        """Move spread (max - min accuracy over gold-at-A..D), question-cluster bootstrap, content switches."""
        moves = [r for r in rows if r['item']['condition'] in MOVES]

        def spread(rs):
            accs = []
            for m in MOVES:
                sub = [r for r in rs if r['item']['condition'] == m]
                if not sub:
                    return float('nan')
                accs.append(100 * sum(bool((r['score'] or {}).get('correct')) for r in sub) / len(sub))
            return max(accs) - min(accs)
        point, lo, hi = stats.cluster_bootstrap(moves, lambda r: r['item']['question_id'], spread, n_boot=2000,
                                                seed=20261003)
        groups = defaultdict(dict)
        for r in moves:
            groups[(r['item']['question_id'], r.get('repeat', 0))][r['item']['condition']] = (r['score'] or {}).get(
                'predicted_content_index')
        complete = {k: v for k, v in groups.items() if len(v) == 4 and all(x is not None for x in v.values())}
        switching = [k for k, v in complete.items() if len(set(v.values())) > 1]
        return {'spread_pp': point, 'spread_ci95_question_bootstrap': [lo, hi], 'groups': len(groups),
                'fully_parsed_groups': len(complete), 'content_switch_groups': len(switching),
                'content_switch_questions': len({q for q, _ in switching}),
                'switch_question_upper95_if_zero': stats.zero_event_upper(len({q for q, _ in groups}))
                if not switching else None}

    # ------------------------------------------------- targets and prior runs
    def archive_rows(self, condition):
        """(item, output) rows for every released record of a condition, built for the archived questions."""
        rows = []
        for rec in archive(condition):
            if rec['type'] != 'result':
                continue
            d = rec['data']
            item = make_item(rec['subject'], d['idx'], condition)
            rows.append({'item': item, 'record': d, 'repeat': 0,
                         'output': {'content': d.get('sampled', ''), 'status': 'ok'}})
        return rows

    def archive_metrics(self, condition):
        variant = self.variant('paper_answer_moving')
        rows = [{**r, 'score': self.score(r['item'], r['output'], variant)} for r in self.archive_rows(condition)]
        return self.aggregate(rows, variant)['conditions'][condition]

    targets = ()


def _arch(condition, key):
    return lambda exp: exp.archive_metrics(condition)[key]


C073MCQSelectionBias.targets = tuple(
    [PaperTarget(f'gpt35_mmlu0s_acc_{c}', v, 0.05, 'Table 1', 'gpt-3.5-turbo-0613', 'paper_answer_moving',
                 'archived_outputs', compute=_arch(c, 'accuracy_pct'),
                 note='released outputs, official load_results scoring; tolerance = rounding to 0.1')
     for c, v in (('original', 67.2), ('move_A', 65.3), ('move_B', 68.5), ('move_C', 74.2), ('move_D', 60.9))]
    + [PaperTarget('gpt35_mmlu0s_rstd_default', 5.5, 0.05, 'Table 2', 'gpt-3.5-turbo-0613', 'paper_answer_moving',
                   'archived_outputs', compute=_arch('original', 'rstd_pp'), note='rounding tolerance'),
       PaperTarget('gpt35_balanced1000_pred_share_C', 32.3, 0.05, 'Section 1', 'gpt-3.5-turbo-0613',
                   'paper_answer_moving', 'unverifiable',
                   note='label-balanced 1,000-question sample, ten runs; sample identities not released')])
