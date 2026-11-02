"""C101 (claim-set key TOY-01) — toy case: "How many r's are in strawberry?" answered 2, not 3.

Claim: asked "How many r's are in strawberry?", chat LLMs often answer 2 instead of the correct 3 (the repeated 'rr' is
undercounted); a control question about a non-repeated letter ("How many s's are in strawberry?", answer 1) is answered
correctly.

Origin: a toy case we added. The strawberry question is a
widely circulated informal 2024 example; it has no paper protocol. Closest published evidence: Fu et al. 2024, "Why Do
Large Language Models (LLMs) Struggle to Count Letters?" (arXiv 2412.18626v1, the only version), which uses
"strawberry" as its one-shot example and shows, on 10,000 random words, that counting errors concentrate on letters
with multiplicity > 1 (Sec. 4, Fig. 10). Their released outputs (repo aMa2210/LLM_CounterLettersWithoutFT @ 3b3eac02)
are snapshotted under data/sources/C101_fu_letter_count/ and anchor the mechanism on GPT-4o (targets below).

Protocol (our minimal design, our claim record; there is no other source)
  three single-turn user prompts, 10 independent samples each at the provider's default temperature:
    a) r_toy      "How many r's are in strawberry?"                                   answer 3 (primary)
    b) r_wording  'How many times does the letter r appear in the word "strawberry"?' answer 3 (wording variant)
    c) s_control  "How many s's are in strawberry?"                                   answer 1 (control)
  Score: the first integer or number word in the answer (regex). Metric: share correct per prompt, exact binomial CI.
  The toy "persists" on a model if (a) accuracy is below (c).

Variants
  toy   default, faithful to the minimal design: the three prompts verbatim from the claim record, no system prompt,
        no temperature sent (provider default), repeats=10. ``max_tokens`` 1024 is sent because originals_v1 (the
        route of the reproduction model) requires an explicit cap; 1024 is also what the panel policies use when the
        source gives none, and the panel policy raises reasoning models to its 4096 floor. Items are ordered a, c, b,
        so ``--limit 2`` is the primary contrast and ``--limit 3`` (MINIMAL_LIMIT) the full design.

Scoring (a deviation from the design's literal "first integer" rule)
  ``stated`` (headline): the count the answer states for the asked letter. Numbered-list markers ("1." / "2)" at line
  start or after "/" or ";") and "N letters/characters" and "position(s) N, N" spans are removed first. Then the
  LAST count bound to the letter or to the question is taken, from: \boxed{N}; "answer: N" / "answer is N"; "N r's",
  "N 'r'", "N Rs", "N letter r"; "N times"; "appears/occurs N"; "total of N". If none is bound, the last remaining
  number is used. Number words: zero-twenty, once/twice/thrice, and "single" (= 1, e.g. "a single s").
  An answer with no number is unparsed and counts as incorrect.
  ``first_number`` (diagnostic, the design's literal rule) and ``cleaned_last`` (the last number after the same
  clean-up) are kept per row. ``agreement`` = share of scored samples where stated == cleaned_last, computed per
  question; ``headline_reliable`` is False when either primary question's agreement is below 0.95 (then hand-check the
  outputs before reading the verdict).
  Denominator: failed calls (no output), truncated outputs (finish_reason length) and empty content are excluded
  from the rate, counted per question, and bound it via ``bounds_with_missing``.

Half-life metric: ``acc(s_control) - acc(r_toy)``, stated scoring; +1 = stronger failure; null 0. CI: Newcombe hybrid
  score interval (Wilson per arm) for a difference of two independent proportions (10 samples per arm; a percentile
  bootstrap degenerates at 0/10 or 10/10). ``persists`` = CI lower bound > 0. ``paper_value`` is None: no paper number
  exists for this prompt. The anchor is our reproduction on openai-direct/gpt-4o-2024-05-13 (the ``gpt-4o`` alias
  during the viral 2024 period); classify panel models relative to that measured value.

Scoring deviations (earlier pipeline validation 2026-10-05)
  * ``clean()`` no longer deletes "N letter(s)/character(s)" when a single (optionally quoted) letter follows, and the
    bound pattern accepts "N letters 'x'": 'The word strawberry contains 3 letters "r".', 'contains one letter
    \'s\'.' and 'There are 3 letter "r"s' were erased to no count (false "incorrect", mostly on the s-control; M102
    10/30, M015 3/30). "10 letters long" / "(10 letters)" are still removed.
  * Reading rule: ``shelflife.primary_value`` = acc(r_toy) (the toy question itself; higher = toy solved) is reported
    as the primary reading next to the difference. ``shelflife.value`` stays acc(s_control) - acc(r_toy) for continuity,
    but ``shelflife.difference_interpretable`` is False when control accuracy is below CONTROL_GATE (0.8): then the
    difference mixes a control failure with the r failure (floor models that fail both score 0, models that miscount
    the control score negative), and should be read through primary_value instead.

Robustness:
  * scoring option ``parse``: 'paper' (default, the stated-count scorer above) or 'robust'. Under 'robust' the paper
    scorer runs first; only when it finds no count, ``robust_count`` tries (1) the stated-count rule on
    ``_robust.normalize``d text (think blocks, LaTeX wrappers, markdown removed), (2) an explicit zero bound to the
    letter ("There are no R's", "does not contain the letter r", a bare "None.") = 0, (3) ``_robust.extract_number``.
    Rows report the path (``parse_path`` paper/robust/None); aggregate reports parsed_paper / parsed_robust /
    unparsed. Truncated and empty outputs stay excluded from the rate (as in the paper rule) under robust_parse.
  * ``robust`` sets option truncated='score' (_robust.truncation_missing): at a 2048-token floor for counting letters
    in one word, a cut-off answer is the model's failure (M045-style letter-by-letter loops), not a cap artefact, so
    it is scored by the stated-count rule on whatever text it has (no count, or empty -> incorrect), stays in the
    rate and is not counted as unparsed or missing; counted as ``truncated_scored``. Empty complete outputs stay
    excluded. Documented exception for robust: cut-off outputs are scored instead of excluded (the module's
    all-variant denominator rule). robust_parse keeps 'missing'.
  * ``headline_reliable`` under 'robust' = ``_robust.reliable`` over the two primary prompts' rows (unparsed +
    truncated/empty/failed <= 10%). The paper rule (stated == cleaned last number on >= 95% of each primary prompt's
    10 samples, i.e. one disagreement flags the model) is kept as ``validity.agreement_rule_reliable``: on the saved
    panel outputs every disagreement was a correct count followed by positions ("3rd, 8th and 9th") or an aside
    about the other letter, except one M206 control answer the paper scorer misreads (see the findings). The
    collector's control-accuracy gate (>= 0.8) is a validity gate and is unchanged.
  * variants ``robust_parse`` (toy requests unchanged, parse='robust'; for offline rescoring) and ``robust``
    (robust_parse with a max_tokens floor of 2048: DeepSeek V3 (M045) hit the 1024 cap on 3/30 toy answers).

Robust != paper on rows the paper scored: none (fallback only when the paper scorer
finds no count; on the 630 saved earlier pipeline outputs only 2 M017 answers took the fallback). The known paper misread
(an M206 control answer read as 3) is deliberately kept.
"""
import collections
import json
import math
import re

from ..paths import DATA, PROJECT
from ..program import SingleTurn
from ..prompts import PromptTemplate, Source
from ..registry import Experiment, PaperTarget, Variant, register
from ..sourcecode import load_functions
from .. import stats
from . import _robust

FU = DATA / 'sources' / 'C101_fu_letter_count'
FU_COMMIT = '3b3eac02923cd54c63a973f5ff6c955b56d1769b'
FU_URL = 'https://github.com/aMa2210/LLM_CounterLettersWithoutFT'
CLAIM_RECORD = 'experiments/run_minimal_setups/runs/earlier/selection/claim_records/C101.json'
REPEATS = 10
CONTROL_GATE = 0.8           # control accuracy below this -> the difference is flagged uninterpretable

SCORING_DEVIATIONS = (
    'clean() keeps "N letter(s)/character(s)" when a single (optionally quoted) letter follows, and "N letters \'x\'" '
    'is a bound count (earlier pipeline validation 2026-10-05; was erased -> false incorrect, mostly on the control)',
    'acc(r_toy) reported as shelflife.primary_value; the difference shelflife.value is flagged '
    'difference_interpretable=False when control accuracy < CONTROL_GATE (0.8)',
)

RUN_VARIANT = 'toy'
ROBUST_CAP = 2048           # robust variant output-cap floor (M045 truncated 3/30 at 1024)
MINIMAL_LIMIT = 3            # items (prompts); each runs REPEATS=10 times -> 30 calls per model
REPRO_MODELS = ['openai-direct/gpt-4o-2024-08-06', 'openai-direct/gpt-4o-2024-05-13']
REPRO_LIMIT = 3              # same design (3 prompts x 10 samples) on each reproduction model

_SPEC = Source(repo='.', path=CLAIM_RECORD, locator='claim_set_record.minimal_design',
               note="our minimal design for the toy case; prompts quoted verbatim there")

PROMPTS = {
    'r_toy': PromptTemplate('r_toy', "How many r's are in strawberry?", style='literal', source=_SPEC,
                            note='(a) verbatim toy prompt'),
    's_control': PromptTemplate('s_control', "How many s's are in strawberry?", style='literal', source=_SPEC,
                                note='(c) control, non-repeated letter'),
    'r_wording': PromptTemplate('r_wording', 'How many times does the letter r appear in the word "strawberry"?',
                                style='literal', source=_SPEC, note='(b) wording variant'),
}
# Order matters for --limit: primary contrast first (a, c), then the extra wording arm (b).
CONDITIONS = (('r_toy', 'r', 3), ('s_control', 's', 1), ('r_wording', 'r', 3))

# ------------------------------------------------------------------------------------------------ scoring
NUMBER_WORDS = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8,
                'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
                'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20,
                'once': 1, 'twice': 2, 'thrice': 3, 'single': 1}
_WORDS = '|'.join(sorted(NUMBER_WORDS, key=len, reverse=True))
_NUM = re.compile(r'(?<![\w.])(\d+)(?![\d])|\b(' + _WORDS + r')\b', re.IGNORECASE)
_N = r'(\d+|' + _WORDS + r')'
_Q = '["\'\u201c\u201d\u2018\u2019`]?'


_QQ = '["\'\u201c\u201d\u2018\u2019`]'   # a required quote
# a letter name follows: quoted ("r", 'r', `r`, "r"s) or bare and then ending the clause ("one letter s." / "... s in")
_LETTER_FOLLOWS = (r'(?!\s*(?:' + _QQ + r'[a-z]' + _QQ + r'?|[a-z]' + _QQ + r')(?:\'?s)?(?![a-z])'
                   r'|\s*[a-z](?:\'?s)?(?:\s*(?:[.!?;()]|$)|\s+(?:in|appears?|occurs?)\b))')


def _value(token):
    return int(token) if token.isdigit() else NUMBER_WORDS[token.lower()]


def numbers(text):
    """All integers / number words in order of appearance."""
    out = []
    for m in _NUM.finditer(text or ''):
        out.append(int(m.group(1)) if m.group(1) is not None else NUMBER_WORDS[m.group(2).lower()])
    return out


def first_number(text):
    found = numbers(text)
    return found[0] if found else None


def clean(text):
    """Remove spans whose numbers are not counts: list markers, 'N letters/characters', 'position(s) N, N and N'."""
    t = (text or '').replace('**', '').replace('__', '')
    t = re.sub(r'(?m)(^|[/;|])[ \t]*\d+[.)](?=\s)', r'\1', t)
    # keep "3 letters 'r'" / "one letter s" (a count bound to a letter); drop "10 letters" / "10 letters long"
    t = re.sub(r'\b' + _N + r'[\s-]*(?:letters?|characters?|chars?)\b' + _LETTER_FOLLOWS, ' ', t,
               flags=re.IGNORECASE)
    t = re.sub(r'\bpositions?\s*(?:#\s*)?\d+(?:\s*(?:,|and|&)\s*(?:#\s*)?\d+)*', ' ', t, flags=re.IGNORECASE)
    t = re.sub(r'\b(?:index|indices)\s*\d+(?:\s*(?:,|and)\s*\d+)*', ' ', t, flags=re.IGNORECASE)
    return t


def _bound_patterns(letter):
    L = re.escape(letter)
    return [
        r'\\boxed\{\s*' + _N + r'\s*\}',
        r'\banswer(?:\s+is)?\s*[:=]?\s*' + _N + r'\b',
        r'\b' + _N + r'\s+(?:letters?\s+)?' + _Q + L + _Q + r'(?:\'?s)?(?![a-z])',
        r'\b' + _N + r'\s+times?\b',
        r'\b(?:appears?|appearing|occurs?|occurring|occurrences?(?:\s+of\s+' + _Q + L + _Q + r')?)\s*(?::|is|are)?\s*'
        + _N + r'\b',
        r'\btotal\s+(?:of\s+)?' + _N + r'\b',
        r'\bcount\s*(?:is|=|:)\s*' + _N + r'\b',
    ]


def stated_count(text, letter):
    """(value, how): the count stated for ``letter``; how = 'bound' | 'last_number' | 'none'."""
    t = clean(text)
    best = None
    for pat in _bound_patterns(letter):
        for m in re.finditer(pat, t, flags=re.IGNORECASE):
            if best is None or m.start() >= best[0]:
                best = (m.start(), _value(m.group(1)))
    if best is not None:
        return best[1], 'bound'
    found = numbers(t)
    return (found[-1], 'last_number') if found else (None, 'none')


def cleaned_last(text):
    found = numbers(clean(text))
    return found[-1] if found else None


_ZERO_WORDS = r'(?:no|zero|none\s+of\s+the)'


def robust_count(text, letter):
    """Fallback count for ``parse='robust'`` (used only when ``stated_count`` finds none): the stated-count rule on
    ``_robust.normalize``d text, else an explicit zero bound to the letter, else ``_robust.extract_number``."""
    t = _robust.normalize(text)
    v, _ = stated_count(t, letter)
    if v is not None:
        return v
    L = re.escape(letter)
    zero = [r'\b' + _ZERO_WORDS + r'\s+(?:letters?\s+|occurrences?\s+of\s+(?:the\s+letter\s+)?)?' + _Q + L + _Q
            + r'(?:\'?s)?(?![a-z])',
            r'\b(?:does\s+not|doesn\'t|do\s+not|don\'t)\s+(?:contain|have|include)\s+(?:any\s+)?(?:the\s+)?'
            r'(?:letter\s+)?' + _Q + L + _Q + r'(?:\'?s)?(?![a-z])',
            r'\b(?:letter\s+)?' + _Q + L + _Q + r'\s+(?:does\s+not|doesn\'t|never)\s+(?:appear|occur)',
            r'^\W*(?:none|zero)\W*$']
    if any(re.search(z, t, flags=re.IGNORECASE) for z in zero):
        return 0
    v = _robust.extract_number(t)
    return int(v) if v is not None and float(v).is_integer() else None


def _rate(rows, key):
    return sum(r['score'][key] for r in rows) / len(rows) if rows else None


def newcombe_diff_ci(k_b, n_b, k_a, n_a, z=1.959963984540054):
    """Newcombe (1998) hybrid score CI for p_b - p_a from Wilson intervals of independent proportions."""
    if not n_a or not n_b:
        return None
    pa, pb = k_a / n_a, k_b / n_b
    la, ua = stats.wilson(k_a, n_a, z)
    lb, ub = stats.wilson(k_b, n_b, z)
    d = pb - pa
    return [d - math.sqrt((pb - lb) ** 2 + (ua - pa) ** 2), d + math.sqrt((ub - pb) ** 2 + (pa - la) ** 2)]


# ---------------------------------------------------------------------------- earlier pipeline runs
# experiments/run_minimal_setups/runs/earlier/runs/<claim>-<variant>-L<N>[-<tag>]/<model>/: one shelflife.runner directory per model.
# Registered: every directory that sent at least one request (complete or partial); tests/test_prior_runs.py
# replays each one against this config. Excluded: aborted, smoke-test and superseded directories, and directories
# that sent nothing. Listed explicitly (written 2026-10-05 from the run directories; nothing is globbed at import).
# Runs made before the 2026-10-05 scorer patch replay byte-identically (the patch changed scoring only); their
# saved scores are pre-patch and the collector rescores them offline.

_R2 = 'experiments/run_minimal_setups/runs/earlier/runs'
_NOTHING_SENT = ('no request reached a provider: every call is not_submitted (circuit breaker open, or stale ledger '
                 'rows after a package-hash change); the directory only records the attempts')


def _unsent(*runs):
    """excluded_runs entries for per-model directories that sent nothing: (run, space-separated models)."""
    return {f'{_R2}/{run}/{m}': _NOTHING_SENT for run, models in runs for m in models.split()}


@register
class C101Strawberry(Experiment):
    claim_id = 'C101'
    claim_set_key = 'TOY-01'
    maturity = 'ready'
    title = "Toy: 'How many r's are in strawberry?' answered 2, not 3"
    claim = ("Asked 'How many r's are in strawberry?', chat LLMs often answer 2 instead of the correct 3, because "
             "tokenization hides individual characters and the repeated 'rr' is undercounted. A control question about "
             "a non-repeated letter ('How many s's are in strawberry?', answer 1) is answered correctly.")
    paper = {'title': 'Why Do Large Language Models (LLMs) Struggle to Count Letters? (closest published evidence; '
                      'the strawberry prompt itself is an informal 2024 example)',
             'authors': 'Fu, Ferrando, Conde, Arriaga, Reviriego', 'year': 2024,
             'url': 'https://arxiv.org/abs/2412.18626'}
    source = _SPEC
    default_variant = 'toy'

    def build_variants(self):
        toy = Variant(
            'toy', "Minimal design: r-question, s-control, r-wording; 10 samples each at provider-default temperature",
            prompts=dict(PROMPTS), params={'max_tokens': 1024}, repeats=REPEATS,
            options={'conditions': [c for c, _, _ in CONDITIONS], 'scorer': 'stated'},
            deviations=(
                'no paper protocol exists for the toy prompt: the protocol is our minimal design '
                '(claim record); "paper" targets come from Fu et al.\'s different JSON letter-count prompt',
                'max_tokens 1024 sent (design says provider defaults; originals_v1 requires an explicit cap; panel '
                'policies use 1024 when the source gives none and raise reasoning models to 4096)',
                'scorer: the design says "first integer or number word"; the headline is the stated count '
                '(number bound to the letter/question, else last number, list markers and "N letters" removed; number '
                'words incl. once/twice/thrice/single). The literal first-number rule is kept as a diagnostic',
                'truncated (finish_reason length) and empty outputs are excluded from the rate and bounded, like '
                'failed calls',
            ))
        robust_parse = toy.override(
            'robust_parse', description='toy requests; paper stated-count scorer first, robust fallback when it finds '
            'no count', options={'parse': 'robust'},
            deviations=('parse=robust: when the stated-count scorer finds no count, robust_count tries the same rule on '
                        '_robust.normalize()d text, an explicit zero bound to the letter ("no r\'s" = 0), then '
                        '_robust.extract_number; headline_reliable = <= 10% unparsed/missing primary rows '
                        '(_robust.reliable) instead of the stated-vs-last-number agreement rule (kept as a diagnostic)',))
        robust = robust_parse.override(
            'robust', description='robust_parse with an output-cap floor for verbose models',
            params=_robust.cap_floor(toy.params, ROBUST_CAP), options={'truncated': 'score'},
            deviations=(f'max_tokens floor {ROBUST_CAP} (toy sends 1024; M045 was truncated on 3/30 answers)',
                        "truncated='score': an answer cut off at the 2048-token floor (or the reasoning cap) is scored "
                        'by the stated-count rule on its text (no count -> incorrect), not excluded'))
        return {'toy': toy, 'robust_parse': robust_parse, 'robust': robust}

    def items(self, variant, limit=None):
        rows = []
        for cond, letter, answer in CONDITIONS:
            if cond in variant.options['conditions']:
                rows.append({'id': f'C101:{cond}', 'condition': cond, 'word': 'strawberry', 'letter': letter,
                             'answer': answer, 'multiplicity': 'strawberry'.count(letter)})
        assert all(x['answer'] == x['multiplicity'] for x in rows)
        return rows[:limit] if limit else rows

    def program(self, variant):
        return SingleTurn(lambda item: [{'role': 'user', 'content': variant.prompts[item['condition']].render()}],
                          variant.params)

    def score(self, item, output, variant):
        output = output or {}
        text = output.get('content') or ''
        truncated = output.get('finish_reason') in {'length', 'max_tokens'} or output.get('status') == 'truncated'
        empty = not text.strip()
        stated, how = stated_count(text, item['letter'])
        path = 'paper' if stated is not None else None
        if stated is None and variant.options.get('parse', 'paper') == 'robust':
            stated = robust_count(text, item['letter'])
            how, path = ('robust', 'robust') if stated is not None else (how, None)
        first, last = first_number(text), cleaned_last(text)
        usable, cut_scored = not (truncated or empty), False
        if truncated and variant.options.get('parse', 'paper') == 'robust' and not _robust.truncation_missing(variant):
            # truncated='score' (robust, 2048-token floor): a count cut off at that budget failed the task; it is
            # scored by the stated-count rule on its text (no count -> incorrect), even when empty
            usable = cut_scored = True
            if path is None:
                path = 'paper'
        return {'stated': stated, 'how': how, 'first_number': first, 'cleaned_last': last,
                'truncated': truncated, 'empty': empty, 'usable': usable,
                **({'truncated_scored': True} if cut_scored else {}),
                'parsed': stated is not None, 'correct': stated == item['answer'],
                'correct_first_number': first == item['answer'], 'answered_two': stated == 2,
                # a robust-path count has no paper-parser number to disagree with (last is None or the same value)
                'agree': stated == last or (path == 'robust' and last is None), 'parse_path': path}

    def aggregate(self, rows, variant):
        by = collections.defaultdict(list)
        for r in rows:
            by[r['item']['condition']].append(r)
        per = {}
        for cond, rs in by.items():
            done = [r for r in rs if r.get('score') is not None]
            scored = [r for r in done if r['score']['usable']]
            k = sum(r['score']['correct'] for r in scored)
            per[cond] = {
                'n': len(rs), 'scored': len(scored), 'failed_calls': len(rs) - len(done),
                'truncated': sum(r['score']['truncated'] for r in done),
                'empty': sum(r['score']['empty'] and not r['score']['truncated'] for r in done),
                'unparsed': sum(not r['score']['parsed'] and not r['score'].get('truncated_scored') for r in scored),
                'parsed_paper': sum(r['score'].get('parse_path', 'paper' if r['score']['parsed'] else None) == 'paper'
                                    for r in scored),
                'parsed_robust': sum(r['score'].get('parse_path') == 'robust' for r in scored),
                'k_correct': k, 'accuracy': k / len(scored) if scored else None,
                'accuracy_ci95_exact': list(stats.clopper_pearson(k, len(scored))) if scored else None,
                'accuracy_bounds_with_missing': list(stats.bounds_with_missing(k, len(scored) - k, len(rs) - len(scored))),
                'accuracy_first_number_rule': _rate(scored, 'correct_first_number'),
                'agreement_stated_vs_cleaned_last': _rate(scored, 'agree'),
                'bound_share': _rate([dict(r, score={'b': r['score']['how'] == 'bound'}) for r in scored], 'b'),
                'answer_counts': dict(collections.Counter(str(r['score']['stated']) for r in scored)),
                'share_answered_two': _rate(scored, 'answered_two'),
            }
        primary = [per[c]['agreement_stated_vs_cleaned_last'] for c in ('r_toy', 's_control') if c in per]
        out = {'per_prompt': per,
               'headline_reliable': bool(primary) and all(x is not None and x >= 0.95 for x in primary)}
        robust_mode = variant.options.get('parse', 'paper') == 'robust'
        if robust_mode:
            prim = [per[c] for c in ('r_toy', 's_control') if c in per]
            n_p = sum(p['n'] for p in prim)
            unp = sum(p['unparsed'] for p in prim)
            miss = sum(p['n'] - p['scored'] for p in prim)
            out['validity'] = {'n_primary': n_p, 'unparsed': unp, 'missing': miss,
                               'valid_rate': (n_p - unp - miss) / n_p if n_p else None,
                               'tolerance': _robust.ROBUST_TOLERANCE,
                               'agreement_rule_reliable': out['headline_reliable']}   # paper rule, now a diagnostic
            out['headline_reliable'] = _robust.reliable(n_p, unp, miss)
            for cond, rs in by.items():
                per[cond]['truncated_scored'] = sum(bool((r.get('score') or {}).get('truncated_scored')) for r in rs)
            out['truncated_scored'] = sum(p['truncated_scored'] for p in per.values())
        out['parsed_paper'] = sum(p['parsed_paper'] for p in per.values())
        out['parsed_robust'] = sum(p['parsed_robust'] for p in per.values())
        out['unparsed'] = sum(p['unparsed'] for p in per.values())
        a = [r for r in by.get('r_toy', []) if r.get('score') is not None and r['score']['usable']]
        c = [r for r in by.get('s_control', []) if r.get('score') is not None and r['score']['usable']]
        ka, kc = sum(r['score']['correct'] for r in a), sum(r['score']['correct'] for r in c)
        value = (kc / len(c) - ka / len(a)) if a and c else None
        ci = newcombe_diff_ci(kc, len(c), ka, len(a))
        out['persists'] = None if ci is None else ci[0] > 0
        acc_c = kc / len(c) if c else None
        interpretable = None if acc_c is None else acc_c >= CONTROL_GATE
        out['shelflife'] = {
            'metric': 'acc_control_minus_acc_r', 'value': value, 'ci': ci,
            'n': len(a) + len(c), 'direction': +1, 'null_value': 0.0, 'paper_value': None,
            'paper_value_note': ('no paper number exists for this prompt; the anchor is our reproduction on '
                                 'openai-direct/gpt-4o-2024-05-13 (gpt-4o alias during the viral 2024 period)'),
            'anchor_model': 'openai-direct/gpt-4o-2024-05-13',
            'headline_reliable': out['headline_reliable'],
            'valid_rate': (out.get('validity') or {}).get('valid_rate'),
            'primary_metric': 'acc_r_toy', 'primary_value': ka / len(a) if a else None,
            'primary_ci': list(stats.clopper_pearson(ka, len(a))) if a else None, 'primary_direction': -1,
            'control_accuracy': acc_c, 'control_gate': CONTROL_GATE,
            'difference_interpretable': interpretable,
            'difference_note': (None if interpretable is not False else
                                f'control accuracy {acc_c:.2f} < {CONTROL_GATE}: the difference mixes a control failure '
                                'with the r failure; read primary_value (acc r_toy) instead'),
            'share_r_toy_answered_two': out['per_prompt'].get('r_toy', {}).get('share_answered_two'),
            'scoring_deviations': list(SCORING_DEVIATIONS),
            'definition': ("Accuracy of the stated count (== 1) on \"How many s's are in strawberry?\" minus accuracy "
                           "(== 3) on \"How many r's are in strawberry?\", 10 samples each; Newcombe score CI."),
        }
        if a and c:
            out['diagnostic_first_number_rule_value'] = (sum(r['score']['correct_first_number'] for r in c) / len(c)
                                                         - sum(r['score']['correct_first_number'] for r in a) / len(a))
        return out

    # ------------------------------------------------------------------ Fu et al. released outputs (mechanism)
    @staticmethod
    def fu_outputs(model='gpt4o'):
        """Fu et al. released outputs: {word: parsed JSON dict or the raw string when it does not parse}."""
        raw = json.loads((FU / 'Results' / f'{model}.json').read_text())
        out = {}
        for word, value in raw.items():
            try:
                out[word] = json.loads(value)
            except (TypeError, ValueError):
                out[word] = value
        return out

    @staticmethod
    def fu_words():
        import csv
        with open(FU / 'Results' / 'Random_10000_words.csv', newline='', encoding='utf-8') as f:
            return list(csv.DictReader(f))

    def fu_word_error_rate(self, model='gpt4o'):
        """Share of the 10,000 words whose returned JSON is not exactly the true letter-count dict (unparsed = error)."""
        truth = json.loads((FU / 'Words' / 'Random_10000_words_letterCount.json').read_text())
        outputs = self.fu_outputs(model)
        words = [r['Word'] for r in self.fu_words()]
        return sum(not (isinstance(outputs[w], dict) and outputs[w] == truth[w]) for w in words) / len(words)

    def fu_multiplicity_error(self, multiplicity, model='gpt4o'):
        """Per-letter error rate at a letter multiplicity, with the authors' own letter_count (Fig. 10 script)."""
        letter_count = load_functions(FU / 'PlotBarchartVSLetterMultiplicity.py', ['letter_count'],
                                      extra_globals={'Counter': collections.Counter})['letter_count']
        outputs = self.fu_outputs(model)
        wrong = total = 0
        for row in self.fu_words():
            c = letter_count(row['Word'], outputs[row['Word']])
            wrong += c[multiplicity]['wrong']
            total += c[multiplicity]['total']
        return wrong / total

    targets = (
        PaperTarget('fu_gpt4o_word_error_rate', 0.17, 0.005, 'arXiv 2412.18626v1 Sec. 4 text ("even the best model '
                    'fails to correctly count the letters in 17% of the words"; best = GPT-4o, Fig. 1)',
                    'gpt-4o (snapshot unstated, Dec 2024)', 'toy', 'archived_outputs',
                    compute=lambda exp: exp.fu_word_error_rate('gpt4o'),
                    note='Fu et al. JSON letter-count prompt on 10,000 words, NOT the toy prompt: anchors the cited '
                         'mechanism on the original model family. Tolerance 0.005 = rounding of "17%" (we get 0.173, '
                         'identical to the authors\' GPT4o_IFCorrect column).'),
        PaperTarget('fu_gpt4o_letter_error_multiplicity2', 0.14, 0.01, 'arXiv 2412.18626v1 Fig. 10 (GPT4o bar at '
                    'multiplicity 2)', 'gpt-4o (snapshot unstated, Dec 2024)', 'toy', 'archived_outputs',
                    compute=lambda exp: exp.fu_multiplicity_error(2, 'gpt4o'),
                    note='Read off the bar chart (~14%); tolerance 0.01 covers figure reading. The authors\' '
                         'letter_count skips letters missing from the returned JSON (kept as in their script).'),
        PaperTarget('fu_gpt4o_letter_error_multiplicity1', 0.003, 0.003, 'arXiv 2412.18626v1 Fig. 10 (GPT4o bar at '
                    'multiplicity 1, near zero)', 'gpt-4o (snapshot unstated, Dec 2024)', 'toy', 'archived_outputs',
                    compute=lambda exp: exp.fu_multiplicity_error(1, 'gpt4o'),
                    note='Read off the bar chart (bar barely visible, < 0.5%); tolerance covers figure reading. Depends on the '
                         'authors\' convention: letters absent from the JSON count in total but never as wrong (counting '
                         'them wrong gives 0.0061, outside the tolerance).'),
    )
