"""C059 — Measuring and Narrowing the Compositionality Gap in Language Models (Press et al. 2022)

Claim-set key: C059 (earlier pipeline). Claim (as adjudicated): LLMs often answer both single-hop sub-questions correctly
yet fail the composed two-hop question (compositionality gap on Compositional Celebrities).

Paper: arXiv 2210.03350v3 (Findings of EMNLP 2023 camera-ready). Sec 2 (CC protocol), Fig 1, Appendix Tables 4-5
(prompts) and Table 6 (per-category 2x2 breakdown, davinci-002, direct 2-shot), footnote 2 in Sec 1 (ChatGPT 42.9%,
GPT-4 23.0%). Headline: davinci-002 answers 45.4% of the 2-hop questions; Table 6 ALL row: both sub-questions right
75.2% (41.8 + 33.4), so the compositionality gap is 33.4 / 75.2 = 44.4% (Fig 1: "roughly constant 40%").

Sources
  * data: ofirpress/self-ask @ 559e56ec, datasets/compositional_celebrities.json (8,693 questions, 17 categories, each
    with its two sub-questions Q1/A1, Q2/A2). No prompts file, exemplar list, scorer or model outputs were released.
  * prompt format: the paper's LaTeX (data/sources/C059_press2022_arxiv_v3/src/tables/cc-prompts.tex, Appendix
    Tables 4-5). tests/claims/test_c059.py re-parses those tables and checks our rendered exemplar blocks byte for byte.

Protocol (variant ``direct_2shot``, default, faithful; also the run variant)
  Every 2-hop question and each of its two sub-questions is asked separately, with a category-specific 2-shot prompt:
      Question: <exemplar 1 question>\\nAnswer: <exemplar 1 answer>\\n\\nQuestion: <exemplar 2 ...>\\nAnswer: <...>\\n\\n
      Question: <test question>\\nAnswer:
  sent as one chat user message. 2-hop prompts use 2-hop exemplars of the same category; sub-question prompts use the
  same exemplar people's sub-questions of the same category and hop (Q1 with Q1s, Q2 with Q2s).
  Exemplars. The paper says the 2 exemplars per category were "randomly chosen examples that have been removed from
  the dataset", but the released file still contains them: both printed prompts use Antonio José de Sucre (Venezuela)
  then Martina Navratilova (Czechia), so for the 14 birthplace categories we use those two people, in that order, and
  remove their rows from evaluation. The birth-date exemplars were never shown; we draw two people present exactly
  once in each of the 3 birth-date categories with random.Random(EXEMPLAR_SEED) and remove their rows. The released
  set minus the exemplar rows has 8,659 questions ("8.6k" in the paper).

Items: one row per question (kind 'hop2' | 'sub1' | 'sub2'); the three rows of one CC entry share a ``unit`` id.
  Units are shuffled within category (random.Random(ORDER_SEED)) and interleaved round-robin over the 17 categories,
  each unit emitting hop2, sub1, sub2 consecutively, so ``--limit 3k`` gives k complete units stratified over
  categories. MINIMAL_LIMIT = 228 = 76 units (selection n_units) x 3 calls.

Scoring (reconstruction: the paper does not state its answer check; Sec 4 names exact match, Appendix reports Cover-EM)
  Normalisation: SQuAD-style (NFKC, lowercase, drop ASCII punctuation, articles, extra spaces); a gold alias that
  normalises to '' (e.g. '$') is compared on the lowercased raw string.
  1. match (HEADLINE; a documented scoring deviation): the answer window is the first LENIENT_LINES (2)
     non-empty lines, cut at a completion-style 'Question:' continuation, 'Answer:' prefixes removed. If it has at most
     LENIENT_MAX_WORDS (30) words, the output is correct iff a normalised gold alias occurs in it as a whole-token span
     (numeric gold: the window's last number, with S/W hemisphere negative, equals the gold); this tolerates a short
     preamble ("X was born in Venezuela. Caracas") and 2-line answers. A longer window falls back to strict EM.
  2. em (diagnostic, the pre-audit headline): the first non-empty line, 'Answer:' and trailing '.' removed, equals a
     gold alias after normalisation (numeric: a single integer, degree sign / unicode minus tolerated).
  3. cover_em (diagnostic): alias as token span of that first line (numeric: its first integer); cover_full: alias
     anywhere in the whole output (used only for the reliability flag).
  Each score also carries ``truncated`` (finish_reason == 'length') and ``empty``.
  Aggregate: the 2x2 table over units with all three outputs, the gap P(2-hop wrong | both subs right)
  POST-STRATIFIED to the released category sizes (w_c = N_c / n_c, recomputed inside the bootstrap), so it estimates
  the same question-weighted quantity as Table 6's ALL row (0.444); the unweighted gap and the strict-EM / cover-EM gaps
  are reported as secondaries. CIs: bootstrap clustered on person (a person can appear in several categories).
  ``paper_value_design_mix`` = 0.397 is Table 6's davinci-002 rows pooled at the minimal design's category mix.
  Per question kind: match / strict-EM rates, the lenient-extractor rescue rate, 'gold in output but not extracted',
  truncation and empty rates. ``headline_reliable`` is False when the 2-hop 'gold in output but not extracted' rate
  exceeds the sub-question mean by more than 10 points, or 2-hop truncation exceeds 2%.

Deviations (all variants)
  * davinci-002 (GPT-3 completion endpoint) is retired; the prompt is sent as one chat user message.
  * Generation parameters are not reported: temperature 0, max_tokens 32 (answers are short entities / numbers).
  * The first exemplar line in Table 5 (2-hop prompt) lacks 'Question: '; treated as a typesetting omission (Table 4
    and the second exemplar have it).
  * Birth-date exemplars are our seeded draw; the scorer is a reconstruction (above).
  * Not registered: CoT / self-ask on CC (Appendix Fig. 8; prompts for CC not given) and Bamboogle / 2Wiki / Musique.

Robustness variants (non-faithful)
  * ``robust_parse``: direct_2shot requests unchanged, options parse='robust'. The paper scorer runs first; only when its
    answer window is None (first two lines > 30 words) and strict EM fails, a short answer is taken from the
    ``_robust.normalize``d text (robust_window): the first line after the last explicit answer marker (not filler such
    as 'the answer is straightforward'), else a leading **bold** span, else a short first line (<= 8 words) opening a
    longer reply, else the last line, else, for a reply of at most 2 lines, its first line or first sentence; the first
    candidate with <= 30 words that is not a reasoning preamble or closing remark ('Step 1', 'To answer ...', "Let's",
    'First', 'We need', 'Identify ...', headings and lead-ins ending in ':', 'If you need ...', 'Note: ...') wins; the
    same token-span / last-number match is applied to it. (Revised 2026-10-10 after the panel robust run: the old
    order took the first line of worked answers, e.g. 'To answer the question, we need to ...', as the answer.) Unparsed rows count as missing in the headline 2x2 table (their unit is
    dropped) instead of as wrong answers. headline_reliable = _robust.reliable per question kind (<= 10% unparsed or
    missing) and no excess 'gold in output but not extracted' on 2-hop; parsed_paper / parsed_robust / unparsed are
    reported per kind.
    Where robust differs from paper on rows the paper scorer DID score (documented exceptions):
      - empty outputs (paper: wrong) are unparsed;
      - (robust_parse only) every truncated output (finish_reason == 'length') is unparsed, matched or not (paper:
        scored on its text).
        The answer may have been cut off, and excluding all of them, not only the failures, keeps the exclusion
        independent of the outcome. Both kinds drop their unit from the headline table;
      - the fallback above re-scores rows whose paper window was None (paper: strict EM on the first line).
  * ``robust``: robust_parse with max_tokens floored at 1024 (_robust.cap_floor; verbose chat models answered the
    2-hop question with a bridge sentence and 13-70% hit the 32-token cap) and min_cap_reasoning 16384 (reasoning
    models M143/M144/M190 returned empty content at the 4096 policy cap), plus option truncated='score'
    (_robust.truncation_missing): at 1024 / 16384 tokens for a one-entity answer a cut-off output is a task failure,
    not a cap artefact, so it is NOT unparsed: it takes the same paper-window / fallback path as any row and, when
    neither finds a short answer (or the text is empty), the paper's strict EM on the first line decides (wrong unless
    it is the gold). Counted as ``truncated_scored`` (per kind and overall). For robust the truncated-output exception
    above therefore does not apply; the documented exception is instead: an EMPTY cut-off output is wrong (paper
    rule), while an empty complete output stays unparsed. robust_parse keeps truncated='missing'.

"""
import functools
import json
import math
import random
import re
import string
import unicodedata
from collections import defaultdict

from ..paths import PROJECT
from ..program import SingleTurn
from ..prompts import PromptTemplate, Source
from ..registry import Experiment, PaperTarget, Variant, register
from .. import stats
from . import _robust

REPO = PROJECT / 'code/shelflife/data/sources/C059_self-ask'
COMMIT = '559e56ec1ba8dc93cf63c4148ec1f4717caf4b60'
PAPER_SRC = PROJECT / 'code/shelflife/data/sources/C059_press2022_arxiv_v3'
SRC = Source(repo=str(REPO), path='datasets/compositional_celebrities.json', locator='Compositional Celebrities release',
             commit=COMMIT, url='https://github.com/ofirpress/self-ask')

EXEMPLAR_SEED = 2210_03350
ORDER_SEED = 20221007
BIRTHPLACE_EXEMPLARS = ('Antonio José de Sucre', 'Martina Navratilova')   # Appendix Tables 4-5, in this order
MINIMAL_LIMIT = 228                     # 76 units (selection n_units) x 3 questions
RUN_VARIANT = 'direct_2shot'
REPEATS = 1
REPRO_MODELS = ['openai/gpt-3.5-turbo-0613', 'openai-direct/gpt-4-0613']
REPRO_LIMIT = 612                       # 204 units = 12 per category x 17
VERBOSE_THRESHOLD = 0.10                # hop2 minus sub 'gold present but not extracted' rate
TRUNCATION_THRESHOLD = 0.02             # hop2 finish_reason == 'length' rate
LENIENT_LINES = 2
N_BOOT = 2000
LENIENT_MAX_WORDS = 30
ROBUST_MIN_CAP_REASONING = 16384        # robust variant: reasoning models emptied the 4096 policy cap
claim_set_key = 'C059'

TEMPLATE = PromptTemplate(
    'user', 'Question: {ex1_q}\nAnswer: {ex1_a}\n\nQuestion: {ex2_q}\nAnswer: {ex2_a}\n\nQuestion: {question}\nAnswer:',
    source=Source(repo=str(PAPER_SRC), path='src/tables/cc-prompts.tex',
                  locator='Appendix Tables 4-5 (tab:ccprompts-subq, tab:ccprompts-fq)',
                  url='https://arxiv.org/abs/2210.03350v3',
                  note='2-shot format; one table row per line, blank row = empty line'),
    note='exemplars are category-specific; test_c059 checks the rendered currency / capital blocks against the LaTeX')

HOPS = {'hop2': ('Question', 'Answer'), 'sub1': ('Q1', 'A1'), 'sub2': ('Q2', 'A2')}
DEVIATIONS = (
    'davinci-002 completion endpoint retired: the 2-shot prompt is sent as one chat user message',
    'generation parameters not reported: temperature 0, max_tokens 32; routes that reject temperature (some reasoning '
    'models under panel_v2) sample at the provider default, and reasoning models get the policy cap (4096)',
    "Table 5's first exemplar line lacks 'Question: ' (typesetting omission); both exemplars use it here",
    'birth-date exemplars not shown in the paper: two people drawn with random.Random(EXEMPLAR_SEED)',
    'answer check not released: headline = gold alias as a token span of the first <=2 answer lines (<=30 words; '
    'numeric: last number, S/W negative); strict first-line exact match and cover-EM kept as diagnostics',
    'headline gap post-stratified to the released category sizes (the design samples categories evenly)',
)


# ---------------------------------------------------------------- data
@functools.lru_cache(maxsize=1)
def _raw():
    return tuple(json.loads((REPO / 'datasets/compositional_celebrities.json').read_text())['data'])


def load_cc():
    """Released CC rows (file order) with their file index added; fresh copies."""
    return [{**r, 'index': i} for i, r in enumerate(_raw())]


def categories():
    out = []
    for r in _raw():
        if r['category'] not in out:
            out.append(r['category'])
    return out


def _ans(value):
    return str(value)


@functools.lru_cache(maxsize=1)
def exemplars():
    """{category: [row, row]} exemplar rows (in prompt order)."""
    rows = load_cc()
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r['category']].append(r)
    out = {}
    date_cats = [c for c in categories() if not c.startswith('birthplace')]
    once = None
    for c in date_cats:
        counts = defaultdict(int)
        for r in by_cat[c]:
            counts[r['person']] += 1
        names = {p for p, k in counts.items() if k == 1}
        once = names if once is None else once & names
    date_people = random.Random(EXEMPLAR_SEED).sample(sorted(once), 2)
    for c in categories():
        people = BIRTHPLACE_EXEMPLARS if c.startswith('birthplace') else date_people
        picked = []
        for p in people:
            hits = [r for r in by_cat[c] if r['person'] == p]
            if len(hits) != 1:
                raise ValueError(f'{c}: exemplar {p} matches {len(hits)} rows')
            picked.append(hits[0])
        out[c] = picked
    return out


def exemplar_people():
    return {c: [r['person'] for r in rows] for c, rows in exemplars().items()}


def render(item):
    ex = item['exemplars']
    return TEMPLATE.render(ex1_q=ex[0][0], ex1_a=ex[0][1], ex2_q=ex[1][0], ex2_a=ex[1][1], question=item['question'])


@functools.lru_cache(maxsize=1)
def _units():
    """Evaluation units (CC rows minus exemplar rows) in limit order."""
    ex = exemplars()
    removed = {(c, r['person']) for c, rows in ex.items() for r in rows}
    by_cat = defaultdict(list)
    for r in load_cc():
        if (r['category'], r['person']) not in removed:
            by_cat[r['category']].append(r)
    rng = random.Random(ORDER_SEED)
    for c in categories():
        rng.shuffle(by_cat[c])
    order, depth = [], max(len(v) for v in by_cat.values())
    for k in range(depth):
        for c in categories():
            if k < len(by_cat[c]):
                order.append(by_cat[c][k])
    return tuple(order)


def build_items():
    ex = exemplars()
    items = []
    for r in _units():
        unit = f'C059:{r["index"]:04d}'
        for kind, (qk, ak) in HOPS.items():
            golds = [_ans(a) for a in r[ak]]
            items.append({'id': f'{unit}:{kind}', 'unit': unit, 'kind': kind, 'category': r['category'],
                          'person': r['person'], 'question': r[qk], 'answers': golds,
                          'numeric': all(isinstance(a, int) for a in r[ak]),
                          'exemplars': [[e[qk], _ans(e[ak][0])] for e in ex[r['category']]]})
    return items


# ---------------------------------------------------------------- scoring
ARTICLES = re.compile(r'\b(a|an|the)\b')
INT = re.compile(r'[+\-\u2212]?\d+')
NUM = re.compile(r'([+\-\u2212]?\d+(?:\.\d+)?)\s*°?\s*([NSEW])?\b')


def normalize(text):
    text = unicodedata.normalize('NFKC', str(text)).lower()
    stripped = ''.join(ch for ch in text if ch not in set(string.punctuation))
    stripped = ' '.join(ARTICLES.sub(' ', stripped).split())
    return stripped if stripped else ' '.join(text.split())


def _strip_answer_prefix(line):
    return re.sub(r'^\**answer\**\s*:\s*\**', '', line.strip(), flags=re.I).strip()


def answer_line(text):
    """Strict: first non-empty line, 'Answer:' prefix and trailing '.' removed."""
    for line in (text or '').splitlines():
        line = line.strip()
        if line:
            return _strip_answer_prefix(line).rstrip('.').strip()
    return ''


def answer_window(text):
    """Lenient (headline): the first LENIENT_LINES non-empty lines, stopping at a completion-style continuation
    ('Question: ...'), each without an 'Answer:' prefix, joined by spaces. Longer than LENIENT_MAX_WORDS words -> None
    (too long to be a short answer: the strict first line is used instead)."""
    lines = []
    for line in (text or '').splitlines():
        line = line.strip()
        if not line:
            continue
        if re.match(r'^\**question\**\s*:', line, flags=re.I):
            break
        lines.append(_strip_answer_prefix(line))
        if len(lines) == LENIENT_LINES:
            break
    window = ' '.join(lines).strip()
    return window if len(window.split()) <= LENIENT_MAX_WORDS else None


def _first_int(text):
    m = INT.search(text.replace('\u2212', '-'))
    return int(m.group().replace('\u2212', '-')) if m else None


def _last_number(text):
    """Last number in text as float, negated for an S/W hemisphere suffix ('66°W' -> -66)."""
    hits = NUM.findall(text.replace('\u2212', '-'))
    if not hits:
        return None
    value, hemi = hits[-1]
    v = float(value)
    return -abs(v) if hemi in ('S', 'W') else v


def _span_match(text, golds):
    p = normalize(text)
    return any(f' {normalize(g)} ' in f' {p} ' for g in golds if normalize(g))


# robust_window: lines that read as the start of a reasoning trace, not as an answer (checked after stripping a list
# marker): 'Step 1: ...', 'To answer the question, we need to ...', "Let's ...", 'First, ...', 'We need ...',
# 'Identify the birthplace ...', courtesy openers, closing remarks ('If you need ...', 'Note: ...'), and any line ending
# in ':' (a heading or lead-in)
PREAMBLE = re.compile(
    r"^(?:step\s*\d+|to\s+(?:answer|determine|find|solve|identify|figure|work|get|address)\b|let'?s\b|let\s+me\b|"
    r"first(?:ly)?\b|next\b|then\b|now\b|we\s+(?:need|must|have|should|can|will)\b|i\s+(?:need|will|must|should|'ll)\b|"
    r"in\s+order\s+to\b|the\s+(?:question|task)\b|this\s+question\b|(?:identify|determine|find|figure|calculate|"
    r"check|recall|confirm|look\s+up|translate|consider|break)\b|okay\b|ok\b|alright\b|sure\b|certainly\b|great\b|"
    r"good\s+question\b|here'?s\b|here\s+is\b|reasoning\b|thinking\b|analysis\b|solution\b|explanation\b|"
    # closing remarks, not answers: 'If you need ...', 'Let me know ...', 'I hope ...', 'Note: ...', 'Please ...'
    r"if\s+you\b|i\s+hope\b|hope\s+this\b|feel\s+free\b|note\b|please\b)",
    re.I)
# a marker that is not an answer: 'the answer is straightforward.', 'the answer is as follows'
MARKER_FILLER = re.compile(r'^(?:quite\s+|very\s+|fairly\s+|pretty\s+)?(?:straightforward|simple|clear|obvious|easy|'
                           r'tricky|complex|complicated|below|as\s+follows|follows|not\s+straightforward)\W*$', re.I)
SHORT_FIRST_LINE_WORDS = 8     # a first line this short opening a longer reply is a short answer ('Caracas', 'Iceland.')


def _list_stripped(line):
    return re.sub(r'^\s*(?:[-*\u2022]|\d+[.)])\s+', '', line).strip()


def is_preamble(line):
    """True if a candidate line reads as a reasoning preamble or heading rather than an answer (robust_window)."""
    s = _list_stripped(line)
    return not s or s.endswith(':') or bool(PREAMBLE.match(s))


def robust_window(text):
    """Fallback short answer (parse='robust', used only when the paper window is None), on the _robust-normalized text
    (lines stop at a completion-style 'Question:' continuation). Candidates in order, the first one that has
    1..LENIENT_MAX_WORDS words and is not a reasoning preamble (is_preamble: 'Step 1', 'To answer ...', "Let's",
    'First', 'We need', 'Identify ...', headings / lines ending in ':') wins:
      1. the first line after the last explicit answer marker (_robust.answer_segment: 'Final answer:', 'the answer
         is', 'Answer:'), unless it is filler ('the answer is straightforward.', MARKER_FILLER);
      2. a leading **bold** span ('**Samuel Beckett** won ...');
      3. a short first line (<= SHORT_FIRST_LINE_WORDS words) opening a longer reply ('Caracas\n\n<explanation>');
      4. the last line (the conclusion of a worked answer);
      5. only for a reply of at most LENIENT_LINES lines (a short answer, not a trace): its first line, then its first
         sentence.
    The rule never looks at the gold, so it is outcome-independent."""
    t = _robust.normalize(text)
    if not t:
        return None
    lines = []
    for line in t.splitlines():
        line = line.strip()
        if not line:
            continue
        if re.match(r'^question\s*:', line, flags=re.I):
            break
        lines.append(line)
    cands = []
    seg = _robust.answer_segment(text)
    if seg != t:
        seg_lines = [x.strip() for x in seg.splitlines() if x.strip()]
        if seg_lines and not MARKER_FILLER.match(seg_lines[0].strip()):
            cands.append(seg_lines[0])
    bold = re.match(r'\s*\*\*([^*\n]+)\*\*', str(text))       # a leading bold answer: '**Samuel Beckett** won ...'
    if bold:
        cands.append(_robust.normalize(bold.group(1)))
    if lines:
        if len(lines[0].split()) <= SHORT_FIRST_LINE_WORDS:
            cands.append(lines[0])
        cands.append(lines[-1])
        if len(lines) <= LENIENT_LINES:
            first_sentence = re.split(r'(?<=[a-z0-9)\]])[.!?]\s+(?=[A-Z(])', lines[0], maxsplit=1)[0]
            cands += [lines[0], first_sentence]
    for c in cands:
        c = _strip_answer_prefix(c).rstrip('.').strip()
        if c and len(c.split()) <= LENIENT_MAX_WORDS and not is_preamble(c):
            return c
    return None


def score_answer(text, golds, numeric, finish_reason=None, parse='paper', cut_missing=True):
    ans = answer_line(text)
    window = answer_window(text)
    if numeric:
        gold = {int(g) for g in golds}
        bare = ans.replace('\u2212', '-').replace('°', '').strip()
        em = bool(re.fullmatch(r'[+-]?\d+', bare)) and int(bare) in gold
        cover = _first_int(ans) in gold
        last = lambda t: _last_number(t)
        match = (last(window) in gold) if window else em
        full = last(text or '') in gold
    else:
        p = normalize(ans)
        em = any(p == normalize(g) for g in golds)
        cover = _span_match(ans, golds)
        match = (_span_match(window, golds) or em) if window is not None else em
        full = _span_match(text or '', golds)
    out = {'answer': ans, 'window': window, 'match': bool(match), 'em': bool(em), 'cover_em': bool(cover or em),
           'cover_full': bool(full or match), 'truncated': finish_reason == 'length',
           'empty': not (text or '').strip()}
    if parse == 'robust':
        # paper first: its decision stands whenever its short-answer window exists (or strict EM matched)
        path, rmatch, rwin = None, bool(match), None
        if out['empty']:
            path = None
        elif window is not None or em:
            path = 'paper'
        else:
            rwin = robust_window(text)
            if rwin is not None:
                path = 'robust'
                rmatch = (_last_number(rwin) in {int(g) for g in golds}) if numeric else _span_match(rwin, golds)
        if out['truncated'] and cut_missing:
            path = None            # robust_parse: cut off = unparsed whatever it contains (outcome-independent)
        elif out['truncated']:
            # robust (truncated='score', generous cap): a cut-off row is a task failure, scored by the paper rule on
            # its text; when neither window nor fallback found a short answer (or it is empty) the paper's strict EM
            # decides (wrong unless the first line is the gold)
            if path is None:
                path, rmatch = 'paper', bool(match)
            out['truncated_scored'] = True
        out.update(match=bool(rmatch), cover_full=bool(full or rmatch), robust_window=rwin, parse_path=path,
                   unparsed=path is None)
    return out


# ---------------------------------------------------------------- aggregation
@functools.lru_cache(maxsize=1)
def released_category_sizes():
    sizes = defaultdict(int)
    for r in _raw():
        sizes[r['category']] += 1
    return dict(sizes)


def _units_table(rows, key):
    """{unit: {'hop2','sub1','sub2': bool, 'category', 'person', 'unit'}} for units with all three scored."""
    by_unit = defaultdict(dict)
    meta = {}
    for r in rows:
        it = r['item']
        meta[it['unit']] = (it['category'], it.get('person', it['unit']))
        by_unit[it['unit']][it['kind']] = None if r.get('score') is None else r['score'][key]
    complete = {u: {**v, 'category': meta[u][0], 'person': meta[u][1], 'unit': u} for u, v in by_unit.items()
                if set(v) == set(HOPS) and all(x is not None for x in v.values())}
    return complete, len(by_unit) - len(complete)


def _gap(units):
    both = [u for u in units if u['sub1'] and u['sub2']]
    return sum(not u['hop2'] for u in both) / len(both) if both else float('nan')


def _gap_weighted(units):
    """Post-stratified gap: category c weighted by w_c = N_c (released size) / n_c (units of c in this sample), i.e.
    sum_c w_c * #(both right & 2-hop wrong)_c / sum_c w_c * #(both right)_c."""
    n = defaultdict(int)
    for u in units:
        n[u['category']] += 1
    sizes = released_category_sizes()
    num = den = 0.0
    for u in units:
        if u['sub1'] and u['sub2']:
            w = sizes[u['category']] / n[u['category']]
            den += w
            num += w * (not u['hop2'])
    return num / den if den else float('nan')


def _person(u):
    return u['person']


def _clean(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else v


def _table(units, key, boot=True):
    """2x2 table and gaps; CIs only when boot (the headline scorer), points otherwise."""
    vals = list(units.values())
    n = len(vals)
    if not n:
        return {'n_units': 0, 'scorer': key, 'compositionality_gap': None, 'compositionality_gap_unweighted': None,
                'gap_ci95_bootstrap': None, 'n_both_subs_right': 0}
    both = [u for u in vals if u['sub1'] and u['sub2']]
    other = [u for u in vals if not (u['sub1'] and u['sub2'])]
    diff = lambda us: (sum(u['sub1'] and u['sub2'] for u in us) - sum(u['hop2'] for u in us)) / len(us)
    if boot:
        gw, lo, hi = stats.cluster_bootstrap(vals, _person, _gap_weighted, n_boot=N_BOOT, seed=0)
        gu, ulo, uhi = stats.cluster_bootstrap(vals, _person, _gap, n_boot=N_BOOT, seed=0)
        d, dlo, dhi = stats.cluster_bootstrap(vals, _person, diff, n_boot=N_BOOT, seed=0)
    else:
        gw, gu, d = _gap_weighted(vals), _gap(vals), diff(vals)
        lo = hi = ulo = uhi = dlo = dhi = None
    return {
        'n_units': n, 'scorer': key,
        'acc_2hop': sum(u['hop2'] for u in vals) / n, 'acc_sub1': sum(u['sub1'] for u in vals) / n,
        'acc_sub2': sum(u['sub2'] for u in vals) / n, 'both_subs_right': len(both) / n,
        'table6_unweighted': {'both_right_2hop_right': sum(u['hop2'] for u in both) / n,
                              'both_right_2hop_wrong': sum(not u['hop2'] for u in both) / n,
                              'one_wrong_2hop_right': sum(u['hop2'] for u in other) / n,
                              'one_wrong_2hop_wrong': sum(not u['hop2'] for u in other) / n},
        'n_both_subs_right': len(both),
        'compositionality_gap': _clean(gw) if both else None,
        'gap_ci95_bootstrap': [_clean(lo), _clean(hi)] if both else None,
        'compositionality_gap_unweighted': _clean(gu) if both else None,
        'gap_unweighted_ci95': [_clean(ulo), _clean(uhi)] if both else None,
        'paired_diff_both_subs_minus_2hop_unweighted': d, 'paired_diff_ci95': [dlo, dhi],
    }


def _kind_diagnostics(rows):
    out = {}
    for kind in HOPS:
        sc = [r['score'] for r in rows if r['item']['kind'] == kind and r.get('score') is not None]
        n = len(sc)
        rate = lambda f: sum(f(s) for s in sc) / n if n else None
        out[kind] = {'n_scored': n,
                     'match_rate': rate(lambda s: s['match']),
                     'strict_em_rate': rate(lambda s: s['em']),
                     'rescued_by_lenient_extractor': rate(lambda s: s['match'] and not s['em']),
                     'gold_in_output_not_extracted': rate(lambda s: s['cover_full'] and not s['match']),
                     'truncated_rate': rate(lambda s: s['truncated']),
                     'empty_rate': rate(lambda s: s['empty'])}
        if sc and 'parse_path' in sc[0]:
            out[kind].update({'n_rows': sum(r['item']['kind'] == kind for r in rows),
                              'missing': sum(r['item']['kind'] == kind and r.get('score') is None for r in rows),
                              'parsed_paper': sum(s['parse_path'] == 'paper' for s in sc),
                              'parsed_robust': sum(s['parse_path'] == 'robust' for s in sc),
                              'unparsed': sum(s['parse_path'] is None for s in sc),
                              'truncated_scored': sum(bool(s.get('truncated_scored')) for s in sc)})
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


_SWEEP = 'experiments/initial-reproduction/runs/runs/C059'


@register
class C059CompositionalityGap(Experiment):
    claim_id = 'C059'
    claim_set_key = 'C059'
    maturity = 'ready'
    title = 'Measuring and Narrowing the Compositionality Gap in Language Models'
    claim = ('LLMs often answer both single-hop sub-questions correctly yet fail the composed two-hop question '
             '(compositionality gap on Compositional Celebrities).')
    paper = {'title': 'Measuring and Narrowing the Compositionality Gap in Language Models',
             'authors': ['Ofir Press', 'Muru Zhang', 'Sewon Min', 'Ludwig Schmidt', 'Noah A. Smith', 'Mike Lewis'],
             'first_date': '2022-10-07', 'venue': 'Findings of EMNLP 2023', 'url': 'https://arxiv.org/abs/2210.03350',
             'version': 'v3 (targets and prompt format)'}
    source = SRC
    default_variant = 'direct_2shot'

    def build_variants(self):
        v = Variant('direct_2shot', 'Sec 2 / Appendix Tables 4-6: direct category-specific 2-shot prompts, 2-hop and '
                    'both sub-questions asked separately', prompts={'user': TEMPLATE},
                    params={'temperature': 0, 'max_tokens': 32}, repeats=REPEATS,
                    options={'exemplar_seed': EXEMPLAR_SEED, 'order_seed': ORDER_SEED}, deviations=DEVIATIONS)
        # Robustness variants: robust_parse rescores saved direct_2shot outputs (identical requests);
        # robust is the variant for new panel runs.
        rparse = v.override('robust_parse', description='direct_2shot requests, parse=robust (paper scorer first, '
                            '_robust short-answer fallback; unparsed rows missing; tolerant reliability)',
                            options={'parse': 'robust'},
                            deviations=('scoring: paper short-answer match first; when its window is None, a short '
                                        'answer line from the _robust-normalized text (marker / bold / first line / first sentence / '
                                        'last line); empty or truncated outputs are unparsed and their unit is '
                                        'dropped from the headline table; reliability = <=10% unparsed per kind',))
        robust = rparse.override('robust', description='robust_parse + max_tokens floor 1024 and min_cap_reasoning '
                                 '16384 so the answer is not cut off on modern models',
                                 params=_robust.cap_floor(v.params),
                                 options={'min_cap_reasoning': ROBUST_MIN_CAP_REASONING, 'truncated': 'score'},
                                 deviations=('max_tokens 32 -> 1024 (cap floor) and reasoning-model cap 4096 -> 16384: '
                                             'verbose models were truncated before the answer',
                                             "truncated='score': at these caps a cut-off output is a failed answer, "
                                             'scored by the paper rule on its text (empty or no short answer -> strict '
                                             'EM, i.e. wrong), not unparsed'))
        return {'direct_2shot': v, 'robust_parse': rparse, 'robust': robust}

    def items(self, variant, limit=None):
        rows = build_items()
        return rows[:limit] if limit else rows

    def program(self, variant):
        extra = ({'min_cap_reasoning': variant.options['min_cap_reasoning']}
                 if variant.options.get('min_cap_reasoning') else None)
        return SingleTurn(lambda item: [{'role': 'user', 'content': render(item)}], variant.params, case_extra=extra)

    def score(self, item, output, variant):
        text = output.get('content') or ''
        return score_answer(text, item['answers'], item['numeric'], output.get('finish_reason'),
                            parse=variant.options.get('parse', 'paper'),
                            cut_missing=_robust.truncation_missing(variant))

    def aggregate(self, rows, variant):
        robust = variant.options.get('parse', 'paper') == 'robust'
        head_rows = rows
        if robust:   # unparsed rows (empty / cut off / no short answer) are missing, not wrong
            head_rows = [{**r, 'score': None} if r.get('score') is not None and r['score'].get('unparsed') else r
                         for r in rows]
        units, missing = _units_table(head_rows, 'match')
        head = _table(units, 'match')
        strict = _table(_units_table(rows, 'em')[0], 'em', boot=False)
        cover = _table(_units_table(rows, 'cover_em')[0], 'cover_em', boot=False)
        kinds = _kind_diagnostics(rows)
        cats = {}
        for c in categories():
            sub = [u for u in units.values() if u['category'] == c]
            if sub:
                cats[c] = {'n_units': len(sub), 'acc_2hop': sum(u['hop2'] for u in sub) / len(sub),
                           'both_subs_right': sum(u['sub1'] and u['sub2'] for u in sub) / len(sub),
                           'compositionality_gap': _clean(_gap(sub))}
        reasons = []
        h, s1, s2 = kinds['hop2'], kinds['sub1'], kinds['sub2']
        if h['n_scored'] and s1['n_scored'] and s2['n_scored']:
            excess = h['gold_in_output_not_extracted'] - (s1['gold_in_output_not_extracted']
                                                          + s2['gold_in_output_not_extracted']) / 2
            if excess > VERBOSE_THRESHOLD:
                reasons.append(f'2-hop outputs contain the gold but not as an extractable short answer {excess:.1%} '
                               'more often than sub-question outputs')
            if h['truncated_rate'] > TRUNCATION_THRESHOLD:
                reasons.append(f'2-hop truncation (finish_reason=length) {h["truncated_rate"]:.1%} > '
                               f'{TRUNCATION_THRESHOLD:.0%}')
        if robust and h['n_scored']:
            reasons = [x for x in reasons if not x.startswith('2-hop truncation')]   # covered by the unparsed rule
            for kind, d in kinds.items():
                if not _robust.reliable(d['n_rows'], d['unparsed'], d['missing']):
                    reasons.append(f'{kind}: {d["unparsed"]} unparsed + {d["missing"]} missing of {d["n_rows"]} rows '
                                   f'> {_robust.ROBUST_TOLERANCE:.0%}')
        reliable = (not reasons) if h['n_scored'] else None
        out = {
            **head, 'strict_em': strict, 'cover_em': cover, 'by_kind': kinds, 'by_category': cats,
            'n_rows': len(rows), 'rows_without_output': sum(r.get('output') is None for r in rows),
            'incomplete_units': missing,
            'headline_reliable': reliable, 'headline_caveats': reasons,
        }
        if robust:
            out['parse'] = {k: sum(d.get(k, 0) for d in kinds.values())
                            for k in ('parsed_paper', 'parsed_robust', 'unparsed', 'truncated_scored')}
            n_all = len(rows)
            out['parsed_paper'], out['parsed_robust'] = out['parse']['parsed_paper'], out['parse']['parsed_robust']
            out['unparsed'] = out['parse']['unparsed']
            out['truncated_scored'] = out['parse']['truncated_scored']
            out['validity_rate'] = (out['parsed_paper'] + out['parsed_robust']) / n_all if n_all else None
        out['shelflife'] = {
            'metric': 'compositionality_gap', 'value': head.get('compositionality_gap'),
            'ci': head.get('gap_ci95_bootstrap'), 'n': head.get('n_both_subs_right', 0), 'direction': +1,
            'null_value': 0.0, 'paper_value': PAPER_GAP_DAVINCI002,
            'paper_value_design_mix': PAPER_GAP_DESIGN_MIX,
            'definition': 'Among CC entries whose two sub-questions are both answered correctly (direct 2-shot, '
                          'short-answer match), the fraction whose composed 2-hop question is answered wrongly, '
                          'post-stratified to the released category sizes (comparable to the question-weighted Table 6 '
                          'value); CI by person-cluster bootstrap.',
            'secondary': {'gap_unweighted': head.get('compositionality_gap_unweighted'),
                          'gap_strict_em': strict.get('compositionality_gap'),
                          'gap_cover_em': cover.get('compositionality_gap')},
        }
        return out

    # ------------------------------------------------------------ targets
    _T6 = 'arXiv 2210.03350v3 Appendix Table 6 (tables/accuracy-generated.tex), ALL row, davinci-002 direct 2-shot'
    targets = (
        PaperTarget('released_questions', 8693, 0, 'self-ask datasets/README.md ("8693 2-hop questions")', 'n/a',
                    'direct_2shot', 'released_data', compute=lambda exp: len(load_cc()),
                    note='anchors data loading; the evaluated set excludes the exemplar rows (8,659 units)'),
        PaperTarget('evaluated_units_8_6k', 8600, 100, 'arXiv v3 Sec 1 ("8.6k 2-hop questions")', 'n/a',
                    'direct_2shot', 'released_data', compute=lambda exp: len(build_items()) / 3,
                    note='8,693 released minus 34 exemplar rows = 8,659; "8.6k" is a truncation, tolerance covers it'),
        PaperTarget('table6_gap_arithmetic', 0.444, 0.001, _T6 + ': 33.4 / (41.8 + 33.4)', 'davinci-002',
                    'direct_2shot', 'released_data', compute=lambda exp: _table6_gap(),
                    note='paper arithmetic from the LaTeX table; the headline paper_value'),
        PaperTarget('table6_categories_reweighted_to_released_sizes', 0.444, 0.002,
                    _T6.replace('ALL row', '17 category rows') + ', weighted by released category sizes',
                    'davinci-002', 'direct_2shot', 'released_data', compute=lambda exp: table6_mix_gap(None),
                    note='the ALL row is the size-weighted pool of the category rows (0.4446): this is the estimand '
                         'the post-stratified headline targets'),
        PaperTarget('table6_gap_at_minimal_design_mix', 0.397, 0.001,
                    _T6.replace('ALL row', '17 category rows') + ', weighted by the MINIMAL_LIMIT category mix',
                    'davinci-002', 'direct_2shot', 'released_data',
                    compute=lambda exp: table6_mix_gap(MINIMAL_LIMIT),
                    note='paper_value_design_mix: what an unweighted pool of the 76-unit design would give for '
                         'davinci-002 (audit M1)'),
        PaperTarget('table6_2hop_accuracy_arithmetic', 0.454, 0.0005, _T6 + ': 41.8 + 3.6 = 45.4 (Sec 2 text)',
                    'davinci-002', 'direct_2shot', 'released_data', compute=lambda exp: _table6_2hop(),
                    note='the Table 6 ALL row reproduces the Sec 2 headline 45.4%'),
        PaperTarget('gap_davinci002', 0.444, 0.06, _T6, 'davinci-002', 'direct_2shot', 'unverifiable',
                    note='davinci-002 completion endpoint retired, no outputs released. Tolerance: ~2 bootstrap SE at '
                         'the REPRO_LIMIT of 204 units'),
        PaperTarget('hop2_accuracy_davinci002', 0.454, 0.05, 'arXiv v3 Sec 2 text', 'davinci-002', 'direct_2shot',
                    'unverifiable', note='retired model'),
        PaperTarget('gap_chatgpt', 0.429, 0.08, 'arXiv v3 Sec 1 footnote 2 ("42.9% for ChatGPT")', 'gpt-3.5-turbo',
                    'direct_2shot', 'live',
                    note='the footnote does not name the snapshot (2023); openai/gpt-3.5-turbo-0613 is the closest '
                         'reachable. The footnote protocol is not described; assumed the Sec 2 protocol'),
        PaperTarget('gap_gpt4', 0.230, 0.08, 'arXiv v3 Sec 1 footnote 2 ("23.0% for GPT-4")', 'gpt-4',
                    'direct_2shot', 'live', note='openai-direct/gpt-4-0613; contamination caveat in the footnote'),
    )


PAPER_GAP_DAVINCI002 = 0.444
PAPER_GAP_DESIGN_MIX = 0.397          # = table6_mix_gap(MINIMAL_LIMIT), asserted by a released_data target

TABLE6_ROWS = {
    'Birthplace/Latitude': 'birthplace_rounded_lat', 'Birthplace/Longitude': 'birthplace_rounded_lng',
    'Birthplace/3166-1 Code': 'birthplace_ccn3', 'Birthplace/Estonian Name': 'birthplace_est_common_name',
    'Birthplace/Urdu Name': 'birthplace_urd_common_name', 'Birthplace/Japanese Name': 'birthplace_jpn_common_name',
    'Birthplace/Domain Name': 'birthplace_tld', 'Birthplace/Spanish Name': 'birthplace_spa_common_name',
    'Birthplace/Russian Name': 'birthplace_rus_common_name', 'Birthplace/Currency': 'birthplace_currency',
    'Birthplace/Currency Abbrv.': 'birthplace_currency_short', 'Birthplace/Currency Symbol': 'birthplace_currency_symbol',
    'Birthplace/Calling Code': 'birthplace_callingcode', 'Birthplace/Capital': 'birthplace_capital',
    "Birth Year/Master's Champion": 'birthyear_masterchamp', 'Birth Year/Lit. Nobel Winner': 'birthyear_nobelLiterature',
    'Birth Date/US President': 'birthdate_uspresident', 'ALL': 'ALL',
}


@functools.lru_cache(maxsize=1)
def table6_rows():
    """{category (or 'ALL'): (both right & 2-hop right, both right & 2-hop wrong, one wrong & right, one wrong & wrong)}"""
    text = (PAPER_SRC / 'src/tables/accuracy-generated.tex').read_text()
    out = {}
    for line in text.splitlines():
        cells = [c.strip() for c in line.split('&')]
        name = re.sub(r'\\textbf\{(.*?)\}', r'\1', cells[0])
        if len(cells) == 5 and name in TABLE6_ROWS:
            out[TABLE6_ROWS[name]] = tuple(float(re.match(r'\s*([\d.]+)\\%', c).group(1)) / 100 for c in cells[1:5])
    if len(out) != 18:
        raise ValueError(f'parsed {len(out)} Table 6 rows')
    return out


def table6_mix_gap(limit):
    """davinci-002 Table 6 category rows pooled with weights = category counts in the first limit/3 units
    (limit None: released category sizes)."""
    rows = table6_rows()
    if limit is None:
        weights = released_category_sizes()
    else:
        weights = defaultdict(int)
        for x in build_items()[:limit]:
            if x['kind'] == 'hop2':
                weights[x['category']] += 1
    num = sum(w * rows[c][1] for c, w in weights.items())
    den = sum(w * (rows[c][0] + rows[c][1]) for c, w in weights.items())
    return num / den


def _table6_gap():
    rr, rw, _, _ = table6_rows()['ALL']
    return rw / (rr + rw)


def _table6_2hop():
    rr, _, wr, _ = table6_rows()['ALL']
    return rr + wr
