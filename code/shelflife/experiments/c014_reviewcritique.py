"""C014 — LLM paper reviews do not discriminate accepted from rejected papers (Du et al. 2024, ReviewCritique).

Claim: GPT-4/Claude-Opus/Gemini-1.5 reviews of 20 ReviewCritique papers average a recommendation of 7.43 for accepted
and 7.47 for rejected papers, while the human reviews of the same papers average 6.41 vs 4.81 (Sec. 4.1
"Recommendation Score"); LLM reviews also contain more expert-flagged deficient segments (13.97% vs 6.27%, Table 1).

Sources
  ReviewCritique repo (jiangshdd/ReviewCritique @ 629932bc) releases data only: the 100 papers with 380 human
  reviews (ReviewCritique.jsonl) and the 20-paper subset with one archived review per LLM (ReviewCritique_LLM.jsonl).
  No generation or analysis code. The prompt is the uninstantiated Table 14 template (paper appendix F); the
  reviewer guidelines, rubric explanations, the two "randomly chosen" example reviews, roles and sampling
  parameters were never released.

Variants
  table14_reconstructed_v1_cap8192  default. Table 14 template (PDF transcription) filled by the initial-reproduction
                     reconstruction (ICLR 2024 reviewer-guide step-by-step text, ICLR-2023-form rubric anchors,
                     two held-out human reviews drawn with seed 20260930), one user message, max_tokens 8192, no
                     temperature. Byte-identical to every cap8192 prior run (initial main + pilots, all panel runs).
  table14_reconstructed_v1          the same messages at max_tokens 4096 (initial pilot, direct GPT-6 pilot, panel
                     GPT-4o run whose API caps output at 4096).
  Both are "faithful" only in the sense that they are the closest runnable instantiation of Table 14: every filled
  placeholder is a declared reconstruction (see ``deviations``).

Scoring
  There is no original scorer: the archived LLM reviews carry a structured ``Recommendation`` field. For generated
  text we keep the four explicit-field parsers the earlier implementations wrote, loaded verbatim from their files:
  ``strict`` (score.py, prespecified), ``v1`` (score_diagnostic.py, Markdown bullet rows, prespecified before the
  initial main run), ``v2_1`` (post hoc, bold rows under a Recommendation heading) and ``v3`` (panel post hoc,
  heading/value/table/prose-with-explicit-number layouts). None infers a score from sentiment. ``v3`` is the headline
  parser (it is the one used for the panel results); all four are reported by ``aggregate``.
  Archived reviews are scored with initial-reproduction archive_anchor.recommendation (structured field, including
  the one Gemini record whose value sits in the key ``Recommendation 8``).

Metric (per model): paper-weighted mean recommendation of accepted vs rejected papers over papers whose three
repeats all parse, accepted-minus-rejected difference, decision AUC (ties 0.5), stratified paper-cluster bootstrap
(5,000 draws, seed 20260930, percentile indices 125/4875, accepted and rejected papers resampled separately, as in
the earlier analyze.py), all-intended missingness bounds (missing papers at 1 or 10), and available-repeat means.

Robustness: parse='robust' runs v3 first and uses robust_recommendation only when v3
returns nothing and flags no conflict; robust vs paper on reviews v3 DID parse: none (tested on 2,573 saved
outputs). The prose fallback ignores prose about another field ('rate soundness 4 and novelty 8') and is
unparsed when its window names several different 1-10 numbers. Variant robust adds min_cap_reasoning 16384
(reasoning models only; max_tokens stays 8192).
Follow-up: robust-only
extraction also recognizes parenthetical rubric scales/"choose one" in Recommendation
headings and a first numeric value followed by numbered justification. This repairs
rubric-echo false positives such as "1. Strong empirical contribution"; real rubric
echoes, conflicting chosen values, quoted/fenced examples and absent numbers remain
unrecovered by the new rule. Previously parsed paper/robust values remain unchanged.

"""
import collections
import re
import functools
import hashlib
import math
import random
import statistics

from ..paths import INITIAL, HELPERS, source_repo
from ..program import SingleTurn
from ..prompts import PromptTemplate, Source
from ..registry import Experiment, PaperTarget, PriorRun, Variant, register
from ..replay import legacy_records, read_jsonl
from ..sourcecode import load_functions
from . import _robust

REPO = source_repo('C014', 'ReviewCritique')
COMMIT = '629932bcfa31879b1df1e011e3cd7188aa3b9b46'
REPRO = INITIAL / 'reproductions/C014'
PANEL_TASK = HELPERS / 'C014'
REPRO_REL = 'experiments/initial-reproduction/reproductions/C014'
SEED = 20260930
PROTOCOL = 'table14_reconstructed_v1'
PARSERS = ('strict', 'v1', 'v2_1', 'v3')
REPEATS = 3
PAPER_URL = 'https://aclanthology.org/2024.emnlp-main.292.pdf'

_REPO_SRC = Source(repo=str(REPO), path='data/ReviewCritique_LLM.jsonl', commit=COMMIT,
                   url='https://github.com/jiangshdd/ReviewCritique')

# Table 14 (paper appendix F), PDF transcription saved by initial-reproduction. The PDF typesets "paper’s" with a
# right single quote; the transcription uses ASCII apostrophes and reconstructs line breaks.
TABLE14 = PromptTemplate.from_source(
    'user', Source(repo=REPRO_REL, path='table14_template.txt', url=PAPER_URL, locator='Table 14 (appendix F, p. 19)',
                   note='PDF transcription; ASCII apostrophes; line breaks reconstructed'),
    kind='lines', lines=(1, 24))


def _fill(name, var):
    return PromptTemplate.from_source(name, Source(repo=REPRO_REL, path='prepare.py', locator=var,
                                                   note='initial-reproduction reconstruction (not released by the authors)'),
                                      style='literal', kind='py_var', var=var)


FILLS = {
    'fill:review_guidelines': PromptTemplate.from_source(
        'fill:review_guidelines', Source(repo=REPRO_REL, path='reviewer_guidelines_reconstructed.txt',
                                         url='https://iclr.cc/Conferences/2024/ReviewerGuide',
                                         note="ICLR 2024 reviewer guide 'step-by-step' subsection (reconstruction)"),
        style='literal', kind='lines', lines=(1, 24)),
    'fill:correctness': _fill('fill:correctness', 'CORRECTNESS'),
    'fill:novelty': _fill('fill:novelty', 'NOVELTY'),
    'fill:recommendation': _fill('fill:recommendation', 'RECOMMENDATION'),
    'fill:confidence': _fill('fill:confidence', 'CONFIDENCE'),
}


def load_data(name):
    return read_jsonl(REPO / 'data' / name)


@functools.lru_cache(maxsize=None)
def archive_recommendation_fn():
    """initial-reproduction archive_anchor.recommendation (structured-field reader for released reviews)."""
    return load_functions(REPRO / 'archive_anchor.py', ['recommendation'])['recommendation']


def archive_recommendation(fields):
    return archive_recommendation_fn()(fields)


@functools.lru_cache(maxsize=None)
def parsers():
    """The earlier implementations' parsers, loaded verbatim (sibling imports supplied explicitly)."""
    strict = load_functions(REPRO / 'score.py', ['score'], constants=('ALLOWED_ANCHORS',))
    v1 = load_functions(REPRO / 'score_diagnostic.py', ['score'], extra_globals={'primary_score': strict['score']})
    v21 = load_functions(REPRO / 'score_diagnostic_v2_1_posthoc.py', ['score'], extra_globals={'v1_score': v1['score']})
    v3 = load_functions(PANEL_TASK / 'score_diagnostic_v3_posthoc.py', ['_clean', '_prose_values', '_values', 'score'],
                        constants=('FIELD', 'TABLE_FIELD', 'PROSE', 'HEADINGISH', 'RUBRIC_ROW', 'SUBLABEL', 'LEAD',
                                   'PAREN', 'SLASH'),
                        extra_globals={'v21_score': v21['score'], 'ALLOWED_ANCHORS': strict['ALLOWED_ANCHORS']})
    return {'strict': strict['score'], 'v1': v1['score'], 'v2_1': v21['score'], 'v3': v3['score']}


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


# ------------------------------------------------------------- robust fallback
# Used only when the headline v3 parser returns no value and did not flag a conflict/ambiguity. Like v3 it reads only
# an explicit number written for the Recommendation field (never a sentiment such as "Accept (borderline)"), but it
# also accepts layouts v3 does not: a number in prose inside the Recommendation section ("I recommend **6**",
# "Therefore, I rate the recommendation as: **8**", "The recommendation is **6**") and an "Overall Score/Rating" field
# ("### Overall Score: **6/10 - marginally above ...**"), the ICLR overall rating that the Recommendation field is.
ROBUST_MIN_CAP_REASONING = 16384
_R_FIELD = re.compile(r'^\s*(?:[>*+-]\s*)?(?:#{1,6}\s*)?(?:\d+[.)]\s*)?(?:Final\s+|Overall\s+)?'
                      r'(?:Recommendation(?:\s+(?:Score|Rating|Level|Decision|for\s+the\s+paper))?|'
                      r'(?<=Overall\s)(?:Score|Rating))'
                      r'(?:\s*[:：–—-]\s*(.*)|\s*)$', re.I)
# A section ends at the next heading, horizontal rule, "Label:" line (Confidence:, Key Reasons:, ...) or bare
# title-case label line ("Confidence" once bold markers are stripped). A sub-label row naming the value
# ("- Choice: 8 (Accept, good paper)", "Rating: 6") is read, not treated as the end of the section.
_R_SUB = re.compile(r'^\s*(?:[>*+-]\s*)?(?:Choice|Rating|Score|Option(?:\s+chosen)?|Value|Decision)\s*[:：]\s*(.*)$', re.I)
_R_STOP = re.compile(r'^\s*(?:#{1,6}\s|[-=_*]{3,}\s*$|(?:[>*+-]\s*)?[A-Z][A-Za-z /&()-]{1,40}[:：]|'
                     r'(?:[>*+-]\s*)?[A-Z][A-Za-z /&-]{2,40}\s*$)')
# Same-line value: a leading 1-10 number ("8", "8 - Accept", "(8)", "[8]", "6/10 - marginally above").
_R_LEAD = re.compile(r'^\s*[(\[]?(\d{1,2})(?:\s*/\s*10)?[)\]]?(?![\d.]\d|\d|\s*%)')
# Prose value: a number right after recommend/rate/score wording, with no digit or sentence end in between
# ("I recommend **6**", "rate the recommendation as: **8**", "The recommendation is **6**").
_R_PROSE = re.compile(r'\b(?:recommend(?:ation)?|rat(?:e|ing)|score)\b[^.\d\n]{0,40}?(\d{1,2})(?:\s*/\s*10)?(?![\d%]|\.\d)',
                      re.I)
_R_STATUSES_SKIP = ('v3_conflicts_with_v2_1_value', 'v3_conflicting_or_ambiguous')
# Prose must name the recommendation itself: a gap naming another review field ("rate soundness 4 and novelty 8")
# is not read, and a prose window that mentions more than one distinct 1-10 number (scale phrases such as "1 to 10",
# "out of 10", "/10" removed) is ambiguous.
_R_OTHER_FIELD = re.compile(r'\b(?:soundness|novelty|originality|clarity|presentation|contribution|significance|'
                            r'quality|confidence|technical|reproducibility|impact|writing)\b', re.I)
_R_SCALE = re.compile(r'\b(?:1|one)\s*(?:to|-|–|—)\s*(?:10|ten)\b|(?:out\s+of|/)\s*10\b', re.I)


def _prose_values(window):
    """Recommendation values written in prose in ``window``; empty when the prose is about another field or names
    several different 1-10 numbers."""
    vals = set()
    for m in _R_PROSE.finditer(window):
        gap = window[m.start():m.start(1)]
        if _R_OTHER_FIELD.search(gap):
            continue
        vals.add(int(m[1]))
    if not vals:
        return vals
    others = {int(x) for x in re.findall(r'(?<![\d.])(\d{1,2})(?![\d%]|\.\d)', _R_SCALE.sub(' ', window))
              if 1 <= int(x) <= 10}
    return vals if len(others | vals) == 1 else {-1}   # -1: ambiguous, never a valid 1..10 value


def robust_recommendation(text):
    """Explicit 1-10 Recommendation value from a noisier layout, or None. For every Recommendation (or Overall
    Score/Rating) field: a leading number of its same-line value, else a number right after recommend/rate/score
    wording in the value line or the first 300 characters of its section (up to the next heading or "Label:" line).
    All values found must agree (one distinct integer in 1..10), otherwise None."""
    if not text:
        return None
    t = _robust._THINK.sub(' ', text)
    t = _robust._OPEN_THINK.sub(' ', t)
    lines = [x.replace('**', '').replace('__', '').replace('`', '').rstrip() for x in t.splitlines()]
    found = set()
    for i, line in enumerate(lines):
        m = _R_FIELD.match(line)
        if not m:
            continue
        rest = (m[1] or '').strip()
        lead = _R_LEAD.match(rest)
        if lead:
            found.add(int(lead[1]))
            continue
        section = [rest]
        for nxt in lines[i + 1:]:
            sub = _R_SUB.match(nxt)
            if sub:
                lead = _R_LEAD.match(sub[1])
                if lead:
                    found.add(int(lead[1]))
                continue
            if _R_FIELD.match(nxt) or _R_STOP.match(nxt):
                break
            section.append(nxt)
        window = '\n'.join(x.strip() for x in section if x.strip())[:300]
        found.update(_prose_values(window))
    return found.pop() if len(found) == 1 and 1 <= next(iter(found)) <= 10 else None


_R_FIELD_METADATA = re.compile(
    r'^(?P<prefix>\s*(?:[>*+-]\s*)?(?:#{1,6}\s*)?(?:\d+[.)]\s*)?'
    r'(?:Final\s+|Overall\s+|Initial\s+)?Recommendation(?:\s+(?:Score|Rating))?)'
    r'\s*\(\s*(?:1\s*[-–—]\s*10|1\s*[/,]\s*3\s*[/,]\s*5\s*[/,]\s*6\s*[/,]\s*8\s*[/,]\s*10|choose\s+one)\s*\)'
    r'(?P<rest>\s*(?:[:：–—-].*)?)$', re.I)

def _unquoted_recovery_text(text):
    """Keep quote/code regions from supplying a new Recommendation field's value.

    A section boundary, rather than deleting the region, also prevents a heading
    immediately before an example from acquiring text after that example.
    This applies only to the new recovery; previously parsed scores stand.
    """
    lines, fence = [], None
    boundary = '## Excluded quoted or fenced example'
    for line in text.splitlines():
        stripped = line.lstrip()
        marker = re.match(r'(`{3,}|~{3,})', stripped)
        if marker:
            token = marker[1]
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
            lines.append(boundary)
        elif fence is not None or stripped.startswith('>'):
            lines.append(boundary)
        else:
            lines.append(line)
    return '\n'.join(lines)


def _normalize_field_metadata(text):
    lines = [];fenced=False
    for line in text.splitlines():
        if line.lstrip().startswith(('```','~~~')):
            fenced=not fenced;lines.append(line);continue
        if fenced or line.lstrip().startswith('>'):
            lines.append(line);continue
        # Formatting does not identify a score; the underlying parser already
        # performs this same markdown normalization.
        clean = line.replace('**','').replace('__','').replace('`','')
        match=_R_FIELD_METADATA.fullmatch(clean)
        lines.append(match['prefix'] + match['rest'] if match else line)
    return '\n'.join(lines)

def _first_numeric_field(text):
    """First numeric value of each explicitly headed Recommendation section.

    Guard real multiple-choice/rubric echoes; `1. Strong empirical contribution`
    is a reason, while `1: strong reject` followed by `8: accept` is a rubric.
    Quoted/fenced examples are outside this new recovery rule.
    """
    lines=_unquoted_recovery_text(text).splitlines();found=set();fenced=False
    rubric=re.compile(r'^\s*(?:[-*+]\s*)?\d+\s*[:.)\-–—]\s*(?:strong(?:ly)?\s+(?:reject|accept)|reject\b|accept\b|marginal|top\b)',re.I)
    for i,raw in enumerate(lines):
        if raw.lstrip().startswith(('```','~~~')):fenced=not fenced;continue
        if fenced or raw.lstrip().startswith('>'):continue
        line=raw.replace('**','').replace('__','').replace('`','')
        field=_R_FIELD.match(line)
        if not field:continue
        rest=(field[1] or '').strip();section=[]
        for nxt in lines[i+1:]:
            clean=nxt.replace('**','').replace('__','').replace('`','')
            if re.match(r'^\s*#{1,6}\s',clean) or _R_FIELD.match(clean) or _R_STOP.match(clean):break
            if clean.strip():section.append(clean)
        value=rest or (section[0] if section else '')
        # A chosen value starts the value line. Other digits near that position
        # make it ambiguous; a /10 denominator is scale metadata, not a value.
        lead=_R_LEAD.match(value)
        if not lead:continue
        vals={int(x) for x in re.findall(r'(?<![\d.])(\d+)(?![\d.])', re.sub(r'/\s*10\b','',value[:25]))}
        if len(vals)!=1:return None
        if sum(bool(rubric.match(x)) for x in [rest]+section)>=2:return None
        if len(vals)==1:
            scalar=next(iter(vals))
            if scalar!=int(lead[1]) or not 1<=scalar<=10:return None
            found.add(scalar)
    return found.pop() if len(found)==1 else None

def _field_metadata_recommendation(text, item):
    """Recover explicit numeric Recommendation fields; keep genuine conflicts unparsed."""
    cleaned = _normalize_field_metadata(_unquoted_recovery_text(text))
    parsed = parsers()['v3'](cleaned, item)
    if parsed['parse_status'] in _R_STATUSES_SKIP:
        return None
    value = parsed['parsed']
    if value is None:
        value = robust_recommendation(cleaned) or _first_numeric_field(cleaned)
    if value is None:
        return None
    # A newly recovered heading value cannot overrule a separate, explicitly
    # stated final recommendation in prose. Quoted examples were removed above.
    plain = cleaned.replace('**', '').replace('__', '').replace('`', '')
    asserted = re.compile(
        r'\b(?:(?:my|the)\s+)?(?:final\s+|overall\s+)?recommendation\s*(?:is|would\s+be|:)\s*'
        r'[(\[]?(\d{1,2})(?![\d.]\d)|'
        r'\bI\s+(?:would\s+)?recommend\s+(?:a\s+(?:score|rating)\s+of\s+)?[(\[]?(\d{1,2})(?![\d.]\d)',
        re.I)
    if any(int(a or b) != value for a, b in asserted.findall(plain)):
        return None
    if re.search(r"\b(?:do\s+not|don't|cannot|can't)\s+recommend\s+[(\[]?\d", plain, re.I):
        return None
    return value


def review_text(review):
    """A released human review as plain text: segments in source order, then 'key: value' score rows."""
    text = '\n'.join(s['segment_text'] for s in review['review'])
    return text + '\n' + '\n'.join(f'{k}: {v.strip()}' for k, v in review['score'].items())


def select_examples(human, target_titles):
    """Two held-out human reviews (one accepted, one rejected paper), seed 20260930, as in prepare.py."""
    rng = random.Random(SEED)
    out = []
    for decision in ('accepted', 'rejected'):
        eligible = [(i, r) for i, r in enumerate(human) if r['decision'] == decision and r['title'].strip() not in target_titles]
        i, row = rng.choice(eligible)
        keys = [k for k, v in row.items() if isinstance(v, dict) and archive_recommendation(v['score']) is not None]
        key = rng.choice(keys)
        out.append({'source_index': i, 'title': row['title'], 'decision': decision, 'review_key': key,
                    'text': review_text(row[key])})
    return out


def auc(accepted, rejected):
    return statistics.mean((a > b) + .5 * (a == b) for a in accepted for b in rejected)


def stratified_bootstrap(accepted, rejected, n_boot=5000, seed=SEED):
    """The earlier analyses' stratified paper-cluster bootstrap: returns (difference CI, AUC CI or None)."""
    rng = random.Random(seed)
    sims, auc_sims = [], []
    for _ in range(n_boot):
        aa = rng.choices(accepted, k=len(accepted))
        bb = rng.choices(rejected, k=len(rejected))
        sims.append(statistics.mean(aa) - statistics.mean(bb))
        auc_sims.append(auc(aa, bb))
    sims.sort()
    auc_sims.sort()
    return [sims[125], sims[4875]], ([auc_sims[125], auc_sims[4875]] if len(set(auc_sims)) > 1 else None)


# ---------------------------------------------------------------------------- earlier pipeline runs
# experiments/run_minimal_setups/runs/earlier/runs/<claim>-<variant>-L<N>[-<tag>]/<model>/: one shelflife.runner directory per model.
# Registered: every directory that sent at least one request (complete or partial); tests/test_prior_runs.py
# replays each one against this config. Excluded: aborted, smoke-test and superseded directories, and directories
# that sent nothing. Listed explicitly (written 2026-10-05 from the run directories; nothing is globbed at import).

_R2 = 'experiments/run_minimal_setups/runs/earlier/runs'
_NOTHING_SENT = ('no request reached a provider: every call is not_submitted (circuit breaker open, or stale ledger '
                 'rows after a package-hash change); the directory only records the attempts')
_ABORTED_WAVE1 = ('wave-1 run aborted 2026-10-04 (OpenRouter credit exhausted, local connect timeouts) and moved '
                  'aside; created before the relaunch core edits and superseded by the tagged reruns; the collector '
                  'skips _aborted*')


def _unsent(*runs):
    """excluded_runs entries for per-model directories that sent nothing: (run, space-separated models)."""
    return {f'{_R2}/{run}/{m}': _NOTHING_SENT for run, models in runs for m in models.split()}


# experiments/run_minimal_setups shelflife.runner runs (registered 2026-10-10; replayed by tests/test_prior_runs.py).


@register
class C014ReviewCritique(Experiment):
    claim_id = 'C014'
    title = 'LLM reviews do not discriminate accepted vs rejected papers (ReviewCritique)'
    claim = ('LLM-generated reviews give uniformly high recommendation scores that do not separate accepted from '
             'rejected papers (7.43 vs 7.47; humans 6.41 vs 4.81) and contain more expert-flagged deficient segments '
             '(13.97% vs 6.27%).')
    paper = {'title': 'LLMs Assist NLP Researchers: Critique Paper (Meta-)Reviewing', 'authors': 'Du et al.',
             'year': 2024, 'venue': 'EMNLP 2024', 'url': 'https://arxiv.org/abs/2406.16253'}
    source = _REPO_SRC
    default_variant = 'table14_reconstructed_v1_cap8192'

    def build_variants(self):
        prompts = {'user': TABLE14, **FILLS}
        deviations = (
            'Table 14 is an uninstantiated template: review_guidelines are the ICLR 2024 reviewer-guide step-by-step '
            'subsection, rubric anchors reconstruct the older ICLR (2023) form, confidence anchors are paraphrases',
            'the authors\' randomly chosen example reviews are unreleased; two held-out human reviews (one accepted, one '
            'rejected paper, seed 20260930) are fixed across papers, models and repeats',
            'both fixed example reviews carry "Recommendation: 6" (ICLR review#4 of the accepted and review#3 of the '
            'rejected example), a shared score anchor the paper\'s random examples would not systematically have',
            'paper attachment unspecified in Table 14 (it ends "write a review for the paper below:"): the paper is '
            'appended as prompt + "\\n\\n" + title + "\\n" + body_text (separator is a reproduction choice)',
            'roles unspecified: one user message, no system message; template transcribed from the PDF (ASCII apostrophes)',
            'sampling parameters and output cap unspecified: temperature omitted (provider default), max_tokens is a '
            'reproduction choice',
            'paper: three LLMs x one archived review per paper; here three independent repeats per model per paper',
            'original models (gpt-4-turbo, gemini-1.5-flash-latest, claude-3-opus-20240229) not re-run',
        )
        base = Variant(PROTOCOL + '_cap8192',
                       'Reconstructed Table 14 prompt, full paper text, one user message, max_tokens 8192, 3 repeats',
                       prompts=prompts, params={'max_tokens': 8192}, repeats=REPEATS,
                       options={'id_suffix': ':cap8192', 'parser': 'v3'}, deviations=deviations)
        cap4096 = Variant(PROTOCOL, 'Reconstructed Table 14 prompt at max_tokens 4096 (initial pilot cap)',
                          prompts=prompts, params={'max_tokens': 4096}, repeats=REPEATS,
                          options={'id_suffix': '', 'parser': 'v3'}, deviations=deviations)
        # Robustness variants. robust_parse: the panel generation variant's requests
        # (cap8192) rescored with parse='robust'; robust: the same plus a 16384-token cap for reasoning models.
        rparse = base.override(
            'robust_parse', description='cap8192 requests; parse=robust (v3 first, explicit-number fallback for '
            'prose/Overall-score layouts; tolerant reliability)', options={'parse': 'robust'},
            deviations=('scoring: v3 parser first; only when it returns no value (and flags no conflict) an explicit '
                        '1-10 number in the Recommendation section prose or an "Overall Score/Rating" field is read '
                        '(robust_recommendation); sentiment-only recommendations stay unparsed',
                        'reliability: headline_reliable = at most 10% of intended paper x repeat slots unparsed or '
                        'missing (_robust.reliable), instead of no flag'))
        robust = rparse.override(
            'robust', description='robust_parse + reasoning-model cap 16384 (panel policy min_cap_reasoning)',
            options={'min_cap_reasoning': ROBUST_MIN_CAP_REASONING},
            deviations=('reasoning models: output cap 8192 -> 16384 (min_cap_reasoning). On the panel, GLM-5.3-Flash '
                        '(M196) spent the whole 8192 cap on hidden reasoning (2 empty reviews) or was cut off before '
                        'the Recommendation field (1 review). Non-reasoning models keep max_tokens 8192, which no '
                        'panel model hit (cap_floor 1024 is below it).',))
        return {base.name: base, cap4096.name: cap4096, rparse.name: rparse, robust.name: robust}

    # ------------------------------------------------------------------ data
    @functools.cached_property
    def _prompt(self):
        rows, human = load_data('ReviewCritique_LLM.jsonl'), load_data('ReviewCritique.jsonl')
        examples = select_examples(human, {r['title'].strip() for r in rows})
        p = self.variant().prompts
        fills = {'review_guidelines': p['fill:review_guidelines'].render(),
                 'Explanation on different correctness scores': p['fill:correctness'].render(),
                 'Explanation on different Technical Novelty and Significance scores': p['fill:novelty'].render(),
                 'Explanation on different Empirical Novelty and Significance scores':
                     p['fill:novelty'].render() + '\nNA: Not applicable.',
                 'Explanation on different recommendation scores': p['fill:recommendation'].render(),
                 'Explanation on different confidence scores': p['fill:confidence'].render(),
                 'review_example_1': examples[0]['text'], 'review_example_2': examples[1]['text']}
        return p['user'].render(**fills), examples

    def items(self, variant, limit=None):
        rows, human = load_data('ReviewCritique_LLM.jsonl'), load_data('ReviewCritique.jsonl')
        by_title = {r['title'].strip(): r for r in human}
        assert collections.Counter(r['decision'] for r in rows) == {'accepted': 10, 'rejected': 10}
        prompt, examples = self._prompt
        out = []
        for i, row in enumerate(rows):
            if limit is not None and i >= limit:
                break
            paired = by_title[row['title'].strip()]
            assert paired['body_text'] == row['body_text'] and paired['decision'] == row['decision']
            out.append({
                'id': f'C014:p{i:02d}:{PROTOCOL}{variant.options["id_suffix"]}', 'paper_id': f'p{i:02d}',
                'decision': row['decision'], 'title': row['title'], 'body_text': row['body_text'], 'prompt': prompt,
                'body_sha256': _sha(row['body_text']),
                'example_reviews': [{k: v for k, v in e.items() if k != 'text'} for e in examples],
                'archive_llm_recommendations': {k: archive_recommendation(v['score']) for k, v in row.items() if isinstance(v, dict)},
                'archive_human_recommendations': [archive_recommendation(v['score']) for v in paired.values() if isinstance(v, dict)]})
        return out

    def program(self, variant):
        def build(item):
            return [{'role': 'user', 'content': item['prompt'] + '\n\n' + item['title'] + '\n' + item['body_text']}]
        cap = variant.options.get('min_cap_reasoning')
        return SingleTurn(build, variant.params, case_extra={'min_cap_reasoning': cap} if cap else None)

    # --------------------------------------------------------------- scoring
    def score(self, item, output, variant):
        text = (output or {}).get('content') or ''
        results = {name: fn(text, item) for name, fn in parsers().items()}
        out = {'recommendation': {k: r['parsed'] for k, r in results.items()},
               'parse_status': {k: r['parse_status'] for k, r in results.items()},
               'strict': results['strict']}
        if variant.options.get('parse') == 'robust':
            v3, status = results['v3']['parsed'], results['v3']['parse_status']
            if v3 is not None:
                value, path = v3, 'paper'
            else:
                # Preserve every existing robust score. A parenthetical rubric scale is heading
                # metadata, not a conflicting chosen score; re-run v3 after removing only known
                # heading metadata, retaining genuine conflicts and numeric-only extraction.
                value = None if status in _R_STATUSES_SKIP else robust_recommendation(text)
                if value is None:
                    value = _field_metadata_recommendation(text, item)
                path = 'robust' if value is not None else None
            out['recommendation']['robust'] = value
            out['parse_status']['robust'] = {'paper': 'parsed_paper', 'robust': 'parsed_robust'}.get(
                path, 'empty' if not text.strip() else 'unparsed')
        return out

    def aggregate(self, rows, variant, repeats=None, paper_ids=None):
        """rows: {'item','score','repeat'} for one model; score None for failed calls (counted missing).
        Every intended paper x repeat slot enters the denominators; ``paper_ids`` restricts the intended papers
        (two-paper pilots)."""
        repeats = variant.repeats if repeats is None else repeats
        all_items = [x for x in self.items(variant) if paper_ids is None or x['paper_id'] in paper_ids]
        slots = {}
        for r in rows:
            key = (r['item']['id'], r['repeat'])
            if key in slots:
                raise ValueError(f'duplicate slot {key}')
            slots[key] = r.get('score')
        robust = variant.options.get('parse') == 'robust'
        headline = 'robust' if robust else variant.options['parser']
        out = {'parser': headline, 'records': len(rows), 'intended_records': repeats * len(all_items),
               'by_parser': {}}
        for parser in PARSERS + (('robust',) if robust else ()):
            papers, parse_counts = [], collections.Counter()
            for item in all_items:
                vals = []
                for rep in range(repeats):
                    s = slots.get((item['id'], rep))
                    vals.append(None if s is None else s['recommendation'][parser])
                    parse_counts['missing_record' if s is None else s['parse_status'][parser]] += 1
                papers.append({'paper': item['paper_id'], 'decision': item['decision'], 'scores': vals,
                               'mean': statistics.mean(vals) if all(v is not None for v in vals) else None,
                               'available_mean': statistics.mean([v for v in vals if v is not None])
                               if any(v is not None for v in vals) else None})
            out['by_parser'][parser] = self._summarize(papers, parse_counts)
        out.update({k: out['by_parser'][headline][k]
                    for k in ('mean_accepted', 'mean_rejected', 'difference', 'difference_ci95', 'auc')})
        if robust:
            pc = out['by_parser']['robust']['parse_status']
            n = out['intended_records']
            missing = pc.get('missing_record', 0)
            unparsed = n - missing - pc.get('parsed_paper', 0) - pc.get('parsed_robust', 0)
            out.update({'parsed_paper': pc.get('parsed_paper', 0), 'parsed_robust': pc.get('parsed_robust', 0),
                        'unparsed': unparsed, 'missing': missing,
                        'validity_rate': (n - unparsed - missing) / n if n else None,
                        'headline_reliable': _robust.reliable(n, unparsed, missing)})
        return out

    @staticmethod
    def _summarize(papers, parse_counts):
        g = {d: [p['mean'] for p in papers if p['decision'] == d and p['mean'] is not None] for d in ('accepted', 'rejected')}
        ga = {d: [p['available_mean'] for p in papers if p['decision'] == d and p['available_mean'] is not None]
              for d in ('accepted', 'rejected')}
        a, b = g['accepted'], g['rejected']
        res = {'parse_status': dict(parse_counts),
               'explicit_scores': sum(v is not None for p in papers for v in p['scores']),
               'complete_papers': {d: len(v) for d, v in g.items()},
               'mean_accepted': statistics.mean(a) if a else None, 'mean_rejected': statistics.mean(b) if b else None,
               'difference': statistics.mean(a) - statistics.mean(b) if a and b else None,
               'auc': auc(a, b) if a and b else None, 'difference_ci95': None, 'auc_ci95': None}
        # A paper-cluster interval needs at least two papers per group (one paper per group gives a degenerate
        # zero-width interval, which the initial analyze.py emitted and its review then suppressed by hand).
        if len(a) >= 2 and len(b) >= 2:
            res['difference_ci95'], res['auc_ci95'] = stratified_bootstrap(a, b)
        bounds = {}
        for d in ('accepted', 'rejected'):
            n = sum(p['decision'] == d for p in papers)
            m = n - len(g[d])
            bounds[d] = [(sum(g[d]) + m) / n, (sum(g[d]) + 10 * m) / n]
        res['difference_missingness_bounds'] = [bounds['accepted'][0] - bounds['rejected'][1],
                                                bounds['accepted'][1] - bounds['rejected'][0]]
        if ga['accepted'] and ga['rejected']:
            res['available_repeat'] = {'mean_accepted': statistics.mean(ga['accepted']),
                                       'mean_rejected': statistics.mean(ga['rejected']),
                                       'difference': statistics.mean(ga['accepted']) - statistics.mean(ga['rejected']),
                                       'papers': {d: len(v) for d, v in ga.items()}}
        allv = [v for p in papers for v in p['scores'] if v is not None]
        res['review_weighted_mean'] = statistics.mean(allv) if allv else None
        res['papers'] = papers
        return res

    # -------------------------------------------------- archived (released) reviews
    def archive_summary(self, which='llm20'):
        """Review-weighted means and expert-deficiency counts of the released reviews (archive_anchor.summarize)."""
        llm = load_data('ReviewCritique_LLM.jsonl')
        human = load_data('ReviewCritique.jsonl')
        by_title = {r['title'].strip(): r for r in human}
        rows = {'llm20': llm, 'human100': human, 'human_matched20': [by_title[r['title'].strip()] for r in llm]}[which]
        groups = collections.defaultdict(list)
        c = collections.Counter()
        for row in rows:
            for key, value in row.items():
                if not isinstance(value, dict):
                    continue
                c['reviews'] += 1
                c['segments'] += len(value['review'])
                deficient = sum(s['reliability'] == 'No' for s in value['review'])
                c['deficient_segments'] += deficient
                c['reviews_with_deficiency'] += bool(deficient)
                score = archive_recommendation(value['score'])
                if score is not None:
                    groups[row['decision']].append(score)
        return {**c, 'segment_deficiency_pct': 100 * c['deficient_segments'] / c['segments'],
                'review_deficiency_pct': 100 * c['reviews_with_deficiency'] / c['reviews'],
                'mean': {d: statistics.mean(v) for d, v in groups.items()}, 'n_scored': {d: len(v) for d, v in groups.items()}}

    # ------------------------------------------------------------- prior runs
    def prior_rows(self, records, variant):
        """Legacy records -> aggregate rows. Error records keep their slot with score None (missing)."""
        items = {x['id']: x for x in self.items(variant)}
        rows = []
        for rec in records:
            item = items[rec['case_id']]
            usable = rec.get('status') in {'ok', 'truncated', 'empty'}
            rows.append({'item': item, 'repeat': rec['repeat'], 'status': rec.get('status'),
                         'score': self.score(item, {'content': rec.get('content') or ''}, variant) if usable else None})
        return rows

    def rescore_prior(self, run, variant_name=None, model=None, repeats=None, paper_ids=None):
        variant = self.variant(variant_name)
        recs = [r for r in legacy_records(run) if model is None or (r.get('record_id') or r.get('model')) == model]
        return self.aggregate(self.prior_rows(recs, variant), variant, repeats=repeats, paper_ids=paper_ids)

    def panel_slots(self):
        """Slot merge of first-pass analyze_panel.load(): reused initial main-run outputs (Sonnet 5.5 -> M206,
        GPT-6.1 -> M207, GLM-5.3), then new runs in priority order; a later run fills a slot only when the earlier
        record is an error, and recovery records (status != error) replace the transport failures they recover.
        not_submitted records are ignored. Returns {(record_id, paper_id, repeat): legacy record}."""
        reused = {'anthropic/claude-sonnet-5.5': 'M206', 'openai-direct/gpt-6.1-sol': 'M207', 'z-ai/glm-5.3': 'GLM-5.3'}
        slots = {}
        for d in read_jsonl(INITIAL / 'runs/2026-09-30-C014-remaining-expanded/responses.jsonl'):
            slots[(reused[d['model']], d['case_id'].split(':')[1], d['repeat'])] = {**d, 'record_id': reused[d['model']]}
        for run in PANEL_RUNS:
            for d in read_jsonl(PANEL_TASK / run / 'responses.jsonl'):
                if d['status'] == 'not_submitted':
                    continue
                k = (d['record_id'], d['case_id'].split(':')[1], d['repeat'])
                old = slots.get(k)
                if 'recovery' in run and d['status'] != 'error' and old is not None and old['status'] not in ('error', 'empty'):
                    # analyze_panel.load() would let a recovery record overwrite a valid output; the frozen panel's
                    # recovery run only covers error/empty slots, so this never fires there (tested). Fail loudly
                    # rather than silently replace a usable output if a future recovery run covers one.
                    raise ValueError(f'recovery record {run} would overwrite usable slot {k} (status {old["status"]})')
                if old is None or old['status'] == 'error' or ('recovery' in run and d['status'] != 'error'):
                    slots[k] = d
        return slots

    def panel_metrics(self, record_id, slots=None):
        """Aggregate one panel model over its merged slots (messages identical across caps; ids by paper)."""
        slots = self.panel_slots() if slots is None else slots
        variant = self.variant()
        items = {x['paper_id']: x for x in self.items(variant)}
        rows = []
        for (rid, pid, rep), rec in slots.items():
            if rid != record_id:
                continue
            item = items[pid]
            assert rec['case']['messages'] == self.program(variant).build(item)
            usable = rec['status'] in {'ok', 'truncated', 'empty'}
            rows.append({'item': item, 'repeat': rep, 'status': rec['status'],
                         'score': self.score(item, {'content': rec.get('content') or ''}, variant) if usable else None})
        return self.aggregate(rows, variant)

    def panel_summary(self, parser='v3', slots=None):
        """Panel-level pooled estimates of first-pass analyze_panel.py (panel_summary_v3) and
        audit_human_baseline.py: paper means averaged over models, then the stratified paper bootstrap."""
        slots = self.panel_slots() if slots is None else slots
        metrics = {rid: self.panel_metrics(rid, slots)['by_parser'][parser] for rid in sorted({k[0] for k in slots})}
        items = self.items(self.variant())
        panel = [r for r in metrics if r not in ('GLM-5.3', 'M196')]   # GLM-5.3 is non-panel; M196 not estimable

        def pooled(models, key):
            per_paper = {}
            for item in items:
                vals = [next(p[key] for p in metrics[r]['papers'] if p['paper'] == item['paper_id']) for r in models]
                vals = [v for v in vals if v is not None]
                per_paper[item['paper_id']] = (item['decision'], statistics.mean(vals), len(vals))
            acc = [v for d, v, _ in per_paper.values() if d == 'accepted']
            rej = [v for d, v, _ in per_paper.values() if d == 'rejected']
            return {'n_models': len(models), 'mean_accepted': statistics.mean(acc), 'mean_rejected': statistics.mean(rej),
                    'difference': statistics.mean(acc) - statistics.mean(rej),
                    'difference_ci95': stratified_bootstrap(acc, rej)[0],
                    'min_models_per_paper': min(n for _, _, n in per_paper.values())}

        full = [r for r, m in metrics.items() if m['complete_papers'] == {'accepted': 10, 'rejected': 10}]
        diffs = [metrics[r]['difference'] for r in panel if metrics[r]['difference'] is not None]
        pos, neg, zero = sum(x > 0 for x in diffs), sum(x < 0 for x in diffs), sum(x == 0 for x in diffs)
        n, k = pos + neg, min(pos, neg)
        human = {d: [statistics.mean(x['archive_human_recommendations']) for x in items if x['decision'] == d]
                 for d in ('accepted', 'rejected')}
        return {'complete_models': full, 'pooled_complete': pooled(full, 'mean'),
                'pooled_available_repeat': pooled(panel, 'available_mean'),
                'sign': {'positive': pos, 'negative': neg, 'zero': zero,
                         'p_two_sided': min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)},
                'models_ci_excluding_zero': [r for r in panel if metrics[r]['difference_ci95'] and
                                             (metrics[r]['difference_ci95'][0] > 0 or metrics[r]['difference_ci95'][1] < 0)],
                'human_paper_weighted': {'mean_accepted': statistics.mean(human['accepted']),
                                         'mean_rejected': statistics.mean(human['rejected']),
                                         'difference': statistics.mean(human['accepted']) - statistics.mean(human['rejected']),
                                         'difference_ci95': stratified_bootstrap(human['accepted'], human['rejected'])[0],
                                         'auc': auc(human['accepted'], human['rejected'])},
                'per_model': metrics}

    targets = (
        PaperTarget('llm_review_weighted_mean_accepted', 7.43, 0.005, 'Sec. 4.1 "Recommendation Score"',
                    'gpt-4-turbo + claude-3-opus + gemini-1.5-flash (archived, 30 reviews)',
                    'table14_reconstructed_v1_cap8192', 'archived_outputs',
                    compute=lambda exp: exp.archive_summary('llm20')['mean']['accepted'],
                    note='tolerance = rounding to 2 decimals',
                    known_discrepancy=('Released ReviewCritique_LLM.jsonl gives 224/30 = 7.467 (Opus 7.6, GPT-4 7.6, '
                                       'Gemini 7.2); the paper\'s 7.43 is not recoverable from the release under any '
                                       'reading of the one anomalous Gemini field (excluding it gives 7.448).')),
        PaperTarget('llm_review_weighted_mean_rejected', 7.47, 0.005, 'Sec. 4.1 "Recommendation Score"',
                    'gpt-4-turbo + claude-3-opus + gemini-1.5-flash (archived, 30 reviews)',
                    'table14_reconstructed_v1_cap8192', 'archived_outputs',
                    compute=lambda exp: exp.archive_summary('llm20')['mean']['rejected'],
                    note='tolerance = rounding to 2 decimals',
                    known_discrepancy=('Released archive gives 226/30 = 7.533 (Opus 7.8, GPT-4 7.8, Gemini 7.0). The '
                                       'released archive still shows no separation (accepted - rejected = -0.067 vs '
                                       'paper -0.04), so the qualitative claim holds on the release.')),
        PaperTarget('human_review_weighted_mean_accepted', 6.41, 0.005, 'Sec. 4.1 "Recommendation Score"',
                    'human reviewers (39 reviews of the 10 accepted matched papers)', 'table14_reconstructed_v1_cap8192',
                    'archived_outputs', compute=lambda exp: exp.archive_summary('human_matched20')['mean']['accepted'],
                    note='250/39 = 6.4103; tolerance = rounding'),
        PaperTarget('human_review_weighted_mean_rejected', 4.81, 0.005, 'Sec. 4.1 "Recommendation Score"',
                    'human reviewers (37 reviews of the 10 rejected matched papers)', 'table14_reconstructed_v1_cap8192',
                    'archived_outputs', compute=lambda exp: exp.archive_summary('human_matched20')['mean']['rejected'],
                    note='178/37 = 4.8108; tolerance = rounding'),
        PaperTarget('llm_deficient_segment_pct', 13.97, 0.005, 'Table 1', 'archived LLM reviews (expert labels)',
                    'table14_reconstructed_v1_cap8192', 'archived_outputs',
                    compute=lambda exp: exp.archive_summary('llm20')['segment_deficiency_pct'],
                    known_discrepancy=('Release has 226 deficient of 1,610 segments (14.04%); Table 1 reports 225 of '
                                       '1,611 (13.97%). Label release differs from the paper by one segment.')),
        PaperTarget('human_deficient_segment_pct', 6.27, 0.005, 'Table 1', 'human reviews, 100 papers (expert labels)',
                    'table14_reconstructed_v1_cap8192', 'archived_outputs',
                    compute=lambda exp: exp.archive_summary('human100')['segment_deficiency_pct'],
                    known_discrepancy=('Release has 712 deficient of 11,376 segments (6.259%); Table 1 reports 713 '
                                       '(6.27%).')),
        PaperTarget('human_reviews_with_deficiency_pct', 71.57, 0.01, 'Table 1', 'human reviews, 100 papers',
                    'table14_reconstructed_v1_cap8192', 'archived_outputs',
                    compute=lambda exp: exp.archive_summary('human100')['review_deficiency_pct'],
                    note='272/380 = 71.579%; the paper truncates to 71.57, so tolerance covers truncation (0.01 pp)'),
        PaperTarget('llm_reviews_with_deficiency_pct', 100.0, 0.005, 'Table 1', 'archived LLM reviews',
                    'table14_reconstructed_v1_cap8192', 'archived_outputs',
                    compute=lambda exp: exp.archive_summary('llm20')['review_deficiency_pct']),
        PaperTarget('llm_difference_original_models', -0.04, 0.005, 'Sec. 4.1', 'gpt-4-turbo / claude-3-opus / gemini-1.5-flash',
                    'table14_reconstructed_v1_cap8192', 'unverifiable',
                    note=('Fresh reviews from the original models were not generated (claude-3-opus and the 2024 '
                          'gemini-1.5-flash-latest alias are retired; prompt fillings unreleased). Panel runs cover '
                          'GPT-4o-2024-05-13 as the closest 2024 snapshot.')),
    )


PANEL_RUNS = ('runs/full-cheap19-2026-10-01', 'runs/full-add-2026-10-01', 'runs/full-cap4096-2026-10-01',
              'runs/full-gpt5-2026-10-02', 'runs/full-dropped4-2026-10-02', 'runs/recovery-2026-10-01')
