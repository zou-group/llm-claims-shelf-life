"""C141 — The Reversal Curse, Experiment 2: real-world celebrity parent/child questions (Berglund et al. 2023).

Claim: models that name a celebrity's parent ("Who is Tom Cruise's mother?") usually cannot answer the reversed
question about the same pair ("Name a child of Mary Lee Pfeiffer."). Paper (arXiv 2309.12288 v4 = ICLR 2024, Exp 2
text identical to v1): GPT-4 identifies the parent "79% of the time, giving us 1573 child-parent pairs"; queried for
the child it "is successful only 33% of the time" (best of 10 samples, footnote). Figure fig:experiment_2_results_1:
gpt-3.5-turbo Parent ~33.8 / Child ~11.3 (mean over 10 samples at temperature 1); Llama-1 7b/30b/65b base models
~7.8-8.0 / 0.2-1.6 (likelihood of the correct completion).

Sources (code/shelflife/data/sources/): C141_reversal_curse = github.com/lukasberglund/reversal_curse @ 6af5f418
(prompts, scorer, released pairs and per-pair results); C141_reversal_curse_arxiv_v4 = arXiv LaTeX v4 (targets).

Items: the released data/celebrity_relations/parent_child_pairs.csv (1,513 rows; GPT-4's temperature-0 answers for
the first 1,000 IMDB celebrities that passed parse_response, NOT checked for correctness). One pair = one CSV row
(row index = stable pair id; two CSV rows are exact duplicates and stay separate rows, as in every released file).
Each pair yields one row per direction: ``parent_query`` (forward: "Who is {child}'s {parent_type}?", target =
parent) and ``child_query`` (reverse: "Name a child of {parent}.", target = child); rows share ``base_id``. Pairs are
ordered by a fixed seeded shuffle (``PAIR_ORDER_SEED``) of CSV rows, so a ``--limit`` prefix is a uniform random
subset of (child, parent_type) pairs, not of celebrities (a prefix can hold a celebrity's mother pair without the
father pair), rather than the most popular celebrities at the top of the IMDB list (popularity drives both
directions). ``limit`` counts pairs; both directions of a pair are always kept together.

Variants
  chat_both          default, faithful. scripts/celebrity_relations/test_parent_child_pairs.py ``reversal_test``
                     chat branch (the gpt-3.5-turbo figure protocol): parent_reversals.py system prompt + 3 few-shot
                     turns, forward and reverse queries, 10 independent samples each at the API default temperature
                     (1), no max_tokens. Per-pair score = share of response lines (all 10 samples pooled) that
                     ``startswith`` the target (case-sensitive, no name parsing).

Scoring (``score``, one sample = one call): the authors' line split (``chat_batch_generate_multiple_messages`` default
``parse``: ``[line.strip() for line in content.strip().split("\\n") if line]``) and their case-sensitive
``line.startswith(target)``; ``gpt4_rule_hit`` additionally requires the authors' ``parse_response`` (loaded from
their file; rejects "I don..." lines, lines of <2 or >5 words, or any word not starting upper-case).
Metrics (``aggregate``, per model; unit = pair; bootstrap clusters = pair ids, 2000 draws):
  mean_over_samples   per pair: sum(correct lines)/sum(lines) over that pair's available samples (exactly the
                      authors' ``query_parent_test``/``query_child_test`` arithmetic); model value = mean over pairs.
  best_of_k           per pair: any available sample with ``gpt4_rule_hit`` (the GPT-4 33% rule); mean over pairs.
  gap                 matched pairs (both directions available): mean(forward mean_over_samples - reverse).
  reverse_given_forward_correct  diagnostic: reverse rates among pairs whose forward query was answered correctly in
                      at least one sample. Labels are GPT-4's unchecked answers; a newer model may "fail" forward
                      because the label is wrong, so this conditions on the model agreeing with the label.
  mean_over_samples_empty_as_wrong_SENSITIVITY  as mean_over_samples, but every available empty sample (no
                      non-empty line, e.g. a reasoning model that spent its cap) counts as one wrong line; its gap is
                      the ``shelflife_sensitivity_empty_as_wrong`` headline.
  lenient_per_sample_share_DIAGNOSTIC  NON-HEADLINE robustness diagnostic, unit = samples: share of samples in which
                      the target appears anywhere (case-insensitive) after stripping markdown (*, _, `, #). Compare it
                      with ``strict_per_sample_share`` (share of samples with a line starting with the target), NOT
                      with the line-pooled mean_over_samples (a multi-line list of children lowers the line share
                      without any format problem). ``format_only_miss`` = lenient hit but no strict line.
  headline_reliable   False when, in either direction, format-only misses, empty samples or truncated samples
                      (status 'truncated' / finish_reason 'length') exceed ``FORMAT_THRESHOLD`` (5%) of available
                      samples, or there are no available samples.
Denominators and missingness: failed calls (score None) are excluded from a pair's pool and counted
(``failed_samples``); a pair with no available sample (or no row at all) in a direction is unavailable for that
direction (excluded from the point estimate, counted, and included in ``bounds_with_missing`` as 0/1). Pairs are
every base id seen in any row plus optional ``expected_base_ids``; the runner supplies every intended row (unsent ones
as not_submitted), and rescoring an interrupted run should pass ``expected_base_ids`` from items(). An available sample with no
non-empty line (empty output) contributes no lines; a pair whose available samples contain no lines at all scores 0
(the authors' code would raise ZeroDivisionError; never happened in their runs) and is counted in
``pairs_without_lines``. Refusals ("I don't know.") are ordinary lines that do not match: they count as wrong and are
also reported (``idk_samples``). Truncated outputs are scored on their text.

Headline (``aggregate()['shelflife']``): forward-minus-reverse gap in mean-over-samples accuracy on matched pairs
(fraction, +1 = supports the claim, null 0). Justification: the claim is an asymmetry within the same fact; the gap
uses the paper's figure metric (mean over 10 samples at T=1, the only metric reported for both directions) and does
not depend on the GPT-4-specific "forward 100% by construction" artefact. Sensitivity: reverse best-of-10 (the 33%
text number) and the empty-as-wrong gap. Paper anchor (``paper_value``): the gpt-3.5-turbo gap on the 1,513 released
rows, 0.2293 (the figure's merged frame gives 0.2243; both are archived targets).

Robustness variants (non-faithful)
  robust_parse       chat_both requests unchanged (same prompts, params, strategy, 10 repeats), options
                     parse='robust', for offline rescoring of saved chat_both outputs. Per sample:
                     1. the authors' rule runs first: if some line starts with the target, is their "I don't know", or
                        is a clean name answer (passes parse_response and is unchanged by markdown / marker / bullet /
                        quote cleaning), the sample keeps the authors' line counts exactly (parse_path 'paper');
                     2. otherwise ``robust_fields`` reads the ``_robust.normalize``d text (markdown emphasis, headings,
                        <think> blocks removed) line by line, after stripping bullets, 'Answer:' / 'A:' / 'Final
                        answer:' markers and wrapping quotes: a direction-specific sentence form anchored at the line
                        start ("X's mother (name) is A", 'The mother of X is A', 'A is the mother of X', 'Her name is
                        A'; reverse: "X's children include A", 'A child of X is A', 'X is the mother of A', 'X has a
                        son named A', 'A is her son') gives the answer field A; else a line that passes parse_response
                        or is name-shaped (<= 6 words, first two capitalized or a name particle) is itself the field.
                        Each field is scored with the authors' case-sensitive ``field.startswith(target)`` and counts
                        as one line in the authors' pooled arithmetic (parse_path 'robust'). The fallback only
                        locates the answer field; it never relaxes the match (case, spelling, nicknames, 'Jr.' vs
                        ', Jr.' still miss). Refusal lines ('Unknown', "I'm not sure ...", 'Sorry, I don't know', "X's
                        mother is not publicly known") are wrong lines (parse_path 'refusal', counted as parsed).
                     3. if nothing is found, the authors' rule on the text stands (all its lines wrong) and the
                        sample is counted ``unparsed`` (empty samples too).
                     Truncated samples (finish_reason 'length') are missing (``_robust.truncation_missing``: the
                     default 'missing'). Reliability: ``_robust.reliable`` per direction (unparsed + missing (failed
                     calls + truncated) <= 10% of samples) and format-only misses (lenient diagnostic hit, no robust
                     hit) <= 10% of available samples; replaces the 5% format / empty / truncated rule.
                     aggregate adds per-direction ``parse`` counts and top-level parsed_paper / parsed_robust /
                     unparsed / truncated_scored / validity_rate, plus flat ``gap`` / ``ci95`` copies of the headline.
                     The headline keys are unchanged.
  robust             robust_parse for new panel runs (the panel): repeats=1 (minimal design of
                     experiments/run_minimal_setups/runs/added-b/minimal_set: 600 pairs x 2 directions x 1 sample; the
                     headline is unchanged in expectation, best-of-k becomes best-of-1), max_tokens 1024 stated
                     explicitly via ``_robust.cap_floor`` (the source sets none; panel policies default to 1024),
                     min_cap_reasoning 16384 through program.case_extra (reasoning routes; policy default 4096), and
                     truncated='score': a cut-off sample is scored on its text by the same paper-then-fallback path
                     (no answer -> no correct line) and counted as ``truncated_scored``.
  Documented exceptions (robust scores differ from the authors' rule on samples that rule did parse):
    * tagged reasoning blocks (<think>, <thinking>, <reasoning>; an unterminated one to the end) are removed before
      the authors' rule, so reasoning lines are not pooled as wrong answer lines;
    * robust_parse only: truncated samples are missing (excluded from the pool and counted), whatever they contain.
  Scoring revision (scoring only, variant fingerprints unchanged):
    * M1: a sentence-form match is rejected when its line carries a negation / retraction / hedge cue (lower-case
      'not', "n't", 'never', 'no', 'incorrect', 'false', 'wrong', 'rumored', 'claim', 'allegedly', 'believe', 'think',
      'might', 'may', 'maybe', 'possibly', 'probably', 'likely', 'seems', 'unclear', 'if', ...; or any of them as the
      line's first word); the line is a wrong line, and a sample with no other field is unparsed. A line that itself
      starts with the target stays on the paper path (the authors' rule credits it).
    * m1: on the fallback path every non-empty cleaned line is pooled (fields + refusal lines + other lines), so a
      bulleted or bold list scores the authors' 1/k like the same list without markup.
    * m2: a sentence-form field must open with a name (two capitalized words or a name particle, a mononym, or
      'A and B'; not 'Who', 'There', 'It', ...); 'no public record' is a refusal.
  On the saved gpt-3.5-turbo-0125 L200 pilot, robust_parse keeps 3,828 of 4,000 samples on the paper path, reads 149
  sentence answers ("Harold Perrineau's mother is Sylvia Perrineau.") and 9 prose refusals, leaves 14 unparsed; the
  gap rises from 0.285 to 0.3285 (forward sentence answers were scored wrong by the strict line rule).

Deviations (all variants): retired paper snapshots (Sep-2023 ``gpt-4`` = gpt-4-0613 is still served directly;
``gpt-3.5-turbo`` = -0613 is retired on OpenAI direct, and the OpenRouter "-0613" route is a GPT-4o-family proxy per
C043/C051 findings, so gpt-3.5-turbo-0125 is the closest substitute; IMDb accessed 2023-06-28 and API 2023-08-17 per
iclr2024_conference.bib, both after the alias moved to -0613); temperature 1 requested explicitly (the source passes
none, the OpenAI default is 1). The source sets no max_tokens. Cap and temperature handling are per-route policy
adaptations recorded per call: originals_v1 adds a 4096 cap; panel policies use 1024 for ordinary routes and 4096 for
reasoning routes, and some routes (e.g. M204/M206/M207) omit temperature, so T=1 is NOT applied there (recorded in
``omitted_unsupported``). gpt-4-0613 must be run with ``--policy originals_v1``: its default route
(initial_openai_v1) fails fast with "Explicit source cap required"; 1,513 released pairs vs "1573" in the paper (1573 is the row count of the notebook's successive
pd.merge of the 5 result files, which cross-multiplies the 2 duplicated pairs: 1509 + 2*2^5).
"""
import csv
import functools
import random
import re
from collections import defaultdict

from ..paths import PROJECT
from ..program import SingleTurn
from ..prompts import PromptTemplate, Source
from ..registry import Experiment, PaperTarget, Variant, register
from ..results import Headline
from ..sourcecode import load_functions
from .. import stats
from . import _robust

REPO = PROJECT / 'code/shelflife/data/sources/C141_reversal_curse'
PAPER_SRC = PROJECT / 'code/shelflife/data/sources/C141_reversal_curse_arxiv_v4'
COMMIT = '6af5f418755d528304e406361e86477637d21c84'
URL = 'https://github.com/lukasberglund/reversal_curse'
DATA = REPO / 'data/celebrity_relations'
PAIRS_CSV = DATA / 'parent_child_pairs.csv'
N_PAIRS = 1513
GPT35_GAP_RELEASED_ROWS = 0.229280   # released gpt-3.5 rows: 0.348381 - 0.119101 (target below recomputes it)
PAIR_ORDER_SEED = 20230921   # arXiv v1 date; fixed seeded shuffle so --limit prefixes are uniform random CSV-row subsets
DIRECTIONS = ('parent_query', 'child_query')
# The notebook merges 5 released per-model result files (plot_parent_child_reversals.ipynb cells 1-2).
NOTEBOOK_MERGE_ORDER = ('llama-7b', 'llama-65b', 'davinci', 'gpt-3.5-turbo', 'llama-30b')
MERGE_KEYS = ('child', 'parent', 'parent_type', 'child_prediction')

PR = 'src/tasks/celebrity_relations/parent_reversals.py'
SRC = Source(repo=str(REPO), path=PR, locator='SYSTEM_PROMPT, get_initial_messages, get_parent_query, get_child_query',
             commit=COMMIT, url=URL)


@functools.lru_cache(maxsize=None)
def official():
    """The authors' SYSTEM_PROMPT and parse_response, compiled from the snapshot without running its module body."""
    return load_functions(REPO / PR, ['parse_response'], constants=('UNKNOWN_STR', 'SYSTEM_PROMPT'))


SYSTEM = PromptTemplate('system', official()['SYSTEM_PROMPT'], style='literal', source=Source(
    repo=str(REPO), path=PR, locator='L11-12 SYSTEM_PROMPT (f-string over UNKNOWN_STR, evaluated from the file)',
    commit=COMMIT, url=URL), note='appendix.tex L189 prints the same text with typographic quotes; the code is used')
PARENT_Q = PromptTemplate('parent_question', "Who is {child}'s {parent_type}?", source=Source(
    repo=str(REPO), path=PR, locator='L28-29 ParentChildPair.ask_for_parent (f-string)', commit=COMMIT, url=URL),
    note='transcribed from the f-string; test_c141 checks byte equality with the authors\' builder on all pairs')
CHILD_Q = PromptTemplate('child_question', 'Name a child of {parent}.', source=Source(
    repo=str(REPO), path=PR, locator='L31-32 ParentChildPair.ask_for_child (f-string)', commit=COMMIT, url=URL),
    note='transcribed from the f-string; test_c141 checks byte equality with the authors\' builder on all pairs')
# get_initial_messages L61-66: (child, parent, parent_type, which question) of the three few-shot turns.
FEW_SHOT = (('Malia Obama', 'Barack Obama', 'father', 'child_query'),
            ('Elon Musk', 'Maye Musk', 'mother', 'parent_query'),
            ('Kathy Pratt', "I don't know.", 'mother', 'parent_query'))


def question(direction, child, parent, parent_type):
    if direction == 'parent_query':
        return PARENT_Q.render(child=child, parent_type=parent_type)
    if direction == 'child_query':
        return CHILD_Q.render(parent=parent)
    raise ValueError(direction)


def chat_messages(item):
    """System prompt, 3 few-shot turns (question, answer), then the query; as get_parent_query/get_child_query."""
    messages = [{'role': 'system', 'content': SYSTEM.render()}]
    for child, parent, parent_type, direction in FEW_SHOT:
        messages.append({'role': 'user', 'content': question(direction, child, parent, parent_type)})
        messages.append({'role': 'assistant', 'content': parent if direction == 'parent_query' else child})
    messages.append({'role': 'user', 'content': question(item['direction'], item['child'], item['parent'],
                                                         item['parent_type'])})
    return messages


# ------------------------------------------------------------------------------------------- scoring helpers
def split_lines(content):
    """openai_chat.py chat_batch_generate_multiple_messages default ``parse`` (transcribed; tested equal)."""
    return [line.strip() for line in content.strip().split('\n') if line]


def parse_response(line):
    """parent_reversals.py parse_response, loaded from the authors' file."""
    return official()['parse_response'](line)


MARKDOWN = re.compile(r'[*_`#]')


def lenient_hit(content, target):
    """NON-HEADLINE diagnostic: case-insensitive containment of the target after removing markdown characters."""
    norm = lambda s: ' '.join(MARKDOWN.sub('', s).split()).casefold()
    return bool(target) and norm(target) in norm(content)


# ------------------------------------------------------------------- robust fallback (parse='robust', the robust-variant conventions)
ROBUST_MIN_CAP_REASONING = 16384   # 'robust': reasoning-model cap (policy min_cap_reasoning via case_extra)
_REASONING_BLOCK = re.compile(r'<(think|thinking|reasoning)>.*?</\1>|<(?:think|thinking|reasoning)>.*\Z', re.S | re.I)
_REL_PARENT = r'(?:mother|father|parent|mom|mum|dad)'
_REL_CHILD = r'(?:child|children|sons?|daughters?|kids?)'
_ADJ = r'(?:(?:known|only|eldest|oldest|youngest|famous|notable|biological|adopted|first|late)\s+)?'
_COP = r'(?:is|was|are|were)'
_POSS = r"[\'’]s?"
# Sentence forms, anchored at the start of a (cleaned) line; group 'a' is the answer field, scored with the authors'
# case-sensitive ``startswith(target)``. Forward (target = parent) and reverse (target = child) are kept separate so a
# sentence naming the other member of the pair is never read as the answer.
_NAMED = rf'(?:{_POSS}\s+(?:full\s+)?name)?'                 # "X's mother's name is A"
_IS_A = rf'{_COP}\s+(?:also\s+)?(?:(?:named|called)\s+)?'      # "is A", "is also named A"
_NAME_IS = r'^(?:his|her|their)\s+(?:full\s+)?name\s+(?:is|was)\s+(?P<a>.+)$'   # "Her name is A"
_SENTENCE = {
    'parent_query': [re.compile(p, re.I) for p in (
        rf'^(?:the\s+)?{_ADJ}{_REL_PARENT}\s+of\s+.+?\s+{_IS_A}(?P<a>.+)$',                 # The mother of X is A
        rf'^.+?{_POSS}\s+{_ADJ}{_REL_PARENT}{_NAMED}\s+{_IS_A}(?P<a>.+)$',                   # X's mother('s name) is A
        rf'^(?:his|her|their)\s+{_ADJ}{_REL_PARENT}{_NAMED}\s+{_IS_A}(?P<a>.+)$',            # His mother is A
        _NAME_IS,
        rf'^(?P<a>.+?)\s+{_COP}\s+(?:the\s+)?{_ADJ}{_REL_PARENT}\s+of\b.*$',                 # A is the mother of X
        rf'^(?P<a>.+?)\s+{_COP}\s+(?:his|her|their|.+?{_POSS})\s+{_ADJ}{_REL_PARENT}\b.*$',  # A is X's / his mother
    )],
    'child_query': [re.compile(p, re.I) for p in (
        rf'^(?:the\s+|a\s+|one\s+)?{_ADJ}{_REL_CHILD}\s+of\s+.+?\s+(?:{_IS_A}|includes?\s+)\s*:?\s*(?P<a>.+)$',  # A child of X is A
        rf'^(?:one\s+of\s+)?.+?{_POSS}\s+{_ADJ}{_REL_CHILD}{_NAMED}\s+(?:{_IS_A}|includes?\s+|including\s+)\s*:?\s*'
        rf'(?P<a>.+)$',                                                                          # X's son is A
        rf'^(?:one\s+of\s+)?(?:his|her|their)\s+{_ADJ}{_REL_CHILD}{_NAMED}\s+(?:{_IS_A}|includes?\s+)\s*:?\s*(?P<a>.+)$',
        _NAME_IS,
        rf'^.+?\s+(?:has|had)\s+(?:a\s+|an\s+|one\s+|two\s+|three\s+|four\s+|five\s+|several\s+)?{_ADJ}{_REL_CHILD}'
        rf'(?:\s*:\s*|\s*,\s*|\s+(?:named|called)\s+)(?P<a>.+)$',                                # X has a son named A
        rf'^.+?\s+{_COP}\s+(?:best\s+known\s+as\s+|known\s+as\s+)?(?:the\s+)?{_REL_PARENT}\s+of\s+(?P<a>.+)$',  # X is the mother of A
        rf'^(?P<a>.+?)\s+{_COP}\s+(?:a|an|the|his|her|their|one\s+of\s+(?:the|his|her|their)|.+?{_POSS})\s+{_ADJ}'
        rf'{_REL_CHILD}\b.*$',                                                                   # A is her / X's son
    )],
}
_MARKER = re.compile(r'^(?:(?:final\s+)?answer\s*[:\-]|a\s*:|the\s+answer\s+is\s*:?)\s*', re.I)
_PARTICLES = {'de', 'van', 'von', 'da', 'del', 'della', 'di', 'du', 'la', 'le', 'bin', 'al', 'ten', 'ter', 'dos', 'das'}
_BULLET = re.compile(r'^(?:[-*•]\s+|\d{1,2}[.)]\s+)')
_QUOTES = '"\'“”‘’`'
_REFUSAL = re.compile(
    r"^(?:sorry|i'?m sorry|i apologi[sz]e|unfortunately)?[,.]?\s*(?:but\s+)?"
    r"(?:i\s+(?:do\s+not|don'?t|cannot|can'?t|could\s*n[o']t|am\s+not|'m\s+not|am\s+unable|'m\s+unable|have\s+no|"
    r"don'?t\s+have|do\s+not\s+have|lack)\b|i'm\s+(?:not|unable)\b|unknown\b|not\s+(?:public(?:ly)?\s+)?(?:known|available)\b|"
    r"no\s+(?:public(?:ly)?\s+)?(?:available\s+)?(?:information|records?|data)\b|there\s+(?:is|are)\s+no\s+(?:public(?:ly)?\s+)?"
    r"(?:available\s+)?(?:information|records?|data)\b)", re.I)
# Audit M1 (2026-10-11): a sentence form is only an answer if the line asserts it. Negation, retraction and hedge cues
# anywhere in the line (lower case, so names such as 'May' are not cues; or capitalized as the first word) reject the
# sentence match: "Some sources claim X's mother is A, but this is incorrect.", "X has not had a son named A.",
# "I believe X's mother is A." The line then counts as a wrong line, never as a field.
_CUES = (r"not|never|no|none|neither|nor|incorrect|false|wrong|untrue|rumou?r(?:s|ed)?|claims?|claimed|allegedly|"
         r"reportedly|supposedly|believe[sd]?|think|thought|might|may|maybe|perhaps|possibly|probably|likely|unsure|"
         r"uncertain|unclear|guess|seems?|apparently|if|whether|unless")
_CUE_ANY = re.compile(rf"\b(?:{_CUES})\b|n['’]t\b")
_CUE_START = re.compile(rf"^(?:{_CUES})\b", re.I)
# Audit m2: a sentence-form field must open with a name (two capitalized words or a name particle, or a single
# capitalized word that is not a function word): "Who is X's mother? A" no longer yields the field 'Who'.
_NOT_NAMES = {'who', 'what', 'which', 'there', 'here', 'it', 'this', 'that', 'these', 'those', 'he', 'she', 'they',
              'his', 'her', 'their', 'the', 'a', 'an', 'i', 'we', 'you', 'some', 'one', 'unknown', 'none', 'yes', 'no'}


def strip_reasoning_blocks(content):
    """Remove tagged reasoning blocks (<think>, <thinking>, <reasoning>; an unterminated one runs to the end)."""
    return _REASONING_BLOCK.sub('\n', content)


def _clean_line(line):
    """A response line with list bullets, answer markers ('Answer:', 'A:', 'Final answer:') and wrapping quotes
    removed (markdown emphasis is already gone after _robust.normalize)."""
    s = line.strip()
    for _ in range(3):
        before = s
        s = _BULLET.sub('', s).strip()
        s = _MARKER.sub('', s).strip()
        s = s.strip(_QUOTES).strip()
        if s == before:
            break
    return s


def _name_shaped(s):
    """A bare name-like answer: <= 6 words, the first two capitalized, not a lead-in ending in ':'."""
    words = s.split()
    return (2 <= len(words) <= 6 and words[0][:1].isupper() and (words[1][:1].isupper() or words[1] in _PARTICLES)
            and not s.endswith(':') and not _is_refusal(s))


def _hedged(s):
    """True if the line carries a negation, retraction or hedge cue (audit M1)."""
    return bool(_CUE_ANY.search(s) or _CUE_START.match(s))


def _name_opening(field):
    """True if a sentence-form field opens with a name (audit m2)."""
    words = field.split()
    if not words:
        return False
    w0 = words[0].strip('.,;:!?()')
    if not w0[:1].isupper() or w0.lower() in _NOT_NAMES:
        return False
    if len(words) == 1 or words[0][-1:] in '.,;:!?)' or words[1] == 'and':   # mononym ('Madonna.', 'Sasha and Kai')
        return True
    return words[1][:1].isupper() or words[1] in _PARTICLES


def _is_clean(line):
    """True if the line is unchanged by _robust.normalize and _clean_line (no markdown, marker, bullet or quotes)."""
    return _clean_line(_robust.normalize(line)) == line


def _is_refusal(s):
    return bool(_REFUSAL.match(s.replace('\u2019', "'")))


def robust_fields(content, direction):
    """Fallback answer fields (parse='robust', used only when the paper rule parsed nothing in the sample).

    On the ``_robust.normalize``d text (reasoning blocks, markdown emphasis/code and headings removed), each line is
    cleaned (bullets, 'Answer:'-style markers, wrapping quotes) and read as:
      * a refusal ('Unknown', "I'm not sure ...", 'Sorry, I don't know', 'No information ...') -> refusal line;
      * a direction-specific sentence form anchored at the line start ("X's mother is A", 'The mother of X is A',
        'A is the mother of X'; reverse: "A child of X is A", "X's children include A", 'X is the mother of A',
        'X has a son named A', 'A is her son') -> field A;
      * a cleaned line that passes the authors' parse_response or is name-shaped -> field = the cleaned line.
    A sentence match is rejected when the line carries a negation / retraction / hedge cue (``_hedged``, audit M1)
    or its field does not open with a name (``_name_opening``, audit m2); the line is then an ordinary wrong line.
    Returns (fields, refusal_lines, other_lines): every non-empty cleaned line is exactly one of the three, so the
    caller pools all of them like the authors' rule pools every line (audit m1: the denominator does not depend on
    bullets or markdown). Correctness is decided by the caller with the authors' case-sensitive
    ``field.startswith(target)``, so this only locates the answer; it never relaxes the match."""
    fields, refusals, other = [], 0, 0
    for raw in split_lines(_robust.normalize(content)):
        s = _clean_line(raw)
        if not s:
            continue
        if _is_refusal(s):
            refusals += 1
            continue
        hit = None
        for rx in _SENTENCE[direction]:
            m = rx.match(s)
            if m:
                hit = _clean_line(m.group('a'))
                break
        if hit is not None:
            if hit and _is_refusal(hit):      # "X's mother is not publicly known."
                refusals += 1
            elif hit and not _hedged(s) and _name_opening(hit):
                fields.append(hit)
            else:
                other += 1
        elif parse_response(s) is not None or _name_shaped(s):
            fields.append(s)
        else:
            other += 1
    return fields, refusals, other


def read_pairs():
    with PAIRS_CSV.open(newline='') as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == N_PAIRS, len(rows)
    return rows


def pair_order():
    order = list(range(N_PAIRS))
    random.Random(PAIR_ORDER_SEED).shuffle(order)
    return order


def read_archived(model='gpt-3.5-turbo'):
    """Released gpt-3.5-turbo per-pair values: (pair key, parent fraction, child fraction of 10 samples), CSV order."""
    with (DATA / f'{model}_reversal_test_results.csv').open(newline='') as fh:
        rows = list(csv.DictReader(fh))
    return [(tuple(r[k] for k in MERGE_KEYS), float(r[f'{model}_can_find_parent']), float(r[f'{model}_can_find_child']))
            for r in rows]


def archived_bars(model='gpt-3.5-turbo', merged=True):
    """Figure bar (fraction) for gpt-3.5-turbo: mean of per-pair values. ``merged`` reproduces the notebook's
    successive inner pd.merge of the 5 files on MERGE_KEYS: a key that occurs c times in each file appears c**5 times
    in the merged frame, so each of its rows carries weight c**4 (n_files - 1). Returns (n_rows, parent, child)."""
    rows = read_archived(model)
    counts = defaultdict(int)
    for key, _, _ in rows:
        counts[key] += 1
    weight = (lambda key: counts[key] ** (len(NOTEBOOK_MERGE_ORDER) - 1)) if merged else (lambda key: 1)
    total = sum(weight(k) for k, _, _ in rows)
    return (total, sum(weight(k) * p for k, p, _ in rows) / total, sum(weight(k) * c for k, _, c in rows) / total)


@register
class C141ReversalCurse(Experiment):
    claim_id = 'C141'
    maturity = 'ready'
    title = 'Reversal Curse: celebrity parent/child questions (Experiment 2)'
    claim = ("Models that can name a celebrity's parent ('Who is Tom Cruise's mother?') usually cannot answer the "
             "reversed question about the same pair ('Name a child of Mary Lee Pfeiffer.'): GPT-4 names the parent for "
             "79% of celebrities but the child in only 33% of reversed pairs, and gpt-3.5-turbo and Llama-1 "
             "7b/30b/65b show the same forward >> reverse asymmetry.")
    paper = {'title': 'The Reversal Curse: LLMs trained on "A is B" fail to learn "B is A"',
             'authors': 'Berglund, Tong, Kaufmann, Balesni, Cooper Stickland, Korbak, Evans', 'year': 2023,
             'url': 'https://arxiv.org/abs/2309.12288', 'version': 'v4 (ICLR 2024; Exp 2 identical to v1)'}
    source = SRC
    default_variant = 'chat_both'
    FORMAT_THRESHOLD = 0.05     # format-only / empty / truncated share of available samples -> headline_reliable False
    N_BOOT = 2000
    BOOT_SEED = 20231006

    def build_variants(self):
        deviations = (
            'paper snapshots: gpt-4 (Sep 2023 alias = gpt-4-0613, still served) and gpt-3.5-turbo (= -0613, retired '
            'on OpenAI direct; the OpenRouter "-0613" route is a GPT-4o-family proxy, so -0125 is the closest '
            'substitute)',
            'temperature 1 requested explicitly; the source passes no temperature (OpenAI API default 1); routes '
            'that reject temperature omit it (recorded in omitted_unsupported), so T=1 is not applied there',
            'source sets no max_tokens; cap handling is a per-route policy adaptation recorded per call '
            '(originals_v1 4096; panel policies 1024 for ordinary and 4096 for reasoning routes); gpt-4-0613 '
            'needs --policy originals_v1 (its default initial_openai_v1 route requires an explicit cap)',
            '10 samples per query are 10 independent repeats, as in the source (10 parallel n=1 calls)',
            'item set: the 1,513 released pairs; the paper reports 1573, the row count of the notebook\'s '
            'successive pd.merge that cross-multiplies 2 duplicated pairs (1509 + 2*2^5)',
        )
        both = Variant(
            'chat_both', 'test_parent_child_pairs.py reversal_test chat branch (gpt-3.5-turbo figure protocol): '
                         'forward + reverse queries, 10 samples at T=1, share of lines starting with the target',
            prompts={'system': SYSTEM, 'parent_question': PARENT_Q, 'child_question': CHILD_Q},
            params={'temperature': 1}, repeats=10, strategy='single_turn_paired_directions',
            options={'directions': DIRECTIONS, 'pair_order_seed': PAIR_ORDER_SEED}, deviations=deviations)
        # Robustness variants; see the module docstring.
        rparse = both.override(
            'robust_parse', description='chat_both requests, parse=robust (authors\' prefix rule first; when no line '
                                        'parses, the same rule on an answer field extracted from markdown / '
                                        'sentence / marker / <think> formats; tolerant reliability)',
            options={'parse': 'robust'},
            deviations=('scoring: authors\' line rule first; a sample in which no line starts with the target, '
                        'passes parse_response or is "I don\'t know" is re-read from the _robust-normalized text '
                        '(answer field after "X\'s mother is" / "Answer:" / quotes / bullets / markdown), then scored '
                        'with the same case-sensitive startswith(target); tagged reasoning blocks are removed before '
                        'the authors\' rule; truncated samples are missing; reliability = _robust.reliable (<= 10% '
                        'unparsed or missing per direction) and <= 10% format-only misses',))
        robust = rparse.override(
            'robust', description='robust_parse + 1 sample per query (minimal design: 600 pairs x 2 directions x 1), '
                                  'explicit max_tokens floor 1024 and min_cap_reasoning 16384 for panel runs',
            params=_robust.cap_floor({**both.params, 'max_tokens': _robust.MIN_ROBUST_TOKENS}), repeats=1,
            options={'min_cap_reasoning': ROBUST_MIN_CAP_REASONING, 'truncated': 'score'},
            deviations=('1 sample per query instead of 10 (minimal_set README: between-pair variance dominates; '
                        'mean_over_samples is unbiased for the per-sample hit rate at any K; best-of-k sensitivity is '
                        'then best-of-1 and not comparable with the paper\'s best-of-10)',
                        'max_tokens 1024 stated explicitly (_robust.cap_floor; the source sets none and panel policies '
                        'already default to 1024), reasoning-model cap 16384 via min_cap_reasoning (policy default '
                        '4096) so hidden reasoning cannot empty the answer',
                        "truncated='score': at these caps a cut-off sample is a failed answer, scored on its text "
                        '(no answer -> wrong), counted as truncated_scored, not missing'))
        return {'chat_both': both, 'robust_parse': rparse, 'robust': robust}

    # ------------------------------------------------------------------ data
    def items(self, variant, limit=None):
        pairs = read_pairs()
        rows = []
        for index in pair_order()[:limit] if limit is not None else pair_order():
            p = pairs[index]
            base = f'C141:p{index:04d}'
            for direction in variant.options['directions']:
                rows.append({'id': f'{base}:{direction}', 'base_id': base, 'pair_index': index,
                             'direction': direction, 'child': p['child'], 'parent': p['parent'],
                             'parent_type': p['parent_type'],
                             'target': p['parent'] if direction == 'parent_query' else p['child'],
                             'gpt4_child_prediction': p['child_prediction'],
                             'gpt4_can_reverse': p['can_reverse'] == 'True'})
        return rows

    def program(self, variant):
        extra = ({'min_cap_reasoning': variant.options['min_cap_reasoning']}
                 if variant.options.get('min_cap_reasoning') else None)
        return SingleTurn(chat_messages, variant.params, case_extra=extra)

    # --------------------------------------------------------------- scoring
    def score(self, item, output, variant):
        content = output.get('content') or ''
        target = item['target']
        truncated = output.get('status') == 'truncated' or output.get('finish_reason') == 'length'
        lines = split_lines(content)
        correct = [line.startswith(target) for line in lines]
        strict_any = any(correct)
        lenient = lenient_hit(content, target)
        idk = lambda ls: any(line.replace('\u2019', "'").startswith("I don't know") for line in ls)
        out = {'direction': item['direction'], 'n_lines': len(lines), 'n_correct_lines': sum(correct),
               'any_line_correct': strict_any,
               'gpt4_rule_hit': any(parse_response(line) is not None and line.startswith(target) for line in lines),
               'idk': idk(lines), 'empty': not lines, 'truncated': truncated,
               'lenient_hit': lenient, 'format_only_miss': lenient and not strict_any}
        if variant.options.get('parse', 'paper') == 'robust':
            out.update(self._robust_score(content, target, item['direction'], truncated, lenient, idk, variant))
        return out

    def _robust_score(self, content, target, direction, truncated, lenient, idk, variant):
        """parse='robust' overrides of the line counts (see the module docstring, 'Robustness variants')."""
        rec = {'paper_n_lines': None, 'paper_n_correct_lines': None, 'parse_path': None, 'refusal': False,
               'missing': False, 'truncated_scored': False, 'robust_fields': []}
        if truncated and _robust.truncation_missing(variant):
            rec.update(missing=True, parse_path='truncated_missing')
            return rec
        rec['truncated_scored'] = truncated
        text = strip_reasoning_blocks(content) if _REASONING_BLOCK.search(content) else content
        lines = split_lines(text)
        correct = [line.startswith(target) for line in lines]
        # The authors' rule "succeeded" when a line starts with the target, is their "I don't know", or is a clean
        # name answer (passes parse_response and needs no cleaning: 'Answer: Maye Musk' passes parse_response but is a
        # marker-wrapped answer, so it goes to the fallback, which can only add the prefix hit on 'Maye Musk').
        paper_parsed = any(correct) or idk(lines) or any(parse_response(line) is not None and _is_clean(line)
                                                         for line in lines)
        rec.update(paper_n_lines=len(lines), paper_n_correct_lines=sum(correct))
        if paper_parsed:
            fields, refusals, other, path = lines, 0, 0, 'paper'
        else:
            fields, refusals, other = robust_fields(text, direction)
            path = 'robust' if fields else ('refusal' if refusals else None)
            if path is None:          # nothing extracted: the authors' rule on the text stands (no correct line)
                fields, other = lines, 0
        hits = [f.startswith(target) for f in fields]
        any_hit = any(hits)
        rec.update(parse_path=path, refusal=path == 'refusal' or (path == 'paper' and idk(lines)),
                   robust_fields=fields if path == 'robust' else [],
                   n_lines=len(fields) + refusals + other, n_correct_lines=sum(hits),
                   any_line_correct=any_hit, non_field_lines=other,
                   gpt4_rule_hit=any(parse_response(f) is not None and h for f, h in zip(fields, hits)),
                   empty=not lines, format_only_miss=lenient and not any_hit)
        return rec

    def _pairs(self, rows, direction, bases=()):
        """Per-pair pooled values for one direction: {base_id: dict}. Every base id in ``bases`` (and every pair
        seen in ``rows`` in any direction) gets an entry, so pairs with no row or no available sample in this
        direction stay in the denominator of ``pairs``/``missing``/``bounds_with_missing``."""
        groups = {b: [] for b in bases}
        for r in rows:
            groups.setdefault(r['item']['base_id'], [])
            if r['item']['direction'] == direction:
                groups[r['item']['base_id']].append(r)
        out = {}
        for base, rs in groups.items():
            # parse='robust' only: a sample marked missing (robust_parse: truncated) leaves the pool like a failed call
            scores = [r['score'] for r in rs if r.get('score') is not None and not r['score'].get('missing')]
            lines = sum(s['n_lines'] for s in scores)
            n_empty = sum(s['empty'] for s in scores)
            out[base] = {
                'samples': len(rs), 'available': len(scores),
                'mean': (sum(s['n_correct_lines'] for s in scores) / lines if lines else 0.0) if scores else None,
                # sensitivity (M1): every available empty sample counts as one wrong line
                'mean_empty_as_wrong': (sum(s['n_correct_lines'] for s in scores) / (lines + n_empty)) if scores else None,
                'no_lines': bool(scores) and lines == 0,
                'best': any(s['gpt4_rule_hit'] for s in scores) if scores else None,
                'any_line': any(s['any_line_correct'] for s in scores) if scores else None,
                'strict_sample_share': sum(s['any_line_correct'] for s in scores) / len(scores) if scores else None,
                'lenient_sample_share': sum(s['lenient_hit'] for s in scores) / len(scores) if scores else None,
                'idk': sum(s['idk'] for s in scores), 'empty': n_empty,
                'truncated': sum(s.get('truncated', False) for s in scores),
                'format_only': sum(s['format_only_miss'] for s in scores)}
        return out

    def _rate(self, pairs, key):
        """Mean over available pairs of pairs[*][key], pair-cluster bootstrap CI, bounds with unavailable pairs."""
        avail = [(b, p[key]) for b, p in sorted(pairs.items()) if p[key] is not None]
        n, k = len(pairs), len(avail)
        mean = lambda xs: sum(float(v) for _, v in xs) / len(xs) if xs else float('nan')
        _, lo, hi = stats.cluster_bootstrap(avail, lambda x: x[0], mean, n_boot=self.N_BOOT, seed=self.BOOT_SEED)
        total = sum(float(v) for _, v in avail)
        return {'pairs': n, 'available_pairs': k, 'rate': total / k if k else None,
                'ci95_pair_bootstrap': [lo, hi] if k else None,
                'bounds_with_missing': [total / n, (total + n - k) / n] if n else None}

    def aggregate(self, rows, variant, expected_base_ids=None):
        """``expected_base_ids`` (optional, e.g. from items() when rescoring an interrupted run) adds pairs that have
        no rows at all; the runner already supplies every intended row (unsent ones with status not_submitted)."""
        directions = variant.options['directions']
        out = {'n': len(rows), 'scored': sum(r.get('score') is not None for r in rows),
               'failed_samples': sum(r.get('score') is None for r in rows), 'directions': {}}
        bases = sorted(set(expected_base_ids or ()) | {r['item']['base_id'] for r in rows})
        per = {d: self._pairs(rows, d, bases) for d in directions}
        for d in directions:
            pairs = per[d]
            avail_samples = sum(p['available'] for p in pairs.values())
            fmt = sum(p['format_only'] for p in pairs.values())
            empty = sum(p['empty'] for p in pairs.values())
            trunc = sum(p['truncated'] for p in pairs.values())
            share = lambda k: k / avail_samples if avail_samples else None
            out['directions'][d] = {
                'mean_over_samples': self._rate(pairs, 'mean'), 'best_of_k': self._rate(pairs, 'best'),
                'mean_over_samples_empty_as_wrong_SENSITIVITY': self._rate(pairs, 'mean_empty_as_wrong'),
                'any_line_startswith': self._rate(pairs, 'any_line'),
                # per-sample shares (unit: samples, not pooled lines): compare lenient with strict on the same unit
                'strict_per_sample_share': self._rate(pairs, 'strict_sample_share'),
                'lenient_per_sample_share_DIAGNOSTIC': self._rate(pairs, 'lenient_sample_share'),
                'samples': sum(p['samples'] for p in pairs.values()), 'available_samples': avail_samples,
                'failed_samples': sum(p['samples'] - p['available'] for p in pairs.values()),
                'pairs_with_failed_samples': sum(p['available'] < p['samples'] for p in pairs.values()),
                'pairs_without_rows': sum(p['samples'] == 0 for p in pairs.values()),
                'pairs_without_lines': sum(p['no_lines'] for p in pairs.values()),
                'idk_samples': sum(p['idk'] for p in pairs.values()),
                'empty_samples': empty, 'empty_share': share(empty),
                'truncated_samples': trunc, 'truncated_share': share(trunc),
                'format_only_miss_samples': fmt, 'format_only_miss_share': share(fmt)}
        robust = variant.options.get('parse', 'paper') == 'robust'
        problems = []
        if robust:
            problems = self._robust_counts(rows, out, directions)
        else:
            for d, v in out['directions'].items():
                for key in ('format_only_miss_share', 'empty_share', 'truncated_share'):
                    if v[key] is None or v[key] > self.FORMAT_THRESHOLD:
                        problems.append(f'{d}.{key}={v[key]}')
        reliable = not problems
        out['headline_reliable'] = reliable
        if not reliable and robust:
            out['headline_caveat'] = ('parse=robust: ' + ', '.join(problems) + f' (tolerance {_robust.ROBUST_TOLERANCE:.0%} '
                                      'of samples per direction unparsed + missing, and of available samples '
                                      'format-only misses)')
        elif not reliable:
            out['headline_caveat'] = (f'above {self.FORMAT_THRESHOLD:.0%} of available samples (or no samples): '
                                      + ', '.join(problems) + '. Format-only misses: compare '
                                      'lenient_per_sample_share_DIAGNOSTIC with strict_per_sample_share. Empty or '
                                      'truncated samples leave the pooled line denominator (authors\' rule): compare '
                                      'mean_over_samples_empty_as_wrong_SENSITIVITY')
        out.update(self._paired(per['parent_query'], per['child_query']))
        gap = out['gap_mean_over_samples']
        out['shelflife'] = Headline(
            metric='forward_minus_reverse_mean_over_samples', value=gap['gap'], n=gap['matched_pairs'],
            direction=1, unit='fraction', null_value=0.0, paper_value=GPT35_GAP_RELEASED_ROWS,
            denominator=('matched pairs (both directions have >=1 available sample); per pair: share of '
                         'response lines over its samples starting with the target (authors\' rule), forward '
                         'minus reverse'),
            ci=tuple(gap['ci95_pair_bootstrap']) if gap['gap'] is not None else None,
            missing=gap['pairs'] - gap['matched_pairs'],
            missingness_note='pairs lacking any available sample in a direction are excluded and counted',
            headline_reliable=reliable,
            reliability_note=('paper anchor = gpt-3.5-turbo(-0613) gap on the 1,513 released rows (0.2293; the '
                              'figure\'s merged frame gives 0.2243). ' + (
                                  'parse=robust: reliable if unparsed + missing <= 10% of samples and format-only '
                                  'misses <= 10% of available samples in both directions' if robust else
                                  'Reliable only if format-only, empty and truncated shares are <= 5% in both '
                                  'directions')),
        ).to_dict()
        if robust:   # flat copies for collectors (same values as the headline)
            out['gap'] = gap['gap']
            out['ci95'] = gap['ci95_pair_bootstrap']
        ew = out['gap_mean_over_samples_empty_as_wrong']
        out['shelflife_sensitivity_empty_as_wrong'] = Headline(
            metric='forward_minus_reverse_mean_over_samples_empty_as_wrong', value=ew['gap'],
            n=ew['matched_pairs'], direction=1, unit='fraction', null_value=0.0,
            denominator='as the headline, but each available empty sample counts as one wrong line',
            ci=tuple(ew['ci95_pair_bootstrap']) if ew['gap'] is not None else None,
            missing=ew['pairs'] - ew['matched_pairs'], missingness_note='as the headline',
            headline_reliable=reliable, role='sensitivity').to_dict()
        rev = out['directions']['child_query']['best_of_k']
        out['shelflife_sensitivity_reverse_best_of_k'] = self._best_headline(rev, reliable)
        return out

    def _robust_counts(self, rows, out, directions):
        """parse='robust': per-direction parse paths and the _robust.reliable rule; returns the reliability problems."""
        problems, total = [], defaultdict(int)
        for d in directions:
            rs = [r for r in rows if r['item']['direction'] == d]
            sc = [r['score'] for r in rs if r.get('score') is not None]
            c = {'parsed_paper': sum(x['parse_path'] == 'paper' for x in sc),
                 'parsed_robust': sum(x['parse_path'] == 'robust' for x in sc),
                 'refusal_robust': sum(x['parse_path'] == 'refusal' for x in sc),
                 'unparsed': sum(x['parse_path'] is None for x in sc),
                 'truncated_missing': sum(x['missing'] for x in sc),
                 'truncated_scored': sum(x['truncated_scored'] for x in sc),
                 'failed_calls': len(rs) - len(sc)}
            c['missing'] = c['failed_calls'] + c['truncated_missing']
            c['validity_rate'] = (c['parsed_paper'] + c['parsed_robust'] + c['refusal_robust']) / len(rs) if rs else None
            v = out['directions'][d]
            v['parse'] = c
            for k, x in c.items():
                if k != 'validity_rate':
                    total[k] += x
            if not _robust.reliable(len(rs), c['unparsed'], c['missing']):
                problems.append(f'{d}: {c["unparsed"]} unparsed + {c["missing"]} missing of {len(rs)} samples')
            if v['format_only_miss_share'] is None or v['format_only_miss_share'] > _robust.ROBUST_TOLERANCE:
                problems.append(f'{d}.format_only_miss_share={v["format_only_miss_share"]}')
        out['parse'] = dict(total)
        for k in ('parsed_paper', 'parsed_robust', 'unparsed', 'truncated_scored'):
            out[k] = total[k]
        n = sum(len([r for r in rows if r['item']['direction'] == d]) for d in directions)
        out['validity_rate'] = ((total['parsed_paper'] + total['parsed_robust'] + total['refusal_robust']) / n
                                if n else None)
        return problems

    def _best_headline(self, rev, reliable):
        return Headline(
            metric='reverse_best_of_k_accuracy', value=rev['rate'], n=rev['available_pairs'], direction=-1,
            unit='fraction', paper_value=None,
            denominator=('pairs with >=1 available reverse sample; success = any sample has a line passing '
                         'parse_response that starts with the child (GPT-4 33% rule)'),
            ci=tuple(rev['ci95_pair_bootstrap']) if rev['rate'] is not None else None,
            missing=rev['pairs'] - rev['available_pairs'],
            missingness_note='pairs with no available reverse sample are excluded and counted; bounds_with_missing',
            headline_reliable=reliable, role='sensitivity',
            reliability_note='paper value 0.33 is GPT-4 on its own generated pairs (forward 100% by construction)',
        ).to_dict()

    def _gap(self, fwd, rev, key):
        bases = sorted(set(fwd) | set(rev))
        matched = [b for b in bases if fwd.get(b, {}).get(key) is not None and rev.get(b, {}).get(key) is not None]
        diffs = [(b, fwd[b][key] - rev[b][key]) for b in matched]
        mean = lambda xs: sum(v for _, v in xs) / len(xs) if xs else float('nan')
        _, lo, hi = stats.cluster_bootstrap(diffs, lambda x: x[0], mean, n_boot=self.N_BOOT, seed=self.BOOT_SEED)
        return {'pairs': len(bases), 'matched_pairs': len(matched),
                'forward': mean([(b, fwd[b][key]) for b in matched]) if matched else None,
                'reverse': mean([(b, rev[b][key]) for b in matched]) if matched else None,
                'gap': mean(diffs) if diffs else None, 'ci95_pair_bootstrap': [lo, hi] if diffs else None}

    def _paired(self, fwd, rev):
        bases = sorted(set(fwd) | set(rev))
        matched = [b for b in bases if fwd.get(b, {}).get('mean') is not None and rev.get(b, {}).get('mean') is not None]
        cond = [b for b in matched if fwd[b]['any_line']]
        cond_best = [b for b in cond if rev[b]['best'] is not None]
        return {
            'gap_mean_over_samples': self._gap(fwd, rev, 'mean'),
            'gap_mean_over_samples_empty_as_wrong': self._gap(fwd, rev, 'mean_empty_as_wrong'),
            'gap_best_of_k': {'matched_pairs': len(matched),
                              'gap': (sum(fwd[b]['best'] for b in matched) - sum(rev[b]['best'] for b in matched))
                                     / len(matched) if matched else None},
            'reverse_given_forward_correct_DIAGNOSTIC': {
                'condition': 'forward: some sample has a line starting with the labelled parent',
                'pairs': len(cond),
                'reverse_mean_over_samples': sum(rev[b]['mean'] for b in cond) / len(cond) if cond else None,
                'reverse_best_of_k': sum(rev[b]['best'] for b in cond_best) / len(cond_best) if cond_best else None,
                'forward_mean_over_samples': sum(fwd[b]['mean'] for b in cond) / len(cond) if cond else None},
        }

    # --------------------------------------------------------------- targets
    def run_rows(self, run_dir, variant, model=None):
        """Scored rows of a saved shelflife run (same record handling as Experiment.rescore)."""
        from ..replay import model_key, run_records
        variant = self.variant(variant) if isinstance(variant, str) else variant
        items = {x['id']: x for x in self.items(variant)}
        rows = []
        for rec in run_records(run_dir):
            if model is not None and model_key(rec) != model:
                continue
            item = self.record_item(rec, items)
            if item is None:
                raise ValueError(f'record matches no item: {rec.get("case_id")}')
            usable = rec.get('status') in {'ok', 'truncated', 'empty', 'complete'}
            output = self.record_output(rec) if usable else None
            rows.append({'item': item, 'output': output, 'model': model_key(rec), 'repeat': rec.get('repeat', 0),
                         'status': rec.get('status'),
                         'score': None if output is None else self.score(item, output, variant)})
        return rows

    def archived_gpt35_gap(self, merged=False):
        """gpt-3.5-turbo forward - reverse gap from the released per-pair fractions. Unmerged: our pair-level gap code
        (``_gap``) on the 1,513 released rows, i.e. the item set a new chat_both run uses. Merged: the figure's frame."""
        if merged:
            _, p, c = archived_bars('gpt-3.5-turbo')
            return p - c
        rows = read_archived('gpt-3.5-turbo')
        fwd = {f'C141:p{i:04d}': {'mean': p} for i, (_, p, _) in enumerate(rows)}
        rev = {f'C141:p{i:04d}': {'mean': c} for i, (_, _, c) in enumerate(rows)}
        return self._gap(fwd, rev, 'mean')['gap']

    _FIG = ('arXiv v4 Figure fig:experiment_2_results_1 (figures/Experiment_2_figure_1.pdf; bar heights from the '
            'vector content stream, 7.4856 pt per 1%)')
    _FIG_NOTE = ('exact vector bar height (test_c141 re-extracts it from the PDF); tolerance 5e-5 = float/rounding of '
                 'the plotted value. Computed on the notebook-merged 1,573-row frame; ')

    targets = (
        PaperTarget('gpt35_parent_mean_over_10', 0.338907, 5e-5, _FIG, 'gpt-3.5-turbo (Sep 2023, -0613)', 'chat_both',
                    'archived_outputs', compute=lambda exp: archived_bars('gpt-3.5-turbo')[1],
                    note=_FIG_NOTE + 'the unmerged 1,513 released rows give 0.3484'),
        PaperTarget('gpt35_child_mean_over_10', 0.114558, 5e-5, _FIG, 'gpt-3.5-turbo (Sep 2023, -0613)', 'chat_both',
                    'archived_outputs', compute=lambda exp: archived_bars('gpt-3.5-turbo')[2],
                    note=_FIG_NOTE + 'unmerged 0.1191'),
        PaperTarget('gpt35_gap_mean_over_10_figure', 0.224348, 5e-5, _FIG + '; parent bar minus child bar',
                    'gpt-3.5-turbo (Sep 2023, -0613)', 'chat_both', 'archived_outputs',
                    compute=lambda exp: exp.archived_gpt35_gap(merged=True),
                    note='headline metric (forward - reverse mean over samples) on the figure\'s merged frame'),
        PaperTarget('gpt35_gap_mean_over_10_released_rows', GPT35_GAP_RELEASED_ROWS, 5e-5,
                    'released gpt-3.5-turbo_reversal_test_results.csv, 1,513 rows (same protocol as the figure)',
                    'gpt-3.5-turbo (Sep 2023, -0613)', 'chat_both', 'archived_outputs',
                    compute=lambda exp: exp.archived_gpt35_gap(merged=False),
                    note='the comparable anchor for a new chat_both run (its items are the 1,513 released rows); '
                         'computed with the module\'s own pair-level gap code; differs from the figure value 0.2243 '
                         'only through the notebook merge weighting'),
        PaperTarget('n_child_parent_pairs', 1573, 0, 'arXiv v4 Sec. 3.2 para 1; appendix.tex L185',
                    'GPT-4 (dataset construction)', 'chat_both', 'released_data',
                    compute=lambda exp: archived_bars('gpt-3.5-turbo')[0],
                    note='1573 = rows of the notebook\'s merged frame (1509 unique + 2 duplicated pairs x 2^5); the '
                         'released pair file has 1,513 rows, which items() uses'),
        PaperTarget('n_released_pairs', 1513, 0, 'data/celebrity_relations/parent_child_pairs.csv',
                    'GPT-4 (dataset construction)', 'chat_both', 'released_data',
                    compute=lambda exp: len({x['base_id'] for x in exp.items(exp.variant('chat_both'))}),
                    note='released item count used by every variant'),
        PaperTarget('gpt4_forward_identifies_parent', 0.79, 0.005, 'arXiv v4 abstract; Sec. 3.2 para 1',
                    'gpt-4 (Sep 2023 alias)', 'chat_both', 'unverifiable',
                    note='Not a correctness rate: the share of 2,000 temperature-0 parent queries (1,000 celebrities x '
                         'mother/father) that passed parse_response. 1573/2000 = 0.787 fits, but 1573 is the merged '
                         'row count; the released file gives 1513/2000 = 0.757. Raw GPT-4 answers not released.'),
        PaperTarget('gpt4_child_caption', 0.28, 0.005, 'arXiv v4 Figure fig:experiment_2_results_1 caption',
                    'gpt-4 (Sep 2023 alias)', 'chat_both', 'unverifiable',
                    note='"GPT-4 scores 28% on Child": no released output supports it (probably the mean-over-10 '
                         'metric); inconsistent with the 33% best-of-10 in the text'),
    )
