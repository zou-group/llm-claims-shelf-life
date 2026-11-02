"""C055 — Overthinking on easy math (Chen et al., "Do NOT Think That Much for 2+3=? On the Overthinking of o1-Like
LLMs", arXiv 2412.21187v2, ICML 2025).

Claim (claim set, adjudicated): reasoning models overthink easy math. They spend many times more output tokens than
non-reasoning models for similar accuracy, and where the trace is visible their first solution is already correct in
most correct cases, so later solution rounds are redundant.
Paper numbers (v2 Table 1, GSM8K test, 1,319 problems): Llama-3.3-70B-Instruct 92.6% accuracy, 220.3 tokens;
Qwen2.5-Math-72B-Instruct 95.8%, 317.4; QwQ-32B-Preview 94.8%, 772.8 (3.1 solutions); DeepSeek-R1 96.4%, 1056.3
(4.3 solutions). v1 used DeepSeek-R1-Preview; all numbers here are from v2 (snapshot in
data/sources/C055_overthinking_arxiv_v2/).

Sources: official repo galaxyChen/overthinking @ 89da2e51 (data/sources/C055_overthinking). It releases the analysis
pipeline only: solution splitting (Llama-3.3-70B), answer extraction (``utils.extract_answer_boxed``, then a
Llama-3.3-70B extractor when there is no \\boxed{}), the correctness judge (string match, then ``math_equal``, then
Omni-Judge), clustering (GPT-4o-mini) and the efficiency metrics (``compute_metrics.py``). No generation script, prompt
or decoding settings are released. The input format (``data/debug.jsonl``) is the raw question as ``problem``; its
first item is GSM8K test item 0 byte for byte. GSM8K test from openai/grade-school-math (data/sources/C055_gsm8k).

Variants
  gsm8k     default, faithful as far as the release allows: one user turn holding the raw GSM8K test question (no
            system prompt, no instruction), temperature 0 (the paper's Table 2 contrasts "Greedy" responses with
            sampled ones; Sec. 2 states no decoding), max_tokens 16000 (unstated in the paper; 16000 is the lowest
            output limit among the panel's and reproduction routes, deepseek-chat / DeepSeek-R1 on OpenRouter, so no
            route rejects it, and it is far above GSM8K lengths: R1 averages 1,056 tokens). All 1,319 test problems
            in a fixed shuffled order (seed 20261004); ``--limit 100`` (MINIMAL_LIMIT) is a uniform random sample, the
            claim record's powered minimal design (n=100, one sample each).

Program: SingleTurn whose ``forward`` also reads the provider ``usage`` from the ledger's raw record (the core
  ``Completion`` does not expose usage; workaround kept in this module, reported to the lead). Total output tokens =
  OpenAI-style ``usage.completion_tokens`` (OpenAI and OpenRouter count hidden reasoning here, with
  ``completion_tokens_details.reasoning_tokens`` as the reasoning part) or Anthropic ``usage.output_tokens`` (thinking
  included).

Scoring (paper-faithful parts kept, judges replaced by rules; deviations)
  extraction: the authors' ``extract_answer_boxed`` (src/utils.py, loaded from the file). With no \\boxed{} the
              authors call a Llama-3.3-70B extractor; the minimal design has no judge, so we take the last number in
              the response instead (``extraction`` = 'boxed' | 'last_number' | 'none' is recorded per row).
  equality:   the authors' ``judge`` order: exact string match, then ``math_equal``. We load the authors'
              ``normalize`` / ``normalize_answer_string`` / ``is_digit`` and reproduce math_equal's string and numeric
              branches (rel_tol 1e-4, percentage variants). The symbolic (sympy) branch and the Omni-Judge fallback
              are not reproduced: GSM8K references are integers, and the numeric branch decides any numeric
              prediction. A response with no extracted answer is incorrect (authors: "non-finished response: False").
  tokens:     provider usage, not the authors' per-model HF tokenizer. Token counts are therefore in each provider's
              own tokenizer units (paper: the model's own tokenizer on the response text).
Aggregate: accuracy over all intended items (failed calls are reported separately and bound the rate), mean / median /
  geometric-mean output tokens over rows with usage, reasoning tokens where reported, truncation count
  (finish_reason length: tokens are then censored at the cap).
  Observed reasoning: panel routes run at provider-default reasoning, so a "reasoning" model in the panel
  table may not reason here (e.g. gemini-2.5-flash showed 0 reasoning tokens in earlier panel_v2 records). Each
  model is classified from its own rows: ``reasoning_share`` = share of rows with reasoning tokens > 0 (OpenAI/
  OpenRouter ``completion_tokens_details.reasoning_tokens``; Anthropic ``output_tokens_details.thinking_tokens``) or,
  where no count is reported, with visible reasoning text; ``observed_reasoning`` = 'reasoning' (>= 0.5),
  'occasional' (> 0), 'none' (0 with counts reported) or 'unknown'. The table flag is kept as
  ``reasoning_model_table``. reasoning_tokens above the total (seen on R1 via OpenRouter) are clipped to the total
  and counted (``reasoning_exceeds_output``).

Half-life metric (headline) ``paired_log_token_ratio_vs_live_llama33``: mean over GSM8K items scored in
  both runs of ln(output tokens of the model) - ln(output tokens of meta-llama/llama-3.3-70b-instruct on the same
  item, the live reference run, REPRO_MODELS[1], same seed and limit). Item-cluster bootstrap CI; direction +1
  (more tokens = stronger overthinking); null 0; paper_value ln(1056.3 / 220.3) = 1.568 (DeepSeek-R1 vs Llama, Table 1)
  is a ratio of means in each model's own tokenizer: the per-item paired value is not recoverable from the paper.
  The dict also carries the accuracy difference to the reference on the paired items (the "similar accuracy"
  condition), counts of failed calls / missing usage / zero-token rows, observed reasoning, and the secondary
  ``log_ratio_vs_paper_llama_constant`` = ln(mean tokens / 220.3) (the pre-audit headline).
  The reference exists only after the Llama run, so the dict is computed by ``shelflife_from_runs(rows, ref_rows)``.
  ``aggregate`` calls it with the reference rows registered via ``set_reference_rows`` / ``reference_rows_from_run``
  (or the module constant REFERENCE_RUN); without them ``value`` is None and ``status`` = 'needs_reference_run'.
Robustness variants (non-faithful)
  Why the panel rescore gave no value: REFERENCE_RUN is None, so ``aggregate`` on a single-model directory has no
  Llama-3.3-70B rows to pair with (status 'needs_reference_run'); round2's collector registered the reference rows by
  hand (set_reference_rows). The robust variants name the saved reference run in options['reference_run']
  (earlier pipeline C055-gsm8k-L100-repro-r5, meta-llama/llama-3.3-70b-instruct, 100/100 complete, same items and requests
  as ``gsm8k``), so an offline rescore of any model directory yields the paired value. The paper variants are unchanged.
  * ``robust_parse``: ``gsm8k`` requests unchanged, options parse='robust'. The authors' boxed extractor and the
    explicit '#### N' / 'answer is N' / 'Answer: N' phrases run first (``parsed_paper``); only when both fail (the
    module's last-number stand-in would apply) the answer is the last number of the last **bold** / \\mathbf{} span
    that is not a step header, else the number after the last answer marker, else _robust.extract_number
    (``parsed_robust``). Modern models bold the final answer and then add a check or caveat with other numbers, which
    the last-number stand-in misread (e.g. M206 15 of 96 such rows). Truncated rows without a paper answer and rows
    without any number are ``unparsed``. Accuracy (and the "similar accuracy" difference vs the reference) use these
    answers; the token-ratio headline is unchanged by parsing. headline_reliable = _robust.reliable(n, unparsed,
    failed calls + rows without usage), also copied into the shelflife dict.
    Where robust differs from paper on rows the paper scorer DID score (documented exceptions): (a) rows where the
    module's last-number stand-in produced the answer are re-read by the fallback above (panel: 92 rows flip
    ``correct``, 88 of them false -> true; accuracy is only the "similar accuracy" gate, the token-ratio headline does
    not depend on it); (b) truncated rows without a boxed / explicit answer are unparsed (paper: last number). (c)
    The GSM8K fallback opts into _robust.extract_number(where='marker_then_last'): the answer is by construction the
    final number of a worked solution.
  * ``robust``: robust_parse with max_tokens 16000 -> 4096 for non-reasoning routes (routes with a 4k output limit
    reject 16000; no the panel non-reasoning answer came near 4096) and min_cap_reasoning 16000 through
    program.case_extra, so reasoning models keep the paper's 16000 budget. Token counts of non-reasoning models
    cut at 4096 would be censored; such rows are counted as truncated. Option truncated='score'
    (_robust.truncation_missing): at the paper's own 16000-token budget (4096 for non-reasoning routes) a row cut off
    before a boxed / explicit answer failed to answer, so it is scored WRONG (no answer; the mid-reasoning number is
    still not taken), not unparsed; counted as ``truncated_scored``. For robust, exception (b) therefore reads: such
    rows are wrong (paper: last number) instead of unparsed. robust_parse keeps truncated='missing'.
Not in the minimal design: the solution-round analysis (">92% first round correct", #solutions, outcome/process
  efficiency) needs the Llama-3.3-70B segmentation and clustering judges on visible traces. ASDIV and MATH500 are
  not registered.
"""
import json
import math
import random
import re
from math import isclose
from pathlib import Path

from ..paths import DATA, PROJECT
from ..program import SingleTurn
from ..prompts import PromptTemplate, Source
from ..registry import Experiment, PaperTarget, Variant, register
from ..sourcecode import load_functions
from .. import models as M
from .. import stats
from . import _robust

REPO = DATA / 'sources' / 'C055_overthinking'
COMMIT = '89da2e51fd96ec4da2f0fca7ed285baa1a7bfb53'
URL = 'https://github.com/galaxyChen/overthinking'
GSM8K = DATA / 'sources' / 'C055_gsm8k' / 'test.jsonl'
ARXIV = DATA / 'sources' / 'C055_overthinking_arxiv_v2' / 'main.tex'
SEED = 20261004
LLAMA_REF_TOKENS = 220.3     # Table 1, GSM8K, Llama-3.3-70B-Instruct #Token
R1_TOKENS = 1056.3           # Table 1, GSM8K, DeepSeek-R1 #Token
REFERENCE_MODEL = 'meta-llama/llama-3.3-70b-instruct'
REFERENCE_RUN = None         # set to the Llama-3.3-70B reproduction run dir (relative to the project) once it exists

RUN_VARIANT = 'gsm8k'
# saved live reference used by the robust variants (offline pairing); the paper variants keep REFERENCE_RUN
ROBUST_MAX_TOKENS = 4096        # robust: non-reasoning routes (4k output limits reject 16000)
ROBUST_MIN_CAP_REASONING = 16000  # robust: reasoning models keep the paper's 16000 budget
ROBUST_REFERENCE_RUN = 'experiments/run_minimal_setups/runs/earlier/runs/C055-gsm8k-L100-repro-r5/meta-llama_llama-3.3-70b-instruct'
MINIMAL_LIMIT = 100
REPRO_MODELS = ['deepseek/deepseek-r1', 'meta-llama/llama-3.3-70b-instruct']
REPRO_LIMIT = 100

SRC = Source(repo=str(REPO), path='data/debug.jsonl', locator='"problem" field: the raw question is the whole input',
             commit=COMMIT, url=URL)
USER = PromptTemplate('user', '{problem}', source=SRC,
                      note='no generation prompt is released; the authors\' input format is the raw problem text')


# ------------------------------------------------------------------------------------- authors' functions
def _authors():
    if not hasattr(_authors, 'cache'):
        fns = load_functions(REPO / 'src' / 'utils.py', ['extract_answer_boxed'])
        names = ['normalize', 'normalize_answer_string', 'is_digit', '_fix_fracs', '_str_is_int', '_str_to_int',
                 '_inject_implicit_mixed_number', '_strip_properly_formatted_commas', '_remove_right_units',
                 '_process_and_or_inside_text', '_remove_left_and_right', '_fix_sqrt', '_fix_interval',
                 '_inject_implicit_mixed_fraction']
        fns.update(load_functions(REPO / 'src' / 'openmathinst_utils.py', names))
        _authors.cache = fns
    return _authors.cache


def extract_answer_boxed(text):
    return _authors()['extract_answer_boxed'](text)


_LAST_NUMBER = re.compile(r'-?\d[\d,]*(?:\.\d+)?')
_ANSWER_PHRASES = [re.compile(r'####\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)'),
                   re.compile(r'(?:final\s+)?answer\s*(?:is|:)\s*:?\s*\**\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)', re.IGNORECASE)]


def extract_answer(text):
    """(answer, how). Authors' boxed extractor; else the last '#### N' / 'answer is N' / 'Answer: N'; else the last
    number. The two fallbacks stand in for the authors' Llama-3.3-70B extractor (no judge in the minimal design)."""
    text = text or ''
    boxed = extract_answer_boxed(text)
    if boxed is not None:
        return boxed, 'boxed'
    for pat in _ANSWER_PHRASES:
        found = pat.findall(text)
        if found:
            return found[-1].rstrip('.'), 'answer_phrase'
    numbers = _LAST_NUMBER.findall(text)
    if numbers:
        return numbers[-1].rstrip('.'), 'last_number'
    return None, 'none'


_BOLD = re.compile(r'\*\*([^*\n]{1,200}?)\*\*|\\mathbf\{((?:[^{}]|\{[^{}]*\})*)\}')
_STEP_HEADER = re.compile(r'^\s*(?:step\s*\d+|\d+[.)])', re.I)
_MARKER = re.compile(r'final\s+answer|the\s+answer\s+is|answer\s*:|answer\s+is', re.I)


def _fmt(x):
    return None if x is None else (str(int(x)) if float(x).is_integer() else repr(float(x)))


def _bold_numbers(text):
    spans = [a or b for a, b in _BOLD.findall(text)]
    return [s for s in spans if not s.strip().endswith(':') and not _STEP_HEADER.match(s)
            and _robust.extract_number(s, 'last') is not None]


def robust_answer(text):
    """Fallback (parse='robust'; only when the boxed extractor and the explicit answer phrases found nothing): the
    number after the last answer marker (a bold number in that line first); else the last number of the last bold /
    \\mathbf{} span that is not a step header ('**3. Total:**'); else _robust.extract_number. Returns a string."""
    text = text or ''
    marks = list(_MARKER.finditer(text))
    if marks:
        seg = text[marks[-1].end():][:300].lstrip('*: ')
        first = seg.split('\n\n')[0]
        seg = first if first.strip() else seg
        b = _bold_numbers(seg)
        if b:
            return _fmt(_robust.extract_number(b[-1], 'last'))
        n = _robust.extract_number(seg, 'first')
        if n is not None:
            return _fmt(n)
    b = _bold_numbers(text)
    if b:
        return _fmt(_robust.extract_number(b[-1], 'last'))
    # GSM8K answers are by construction the final number of a worked solution (the authors' own stand-in reads the
    # last number), so the permissive last-number fallback is opted into explicitly here.
    return _fmt(_robust.extract_number(text, where='marker_then_last'))


def _strip_prefix(answer):
    """Post-audit tolerance: drop a leading 'x =' and LaTeX-escaped dollar signs before the authors' normalisation."""
    answer = re.sub(r'^\s*[a-zA-Z]\s*=\s*', '', answer)
    return answer.replace('\\$', '$')


def rule_equal(prediction, reference):
    """math_equal (openmathinst_utils.py) steps 0-1: normalisation, string comparison, numeric comparison with
    rel_tol 1e-4 and percentage variants. The sympy symbolic step is not reproduced (returns False there)."""
    f = _authors()
    prediction = f['normalize_answer_string'](f['normalize'](prediction))
    reference = f['normalize_answer_string'](f['normalize'](reference))
    if isinstance(prediction, str) and len(prediction) > 1000:
        prediction = prediction[:1000]
    if isinstance(prediction, str) and isinstance(reference, str):
        if prediction.strip().lower() == reference.strip().lower():
            return True
        if prediction.replace(' ', '') == reference.replace(' ', ''):
            return True
    try:
        if f['is_digit'](prediction)[0] and f['is_digit'](reference)[0]:
            p, r = f['is_digit'](prediction)[1], f['is_digit'](reference)[1]
            return any(isclose(x, p, rel_tol=1e-4) for x in (r / 100, r, r * 100))
    except Exception:
        pass
    return False


def judge_correct(extracted, expected):
    """The authors' extract_and_eval.judge ordering, with rule correctness in place of the Omni-Judge fallback."""
    if extracted is None or expected is None:
        return False
    if extracted.strip() == expected.strip():
        return True
    try:
        return bool(rule_equal(extracted, expected) or rule_equal(_strip_prefix(extracted), expected))
    except Exception:   # authors: any exception in math_equal -> rule_correctness False
        return False


# ------------------------------------------------------------------------------------- usage-aware program
def usage_tokens(response):
    """Output/reasoning/prompt tokens from a provider response (OpenAI/OpenRouter or Anthropic shape). A null count
    stays None (missing), never 0. reasoning_tokens above the total are clipped to it and flagged."""
    u = (response or {}).get('usage') or {}
    if 'completion_tokens' in u:
        total, details, prompt = u.get('completion_tokens'), u.get('completion_tokens_details') or {}, u.get('prompt_tokens')
        reasoning = details.get('reasoning_tokens')
    elif 'output_tokens' in u:
        total, details, prompt = u.get('output_tokens'), u.get('output_tokens_details') or {}, u.get('input_tokens')
        reasoning = details.get('thinking_tokens', details.get('reasoning_tokens'))
    else:
        return {'output_tokens': None, 'reasoning_tokens': None, 'prompt_tokens': None, 'reasoning_exceeds_output': False}
    exceeds = total is not None and reasoning is not None and reasoning > total
    return {'output_tokens': total, 'reasoning_tokens': min(reasoning, total) if exceeds else reasoning,
            'prompt_tokens': prompt, 'reasoning_exceeds_output': exceeds}


class UsageSingleTurn(SingleTurn):
    """SingleTurn that also returns the provider's token usage, read from the ledger's raw record of this call."""
    name = 'single_turn_with_usage'

    def forward(self, item, lm):
        c = lm(self.build(item), self.params_for(item), step='main')
        usage = usage_tokens(None)
        run_dir = getattr(lm.client, 'run_dir', None)
        if run_dir is not None and c.cache_key:
            raw = Path(run_dir) / 'raw' / f'{c.cache_key}.json'
            if raw.exists():
                usage = usage_tokens(json.loads(raw.read_text()).get('response'))
        return {'content': c.content, 'reasoning': c.reasoning, 'finish_reason': c.finish_reason, 'status': c.status,
                **usage}


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def observed_reasoning(scored):
    """Reasoning behaviour observed in a model's own rows (not the panel table flag)."""
    counted = [r for r in scored if r['score'].get('reasoning_tokens') is not None]
    texted = [r for r in scored if r['score'].get('reasoning_tokens') is None]
    flags = [r['score']['reasoning_tokens'] > 0 for r in counted]
    flags += [r['score'].get('reasoning_text', False) for r in texted if r['score'].get('reasoning_text')]
    basis = len(counted) + sum(bool(r['score'].get('reasoning_text')) for r in texted)
    share = sum(flags) / len(scored) if scored and basis else None
    label = ('unknown' if share is None else 'reasoning' if share >= 0.5 else 'occasional' if share > 0 else 'none')
    return {'reasoning_share': share, 'observed_reasoning': label,
            'mean_reasoning_tokens_when_reasoning': _mean([r['score']['reasoning_tokens'] for r in counted
                                                           if r['score']['reasoning_tokens'] > 0])}


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
class C055Overthinking(Experiment):
    claim_id = 'C055'
    claim_set_key = 'C055'
    maturity = 'ready'
    title = 'Overthinking on easy math (reasoning-model output tokens on GSM8K)'
    claim = ('Reasoning models overthink easy math: they spend many times more output tokens than non-reasoning models '
             'for similar accuracy, and (where the trace is visible) their first solution is already correct in most '
             'correct cases, so later solution rounds are redundant.')
    paper = {'title': 'Do NOT Think That Much for 2+3=? On the Overthinking of o1-Like LLMs',
             'authors': 'Chen, Xu, Liang, He, Pang, Yu, Song, Liu, Zhou, Zhang, Wang, Tu, Mi, Yu', 'year': 2024,
             'url': 'https://arxiv.org/abs/2412.21187v2'}
    source = SRC
    default_variant = 'gsm8k'

    def build_variants(self):
        gsm8k = Variant(
            'gsm8k', 'GSM8K test, raw question as the only user turn, greedy, one sample per problem',
            prompts={'user': USER}, params={'temperature': 0, 'max_tokens': 16000}, repeats=1,
            options={'dataset': 'gsm8k_test', 'order_seed': SEED},
            deviations=(
                'generation prompt, system prompt and decoding are not released or stated: raw question as the only '
                'user turn (the authors\' input format), temperature 0 (Table 2 "Greedy"), max_tokens 16000',
                'hosted APIs instead of local weights; QwQ-32B-Preview is not served (not in the OpenRouter catalogue)',
                'tokens are provider usage counts (hidden reasoning included), not the model\'s HF tokenizer on the text',
                'no Llama-3.3-70B answer extractor and no Omni-Judge: last-number fallback and rule equality only',
                'minimal design: a seeded uniform sample of 100 of the 1,319 problems (--limit 100); ASDIV, MATH500 and '
                'the solution-round analysis are not run',
            ))
        # Robustness variants: robust_parse rescores saved gsm8k outputs (identical requests) and pairs
        # them with the saved live Llama reference; robust (new panel runs) needs no request change.
        rparse = gsm8k.override(
            'robust_parse', description='gsm8k requests, parse=robust (paper extraction first, bold/marker number '
            'fallback instead of the last-number stand-in), paired with the saved live Llama-3.3-70B reference',
            options={'parse': 'robust', 'reference_run': ROBUST_REFERENCE_RUN},
            deviations=('scoring: boxed / explicit answer phrase first; otherwise the bold or marked final number '
                        '(_robust fallback) instead of the last number; truncated rows without an answer unparsed; '
                        'reliability = <=10% unparsed or missing rows',
                        'paired reference: the saved earlier pipeline repro-r5 Llama-3.3-70B run (100/100 complete)'))
        robust = rparse.override('robust', description='robust_parse for new panel runs: max_tokens 4096 for '
                                 'non-reasoning routes, min_cap_reasoning 16000 for reasoning models',
                                 params={'max_tokens': ROBUST_MAX_TOKENS},
                                 options={'min_cap_reasoning': ROBUST_MIN_CAP_REASONING, 'truncated': 'score'},
                                 deviations=('max_tokens 16000 -> 4096 (routes with 4k output limits reject 16000); '
                                             'reasoning models keep 16000 via min_cap_reasoning',
                                             "truncated='score': a row cut off before a boxed / explicit answer is "
                                             'wrong (no answer within the paper budget), not unparsed'))
        return {'gsm8k': gsm8k, 'robust_parse': rparse, 'robust': robust}

    def items(self, variant, limit=None):
        rows = []
        for idx, line in enumerate(GSM8K.read_text().splitlines()):
            d = json.loads(line)
            answer = d['answer'].split('####')[-1].strip().replace(',', '')
            rows.append({'id': f'C055:gsm8k:test-{idx:04d}', 'dataset': 'gsm8k', 'index': idx,
                         'problem': d['question'], 'expected_answer': answer})
        random.Random(variant.options['order_seed']).shuffle(rows)
        return rows[:limit] if limit else rows

    def program(self, variant):
        extra = ({'min_cap_reasoning': variant.options['min_cap_reasoning']}
                 if variant.options.get('min_cap_reasoning') else None)
        return UsageSingleTurn(lambda item: [{'role': 'user', 'content': variant.prompts['user'].render(
            problem=item['problem'])}], variant.params, case_extra=extra)

    def score(self, item, output, variant):
        # The paper scores the whole response (R1's thinking + answer). Some OpenRouter providers (Novita, R1, seen in
        # round2 repro-r5: 42/100 rows) return content=None and put the whole response, final answer included, in
        # `reasoning`. Scoring content only then marks correct answers wrong. Fallback (reproduction pass): when content is
        # empty, extract from the reasoning text; `answer_source` records which text was used.
        content = (output or {}).get('content') or ''
        source = 'content'
        if not content.strip() and ((output or {}).get('reasoning') or '').strip():
            content, source = output['reasoning'], 'reasoning_fallback'
        text = content
        extracted, how = extract_answer(text)
        tokens = (output or {}).get('output_tokens')
        truncated = ((output or {}).get('finish_reason') in {'length', 'max_tokens'}
                     or (output or {}).get('status') == 'truncated')
        extra = {}
        if variant.options.get('parse', 'paper') == 'robust':
            if how in ('boxed', 'answer_phrase'):
                path = 'paper'
            else:
                extracted, path = _robust.fallback(None, robust_answer, text)
                how = 'robust' if path else 'none'
                if truncated:          # cut off before a final answer: a mid-reasoning number is not the answer
                    extracted, path, how = None, None, 'none'
                    if not _robust.truncation_missing(variant):
                        # truncated='score' (robust, 16000-token budget): no final answer within the budget is a
                        # wrong answer (judge_correct(None) is False), not unparsed
                        path = 'paper'
                        extra['truncated_scored'] = True
            extra.update({'parse_path': path, 'unparsed': path is None})
        return {'extracted_answer': extracted, 'extraction': how, 'boxed': how == 'boxed', 'answer_source': source,
                **extra,
                'correct': judge_correct(extracted, item['expected_answer']),
                'output_tokens': tokens, 'reasoning_tokens': (output or {}).get('reasoning_tokens'),
                'reasoning_text': bool(((output or {}).get('reasoning') or '').strip()),
                'reasoning_exceeds_output': bool((output or {}).get('reasoning_exceeds_output')),
                'truncated': (output or {}).get('finish_reason') in {'length', 'max_tokens'}
                             or (output or {}).get('status') == 'truncated'}

    def aggregate(self, rows, variant):
        scored = [r for r in rows if r.get('score') is not None]
        acc = stats.rate_summary(rows, lambda r: r['score']['correct'], seed=SEED)
        with_tokens = [r for r in scored if r['score']['output_tokens'] is not None]
        toks = [r['score']['output_tokens'] for r in with_tokens]
        reasoning = [r['score']['reasoning_tokens'] for r in with_tokens if r['score']['reasoning_tokens'] is not None]
        point, lo, hi = self._mean_tokens_ci(with_tokens)
        model = rows[0].get('model') if rows else None
        entry = (M.MODELS.get(model) or M.PANEL.get(model)) if model else None
        out = {
            'n': len(rows), 'scored': len(scored), 'failed_calls': len(rows) - len(scored),
            'accuracy': acc['rate'], 'accuracy_ci95': acc['ci95'], 'accuracy_bounds_with_missing': acc['bounds_with_missing'],
            'boxed_share': _mean([r['score']['boxed'] for r in scored]),
            'no_answer': sum(r['score']['extraction'] == 'none' for r in scored),
            'with_usage': len(with_tokens), 'missing_usage': len(scored) - len(with_tokens),
            'zero_token_rows': sum(t == 0 for t in toks),
            'mean_output_tokens': point, 'mean_output_tokens_ci95': [lo, hi],
            'median_output_tokens': sorted(toks)[len(toks) // 2] if toks else None,
            'geomean_output_tokens': math.exp(_mean([math.log(max(t, 1)) for t in toks])) if toks else None,
            'mean_reasoning_tokens': _mean(reasoning), 'with_reasoning_count': len(reasoning),
            'reasoning_exceeds_output': sum(r['score'].get('reasoning_exceeds_output', False) for r in scored),
            'truncated': sum(r['score']['truncated'] for r in scored),
            'reasoning_model_table': entry[4] if entry and len(entry) > 4 else None,
            'token_ratio_vs_paper_llama33': point / LLAMA_REF_TOKENS if point else None,
            **observed_reasoning(scored),
        }
        ref = self.reference_rows()
        robust = variant.options.get('parse', 'paper') == 'robust'
        if ref is None and variant.options.get('reference_run'):
            ref = self.robust_reference_rows(variant)
        out['shelflife'] = self.shelflife_from_runs(rows, ref)
        if robust:
            for k, path in (('parsed_paper', 'paper'), ('parsed_robust', 'robust'), ('unparsed', None)):
                out[k] = sum(r['score'].get('parse_path') == path for r in scored)
            out['truncated_scored'] = sum(bool(r['score'].get('truncated_scored')) for r in scored)
            out['validity_rate'] = (out['parsed_paper'] + out['parsed_robust']) / len(rows) if rows else None
            missing = out['failed_calls'] + out['missing_usage']
            out['headline_reliable'] = _robust.reliable(len(rows), out['unparsed'], missing)
            out['shelflife'].update(headline_reliable=out['headline_reliable'], validity_rate=out['validity_rate'],
                                   reference_run=variant.options.get('reference_run') if ref is not None else None)
        return out

    _robust_ref_cache = {}

    def robust_reference_rows(self, variant):
        """The saved live Llama-3.3-70B reference (options['reference_run']) scored with this variant; cached."""
        key = (variant.options['reference_run'], variant.name)
        if key not in self._robust_ref_cache:
            path = PROJECT / variant.options['reference_run']
            self._robust_ref_cache[key] = (self.reference_rows_from_run(path, variant=variant)
                                           if (path / 'responses.jsonl').exists() else None)
        return self._robust_ref_cache[key]

    @staticmethod
    def _mean_tokens_ci(with_tokens):
        mean_tokens = lambda rs: _mean([r['score']['output_tokens'] for r in rs])
        return stats.cluster_bootstrap(with_tokens, lambda r: r['item']['id'], mean_tokens, n_boot=2000, seed=SEED)

    # ------------------------------------------------------------------------------- live reference (Llama)
    _reference = None

    def set_reference_rows(self, rows):
        """Register the reference model's scored rows (e.g. from reference_rows_from_run); None clears it."""
        type(self)._reference = rows

    def reference_rows(self):
        if self._reference is None and REFERENCE_RUN is not None:
            self.set_reference_rows(self.reference_rows_from_run(PROJECT / REFERENCE_RUN))
        return self._reference

    def reference_rows_from_run(self, run_dir, model=REFERENCE_MODEL, variant=None):
        """Scored rows of one model from a saved run (runner records carry the program output incl. usage)."""
        from ..replay import model_key, run_records
        variant = self.variant(variant) if isinstance(variant, str) or variant is None else variant
        items = {x['id']: x for x in self.items(variant)}
        rows = []
        for rec in run_records(run_dir):
            if model_key(rec) != model:
                continue
            item = self.record_item(rec, items)
            usable = rec.get('status') in {'ok', 'truncated', 'empty', 'complete'}
            output = self.record_output(rec) if usable else None
            rows.append({'item': item, 'output': output, 'model': model, 'repeat': rec.get('repeat', 0),
                         'score': None if output is None else self.score(item, output, variant)})
        return rows

    def shelflife_from_runs(self, rows, ref_rows=None):
        """The headline dict: paired per-item log token ratio vs the live Llama-3.3-70B reference on the same items."""
        scored = [r for r in rows if r.get('score') is not None]
        with_tokens = [r for r in scored if r['score']['output_tokens'] is not None]
        point = self._mean_tokens_ci(with_tokens)[0]
        hl = {'metric': 'paired_log_token_ratio_vs_live_llama33', 'value': None, 'ci': None, 'n': 0,
              'direction': +1, 'null_value': 0.0, 'paper_value': math.log(R1_TOKENS / LLAMA_REF_TOKENS),
              'paper_value_note': ('ln of the ratio of Table 1 means (DeepSeek-R1 1056.3 / Llama-3.3-70B 220.3, each '
                                   'in its own tokenizer); the per-item paired value is not recoverable from the paper'),
              'definition': ('mean over GSM8K test items scored in both runs of ln(total output tokens of the model, '
                             'provider usage incl. hidden reasoning) - ln(tokens of meta-llama/llama-3.3-70b-instruct on '
                             'the same item); item bootstrap CI'),
              'reference_model': REFERENCE_MODEL, 'failed_calls': len(rows) - len(scored),
              'missing_usage': len(scored) - len(with_tokens),
              'accuracy': _mean([r['score']['correct'] for r in scored]),
              'log_ratio_vs_paper_llama_constant': math.log(point / LLAMA_REF_TOKENS) if point else None,
              **observed_reasoning(scored)}
        if not ref_rows:
            hl['status'] = 'needs_reference_run'
            return hl
        paired = self.paired_log_ratio(rows, ref_rows)
        hl.update(value=paired['mean_log_ratio'] if paired['n_pairs'] else None,
                  ci=paired['ci95'] if paired['n_pairs'] else None, n=paired['n_pairs'],
                  accuracy_difference_vs_reference=paired['accuracy_difference'],
                  zero_token_pairs=paired['zero_token_pairs'], status='ok' if paired['n_pairs'] else 'no_pairs',
                  ratio_of_means_vs_live_reference=paired['ratio_of_means'])
        return hl

    @staticmethod
    def paired_log_ratio(rows, reference_rows, n_boot=2000, seed=SEED):
        """Per-item ln(tokens_model) - ln(tokens_reference) over items both runs scored with usage (the record's
        paired primary contrast). Zero counts are floored at 1 token and counted. Returns mean, item bootstrap CI,
        n, the accuracy difference on the same items, and the ratio of means on them."""
        usable = lambda r: r.get('score') is not None and r['score']['output_tokens'] is not None
        ref = {r['item']['id']: r['score'] for r in reference_rows if usable(r)}
        pairs = [(r['item']['id'], r['score']['output_tokens'], ref[r['item']['id']]['output_tokens'],
                  int(r['score']['correct']) - int(ref[r['item']['id']]['correct']))
                 for r in rows if usable(r) and r['item']['id'] in ref]
        mean = lambda ps: _mean([math.log(max(p[1], 1)) - math.log(max(p[2], 1)) for p in ps])
        point, lo, hi = stats.cluster_bootstrap(pairs, lambda p: p[0], mean, n_boot=n_boot, seed=seed)
        return {'mean_log_ratio': point, 'ci95': [lo, hi], 'n_pairs': len(pairs),
                'zero_token_pairs': sum(p[1] == 0 or p[2] == 0 for p in pairs),
                'accuracy_difference': _mean([p[3] for p in pairs]),
                'ratio_of_means': (_mean([p[1] for p in pairs]) / _mean([p[2] for p in pairs])
                                   if pairs and _mean([p[2] for p in pairs]) else None)}

    # ------------------------------------------------------------------------------------ offline anchors
    @staticmethod
    def authors_debug_rows():
        """The authors' released debug outputs with their pipeline's rule/final correctness."""
        path = REPO / 'data' / 'ot_tmp' / 'debug.jsonl.split.eval.jsonl'
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def debug_rule_agreement(self):
        """Share of the authors' debug items where our boxed extraction + rule equality equals their rule_correctness."""
        rows = self.authors_debug_rows()
        agree = 0
        for r in rows:
            extracted = extract_answer_boxed(r['response'] if isinstance(r['response'], str) else r['response'][0])
            try:
                ours = bool(rule_equal(extracted, r['expected_answer'])) if extracted is not None else False
            except Exception:
                ours = False
            agree += ours == r['rule_correctness'] and extracted == r['extracted_answer']
        return agree / len(rows)

    targets = (
        PaperTarget('gsm8k_test_size', 1319, 0, 'arXiv 2412.21187v2 Sec. 2.1 ("The test set includes 1,319 problems")',
                    'n/a', 'gsm8k', 'released_data', compute=lambda exp: len(exp.items(exp.variant('gsm8k'))),
                    note='anchors data loading only'),
        PaperTarget('authors_debug_rule_correctness_agreement', 1.0, 0.0, 'official repo data/ot_tmp/'
                    'debug.jsonl.split.eval.jsonl (released pipeline output on 5 debug responses)', 'QwQ-32B-Preview',
                    'gsm8k', 'archived_outputs', compute=lambda exp: exp.debug_rule_agreement(),
                    note='Our extraction (authors\' extract_answer_boxed) and rule equality reproduce the authors\' '
                         'extracted_answer and rule_correctness on every released debug output (incl. a GSM8K item, a '
                         'MATH item the rule marks wrong and the LLM judge right, and a wrong answer). Not a Table 1 '
                         'number: the authors released no full outputs.'),
        PaperTarget('r1_gsm8k_mean_tokens', R1_TOKENS, 150, 'arXiv 2412.21187v2 Table 1 (GSM8K, DeepSeek-R1 #Token)',
                    'deepseek/deepseek-r1', 'gsm8k', 'live',
                    note='Live on deepseek/deepseek-r1 via OpenRouter (usage is already in R1\'s tokenizer). '
                         'Tolerance ~2 SE of a 100-item mean (SD ~600-700 tokens). A larger gap should be recorded as a '
                         'known discrepancy naming the unreleased prompt/decoding and the serving stack.'),
        PaperTarget('llama33_gsm8k_mean_tokens', LLAMA_REF_TOKENS, 60, 'arXiv 2412.21187v2 Table 1 (GSM8K, '
                    'Llama-3.3-70B-Instruct #Token)', 'meta-llama/llama-3.3-70b-instruct', 'gsm8k', 'live',
                    note='Live; same tolerance reasoning.'),
        PaperTarget('r1_gsm8k_accuracy', 0.964, 0.05, 'arXiv 2412.21187v2 Table 1 (GSM8K, DeepSeek-R1 Accuracy 96.4)',
                    'deepseek/deepseek-r1', 'gsm8k', 'live',
                    note='Live; our rule scoring lacks the Omni-Judge fallback, so accuracy may be slightly lower.'),
        PaperTarget('qwq_gsm8k_mean_tokens', 772.8, 200, 'arXiv 2412.21187v2 Table 1 (GSM8K, QwQ-32B-Preview #Token)',
                    'QwQ-32B-Preview', 'gsm8k', 'unverifiable',
                    note='QwQ-32B-Preview is not served by any reachable route; no full outputs released.'),
    )
    # reproductions (G06): R1 greedy rerun on the first 50 seeded items (served by Novita).
