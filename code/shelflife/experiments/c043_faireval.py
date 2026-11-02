"""C043 — LLM-judge position bias (Wang et al. 2023, "Large Language Models are not Fair Evaluators").

Claim: swapping the order of two responses flips a pairwise judge's verdict for a large fraction of items
(GPT-4 judge, Vicuna-13B vs ChatGPT on the 80 Vicuna-bench questions: conflict 37/80 = 46.25%, Table 2).

Variants
  faireval_vanilla   default, faithful. The authors' own wrapper, FairEval.py ``gen_prompt``'s
                     ``prompt_template`` ('Answer' markers, single newline before the End markers, "[System]\n{prompt}\n"),
                     filled with the Table 1 score-first rubric taken byte-exact from FastChat prompt.jsonl prompt_id 1
                     ``defaults.prompt`` (Table 1 reproduces it word for word; paper v1 Table 1, posted 2023-05-29 with
                     the 37/80 result, shows 'Answer' markers and cites FastChat). System message from FairEval.py,
                     temperature 0, max_tokens 512.

Scoring
  faireval_vanilla: FastChat ``parse_score`` (first line, commas -> spaces, exactly two space-separated numbers),
  which is the parser of the pipeline the template comes from; FastChat maps failures to (-1, -1), which we report
  as unparsed rather than as a tie, and scores outside 1-10 (fused digits like '735 6') are unparsed (range check:
  documented deviation). FastChat splits on single spaces without stripping, so whitespace/markdown artefacts of
  modern APIs ('\n\n8 9', '8 9  ', '8\t7', '**8 7**') are unparsed. Therefore every vanilla score also carries a
  ``fastchat_ws`` verdict (FastChat parse_score after stripping surrounding whitespace and '**'/'`' and collapsing
  whitespace runs on the first line: a documented diagnostic deviation), and ``aggregate`` reports both, the count of
  orders unparsed by FastChat only because of whitespace/markdown, and ``headline_reliable`` (False when such
  format-only failures exceed 1% of scored orders). Read ``conflict_rate`` (paper parser) only together with
  ``headline_reliable``.
Metric: conflict rate = share of question pairs (both orders parsed) whose Vicuna verdict (win/tie/loss) differs
  between Vicuna-first and Vicuna-second; clusters = questions. Failed requests stay in the denominator of the
  missingness bounds (they are rows with no verdict), they are not dropped.
Robustness:
  robust_parse       faireval_vanilla requests unchanged, scoring option parse='robust': the FastChat parser runs first
                     and its verdict stands wherever it parses; only when it returns nothing, (1) the fastchat_ws
                     whitespace/markdown normalisation, then (2) ``robust_score_line``: an explicit pair of in-range
                     scores at the start of the review after _robust.normalize (<think> blocks, markdown, LaTeX removed):
                     a labelled or bare score line ('Scores: 7 8', 'Score line:' + '5 5', '6, 7', '5 and 6') on the
                     first non-empty line (or the line after a bare 'Score(s)/Score line:' label), the sentence
                     'Assistant 1 scores 7 and Assistant 2 scores 3' (or "Assistant 1's score is 4 and Assistant 2's is
                     4") in the first sentence, or labelled 'Assistant N: x' lines. Fused digits ('67', '456 7 8'),
                     prose verdicts without two scores, and out-of-range values stay unparsed. The headline uses the
                     robust verdicts; aggregate reports parsed_paper / parsed_robust / unparsed, the validity rate,
                     the paper-parser conflict rate under 'paper', and headline_reliable = _robust.reliable (<= 10% of
                     orders unparsed or missing). Under parse='robust' the 'whitespace_tolerant' block carries the
                     robust conflicts (robust parsing is a superset of the whitespace normalisation), so the
                     run_minimal_setups collector, which reads whitespace_tolerant.conflict_rate, picks it up unchanged.
  robust vs paper on orders the FastChat parser DID parse: none (tested on every saved the panel order). Note: the
                     whitespace_tolerant block is overwritten with the robust verdicts under parse='robust' only.
  robust             robust_parse + max_tokens floor 1024 (_robust.cap_floor) and min_cap_reasoning 16384: on the panel a
                     reasoning model (M144) spent the policy's 4096-token reasoning cap on 27/160 orders and returned
                     no review text on 23. The score line comes first, so non-reasoning models stop long before the cap.
  score_line_last    (robust follow-up 2026-10-10, design/robust_followup_diagnosis.md) when the first-line rules find
                     nothing, ``robust_score_line_last`` reads an in-range pair standing alone on the LAST line (the
                     review wrote its evaluation first: Claude Sonnet 5.5 did so on 30 of its 35 unparsed the panel
                     orders). A digits-only first line (fused '456 7 8') blocks it. These orders are de facto
                     evidence-first, so aggregate reports them as ``score_line_last`` (included in parsed_robust).
                     Robust vs paper on orders FastChat parsed: still none (the fallback only sees unparsed orders).
FastChat's own pipeline picks a category-specific reviewer prompt (prompt_id 2/3 for coding/math, reviewer.jsonl);
the paper states Table 1 was used for all questions, so all vanilla variants use the general prompt for every item.
"""
import hashlib
import json
import math
import re
from collections import defaultdict

from ..paths import DATA, source_repo
from ..program import SingleTurn
from ..prompts import PromptTemplate, Source
from ..registry import Experiment, PaperTarget, PriorRun, Variant, register
from ..sourcecode import load_functions
from .. import stats
from . import _robust

REPO = source_repo('C043', 'FairEval')
COMMIT = '453c117a69159fd7b7de2f4e0519bac57bee21f4'
FASTCHAT = DATA / 'sources' / 'C043_fastchat'
HUMAN = {'CHATGPT': 'chatgpt', 'VICUNA13B': 'vicuna', 'TIE': 'tie'}
PAIRINGS = {'vicuna_vs_chatgpt': ('chatgpt', 'answer_gpt35.jsonl')}

_REPO_SRC = Source(repo=str(REPO), path='FairEval.py', locator='gen_prompt', commit=COMMIT,
                   url='https://github.com/i-Eval/FairEval')
SYSTEM = PromptTemplate('system', 'You are a helpful and precise assistant for checking the quality of the answer.',
                        style='literal', source=_REPO_SRC, note='gen_prompt sys_prompt; FastChat prompt.jsonl identical')

_FAIREVAL_WRAPPER_SRC = Source(repo=str(REPO), path='FairEval.py', locator='gen_prompt: prompt_template',
                               commit=COMMIT, url='https://github.com/i-Eval/FairEval')
FAIREVAL_WRAPPER = PromptTemplate.from_source(
    'wrapper', _FAIREVAL_WRAPPER_SRC, kind='py_var', var='prompt_template',
    note="the authors' own judge-prompt wrapper ('Answer' markers); {prompt} receives the rubric")


def _fastchat_rubric():
    rows = [json.loads(line) for line in (FASTCHAT / 'prompt.jsonl').read_text().splitlines()]
    return next(r for r in rows if r['prompt_id'] == 1)['defaults']['prompt']


FASTCHAT_RUBRIC = PromptTemplate('rubric', _fastchat_rubric(), style='literal', source=Source(
    repo=str(FASTCHAT), path='prompt.jsonl', locator='prompt_id 1 defaults.prompt (paper Table 1 rubric, word for word)',
    url='https://github.com/lm-sys/FastChat/blob/4960ca702c66b9adaa65945746dba34f8d2c8ddc/fastchat/eval/table/prompt.jsonl'))


def fastchat_parse_score(review):
    """FastChat eval_gpt_review.parse_score, loaded from the snapshot (prints suppressed)."""
    fn = load_functions(FASTCHAT / 'eval_gpt_review.py', ['parse_score'],
                        extra_globals={'logger': type('L', (), {'error': staticmethod(lambda *a, **k: None)})(),
                                       'print': lambda *a, **k: None})['parse_score']
    return fn(review)


def fastchat_ws_parse_score(review):
    """Diagnostic: FastChat parse_score after normalising whitespace/markdown artefacts only (strip the review, drop
    '**' and '`', strip the first line and collapse its whitespace runs to one space). Never splits fused digits."""
    text = review.replace('**', '').replace('`', '').strip()
    first, _, rest = text.partition('\n')
    first = re.sub(r'\s+', ' ', first.strip())
    return fastchat_parse_score(first + ('\n' + rest if rest else ''))


_SCORE = r'(\d+(?:\.\d+)?)(?:\s*/\s*10)?'
_LABEL = r'(?:(?:final\s+|overall\s+)?(?:scores?|ratings?)(?:\s+line)?(?:\s+for\s+assistants?\s+1\s+and\s+2)?' \
         r'|assistant\s+1\s*/\s*assistant\s+2\s+scores?)'
_PAIR_LINE = re.compile(rf'(?i)^(?:{_LABEL}\s*[:=\-]?\s*)?\(?\s*{_SCORE}\s*(?:,|;|\||\band\b|\s)\s*{_SCORE}\s*\)?\s*[.]?$')
_BARE_LABEL = re.compile(rf'(?i)^{_LABEL}\s*[:=\-]?$')
_A = r"assistant\s+{n}(?:'s)?\s*(?:scores?\s*(?:is\s*)?|score\s+is|receives\s*(?:a\s+(?:score\s+of\s+)?)?|gets\s*(?:a\s+)?|:|is)\s*"
_SENTENCE = re.compile(rf"(?i)^{_A.format(n=1)}{_SCORE}\s*(?:,|;)?\s*(?:and|while|,|;)\s*" +
                       rf"(?:{_A.format(n=2)}|assistant\s+2's\s+is\s*|assistant\s+2\s+)" + rf"{_SCORE}\b")
_LABELLED = re.compile(rf'(?im)^\s*(?:the\s+)?(?:score\s+(?:of|for)\s+(?:the\s+)?)?assistant\s+([12])\s*(?:score)?\s*[:=\-]\s*'
                       rf'{_SCORE}\s*[.]?\s*$')


def robust_score_line(text):
    """Robust fallback (parse='robust' only): an explicit pair of in-range scores stated at the start of the review.
    Returns [s1, s2] or None. Never splits fused digits and never infers scores from a prose verdict."""
    t = _robust.normalize(text)
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if not lines:
        return None
    candidates = [lines[0]] + ([lines[1]] if len(lines) > 1 and _BARE_LABEL.match(lines[0]) else [])
    scores = None
    for line in candidates:
        m = _PAIR_LINE.match(line)
        if m:
            scores = [float(m.group(1)), float(m.group(2))]
            break
    if scores is None:
        first_sentence = re.split(r'(?<=[.!?])\s|\n', lines[0], maxsplit=1)[0]
        m = _SENTENCE.match(first_sentence)
        if m:
            scores = [float(m.group(1)), float(m.group(2))]
    if scores is None:
        values = {}
        for m in _LABELLED.finditer(t):
            values.setdefault(int(m.group(1)), float(m.group(2)))
        if len(values) == 2:
            scores = [values[1], values[2]]
    if scores is None or not all(math.isfinite(v) and 1 <= v <= 10 for v in scores):
        return None
    return scores


def robust_score_line_last(text):
    """Second robust fallback (parse='robust' only, after robust_score_line found nothing): the review states its
    evaluation first and the score pair alone on its LAST non-empty line ('Assistant 1 has ... \\n\\n2 4'; Claude
    Sonnet 5.5, 30 of 35 unparsed the panel orders). Same pair grammar and 1-10 range as the first-line rule. Not used
    when the first line is itself a digits-only score attempt (fused digits such as '456 7 8' stay unparsed) or when
    the review has a single line. Such reviews are in effect evidence-first (the FairEval EC order), so aggregate
    counts them separately as ``score_line_last``. Returns [s1, s2] or None."""
    lines = [ln.strip() for ln in _robust.normalize(text).splitlines() if ln.strip()]
    if len(lines) < 2 or re.fullmatch(r'[\d\s.,;:/|()\-]+', lines[0]):
        return None
    m = _PAIR_LINE.match(lines[-1])
    if not m:
        return None
    scores = [float(m.group(1)), float(m.group(2))]
    return scores if all(math.isfinite(v) and 1 <= v <= 10 for v in scores) else None


def _jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


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
_SMOKE = ('pre-launch route/policy smoke test (runs/smoke*, about $0.02) made before the M204 capabilities fix and '
          'policy changes, so its requests no longer replay; not used by the analysis')


def _unsent(*runs):
    """excluded_runs entries for per-model directories that sent nothing: (run, space-separated models)."""
    return {f'{_R2}/{run}/{m}': _NOTHING_SENT for run, models in runs for m in models.split()}


@register
class C043FairEval(Experiment):
    claim_id = 'C043'
    title = 'LLM-judge position bias (FairEval)'
    claim = ('Swapping the order of two responses changes a pairwise LLM judge\'s verdict for a large fraction of '
             'items (GPT-4 conflict 46.25%, ChatGPT 82.5% on Vicuna-13B vs ChatGPT).')
    paper = {'title': 'Large Language Models are not Fair Evaluators', 'authors': 'Wang et al.', 'year': 2023,
             'url': 'https://arxiv.org/abs/2305.17926'}
    source = _REPO_SRC
    default_variant = 'faireval_vanilla'
    VANILLA_FORMAT_THRESHOLD = 0.01   # share of orders unparsed only because of whitespace/markdown -> headline flagged

    def build_variants(self):
        vanilla_dev = ('original judge snapshot unknown (paper used the May-2023 `gpt-4` alias)',
                       'vanilla max_tokens unspecified in the paper; 512 taken from FairEval.py')
        faireval = Variant(
            'faireval_vanilla', "FairEval.py gen_prompt wrapper ('Answer' markers) + Table 1 rubric (FastChat "
                                'prompt_id 1 defaults.prompt), temperature 0',
            prompts={'system': SYSTEM, 'wrapper': FAIREVAL_WRAPPER, 'rubric': FASTCHAT_RUBRIC},
            params={'temperature': 0, 'max_tokens': 512}, strategy='pairwise_both_orders',
            options={'pairing': 'vicuna_vs_chatgpt', 'protocol': 'faireval_vanilla', 'parser': 'fastchat'},
            deviations=vanilla_dev + (
                'the vanilla prompt is assembled, not released as such: FairEval.py wrapper (released for the MEC '
                'rubric) with the Table 1 rubric; paper v1 Table 1 shows the same Answer markers; the v2 PDF shows '
                'lowercase "response" markers',))
        robust_parse = faireval.override(
            'robust_parse', description='faireval_vanilla requests, parse=robust (FastChat parser first, then '
                                        'whitespace/markdown normalisation and an explicit score-line fallback)',
            options={'parse': 'robust'},
            deviations=("scoring: FastChat parse_score first; only when it fails, fastchat_ws and robust_score_line "
                        "(labelled or bare score pair on the first line, 'Assistant 1 scores x and Assistant 2 scores "
                        "y', 'Assistant N: x' lines; in range 1-10); reliability = <=10% of orders unparsed or missing",))
        robust = robust_parse.override(
            'robust', description='robust_parse + max_tokens floor 1024 and reasoning cap 16384 so the review is not '
                                  'cut off on modern (reasoning) models',
            params=_robust.cap_floor(faireval.params), options={'min_cap_reasoning': 16384},
            deviations=('max_tokens 512 -> 1024 (cap floor) and reasoning-model cap 4096 -> 16384 (min_cap_reasoning): '
                        'a reasoning judge spent the 4096 cap on hidden reasoning and returned no review on 23/160 '
                        'orders',))
        return {
            'faireval_vanilla': faireval,
            'robust_parse': robust_parse,
            'robust': robust,
        }

    # ------------------------------------------------------------------ data
    def items(self, variant, limit=None):
        pairing = variant.options['pairing']
        other, other_file = PAIRINGS[pairing]
        questions = _jsonl(REPO / 'question.jsonl')
        vicuna = _jsonl(REPO / 'answer/answer_vicuna-13b.jsonl')
        others = _jsonl(REPO / 'answer' / other_file)
        human = [HUMAN[x] for x in (REPO / 'review/review_gpt35_vicuna-13b_human.txt').read_text().splitlines()]
        assert len(questions) == len(vicuna) == len(others) == len(human) == 80
        rows = []
        for i, (q, v, o) in enumerate(zip(questions, vicuna, others)):
            if limit is not None and i >= limit:
                break
            assert q['question_id'] == v['question_id'] == o['question_id']
            pair_id = f'C043:{pairing}:q{q["question_id"]:03d}'
            for vicuna_first in (True, False):
                a1, a2 = (v, o) if vicuna_first else (o, v)
                order = 'vicuna_first' if vicuna_first else 'vicuna_second'
                rows.append({
                    'id': f'{pair_id}:{variant.options["protocol"]}:{order}', 'pair_id': pair_id, 'order': order,
                    'question_id': q['question_id'], 'category': q['category'], 'question': q['text'],
                    'answer_1': a1['text'], 'answer_2': a2['text'],
                    'assistant_models': ['vicuna', other] if vicuna_first else [other, 'vicuna'],
                    'human_winner': human[i],
                    'candidate_sha256': sorted(hashlib.sha256(a['text'].encode()).hexdigest() for a in (v, o))})
        return rows

    def program(self, variant):
        def build(item):
            system = variant.prompts['system'].render()
            user = variant.prompts['wrapper'].render(question=item['question'], answer_1=item['answer_1'],
                                                     answer_2=item['answer_2'],
                                                     prompt=variant.prompts['rubric'].render())
            return [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]
        cap = variant.options.get('min_cap_reasoning')
        return SingleTurn(build, variant.params, case_extra=None if cap is None else {'min_cap_reasoning': cap})

    # --------------------------------------------------------------- scoring
    def parse(self, text, variant, parser=None):
        parser = parser or variant.options['parser']
        if parser == 'fastchat':
            scores = fastchat_parse_score(text)
        elif parser == 'fastchat_ws':
            scores = fastchat_ws_parse_score(text)
        else:
            raise ValueError(parser)
        if scores == [-1, -1]:
            return None
        scores = [float(s) for s in scores]
        # Range check added (deviation from the original parsers, which accept any float): modern outputs contain
        # fused digits such as '735 6' that the original pipeline would have scored as a 735-point win.
        return scores if all(math.isfinite(v) and 1 <= v <= 10 for v in scores) else None

    @staticmethod
    def _verdict(item, scores):
        identity = dict(zip(item['assistant_models'], scores))
        a, b = item['assistant_models']
        return identity, 'tie' if identity[a] == identity[b] else (a if identity[a] > identity[b] else b)

    def score(self, item, output, variant, parser=None):
        text = output.get('content') or ''
        parser = parser or variant.options['parser']
        out = {'parser': parser, 'call_ok': output.get('status', 'ok') in {'ok', 'truncated', 'empty'}}
        if parser == 'fastchat':
            ws = self.parse(text, variant, 'fastchat_ws')
            out['verdict_ws'] = None if ws is None else self._verdict(item, ws)[1]
        scores = self.parse(text, variant, parser)
        if variant.options.get('parse', 'paper') == 'robust':
            out['verdict_paper'] = None if scores is None else self._verdict(item, scores)[1]
            out['parse_path'] = 'paper' if scores is not None else None
            if scores is None:
                out['format_only_failure'] = parser == 'fastchat' and out.get('verdict_ws') is not None
                scores = (self.parse(text, variant, 'fastchat_ws') if parser == 'fastchat' else None) \
                    or robust_score_line(text)
                if scores is None:
                    scores = robust_score_line_last(text)
                    if scores is not None:
                        out['score_line_last'] = True
                out['parse_path'] = 'robust' if scores is not None else None
        if scores is None:
            out['format_only_failure'] = parser == 'fastchat' and out['verdict_ws'] is not None
            return {**out, 'parsed': False, 'verdict': None}
        identity, verdict = self._verdict(item, scores)
        return {**out, 'parsed': True, 'scores': scores, 'identity_scores': identity, 'verdict': verdict,
                'format_only_failure': out.get('format_only_failure', False),
                'human_agreement': None if item['human_winner'] is None else verdict == item['human_winner']}

    def aggregate(self, rows, variant):
        out = self._conflicts(rows, 'verdict')
        scores = [r.get('score') or {} for r in rows]
        out['orders'] = len(rows)
        out['unparsed_orders'] = sum(s.get('verdict') is None for s in scores)
        out['failed_calls'] = sum(s.get('call_ok') is False or not s for s in scores)
        if any('verdict_ws' in s for s in scores):
            fmt = sum(bool(s.get('format_only_failure')) for s in scores)
            ws = self._conflicts(rows, 'verdict_ws')
            out['whitespace_tolerant'] = {k: ws[k] for k in ('pairs', 'complete_pairs', 'conflicts', 'conflict_rate',
                                                             'conflict_missingness_bounds')}
            out['format_only_unparsed_orders'] = fmt
            out['headline_reliable'] = fmt <= self.VANILLA_FORMAT_THRESHOLD * len(rows)
            if not out['headline_reliable']:
                out['headline_caveat'] = (f'{fmt}/{len(rows)} orders are unparsed by the paper (FastChat) parser only '
                                          'because of whitespace/markdown around the score line; see '
                                          "'whitespace_tolerant' for the conflict rate without these format failures")
        if variant.options.get('parse', 'paper') == 'robust':
            out.pop('headline_caveat', None)
            paths = [s.get('parse_path') for s in scores if s and s.get('call_ok') is not False]
            missing = out['failed_calls']
            unparsed = sum(p is None for p in paths)
            paper = self._conflicts(rows, 'verdict_paper')
            out['paper'] = {k: paper[k] for k in ('pairs', 'complete_pairs', 'conflicts', 'conflict_rate',
                                                  'conflict_missingness_bounds')}
            out['parsed_paper'] = sum(p == 'paper' for p in paths)
            out['parsed_robust'] = sum(p == 'robust' for p in paths)
            out['score_line_last'] = sum(bool(s.get('score_line_last')) for s in scores if s)
            out['unparsed'] = unparsed
            out['missing'] = missing
            out['validity_rate'] = (len(rows) - unparsed - missing) / len(rows) if rows else None
            out['headline_reliable'] = _robust.reliable(len(rows), unparsed, missing)
            out['whitespace_tolerant'] = {k: out[k] for k in ('pairs', 'complete_pairs', 'conflicts', 'conflict_rate',
                                                              'conflict_missingness_bounds')}
            out['whitespace_tolerant_note'] = ("parse='robust': this block repeats the robust headline (FastChat, then "
                                               'whitespace/markdown normalisation, then the explicit score-line '
                                               'fallback) so collectors reading whitespace_tolerant see it')
        return out

    def _conflicts(self, rows, key):
        groups = defaultdict(dict)
        for r in rows:
            s = r.get('score') or {}
            groups[(r['item']['pair_id'], r['repeat'])][r['item']['order']] = s.get(key)
        pairs = []
        for (pair_id, repeat), orders in sorted(groups.items()):
            first, second = orders.get('vicuna_first'), orders.get('vicuna_second')
            pairs.append({'pair_id': pair_id, 'repeat': repeat, 'complete': first is not None and second is not None,
                          'conflict': (first != second) if first is not None and second is not None else None,
                          'vicuna_first_win': first == 'vicuna', 'vicuna_second_win': second == 'vicuna'})
        complete = [p for p in pairs if p['complete']]
        k, n = len(complete), len(pairs)
        conflicts = sum(p['conflict'] for p in complete)
        rate = lambda ps: sum(p['conflict'] for p in ps) / len(ps) if ps else float('nan')
        point, lo, hi = stats.cluster_bootstrap(complete, lambda p: p['pair_id'], rate, n_boot=2000, seed=20260930)
        return {'pairs': n, 'complete_pairs': k, 'conflicts': conflicts,
                'conflict_rate': conflicts / k if k else None, 'conflict_ci95_question_bootstrap': [lo, hi],
                'conflict_missingness_bounds': list(stats.bounds_with_missing(conflicts, k - conflicts, n - k)) if n else None,
                'vicuna_win_rate_first': sum(p['vicuna_first_win'] for p in complete) / k if k else None,
                'vicuna_win_rate_second': sum(p['vicuna_second_win'] for p in complete) / k if k else None}

    # ------------------------------------------------- targets and prior runs
    targets = (
        PaperTarget('chatgpt_conflict_rate_vicuna_vs_chatgpt', 66 / 80, 0.10, 'Table 2', 'gpt-3.5-turbo-0301',
                    'faireval_vanilla', 'unverifiable',
                    note='gpt-3.5-turbo-0301 is retired and its judge outputs were not released'),
    )

