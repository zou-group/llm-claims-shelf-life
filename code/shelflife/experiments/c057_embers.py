"""C057 — Embers of autoregression: shift-cipher decoding (McCoy et al. 2023, arXiv 2309.13638).

Claim: decoding accuracy tracks task and output probability. GPT-4 (gpt-4-0613) decodes rot-1 0.82, rot-3 0.76 but
rot-2 0.02 (Sec 5.1, Fig 5.2), and decodes rot-13 at 0.51 when the answer is a high-probability sentence vs 0.13 when
low-probability (Sec 6.1, Fig 6.1). GPT-3.5 (gpt-3.5-turbo-0613): rot-13 0.21, every other shift 0.00 (Sec 5.1).

Sources: official repo tommccoy1/embers-of-autoregression @ 5dd6c91a. Prompts are the authors' stimuli
(``stimuli/<set>.jsonl`` ``task_instruction % input`` == ``instruction_plus_input``), sent by ``run_openai.py`` as a
single user message with temperature 0.0, top_p 1, stop None; max_tokens 200 per the README commands for the
rot13/shift tasks (``run_openai.py``'s argparse default is 1000, the README overrides it). The authors released every
model output (``logs/<set>_<model>_temp=0.0_n=1.json``), used here as ``archived_outputs`` targets.

Variants
  paper_base     default, faithful. The claim's four sets: shift_2 (rot-2, high-prob sentences), rot13dec_{high,medium,
                 low}prob. Ids ``C057/<set>/<index:03d>/r0`` are the initial-reproduction ids (the '/r0' suffix is a
                 legacy prepare.py repeat tag; repeats are runner repeats here). Used by the pilot, direct-0613 runs and
                 the direct Sonnet pilot.

Scoring (``score``)
  official_repo_correct (primary, ``correct``): the authors' ``evaluation/eval_shift.py`` / ``eval_rot13.py`` loop
      (non-Llama branch): strip one boundary '"' from gt and response, then ``gt in res`` (substring). This is the
      scorer that produced the published numbers: ``evaluation/tsv_rot13.py`` (same rule) feeds the figure tables, and
      ``evaluation/figures/embers_0915_shifts.R`` hard-codes GPT-3.5 rot-13 decoding accuracies 0.21/0.07/0.04
      (high/medium/low), which only the substring rule yields (Appendix C.1 exact match gives 0.21/0.06/0.03).
  paper_exact: Appendix C.1 as written — "We ignored quotation marks at the start and end of the answer, but otherwise
      the model's output had to exactly match the ground truth" (one leading/trailing ASCII '"' removed, nothing else).
      Identical to the official rule on all 2,800 released GPT-4 decode outputs; differs on 2 GPT-3.5 outputs
      (correct sentence after 'Sure, here is the decoded message:'). On modern verbose outputs the two diverge widely.
  extracted_exact (diagnostic only, not a paper metric): initial-reproduction's target-blind answer extractor
      (reproductions/C057/score.py, v3) followed by exact match; guards against the substring rule crediting a correct
      intermediate string that the model later revises.
  Edit distance (paper's secondary metric) is the official Levenshtein on boundary-quote-stripped strings.

Aggregation: per set, accuracy = correct / responses scored (truncated and empty responses are scored as the original
pipeline would score them; calls without a response are missing and reported with bounds). Also reported: accuracy
among status 'ok' only (the prior expanded analysis convention) and paired sentence-family contrasts
(rot13 high − rot2, rot13 high − rot13 low) with a cluster bootstrap over sentence families.

Provider refusals. A call that ends with a content-filter/refusal finish and no content
(OpenRouter ``finish_reason: content_filter`` / ``native_finish_reason: refusal``, Anthropic ``stop_reason: refusal``)
is a *refusal*. The paper-faithful headline (``accuracy``, the unsuffixed contrasts) keeps scoring it as a wrong
answer: the 2023 GPT runs had no such refusals, and the original loop would have retried an API error or crashed on
null content, so there is no original convention to follow. Refusals are strongly condition-dependent for modern
models (Claude Sonnet 5.5 in the expanded run: 75 low / 51 rot-2 / 42 medium / 37 high of 100), so the headline
contrast for such a model mostly measures refusal rate. Therefore every cell also reports ``refusals``,
``refusal_rate`` and per metric ``accuracy_answered`` (refusals excluded) with ``bounds_refusals_unknown``; every
contrast has an ``:answered`` twin restricted to sentence families answered in both conditions; each contrast says
``refusal_confounded``; and the top level has ``refusals`` and ``headline_reliable`` (False when any refusal occurred)
with a ``headline_caveat``. Content-filter finishes that still returned text (7 Claude outputs) are scored normally and
counted as ``content_filter_with_content``.

Robustness. On the panel the floor models (gpt-4o-mini
M015, Mistral Small 3 M049: both arms 0) decode letter by letter ("V -> T, q -> o, ...") and hit the paper's 200-token
cap before writing the sentence (85% / 83% of responses cut off); reasoning models also lose answers at 4,096. Those
zeros are truncation, not decoding failure.
  Scoring option ``parse``: 'paper' (default) or 'robust'. Under 'robust', ``official_repo_correct`` (and ``correct``)
  is the official substring rule on the raw text; only when that fails is the same rule applied to
  ``_robust.normalize``d response and ground truth (think blocks, markdown emphasis, LaTeX wrappers, unicode dashes).
  The paper value stays as ``official_repo_correct_paper``. EVERY response that was cut off (finish_reason 'length' /
  status 'truncated'), whether its text happens to contain the answer or not, and every empty response without a
  provider refusal is ``unparsed``: under 'robust' it is excluded from every accuracy and contrast (counted, like a
  missing call). The treatment does not depend on correctness, so it cannot select on the outcome (the first version
  dropped only wrong cut-off rows, which biased accuracy up).
  Documented exceptions (rows the paper rule scored that robust treats differently): (a) cut-off rows, scored by the
  paper rule (right or wrong), are missing under robust; (b) empty non-refusal rows, wrong under the paper rule, are
  missing; (c) a wrong row whose normalized text contains the normalized truth becomes correct (no panel row). Refusals stay
  refusals (wrong in ``accuracy``, excluded in ``*_answered``). ``headline_reliable`` = ``_robust.reliable(n,
  unparsed + refusals, missing)`` over the cells of the headline contrast (rot13dec high vs low) instead of "no refusal
  at all"; parsed_paper / parsed_robust / unparsed are reported.
  robust_parse  paper_base with parse='robust' (identical requests; offline rescoring of the panel outputs).
  robust        robust_parse + max_tokens floor 2048 (letter-by-letter decoding of a 30-word sentence needs ~1,000
                tokens; 1,024 would still cut the longest low-probability sentences) + min_cap_reasoning 16384 +
                truncated='score': at these caps a cut-off response is a decoding failure, not a cap artefact, so it
                is scored by the official substring rule on its text (almost always wrong) instead of unparsed. On the
                the panel robust run, excluding them flagged 6 of 11 models unreliable (M045 never finished 40% of
                letter-by-letter decodes at 2,048; M183/M196 looped for 16k reasoning tokens) and dropped exactly the
                hardest (low-probability) items. Exception (d) for robust: cut-off rows are scored, as in the paper.
"""
import functools
import hashlib
import json
import math
from collections import Counter, defaultdict

from ..paths import INITIAL, source_repo
from ..program import SingleTurn
from ..prompts import PromptTemplate, Source
from ..registry import Experiment, PaperTarget, PriorRun, Variant, register
from ..replay import legacy_records
from ..sourcecode import load_functions
from .. import stats
from . import _robust

REPO = source_repo('C057', 'embers-of-autoregression')
COMMIT = '5dd6c91ad57cd2d913402b98e4253fda806fbe93'
URL = 'https://github.com/tommccoy1/embers-of-autoregression'
PRIOR = INITIAL / 'reproductions/C057'
RUNS = INITIAL / 'runs'
BASE_SETS = ('shift_2', 'rot13dec_highprob', 'rot13dec_mediumprob', 'rot13dec_lowprob')
ARCHIVED_MODELS = ('gpt-4-0613', 'gpt-3.5-turbo-0613')
USABLE = {'ok', 'truncated', 'empty'}
REFUSAL_FINISH = {'content_filter', 'refusal'}


def is_refusal(output):
    """Provider refusal: content-filter/refusal finish with no returned text."""
    output = output or {}
    return output.get('finish_reason') in REFUSAL_FINISH and not (output.get('content') or '').strip()


def legacy_finish_reason(rec):
    """finish reason of a legacy record (OpenRouter/OpenAI choices[0].finish_reason or Anthropic stop_reason)."""
    resp = rec.get('response') if isinstance(rec.get('response'), dict) else {}
    choices = resp.get('choices') or [{}]
    return rec.get('finish_reason') or choices[0].get('finish_reason') or resp.get('stop_reason')

# Frozen initial-reproduction scorers (sha256 of the file each run snapshotted), for rescoring parity.
PRIOR_SCORERS = {
    'v1': (RUNS / '2026-09-30-C057-pilot/code_snapshot/score.py',
           'e178df8d4b99ccf263d824ab6c1057aea2b95c598a81c8fc61dcec2b1e6c2ba3'),
    'v2': (RUNS / '2026-09-30-C057-expanded/code_snapshot/score.py',
           '2f8217427fa3969162e37ea1582b0740f4392072c7ab19f1ce326b6c30377bf0'),
    'v3': (PRIOR / 'score.py', '96cd1d61978d189b8632d0a931ebd929cbeaf0d372a52de290eff8643e7a5c6f'),
}


@functools.lru_cache(maxsize=None)
def prior_scorer(version):
    path, sha = PRIOR_SCORERS[version]
    if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
        raise RuntimeError(f'{path} changed since it was frozen')
    return load_functions(path, ['score', 'parse_response', 'unwrap', 'boundary_quotes', 'edit_distance'])


@functools.lru_cache(maxsize=None)
def stimuli(name):
    return tuple(json.loads(line) for line in (REPO / 'stimuli' / f'{name}.jsonl').read_text().splitlines() if line.strip())


@functools.lru_cache(maxsize=None)
def archived_log(name, model):
    return json.loads((REPO / 'logs' / f'{name}_{model}_temp=0.0_n=1.json').read_text())


def shift_of(name):
    return int(name.split('_')[1]) if name.startswith('shift_') else 13


def probability_of(name):
    return name.split('_')[-1] if name.startswith('rot13') else 'highprob'


def caesar(text, shift):
    """ASCII Caesar shift preserving case and non-letters (stimulus_generator_shift.py rot_encode semantics)."""
    out = []
    for c in text:
        base = ord('a') if 'a' <= c <= 'z' else ord('A') if 'A' <= c <= 'Z' else None
        out.append(c if base is None else chr(base + (ord(c) - base + shift) % 26))
    return ''.join(out)


def template(name):
    rows = stimuli(name)
    instructions = {r['task_instruction'] for r in rows}
    assert len(instructions) == 1, name
    text = instructions.pop()
    # '%s' is the only %-directive, so literal replacement equals the authors' `task_instruction % input`.
    assert text.count('%') == 1 and '%s' in text, name
    return PromptTemplate(f'user:{name}', text, style='replace', tokens={'%s': 'input'},
                          source=Source(repo=str(REPO), path=f'stimuli/{name}.jsonl',
                                        locator='task_instruction (identical on all rows); instruction_plus_input = task_instruction % input',
                                        commit=COMMIT, url=URL),
                          note='verified byte-equal to instruction_plus_input and to the prompts in the released logs')


# ------------------------------------------------------------------ scorers
def strip_boundary_quotes(text):
    if text.startswith('"'):
        text = text[1:]
    if text.endswith('"'):
        text = text[:-1]
    return text


def paper_exact(response, expected):
    """Appendix C.1: ignore quotation marks at the start and end; otherwise exact match."""
    return strip_boundary_quotes(response) == expected


def official_repo_correct(response, expected):
    """evaluation/eval_rot13.py and eval_shift.py (non-Llama branch). gts are '"' + correct_output + '"' (run_openai.py)."""
    gt = '"' + expected + '"'
    if gt[0] == '"':
        gt = gt[1:]
    if gt[-1] == '"':
        gt = gt[:-1]
    res = response
    if len(res) > 0 and res[0] == '"':
        res = res[1:]
    if len(res) > 0 and res[-1] == '"':
        res = res[:-1]
    return gt in res


def robust_correct(item, output, official, cut_missing=True):
    """parse='robust' fields (see module docstring): fallback correctness on normalized text and the unparsed flag.
    cut_missing=False (variant option truncated='score'): a cut-off row is scored like any other answer."""
    text = output.get('content') if isinstance(output.get('content'), str) else ''
    expected = item['expected']
    path = 'paper' if official else None
    correct = official
    if not official:
        norm = _robust.normalize(text)
        if norm and official_repo_correct(norm, _robust.normalize(expected)):
            correct, path = True, 'robust'
    cut = output.get('finish_reason') == 'length' or output.get('status') == 'truncated'
    # Every cut-off row is unparsed (missing), right or wrong: excluding only the wrong ones selects on the outcome
    # and biases accuracy up (M143 lowprob 0.09 -> 0.47). Empty non-refusal rows likewise.
    reason = 'truncated' if cut and cut_missing else (
        'empty' if not text.strip() and not is_refusal(output) and not cut else None)
    if reason is None and path is None:
        path = 'paper'      # a complete answer the paper rule scores (wrong); refusals keep path 'paper' too
    return {'official_repo_correct_paper': official, 'official_repo_correct': correct, 'correct': correct,
            'unparsed': reason is not None, 'unparse_reason': reason, 'parse_path': None if reason else path,
            'truncated_scored': bool(cut and not cut_missing)}


def levenshtein(a, b):
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def official_edit_distance(response, expected):
    """eval scripts: Levenshtein.distance(gt, res) after stripping one boundary quote from each."""
    return levenshtein(expected, strip_boundary_quotes(response))


def _wilson(k, n):
    return list(stats.wilson(k, n)) if n else None


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
class C057Embers(Experiment):
    claim_id = 'C057'
    title = 'Embers of autoregression: shift-cipher decoding'
    claim = ('LLM accuracy tracks task and output probability: GPT-4 decodes rot-1 0.82 and rot-3 0.76 but rot-2 0.02, '
             'and decodes rot-13 at 0.51 for high-probability vs 0.13 for low-probability output sentences.')
    paper = {'title': 'Embers of Autoregression: Understanding Large Language Models Through the Problem They are '
                      'Trained to Solve', 'authors': 'McCoy, Yao, Friedman, Hardy, Griffiths', 'year': 2023,
             'url': 'https://arxiv.org/abs/2309.13638'}
    source = Source(repo=str(REPO), path='run_openai.py', locator='solve_file', commit=COMMIT, url=URL)
    default_variant = 'paper_base'

    def build_variants(self):
        params = {'temperature': 0.0, 'top_p': 1, 'max_tokens': 200, 'stop': None}
        deviations = (
            'original runner retried failed calls indefinitely (models/openai.py); here a failed call is recorded as '
            'missing and bounded, not retried forever',
            'stimuli control length in GPT-3.5/GPT-4 tokens only; that control does not carry over to other tokenizers',
            'gpt-3.5-turbo-0613 is retired; gpt-4-0613 is reachable only through the direct OpenAI API (OpenRouter '
            '`openai/gpt-4` is an unpinned alias, not the paper snapshot)',
        )
        base = Variant('paper_base', 'Official one-demonstration decode prompts, rot-2 vs rot-13 high/medium/low',
                       prompts={f'user:{s}': template(s) for s in BASE_SETS}, params=params,
                       options={'sets': BASE_SETS, 'id_suffix': ''}, deviations=deviations)
        robust_parse = base.override(
            'robust_parse', options={'parse': 'robust'},
            description='paper_base scored with the robust fallback (identical requests)',
            deviations=('parse=robust: official substring rule re-applied to normalized text only when it fails on the '
                        'raw text; wrong answers that were cut off or empty are unparsed (excluded, counted) instead '
                        'of wrong; reliable when <= 10% of headline rows are unparsed, refused or missing',))
        robust = robust_parse.override(
            'robust', params=_robust.cap_floor(params, 2048), options={'min_cap_reasoning': 16384, 'truncated': 'score'},
            description='robust_parse with a 2,048-token cap floor and min_cap_reasoning 16384 for panel runs; '
                        'responses cut off at that cap are scored (a failure to decode), not dropped',
            deviations=('max_tokens 200 -> 2048 (letter-by-letter decoders exhaust 200); reasoning models get a '
                        '16,384-token cap; the paper\'s 200-token cap also limited GPT-4 to answering directly, so '
                        'step-by-step decoding is now possible',))
        return {v.name: v for v in (base, robust_parse, robust)}

    # ------------------------------------------------------------------ data
    def items(self, variant, limit=None):
        """limit applies per set (same ordered prefix of every set), which keeps sentence families paired."""
        rows = []
        for name in variant.options['sets']:
            shift = shift_of(name)
            for index, row in enumerate(stimuli(name)):
                if limit is not None and index >= limit:
                    break
                assert caesar(row['input'], -shift) == row['correct_output'], (name, index)
                rows.append({'id': f'C057/{name}/{index:03d}/r0{variant.options["id_suffix"]}', 'set': name,
                             'shift': shift,
                             'output_probability': probability_of(name), 'source_index': index,
                             'pair_id': f'C057:sentence{index:03d}', 'input': row['input'],
                             'expected': row['correct_output']})
        return rows

    def program(self, variant):
        def build(item):
            return [{'role': 'user', 'content': variant.prompts[f'user:{item["set"]}'].render(input=item['input'])}]
        cap = variant.options.get('min_cap_reasoning')
        return SingleTurn(build, variant.params, case_extra={'min_cap_reasoning': cap} if cap else None)

    # --------------------------------------------------------------- scoring
    def score(self, item, output, variant, distances=True):
        text = output.get('content')
        if not isinstance(text, str):
            text = ''
        expected = item['expected']
        parsed, method = prior_scorer('v3')['parse_response'](text)
        official = official_repo_correct(text, expected)
        out = {'correct': official, 'official_repo_correct': official, 'paper_exact': paper_exact(text, expected),
               'extracted_exact': parsed == expected, 'parsed_answer': parsed, 'parse_method': method}
        if distances:
            out['edit_distance'] = official_edit_distance(text, expected)
        if variant.options.get('parse', 'paper') == 'robust':
            out.update(robust_correct(item, output, official, _robust.truncation_missing(variant)))
        return out

    METRICS = ('official_repo_correct', 'paper_exact', 'extracted_exact')

    @staticmethod
    def _usable(r, variant):
        """Row enters accuracies/contrasts: scored, usable status and (parse='robust') not unparsed."""
        return bool(r.get('score')) and (r.get('output') or {}).get('status') in USABLE and not (
            variant.options.get('parse', 'paper') == 'robust' and r['score'].get('unparsed'))

    def aggregate(self, rows, variant):
        by_set = defaultdict(list)
        for r in rows:
            by_set[r['item']['set']].append(r)
        conditions = {}
        for name in variant.options['sets']:
            rs = by_set.get(name, [])
            if not rs:
                continue
            statuses = Counter((r.get('output') or {}).get('status') or r.get('status') or 'no_output' for r in rs)
            scored = [r for r in rs if self._usable(r, variant)]
            ok = [r for r in scored if r['output']['status'] == 'ok']
            answered = [r for r in scored if not is_refusal(r['output'])]
            refusals = len(scored) - len(answered)
            missing = len(rs) - len(scored)
            cell = {'n': len(rs), 'scored': len(scored), 'missing': missing, 'statuses': dict(statuses),
                    'refusals': refusals, 'refusal_rate': refusals / len(rs),
                    'content_filter_with_content': sum((r['output'].get('finish_reason') in REFUSAL_FINISH)
                                                       for r in answered)}
            for metric in self.METRICS:
                k = sum(bool(r['score'][metric]) for r in scored)
                k_ok = sum(bool(r['score'][metric]) for r in ok)
                k_ans = sum(bool(r['score'][metric]) for r in answered)
                cell[metric] = {
                    'correct': k, 'accuracy': k / len(scored) if scored else None, 'wilson95': _wilson(k, len(scored)),
                    'bounds_missing_unknown': list(stats.bounds_with_missing(k, len(scored) - k, missing)),
                    'correct_ok': k_ok, 'ok_n': len(ok), 'accuracy_ok': k_ok / len(ok) if ok else None,
                    'bounds_nonok_unknown': list(stats.bounds_with_missing(k_ok, len(ok) - k_ok, len(rs) - len(ok))),
                    'correct_answered': k_ans, 'answered_n': len(answered),
                    'accuracy_answered': k_ans / len(answered) if answered else None,
                    'bounds_refusals_unknown': list(stats.bounds_with_missing(k_ans, len(answered) - k_ans,
                                                                              len(rs) - len(answered)))}
            cell['accuracy'] = cell['official_repo_correct']['accuracy']
            cell['accuracy_answered'] = cell['official_repo_correct']['accuracy_answered']
            conditions[name] = cell
        contrasts = {}
        for hi, lo in (('rot13dec_highprob', 'shift_2'), ('rot13dec_highprob', 'rot13dec_lowprob')):
            if hi in conditions and lo in conditions:
                rates = [conditions[hi]['refusal_rate'], conditions[lo]['refusal_rate']]
                confounded = conditions[hi]['refusals'] + conditions[lo]['refusals'] > 0
                for metric in ('official_repo_correct', 'paper_exact'):
                    for answered_only in (False, True):
                        c = self._contrast(by_set[hi], by_set[lo], metric, answered_only, variant)
                        c.update(refusal_rates=rates, refusal_confounded=confounded and not answered_only)
                        contrasts[f'{hi}-{lo}:{metric}' + (':answered' if answered_only else '')] = c
        refusals = sum(c['refusals'] for c in conditions.values())
        out = {'conditions': conditions, 'contrasts': contrasts, 'refusals': refusals,
               'headline_reliable': refusals == 0}
        if variant.options.get('parse', 'paper') == 'robust':
            return self._robust_summary(out, rows, by_set, variant)
        if refusals:
            out['headline_caveat'] = (
                f'{refusals} provider refusals (content filter, no text) are scored as wrong in accuracy and the '
                'unsuffixed contrasts; refusal rates differ by condition '
                + str({k: round(c['refusal_rate'], 3) for k, c in conditions.items()})
                + '. Use accuracy_answered and the :answered contrasts to separate decoding from refusal.')
        return out

    def _robust_summary(self, out, rows, by_set, variant):
        """parse='robust' counts and the tolerant reliability rule (headline cells rot13dec high/low)."""
        scores = [r['score'] for r in rows if r.get('score') and (r.get('output') or {}).get('status') in USABLE]
        paths = Counter(sc.get('parse_path') for sc in scores)
        head = by_set.get('rot13dec_highprob', []) + by_set.get('rot13dec_lowprob', [])
        n = len(head)
        unparsed = sum(bool(self._scored(r) and r['score'].get('unparsed')) for r in head)
        refused = sum(bool(self._scored(r) and is_refusal(r['output'])) for r in head)
        missing = sum(not self._scored(r) for r in head)
        out.update(parse='robust', parsed_paper=paths.get('paper', 0), parsed_robust=paths.get('robust', 0),
                   unparsed=paths.get(None, 0), unparsed_truncated=sum(sc.get('unparse_reason') == 'truncated'
                                                                       for sc in scores),
                   truncated_scored=sum(bool(sc.get('truncated_scored')) for sc in scores),
                   headline_rows=n, headline_unparsed=unparsed, headline_refusals=refused, headline_missing=missing,
                   validity_rate=(n - unparsed - refused - missing) / n if n else None,
                   headline_reliable=_robust.reliable(n, unparsed + refused, missing))
        out.pop('headline_caveat', None)
        if not out['headline_reliable']:
            out['headline_caveat'] = (f'rot13dec high/low: {unparsed} unparsed (cut off or empty), {refused} refusals, '
                                      f'{missing} missing of {n} rows (> 10%)')
        return out

    @staticmethod
    def _scored(r):
        return bool(r.get('score')) and (r.get('output') or {}).get('status') in USABLE

    def _contrast(self, hi_rows, lo_rows, metric, answered_only=False, variant=None):
        """Paired difference over sentence families (same source_index and repeat), both responses scored (and, with
        ``answered_only``, neither a refusal; with parse='robust', neither unparsed)."""
        robust = variant is not None and variant.options.get('parse', 'paper') == 'robust'

        def table(rs):
            return {(r['item']['source_index'], r.get('repeat', 0)): bool(r['score'][metric]) for r in rs
                    if r.get('score') and (r.get('output') or {}).get('status') in USABLE
                    and not (answered_only and is_refusal(r['output']))
                    and not (robust and r['score'].get('unparsed'))}
        a, b = table(hi_rows), table(lo_rows)
        pairs = [{'cluster': k[0], 'diff': int(a[k]) - int(b[k])} for k in sorted(set(a) & set(b))]
        mean = lambda ps: sum(p['diff'] for p in ps) / len(ps) if ps else float('nan')
        point, lo, hi = stats.cluster_bootstrap(pairs, lambda p: p['cluster'], mean, n_boot=2000, seed=20260930)
        return {'metric': metric, 'paired_n': len(pairs), 'difference': point if pairs else None,
                'ci95_sentence_bootstrap': [lo, hi]}

    # ------------------------------------------------- targets and prior runs
    def rescore_archived(self, model, variant_name):
        """Score the authors' released outputs (logs/) for one original model with this config's scorer."""
        variant = self.variant(variant_name)
        program = self.program(variant)
        rows = []
        for item in self.items(variant):
            log = archived_log(item['set'], model)
            i = item['source_index']
            assert log['prompts'][i] == program.build(item)[0]['content'], item['id']
            assert log['gts'][i] == '"' + item['expected'] + '"', item['id']
            output = {'content': log['res'][i], 'status': 'ok'}
            rows.append({'item': item, 'output': output, 'repeat': 0, 'model': model,
                         'score': self.score(item, output, variant, distances=False)})
        return self.aggregate(rows, variant)

    def prior_rows(self, records, variant_name, model=None):
        """Turn legacy response records into aggregate rows (non-usable statuses keep output=None)."""
        variant = self.variant(variant_name)
        items = {x['id']: x for x in self.items(variant)}
        rows = []
        for rec in records:
            if model is not None and rec.get('model') != model:
                continue
            item = items[rec['case_id']]
            status = rec.get('status')
            output = ({'content': rec.get('content') or '', 'status': status, 'finish_reason': legacy_finish_reason(rec)}
                      if status in USABLE else None)
            rows.append({'item': item, 'output': output, 'repeat': int(rec.get('repeat') or 0), 'model': rec.get('model'),
                         'status': status,
                         'score': self.score(item, output, variant, distances=False) if output else None})
        return rows

    def rescore_prior(self, run, variant_name):
        records = legacy_records(RUNS / run)
        return self.aggregate(self.prior_rows(records, variant_name), self.variant(variant_name))

    targets = (
        PaperTarget('gpt4_rot13_highprob_accuracy', 0.51, 0.005, 'Sec 6.1 text / Fig 6.1', 'gpt-4-0613', 'paper_base',
                    'archived_outputs', compute=lambda e: e.rescore_archived('gpt-4-0613', 'paper_base')['conditions']['rot13dec_highprob']['accuracy']),
        PaperTarget('gpt4_rot13_lowprob_accuracy', 0.13, 0.005, 'Sec 6.1 text / Fig 6.1', 'gpt-4-0613', 'paper_base',
                    'archived_outputs', compute=lambda e: e.rescore_archived('gpt-4-0613', 'paper_base')['conditions']['rot13dec_lowprob']['accuracy']),
        PaperTarget('gpt4_rot13_mediumprob_accuracy', 0.22, 0.005, 'Fig 3.1/6.1 (embers_0915_shifts.R gpt4_dec_acc)',
                    'gpt-4-0613', 'paper_base', 'archived_outputs',
                    compute=lambda e: e.rescore_archived('gpt-4-0613', 'paper_base')['conditions']['rot13dec_mediumprob']['accuracy']),
        PaperTarget('gpt35_rot13_mediumprob_accuracy', 0.07, 0.005, 'Fig 3.1/6.1 (embers_0915_shifts.R gpt35_dec_acc)',
                    'gpt-3.5-turbo-0613', 'paper_base', 'archived_outputs',
                    compute=lambda e: e.rescore_archived('gpt-3.5-turbo-0613', 'paper_base')['conditions']['rot13dec_mediumprob']['accuracy'],
                    note='Appendix C.1 exact match would give 0.06; the published value comes from the substring scorer'),
        PaperTarget('gpt35_rot13_lowprob_accuracy', 0.04, 0.005, 'Fig 3.1/6.1 (embers_0915_shifts.R gpt35_dec_acc)',
                    'gpt-3.5-turbo-0613', 'paper_base', 'archived_outputs',
                    compute=lambda e: e.rescore_archived('gpt-3.5-turbo-0613', 'paper_base')['conditions']['rot13dec_lowprob']['accuracy'],
                    note='Appendix C.1 exact match would give 0.03; the published value comes from the substring scorer'),
        # Fresh calls to the same snapshot: the authors' own duplicate (shift_13 vs rot13dec_highprob, identical prompts)
        # flips 5/100 GPT-4 items (net 1), so a same-snapshot rerun is expected within a few points; 0.05 ≈ 2 SD of that.
        PaperTarget('gpt4_rot2_accuracy_direct_rerun', 0.02, 0.05, 'Sec 5.1', 'gpt-4-0613 (direct OpenAI, 2026 rerun)',
                    'paper_base', 'prior_run',
                    compute=lambda e: e.rescore_prior('2026-09-30-C057-direct0613-full', 'paper_base')['conditions']['shift_2']['accuracy']),
        PaperTarget('gpt4_rot13_highprob_accuracy_direct_rerun', 0.51, 0.05, 'Sec 6.1', 'gpt-4-0613 (direct OpenAI, 2026 rerun)',
                    'paper_base', 'prior_run',
                    compute=lambda e: e.rescore_prior('2026-09-30-C057-direct0613-full', 'paper_base')['conditions']['rot13dec_highprob']['accuracy']),
        PaperTarget('gpt4_rot13_lowprob_accuracy_direct_rerun', 0.13, 0.05, 'Sec 6.1', 'gpt-4-0613 (direct OpenAI, 2026 rerun)',
                    'paper_base', 'prior_run',
                    compute=lambda e: e.rescore_prior('2026-09-30-C057-direct0613-full', 'paper_base')['conditions']['rot13dec_lowprob']['accuracy']),
    )

