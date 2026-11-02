"""C023 — non-uniform random-number / baby-name generation (Zhang, Schwarzschild, Carlini, Kolter & Ippolito 2024,
"Forcing Diffuse Distributions out of Language Models", arXiv 2404.10859).

Claim: instruction-tuned LLMs asked for a random number between 1 and 10 produce highly non-uniform outputs (baseline
entropy 1.16 / 1.07 / 1.88 nats for Gemma-7B-it / Llama-2-13B-chat / Mistral-7B-Instruct-v0.1 vs ln 10 = 2.30;
Coverage-1000 8 / 7 / 9), and random first names collapse similarly (entropy 3.19 / 3.24 / 4.02 vs 6.43 natural,
coverage 129-225 vs 714). Paper v1 Appendix B.1 Table 4 = v2 Table 1.

Protocol in the source (src/generate.py @ 7170ab05): one user turn (the prompt file's ``request`` through the chat
template, no system prompt), do_sample, temperature 1.0, top_p 1.0, top_k None (disabled), max_new_tokens 50, 1,000
generations per prompt (batches of 20, no seed). Generation stats are entropy (nats) and coverage of the empirical
distribution over 1,000 samples.

Variants
  rng0              default. prompts/numbers/rng-0.json, 1,000 independent draws, T=1, top_p=1, top_k=0 (OpenRouter's
                    encoding of 'disabled'), max_tokens 50. Each draw is its own item (ids ``C023-rng-0-draw-00000`` ...)
                    as in every earlier run; repeats=1.
  baby_names        prompts/baby-names/train.json request, same decoding, 1,000 draws. Not run by any prior run.

Scoring (the source has no output parser; its prompt files list the valid target strings)
  strict:   the whole output, outer whitespace stripped, equals a released target string ``{n}``. This equals the
            frozen initial-reproduction/panel ``score.py`` (tested on every saved output).
  semantic: sensitivity rule of the same score.py (bare integer or braced integer with inner whitespace/sign/zeros).
  baby names: whole output ``{Name}`` with Name a single alphabetic token (apostrophes/hyphens allowed). Our rule; the
            source releases only 200 training targets, not the paper's test-time name parser.
Metrics (headline = paper definition)
  ``metrics`` (faithful): paper §2.2 Eq. 3-4 (v2 p.3; v1 identical) define Entropy(p~) = -sum_{y in V*} p~(y) log p~(y)
            over the empirical distribution of the generations y_1..y_N, and Coverage-N = |{y_i}|, the number of unique
            generations. V* is the space of all token sequences, so invalid generations are outcomes too. We use the
            content of every observed generation with outer whitespace stripped (HF ``batch_decode`` gives
            near-canonical strings, API transports add/strip whitespace inconsistently), uncapped coverage; an observed
            empty generation (normal stop) is the outcome ''. Residual ambiguity: footnote 5 says braces were requested
            "to simplify output parsing", but no parser is in the repository and Eq. 3-4 are over generations.
  ``valid_only_metrics`` (diagnostic, NOT the paper's metric): entropy/coverage conditional on strict-valid outputs
            (coverage capped at k), as both earlier analyses (initial-reproduction analyze.py, panel analyze_panel.py)
            computed them; all prior-run parity is checked on this field. ``semantic_metrics``: sensitivity parse.
  Invalid and unknown counts are always reported alongside.
Completion status (both metrics): an output counts as observed if status 'ok', or 'empty' with a normal stop reason
  (initial-reproduction rule). Truncated (finish 'length'; in saved runs almost all are reasoning models whose
  visible content is empty because reasoning consumed the raised cap), errored, or not-submitted calls are unknown,
  i.e. not generations in the paper's sense; they are counted, never silently dropped.
"""
import collections
import json
import math
import re

from ..paths import HELPERS, source_repo
from ..program import SingleTurn
from ..prompts import PromptTemplate, Source
from ..registry import Experiment, PaperTarget, PriorRun, Variant, register
from ..replay import legacy_records
from .. import stats

REPO = source_repo('C023', 'diffuse-distributions')
COMMIT = '7170ab056d790fb3f209a3f5cca2dffe59266996'
URL = 'https://github.com/y0mingzhang/diffuse-distributions'
DRAWS = 1000
STOP_REASONS = {'stop', 'end_turn', 'tool_use', 'refusal', 'content_filter'}
CONDITIONS = {'rng-0': 'prompts/numbers/rng-0.json', 'baby-names': 'prompts/baby-names/train.json'}
K = {'rng-0': 10, 'baby-names': None}

_GEN_SRC = Source(repo=str(REPO), path='src/generate.py', locator='generate (generation_config) / main defaults',
                  commit=COMMIT, url=URL)


def _prompt(condition):
    src = Source(repo=str(REPO), path=CONDITIONS[condition], locator='request', commit=COMMIT, url=URL)
    return PromptTemplate.from_source(condition, src, style='literal', kind='json', json_path=['request'])


def _targets(condition):
    return json.loads((REPO / CONDITIONS[condition]).read_text())['targets']


# ----------------------------------------------------------------------------------------------- parsing
_SEMANTIC = re.compile(r'(?:\{\s*([+-]?[0-9]+)\s*\}|([+-]?[0-9]+))')
_NAME = re.compile(r"\{\s*([A-Za-z][A-Za-z'\-]*)\s*\}")


def parse_number(content, condition):
    """Strict (released target string) and semantic (sensitivity) parse of one RNG output."""
    if not isinstance(content, str):
        return {'strict_valid': False, 'semantic_valid': False, 'value': None, 'parse_status': 'non_text'}
    text = content.strip()
    k = K[condition]
    strict_valid = text in _TARGET_SETS[condition]
    m = _SEMANTIC.fullmatch(text)
    value = int(next(x for x in m.groups() if x is not None)) if m else None
    in_range = value is not None and 1 <= value <= k
    status = ('strict_target' if strict_valid else 'format_violation' if in_range else
              'out_of_range' if value is not None else 'empty' if not text else 'unparsed')
    return {'strict_valid': strict_valid, 'semantic_valid': in_range, 'value': value, 'parse_status': status}


def parse_name(content):
    if not isinstance(content, str):
        return {'strict_valid': False, 'value': None, 'parse_status': 'non_text'}
    text = content.strip()
    m = _NAME.fullmatch(text)
    return {'strict_valid': bool(m), 'value': m.group(1) if m else None,
            'parse_status': 'strict_target' if m else 'empty' if not text else 'unparsed'}


_TARGET_SETS = {'rng-0': frozenset(_targets('rng-0'))}


def observed(output):
    """Initial-reproduction completion rule: ok, or empty with a normal stop reason."""
    if output is None:
        return False
    status = output.get('status', 'ok')
    return status == 'ok' or (status == 'empty' and output.get('finish_reason') in STOP_REASONS)


# ------------------------------------------------------------------------------------------- metrics
def entropy(counts):
    n = sum(counts)
    return -sum(c / n * math.log(c / n) for c in counts if c) if n else None


def generation_outcome(content):
    """Paper Eq. 3-4 outcome of one observed generation: its text with outer whitespace stripped. An observed
    generation with no content (status 'empty', or null content) is the empty generation ''."""
    if content is None:
        return ''
    if not isinstance(content, str):
        raise TypeError(f'non-text generation content: {content!r}')
    return content.strip()


def generation_distribution(outcomes):
    """Paper Eq. 3-4 on the outcomes of all observed generations (open outcome space, uncapped coverage)."""
    d = distribution(outcomes, None)
    if d is not None:
        d['n_generations'] = d.pop('n_valid')
    return d


def distribution(values, k):
    """Paper metrics on a list of valid outcomes. k=None: open outcome space (names)."""
    counter = collections.Counter(values)
    n = len(values)
    if not n:
        return None
    counts = [counter.get(v, 0) for v in range(1, k + 1)] if k else list(counter.values())
    h = max(0.0, entropy(counts))
    mode_count = max(counts)
    if k:  # lowest value wins ties (counts.index(max(counts)) in the panel analysis)
        mode_value = counts.index(mode_count) + 1
    else:  # names: alphabetically first among the most frequent
        mode_value = min(v for v, c in counter.items() if c == mode_count)
    out = {'n_valid': n, 'entropy_nats': h, 'coverage': sum(c > 0 for c in counts),
           'mode_value': mode_value, 'mode_share': mode_count / n, 'counts': dict(sorted(counter.items(), key=str))}
    if k:
        p = [c / n for c in counts]
        out.update({'uniform_entropy': math.log(k), 'kl_emp_to_uniform': math.log(k) - h,
                    'kl_uniform_to_emp_infinite': out['coverage'] < k,
                    'tv_to_uniform': sum(abs(x - 1 / k) for x in p) / 2})
    return out


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
class C023RandomNumbers(Experiment):
    claim_id = 'C023'
    title = 'Non-uniform random number / name generation (diffuse distributions)'
    claim = ('Instruction-tuned LLMs asked to pick a number at random between 1 and 10 produce highly non-uniform '
             'outputs (entropy 1.07-1.88 nats vs 2.30 uniform; coverage 7-9/10 in 1,000 draws); random first names '
             'collapse similarly (entropy 3.19-4.02 vs 6.43 natural).')
    paper = {'title': 'Forcing Diffuse Distributions out of Language Models',
             'authors': 'Zhang, Schwarzschild, Carlini, Kolter, Ippolito', 'year': 2024,
             'url': 'https://arxiv.org/abs/2404.10859'}
    source = _GEN_SRC
    default_variant = 'rng0'

    def build_variants(self):
        params = {'temperature': 1.0, 'top_p': 1.0, 'max_tokens': 50, 'top_k': 0}
        common = ('original sampled 1,000 generations from local HF weights in batches of 20 with no seed; here each '
                  'draw is an independent single API request (one item per draw, repeats=1)',
                  'top_k=None (HF: disabled) is sent as top_k=0, OpenRouter\'s documented "disabled" value',
                  'API chat templates, server weights and server-side sampling are not inspectable',
                  'headline entropy/coverage follow paper Eq. 3-4 over all observed generations (outer whitespace '
                  'stripped; the paper decodes with HF batch_decode); truncated/errored calls are unknown, not '
                  'generations. The strict-valid-conditional metric of the earlier runs is reported separately as '
                  'valid_only_metrics (diagnostic)')
        rng0 = Variant('rng0', 'Released RNG prompt 1-10 (rng-0.json), 1,000 independent draws, T=1, top_p=1, '
                       'top_k disabled, 50 new tokens', prompts={'rng-0': _prompt('rng-0')}, params=params,
                       options={'conditions': ('rng-0',), 'draws': DRAWS}, deviations=common)
        return {
            'rng0': rng0,
            'baby_names': Variant('baby_names', 'Released BabyNames prompt (baby-names/train.json request), 1,000 '
                                  'independent draws, same decoding', prompts={'baby-names': _prompt('baby-names')},
                                  params=params, options={'conditions': ('baby-names',), 'draws': DRAWS},
                                  deviations=common + (
                                      'name parser ({Name}, single alphabetic token) is ours: the repository releases '
                                      'only 200 training targets, not the paper\'s test-time parsing or the natural '
                                      'US-2022 reference sample (714 names, entropy 6.43)',)),
        }

    # ------------------------------------------------------------------------------------------ data
    def items(self, variant, limit=None):
        conditions = variant.options['conditions']
        rows = []
        for draw in range(variant.options['draws']):
            for cond in conditions:
                if limit is not None and len(rows) >= limit:
                    return rows
                rows.append({'id': f'C023-{cond}-draw-{draw:05d}', 'condition': cond, 'k': K[cond],
                             'draw_index': draw, 'diagnostic_block': draw // 100})
        return rows

    def program(self, variant):
        def build(item):
            return [{'role': 'user', 'content': variant.prompts[item['condition']].render()}]
        return SingleTurn(build, variant.params, case_extra={})

    # --------------------------------------------------------------------------------------- scoring
    def score(self, item, output, variant):
        content = (output or {}).get('content')
        if item['condition'] == 'baby-names':
            return parse_name(content)
        return parse_number(content, item['condition'])

    def aggregate(self, rows, variant):
        by_cond = collections.defaultdict(list)
        for r in rows:
            by_cond[r['item']['condition']].append(r)
        out = {}
        for cond, rs in sorted(by_cond.items()):
            k = K[cond]
            valid, semantic, generations, invalid, unknown = [], [], [], 0, 0
            for r in rs:
                s = r.get('score')
                if not observed(r.get('output')) or s is None:
                    unknown += 1
                    continue
                generations.append(generation_outcome((r.get('output') or {}).get('content')))
                if s['strict_valid']:
                    valid.append(s['value'])
                else:
                    invalid += 1
                if s.get('semantic_valid'):
                    semantic.append(s['value'])
            metrics = generation_distribution(generations)
            valid_only = distribution(valid, k)
            row = {'attempted': len(rs), 'n_generations': len(generations), 'n_valid': len(valid),
                   'observed_invalid': invalid, 'unknown': unknown,
                   'strict_valid_fraction_of_observed': len(valid) / (len(valid) + invalid) if valid or invalid else None,
                   'metrics': metrics,
                   'metric_definition': 'paper Eq. 3-4: all observed generations, outer whitespace stripped',
                   'valid_only_metrics': valid_only,
                   'valid_only_metric_definition': 'diagnostic: strict-valid outputs only (earlier runs\' metric)',
                   'semantic_metrics': distribution(semantic, k) if k else None}
            ids = lambda xs: list(range(len(xs)))  # each draw is its own cluster (single prompt per condition)
            if generations:
                h = lambda idx: generation_distribution([generations[i] for i in idx])['entropy_nats']
                _, lo, hi = stats.cluster_bootstrap(ids(generations), lambda i: i, h, n_boot=2000, seed=20261003)
                row['entropy_ci95_draw_bootstrap'] = [lo, hi]
                if metrics['mode_share'] == 1.0:
                    row['mode_share_one_sided95_lower'] = 0.05 ** (1 / len(generations))
            if valid:
                h = lambda idx: distribution([valid[i] for i in idx], k)['entropy_nats']
                _, lo, hi = stats.cluster_bootstrap(ids(valid), lambda i: i, h, n_boot=2000, seed=20261003)
                row['valid_only_entropy_ci95_draw_bootstrap'] = [lo, hi]
                if valid_only['mode_share'] == 1.0:
                    row['valid_only_mode_share_one_sided95_lower'] = 0.05 ** (1 / len(valid))
            out[cond] = row
        return out

    # ------------------------------------------------------------------------------- prior rescoring
    def prior_rows(self, run, model=None):
        """Rows of a saved legacy run in this registry's row format (score recomputed with our scorer)."""
        variant = self.variant('rng0')  # paper parse; scoring reads only the item's condition
        rows = []
        for rec in legacy_records(run):
            if model is not None and (rec.get('record_id') or rec.get('model')) != model:
                continue
            cond = 'rng-0'  # every saved legacy record is an rng-0 draw
            item = {'id': rec['case_id'], 'condition': cond, 'k': K[cond]}
            output = {'content': rec.get('content'), 'status': rec.get('status'),
                      'finish_reason': rec.get('finish_reason')}
            rows.append({'item': item, 'output': output, 'model': rec.get('record_id') or rec.get('model'),
                         'score': self.score(item, output, variant), 'record': rec})
        return rows

    def rescore_prior(self, run, model=None):
        return self.aggregate(self.prior_rows(run, model), self.variant('rng0'))

    targets = tuple(
        PaperTarget(f'{task}_{stat}_{name}', value, tol, 'v1 App. B.1 Table 4 = v2 Table 1', model, variant,
                    'unverifiable', note=('original open-weight checkpoints sampled locally with HF; no generations or '
                                          'count vectors were released and no offline/API path reproduces local '
                                          'full-softmax sampling. Compare with aggregate()[\'metrics\'] (paper Eq. '
                                          '3-4, all generations), not valid_only_metrics. Tolerance: rounding only.'))
        for task, variant, rows in (
            ('rng', 'rng0', (('gemma', 'google/gemma-7b-it', 1.16, 8), ('llama2', 'meta-llama/Llama-2-13b-chat-hf', 1.07, 7),
                             ('mistral', 'mistralai/Mistral-7B-Instruct-v0.1', 1.88, 9))),
            ('babynames', 'baby_names', (('gemma', 'google/gemma-7b-it', 3.19, 170),
                                         ('llama2', 'meta-llama/Llama-2-13b-chat-hf', 3.24, 129),
                                         ('mistral', 'mistralai/Mistral-7B-Instruct-v0.1', 4.02, 225))))
        for name, model, h, cov in rows
        for stat, value, tol in (('entropy', h, 0.005), ('coverage', cov, 0)))


PANEL_DIR = HELPERS / 'C023'
