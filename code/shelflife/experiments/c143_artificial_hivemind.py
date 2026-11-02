"""C143 — Artificial Hivemind: within- and cross-model response homogeneity on Infinity-Chat100.

Paper: Jiang, Chai, Li, Liu, Fok, Dziri, Tsvetkov, Sap, Albalak, Choi, "Artificial Hivemind: The Open-Ended Homogeneity of
Language Models (and Beyond)", arXiv 2510.22954 v2 (camera-ready, 2026-10-06; NeurIPS 2025 D&B oral). Snapshot:
code/shelflife/data/sources/C143_hivemind_arxiv_v2/. Sec. 3 (sections/3_lack_of_gen_diversity.tex), App. A
(sections/appendix/repetition_analysis.tex), App. tables within_model_similarity_full / across_model_similarity_full.

Claim: 50 samples per query (top-p 0.9, T 1.0, max 2048 tokens) on the 100 Infinity-Chat100 queries are highly similar
within a model (text-embedding-3-small cosine): "in 79% of cases, the average similarity exceeds 0.8" (25 main models,
Sec. 3 line 25), and across models ("ranges from 71% to 82%", Sec. 3 line 42; port-computed table off-diagonal mean 0.759, not printed).
Original design: run only models released after the paper
(mid/late 2025) and compare with the paper's printed numbers; do not rerun the paper's models. On 2026-10-19 the
we later filled all missing panel runs; top_p_r20_panel_completion includes the five originally omitted
panel models while preserving the earlier variants and printed reference series.

Sources
  Queries: HF liweijiang/infinite-chats-eval @ e6177df (code/shelflife/data/sources/C143_infinity_chat_hf/), column
  ``query``, 100 rows, sent verbatim as one user turn with no system prompt (App. A mentions none). Ids C143:qNNN follow
  the HF row order (the authors align queries across models by file order, ``zip``).
  Scorer: the authors' analysis scripts (repo liweijiang/artificial-hivemind @ ff3b875, scripts only; helper modules
  src.utils.main_utils / src.adhoc.constants and every generation/embedding file are unreleased). Re-implemented in
  _c143_similarity.py and tested against the same formulas on literal fixtures.
  Targets: the paper's LaTeX tables, parsed to derived/ CSVs (79-model within histograms, 79x79 cross matrix).
  The paper's responses and embeddings are NOT public, so no target checks our scorer on the paper's outputs.

Variants
  top_p           default, faithful: App. A top-p protocol (T 1.0, top_p 0.9, max_tokens 2048, 50 samples = repeats 50).
  top_p_r20       NOT faithful (cost-reduced): 20 samples per query (190 pairs). The protocol we chose for
                  the new-model run (2026-10-09). Per-query means stay unbiased but noisier; the paper's own paraphrase
                  experiment (App. A.4) used 20 samples.
  top_p_r20_panel_completion  Same decoding and 20 repeats as top_p_r20; fills the five
                  original-paper model slots added later.

Forced sampling deviation: GPT-6 Luna (M204), Claude Sonnet 5.5 (M206) and GPT-6.1 Sol (M207) reject temperature and
top_p (data/model_capabilities.json, data/panel_capabilities.json); policy panel_v2 omits them, so these models
sample at provider defaults (T = 1, top_p = 1) with provider-default reasoning, and the runner raises the output cap
to >= 4096 for reasoning models. ``shelflife plan`` shows the adaptation. Direction of the effect is not known: top_p 1
widens sampling, but default reasoning may make answers converge. The paper's own four OpenAI reasoning models (o1,
o1-mini, o1-preview, o3-mini; App. table rows only, not in Table full_model_list) could not have used top_p 0.9 either
(fixed T = 1 / top_p = 1 APIs, hidden reasoning; whether the 2048 cap counted reasoning tokens is unknown). They are
the decoding-matched comparators: within share > 0.8 64/78/73/69% (mean 71.0%, vs 79.08% for the main 25), intra-group
cross mean 0.774. ``compare_to_paper`` reports both reference groups (audit M1); their gap to the main 25 is
confounded with model family.

Duplicates: a (model, item, repeat) seen twice (same run passed twice, overlapping reruns) or a repeat index outside the
variant's range raises in ``query_pools`` instead of being pooled (a duplicate text adds a cosine-1 pair). A --limit or
partial run is aggregated against the variant's full 100-query item set, so it is never headline-reliable.

Embeddings (outside the runner): score() is pure and records the sha256 of each response's exact text. The embedding
step is experiments/run_minimal_setups/runs/added-b/scripts/embed_hivemind.py (paid; runs only with --execute and --max-usd),
which writes .npz files (sha256 -> text-embedding-3-small vector, default 1536 dims, whole response, no truncation or
``dimensions`` override: the paper names only the model) into the cache directory ``variant.options['embedding_cache']``
(override: env SHELFLIFE_C143_EMBEDDING_CACHE). aggregate() reads that cache offline. Responses without a cached vector
are counted as ``missing_embedding``, never dropped silently.

Metric contract (per model, aggregate)
  unit: query (100). Per query: mean cosine over the distinct pairs of its embedded responses (self-pairs excluded:
  main_analysis_within_model.py, triu k=1). Defined for >= 2 embedded responses.
  headline: share of queries with defined mean >= 0.8 = the paper's two top histogram bins [0.8, 0.9) + [0.9, 1.0]
  (np.histogram(bins=10, range=(0, 1))); the paper's "> 0.8" is this bin rule. Denominator: queries with a defined mean
  (the authors' scripts skip queries without embeddings: ``if "embeddings" not in d: continue``). Bounds count every
  undefined query as below / above 0.8. CI: bootstrap over queries.
  Failed calls (output None), empty or whitespace-only outputs (the OpenAI embeddings API rejects empty input; the
  paper's handling is not documented) and responses missing from the cache are excluded from that query's pool and
  counted. Refusals are embedded like any other text (the authors' scripts embed every response; no refusal filter);
  ``refusal_like`` is a crude prefix diagnostic count only. Truncated outputs (finish_reason length) are embedded on
  their text and counted. Reasoning traces are never embedded (content only).
  headline_reliable is False if any intended response is unavailable for embedding (failed, empty, missing vector,
  absent row) or any query is undefined.
  Also reported: the paper histogram (percent per bin), mean of the per-query means (Schaeffer et al.'s c-bar), and the
  self-pairs-included value (the paper's cross-matrix diagonal convention, 1/n + (1 - 1/n) * mean, ~ +0.003 at n = 50
  for means near 0.86; at n = 20 the shift is ~ +0.007).
  ``compare_to_paper`` places the within-model share in the paper's main-25 and reasoning-4 distributions.
  Cross-model similarity is not computed for our runs; the paper's cross-model numbers are checked as targets only.

Interpretation caveat (Schaeffer et al. 2026, arXiv 2609.33936, snapshot C143_schaeffer2026_reeval_arxiv_v1): the
paper's reference null (unrelated prompts, 0.1-0.2) is undemanding; same-prompt responses with genuinely different ideas
already have median cosine 0.70-0.72 and 20-32% of such pairs exceed 0.8. A high share > 0.8 therefore does not by
itself show conceptual collapse; no human baseline exists. Reported as ``reliability_note``, not corrected.
"""
import functools
from dataclasses import replace
import json
import os
import re
from collections import defaultdict

import numpy as np

from ..paths import DATA, PROJECT
from ..program import SingleTurn
from ..prompts import PromptTemplate, Source
from ..registry import Experiment, PaperTarget, Variant, register
from ..results import Headline
from .. import stats
from . import _c143_similarity as sim

REPO = PROJECT / 'code/shelflife/data/sources/C143_hivemind'
COMMIT = 'ff3b875669db2c6d8bb9f3970fb383013d936d60'
HF = DATA / 'sources' / 'C143_infinity_chat_hf'
QUERIES = HF / 'infinite-chats-eval' / 'train.parquet'
PAPER = DATA / 'sources' / 'C143_hivemind_arxiv_v2'
DERIVED = PAPER / 'derived'
EMBEDDING_MODEL = 'text-embedding-3-small'
DEFAULT_CACHE = 'experiments/run_minimal_setups/runs/added-b/data/hivemind/embeddings/text-embedding-3-small'
THRESHOLD = 0.8
N_QUERIES = 100

SRC = Source(repo=str(REPO), path='src/model_calibration_analysis/main_analysis_within_model.py',
             locator='get_pairwise_similarity / main_model_similarity / main_heatmap_transposed', commit=COMMIT,
             url='https://github.com/liweijiang/artificial-hivemind')
QUERY_SRC = Source(repo=str(HF), path='infinite-chats-eval/train.parquet', locator="column 'query'",
                   url='https://huggingface.co/datasets/liweijiang/infinite-chats-eval',
                   note='HF revision e6177df454f82f6f0ed140d74b8b4844264b40fe; the query is the whole user turn')
USER = PromptTemplate('user', '{query}', source=QUERY_SRC,
                      note='identity template: the released query verbatim, no system prompt (App. A names none)')
REFUSAL_PREFIX = re.compile(r"^\s*(I'm sorry|I am sorry|I can't|I cannot|I can’t|I won't|I will not|As an AI)", re.I)


@functools.lru_cache(maxsize=None)
def load_queries():
    import pandas as pd
    return tuple(pd.read_parquet(QUERIES)['query'].tolist())


# --------------------------------------------------------------------------------------------- paper reference data
@functools.lru_cache(maxsize=None)
def paper_tables():
    """(within histogram DataFrame [percent], cross matrix DataFrame [percent], main-25 names) from derived/."""
    import pandas as pd
    within = pd.read_csv(DERIVED / 'paper_within_model_hist_79.csv', index_col=0)
    cross = pd.read_csv(DERIVED / 'paper_across_model_similarity_79x79.csv', index_col=0)
    main25 = json.loads((DERIVED / 'paper_main25_models.json').read_text())['main25']
    return within, cross, tuple(main25)


# The paper's four OpenAI reasoning models (App. table within_model_similarity_full l.73-76; absent from Table
# full_model_list and the text). Their APIs fix T = 1 / top_p = 1 and reason by default, so they were sampled under the
# same forced adaptation as our target models: the decoding-matched comparators (audit M1).
REASONING4 = ('o1-2024-12-17', 'o1-mini-2024-09-12', 'o1-preview-2024-09-12', 'o3-mini-2025-01-31')


def paper_within_share(models):
    """{model: % of queries with mean >= 0.8} (top two bins) from the paper's within table."""
    within, _, _ = paper_tables()
    return {m: float(within.loc[m, '0.9-1.0'] + within.loc[m, '0.8-0.9']) for m in models}


def paper_main25_within_share():
    return paper_within_share(paper_tables()[2])


def paper_reasoning4_within_share():
    return paper_within_share(REASONING4)


def paper_cross_pairs(group, other=None):
    """{(a, b): mean cosine (fraction)}: distinct pairs within ``group``, or all group x other pairs."""
    _, cross, _ = paper_tables()
    if other is None:
        return {(a, b): float(cross.loc[a, b]) / 100 for i, a in enumerate(group) for b in group[i + 1:]}
    return {(a, b): float(cross.loc[a, b]) / 100 for a in group for b in other}


def paper_main25_cross_offdiag():
    """{(a, b): mean cosine (fraction)} for the 300 distinct main-25 pairs."""
    return paper_cross_pairs(paper_tables()[2])


def paper_reference_groups():
    """Within-share reference distributions (fractions) for the main 25 (top_p 0.9, no reasoning) and the
    decoding-matched reasoning-4 group (provider-default sampling + hidden reasoning)."""
    return {'main25': [v / 100 for v in paper_main25_within_share().values()],
            'reasoning4': [v / 100 for v in paper_reasoning4_within_share().values()]}


def _percentile_rank(value, reference):
    ref = np.asarray(list(reference), dtype=float)
    return float((ref < value).mean() * 100 + (ref == value).mean() * 50)


def compare_to_paper(within_share):
    """Place new within-model shares ({model: fraction >= 0.8}) in the paper's reference distributions (main 25 and
    the decoding-matched reasoning-4 = o1, o1-mini, o1-preview, o3-mini)."""
    refs = paper_reference_groups()

    def place(v):
        d = {}
        for g, ref in refs.items():
            d[f'percentile_in_paper_{g}'] = _percentile_rank(v, ref)
            d[f'minus_paper_{g}_mean'] = v - float(np.mean(ref))
        return d

    return {m: None if v is None else {'share_ge_0.8': v, **place(v)} for m, v in within_share.items()}


# ----------------------------------------------------------------------------------------------- pooled embeddings
def cache_path(variant):
    path = os.environ.get('SHELFLIFE_C143_EMBEDDING_CACHE') or variant.options['embedding_cache']
    return path if os.path.isabs(path) else str(PROJECT / path)


def _pool_status(row, vectors):
    """('embedded', vec) | ('failed'|'empty'|'missing_embedding', None) for one runner row."""
    score = row.get('score')
    if row.get('output') is None or score is None:
        return 'failed', None
    if score['empty']:
        return 'empty', None
    vec = vectors.get(score['sha256'])
    return ('embedded', vec) if vec is not None else ('missing_embedding', None)


def query_pools(rows, vectors, repeats=None):
    """{item id: {'vectors': [...], counts by status, 'truncated', 'refusal_like', 'repeats_present'}} for ONE model.

    Raises ValueError on a duplicate (item id, repeat) (e.g. the same run directory passed twice, or overlapping
    reruns: a duplicated text would add a cosine-1 pair) and, when ``repeats`` is given, on a repeat index outside
    0..repeats-1 (a run made with a different --repeats than the variant). Nothing is clamped (audit M2)."""
    pools = defaultdict(lambda: {'vectors': [], 'rows': 0, 'embedded': 0, 'failed': 0, 'empty': 0,
                                 'missing_embedding': 0, 'truncated': 0, 'refusal_like': 0, 'repeats_present': set()})
    models = {r.get('model') for r in rows}
    if len(models) > 1:
        raise ValueError(f'query_pools takes one model\'s rows, got {sorted(map(str, models))}')
    for r in sorted(rows, key=lambda r: (r['item']['id'], int(r.get('repeat') or 0))):
        p = pools[r['item']['id']]
        rep = int(r.get('repeat') or 0)
        if rep in p['repeats_present']:
            raise ValueError(f'duplicate row for (item {r["item"]["id"]}, repeat {rep}): pass each run once and do not '
                             'merge overlapping runs')
        if repeats is not None and not 0 <= rep < repeats:
            raise ValueError(f'repeat {rep} outside 0..{repeats - 1} for item {r["item"]["id"]}: aggregate with the '
                             'variant whose repeats match the run')
        p['repeats_present'].add(rep)
        p['rows'] += 1
        status, vec = _pool_status(r, vectors)
        p[status] += 1
        if vec is not None:
            p['vectors'].append(vec)
        s = r.get('score') or {}
        p['truncated'] += bool(s.get('truncated'))
        p['refusal_like'] += bool(s.get('refusal_like'))
    return dict(pools)


def within_summary(pools, repeats, threshold=THRESHOLD, expected_ids=None):
    """Per-query means and the paper's within-model statistics from ``query_pools`` output. ``expected_ids``: the
    intended query ids (default: the pooled ids); queries without rows count as undefined with ``repeats`` absent rows."""
    expected = sorted(pools) if expected_ids is None else sorted(set(expected_ids))
    unknown = sorted(set(pools) - set(expected))
    if unknown:
        raise ValueError(f'{len(unknown)} pooled query ids are not intended items: {unknown[:3]}')
    empty_pool = {'vectors': [], 'rows': 0, 'embedded': 0, 'failed': 0, 'empty': 0, 'missing_embedding': 0,
                  'truncated': 0, 'refusal_like': 0, 'repeats_present': set()}
    per_query = []
    for qid in expected:
        p = pools.get(qid, empty_pool)
        E = np.asarray(p['vectors'])
        m = sim.within_pairs_mean(E) if p['embedded'] >= 2 else None
        per_query.append({'id': qid, 'mean': m, 'n_embedded': p['embedded'], 'embedded': p['embedded'],
                          'se_jackknife': sim.jackknife_pairs_se(E) if p['embedded'] >= 3 else None,
                          'mean_with_self_pairs': None if m is None else sim.with_self_pairs(m, p['embedded']),
                          'absent': repeats - len(p['repeats_present']),
                          **{k: p[k] for k in ('rows', 'failed', 'empty', 'missing_embedding', 'truncated',
                                               'refusal_like')}})
    defined = [q for q in per_query if q['mean'] is not None]
    means = [q['mean'] for q in defined]
    n_q = len(per_query)
    k = sum(m >= threshold for m in means)
    undefined = n_q - len(defined)
    share = k / len(defined) if defined else None
    _, lo, hi = (stats.cluster_bootstrap(defined, lambda q: q['id'], lambda qs: sum(q['mean'] >= threshold for q in qs) / len(qs),
                                         n_boot=2000, seed=20261009) if defined else (None, None, None))
    tot = lambda key: sum(q[key] for q in per_query)
    ses = [q['se_jackknife'] for q in defined if q['se_jackknife'] is not None]
    near = sum(abs(q['mean'] - threshold) <= 2 * q['se_jackknife'] for q in defined if q['se_jackknife'] is not None)
    return {
        'queries_intended': n_q, 'queries_with_rows': sum(q['rows'] > 0 for q in per_query),
        'queries_defined': len(defined),
        'queries_undefined': undefined, 'queries_full_pool': sum(q['n_embedded'] == repeats for q in per_query),
        'share_ge_threshold': share, 'k_ge_threshold': k, 'threshold': threshold,
        'share_ci95_query_bootstrap': [lo, hi] if defined else None,
        'share_bounds_with_undefined': list(stats.bounds_with_missing(k, len(defined) - k, undefined)) if n_q else None,
        'per_query_se_jackknife_median': float(np.median(ses)) if ses else None,
        'queries_within_2se_of_threshold': near,
        'histogram_percent': sim.paper_histogram(means)['percent'],
        'histogram_out_of_range': sim.paper_histogram(means)['out_of_range'],
        'mean_of_query_means': float(np.mean(means)) if means else None,
        'mean_of_query_means_with_self_pairs': float(np.mean([q['mean_with_self_pairs'] for q in defined])) if defined else None,
        'responses_intended': n_q * repeats, 'responses_embedded': tot('embedded'), 'rows': tot('rows'),
        'failed_calls': tot('failed'), 'empty_outputs': tot('empty'), 'missing_embedding': tot('missing_embedding'),
        'absent_rows': tot('absent'), 'truncated_outputs': tot('truncated'), 'refusal_like_outputs': tot('refusal_like'),
        'per_query': per_query,
    }


@register
class C143ArtificialHivemind(Experiment):
    claim_id = 'C143'
    maturity = 'ready'
    title = 'Artificial Hivemind: within- and cross-model homogeneity (Infinity-Chat100)'
    claim = ('On 100 open-ended Infinity-Chat100 queries, 50 samples per query (top-p 0.9, T 1.0) are highly similar '
             'within a model (text-embedding-3-small cosine; mean pairwise similarity > 0.8 for 79% of queries over 25 '
             'models) and across models (cross-model mean similarity 0.71-0.82, mean 0.759).')
    paper = {'title': 'Artificial Hivemind: The Open-Ended Homogeneity of Language Models (and Beyond)',
             'authors': 'Jiang et al.', 'year': 2025, 'url': 'https://arxiv.org/abs/2510.22954v2'}
    source = SRC
    default_variant = 'top_p'

    def build_variants(self):
        common_dev = (
            'models under test are post-paper releases (our design); the paper\'s 79 models are not rerun',
            'n=50 samples are 50 independent single-sample calls (repeats), as the paper\'s "independently generates"',
            'embeddings come from an experiment-local script outside the runner (scripts/embed_hivemind.py), cached by '
            'sha256 of the response text; default 1536 dims assumed (paper names only text-embedding-3-small)',
            'models that reject temperature/top_p (M204, M206, M207 under panel_v2) are sampled at provider defaults '
            '(T=1, top_p=1) with provider-default reasoning; reasoning models get output cap >= 4096 instead of 2048',
            'empty outputs cannot be embedded and are excluded from their query pool (counted); paper handling unknown',
        )
        options = {'embedding_model': EMBEDDING_MODEL, 'embedding_cache': DEFAULT_CACHE, 'threshold': THRESHOLD}
        top_p = Variant('top_p', 'App. A top-p protocol: 100 queries x 50 samples, T 1.0, top_p 0.9, max 2048',
                        prompts={'user': USER}, params={'temperature': 1.0, 'top_p': 0.9, 'max_tokens': 2048},
                        repeats=50, options=options, deviations=common_dev)
        variants = {
            'top_p': top_p,
            'top_p_r20': top_p.override(
                'top_p_r20', description='top_p protocol with 20 samples per query (cost-reduced; the new-model run)', repeats=20,
                deviations=('20 samples per query instead of 50 (190 instead of 1,225 pairs per query): per-query '
                            'means unbiased but noisier, histogram tails wider; self-pair diagonal shift 1/20',)),
        }
        # Preserve historic variants and fingerprints; only the new run's model-scope metadata changes.
        r20 = variants['top_p_r20']
        variants['top_p_r20_panel_completion'] = replace(
            r20, name='top_p_r20_panel_completion', parent='top_p_r20',
            description='top_p_r20 protocol, including the five original-paper panel models '
                        '(coverage completion)',
            deviations=tuple(d for d in r20.deviations if d != common_dev[0]) + (
                'model scope expanded to include original-paper panel models; '
                'fresh outputs are separate from printed paper-reference values',))
        return variants

    def items(self, variant, limit=None):
        queries = load_queries()
        assert len(queries) == N_QUERIES and len(set(queries)) == N_QUERIES
        rows = [{'id': f'C143:q{i:03d}', 'query_index': i, 'query': q} for i, q in enumerate(queries)]
        return rows[:limit] if limit is not None else rows

    def program(self, variant):
        return SingleTurn(lambda item: [{'role': 'user', 'content': variant.prompts['user'].render(query=item['query'])}],
                          variant.params)

    def score(self, item, output, variant):
        text = output.get('content') or ''
        empty = not text.strip()
        return {'sha256': None if empty else sim.text_sha256(text), 'empty': empty, 'n_chars': len(text),
                'truncated': output.get('finish_reason') in {'length', 'max_tokens'} or output.get('status') == 'truncated',
                'refusal_like': bool(REFUSAL_PREFIX.match(text))}

    def aggregate(self, rows, variant, vectors=None, expected_ids=None):
        """Within-model metrics for one model. ``vectors`` ({sha256: vector}) defaults to the variant's cache;
        ``expected_ids`` (intended query ids) defaults to the variant's full item set, so a --limit run reports the
        missing queries as undefined and is never headline-reliable (audit m1)."""
        if vectors is None:
            vectors, cache_info = sim.load_embedding_cache(cache_path(variant), variant.options['embedding_model'])
        else:
            cache_info = {'path': None, 'exists': True, 'vectors': len(vectors), 'injected': True}
        threshold = variant.options['threshold']
        if expected_ids is None:
            expected_ids = [x['id'] for x in self.items(variant)]
        out = within_summary(query_pools(rows, vectors, variant.repeats), variant.repeats, threshold, expected_ids)
        out['embedding_cache'] = cache_info
        unavailable = out['failed_calls'] + out['empty_outputs'] + out['missing_embedding'] + out['absent_rows']
        reliable = unavailable == 0 and out['queries_undefined'] == 0 and out['queries_defined'] > 0
        out['headline_reliable'] = reliable
        if not cache_info['exists']:
            out['headline_caveat'] = f'embedding cache not found at {cache_info["path"]}: every response is missing'
        elif unavailable:
            out['headline_caveat'] = (f'{unavailable}/{out["responses_intended"]} intended responses unavailable for '
                                      f'embedding (failed {out["failed_calls"]}, empty {out["empty_outputs"]}, missing '
                                      f'vector {out["missing_embedding"]}, absent rows {out["absent_rows"]})')
        caveat = ('paper_value 0.79 = main-25 models (top_p 0.9, no reasoning); the decoding-matched paper reference '
                  'is its four o-series reasoning models, 0.71 (see paper_comparison). Schaeffer et al. 2026: '
                  'same-prompt responses with different ideas already have median cosine 0.70-0.72 and 20-32% of such '
                  'pairs exceed 0.8; no human baseline; the paper\'s null (unrelated prompts) is undemanding. '
                  'With 20 samples, see queries_within_2se_of_threshold')
        out['shelflife'] = Headline(
            metric='share_queries_mean_pairwise_cosine_ge_0.8', value=out['share_ge_threshold'],
            n=out['queries_defined'], direction=1, unit='fraction',
            denominator='queries with >= 2 embedded responses (paper: queries with embeddings); per-query mean over '
                        'distinct response pairs, self-pairs excluded; paper bins [0.8, 1.0]',
            ci=tuple(out['share_ci95_query_bootstrap']) if out['share_ci95_query_bootstrap'] else None,
            paper_value=0.79, missing=out['queries_undefined'],
            missingness_note='responses without an embedding (failed, empty, not in cache) leave their query pool; '
                             'queries with < 2 embedded responses are undefined and bounded in share_bounds_with_undefined',
            headline_reliable=reliable, reliability_note=caveat).to_dict()
        out['shelflife_mean_cosine'] = Headline(
            metric='mean_within_model_pairwise_cosine', value=out['mean_of_query_means'], n=out['queries_defined'],
            direction=1, unit='cosine', denominator='mean over defined queries of the per-query off-diagonal mean',
            missing=out['queries_undefined'], headline_reliable=reliable, role='sensitivity',
            reliability_note='Schaeffer et al. 2026 report c-bar = 0.805 for 15 newer API models on 12 topics; '
                             'paper diagonals (self-pairs included) average 0.858 for the main 25').to_dict()
        out['paper_comparison'] = compare_to_paper({'this_model': out['share_ge_threshold']})
        return out

    # ------------------------------------------------------------------------------------------- saved-run helpers
    def run_rows(self, run_dir, variant=None):
        """{model: runner-shaped rows} from a saved shelflife run (every response stream)."""
        from ..replay import model_key, run_records
        variant = self.variant(variant) if isinstance(variant, str) or variant is None else variant
        items = {x['id']: x for x in self.items(variant)}
        rows = defaultdict(list)
        for rec in run_records(run_dir):
            item = self.record_item(rec, items)
            if item is None:
                raise ValueError(f'record matches no C143 item: {rec.get("item_id") or rec.get("case_id")}')
            usable = rec.get('status') in {'ok', 'truncated', 'empty', 'complete'}
            output = self.record_output(rec) if usable else None
            m = model_key(rec)
            rows[m].append({'item': item, 'output': output, 'model': m, 'repeat': rec.get('repeat', 0),
                            'status': rec.get('status'),
                            'score': None if output is None else self.score(item, output, variant)})
        return dict(rows)

    # ------------------------------------------------------------------------------------------------- targets
    @staticmethod
    def _t_main25_within():
        return float(np.mean(list(paper_main25_within_share().values())))

    @staticmethod
    def _t_all79_bin(label):
        within, _, _ = paper_tables()
        return float(within[label].mean())

    @staticmethod
    def _t_cross(a, b):
        _, cross, _ = paper_tables()
        return float(cross.loc[a, b]) / 100

    _V2 = 'arXiv 2510.22954v2 '
    targets = (
        PaperTarget('main25_within_share_gt_0.8_percent', 79.0, 0.5,
                    _V2 + 'Sec. 3 (sections/3_lack_of_gen_diversity.tex line 25: "in 79% of cases, the average '
                    'similarity exceeds 0.8"); Fig. 2 Average column; recomputed from App. Table '
                    'tab:within_model_similarity_full rows of the 25 Fig. 2 models (0.9-1.0 + 0.8-0.9)',
                    '25 main models (2024-2025)', 'top_p', 'released_data',
                    compute=lambda exp: exp._t_main25_within(),
                    note='table rows recompute 79.08; tolerance 0.5 = the text rounds to an integer percent; the Fig. 2 '
                         'raster prints Average 43.8 + 35.2 = 79.0 with the same roster. Anchors the bin rule (top two '
                         'bins) and the 25-model roster, not our scorer (paper outputs unreleased). Roster ambiguity: '
                         'Table full_model_list marks gpt-4o-2024-08-06 "In Main", but Figs. 2 and 4 use '
                         'gpt-4o-2024-11-20 (followed here); with 08-06 the share would be 78.84'),
        PaperTarget('all79_within_average_bin_0.9_1.0_percent', 33.65, 0.0051,
                    _V2 + 'App. Table tab:within_model_similarity_full, Average row (tables/within_model_similarity_full.tex line 16)',
                    '79 models', 'top_p', 'released_data', compute=lambda exp: exp._t_all79_bin('0.9-1.0'),
                    note='mean of the 79 printed rows; tolerance = 2-decimal rounding'),
        PaperTarget('all79_within_average_bin_0.8_0.9_percent', 36.47, 0.0051,
                    _V2 + 'App. Table tab:within_model_similarity_full, Average row',
                    '79 models', 'top_p', 'released_data', compute=lambda exp: exp._t_all79_bin('0.8-0.9'),
                    note='mean of the 79 printed rows; tolerance = 2-decimal rounding'),
        PaperTarget('cross_deepseek_v3_qwen_max', 0.82, 0.005,
                    _V2 + 'Sec. 3 line 42 ("DeepSeek-V3 and qwen-max-2025-01-25 share a similarity of 0.82"); App. '
                    'tables tab:across_model_similarity_full_1-5', 'DeepSeek-V3 x qwen-max-2025-01-25', 'top_p',
                    'released_data', compute=lambda exp: exp._t_cross('DeepSeek-V3', 'qwen-max-2025-01-25'),
                    note='table 81.6; tolerance = 2-decimal rounding of the text'),
        PaperTarget('cross_deepseek_v3_gpt4o_1120', 0.81, 0.005, _V2 + 'Sec. 3 line 42 ("reach 0.81"); App. tables',
                    'DeepSeek-V3 x gpt-4o-2024-11-20', 'top_p', 'released_data',
                    compute=lambda exp: exp._t_cross('DeepSeek-V3', 'gpt-4o-2024-11-20'),
                    note='table 81.4; tolerance = 2-decimal rounding of the text'),
        PaperTarget('main25_cross_offdiag_max', 0.82, 0.005, _V2 + 'Sec. 3 line 42 ("ranges from 71% to 82%")',
                    '25 main models', 'top_p', 'released_data',
                    compute=lambda exp: max(paper_main25_cross_offdiag().values()),
                    note='table max 82.1 (qwen-max x qwen-plus)'),
        PaperTarget('main25_cross_offdiag_min', 0.71, 0.005, _V2 + 'Sec. 3 line 42 ("ranges from 71% to 82%")',
                    '25 main models', 'top_p', 'released_data',
                    compute=lambda exp: min(paper_main25_cross_offdiag().values()),
                    note='the Fig. 4 heatmap (raster in figures/model_pairwise_similarity_combined_top_p.pdf, read via '
                         'pdfimages) prints .69 for gemini-1.5-pro x c4ai-command-r-plus-08-2024 and .70 for '
                         'gemini-2.0-flash x command-r-plus, agreeing with the App. tables',
                    known_discrepancy=('the App. tables (69.5) and the Fig. 4 heatmap (.69) agree that the 25-model '
                                       'minimum is gemini-1.5-pro x c4ai-command-r-plus-08-2024 = 0.695; the text\'s '
                                       '"71%" is the outlier (a text-level approximation or error, not a subset)')),
        PaperTarget('main25_cross_offdiag_mean_port_computed', 0.759, 0.0005,
                    _V2 + 'NOT PRINTED in the paper (text gives only "71% to 82%"): port-computed mean of the 300 '
                    'distinct main-25 pairs of App. tables tab:across_model_similarity_full_1-5', '25 main models',
                    'top_p', 'released_data',
                    compute=lambda exp: float(np.mean(list(paper_main25_cross_offdiag().values()))),
                    note='a derived reference, not a paper check: value from the hivemind sources brief (a separate '
                         'parse script); guards the roster and name mapping used by compare_to_paper; tolerance = '
                         '3-decimal rounding. The auditor recomputed 75.94 independently'),
        PaperTarget('reasoning4_within_share_gt_0.8_mean_percent', 71.0, 0.005,
                    _V2 + 'App. Table tab:within_model_similarity_full rows o1-2024-12-17 (27+37), o1-mini-2024-09-12 '
                    '(38+40), o1-preview-2024-09-12 (40+33), o3-mini-2025-01-31 (34+35) (tables/'
                    'within_model_similarity_full.tex l.73-76); the mean is port-computed, not printed',
                    'o1 / o1-mini / o1-preview / o3-mini', 'top_p', 'released_data',
                    compute=lambda exp: float(np.mean(list(paper_reasoning4_within_share().values()))),
                    note='decoding-matched reference for post-paper reasoning models (fixed T=1/top_p=1 APIs, hidden '
                         'reasoning): 64, 78, 73, 69 -> 71.0; tolerance = exact sum of printed integers'),
        PaperTarget('reasoning4_cross_intragroup_mean_port_computed', 0.7743, 0.0005,
                    _V2 + 'App. tables tab:across_model_similarity_full_1-5, the 6 distinct pairs among the o-series '
                    'models (75.2, 76.6, 78.5, 80.2, 76.9, 77.2); mean port-computed, not printed',
                    'o1 / o1-mini / o1-preview / o3-mini', 'top_p', 'released_data',
                    compute=lambda exp: float(np.mean(list(paper_cross_pairs(REASONING4).values()))),
                    note='decoding-matched cross reference for pairs of post-paper reasoning models; tolerance = '
                         'rounding of 0.774333'),
        PaperTarget('random_cross_prompt_pairs_in_0.1_0.2_percent', 100.0, 0.5, _V2 + 'Fig. 2 caption',
                    'global response pool', 'top_p', 'unverifiable', note='no released responses'),
        PaperTarget('top50_unique_models_mean', 8.0, 0.5, _V2 + 'Sec. 3 line 70; Fig. 5 (raster)', '25 main models',
                    'top_p', 'unverifiable', note='"~8"; figure only, no released responses'),
        PaperTarget('gpt4o_mini_within_share_gt_0.8_percent', 87.0, 0.5,
                    _V2 + 'App. Table tab:within_model_similarity_full row gpt-4o-mini-2024-07-18 (53 + 34)',
                    'gpt-4o-mini-2024-07-18', 'top_p', 'live',
                    note='the only main-25 model still routable with top_p 0.9 (M015); a rerun would anchor our '
                         'generation + embedding pipeline. Not planned: paper models were not rerun'),
    )
