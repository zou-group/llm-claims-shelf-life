"""C143 Artificial Hivemind: queries vs the HF release, similarity math vs the authors' functions and literal fixtures,
paper-table parsing vs an independent LaTeX parser, and the denominator / missing-embedding rules."""
import dataclasses
import hashlib
import json
import os
import re

import numpy as np
import pytest

from shelflife import registry
from shelflife.paths import DATA
from shelflife.sourcecode import load_functions

exp = registry.get('C143')
pytestmark = pytest.mark.skipif(exp.maturity == 'draft' and os.environ.get('SHELFLIFE_INCLUDE_DRAFTS') != '1',
                                reason='draft claim: run `python -m shelflife.cli check C143`')

from shelflife.experiments import c143_artificial_hivemind as mod  # noqa: E402
from shelflife.experiments import _c143_similarity as sim  # noqa: E402

HF_EVAL = DATA / 'sources/C143_infinity_chat_hf/infinite-chats-eval/train.parquet'
TABLES = DATA / 'sources/C143_hivemind_arxiv_v2/src/tables'
AUTHORS = DATA / 'sources/C143_hivemind/src/model_calibration_analysis'
SQ2 = 1 / np.sqrt(2)
E1, E2, E3 = [1.0, 0.0], [0.0, 1.0], [SQ2, SQ2]


# ------------------------------------------------------------------------------------------------ queries / prompts
def _hf_queries():
    pq = pytest.importorskip('pyarrow.parquet')
    return pq.read_table(HF_EVAL).column('query').to_pylist()


def test_hf_snapshot_hash():
    digest = hashlib.sha256(HF_EVAL.read_bytes()).hexdigest()
    assert digest.startswith('eec7277f'), digest   # C143_infinity_chat_hf/PROVENANCE.md


@pytest.mark.parametrize('name', sorted(exp.variants))
def test_requests_are_the_hf_queries_verbatim(name):
    queries = _hf_queries()
    assert len(queries) == 100 and len(set(queries)) == 100
    v = exp.variant(name)
    items = exp.items(v)
    assert [x['id'] for x in items] == [f'C143:q{i:03d}' for i in range(100)]
    prog = exp.program(v)
    for item, q in zip(items, queries):
        msgs = prog.preview(item)
        assert msgs == [{'role': 'user', 'content': q}]           # one user turn, no system prompt
        assert msgs[0]['content'].encode('utf-8') == q.encode('utf-8')
    assert exp.items(v, limit=0) == [] and len(exp.items(v, limit=3)) == 3


def test_known_query_wordings():
    queries = _hf_queries()
    assert queries[0] == 'Write a short story about a colorful toad goes on an adventure in 50 words.'
    assert 'Write a metaphor involving time.' in queries     # paper text says "about time" (Schaeffer et al.)
    assert 'Generate a joke about electric vehicles.' in queries


def test_variant_params_and_labels():
    v = exp.variant('top_p')
    assert v.faithful and v.repeats == 50 and exp.default_variant == 'top_p'
    assert v.params == {'temperature': 1.0, 'top_p': 0.9, 'max_tokens': 2048}     # App. A
    r20 = exp.variant('top_p_r20')
    assert not r20.faithful and r20.repeats == 20 and r20.params == v.params
    assert v.options['embedding_model'] == 'text-embedding-3-small'


def test_no_auxiliary_models():
    for name in exp.variants:
        assert not tuple(getattr(exp.program(exp.variant(name)), 'auxiliary_models', ()) or ())


def test_panel_completion_keeps_decoding_and_updates_scope():
    """New coverage includes original-paper models without claiming that historic runs did so."""
    old = exp.variant('top_p_r20')
    new = exp.variant('top_p_r20_panel_completion')
    assert new.params == {'temperature': 1.0, 'top_p': 0.9, 'max_tokens': 2048}
    assert new.repeats == 20 and new.faithful is False and new.parent == 'top_p_r20'
    assert new.options == old.options and new.prompts == old.prompts
    assert any("paper's 79 models are not rerun" in d for d in old.deviations)
    assert not any("paper's 79 models are not rerun" in d for d in new.deviations)
    assert any('model scope expanded' in d for d in new.deviations)


def test_target_models_reject_sampling_params():
    """The forced sampling deviation in the docstring is grounded in the capability files."""
    caps50 = json.loads((DATA / 'model_capabilities.json').read_text())
    caps = json.loads((DATA / 'panel_capabilities.json').read_text())
    for m in ('M204', 'M207'):
        assert {'temperature', 'top_p'} <= set(caps50[m]['rejected_params'])
    assert {'temperature', 'top_p'} <= set(caps['M206']['rejected_params'])


# ----------------------------------------------------------------------------------------------------- scoring
@pytest.mark.parametrize('output,expected', [
    ({'content': 'hello'}, {'sha256': '2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824',
                            'empty': False, 'truncated': False, 'refusal_like': False}),
    ({'content': ''}, {'sha256': None, 'empty': True}),
    ({'content': None}, {'sha256': None, 'empty': True}),
    ({'content': '  \n\t'}, {'sha256': None, 'empty': True}),
    ({'content': "I'm sorry, but I can't write that."}, {'empty': False, 'refusal_like': True}),
    ({'content': '**Time** is a river.', 'finish_reason': 'length'}, {'truncated': True, 'refusal_like': False}),
    ({'content': 'Time is a river.', 'status': 'truncated'}, {'truncated': True}),
])
def test_score_fields(output, expected):
    s = exp.score(exp.items(exp.variant(), limit=1)[0], output, exp.variant())
    for k, val in expected.items():
        assert s[k] == val, k


def test_sha_is_of_exact_text_not_stripped():
    v = exp.variant()
    item = exp.items(v, limit=1)[0]
    assert exp.score(item, {'content': 'a '}, v)['sha256'] != exp.score(item, {'content': 'a'}, v)['sha256']


# ------------------------------------------------------------------------------------------- similarity math
def test_within_pairs_literal():
    assert sim.within_pairs_mean([E1, E2, E3]) == pytest.approx((0 + SQ2 + SQ2) / 3)
    assert sim.with_self_pairs((0 + 2 * SQ2) / 3, 3) == pytest.approx((3 + 4 * SQ2) / 9)
    assert sim.within_pairs_mean([[3.0, 0.0], [0.0, 0.5]]) == pytest.approx(0.0)      # scale invariant
    assert sim.within_pairs_mean([E1]) is None


def test_self_pair_shift_at_paper_scale():
    # paper diagonal = 0.02 + 0.98 * offdiag at n = 50: ~ +0.003 for means near 0.85
    assert sim.with_self_pairs(0.85, 50) - 0.85 == pytest.approx(0.003)
    assert sim.with_self_pairs(0.85, 20) - 0.85 == pytest.approx(0.0075)


def _authors(file, names):
    return load_functions(AUTHORS / file, names, extra_globals={'np': np})


def test_histogram_bin_edges_match_paper():
    h = sim.paper_histogram([0.8, 0.7999999, 1.0, 0.0, 0.9, 1.0 + 1e-12, None])
    assert h['n'] == 6 and h['out_of_range'] == 1
    pct = h['percent']
    assert pct['0.8-0.9'] == pytest.approx(100 / 6)          # 0.8 exactly -> [0.8, 0.9)
    assert pct['0.7-0.8'] == pytest.approx(100 / 6)
    assert pct['0.9-1.0'] == pytest.approx(200 / 6)          # 0.9 and 1.0 (last bin closed)
    assert pct['0.0-0.1'] == pytest.approx(100 / 6)
    assert sum(pct.values()) == pytest.approx(500 / 6)       # overshoot stays in the denominator (authors' code)


# -------------------------------------------------------------------------------------------- aggregate fixtures
def _rows(spec, model='X'):
    """spec: [(query index, content or None for a failed call)]; repeats numbered 0.. per query"""
    items = exp.items(exp.variant())
    rows, next_rep = [], {}
    for qi, content in spec:
        rep = next_rep.get(qi, 0)
        next_rep[qi] = rep + 1
        item = items[qi]
        if content is None:
            rows.append({'item': item, 'output': None, 'score': None, 'model': model, 'repeat': rep, 'status': 'call_error'})
            continue
        out = {'content': content, 'finish_reason': 'stop'}
        rows.append({'item': item, 'output': out, 'score': exp.score(item, out, exp.variant()), 'model': model,
                     'repeat': rep, 'status': 'complete'})
    return rows


def _vec(*pairs):
    return {sim.text_sha256(t): np.asarray(v, dtype=np.float32) for t, v in pairs}


def test_aggregate_denominators_and_missing():
    v = dataclasses.replace(exp.variant('top_p_r20'), repeats=4)
    vectors = _vec(('same', E1), ('east', E1), ('north', E2), ('alone', E3))
    rows = _rows([(0, 'same'), (0, 'same'), (0, 'same'),                       # q0: 3 identical -> mean 1.0, 1 absent
                  (1, 'east'), (1, 'north'), (1, None), (1, ''),               # q1: mean 0.0, 1 failed, 1 empty
                  (2, 'alone'), (2, 'not in cache')])                          # q2: 1 embedded -> undefined, 2 absent
    out = exp.aggregate(rows, v, vectors=vectors, expected_ids=['C143:q000', 'C143:q001', 'C143:q002'])
    assert (out['queries_intended'], out['queries_defined'], out['queries_undefined']) == (3, 2, 1)
    assert out['k_ge_threshold'] == 1 and out['share_ge_threshold'] == pytest.approx(0.5)
    assert out['share_bounds_with_undefined'] == pytest.approx([1 / 3, 2 / 3])
    assert (out['failed_calls'], out['empty_outputs'], out['missing_embedding'], out['absent_rows']) == (1, 1, 1, 3)
    assert out['responses_embedded'] == 6 and out['responses_intended'] == 12
    assert out['mean_of_query_means'] == pytest.approx(0.5)
    assert out['mean_of_query_means_with_self_pairs'] == pytest.approx(((1 / 3 + 2 / 3 * 1.0) + (0.5 + 0.5 * 0.0)) / 2)
    assert out['histogram_percent']['0.9-1.0'] == pytest.approx(50) and out['histogram_percent']['0.0-0.1'] == pytest.approx(50)
    assert out['headline_reliable'] is False and 'unavailable' in out['headline_caveat']
    h = out['shelflife']
    assert h['value'] == pytest.approx(0.5) and h['n'] == 2 and h['missing'] == 1 and h['paper_value'] == 0.79


def test_aggregate_complete_is_reliable():
    v = dataclasses.replace(exp.variant('top_p_r20'), repeats=2)
    out = exp.aggregate(_rows([(0, 'a'), (0, 'b')]), v, vectors=_vec(('a', E1), ('b', E3)), expected_ids=['C143:q000'])
    assert out['headline_reliable'] is True and out['share_ge_threshold'] == 0.0
    assert out['mean_of_query_means'] == pytest.approx(SQ2)


def test_aggregate_without_cache_reports_everything_missing(tmp_path, monkeypatch):
    monkeypatch.setenv('SHELFLIFE_C143_EMBEDDING_CACHE', str(tmp_path / 'absent'))
    out = exp.aggregate(_rows([(0, 'a'), (0, 'b')]), exp.variant())
    assert out['embedding_cache']['exists'] is False and out['missing_embedding'] == 2
    assert out['shelflife']['value'] is None and out['headline_reliable'] is False


def test_embedding_cache_files(tmp_path):
    a, b = sim.text_sha256('a'), sim.text_sha256('b')
    np.savez(tmp_path / '1.npz', sha256=np.array([a, b]), embedding=np.array([E1, E2], dtype=np.float32),
             model=np.array('text-embedding-3-small'))
    np.savez(tmp_path / '2.npz', sha256=np.array([a]), embedding=np.array([E2], dtype=np.float32))   # conflicting dup
    np.savez(tmp_path / '3.npz', sha256=np.array([sim.text_sha256('c')]), embedding=np.array([E1], dtype=np.float32),
             model=np.array('text-embedding-3-large'))
    vectors, info = sim.load_embedding_cache(tmp_path)
    assert set(vectors) == {a, b} and vectors[a].tolist() == E1
    assert info['duplicate_conflicts'] == 1 and len(info['skipped_files_wrong_model']) == 1 and info['dims'] == [2]
    monkey = dataclasses.replace(exp.variant(), options={**exp.variant().options, 'embedding_cache': str(tmp_path)})
    out = exp.aggregate(_rows([(0, 'a'), (0, 'b'), (0, 'c')]), monkey)
    assert out['responses_embedded'] == 2 and out['missing_embedding'] == 1


# --------------------------------------------------------------------------- paper tables: independent LaTeX parse
def _norm(name):
    name = name.split('/')[-1].replace('Meta-', '')
    return 'Llama-3.1-405B-Instruct' if name == 'Llama-3.1-405B-Instruct-Turbo' else name


def _latex_within():
    rows = {}
    for line in (TABLES / 'within_model_similarity_full.tex').read_text().splitlines():
        m = re.match(r'\s*([^&\\%][^&]*?)\s*((?:&\s*[\d.]+\s*){10})\\\\', line)
        if m and 'textbf' not in line:
            rows[_norm(m.group(1))] = [float(x) for x in re.findall(r'[\d.]+', m.group(2))]
    return rows


def _latex_cross():
    cells = {}
    for part in (TABLES / 'across_model_similarity_full.tex').read_text().split('\\begin{tabular}')[1:]:
        header, body = part.split('\\midrule', 1)
        cols = [_norm(c) for c in re.findall(r'\\rotatebox\{90\}\{([^}]*)\}', header)]
        for line in body.split('\\bottomrule')[0].splitlines():
            parts = [p.strip() for p in line.strip().rstrip('\\').split('&')]
            if len(parts) == len(cols) + 1 and not line.strip().startswith('%'):
                for c, val in zip(cols, parts[1:]):
                    cells[(_norm(parts[0]), c)] = float(val)
    return cells


def test_main25_roster_present_in_both_tables():
    within, cross, main25 = mod.paper_tables()
    assert len(main25) == 25 == len(set(main25))
    assert set(main25) <= set(within.index) and set(main25) <= set(cross.index)


def test_target_values_literal():
    t = {x.name: x for x in exp.targets}
    assert t['main25_within_share_gt_0.8_percent'].compute(exp) == pytest.approx(79.08)
    assert t['main25_cross_offdiag_min'].compute(exp) == pytest.approx(0.695)       # documented discrepancy vs 0.71
    assert t['main25_cross_offdiag_mean_port_computed'].compute(exp) == pytest.approx(0.75944, abs=1e-5)


# ------------------------------------------------------------------------------------- scoring revision
def test_duplicate_rows_are_rejected_not_pooled():
    """Audit M2: the same rows twice (e.g. a run dir passed twice) must raise."""
    v = dataclasses.replace(exp.variant('top_p_r20'), repeats=2)
    vectors = _vec(('a', E1), ('b', E2))
    rows = _rows([(0, 'a'), (0, 'b')])
    assert exp.aggregate(rows, v, vectors=vectors, expected_ids=['C143:q000'])['rows'] == 2
    with pytest.raises(ValueError, match='duplicate'):
        exp.aggregate(rows + rows, v, vectors=vectors, expected_ids=['C143:q000'])


def test_extra_repeats_are_rejected_not_clamped():
    v = dataclasses.replace(exp.variant('top_p_r20'), repeats=2)
    rows = _rows([(0, 'a'), (0, 'b'), (0, 'c')])              # repeats 0, 1, 2 with a 2-repeat variant
    with pytest.raises(ValueError, match='outside 0..1'):
        exp.aggregate(rows, v, vectors=_vec(('a', E1), ('b', E2), ('c', E3)), expected_ids=['C143:q000'])


def test_mixed_models_rejected_by_pools():
    with pytest.raises(ValueError, match="one model"):
        mod.query_pools(_rows([(0, 'a')], 'X') + _rows([(0, 'b')], 'Y'), {})


def test_limited_run_is_measured_against_all_100_queries():
    """Audit m1: a complete 10-query run is not headline-reliable; the other 90 queries are undefined and absent."""
    v = dataclasses.replace(exp.variant('top_p_r20'), repeats=2)
    spec = [(q, f'q{q}r{r}') for q in range(10) for r in range(2)]
    vectors = _vec(*[(t, E1 if r == 0 else E3) for (q, t), r in zip(spec, [0, 1] * 10)])
    out = exp.aggregate(_rows(spec), v, vectors=vectors)
    assert (out['queries_intended'], out['queries_with_rows'], out['queries_defined'], out['queries_undefined']) == (100, 10, 10, 90)
    assert out['absent_rows'] == 180 and out['responses_intended'] == 200 and out['responses_embedded'] == 20
    assert out['share_ge_threshold'] == 0.0 and out['share_bounds_with_undefined'] == pytest.approx([0.0, 0.9])
    assert out['headline_reliable'] is False


def test_unknown_query_ids_raise():
    with pytest.raises(ValueError, match='not intended'):
        mod.within_summary({'C143:q999': mod.query_pools(_rows([(0, 'a')]), {})['C143:q000']}, 1,
                           expected_ids=['C143:q000'])


def test_jackknife_matches_brute_force_leave_one_out():
    rng = np.random.default_rng(11)
    E = rng.normal(size=(7, 5)) + 2
    loo = np.array([sim.within_pairs_mean(np.delete(E, i, axis=0)) for i in range(7)])
    expected = np.sqrt(6 / 7 * ((loo - loo.mean()) ** 2).sum())
    assert sim.jackknife_pairs_se(E) == pytest.approx(expected)
    assert sim.jackknife_pairs_se(E[:2]) is None


def test_aggregate_reports_near_threshold_counts():
    v = dataclasses.replace(exp.variant('top_p_r20'), repeats=3)
    vectors = _vec(('a', E1), ('b', E1), ('c', [1.0, 0.05]))
    out = exp.aggregate(_rows([(0, 'a'), (0, 'b'), (0, 'c')]), v, vectors=vectors, expected_ids=['C143:q000'])
    q = out['per_query'][0]
    assert q['se_jackknife'] is not None and out['per_query_se_jackknife_median'] == pytest.approx(q['se_jackknife'])
    assert out['queries_within_2se_of_threshold'] == 0          # mean ~0.999, far above 0.8


# --------------------------------------------------------------------------- new-model runs (rescoring parity)
RUNS = 'experiments/run_minimal_setups/runs/added-b/runs/C143-top_p_r20-Lall/'
CACHE = mod.PROJECT / mod.DEFAULT_CACHE


def _require_cache():
    if not any(CACHE.glob('*.npz')):
        pytest.skip(f'embedding cache not present: {CACHE}')


