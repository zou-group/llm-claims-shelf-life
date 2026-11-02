"""C046 LC-AlpacaEval: annotator prompt/order/parser against the official alpaca_eval code and released annotations,
the prior binary scorer against the earlier implementation, and rescoring parity with its reported numbers."""
import json
import math
import statistics

import pytest

from shelflife import registry
from shelflife.ledger import MockTransport
from shelflife.paths import INITIAL
from shelflife.replay import legacy_records, read_jsonl
from shelflife.runner import run
from shelflife.sourcecode import load_functions
from shelflife.experiments import c046_alpacaeval_length as m

exp = registry.get('C046')
REPRO = INITIAL / 'reproductions/C046'
RUNS = {name: INITIAL / f'runs/2026-09-30-C046-{name}' for name in ('expanded', 'directGPT6-expanded')}
BINARY_RUNS = ('expanded', 'directGPT6-expanded')


def prior_code():
    """initial-reproduction scorer, compiled without importing (no __pycache__ under experiments/)."""
    base = load_functions(REPRO / 'score.py', ['score', 'score_response', 'grouped_pairs'])
    return {'strict': base['score']}


# ------------------------------------------------------------------ prompts and order


def test_seeded_switch_reproduces_every_released_judgment():
    """Official seed 'is_switched_outputs'+instruction+'0' + official logprob_parser recovers all 1,596 released
    preferences (< 1e-9); the processor-class default seed 123 does not (the annotator passes seed 0)."""
    for seed, ok in ((0, True), (123, False)):
        errs = []
        for c in m.CONDITIONS:
            for r in m.annotations(c):
                if r['raw_completion'] is None:
                    assert r['output_1'] == r['output_2'] and float(r['preference']) == 1.5
                    continue
                p = m.official_logprob_preference(m._parse_raw(r['raw_completion']))[0]
                p = 3 - p if m.is_switched(r['instruction'], seed) else p
                errs.append(abs(p - float(r['preference'])))
        assert len(errs) == 1596
        assert bool(max(errs) < 1e-9) == ok


def test_items_order_matches_switch_and_identity_mapping():
    for x in exp.items(exp.variant()):
        r = m.annotations(x['condition'])[x['instruction_index']]
        assert (x['order'] == 'candidate_first') == m.is_switched(r['instruction'])
        cand = x['output_1'] if x['slot_identities'][0] == 'candidate' else x['output_2']
        assert cand == r['output_2'] and r['generator_2'].endswith(x['condition'])


def test_annotation_rows_align_across_conditions_and_baseline():
    c, v = m.annotations('concise'), m.annotations('verbose')
    base = m.baseline_outputs()
    for a, b, o in zip(c, v, base):
        assert a['instruction'] == b['instruction'] == o['instruction'] and a['output_1'] == b['output_1'] == o['output']


# ------------------------------------------------------------------ subset sampling


# ------------------------------------------------------------------ weighted scorer
def _lp(tokens, emitted=None, extra=0):
    top = [{'token': t, 'logprob': lp} for t, lp in tokens]
    content = [{'token': emitted or tokens[0][0], 'logprob': tokens[0][1], 'top_logprobs': top}]
    content += [{'token': 'x', 'logprob': 0.0, 'top_logprobs': []}] * extra
    return {'content': content}


def test_weighted_scorer_equals_official_batch_parser_on_archive():
    """log_prob_index=0 equals the configured 'batch' mode on every released single-token completion."""
    for c in m.CONDITIONS:
        for r in m.annotations(c):
            if r['raw_completion'] is None:
                continue
            raw = m._parse_raw(r['raw_completion'])
            assert len(raw['logprobs']['content']) == 1
            ours = exp.weighted_preference(raw['logprobs'])[0]
            assert ours == m.official_logprob_preference(raw)[0]


def test_weighted_scorer_edge_cases():
    wp = exp.weighted_preference
    p, status, aligned, _ = wp(_lp([('m', math.log(0.75)), ('M', math.log(0.25))]))
    assert abs(p - 1.25) < 1e-12 and status == 'normalized_logprobs' and aligned
    assert wp(_lp([('M', -0.1), ('Both', -3)]))[:2] == (2.0, 'normalized_logprobs_one_sentinel_censored')
    assert wp(_lp([('m', -0.1)]))[:2] == (1.0, 'normalized_logprobs_one_sentinel_censored')
    assert wp(_lp([('The', -0.1), ('Model', -2)]))[:2] == (None, 'missing_sentinel_logprobs')
    assert wp(None)[:2] == (None, 'missing_logprobs') and wp({'content': []})[:2] == (None, 'missing_logprobs')
    p, _, aligned, n = wp(_lp([('</think>', -0.01), ('m', -5), ('M', -6)], emitted='</think>', extra=3))
    assert p is not None and not aligned and n == 4      # mechanical value kept, flagged as misaligned
    item = next(x for x in exp.items(exp.variant()) if not x['identical_outputs'] and x['slot_identities'][0] == 'reference')
    s = exp.score(item, {'logprobs': _lp([('m', math.log(0.9)), ('M', math.log(0.1))])}, exp.variant())
    assert abs(s['candidate_preference'] - 0.1) < 1e-12 and s['parsed'] == 'reference'
    ident = next(x for x in exp.items(exp.variant()) if x['identical_outputs'])
    assert exp.score(ident, {'logprobs': None}, exp.variant())['candidate_preference'] == 0.5


# ------------------------------------------------------------------ binary scorers vs prior code


def test_binary_case_sensitivity_and_identity():
    v = exp.variant('binary_paired')
    x = next(i for i in exp.items(v) if i['order'] == 'candidate_first' and not i['identical_outputs'])
    assert exp.score(x, {'content': 'm'}, v)['parsed'] == 'candidate'
    assert exp.score(x, {'content': 'M'}, v)['parsed'] == 'reference'


# ------------------------------------------------------------------ rescoring parity with reports
def _check_binary_summary(ours, theirs):
    for c in m.CONDITIONS:
        a, b = ours[c], theirs['variants'][c]
        assert (a['intended_pair_repeats'], a['complete_pair_repeats']) == (b['intended_pair_repeats'], b['complete_pair_repeats'])
        for key in ('candidate_win_fraction_complete', 'order_conflict_rate_complete'):
            assert (a[key] is None and b[key] is None) or abs(a[key] - b[key]) < 1e-12, key
        assert all(abs(p - q) < 1e-12 for p, q in zip(a['candidate_win_fraction_missingness_bounds'],
                                                      b['candidate_win_fraction_missingness_bounds']))
        lk = 'longer_answer_preference_fraction_complete_abs_delta_gt30'
        assert (a['longer_answer_preference_abs_delta_gt30'] is None and b[lk] is None) or \
            abs(a['longer_answer_preference_abs_delta_gt30'] - b[lk]) < 1e-12
        assert a['eligible_longer_comparisons'] == b['eligible_longer_comparisons']
        assert a['analytic_identical_pair_repeats'] == b['analytic_identical_pair_repeats']
    assert ours['matched_complete_instruction_repeats'] == theirs['matched_complete_instruction_repeats']
    assert ours['matched_intended_instruction_repeats'] == theirs['matched_intended_instruction_repeats']
    assert abs(ours['verbose_minus_concise_complete'] - theirs['verbose_minus_concise_complete']) < 1e-12
    assert all(abs(p - q) < 1e-12 for p, q in zip(ours['difference_missingness_bounds'], theirs['difference_missingness_bounds']))


# ------------------------------------------------------------------ archive anchors


# ------------------------------------------------------------------ program replay with MockTransport


# ------------------------------------------------------------------ length-controlled win rate (refit)
def _released_fit(model):
    from shelflife.experiments import _c046_lc as lc
    rows = m._results(model)
    assert [r['instruction'] for r in rows] == [b['instruction'] for b in m.baseline_outputs()]
    pref = [math.nan if r['preference'] is None else float(r['preference']) for r in rows]
    return lc.fit_length_controlled(pref, [len(r['output_1']) for r in rows], [len(r['output_2']) for r in rows],
                                    m.instruction_difficulty().values)


def test_figure3_cells_all_targeted():
    """Audit minor 3: all 30 Figure 3 cells (6 models x 3 prompts x raw/LC, minus the fixed 50.0 standard cells of
    the baseline) are targets."""
    names = {x.name for x in exp.targets}
    fig = [n for n in names if n.startswith('fig3_')]
    assert len(fig) == 24
    assert {'gpt4_1106_concise_win_rate', 'gpt4_1106_verbose_win_rate', 'gpt4_1106_lc_concise', 'gpt4_1106_lc_verbose',
            'claude21_concise_win_rate', 'gpt35_1106_verbose_win_rate'} <= names


# ------------------------------------------------------------------ robustness
from pathlib import Path  # noqa: E402

R2 = Path(__file__).resolve().parents[3] / 'experiments/run_minimal_setups/runs/earlier/runs'


@pytest.mark.parametrize('text,slot', [
    ('Model m gives a thorough guide.\n\nMost human evaluators would prefer the more thorough response, Model m.\n\nm', 0),
    ('Overall, I think M is somewhat better.\n\nM', 1),                    # explanation then identifier (M206)
    ('Both are fine.\n\n**M**', 1), ('Reasoning...\nFinal answer: m', 0), ('Analysis.\nModel M', 1),
    ('<think>m vs M</think>\n"m"', 0), ('m\n\nBecause its instructions are accurate.', 0),   # identifier first
    ('m\nM', 1),                                                           # last line wins
    ('To determine the best model identifier based on the given instruction, let\'s analyze the outputs.\n\n1. **Re', None),
    ('', None), ("I'm not able to compare these.", None),
    ('**Model M** is better because its design makes sense.\n\nIn **Model m**, the turbine', None),   # truncated prose
    # verdict clause on the last line (M066 robust run, 2026-10-10 follow-up)
    ('Analysis of both outputs.\n\nThe best answer is M.', 1), ('Analysis.\n\nThe best answer is: m', 0),
    ('Analysis.\n\nTherefore, the best model identifier is: M', 1),
    ('Analysis.\n\nBased on the analysis, the best model identifier is **m**.', 0),
    ('Analysis.\n\nBest Model Identifier: M', 1), ('Analysis.\n\nTherefore, the model identifier of the best model is: m', 0),
    ('Analysis.\n\nThe best answer is not m.', None), ('Analysis.\n\nThe best answer is m or M.', None),
    ('Analysis.\n\nFocusing on quality and relevance, m', None),
])
def test_robust_binary_fallback(text, slot):
    assert m.prior_strict(text, False)[0] is None
    assert m.parse_binary_robust(text) == slot


def test_robust_score_keeps_strict_and_is_case_sensitive():
    rp, bp = exp.variant('robust'), exp.variant('binary_paired')
    item = next(x for x in exp.items(bp) if not x['identical_outputs'])
    for text in ('m', 'M', ' M\n'):
        assert {k: v for k, v in exp.score(item, {'content': text}, rp).items() if k != 'parse_path'} == \
            exp.score(item, {'content': text}, bp)
        assert exp.score(item, {'content': text}, rp)['parse_path'] == 'paper'
    s = exp.score(item, {'content': 'Explanation.\n\nM'}, rp)
    assert s['parse_path'] == 'robust' and s['parsed'] == item['slot_identities'][1]
    assert exp.score(item, {'content': 'Explanation.\n\nM'}, bp)['parsed'] is None


def test_robust_variants_params():
    bp, rb = (exp.variant(n) for n in ('binary_paired', 'robust'))
    assert (rb.prompts, rb.strategy, rb.repeats) == (bp.prompts, bp.strategy, bp.repeats)
    assert rb.options == {**bp.options, 'parse': 'robust'} and not rb.faithful
    assert rb.params == {'temperature': 0, 'max_tokens': 1024} and rb.options['parse'] == 'robust'
    assert [x['id'] for x in exp.items(rb)] == [x['id'] for x in exp.items(bp)]
    w = exp.variant('weighted_frozen')
    assert w.faithful and 'parse' not in w.options and w.params == m.ANNOTATOR_PARAMS   # logprob judge untouched


def test_robust_binary_reliability_counts_observed_repeats():
    rp = exp.variant('robust')
    rows = []
    for k, item in enumerate(exp.items(rp)):
        text = 'nope' if k < 20 else 'm'
        out = {'content': None if item['identical_outputs'] else text}
        rows.append({'item': item, 'repeat': 0, 'output': out, 'score': exp.score(item, out, rp)})
    agg = exp.aggregate(rows, rp)
    sent = sum(not x['identical_outputs'] for x in exp.items(rp))
    assert agg['judgments_sent'] == sent and agg['failed_or_missing'] == 0      # 1 repeat run, not 3
    assert agg['unparsed'] == sum(not x['identical_outputs'] for x in exp.items(rp)[:20])
    assert agg['headline_reliable'] is True


