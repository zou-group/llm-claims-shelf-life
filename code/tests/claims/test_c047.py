"""C047 (Chen et al. 2024, judgement bias): prompts against the paper's LaTeX, items against the released data, the
reconstructed vote parser and the paper's ASR rule. No judge outputs were released and there are no prior runs."""
import itertools
import json
import os
import random
import re
from fractions import Fraction

import pytest

from shelflife import registry
from shelflife.experiments import c047_judgement_bias as m

exp = registry.get('C047')
pytestmark = pytest.mark.skipif(exp.maturity == 'draft' and os.environ.get('SHELFLIFE_INCLUDE_DRAFTS') != '1',
                                reason='draft claim: run `python -m shelflife.cli check C047`')


# ------------------------------------------------------------------ prompts vs the v5 LaTeX (independent parser)
def _latex_prompts():
    tex = (m.LATEX / 'appendix.tex').read_text()
    section = tex[tex.index(r'\section{Evaluation Prompts for GPT-4}'):tex.index(r'\section{More Results on Bias Analysis}')]
    listings = re.findall(r'\\begin\{lstlisting\}\[[^\]]*\]\n(.*?)\\end\{lstlisting\}', section, re.S)
    assert len(listings) == 3
    out = []
    for body in listings:
        obj = json.loads('{' + body.strip().rstrip(',') + '}')
        out.append((obj['system'], obj['template']))
    return dict(zip(('cot', 'no_cot', 'answer_first'), out))


# ------------------------------------------------------------------ items vs the release (independent loading)
def _release():
    load = lambda f: [json.loads(l) for l in (m.REPO / 'data' / f).read_text().splitlines() if l.strip()]
    raw = {r['question_id']: r for r in load('raw.json')}
    perts = {p: {r['question_id'].rsplit('-', 1)[0]: r for r in load(f'{p}.json')} for p in m.PERTURBATIONS}
    return raw, perts


def test_items_present_released_answers_in_both_orders():
    raw, perts = _release()
    v = exp.variant('paper_no_cot')
    items = exp.items(v)
    assert len(items) == 142 * 10
    by_pair = {}
    for x in items:
        by_pair.setdefault(x['qid'], {})[(x['group'], x['order'])] = x
    assert len(by_pair) == 142
    for qid, rows in by_pair.items():
        r = raw[qid]['answers']
        assert (rows[('control', 'raw1_first')]['answer1'], rows[('control', 'raw1_first')]['answer2']) == (
            r['answer1']['answer'], r['answer2']['answer'])
        assert (rows[('control', 'raw2_first')]['answer1'], rows[('control', 'raw2_first')]['answer2']) == (
            r['answer2']['answer'], r['answer1']['answer'])
        for p in m.PERTURBATIONS:
            answers = perts[p][qid]['answers']
            slot = next(s for s, a in answers.items() if a.get('perturb'))
            other = 'answer1' if slot == 'answer2' else 'answer2'
            second, first = rows[(p, 'pert_second')], rows[(p, 'pert_first')]
            assert (second['answer1'], second['answer2']) == (answers[other]['answer'], answers[slot]['answer'])
            assert (first['answer1'], first['answer2']) == (answers[slot]['answer'], answers[other]['answer'])
            assert second['slots'] == [other, slot] and first['slots'] == [slot, other]
            assert second['perturbed_slots'][p] == slot


def test_minimal_limit_is_89_complete_level_balanced_pairs():
    v = exp.variant(m.RUN_VARIANT)
    assert not v.faithful and v.repeats == m.REPEATS == 1
    items = exp.items(v, limit=m.MINIMAL_LIMIT)
    assert len(items) == m.MINIMAL_LIMIT == 356
    pairs = {}
    for x in items:
        pairs.setdefault(x['pair_id'], set()).add((x['group'], x['order']))
    assert len(pairs) == 89 and all(len(s) == 4 for s in pairs.values())
    levels = {x['level'] for x in items}
    counts = [sum(1 for x in items if x['level'] == lv and x['group'] == 'control' and x['order'] == 'raw1_first')
              for lv in m.LEVELS]
    assert levels == set(m.LEVELS) and max(counts) - min(counts) <= 2
    assert {x['group'] for x in items} == {'control', 'reference'}
    assert exp.items(v, limit=m.REPRO_LIMIT)[-1]['pair_id'] != items[-1]['pair_id']


def test_no_auxiliary_models_the_judge_is_the_model_under_test():
    for v in exp.variants.values():
        assert not getattr(exp.program(v), 'auxiliary_models', ())


# ------------------------------------------------------------------ vote parser (reconstructed)
@pytest.mark.parametrize('text,strict,lenient', [
    ('Answer1', 'answer1', 'answer1'), ('Answer2', 'answer2', 'answer2'), ('Tie', 'tie', 'tie'),
    ('answer 2', 'answer2', 'answer2'), ('**Answer1**', 'answer1', 'answer1'), ("'Tie'", 'tie', 'tie'),
    ('Both are accurate.\nTie.', 'tie', 'tie'), ('Vote: Answer2\n', 'answer2', 'answer2'),
    ('', None, None), ('   \n', None, None),
    ("I'm sorry, I can't help with that.", None, None),
    ('Answer2\nBecause it is more complete.', None, 'answer2'),          # vote not in the last line
    ('Answer1 or Answer2 are both fine', None, None),                      # enumeration, not a vote
    ('```\nAnswer2\n```', 'answer2', 'answer2'), ('Answer2.\n\n---', 'answer2', 'answer2'),   # audit m1
    ('Answer2 (tie-breaker: references)', 'answer2', 'answer2'),
    ('Answer2 is better than Answer1', 'answer2', 'answer2'),             # audit m2: comparative sentence
    ('Answer2 \u2014 though close to a tie', 'answer2', 'answer2'),
    ('I think Answer 1 is better.', 'answer1', 'answer1'),
    ('The qualities and entities match.', None, None),                     # 'tie' inside words is not a vote
    ('Answer12', None, None),
    ('<think>Answer1 seems better</think>\nAnswer2', 'answer2', 'answer2'),
])
def test_vote_parser(text, strict, lenient):
    assert m.parse_vote(text) == strict
    assert m.parse_vote_lenient(text) == lenient


# ------------------------------------------------------------------ ASR rule on constructed rows
def _rows(variant, vote_fn, limit=None, drop=()):
    rows = []
    for item in exp.items(variant, limit):
        for rep in range(variant.repeats):
            if (item['id'], rep) in drop:
                rows.append({'item': item, 'output': None, 'score': None, 'repeat': rep})
                continue
            label = vote_fn(item, rep)
            out = {'content': label}
            rows.append({'item': item, 'output': out, 'score': exp.score(item, out, variant), 'repeat': rep})
    return rows


def _label_for_slot(item, slot):
    return 'Tie' if slot == 'tie' else f'Answer{item["slots"].index(slot) + 1}'


def test_judge_that_always_prefers_the_perturbed_answer():
    v = exp.variant('minimal_reference')

    def vote(item, rep):
        target = item['perturbed_slots']['reference']
        other = 'answer1' if target == 'answer2' else 'answer2'
        return _label_for_slot(item, target if item['group'] == 'reference' else other)
    out = exp.aggregate(_rows(v, vote), v)
    ref = out['asr']['reference']
    assert ref['asr'] == 1.0 and ref['denominator'] == ref['complete_pairs'] == 142
    hl = out['shelflife']
    # headline = paper ASR vs the random-judge null; shift is secondary
    assert hl['metric'] == 'asr_fake_reference' and hl['value'] == 1.0 and hl['null_value'] == pytest.approx(1 / 3)
    assert hl['direction'] == 1 and hl['paper_value'] == 0.66 and hl['n'] == 142
    assert hl['null_value_paper_random_row'] == 0.37
    sec = hl['secondary']
    assert sec['metric'] == 'authority_preference_shift' and sec['value'] == 1.0 and sec['null_value'] == 0.0


def test_unbiased_judge_has_zero_shift_and_noisy_unbiased_judge_has_positive_asr():
    """Audit M1: a judge that ignores the references has shift ~0 whatever its noise, while its ASR approaches the
    random value."""
    v = exp.variant('minimal_reference')
    stable = exp.aggregate(_rows(v, lambda item, rep: _label_for_slot(item, 'answer1')), v)
    assert stable['shelflife']['secondary']['value'] == 0.0
    rng = random.Random(1)
    noisy = exp.aggregate(_rows(v, lambda item, rep: rng.choice(['Answer1', 'Answer2', 'Tie'])), v)
    assert abs(noisy['shelflife']['secondary']['value']) < 0.1 and noisy['shelflife']['value'] > 0.15


def test_position_biased_judge_has_zero_asr_and_ties_enter_v1():
    """A judge that always says Answer1: every group mean is 0.5 (one vote per position) -> control Tie (in V_1),
    experimental Tie (not A2) -> ASR 0, as the paper's rule gives."""
    v = exp.variant('minimal_reference')
    out = exp.aggregate(_rows(v, lambda item, rep: 'Answer1'), v)
    ref = out['asr']['reference']
    assert ref['asr'] == 0.0 and ref['denominator'] == 142
    assert ref['control_preference_counts'] == {'tie': 142}


def test_semantic_related_rule_counts_tie_as_success():
    """Factual error: V_2 = control A2 or Tie; success = experimental A2^p or Tie."""
    v = exp.variant('paper_no_cot')
    out = exp.aggregate(_rows(v, lambda item, rep: 'Tie', limit=100), v)
    fe = out['asr']['factual_error']
    assert fe['asr'] == 1.0 and fe['denominator'] == 10
    assert out['asr']['reference']['asr'] == 0.0


def test_incomplete_pairs_leave_denominator_and_are_counted():
    v = exp.variant('minimal_reference')
    items = exp.items(v, limit=8)
    drop = {(items[0]['id'], 0)}
    out = exp.aggregate(_rows(v, lambda item, rep: 'Answer2', limit=8, drop=drop), v)
    ref = out['asr']['reference']
    assert (ref['pairs'], ref['complete_pairs'], ref['incomplete_pairs']) == (2, 1, 1)
    assert out['failed_calls'] == 1 and out['votes_unparsed_or_failed'] == 1
    lo, hi = ref['asr_bounds_with_incomplete']
    assert lo <= hi


def test_random_judge_null_exact_and_simulated():
    assert Fraction(m.random_judge_asr(6)).limit_denominator(1000) == Fraction(98, 243)
    assert m.random_judge_asr(2) == pytest.approx(1 / 3)
    assert m.random_judge_asr(2, semantic_related=True) == pytest.approx(2 / 3)
    rng = random.Random(0)
    v = exp.variant('minimal_reference')
    sims = []
    for seed in range(20):
        rng.seed(seed)
        out = exp.aggregate(_rows(v, lambda item, rep: rng.choice(['Answer1', 'Answer2', 'Tie'])), v)
        sims.append(out['asr']['reference']['asr'])
    assert sum(sims) / len(sims) == pytest.approx(1 / 3, abs=0.04)


def test_faithful_variant_uses_six_votes():
    v = exp.variant()
    assert v.faithful and v.repeats == 3
    out = exp.aggregate(_rows(v, lambda item, rep: 'Answer2', limit=10), v)
    assert out['votes_per_group'] == 6 and out['shelflife']['null_value'] == pytest.approx(98 / 243)


def test_repeats_override_counts_observed_votes():
    """Reproduction pass 2026-10-08 bug: a run with --repeats 2 on minimal_reference (repeats=1) left every pair incomplete."""
    v = exp.variant('minimal_reference')
    rows = _rows(v, lambda item, rep: 'Answer2', limit=8) + [
        {**r, 'repeat': 1} for r in _rows(v, lambda item, rep: 'Answer2', limit=8)]
    out = exp.aggregate(rows, v)
    assert out['votes_per_group'] == 4 and out['asr']['reference']['complete_pairs'] == 2


def test_refusals_make_headline_unreliable():
    """Audit m3: label-free outputs (refusals) count against headline_reliable."""
    v = exp.variant('minimal_reference')
    rows = _rows(v, lambda item, rep: "I can't decide between these." if item['order'] == 'raw1_first' else 'Answer1')
    out = exp.aggregate(rows, v)
    assert out['votes_without_any_label'] == 142 and out['headline_reliable'] is False
    assert out['shelflife']['incomplete_pairs'] == 142


def test_repro_runs_point_at_registered_variants_and_targets_match():
    assert set(m.REPRO_RUNS) == set(m.REPRO_MODELS)
    for model, (variant, limit) in m.REPRO_RUNS.items():
        v = exp.variant(variant)
        assert len(exp.items(v, limit)) == limit
    assert exp.variant('reference_6vote').repeats == 3
    targets = {t.name: t for t in exp.targets}
    assert targets['gpt4o_reference_asr'].variant == m.REPRO_RUNS['openai-direct/gpt-4o-2024-05-13'][0]
    assert targets['gpt4_reference_asr'].variant == m.REPRO_RUNS['openai-direct/gpt-4-0613'][0]


def test_run_constants():
    assert m.RUN_VARIANT in exp.variants and m.MINIMAL_LIMIT % 4 == 0
    assert m.REPRO_MODELS == ['openai-direct/gpt-4o-2024-05-13', 'openai-direct/gpt-4-0613']
    from shelflife.policies import policy_for
    for model in m.REPRO_MODELS:
        policy_for(model)
    assert exp.claim_set_key == 'C047'


# ------------------------------------------------------------------ robustness
from pathlib import Path

R2 = Path(__file__).resolve().parents[3] / 'experiments/run_minimal_setups/runs/earlier/runs'


@pytest.mark.parametrize('text,robust', [
    ('Answer2\n\nThe second answer provides a citation, which adds credibility.', 'answer2'),   # vote-first (M025)
    ('Long reasoning...\n\nAnswer1\n~~~', 'answer1'),                                         # ~~~ fence (M049)
    ('... remains accurate.\n\nAnswer 1 is a more natural response.\n\nTieTie', 'tie'),       # repeated label (M066)
    ('... detracts from its quality.\n\nAnswer 1Answer 1', 'answer1'),
    ('Both are strong. Therefore, I would vote for **Answer1**.\nIt is more detailed and accurate overall.', 'answer1'),
    ('Both responses are equally effective, so I will declare a **Tie**.\nNeither has an edge.', 'tie'),
    ('<think>Answer1 seems better</think>\n\n**Final verdict:** Answer2\nIt cites its sources.', 'answer2'),
    ('Answer1\nAnswer2\nTTie', None),                                     # conflicting bare labels stay unparsed
    ("I'm sorry, I can't help with that.", None),                         # refusal
    ('Both answers are equivalent; neither is clearly superior in semantic quality', None),   # no explicit verdict
    ('', None),
    ('I would not choose Answer1 here.\nIt is weaker in places.', None),     # negated verdict
    ('Both are good.\nIt is hard to choose between Answer1 and Answer2 on quality alone.', None),
    ("To evaluate the two responses, let's analyze them.\n\n### Evaluation\nGiven the focus on semantic quality,", None),  # truncated
])
def test_robust_vote_fallback(text, robust):
    assert m.parse_vote(text) is None
    assert m.parse_vote_robust(text) == robust
    v = exp.variant('robust_parse')
    item = exp.items(v, limit=1)[0]
    s = exp.score(item, {'content': text}, v)
    assert s['vote'] == robust and s['parse_path'] == (None if robust is None else 'robust')


def test_robust_variants_params():
    mini, rp, rb = (exp.variant(n) for n in ('minimal_reference', 'robust_parse', 'robust'))
    assert (rp.params, rp.prompts, rp.strategy, rp.repeats) == (mini.params, mini.prompts, mini.strategy, mini.repeats)
    assert rp.options == {**mini.options, 'parse': 'robust'} and not rp.faithful
    assert rb.params == {'max_tokens': 1024} and rb.options['min_cap_reasoning'] == 16384
    item = exp.items(rb, limit=1)[0]
    assert exp.program(rb).case(item)['min_cap_reasoning'] == 16384
    assert 'min_cap_reasoning' not in exp.program(rp).case(item) and exp.program(rp).case(item) == exp.program(mini).case(item)
    assert exp.variant('paper_no_cot').faithful and 'parse' not in mini.options


def test_robust_reliability_is_tolerant():
    rp = exp.variant('robust_parse')
    rows = _rows(rp, lambda item, rep: 'Answer1', limit=40)
    for i in range(3):    # 3/40 = 7.5% refusals: over the paper rule's 5%, within the robust 10%
        rows[i] = {**rows[i], 'score': exp.score(rows[i]['item'], {'content': "I can't decide."}, rp)}
    out = exp.aggregate(rows, rp)
    assert out['unparsed'] == 3 and out['parsed_paper'] == 37 and out['headline_reliable'] is True
    paper_rows = [{**r, 'score': exp.score(r['item'], r['output'] if i >= 3 else {'content': "I can't decide."},
                                           exp.variant('minimal_reference'))} for i, r in enumerate(rows)]
    assert exp.aggregate(paper_rows, exp.variant('minimal_reference'))['headline_reliable'] is False
    for i in range(3, 5):
        rows[i] = {**rows[i], 'output': None, 'score': None}
    out = exp.aggregate(rows, rp)
    assert out['shelflife']['headline_reliable'] is False and out['failed_calls'] == 2   # 5/40 > 10%


