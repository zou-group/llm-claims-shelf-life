"""C014 ReviewCritique: prompt reconstruction, parsers, archive replay, and rescoring of every earlier run."""
import collections
import json
import math
import random
import statistics

import pytest

from shelflife import registry
from shelflife.experiments import c014_reviewcritique as m
from shelflife.ledger import MockTransport
from shelflife.paths import INITIAL, HELPERS
from shelflife.replay import legacy_records, read_jsonl
from shelflife.runner import run
from shelflife.sourcecode import load_functions

exp = registry.get('C014')
REPRO = INITIAL / 'reproductions/C014'
TASK = HELPERS / 'C014'
MAIN = INITIAL / 'runs/2026-09-30-C014-remaining-expanded'
CAP8 = 'table14_reconstructed_v1_cap8192'
CAP4 = 'table14_reconstructed_v1'


def close(a, b, tol=1e-9):
    if a is None or b is None:
        return a is b
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(close(a[k], b[k], tol) for k in a)
    return math.isclose(a, b, rel_tol=0, abs_tol=tol)


# ----------------------------------------------------------------------------- prompts and items
@pytest.mark.parametrize('variant,files', [
    (CAP4, [REPRO / 'cases.jsonl', REPRO / 'cases_pilot2.jsonl', TASK / 'cases_cap4096.jsonl']),
    (CAP8, [REPRO / 'cases_cap8192.jsonl', REPRO / 'cases_pilot2_cap8192.jsonl', TASK / 'cases.jsonl']),
])
def test_cases_byte_identical_to_frozen_case_files(variant, files):
    """Every frozen cases file (initial and panel) equals our rendering in id, messages and params."""
    ours = {c['id']: c for c in exp.cases(exp.variant(variant))}
    for f in files:
        theirs = read_jsonl(f)
        assert theirs
        for c in theirs:
            assert ours[c['id']]['messages'] == c['messages'] and ours[c['id']]['params'] == c['params'], (f, c['id'])


def test_case_metadata_matches_frozen_reconstruction():
    """Example-review selection (seed 20260930), archive score vectors and decisions equal prepare.py's metadata."""
    frozen = {c['id']: c['metadata'] for c in read_jsonl(REPRO / 'cases.jsonl')}
    for item in exp.items(exp.variant(CAP4)):
        meta = frozen[item['id']]
        assert item['decision'] == meta['decision'] and item['title'] == meta['title']
        assert item['body_sha256'] == meta['body_sha256']
        assert item['archive_llm_recommendations'] == meta['archive_llm_recommendations']
        assert item['archive_human_recommendations'] == meta['archive_human_recommendations']
        assert [{k: v for k, v in e.items()} for e in item['example_reviews']] == [
            {k: v for k, v in e.items() if k != 'text_sha256'} for e in meta['examples']]
        assert m._sha(item['prompt'] + '\n\n' + item['title'] + '\n' + item['body_text']) == meta['prompt_sha256']


def test_template_matches_paper_text_and_held_out_examples():
    """Table 14 transcription equals the saved arXiv text (lines 201-224) word for word; examples are held out."""
    src = (m.INITIAL.parent / 'claim-sources/judges_reviewers/sources/'
           'arx_2406.16253.txt').read_text().splitlines()
    assert '\n'.join(line.split(': ', 1)[1] for line in src[200:224]) == m.TABLE14.text
    items = exp.items(exp.variant())
    titles = {x['title'].strip() for x in items}
    assert all(e['title'].strip() not in titles for e in items[0]['example_reviews'])
    assert [e['decision'] for e in items[0]['example_reviews']] == ['accepted', 'rejected']
    text = items[0]['prompt']
    assert '{' not in m.TABLE14.render(**{f: '' for f in m.TABLE14.fields()}).replace('[YES or NO]', '')
    # Both reconstructed examples end with "Recommendation: 6" (noted by the panel report as a possible anchor).
    assert text.count('\nRecommendation: 6\n') == 2
    assert collections.Counter(x['decision'] for x in items) == {'accepted': 10, 'rejected': 10}


def test_item_ids_and_decisions_follow_release_order():
    items = exp.items(exp.variant())
    assert [x['paper_id'] for x in items] == [f'p{i:02d}' for i in range(20)]
    released = [r['decision'] for r in m.load_data('ReviewCritique_LLM.jsonl')]
    assert [x['decision'] for x in items] == released
    assert items[0]['decision'] == 'rejected' and items[3]['decision'] == 'accepted'   # pilot papers p00 / p03


# ----------------------------------------------------------------------------- parsers


def test_panel_parser_copies_identical_to_initial_reproduction():
    for name in ('score.py', 'score_diagnostic.py', 'score_diagnostic_v2_1_posthoc.py', 'analyze.py'):
        assert (TASK / name).read_bytes() == (REPRO / name).read_bytes(), name


@pytest.mark.parametrize('text,expected', [
    ('Recommendation: 8', {'strict': 8, 'v1': 8, 'v2_1': 8, 'v3': 8}),
    ('- **Recommendation: 5**', {'strict': None, 'v1': 5, 'v2_1': 5, 'v3': 5}),
    ('## Recommendation\n\n**5: marginally below the acceptance threshold**', {'strict': None, 'v1': None, 'v2_1': 5, 'v3': 5}),
    ('## Recommendation\n**5** – marginally below the acceptance threshold', {'strict': None, 'v1': None, 'v2_1': 5, 'v3': 5}),
    ('### Recommendation\nRating: 8 (accept, good paper)', {'strict': None, 'v1': None, 'v2_1': None, 'v3': 8}),
    ('| Recommendation | 6 - marginally above |', {'strict': None, 'v1': None, 'v2_1': None, 'v3': 6}),
    ('Recommendation:\nI recommend acceptance with a rating of 8.', {'strict': None, 'v1': None, 'v2_1': None, 'v3': 8}),
    # rubric echo, conflicts, out of range, non-integer, prose sentiment, empty, correctness-only
    ('Recommendation:\n10: strong accept\n8: accept, good paper\n6: marginally above', dict.fromkeys(m.PARSERS)),
    ('Recommendation: 8\nRecommendation: 6', dict.fromkeys(m.PARSERS)),
    ('Recommendation: 11', dict.fromkeys(m.PARSERS)),
    ('Recommendation: 7.5', dict.fromkeys(m.PARSERS)),
    ('I recommend acceptance.', dict.fromkeys(m.PARSERS)),
    ('', dict.fromkeys(m.PARSERS)),
    ('Correctness: 4\nConfidence: 5', dict.fromkeys(m.PARSERS)),
    ('Recommendation: 8\n### Final Recommendation\n6: marginally above', {'strict': 8, 'v1': 8, 'v2_1': 8, 'v3': None}),
])
def test_parser_edge_cases(text, expected):
    item = exp.items(exp.variant(), limit=1)[0]
    assert exp.score(item, {'content': text}, exp.variant())['recommendation'] == expected


def test_parsers_ignore_decision_labels():
    items = exp.items(exp.variant(), limit=4)
    acc, rej = items[3], items[0]
    for text in ('Recommendation: 8', '### Recommendation\nRating: 6 (marginal)'):
        assert exp.score(acc, {'content': text}, exp.variant())['recommendation'] == \
               exp.score(rej, {'content': text}, exp.variant())['recommendation']


def test_archive_reader_matches_prior_and_handles_gemini_key_anomaly():
    """archive_anchor.recommendation: 'Recommendation 8' key (Gemini, one paper) reads as 8; human 'rating' fields."""
    assert m.archive_recommendation({'Recommendation 8': ' accept, good paper'}) == 8.0
    assert m.archive_recommendation({'Recommendation': ' 6: marginally above'}) == 6.0
    assert m.archive_recommendation({'rating': '3: reject'}) == 3.0
    assert m.archive_recommendation({'Recommendation': ' 8', 'rating': '6'}) is None    # conflicting -> missing
    assert m.archive_recommendation({'Confidence': '4'}) is None


# ----------------------------------------------------------------------------- archive replay
def test_archive_summary_matches_archive_anchor_stats():
    """archive_anchor_stats.json: counts and review-weighted means for llm20, human100, human_matched20."""
    stats = json.loads((REPRO / 'archive_anchor_stats.json').read_text())
    for which in ('llm20', 'human100', 'human_matched20'):
        ours = exp.archive_summary(which)
        assert {k: ours[k] for k in stats[which]['counts']} == stats[which]['counts']
        for d, v in stats[which]['review_weighted_decision_means'].items():
            assert ours['n_scored'][d] == v['n'] and math.isclose(ours['mean'][d], v['mean'])
    assert exp.archive_summary('human_matched20')['n_scored'] == {'accepted': 39, 'rejected': 37}


def test_matched_archive_paper_weighted_baseline():
    """remaining_matched_archive_baseline.json: human paper means 6.3933/4.7917 (AUC .905); LLM pool 7.467/7.533."""
    base = json.loads((REPRO / 'remaining_matched_archive_baseline.json').read_text())['baselines']
    items = exp.items(exp.variant())
    h = {d: [statistics.mean(x['archive_human_recommendations']) for x in items if x['decision'] == d] for d in ('accepted', 'rejected')}
    assert math.isclose(statistics.mean(h['accepted']), base['human_matched20']['accepted_paper_mean'])
    assert math.isclose(statistics.mean(h['rejected']), base['human_matched20']['rejected_paper_mean'])
    assert m.auc(h['accepted'], h['rejected']) == base['human_matched20']['decision_auc'] == 0.905
    for key in ('claude_opus', 'gpt4', 'gemini_pro_1.5'):
        for d in ('accepted', 'rejected'):
            vals = [x['archive_llm_recommendations'][key] for x in items if x['decision'] == d]
            assert math.isclose(statistics.mean(vals), base[key][f'{d}_paper_mean'])


# ----------------------------------------------------------------------------- prior-run rescoring
def _initial_analysis(path, key):
    return {r['model']: r for r in json.loads((REPRO / path).read_text())[key]['models']} if key else \
        {r['model']: r for r in json.loads((REPRO / path).read_text())['models']}


def _check_against_analyze(ours, theirs):
    """Compare our per-parser summary with initial analyze.summarize output for one model."""
    g = theirs['groups']
    assert ours['complete_papers'] == {d: g[d]['complete_papers'] for d in ('accepted', 'rejected')}
    assert close(ours['mean_accepted'], g['accepted']['mean_complete_paper_scores'])
    assert close(ours['mean_rejected'], g['rejected']['mean_complete_paper_scores'])
    assert close(ours['difference'], theirs['accepted_minus_rejected_score_difference'])
    assert close(ours['auc'], theirs['decision_auc_complete_paper_mean_scores'])
    assert close(ours['difference_missingness_bounds'], theirs['difference_missingness_bounds'])
    assert [p['scores'] for p in ours['papers']] == [p['scores_by_repeat'] for p in theirs['papers']]
    if min(ours['complete_papers'].values()) >= 2:
        assert close(ours['difference_ci95'], theirs['stratified_paper_cluster_difference_percentile95'])
        assert close(ours['auc_ci95'], theirs['stratified_paper_cluster_auc_percentile95'])
    else:
        # analyze.py emitted a zero-width interval from one paper per group; we report none (documented).
        assert ours['difference_ci95'] is None
        ci = theirs['stratified_paper_cluster_difference_percentile95']
        assert ci is None or ci[0] == ci[1]


@pytest.mark.parametrize('model', ['anthropic/claude-sonnet-5.5', 'openai-direct/gpt-6.1-sol', 'z-ai/glm-5.3'])
def test_main_run_rescoring_reproduces_initial_analyses(model):
    """remaining_main_analysis.json (strict primary + v1) and remaining_main_posthoc_format_v2_1.json, all fields."""
    ours = exp.rescore_prior(MAIN, CAP8, model=model)
    _check_against_analyze(ours['by_parser']['strict'], _initial_analysis('remaining_main_analysis.json', 'primary')[model])
    _check_against_analyze(ours['by_parser']['v1'],
                           _initial_analysis('remaining_main_analysis.json', 'explicit_bullet_diagnostic_v1')[model])
    _check_against_analyze(ours['by_parser']['v2_1'], _initial_analysis('remaining_main_posthoc_format_v2_1.json', None)[model])


def test_main_run_headline_numbers_in_review():
    """REMAINING_EXPANDED_REVIEW.md: v1 Sonnet 55/60, 8/7 papers, -0.387 [-1.131, 0.232], AUC 0.420 [0.161, 0.705];
    native 60/60, 0.167 [-0.467, 0.800], AUC 0.58; GLM 29/60, bounds [-8.133, 8.067]; strict 53/1/26;
    v2.1 Sonnet 5.033/5.167, -0.133 [-0.767, 0.500], AUC 0.495 [0.275, 0.740]."""
    s = exp.rescore_prior(MAIN, CAP8, model='anthropic/claude-sonnet-5.5')['by_parser']
    g = exp.rescore_prior(MAIN, CAP8, model='openai-direct/gpt-6.1-sol')['by_parser']
    z = exp.rescore_prior(MAIN, CAP8, model='z-ai/glm-5.3')['by_parser']
    r3 = lambda xs: [round(x, 3) for x in xs]
    assert (s['strict']['explicit_scores'], g['strict']['explicit_scores'], z['strict']['explicit_scores']) == (53, 1, 26)
    assert (s['v1']['explicit_scores'], g['v1']['explicit_scores'], z['v1']['explicit_scores']) == (55, 60, 29)
    assert s['v1']['complete_papers'] == {'accepted': 8, 'rejected': 7}
    assert r3([s['v1']['difference'], *s['v1']['difference_ci95'], s['v1']['auc'], *s['v1']['auc_ci95']]) == \
        [-0.387, -1.131, 0.232, 0.42, 0.161, 0.705]
    assert r3([g['v1']['difference'], *g['v1']['difference_ci95'], *g['v1']['auc_ci95']]) == [0.167, -0.467, 0.8, 0.355, 0.795]
    assert z['v1']['complete_papers'] == {'accepted': 1, 'rejected': 1}
    assert r3(z['v1']['difference_missingness_bounds']) == [-8.133, 8.067]
    v = s['v2_1']
    assert r3([v['mean_accepted'], v['mean_rejected'], v['difference'], *v['difference_ci95'], v['auc'], *v['auc_ci95']]) == \
        [5.033, 5.167, -0.133, -0.767, 0.5, 0.495, 0.275, 0.74]
    # v3 changes nothing for the reused models (panel REPORT.md claim)
    for b in (s, g, z):
        assert [p['scores'] for p in b['v3']['papers']] == [p['scores'] for p in b['v2_1']['papers']]


def test_cap8192_pilots_reproduce_all_pilot_analysis():
    """remaining_all_pilot_analysis.json: routed cap8192 pilot + native cap8192 pilot, repeats=1, papers p00/p03."""
    papers = {'p00', 'p03'}
    for key, parser in (('primary', 'strict'), ('explicit_bullet_diagnostic_v1', 'v1')):
        theirs = _initial_analysis('remaining_all_pilot_analysis.json', key)
        for model, run_dir in (('anthropic/claude-sonnet-5.5', 'cap8192-pilot'), ('z-ai/glm-5.3', 'cap8192-pilot'),
                               ('openai-direct/gpt-6.1-sol', 'remaining-native-pilot')):
            ours = exp.rescore_prior(INITIAL / f'runs/2026-09-30-C014-{run_dir}', CAP8, model=model, repeats=1,
                                     paper_ids=papers)['by_parser'][parser]
            t = theirs[model]
            # the pilot cases file lists accepted p03 before rejected p00
            assert [p['scores'] for p in ours['papers']] == [p['scores_by_repeat'] for p in sorted(t['papers'], key=lambda p: p['paper_id'])]
            assert close(ours['difference'], t['accepted_minus_rejected_score_difference'])
    # REMAINING_PILOT_REVIEW.md: native p00=5, p03=6 (v1); Sonnet 5/6; GLM 6/6
    got = {model: exp.rescore_prior(INITIAL / f'runs/2026-09-30-C014-{d}', CAP8, model=model, repeats=1, paper_ids=papers)
           ['by_parser']['v1']['papers'] for model, d in (('openai-direct/gpt-6.1-sol', 'remaining-native-pilot'),
                                                           ('anthropic/claude-sonnet-5.5', 'cap8192-pilot'),
                                                           ('z-ai/glm-5.3', 'cap8192-pilot'))}
    assert {k: [p['scores'][0] for p in v] for k, v in got.items()} == {
        'openai-direct/gpt-6.1-sol': [5, 6], 'anthropic/claude-sonnet-5.5': [5, 6], 'z-ai/glm-5.3': [6, 6]}


def test_cap4096_pilots_match_live_pilot_review():
    """LIVE_PILOT_REVIEW.md: Sonnet p00=6, p03 truncated; native 5/5 under v1 (0/2 strict); GLM no score."""
    papers = {'p00', 'p03'}
    sc = lambda run_dir, model, parser: [p['scores'][0] for p in exp.rescore_prior(
        INITIAL / f'runs/2026-09-30-C014-{run_dir}', CAP4, model=model, repeats=1, paper_ids=papers)['by_parser'][parser]['papers']]
    assert sc('pilot', 'anthropic/claude-sonnet-5.5', 'strict') == [6, None]
    assert sc('pilot', 'z-ai/glm-5.3', 'v3') == [None, None]
    assert sc('directGPT6-pilot', 'openai-direct/gpt-6.1-sol', 'strict') == [None, None]
    assert sc('directGPT6-pilot', 'openai-direct/gpt-6.1-sol', 'v1') == [5, 5]


PANEL_RESULTS = json.loads((TASK / 'results.json').read_text())
PANEL_KEYS = ('explicit_scores', 'complete_papers', 'mean_accepted', 'mean_rejected', 'difference', 'auc',
              'difference_ci95', 'auc_ci95', 'available_repeat')


@pytest.fixture(scope='module')
def panel_slots():
    return exp.panel_slots()


@pytest.mark.parametrize('row', PANEL_RESULTS['models'], ids=[r['record_id'] for r in PANEL_RESULTS['models']])
def test_panel_results_reproduced_by_rescoring(row, panel_slots):
    """results.json per model and parser (strict, v1, v2.1, v3): per-paper scores and every summary statistic.

    Missingness bounds agree to 1e-12 only: results.json carries values (e.g. M119 v3 0.0) that the current
    analyze_panel.py itself computes as -8.9e-16, i.e. the file came from an arithmetically reordered earlier
    revision; floating-point noise only."""
    ours = exp.panel_metrics(row['record_id'], panel_slots)['by_parser']
    for parser in m.PARSERS:
        theirs = row['by_parser'][parser]
        assert [p['scores'] for p in ours[parser]['papers']] == [p['scores'] for p in theirs['papers']], parser
        for key in PANEL_KEYS:
            assert ours[parser].get(key) == theirs.get(key), (parser, key)
        assert close(ours[parser]['difference_missingness_bounds'], theirs['difference_missingness_bounds'], 1e-12)
        assert close(ours[parser]['review_weighted_mean'], theirs['overall_mean_score'])


def test_panel_pooled_estimates_and_human_baseline(panel_slots):
    """results.json panel_summary_v3 (24 complete models, +0.068 [-0.265, +0.422]) and audit_2026_10_02
    (all-29 available-repeat +0.08 [-0.25, +0.44]; human +1.60 [0.875, 2.342], AUC 0.905; signs 18/8/3, p=0.08)."""
    ours = exp.panel_summary('v3', panel_slots)
    ps = PANEL_RESULTS['panel_summary_v3']
    assert ours['complete_models'] == ps['models_all_papers_complete']
    pc = ours['pooled_complete']
    assert [pc['mean_accepted'], pc['mean_rejected'], pc['difference'], pc['difference_ci95']] == \
        [ps['panel_mean_accepted'], ps['panel_mean_rejected'], ps['panel_difference'], ps['panel_difference_ci95_paper_cluster']]
    audit = PANEL_RESULTS['audit_2026_10_02']
    pa, tp = ours['pooled_available_repeat'], audit['pooled_all_estimable_available_repeat_v3']
    assert pa['n_models'] == tp['n_models'] == 29
    assert [pa['difference'], pa['difference_ci95'], pa['min_models_per_paper']] == \
        [tp['difference'], tp['difference_ci95_paper_cluster'], tp['min_models_per_paper']]
    h, th = ours['human_paper_weighted'], audit['human_baseline']
    assert [h['difference'], h['difference_ci95'], h['auc']] == [th['difference'], th['difference_ci95_paper_cluster'], th['auc']]
    sign = audit['sign_pattern_v3']
    assert (ours['sign']['positive'], ours['sign']['negative'], ours['sign']['zero']) == (sign['positive'], sign['negative'], sign['zero']) == (18, 8, 3)
    assert ours['sign']['p_two_sided'] == sign['exact_two_sided_sign_test_p_excluding_ties']
    assert ours['models_ci_excluding_zero'] == [] == ps['models_with_ci_excluding_zero']


def test_panel_slot_merge_provenance(panel_slots):
    """Every merged slot's messages equal ours; 31 deployments; GPT-4o slots all come from the cap4096 run;
    recovery records fill exactly the 21 slots they recovered (one recovery stayed empty and is retained)."""
    rids = collections.Counter(k[0] for k in panel_slots)
    assert len(rids) == 31 and set(rids.values()) == {60}
    assert {d['case']['params']['max_tokens'] for k, d in panel_slots.items() if k[0] == 'M001'} == {4096}
    rec = [d for d in panel_slots.values() if 'recovers' in d]
    assert len(rec) == 22 and collections.Counter(d['status'] for d in rec) == {'ok': 21, 'empty': 1}
    assert not any(d['status'] in ('error', 'not_submitted') for d in panel_slots.values())


# ----------------------------------------------------------------------------- program replay
def test_program_replays_saved_main_run_requests(tmp_path):
    """Run the program on two papers with a MockTransport returning saved Sonnet outputs: identical messages and
    max_tokens reach the transport, and scores equal the stored strict scores."""
    saved = {(r['case_id'], r['repeat']): r for r in legacy_records(MAIN) if r['model'] == 'anthropic/claude-sonnet-5.5'}
    items = [x for x in exp.items(exp.variant()) if x['paper_id'] in ('p00', 'p03')]
    by_content = {}
    for x in items:
        by_content[exp.program(exp.variant()).build(x)[0]['content']] = x['id']

    def responder(payload):
        return saved[(by_content[payload['messages'][-1]['content']], 0)]['content']
    t = MockTransport(responder)
    out = run(exp, exp.variant(), ['M001'], tmp_path / 'run', ledger=tmp_path / 'l.sqlite3', ceiling_usd=100,
              transport=t, items=items, repeats=1, progress_every=0)
    assert [r['status'] for r in out['rows']] == ['complete', 'complete'], [r.get('error') for r in out['rows']]
    assert len(t.payloads) == 2
    for p in t.payloads:
        case_id = by_content[p['messages'][-1]['content']]
        assert p['messages'] == saved[(case_id, 0)]['case']['messages']
    for row in out['rows']:
        assert row['score']['strict'] == saved[(row['item_id'], 0)]['score']


def test_panel_report_table_matches_rescoring(panel_slots):
    """Every row of tasks/C014/REPORT.md's v3 results table (means, difference, CI, AUC, parsed counts with the v1
    count in parentheses, complete papers) equals our rescoring to the printed precision. GLM-5.3 (1 paper/group)
    and M196 print 'not estimable' where we report no interval."""
    import re
    lines = [l for l in (TASK / 'REPORT.md').read_text().splitlines() if re.match(r'\| [^|]+ \| (M\d{3}|—) \|', l)]
    assert len(lines) == 31
    f = lambda x: 'n/a' if x is None else f'{x:.2f}'
    for line in lines:
        c = [x.strip() for x in line.strip('|').split('|')]
        rid = 'GLM-5.3' if c[1] == '—' else c[1]
        by = exp.panel_metrics(rid, panel_slots)['by_parser']
        v = by['v3']
        ci = v['difference_ci95']
        ci_text = 'not estimable' if ci is None else f'[{ci[0]:+.2f}, {ci[1]:+.2f}]'
        parsed = f"{v['explicit_scores']}/60" + ('' if rid in ('GLM-5.3', 'M196') else f" ({by['v1']['explicit_scores']})")
        estimable = ci is not None
        mine = [f(v['mean_accepted']), f(v['mean_rejected']), f(v['difference']) if estimable else 'n/a', ci_text,
                f(v['auc']) if estimable else '—', parsed,
                f"{v['complete_papers']['accepted']}/{v['complete_papers']['rejected']}"]
        theirs = [c[3], c[4], c[5], c[6].rstrip('†*').replace('[0, 0]', '[+0.00, +0.00]'), c[7], c[8], c[9]]
        assert mine == theirs, rid


def test_recovery_record_cannot_overwrite_a_usable_slot(tmp_path, monkeypatch):
    """Audit minor 3: analyze_panel.load() lets any non-error recovery record replace its slot. The frozen panel's
    recovery run only covers error/empty slots (16 empty + 5 error -> ok, 1 empty -> empty), so our merge is
    identical there; a recovery record aimed at a usable slot now raises instead of silently replacing it."""
    sonnet = next(r for r in read_jsonl(MAIN / 'responses.jsonl') if r['model'] == 'anthropic/claude-sonnet-5.5' and r['status'] == 'ok')
    run_dir = tmp_path / 'runs/recovery-fake'
    run_dir.mkdir(parents=True)
    (run_dir / 'responses.jsonl').write_text(json.dumps({**sonnet, 'record_id': 'M206', 'content': 'replacement'}) + '\n')
    monkeypatch.setattr(m, 'PANEL_TASK', tmp_path)
    monkeypatch.setattr(m, 'PANEL_RUNS', ('runs/recovery-fake',))
    with pytest.raises(ValueError, match='would overwrite usable slot'):
        exp.panel_slots()


def test_reconstruction_choices_are_declared_deviations():
    """Audit minors 1-2: the shared 'Recommendation: 6' anchor and the paper-join separator are declared."""
    _, examples = exp._prompt
    assert all('Recommendation: 6' in e['text'] for e in examples)
    for v in exp.variants.values():
        devs = ' '.join(v.deviations)
        assert 'Recommendation: 6' in devs and 'title + "\\n" + body_text' in devs
    x = exp.items(exp.variant(), limit=1)[0]
    assert exp.program(exp.variant()).build(x)[0]['content'] == x['prompt'] + '\n\n' + x['title'] + '\n' + x['body_text']


# ----------------------------------------------------------------------------- robustness
from shelflife.paths import EXPERIMENTS  # noqa: E402

ROBUST_SOURCE_RUNS = (
    *(EXPERIMENTS / f'earlier/runs/C014-{CAP8}-Lall-r5/{mod}' for mod in 'M049 M066 M132 M171 M183 M190'.split()),
    EXPERIMENTS / f'earlier/runs/C014-{CAP8}-Lall-w1b/M204',
    EXPERIMENTS / f'run_minimal_setups/runs/C014-{CAP8}-Lall-gap/M196',
)


@pytest.mark.parametrize('text,expected', [
    # prose inside the Recommendation section (M066 layouts)
    ('### Recommendation\n\nBased on the review, I recommend **6** (marginally above the acceptance threshold).', 6),
    ('### Recommendation\n\nThe paper is recommended for acceptance. The recommendation is **6**, indicating ...', 6),
    ('### Recommendation\n\nThe paper is well written. Therefore, I rate the recommendation as: **8**\n\n'
     '### Confidence\n\nI rate my confidence as: **4**', 8),
    # Overall score field (GLM-5.3-Flash), bracketed and sub-labelled values (gpt-4-turbo)
    ('### Overall Score: **6/10 — marginally above the acceptance threshold**\n\nSolid paper.', 6),
    ('**Recommendation:** [8] – The paper is a good fit for the conference.', 8),
    ('**Recommendation:**\n- Choice: 8 (Accept, good paper)\n', 8),
    ('### Recommendation\n- **Recommendation Level:** 8 - accept, good paper.', 8),
    ('<think>Recommendation: 3 maybe?</think>\n### Recommendation\n\nI recommend **8**.', 8),
    # stays unparsed: sentiment only, the confidence value must not leak, enumerations, truncation, conflicts
    ('**Recommendation: Accept (borderline), conditional on:** (i) reporting accuracy', None),
    # prose about another field, or several different numbers, is not a recommendation value
    ('## Recommendation\nOverall, I would rate soundness 4 and novelty 8.', None),
    ('## Recommendation\nTherefore, I recommend 6, though a 7 would also be fair.', None),
    ('## Recommendation\nOn a scale of 1 to 10, I recommend **6**.', 6),
    ('**Recommendation**\n\n* I would recommend the paper for acceptance.\n\n**Confidence**\n\n* 5', None),
    ('**Recommendation:** **8 – Accept**\n**Key Reasons:** (1) strong evidence; (2) new dataset', 8),
    ('### 7. Recommendation Summary\nThe submission convincingly demonstrates gains in Table 3.', None),
    ('## Strengths\n1. Novel task\n## Weaknesses\n2. Missing baselines\n## Recommen', None),
    ('### Recommendation\nI recommend 6.\n### Overall Score: 8/10', None),
    ('### Recommendation\nI would rate it 11.', None),
    ('', None),
])
def test_robust_fallback_modern_formats(text, expected):
    assert m.robust_recommendation(text) == expected


@pytest.mark.parametrize('text,value,path', [
    ('Recommendation: 8', 8, 'parsed_paper'),
    ('### Recommendation\n\nI recommend **6** overall.', 6, 'parsed_robust'),
    ('I recommend acceptance.', None, 'unparsed'),
    ('', None, 'empty'),
    # v3 flags conflicting explicit fields: the fallback must not pick one
    ('Recommendation: 8\n### Final Recommendation\n6: marginally above', None, 'unparsed'),
])
def test_robust_scoring_paths(text, value, path):
    item = exp.items(exp.variant(), limit=1)[0]
    s = exp.score(item, {'content': text}, exp.variant('robust_parse'))
    assert s['recommendation']['robust'] == value and s['parse_status']['robust'] == path
    paper = exp.score(item, {'content': text}, exp.variant())
    assert 'robust' not in paper['recommendation'] and paper['recommendation'] == {
        k: v for k, v in s['recommendation'].items() if k != 'robust'}


def _saved_texts():
    out = []
    for run_dir in ROBUST_SOURCE_RUNS:
        for f in sorted(run_dir.glob('responses*.jsonl')):
            out += [(run_dir.name, r['item_id'].split(':')[1], r['output'].get('content') or '')
                    for r in read_jsonl(f) if r['status'] == 'complete']
    for (rid, pid, _), rec in exp.panel_slots().items():
        if rec['status'] in ('ok', 'truncated'):
            out.append((rid, pid, rec.get('content') or ''))
    return out


def test_robust_equals_paper_wherever_paper_parses_on_saved_outputs():
    """Over every saved the panel (round2 + gap run) and panel output: parse='robust' returns the v3 value wherever v3
    parses, and the fallback alone never contradicts v3 (it agrees or returns None); it recovers a few M066/M196
    prose layouts."""
    texts = _saved_texts()
    assert len(texts) > 2000
    items = {x['paper_id']: x for x in exp.items(exp.variant())}
    rv = exp.variant('robust_parse')
    paths, agree = collections.Counter(), 0
    for _, pid, text in texts:
        s = exp.score(items[pid], {'content': text}, rv)
        v3 = s['recommendation']['v3']
        paths[s['parse_status']['robust']] += 1
        if v3 is not None:
            assert s['recommendation']['robust'] == v3
            fb = m.robust_recommendation(text)
            assert fb in (None, v3)
            agree += fb == v3
    assert paths['parsed_paper'] > 2000 and paths['parsed_robust'] >= 6 and agree > 1000


def test_robust_variants_requests_and_params():
    base, rp, rb = exp.variant(CAP8), exp.variant('robust_parse'), exp.variant('robust')
    assert not rp.faithful and not rb.faithful and base.faithful
    assert rp.params == base.params == rb.params == {'max_tokens': 8192}
    assert rp.options['parse'] == rb.options['parse'] == 'robust' and 'parse' not in base.options
    assert rb.options['min_cap_reasoning'] == m.ROBUST_MIN_CAP_REASONING == 16384
    cb, cp, cr = (exp.cases(v) for v in (base, rp, rb))
    assert cb == cp                                               # robust_parse: identical requests (offline rescore)
    assert [{k: v for k, v in c.items() if k != 'min_cap_reasoning'} for c in cr] == cb
    assert all(c['min_cap_reasoning'] == 16384 for c in cr) and all('min_cap_reasoning' not in c for c in cb)


def test_robust_aggregate_counts_and_reliability():
    rv = exp.variant('robust_parse')
    items = exp.items(rv)
    rows = []
    for i, item in enumerate(items):
        for rep in range(3):
            if (i, rep) == (0, 0):
                continue                                          # one missing call
            text = 'Recommendation: 8' if i % 2 else '### Recommendation\nI recommend **6**.'
            if (i, rep) in ((1, 0), (2, 0), (3, 0), (4, 0), (5, 0)):
                text = 'I recommend acceptance.'                  # five unparsed
            rows.append({'item': item, 'repeat': rep, 'score': exp.score(item, {'content': text}, rv)})
    out = exp.aggregate(rows, rv)
    assert (out['parsed_paper'], out['parsed_robust'], out['unparsed'], out['missing']) == (27, 27, 5, 1)
    assert out['headline_reliable'] is True and out['validity_rate'] == 54 / 60 and out['parser'] == 'robust'
    rows2 = [r for r in rows if not (r['item']['paper_id'] == 'p19' and r['repeat'] == 2)]
    assert exp.aggregate(rows2, rv)['headline_reliable'] is False  # 7/60 > 10%
    paper = exp.aggregate(rows, exp.variant())
    assert 'headline_reliable' not in paper and paper['parser'] == 'v3'


@pytest.mark.parametrize('text,expected', [
    ('## Recommendation (1/3/5/6/8/10)\n**6: marginally above the acceptance threshold**\n## Confidence\n4', 6),
    ('## Recommendation (1–10)\n**5: marginally below the acceptance threshold**', 5),
    ('## Recommendation (choose one)\n**8: accept, good paper**', 8),
    ('**Recommendation (1,3,5,6,8,10): 6**', 6),
    ('## Recommendation (1–10)\n**6: marginally above threshold**\nKey reasons:\n1. Strong empirical contribution\n2. Some limitations.', 6),
    ('## Recommendation (1–10)\n6: marginally above threshold\n## Final Recommendation\n8: accept, good paper', None),
    ('## Recommendation (choose one)\n6 or 8', None),
    ('## Recommendation (choose one)\n1: strong reject\n3: reject\n5: marginally below threshold\n6: marginally above threshold\n8: accept, good paper', None),
    ('## Recommendation (choose one)\nMarginally above threshold', None),
    ('## Recommendation (1–10)\n11: excellent', None),
    ('## Confidence\n4: confident', None),
    ('> ## Recommendation (choose one)\n> 6: above threshold', None),
    ('```\n## Recommendation (choose one)\n6: above threshold\n```', None),
    # A plain heading cannot acquire its chosen value from a quoted/code example.
    ('## Recommendation (choose one)\n```\n6: marginally above threshold\n```', None),
    ('## Recommendation (choose one)\n> 6: marginally above threshold', None),
    ('## Recommendation (choose one)\n~~~~python\n6: marginally above threshold\n~~~~', None),
    # Keep a real field's chosen value while ignoring an explicitly quoted example.
    ('## Recommendation (choose one)\n6: marginally above threshold\n\n> ## Recommendation (choose one)\n> 8: accept, good paper', 6),
    ('## Recommendation (choose one)\n6: marginally above threshold\n\n```\n## Recommendation (choose one)\n8: accept, good paper\n```', 6),
    # Excluding an example cannot hide a separate, real conflicting final score.
    ('## Recommendation (choose one)\n6: marginally above threshold\n> Example: 6\n## Final Recommendation (choose one)\n8: accept, good paper', None),
    ('## Recommendation (choose one)\n6: marginally above threshold\nMy final recommendation is 8.', None),
    ('## Recommendation (choose one)\n6: marginally above threshold\nI recommend 8.', None),
    ('## Recommendation (choose one)\n6: marginally above threshold\nI do not recommend 6; instead 8.', None),
    ('## Recommendation (choose one)\n6: marginally above threshold\nI recommend 6 rather than 8.', 6),
])
def test_robust_recommendation_heading_metadata_and_numbered_reasons(text, expected):
    # Literal field values are independent of the implementation and paper outcomes.
    item = exp.items(exp.variant('robust_parse'))[0]
    result = exp.score(item, {'content': text}, exp.variant('robust_parse'))
    assert result['recommendation']['robust'] == expected
    # The original released parser always remains the paper measurement.
    assert result['recommendation']['v3'] == m.parsers()['v3'](text, item)['parsed']


@pytest.mark.parametrize('text', [
    '> ## Recommendation (choose one)\n> 6: above threshold',
    '```\n## Recommendation (choose one)\n6: above threshold\n```',
    '## Recommendation (choose one)\n```\n6: marginally above threshold\n```',
    '## Recommendation (choose one)\n> 6: marginally above threshold',
])
def test_new_recommendation_recovery_ignores_quoted_and_fenced_examples(text):
    assert m._first_numeric_field(m._normalize_field_metadata(text)) is None
