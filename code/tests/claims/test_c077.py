"""C077 gender-stereotyped pronoun resolution: prompts against the paper, the two-turn session + LLM coder program
against every saved panel request, coder label parsing/mapping, and rescoring of the panel results."""
import collections
import functools
import json
from pathlib import Path

import pytest

from shelflife import registry
from shelflife.experiments import c077_gender_pronoun_bias as m
from shelflife.ledger import LedgerClient, MockTransport
from shelflife.replay import read_jsonl
from shelflife.runner import run
from shelflife.sourcecode import load_functions

exp = registry.get('C077')
TASK = m.TASK
RESULTS = json.loads((TASK / 'results.json').read_text())
BY_MODEL = {x['record_id']: x for x in RESULTS['models']}


# --------------------------------------------------------------------------------------------- prompts
def build_cases():
    return load_functions(TASK / 'build_cases.py', ['cap', 'prompt'], constants=('SCHEMAS',))


def test_schemas_derived_from_paper_equal_panel_reconstruction():
    """Templates and questions derived from the printed sentences equal build_cases.SCHEMAS (all 15)."""
    assert m.schemas() == [tuple(s) for s in build_cases()['SCHEMAS']]


def test_printed_sentence_one_is_rendered_exactly():
    item = next(x for x in exp.items(exp.variant()) if x['id'] == 'S01_MF_she')
    assert m.turn1_text(item, exp.variant()) == m.PRINTED.text.split('\n')[0]
    # every printed sentence 2-15 is the MF/she sentence of its schema (modulo the final period and schema 14's article)
    lines = m.PRINTED.text.strip().split('\n')
    for i, line in enumerate(lines[1:], 2):
        item = next(x for x in exp.items(exp.variant()) if x['id'] == f'S{i:02d}_MF_she')
        expected = line.rstrip('.').replace('consulted with florist', 'consulted with the florist')
        assert item['sentence'] == expected, i


def test_followup_matches_printed_schema():
    printed = m.FOLLOWUP_PRINTED.text
    for pron in ('he', 'she'):
        ours = m.FOLLOWUP.render(pronoun=pron)
        assert printed.replace('“', '').replace('”', '').replace('{"he", "she"}', pron) == ours


def test_main_items_equal_panel_case_files():
    main = {x['id']: x for x in read_jsonl(TASK / 'cases_main.jsonl')}
    v = exp.variant()
    items = exp.items(v)
    assert [x['id'] for x in items] == list(main)
    for x in items:
        c = main[x['id']]
        assert exp.program(v).preview(x) == c['messages'] and c['params'] == v.params
        assert m.followup_text(x, v) == c['meta']['followup']
        assert {k: x[k] for k in c['meta'] if k != 'followup'} == {k: v_ for k, v_ in c['meta'].items() if k != 'followup'}


def test_coder_sees_subject_as_A_and_never_the_stereotype():
    v = exp.variant()
    item = next(x for x in exp.items(v) if x['id'] == 'S01_FM_she')
    msgs = m.coder_messages(item, 'r1', 'r2', v)
    assert msgs[1]['content'].startswith('PERSON_A: nurse\nPERSON_B: doctor\n')
    assert 'congruent' not in msgs[1]['content'] and 'stereotyp' not in msgs[0]['content'].lower()
    row = exp.session_row(item, {'t1_choice': 'A', 't1_ambiguity': False, 't2_answer': 'yes', 'explanation': 'subject'})
    assert row['cong'] and row['fem'] and row['subj'] and row['pick']
    row = exp.session_row(item, {'t1_choice': 'B', 't1_ambiguity': True, 't2_answer': 'no', 'explanation': 'context'})
    assert row['incong'] and row['male'] and not row['subj']
    row = exp.session_row(item, {'t1_choice': 'none', 't1_ambiguity': True, 't2_answer': 'yes', 'explanation': 'ambiguous'})
    assert not row['pick'] and row['none'] and not row['cong']


# ----------------------------------------------------------------------------------- saved panel requests
@functools.lru_cache(maxsize=None)
def raw_index():
    out = {}
    for d in (TASK / 'runs').iterdir():
        if (d / 'raw').is_dir():
            for f in (d / 'raw').glob('*.json'):
                out[f.stem] = f
    return out


def raw(key):
    return json.loads(raw_index()[key].read_text())


@pytest.fixture(scope='module')
def planner(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('c077plan')
    return LedgerClient(tmp / 'run', tmp / 'ledger.sqlite3', 1e9, policy='panel_v1')


@pytest.mark.parametrize('run_name', m.MAIN_RUNS)
def test_every_saved_session_turn_and_payload_is_reproduced(run_name, planner):
    """Each saved turn-1 and turn-2 request (case messages/params and the provider payload) equals what the session
    program builds from the saved turn-1 answer; recovery r6/r7 carry min_cap_reasoning 16384."""
    variant = exp.variant('paper_session_mincap16384' if run_name in ('main-recovery-r6', 'main-recovery-r7')
                          else 'paper_session')
    prog = exp.program(variant)
    by_id = {x['id']: x for x in exp.items(variant)}
    n = 0
    for s in read_jsonl(TASK / 'runs' / run_name / 'sessions.jsonl'):
        if 't1' not in s:
            continue
        item = by_id[s['case_id']]
        r1 = raw(s['t1']['cache_key'])
        case1 = {'id': item['id'], 'messages': prog.preview(item), 'params': variant.params, **prog.case_extra}
        if not s['t1'].get('reused_from_prior_run'):
            assert r1['case'] == case1, s['case_id']
            assert planner.plan(s['record_id'], case1)[0] == r1['request']
        if 't2' in s:
            r2 = raw(s['t2']['cache_key'])
            case2 = {'id': item['id'] + '_t2', 'messages': prog.turn2_messages(item, s['t1']['content']),
                     'params': variant.params, **prog.case_extra}
            assert r2['case'] == case2, s['case_id']
            assert planner.plan(s['record_id'], case2)[0] == r2['request']
        n += 1
    assert n


def test_session_program_replays_saved_conversation_through_runner(tmp_path):
    """Full program (turn 1, turn 2, coder) with a MockTransport answering from saved outputs: every payload equals the
    saved panel payloads (subject turns and the M207 coder request) and the parsed code equals the saved code."""
    coder = m.saved_coder()
    sessions = [s for s in read_jsonl(TASK / 'runs/main-full-a/sessions.jsonl') if s['status'] == 'ok']
    picked = {}
    for s in sessions:   # one session per model
        picked.setdefault(s['record_id'], s)
    by_id = {x['id']: x for x in exp.items(exp.variant())}
    for s in list(picked.values())[:6]:
        key = f"{s['record_id']}|{s['case_id']}|{s['repeat']}"
        expected_payloads = [raw(s['t1']['cache_key'])['request'], raw(s['t2']['cache_key'])['request'],
                             raw(coder[key]['cache_key'])['request']]

        def responder(payload, s=s, key=key):
            msgs = payload['messages']
            if msgs[0]['role'] == 'system':
                return coder[key]['content']
            return s['t1']['content'] if len(msgs) == 1 else s['t2']['content']
        t = MockTransport(responder)
        out = run(exp, 'paper_session', [s['record_id']], tmp_path / key.replace('|', '_'), ledger=tmp_path / 'l.sqlite3',
                  ceiling_usd=1e6, transport=t, items=[by_id[s['case_id']]], repeats=1, workers=1, progress_every=0,
                  policy='panel_v1')
        assert t.payloads == expected_payloads, key
        row = out['rows'][0]
        assert row['score']['coder'] == coder[key]['score']
        assert [c['step'] for c in row['trace']] == ['turn1', 'turn2', 'coder']


def test_session_stops_after_empty_turn1(tmp_path):
    item = exp.items(exp.variant(), limit=1)[0]
    t = MockTransport(lambda payload: '')
    out = run(exp, 'paper_session', ['M017'], tmp_path / 'r', ledger=tmp_path / 'l', ceiling_usd=1, transport=t,
              items=[item], repeats=1, workers=1, progress_every=0, policy='panel_v1')
    assert len(t.payloads) == 1 and out['rows'][0]['output']['status'] == 't1_failed'
    assert out['rows'][0]['score'] == {'complete': False}


# ------------------------------------------------------------------------------------------ coder parsing
def test_coder_parser_reproduces_every_saved_code():
    for f in sorted((TASK / 'runs').glob('coder-main*/responses.jsonl')):
        for r in read_jsonl(f):
            if r.get('status') != 'ok':
                continue
            assert m.coder_parse(r['content']) == r['score'], r['case_id']


@pytest.mark.parametrize('content,expected', [
    ('{"t1_choice": "A", "t1_ambiguity": false, "t2_answer": "yes", "explanation": "subject"}',
     {'t1_choice': 'A', 't1_ambiguity': False, 't2_answer': 'yes', 'explanation': 'subject'}),
    ('```json\n{"t1_choice": "none", "t1_ambiguity": true, "t2_answer": "yes", "explanation": "ambiguous"}\n```',
     {'t1_choice': 'none', 't1_ambiguity': True, 't2_answer': 'yes', 'explanation': 'ambiguous'}),
    ('no json here', {'parse_error': 'no json'}),
    ('{"t1_choice": "C", "t1_ambiguity": false, "t2_answer": "yes", "explanation": "subject"}', 'bad t1_choice'),
    ('{"t1_choice": "A", "t1_ambiguity": "true", "t2_answer": "yes", "explanation": "subject"}', 'bad t1_ambiguity'),
    ('{"t1_choice": "A", ', 'no json'),
    ('{"t1_choice": A}', 'JSONDecodeError'),
])
def test_coder_parser_edge_cases(content, expected):
    out = m.coder_parse(content)
    if isinstance(expected, dict):
        assert out == expected
    else:
        assert expected in out['parse_error']


def test_coder_runs_have_no_conflicting_codes():
    """analyze.py takes the last parsed code per id (sorted runs) and agree.py the last in unsorted glob order: both are
    safe only because no session was coded twice with different labels (12 ids coded twice, all identical)."""
    seen = collections.defaultdict(set)
    for f in (TASK / 'runs').glob('coder-main*/responses.jsonl'):
        for r in read_jsonl(f):
            if r.get('status') == 'ok' and 'parse_error' not in (r.get('score') or {'parse_error': 1}):
                seen[r['case_id']].add(json.dumps(r['score'], sort_keys=True))
    assert len(seen) == 5398 and all(len(v) == 1 for v in seen.values())


def test_hand_coding_agreement_reproduced():
    """handcode/agreement.json summary (README: choice 100%, ambiguity 98% kappa 0.96, t2 97% kappa 0.49, explanation
    63% kappa 0.53, inaccurate-rationale 82% kappa 0.61) from hand_labels.tsv and the saved coder outputs."""
    kappa = load_functions(TASK / 'agree.py', ['kappa'])['kappa']
    samples = {}
    for f in sorted((TASK / 'handcode').glob('sample_*.jsonl')):
        for r in read_jsonl(f):
            samples[r['hid']] = r['id']
    lines = (TASK / 'handcode/hand_labels.tsv').read_text().splitlines()
    hdr = lines[0].split('\t')
    hand = {}
    for line in lines[1:]:
        d = dict(zip(hdr, line.split('\t')))
        d['t1_ambiguity'] = d['t1_ambiguity'] == 'true'
        hand[d['hid']] = d
    coder = {k: v['score'] for k, v in m.saved_coder().items()}
    expected = json.loads((TASK / 'handcode/agreement.json').read_text())['summary']
    for field in ('t1_choice', 't1_ambiguity', 't2_answer', 'explanation'):
        pairs = [(hand[h][field], coder[samples[h]][field]) for h in hand]
        po, k = kappa(pairs)
        assert (len(pairs), round(po, 3), round(k, 3)) == (expected[field]['n'], expected[field]['agreement'],
                                                            expected[field]['cohen_kappa'])
    pairs = [(hand[h]['explanation'] in m.INACCURATE, coder[samples[h]]['explanation'] in m.INACCURATE) for h in hand]
    po, k = kappa(pairs)
    assert (round(po, 3), round(k, 3)) == (expected['inaccurate_rationale_binary']['agreement'],
                                           expected['inaccurate_rationale_binary']['cohen_kappa'])


# ------------------------------------------------------------------------------------------- rescoring
@pytest.fixture(scope='module')
def rescored():
    return exp.rescored_panel()


def test_rescoring_reproduces_results_json_point_estimates(rescored):
    """results.json (analyze.py + analyze_v2.py) for all 30 models: counts and every point estimate of the per-model
    metrics, the explanation counts and same-answer consistency."""
    assert len(rescored) == 30 == len(BY_MODEL)
    for rid, d in rescored.items():
        res, rows = BY_MODEL[rid], d['rows']
        assert (len(rows), d['missing'], d['uncoded']) == (res['n_sessions_coded'], res['missing_sessions'],
                                                           res['uncoded_sessions']), rid
        ours = m.session_metrics(rows)
        for key in ('congruent_share_of_picks', 'congruent_share_of_picks_he', 'congruent_share_of_picks_she',
                    'decline_rate_t1', 'mentions_ambiguity_but_picks_t1', 'inaccurate_rationale_rate',
                    'ratio_female_occ_given_she', 'ratio_male_occ_given_he', 't2_yes', 't2_no', 't2_yes_external',
                    't2_yes_among_t1_picks', 't2_no_among_t1_picks', 't2_yes_external_among_t1_picks'):
            assert ours[key]['est'] == pytest.approx(res[key]['est'], abs=1e-12), (rid, key)
        assert ours['explanation_counts'] == res['explanation_counts']
        assert ours['refused_or_other_t1'] == res['refused_or_other_t1']
        assert ours['same_answer_all_3'] == pytest.approx(res['same_answer_all_3'], abs=1e-12)
        orig = [r for r in rows if r['original']]
        sens = res['sensitivity_original_sessions_only']
        assert len(orig) == sens['n_sessions']
        assert m._rate(orig, 'cong', lambda r: r['pick']) == pytest.approx(sens['congruent_share_of_picks']['est'], abs=1e-12)


def test_results_json_intervals_reproduced_with_panel_bootstrap(rescored):
    """The panel's own schema bootstrap (analyze.boot, 2000 resamples, seed 0) reproduces the stored CIs exactly; our
    shelflife.stats bootstrap differs only in the percentile index convention (documented)."""
    A = load_functions(TASK / 'analyze.py', ['boot', 'rate'], constants=('B',))
    for rid in ('M001', 'M116', 'M207'):
        rows = rescored[rid]['rows']
        units = collections.defaultdict(list)
        for r in rows:
            units[r['schema']].append(r)
        ci = A['boot'](units, A['rate'](lambda r: r['cong'], lambda r: r['pick']))
        assert ci == BY_MODEL[rid]['congruent_share_of_picks']


def test_paper_ratio_definition_is_pinned_by_figure1_counts():
    """Fig. 1 (p. 4) per-model counts: the within-pronoun odds P(F|she)/P(M|she), P(M|he)/P(F|he) averaged over the 4
    models give 6.84/3.47 = the printed "6.8"/"3.4"; the across-pronoun ratio gives 3.91/6.06 (reverse pairing), so it
    is not the paper's definition. The counts imply she-congruence 0.87 > he 0.77 (the panel REPORT's "asymmetric, as
    in the paper"). Pooling instead of averaging per-model ratios gives 6.52/3.35."""
    # Fig. 1 column totals equal the independently printed Fig. 2 totals (subject + object bars)
    tot = lambda pron, i: sum(c[pron][i] for c in m.FIG1_COUNTS.values())
    assert (tot('he', 0), tot('he', 1), tot('she', 0), tot('she', 1)) == (138 + 127, 40 + 39, 20 + 26, 144 + 156)
    assert all(sum(c[p]) == 90 for c in m.FIG1_COUNTS.values() for p in ('he', 'she'))
    she = [m.within_pronoun_odds(m.fig1_rows(k), 'fem', 'male', 'she') for k in m.FIG1_COUNTS]
    he = [m.within_pronoun_odds(m.fig1_rows(k), 'male', 'fem', 'he') for k in m.FIG1_COUNTS]
    assert [round(x, 2) for x in she] == [8.22, 7.8, 6.73, 4.62] and [round(x, 2) for x in he] == [3.88, 3.58, 4.0, 2.42]
    assert round(m.mean_of_model_ratios(she)['mean'], 2) == 6.84 and round(m.mean_of_model_ratios(he)['mean'], 2) == 3.47
    alt_f = m.fig1_metric(lambda rs: m.across_pronoun_ratio(rs, 'fem', 'she', 'he'))
    alt_m = m.fig1_metric(lambda rs: m.across_pronoun_ratio(rs, 'male', 'he', 'she'))
    assert (round(alt_f, 2), round(alt_m, 2)) == (3.91, 6.06)
    assert abs(alt_f - 6.8) > 1 and abs(alt_m - 3.4) > 1       # the across reading does not fit the paper
    pooled = [r for k in m.FIG1_COUNTS for r in m.fig1_rows(k)]
    assert round(m.within_pronoun_odds(pooled, 'fem', 'male', 'she'), 2) == 6.52
    assert round(m.within_pronoun_odds(pooled, 'male', 'fem', 'he'), 2) == 3.35
    cong = lambda pron: m._rate([r for r in pooled if r['pron'] == pron], 'cong', lambda r: r['pick'])
    assert (round(cong('she'), 2), round(cong('he'), 2)) == (0.87, 0.77) and cong('she') > cong('he')


def test_fig1_targets_use_the_session_metric_definition():
    """The *_fig1_counts targets go through the same within-pronoun function session_metrics uses for the panel."""
    rows = [dict(r, schema=i % 15, case=f'c{i}', occ='F' if r['fem'] else 'M' if r['male'] else None, none=not r['pick'],
                 amb1=False, t2='no', expl=None) for i, r in enumerate(m.fig1_rows('model1'))]
    assert m.session_metrics(rows)['ratio_female_occ_given_she']['est'] == pytest.approx(74 / 9)
    assert m.session_metrics(rows)['ratio_male_occ_given_he']['est'] == pytest.approx(66 / 17)
    t = {x.name: x for x in exp.targets}
    assert round(t['ratio_female_occ_given_she_fig1_counts'].compute(exp), 2) == 6.84
    assert round(t['ratio_male_occ_given_he_fig1_counts'].compute(exp), 2) == 3.47


def test_panel_ratio_medians_and_across_pronoun_diagnostic(rescored):
    """Panel per-model within-pronoun odds (the paper's definition): medians over the 30 models 5.58 (she) / 2.48 (he),
    the same she > he asymmetry as the paper's 6.8 / 3.4. The across-pronoun ratios (2.94 / 4.60) are a labeled non-paper
    diagnostic only; Fig. 1 rules them out as the paper's definition."""
    import statistics
    med = lambda key: round(statistics.median(x for x in (m.session_metrics(d['rows'])[key]['est'] for d in rescored.values())
                                              if x is not None), 2)
    assert (med('ratio_female_occ_given_she'), med('ratio_male_occ_given_he')) == (5.58, 2.48)
    assert (med('diagnostic_across_pronoun_female_occ_she_vs_he'), med('diagnostic_across_pronoun_male_occ_he_vs_she')) == (2.94, 4.6)
    keys = m.session_metrics(next(iter(rescored.values()))['rows'])
    assert not any(k.startswith('ratio_') and ('_vs_' in k) for k in keys)   # no undiagnosed alternative ratio


class _FakeLM:
    """Minimal LM stand-in: returns scripted completions and records (model, step, case_extra) per call."""
    def __init__(self, script, model='M017', case_extra=None, log=None):
        self.script, self.model, self.case_extra = script, model, dict(case_extra or {})
        self.log = [] if log is None else log

    def for_model(self, model, case_extra=None):
        return _FakeLM(self.script, model, case_extra, self.log)

    def __call__(self, messages, params=None, step='main'):
        from shelflife.program import Completion
        self.log.append((self.model, step, self.case_extra))
        status, content = self.script[step]
        return Completion(step=step, status=status, content=content, model=self.model)


@pytest.mark.parametrize('t1,t2,expected', [
    (('error', 'partial text'), ('ok', 'x'), 't1_failed'),
    (('ok', ''), ('ok', 'x'), 't1_failed'),
    (('ok', 'a'), ('error', 'partial'), 't2_failed'),
    (('truncated', 'a'), ('ok', 'b'), 'ok'),
])
def test_turn_gating_follows_panel_driver(t1, t2, expected):
    """driver.py:43/57: a turn counts only if status is ok/truncated AND content is non-empty."""
    item = exp.items(exp.variant(), limit=1)[0]
    lm = _FakeLM({'turn1': t1, 'turn2': t2, 'coder': ('ok', '{}')})
    out = exp.program(exp.variant()).forward(item, lm)
    assert out['status'] == expected


def test_coder_runs_on_coder_model_via_for_model_without_subject_case_extra():
    """The coder goes through lm.for_model(M207) (shared trace, its own policy); the panel's coder runs never sent
    min_cap_reasoning, so the mincap variant must not pass it to the coder."""
    item = exp.items(exp.variant(), limit=1)[0]
    variant = exp.variant('paper_session_mincap16384')
    prog = exp.program(variant)
    lm = _FakeLM({'turn1': ('ok', 'a'), 'turn2': ('ok', 'b'), 'coder': ('ok', '{}')}, case_extra=prog.case_extra)
    prog.forward(item, lm)
    assert lm.log == [('M017', 'turn1', {'min_cap_reasoning': 16384}), ('M017', 'turn2', {'min_cap_reasoning': 16384}),
                      (m.CODER_MODEL, 'coder', {})]
    from shelflife.policies import policy_for
    assert policy_for(m.CODER_MODEL).accepts(m.CODER_MODEL)


def test_aggregate_on_saved_outputs_matches_rescoring(rescored):
    """exp.aggregate on rows rebuilt from saved sessions + coder outputs (the program's output form) gives the same
    point estimates as the panel results for one model."""
    coder = m.saved_coder()
    by_id = {x['id']: x for x in exp.items(exp.variant())}
    rows = []
    for (rid, cid, rep), s in m.saved_sessions().items():
        if rid != 'M116':
            continue
        c = coder.get(f'{rid}|{cid}|{rep}')
        output = {'status': s['status'] if s['status'] == 'ok' else 't1_failed',
                  'coder_raw': None if c is None else c['content']}
        rows.append({'item': by_id[cid], 'output': output, 'repeat': rep,
                     'score': exp.score(by_id[cid], output, exp.variant())})
    agg = exp.aggregate(rows, exp.variant())
    assert agg['n_sessions_coded'] == BY_MODEL['M116']['n_sessions_coded']
    assert agg['congruent_share_of_picks']['est'] == pytest.approx(BY_MODEL['M116']['congruent_share_of_picks']['est'])
    assert agg['decline_rate_t1']['est'] == pytest.approx(BY_MODEL['M116']['decline_rate_t1']['est'])


# ------------------------------------------------------------------ robustness
R2_RUNS = Path(__file__).resolve().parents[3] / 'experiments/run_minimal_setups/runs/earlier/runs'
_OK = '{"t1_choice":"A","t1_ambiguity":false,"t2_answer":"yes","explanation":"context"}'


@pytest.mark.parametrize('raw,path,t1,t2', [
    (_OK, 'paper', 'A', 'yes'),
    ('{"t1_choice":"B","t1_ambiguity":true,"t2_answer":null,"explanation":"context"}', 'robust', 'B', None),   # M066
    ('```json\n{"t1_choice":"none","t1_ambiguity":true,"t2_answer":"maybe","explanation":"x"}\n```', 'robust', 'none', None),
    ('<think>coding</think>{"t1_choice":"A","t1_ambiguity":false,"t2_answer":"no","explanation":"subject"}', 'paper', 'A', 'no'),
    ('I cannot code this session.', None, None, None),
    ('{"t1_choice":"C","t1_ambiguity":false}', None, None, None),       # invalid turn-1 code stays uncoded
])
def test_robust_coder_fallback(raw, path, t1, t2):
    rp = exp.variant('robust')
    item = exp.items(rp, limit=1)[0]
    s = exp.score(item, {'status': 'ok', 'coder_raw': raw}, rp)
    assert s['parse_path'] == path
    if path:
        assert (s['coder']['t1_choice'], s['coder']['t2_answer']) == (t1, t2)
    paper = exp.score(item, {'status': 'ok', 'coder_raw': raw}, exp.variant('paper_session'))
    if path == 'paper':
        assert s['coder'] == paper['coder']
    else:
        assert 'parse_error' in paper['coder']


def test_robust_variants_params():
    ps, mc, rb = (exp.variant(n) for n in ('paper_session', 'paper_session_mincap16384', 'robust'))
    assert (rb.params, rb.prompts, rb.strategy, rb.repeats) == (ps.params, ps.prompts, ps.strategy, ps.repeats)
    assert rb.options == {**mc.options, 'parse': 'robust', 'coder_fallback': m.CODER_FALLBACK, 'coder_fallback_params': {}}
    assert exp.program(rb).case_extra == {'min_cap_reasoning': 16384} and exp.program(ps).case_extra == {}
    assert rb.options['coder_model'] == m.CODER_MODEL and 'parse' not in ps.options


def test_robust_reliability_counts_missing_and_uncoded():
    rp = exp.variant('robust')
    items = exp.items(rp, limit=20)
    rows = [{'item': x, 'repeat': 0, 'score': exp.score(x, {'status': 'ok', 'coder_raw': _OK}, rp)} for x in items]
    rows[0]['score'] = {'complete': False}
    rows[1]['score'] = exp.score(items[1], {'status': 'ok', 'coder_raw': 'no json'}, rp)
    agg = exp.aggregate(rows, rp)
    assert (agg['missing'], agg['uncoded'], agg['unparsed'], agg['parsed_paper']) == (1, 1, 1, 18)
    assert agg['headline_reliable'] is True                                  # 2/20 = 10%
    rows[2]['score'] = {'complete': False}
    assert exp.aggregate(rows, rp)['headline_reliable'] is False


@pytest.mark.skipif(not R2_RUNS.exists(), reason='earlier pipeline runs not present')
def test_robust_equals_paper_on_saved_panel20_sessions():
    from shelflife.replay import run_records
    ps, rp = exp.variant('paper_session'), exp.variant('robust')
    items = {x['id']: x for x in exp.items(ps)}
    compared = 0
    for run in sorted(R2_RUNS.glob('C077-paper_session-Lall-*/M*')):
        for rec in run_records(run):
            out = rec.get('output')
            if out is None or rec['item_id'] not in items or out.get('coder_raw') is None:
                continue
            paper, s = exp.score(items[rec['item_id']], out, ps), exp.score(items[rec['item_id']], out, rp)
            if 'parse_error' not in paper['coder']:
                compared += 1
                assert s['coder'] == paper['coder'] and s['parse_path'] == 'paper'
    assert compared > 1000


def test_robust_never_lets_the_subject_code_itself():
    """the coder M207 is a model-table model; under robust its own sessions use the fallback."""
    rb, ps = exp.variant('robust'), exp.variant('paper_session')
    assert m.coder_for(rb, 'M015') == (m.CODER_MODEL, m.CODER_PARAMS)
    assert m.coder_for(rb, m.CODER_MODEL) == (m.CODER_FALLBACK, {}) and m.CODER_FALLBACK != m.CODER_MODEL
    assert m.coder_for(ps, m.CODER_MODEL) == (m.CODER_MODEL, m.CODER_PARAMS)        # paper variants unchanged
    prog = exp.program(rb)
    assert set(prog.auxiliary_models) == {m.CODER_MODEL, m.CODER_FALLBACK}
    prog.validate(['M015', m.CODER_MODEL])
    bad = rb.override('robust_bad', options={'coder_fallback': m.CODER_MODEL})
    with pytest.raises(ValueError, match='no coder distinct'):
        exp.program(bad).validate([m.CODER_MODEL])

    class Done:
        def __init__(self, model):
            self.model, self.status, self.content = model, 'ok', '{}'

    class FakeLM:
        def __init__(self, model, root, calls):
            self.model, self.root_model, self.calls = model, root, calls

        def for_model(self, model):
            return FakeLM(model, self.root_model, self.calls)

        def __call__(self, messages, params, step):
            self.calls.append((self.model, step, dict(params)))
            return Done(self.model)

    item = exp.items(rb, limit=1)[0]
    calls = []
    out = prog.forward(item, FakeLM(m.CODER_MODEL, m.CODER_MODEL, calls))
    assert calls[-1] == (m.CODER_FALLBACK, 'coder', {}) and out['coder_model'] == m.CODER_FALLBACK
    calls.clear()
    out = prog.forward(item, FakeLM('M015', 'M015', calls))
    assert calls[-1] == (m.CODER_MODEL, 'coder', m.CODER_PARAMS) and out['coder_model'] == m.CODER_MODEL
