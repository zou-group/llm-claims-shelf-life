"""C010 Walters & Wilder: literal prompts from the supplements, the continuation program replayed against every
earlier run, and the automated scorer against its saved verdicts."""
import json
import os
import xml.etree.ElementTree as ET

import pytest

from shelflife import registry
from shelflife.experiments import c010_fabricated_citations as m
from shelflife.ledger import MockTransport
from shelflife.paths import INITIAL, HELPERS, PROJECT
from shelflife.program import CallFailed, Completion
from shelflife.replay import read_jsonl
from shelflife.runner import run

exp = registry.get('C010')
TASK = HELPERS / 'C010'
REPRO = INITIAL / 'reproductions/C010'
FULL = os.environ.get('SHELFLIFE_FULL') == '1'


# ------------------------------------------------------------------------------------------------ prompts
def test_template_renders_all_84_archived_prompts_byte_for_byte():
    """Appendix 2: both ChatGPT models received the identical literal prompt for every topic."""
    archive = m.archive_transcripts()
    items = {x['topic_number']: x for x in exp.items(exp.variant())}
    assert len(items) == 42
    for (ui, topic), a in archive.items():
        assert exp.program(exp.variant()).preview(items[topic]) == [{'role': 'user', 'content': a['prompt']}], (ui, topic)


def test_questions_differ_from_appendix1_only_by_spacing_quotes_and_one_typo():
    diffs = {}
    for x in exp.items(exp.variant()):
        a1 = ' '.join(x['appendix1_topic'].split())
        if a1 != x['question']:
            diffs[x['topic_number']] = (a1, x['question'])
    assert sorted(diffs) == [17, 22, 25, 32]
    assert 'sychronous' in diffs[17][1] and 'synchronous' in diffs[17][0]       # typo preserved as sent
    for t in (22, 25, 32):
        assert diffs[t][0].replace('"', "'") == diffs[t][1]


def test_variants_and_options():
    v = exp.variants
    assert v['paper_chatgpt_ui'].faithful and v['paper_chatgpt_ui'].params == {'max_tokens': 8192}
    assert m.CONTINUE.render() == 'Please continue'
    # robust (2026-10-10) scores like its parent paper_chatgpt_ui_verify_online (online lookups)
    online = {'paper_chatgpt_ui_verify_online', 'cap4096_verify_online', 'robust'}
    assert all(v[k].options['verify_online'] for k in online)
    assert all(not x.options['verify_online'] for k, x in v.items() if k not in online)
    assert v['cap4096_verify_online'].params == {'max_tokens': 4096} and not v['cap4096_verify_online'].faithful
    assert v['cap4096_verify_online'].options['verify_online'] and v['cap4096_verify_online'].options['cap_fallback'] is None


# ----------------------------------------------------------------------------- paper tables (author coding)
def _real(ui):
    return [c for c in m.coded_citations() if c['ui_model'] == ui and c['reality'] == 'real']


# ------------------------------------------------------------------------- program replay against prior runs
class ReplayLM:
    """Stands in for shelflife.program.LM: returns the saved stage of each turn and records our requests."""

    def __init__(self, stages):
        self.stages, self.trace, self.requests = stages, [], []

    def __call__(self, messages, params=None, step='main'):
        stage = self.stages[len(self.trace)]
        self.requests.append({'step': step, 'messages': messages, 'params': dict(params or {})})
        c = Completion(step=step, status=stage['status'], content=stage.get('content') or '',
                       finish_reason=stage.get('finish_reason'), error=stage.get('error_body') or stage.get('error'))
        self.trace.append(c)
        if not c.usable:
            raise CallFailed(c)
        return c


def _replay(prior, record, stages):
    items = {x['id']: x for x in exp.items(exp.variant())}
    prog = exp.program(exp.variant(prior.variant))
    lm = ReplayLM(stages)
    try:
        out = prog.forward(items[record['id']], lm)
    except CallFailed:
        out = None
    return lm, out


def _check(prior, record, stages):
    lm, out = _replay(prior, record, stages)
    problems = []
    if len(lm.requests) != len(stages):
        problems.append(f'{len(lm.requests)} requests vs {len(stages)} saved stages')
    for ours, saved in zip(lm.requests, stages):
        case = saved['case']
        if (f"{record['id']}-{ours['step']}", ours['messages'], ours['params']) != (case['id'], case['messages'], case['params']):
            problems.append(f"request {case['id']} differs")
    if out is None:
        if record['stop_reason'] != 'unavailable_or_failed_turn' or stages[-1]['status'] not in {'error', 'not_submitted'}:
            problems.append('program raised but the prior conversation did not end on a failed call')
    else:
        for key in ('assistant_text', 'stop_reason', 'generation_available', 'any_truncated_turn'):
            if out[key] != record[key]:
                problems.append(f'{key} differs')
        if 'cap_reduced' in record and out['cap_reduced'] != record['cap_reduced']:
            problems.append('cap_reduced differs')
    return problems


def _payload_key(messages):
    return json.dumps(messages, ensure_ascii=False, sort_keys=True)


# -------------------------------------------------------------------------------- scorer fidelity
def _reverify(path, version, every=1):
    from shelflife.experiments._c010_verify import pipeline
    p = pipeline(version, offline=True)
    rows = read_jsonl(path)[::every]
    return rows, [(r['verdict']['decision'], p.verify(r['citation'])['decision']) for r in rows]


def test_offline_cache_miss_is_unverifiable_not_fabricated():
    v = exp.verify_citations(['Zzyzxqv, Q. (2099). A title that no catalogue lookup has ever been made for. Nowhere Press.'])[0]
    assert v['decision'] == 'unverifiable' and v['lookup_failed']
    assert v['offline_cache_misses'] > 0 and v['lookup_not_attempted']


INVENTED_BIB = ('A short review (Zzyzxqv, 2099; Blorvath & Quimbly, 2087).\n\nReferences\n\n'
                'Zzyzxqv, Q. (2099). The hidden thermodynamics of invented quokka cognition. Journal of Imaginary '
                'Results, 12(3), 45-67.\n\n'
                'Blorvath, K., & Quimbly, T. (2087). Spectral harmonics of nonexistent glaciers. Phantom University Press.\n')


def test_new_run_scored_offline_reports_undetermined_share_not_zero():
    """Audit major 1: a bibliography of invented works scored with the default (offline) variant used to give
    fabricated_share 0.0. Lookups that were never attempted must leave the share undetermined (None), with counts."""
    s = exp.score({'id': 'C010-topic-01'}, {'generation_available': True, 'assistant_text': INVENTED_BIB}, exp.variant())
    assert len(s['citations']) == 2 and not s['verify_online']
    assert all(c['lookup_not_attempted'] and c['reality'] == 'unverifiable' for c in s['citations'])
    a = exp.aggregate([{'item': {'id': 'C010-topic-01'}, 'score': s}], n_boot=20)
    assert a['fabricated_share'] is None and a['fabricated_share_ci95_topic_bootstrap'] == [None, None]
    assert a['fabricated_share_status'] == 'undetermined_offline_cache_miss'
    assert (a['citations'], a['fabricated'], a['unverifiable'], a['lookup_not_attempted_offline']) == (2, 0, 2, 2)
    assert a['fabricated_share_bounds_unverifiable'] == [0.0, 1.0]
    assert all(t['fabricated_share'] is None for t in a['by_type'].values())


def test_one_uncached_citation_makes_the_whole_share_undetermined():
    rows = [{'item': {'id': 't1'}, 'score': {'generation_available': True, 'citations': [
        {'citation': 'a', 'reality': 'real'}, {'citation': 'b', 'reality': 'fabricated'},
        {'citation': 'c', 'reality': 'unverifiable', 'lookup_failed': True, 'lookup_not_attempted': True}]}}]
    assert exp.aggregate(rows, n_boot=20)['fabricated_share'] is None
    rows[0]['score']['citations'][2]['lookup_not_attempted'] = False   # an attempted lookup that failed (online)
    a = exp.aggregate(rows, n_boot=20)
    assert a['fabricated_share'] == pytest.approx(1 / 3) and a['fabricated_share_status'] == 'determined'


def test_verify_online_variant_is_explicit_and_uses_the_network_layer(monkeypatch, tmp_path):
    """Opt-in online scoring: same generation protocol, network-backed lookups for uncached URLs. The network is
    mocked (no real calls): every catalogue answers 200 with no hits, so both invented works become fabricated."""
    import io
    import urllib.request
    from shelflife.experiments import _c010_verify as cv
    online = exp.variant('paper_chatgpt_ui_verify_online')
    base = exp.variant()
    assert online.options['verify_online'] is True and base.options['verify_online'] is False
    assert (online.prompts, online.params, online.strategy, online.faithful) == (base.prompts, base.params, base.strategy, True)
    calls = []

    class Resp(io.BytesIO):
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        calls.append(req.full_url)
        return Resp(json.dumps({'message': {'items': [], 'total-results': 0}, 'docs': [], 'numFound': 0,
                                'data': [], 'total': 0}).encode())

    monkeypatch.setattr(urllib.request, 'urlopen', fake_urlopen)
    monkeypatch.setattr(cv, 'LOCAL_CACHE', tmp_path)
    p = cv.pipeline('v2', offline=False)
    monkeypatch.setattr(p.module, 'CACHE', tmp_path)
    monkeypatch.setattr(p.module, 'time', cv._NoSleep(p.module.time))
    s = exp.score({'id': 'C010-topic-01'}, {'generation_available': True, 'assistant_text': INVENTED_BIB}, online)
    assert calls and all(u.startswith('https://') for u in calls)
    assert s['verify_online'] and [c['reality'] for c in s['citations']] == ['fabricated', 'fabricated']
    assert all(c['offline_cache_misses'] == 0 and not c['lookup_not_attempted'] for c in s['citations'])
    a = exp.aggregate([{'item': {'id': 'C010-topic-01'}, 'score': s}], n_boot=20)
    assert a['fabricated_share'] == 1.0 and a['fabricated_share_status'] == 'determined'


def test_unavailable_conversations_holding_a_bibliography_are_counted():
    """Audit minor 3: a conversation that failed only on an unneeded continuation stays missing but is counted."""
    v = exp.variant()
    rows = [{'item': {'id': cid}, 'score': exp.score({'id': cid}, {'generation_available': False, 'assistant_text': t}, v)}
            for cid, t in (('C010-topic-04', INVENTED_BIB), ('C010-topic-05', 'An essay cut off before its'))]
    flagged = sorted(r['item']['id'] for r in rows if r['score']['unavailable_with_extractable_bibliography'])
    assert flagged == ['C010-topic-04']
    assert exp.aggregate(rows, n_boot=1)['missing_with_extractable_bibliography'] == 1


@pytest.mark.parametrize('output,expected', [
    ({'assistant_text': '', 'generation_available': False}, {'has_bibliography': None, 'citations': []}),
    ({'assistant_text': 'An essay with no references at all.', 'generation_available': True}, {'has_bibliography': False, 'citations': []}),
])
def test_score_edge_cases(output, expected):
    s = exp.score({'id': 'x'}, output, exp.variant())
    assert {k: s[k] for k in expected} == expected


def test_unavailable_generation_is_missing_not_zero_fabrication():
    rows = [{'item': {'id': 't1'}, 'score': {'generation_available': False, 'citations': []}},
            {'item': {'id': 't2'}, 'score': {'generation_available': True, 'has_bibliography': True,
                                             'citations': [{'citation': 'a', 'reality': 'fabricated'}, {'citation': 'b', 'reality': 'real'}]}}]
    a = exp.aggregate(rows, n_boot=50)
    assert (a['missing_generation'], a['citations'], a['fabricated_share']) == (1, 2, 0.5)


# ---------------------------------------------------------------------- extraction


# ------------------------------------------------------------------- robustness for large panels (2026-10-10)
R2_RUNS = PROJECT / 'experiments/run_minimal_setups/runs/earlier/runs'
ENTRY_A = 'Courneya, K. S., & Carron, A. V. (1992). The home advantage in sport competitions: A literature review. Journal of Sport and Exercise Psychology, 14(1), 13-27.'
ENTRY_B = 'Pollard, R. (2002). Evidence of a reduced home advantage when a team moves to a new stadium. Journal of Sports Sciences, 20(12), 969-973.'
BODY = 'Home advantage is well documented (Courneya & Carron, 1992; Pollard, 2002). Fans matter.\n\n'


def test_robust_variants_params():
    v = exp.variants
    gen, rb = v['paper_chatgpt_ui_verify_online'], v['robust']
    assert rb.params == gen.params == {'max_tokens': 8192} and not rb.faithful and rb.strategy == gen.strategy
    assert rb.options == {**gen.options, 'parse': 'robust', 'cap_fallback': 4096, 'stop_rule': 'robust',
                          'min_cap_reasoning': 32768}
    assert exp.program(rb).case_extra == {'min_cap_reasoning': 32768}
    assert all(not getattr(exp.program(x), 'case_extra', None) for k, x in v.items() if k != 'robust')
    # faithful / earlier variants keep the paper parser and the prior stop rule
    assert all('parse' not in x.options and 'stop_rule' not in x.options
               for k, x in v.items() if k != 'robust')
    assert v['paper_chatgpt_ui'].faithful and gen.faithful


@pytest.mark.parametrize('text,path,n', [
    ('Essay. ' + BODY + 'References\n\n' + ENTRY_A + '\n\n' + ENTRY_B + '\n', 'paper', 2),
    ('Essay. ' + BODY + '**References:**\n\n1. ' + ENTRY_A + '\n2. ' + ENTRY_B + '\n', 'paper', 2),
    # duplicated in-text citation after each entry (M066 topic 29): v1 INTEXT_END rejects every line
    ('Essay. ' + BODY + 'References:\n\n' + ENTRY_A + ' (Courneya & Carron, 1992)\n\n' + ENTRY_B + ' (Pollard, 2002)\n',
     'robust', 2),
    # heading the paper extractor does not know
    ('Essay. ' + BODY + '## Literature Cited\n\n' + ENTRY_A + '\n' + ENTRY_B + '\n', 'robust', 2),
    # <think> block in the content, then a bibliography under a heading the paper extractor does not know
    ('<think>\nI should cite real works on home advantage (1992).\n</think>\n' + BODY + '**Sources**\n\n- ' + ENTRY_A + '\n- ' + ENTRY_B,
     'robust', 2),
    # bibliography cut off mid-list (truncated turn): the complete entry before the cut is still extracted
    ('Essay. ' + BODY + 'References\n\n' + ENTRY_A + '\n\nPollard, R. (2002). Evidence of a reduced home adv', 'paper', 2),
    ('I cannot write this paper with verified references.', None, 0),
    ('flavor flavored taste tasting flavorful seasoning seasoned salts pepper peppercorns spices paprika', None, 0),
    ('', None, 0),
])
def test_robust_extraction_modern_formats(text, path, n):
    e, p = m.extract_bibliography(text, 'v2', 'robust')
    assert (p, len(e['entries'])) == (path, n)
    if path:
        assert ENTRY_A in e['entries'][0]
    # the paper parser alone: same result wherever it finds entries, nothing otherwise
    pe, pp = m.extract_bibliography(text, 'v2', 'paper')
    assert pe['entries'] == (e['entries'] if path == 'paper' else [])
    assert pp == ('paper' if path == 'paper' else None)


def _panel_conversations():
    """The earlier model table's conversations as build_citations(_v2).py selected them: reused M206/M207 transcripts,
    then runs/*/ in sorted order, a later run replacing only an earlier *failed* attempt."""
    convs = {}
    prior_map = {'anthropic/claude-sonnet-5.5': 'M206', 'openai-direct/gpt-6.1-sol': 'M207'}
    for k, text in json.loads((TASK / 'validation/prior_transcripts.json').read_text()).items():
        model, cid = k.split('|')
        convs[(prior_map[model], cid)] = {'assistant_text': text, 'generation_available': True,
                                          'stop_reason': 'normal_stop_with_bibliography_candidate'}
    for f in sorted(TASK.glob('runs/*/conversations.jsonl')):
        for r in read_jsonl(f):
            prev = convs.get((r['record_id'], r['id']))
            if prev and prev.get('generation_available') and prev.get('stop_reason') != 'unavailable_or_failed_turn':
                continue
            convs[(r['record_id'], r['id'])] = r
    return convs


def _robust_rows(paths):
    rows = []
    for i, p in enumerate(paths):
        if p == 'missing':
            rows.append({'item': {'id': f't{i}'}, 'score': {'generation_available': False, 'citations': [], 'parse_path': None}})
        elif p == 'call_error':
            rows.append({'item': {'id': f't{i}'}, 'output': None, 'score': None})
        else:
            cits = [{'citation': 'a', 'reality': 'fabricated'}, {'citation': 'b', 'reality': 'real'}] if p else []
            rows.append({'item': {'id': f't{i}'}, 'score': {'generation_available': True, 'has_bibliography': bool(p),
                                                             'parse_path': p, 'citations': cits}})
    return rows


@pytest.mark.parametrize('paths,counts,reliable', [
    (['paper'] * 18 + ['robust', None], (18, 1, 1, 0), True),            # 1/20 unparsed
    (['paper'] * 18 + ['missing', 'call_error'], (18, 0, 0, 2), True),   # 2/20 missing = 10%
    (['paper'] * 17 + [None, 'missing', 'call_error'], (17, 0, 1, 2), False),
    (['paper'] * 7 + ['missing'] * 35, (7, 0, 0, 35), False),            # M196 the panel pattern
])
def test_robust_reliability_counts(paths, counts, reliable):
    rows = _robust_rows(paths)
    a = exp.aggregate(rows, exp.variant('robust'), n_boot=20)
    assert (a['parsed_paper'], a['parsed_robust'], a['unparsed'], a['missing_generation']) == counts
    assert a['headline_reliable'] is reliable
    assert a['validity_rate'] == pytest.approx((counts[0] + counts[1]) / len(paths))
    # the paper variant reports no robust fields and no reliability flag (unchanged output)
    p = exp.aggregate(rows, exp.variant('paper_chatgpt_ui_verify_online'), n_boot=20)
    assert not {'parsed_paper', 'parsed_robust', 'unparsed', 'headline_reliable'} & set(p)
    assert p['fabricated_share'] == a['fabricated_share']


class _ScriptLM:
    """Fake LM: a scripted (status, content, finish, error) per call; records steps and params."""

    def __init__(self, script):
        self.script, self.trace, self.requests = list(script), [], []

    def __call__(self, messages, params=None, step='main'):
        status, content, finish, error = self.script[len(self.trace)]
        self.requests.append({'step': step, 'params': dict(params or {})})
        c = Completion(step=step, status=status, content=content, finish_reason=finish, error=error)
        self.trace.append(c)
        if not c.usable:
            raise CallFailed(c)
        return c


def test_robust_stop_rule_recognises_bold_colon_heading():
    """BIB_HEADING misses '**References:**' and sends an unneeded 'Please continue'; stop_rule='robust' stops."""
    paper = 'Essay. ' + BODY + '**References:**\n\n' + ENTRY_A + '\n' + ENTRY_B
    item = exp.items(exp.variant(), limit=1)[0]
    script = [('ok', paper, 'stop', None), ('ok', 'Nothing more to add.', 'stop', None), ('ok', 'Done.', 'stop', None)]
    lm = _ScriptLM(script)
    out = exp.program(exp.variant('paper_chatgpt_ui_verify_online')).forward(item, lm)
    assert len(lm.requests) == 3 and out['stop_reason'] == 'resource_cap_without_confirmed_bibliography_completion'
    lm = _ScriptLM(script)
    out = exp.program(exp.variant('robust')).forward(item, lm)
    assert len(lm.requests) == 1 and out['stop_reason'] == 'normal_stop_with_bibliography_candidate'
    assert lm.requests[0]['params'] == {'max_tokens': 8192}


def test_robust_cap_rejection_restarts_at_4096():
    item = exp.items(exp.variant(), limit=1)[0]
    paper = 'Essay. ' + BODY + 'References\n\n' + ENTRY_A
    lm = _ScriptLM([('error', '', None, 'max_tokens: 8192 > 4096, which is the maximum allowed'),
                    ('ok', paper, 'stop', None)])
    out = exp.program(exp.variant('robust')).forward(item, lm)
    assert [r['step'] for r in lm.requests] == ['turn-0', 'cap4096-turn-0']
    assert [r['params']['max_tokens'] for r in lm.requests] == [8192, 4096]
    assert out['cap_reduced'] == 4096 and out['generation_available']


def test_robust_reasoning_cap_reaches_payload_only_for_reasoning_models():
    """min_cap_reasoning travels as case_extra: the panel policy sends 32768 to a reasoning model (M207) and keeps the
    paper 8192 for a non-reasoning one (M001). Offline plan (no scoring, so no catalogue lookups)."""
    from shelflife.planning import plan_program
    v = exp.variant('robust')
    paper = 'Essay. ' + BODY + 'References\n\n' + ENTRY_A + '\n' + ENTRY_B
    pl = plan_program(exp.program(v), exp.items(v, limit=1), ['M207', 'M001'], policy='panel_v2',
                      responder=lambda request: paper)
    caps = {c['model']: c['payload'].get('max_completion_tokens', c['payload'].get('max_tokens')) for c in pl['calls']}
    assert caps == {'M207': 32768, 'M001': 8192}
    gen = exp.variant('paper_chatgpt_ui_verify_online')
    pl = plan_program(exp.program(gen), exp.items(gen, limit=1), ['M207'], policy='panel_v2',
                      responder=lambda request: paper)
    assert pl['calls'][0]['payload']['max_completion_tokens'] == 8192   # generation variant unchanged


