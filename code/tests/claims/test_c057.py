"""C057 Embers shift ciphers: prompts, ground truth and scorers against the authors' repo, and rescoring of earlier runs."""
import contextlib
import io
import json
import re
import sys
import types
from collections import Counter

import pytest

from shelflife import registry
from shelflife.experiments import c057_embers as m
from shelflife.paths import INITIAL, PROJECT
from shelflife.replay import legacy_records
from shelflife.sourcecode import load_functions

exp = registry.get('C057')
RUNS = INITIAL / 'runs'
STATS = json.loads((m.PRIOR / 'expanded_stats.json').read_text())


# ------------------------------------------------------------------ prompts and data
@pytest.mark.parametrize('variant', ['paper_base'])
def test_prompts_byte_equal_to_stimuli_and_released_logs(variant):
    """Every rendered prompt equals instruction_plus_input and the prompt the authors sent (logs/, both models)."""
    v = exp.variant(variant)
    prog = exp.program(v)
    items = exp.items(v)
    assert len(items) == 100 * len(v.options['sets'])
    for item in items:
        row = m.stimuli(item['set'])[item['source_index']]
        msgs = prog.build(item)
        assert msgs == [{'role': 'user', 'content': row['instruction_plus_input']}]
        assert row['task_instruction'] % row['input'] == row['instruction_plus_input']
        for model in m.ARCHIVED_MODELS:
            assert m.archived_log(item['set'], model)['prompts'][item['source_index']] == msgs[0]['content']
        assert prog.params_for(item) == {'temperature': 0.0, 'top_p': 1, 'max_tokens': 200, 'stop': None}


def test_official_client_sends_one_user_message_with_temperature_0_top_p_1():
    """models/openai.py: gpts wraps each prompt as a lone user message; chatgpts adds top_p=1. run_openai.py forces
    temperature=0.0 and the README runs the shift/rot13 tasks with --max_tokens 200."""
    seen = {}
    ns = load_functions(m.REPO / 'models/openai.py', ['gpts', 'chatgpts'], extra_globals={
        'asyncio': types.SimpleNamespace(run=lambda x: x),
        'generate_from_openai_chat_completion': lambda **kw: seen.update(kw) or [
            {'choices': [{'message': {'content': 'x'}}]} for _ in kw['messages_list']]})
    assert ns['gpts'](['P1', 'P2'], model='gpt-4-0613', temperature=0.0, max_tokens=200) == ['x', 'x']
    assert seen == {'model': 'gpt-4-0613', 'messages_list': [[{'role': 'user', 'content': 'P1'}],
                                                             [{'role': 'user', 'content': 'P2'}]],
                    'temperature': 0.0, 'max_tokens': 200, 'top_p': 1, 'stop': None}
    src = (m.REPO / 'run_openai.py').read_text()
    assert "gpts(prompts, model=model, temperature=0.0, max_tokens=max_tokens)" in src
    readme = (m.REPO / 'README.md').read_text()
    assert ('run_openai.py --tasks rot13enc,rot13dec,rot12enc,rot12dec --conditions highprob,mediumprob,lowprob,'
            'adversarial --max_tokens 200  --model gpt-4-0613') in readme
    assert 'run_openai.py --tasks shift --conditions 1,2,3' in readme and '--max_tokens 200  --model gpt-4-0613' in readme


def test_ground_truth_is_the_caesar_decoding_and_sentence_families_are_paired():
    v = exp.variant('paper_base')
    high = {x['source_index']: x['expected'] for x in exp.items(v) if x['set'] == 'rot13dec_highprob'}
    for item in exp.items(v):
        assert m.caesar(item['expected'], item['shift']) == item['input']
        assert m.caesar(item['input'], 26 - item['shift']) == item['expected']
        if item['set'] == 'shift_2':
            assert item['expected'] == high[item['source_index']]  # shift sets use the same 100 high-prob sentences
    # rot13dec_highprob and shift_13 are byte-identical stimuli (the authors ran them as two separate calls)
    assert [r['instruction_plus_input'] for r in m.stimuli('shift_13')] == \
           [r['instruction_plus_input'] for r in m.stimuli('rot13dec_highprob')]


# ------------------------------------------------------------------ scorers
def _run_official_eval(script):
    """Execute the authors' evaluation script unmodified (Levenshtein stubbed: only accuracies are compared)."""
    stub = types.ModuleType('Levenshtein')
    stub.distance = lambda a, b: 0
    buf = io.StringIO()
    saved = sys.modules.get('Levenshtein')
    sys.modules['Levenshtein'] = stub
    try:
        with contextlib.chdir(m.REPO / 'evaluation'), contextlib.redirect_stdout(buf):
            exec(compile((m.REPO / 'evaluation' / script).read_text(), script, 'exec'), {'__name__': '__main__'})
    finally:
        if saved is None:
            sys.modules.pop('Levenshtein', None)
        else:
            sys.modules['Levenshtein'] = saved
    accs, model, task = {}, None, None
    for line in buf.getvalue().splitlines():
        if line.startswith('TASK:'):
            task = line.split()[1]
        elif line.strip() in {'gpt-3.5-turbo-0613', 'gpt-4-0613', 'llama-3-70b-chat-hf', 'claude-3-opus-20240229',
                              'gemini-1.0-pro-001'}:
            model = line.strip()
        elif ' acc: ' in line:
            accs[(task, model, line.split()[0])] = float(line.split()[2])
    return accs


def test_official_eval_scripts_equal_our_primary_scorer_on_all_released_outputs():
    """eval_shift.py (shift sets) and eval_rot13.py (decode sets), executed as released, give the same per-set accuracy
    as our official_repo_correct on the GPT-3.5 and GPT-4 logs."""
    official = {**_run_official_eval('eval_shift.py'), **_run_official_eval('eval_rot13.py')}
    checked = 0
    for model in m.ARCHIVED_MODELS:
        conds = exp.rescore_archived(model, 'paper_base')['conditions']
        for name, cell in conds.items():
            assert cell['accuracy'] == official[('dec', model, name)], (model, name)
            checked += 1
    assert checked == 2 * 4


def test_primary_scorer_equals_the_figure_tables_item_by_item():
    """evaluation/table_rot13dec_<model>.tsv (input to the published regressions/figures): per-item 'correct'."""
    for model in m.ARCHIVED_MODELS:
        rows = [l.split('\t') for l in (m.REPO / f'evaluation/table_rot13dec_{model}.tsv').read_text().splitlines()[1:]]
        assert len(rows) == 300
        ours = []
        for name in ('rot13dec_highprob', 'rot13dec_mediumprob', 'rot13dec_lowprob'):
            log = m.archived_log(name, model)
            ours += [m.official_repo_correct(r, s['correct_output']) for r, s in zip(log['res'], m.stimuli(name))]
        assert [r[-1].strip() == '1' for r in rows] == ours


def test_published_figure_values_come_from_the_substring_scorer_not_appendix_c1():
    """embers_0915_shifts.R hard-codes the Fig 3.1/6.1 decode accuracies (high, medium, low)."""
    src = (m.REPO / 'evaluation/figures/embers_0915_shifts.R').read_text()
    fig = {k: [float(x) for x in re.search(rf'{k} <- c\(([^)]*)\)', src).group(1).split(',')]
           for k in ('gpt35_dec_acc', 'gpt4_dec_acc')}
    sets = ('rot13dec_highprob', 'rot13dec_mediumprob', 'rot13dec_lowprob')
    for key, model in (('gpt35_dec_acc', 'gpt-3.5-turbo-0613'), ('gpt4_dec_acc', 'gpt-4-0613')):
        conds = exp.rescore_archived(model, 'paper_base')['conditions']
        assert [conds[s]['official_repo_correct']['accuracy'] for s in sets] == fig[key]
    exact35 = exp.rescore_archived('gpt-3.5-turbo-0613', 'paper_base')['conditions']
    assert [exact35[s]['paper_exact']['accuracy'] for s in sets] == [0.21, 0.06, 0.03]  # != published 0.21, 0.07, 0.04


def test_cached_acc_fields_in_released_logs_are_stale_for_gpt35():
    """The logs' own acc field undercounts GPT-3.5 (e.g. 0.16 vs 0.21 for rot13 high); the paper uses 0.21."""
    assert m.archived_log('rot13dec_highprob', 'gpt-3.5-turbo-0613')['acc'] == 0.16
    assert m.archived_log('shift_13', 'gpt-3.5-turbo-0613')['acc'] == 0.17
    assert exp.rescore_archived('gpt-3.5-turbo-0613', 'paper_base')['conditions']['rot13dec_highprob']['accuracy'] == 0.21


@pytest.mark.parametrize('text,official,exact', [
    ('"But this time."', True, True),
    ('But this time.', True, True),
    ('"But this time."\n', True, False),                 # trailing newline: exact fails, substring holds
    (' But this time.', True, False),
    ('Original text: "But this time."', True, False),
    ('**But this time.**', True, False),
    ('“But this time.”', True, False),                   # typographic quotes are not stripped
    ('But this time', False, False),
    ('but this time.', False, False),                     # case-sensitive
    ('', False, False),
    ('Candidate: But this time.\nFinal answer: But that time.', True, False),  # substring credits a revised answer
])
def test_scorer_edge_cases(text, official, exact):
    item = {'expected': 'But this time.'}
    s = exp.score(item, {'content': text}, exp.variant())
    assert (s['correct'], s['official_repo_correct'], s['paper_exact']) == (official, official, exact)


def test_scorer_handles_missing_content_and_edit_distance():
    item = {'expected': 'abc'}
    assert exp.score(item, {'content': None}, exp.variant())['correct'] is False
    assert exp.score(item, {'content': '"abd"'}, exp.variant())['edit_distance'] == 1
    assert m.levenshtein('kitten', 'sitting') == 3


def test_diagnostic_extractor_is_target_blind_v3():
    s = exp.score({'expected': 'X'}, {'content': 'Reasoning...\n\n**Original text:** "Hello there."'}, exp.variant())
    assert s['parsed_answer'] == 'Hello there.' and s['parse_method'] == 'last_answer_label' and not s['extracted_exact']


# ------------------------------------------------------------------ earlier runs
SAVED_SCORER = {'2026-09-30-C057-pilot': 'v1', '2026-09-30-C057-direct0613-pilot': 'v2',
                '2026-09-30-C057-direct0613-full': 'v2', '2026-09-30-C057-directSonnet-pilot': 'v3'}


@pytest.mark.parametrize('run', sorted(SAVED_SCORER))
def test_frozen_scorers_reproduce_saved_scores_and_our_metrics_agree(run):
    """Each run's saved score dict is reproduced by the scorer file it snapshotted; our paper_exact and official
    scorers agree with its paper_exact / official_repo_correct fields on every saved output."""
    frozen = m.prior_scorer(SAVED_SCORER[run])['score']
    recs = legacy_records(RUNS / run)
    scored = [r for r in recs if 'score' in r]
    assert scored and all(r['status'] == 'error' for r in recs if 'score' not in r)
    for r in scored:
        assert frozen(r.get('content'), r['case']) == r['score'], r['case_id']
        item = {'expected': r['case']['expected']}
        ours = exp.score(item, {'content': r.get('content')}, exp.variant(), distances=False)
        assert ours['paper_exact'] == r['score']['paper_exact']
        assert ours['official_repo_correct'] == r['score']['official_repo_correct']


def test_pilot_summary_reproduced():
    """pilot_summary_v2.json: per model x condition, statuses and paper/substring/extracted(v2) counts."""
    summary = json.loads((m.PRIOR / 'pilot_summary_v2.json').read_text())
    recs = legacy_records(RUNS / '2026-09-30-C057-pilot')
    v2 = m.prior_scorer('v2')['score']
    for s in summary:
        rows = exp.prior_rows([r for r in recs if r['case']['condition'] == s['condition']], 'paper_base', s['model'])
        cell = exp.aggregate(rows, exp.variant())['conditions'][s['condition']]
        assert cell['n'] == s['n'] and cell['statuses'] == s['statuses']
        assert cell['paper_exact']['correct'] == s['paper_exact_all']
        assert cell['official_repo_correct']['correct'] == s['repo_substring_all']
        sub = [r for r in recs if r['model'] == s['model'] and r['case']['condition'] == s['condition']]
        assert sum(v2(r['content'], r['case'])['extracted_exact'] for r in sub) == s['extracted_all']
        assert sum(v2(r['content'], r['case'])['extracted_exact'] for r in sub if r['status'] == 'ok') == s['extracted_ok']


def test_direct_gpt4_0613_headline_and_contrasts():
    """EXPANDED_REVIEW.md: direct gpt-4-0613 rot2/high/medium/low 2/49/22/14 (archived 2/51/22/13); high-rot2 47 pp,
    high-low 35 pp; 393/400 items agree with the archived outputs."""
    out = exp.rescore_prior('2026-09-30-C057-direct0613-full', 'paper_base')
    assert [out['conditions'][s]['official_repo_correct']['correct'] for s in m.BASE_SETS] == [2, 49, 22, 14]
    assert [out['conditions'][s]['paper_exact']['correct'] for s in m.BASE_SETS] == [2, 49, 22, 14]
    assert out['contrasts']['rot13dec_highprob-shift_2:official_repo_correct']['difference'] == pytest.approx(0.47)
    assert out['contrasts']['rot13dec_highprob-rot13dec_lowprob:official_repo_correct']['difference'] == pytest.approx(0.35)
    direct = legacy_records(RUNS / '2026-09-30-C057-direct0613-full')
    flips = Counter()
    for r in direct:
        name, idx = r['case']['condition'], r['case']['metadata']['source_index']
        old = m.official_repo_correct(m.archived_log(name, 'gpt-4-0613')['res'][idx], r['case']['expected'])
        new = m.official_repo_correct(r['content'], r['case']['expected'])
        flips[(name, old, new)] += old != new
    conc = {c['condition']: (c['correct_to_incorrect'], c['incorrect_to_correct']) for c in STATS['historical_concordance']}
    for name in m.BASE_SETS:
        assert (flips[(name, True, False)], flips[(name, False, True)]) == conc[name]
    assert sum(flips.values()) == 7


def test_prior_requests_follow_the_registry_and_transport_adaptations_are_logged():
    """Raw requests: original-cap runs sent the case params unchanged to gpt-4-0613; modern endpoints in the pilot had
    their cap raised to 4096 by the transport (recorded in adaptations), so pilot modern-model calls are not at the
    paper's 200-token cap."""
    for r in legacy_records(RUNS / '2026-09-30-C057-direct0613-full'):
        req = eval(r['request']) if isinstance(r['request'], str) else r['request']
        assert req['model'] == 'gpt-4-0613' and req['messages'] == r['case']['messages']
        assert {k: req[k] for k in ('temperature', 'top_p', 'max_tokens', 'stop')} == r['case']['params']
    raised = Counter()
    for r in legacy_records(RUNS / '2026-09-30-C057-pilot'):
        a = r['adaptations'] if isinstance(r['adaptations'], dict) else eval(r['adaptations'])
        raised[(r['model'], a['effective_cap'])] += 1
    assert raised == Counter({('openai/gpt-4', 200): 12, ('openai/gpt-3.5-turbo-0613', 200): 12,
                              ('anthropic/claude-sonnet-5.5', 4096): 12, ('openai/gpt-6.1-sol', 4096): 12,
                              ('z-ai/glm-5.3', 4096): 12})


# ------------------------------------------------------------------ refusals, requests, all shifts
def _req(r):
    return r['request'] if isinstance(r['request'], dict) else eval(r['request'])


def _adapt(r):
    return r['adaptations'] if isinstance(r['adaptations'], dict) else eval(r['adaptations'])


def test_refusal_detection_on_program_outputs():
    v = exp.variant()
    item = exp.items(v, limit=1)[0]
    rows = [{'item': item, 'repeat': 0, 'output': o, 'score': exp.score(item, o, v, distances=False)} for o in (
        {'content': '', 'status': 'empty', 'finish_reason': 'content_filter'},
        {'content': '', 'status': 'empty', 'finish_reason': 'refusal'},
        {'content': '', 'status': 'empty', 'finish_reason': 'stop'},
        {'content': item['expected'], 'status': 'ok', 'finish_reason': 'content_filter'})]
    assert [m.is_refusal(r['output']) for r in rows] == [True, True, False, False]
    cell = exp.aggregate(rows, v)['conditions'][item['set']]
    assert (cell['refusals'], cell['content_filter_with_content'], cell['accuracy'], cell['accuracy_answered']) == (
        2, 1, 0.25, 0.5)


def test_direct_sonnet_pilot_and_transport_adaptations_per_model():
    """Audit m1/m2: what the modern-model runs actually sent (core-policy adaptations, logged per record)."""
    for r in legacy_records(RUNS / '2026-09-30-C057-directSonnet-pilot'):
        req = _req(r)
        assert req['model'] == 'claude-sonnet-5-5' and req['max_tokens'] == 4096
        assert req['thinking']['type'] == 'adaptive' and 'temperature' not in req and 'top_p' not in req
        assert _adapt(r)['omitted_sampling'] == {'temperature': 0.0, 'top_p': 1}
    expected = {'anthropic/claude-sonnet-5.5': ({'max_tokens'}, {'top_p'}),
                'openai/gpt-3.5-turbo-0613': ({'max_completion_tokens'}, set()),
                'openai/gpt-4': ({'max_tokens'}, set()),
                'openai/gpt-6.1-sol': ({'max_tokens'}, {'stop', 'temperature', 'top_p'}),
                'z-ai/glm-5.3': ({'max_tokens'}, set())}
    for run in ('2026-09-30-C057-pilot',):
        for r in legacy_records(RUNS / run):
            cap_key, omitted = expected[r['model']]
            req = _req(r)
            assert cap_key <= set(req) and set(_adapt(r).get('omitted_unsupported', {})) == omitted, (run, r['model'])
            assert not omitted & set(req)


# ------------------------------------------------------------------ robustness
R2 = PROJECT / 'experiments/run_minimal_setups/runs/earlier/runs'
ROBUST_ITEM = {'set': 'rot13dec_highprob', 'expected': 'To this day, we continue to follow these principles.',
               'source_index': 0}


@pytest.mark.parametrize('output,correct,unparsed,path', [
    ({'content': 'To this day, we continue to follow these principles.', 'status': 'ok', 'finish_reason': 'stop'},
     True, False, 'paper'),
    ({'content': 'Decoded: "To this day, we continue to follow these principles."', 'status': 'ok'}, True, False, 'paper'),
    ({'content': '**To this day**, we continue to follow these principles.', 'status': 'ok'}, True, False, 'robust'),
    ({'content': '<think>shift each letter</think>\n**Final answer:** To this day, we *continue* to follow these '
                 'principles.', 'status': 'ok'}, True, False, 'robust'),
    ({'content': 'V -> T\nq -> o\nv -> t\nj -> h', 'status': 'truncated', 'finish_reason': 'length'}, False, True, None),
    ({'content': '', 'status': 'truncated', 'finish_reason': 'length'}, False, True, None),
    ({'content': 'To this day, we continue to follow principles.', 'status': 'ok', 'finish_reason': 'stop'},
     False, False, 'paper'),                                                       # complete wrong answer stays wrong
    ({'content': 'To this day, we continue to follow these principles.', 'status': 'truncated',
      'finish_reason': 'length'}, True, True, None),     # correct before the cut: still missing (no outcome selection)
    ({'content': '', 'status': 'ok', 'finish_reason': 'content_filter'}, False, False, 'paper'),   # refusal stays
])
def test_robust_scoring_modern_formats(output, correct, unparsed, path):
    rp = exp.variant('robust_parse')
    s = exp.score(ROBUST_ITEM, output, rp)
    assert (s['correct'], s['official_repo_correct'], s['unparsed'], s['parse_path']) == (correct, correct, unparsed, path)
    paper = exp.score(ROBUST_ITEM, output, exp.variant('paper_base'))
    assert s['official_repo_correct_paper'] == paper['official_repo_correct']
    if path == 'paper':
        assert s['correct'] == paper['correct']


def test_robust_variants_params():
    base, rp, rb = (exp.variant(n) for n in ('paper_base', 'robust_parse', 'robust'))
    assert (rp.params, rp.prompts, rp.strategy) == (base.params, base.prompts, base.strategy) and not rp.faithful
    assert rp.options == {**base.options, 'parse': 'robust'}
    assert rb.params == {**base.params, 'max_tokens': 2048} and rb.options['min_cap_reasoning'] == 16384
    assert [x['id'] for x in exp.items(rp)] == [x['id'] for x in exp.items(base)]
    case = exp.program(rb).case(exp.items(rb, limit=1)[0])
    assert case['min_cap_reasoning'] == 16384 and case['params']['max_tokens'] == 2048
    assert exp.program(rp).case(exp.items(rp, limit=1)[0]) == exp.program(base).case(exp.items(base, limit=1)[0])
    assert base.faithful and 'parse' not in base.options


def test_robust_cut_off_rows_are_missing_whatever_their_outcome():
    """a cut-off row is missing in both arms whether right or wrong, so accuracy is not
    biased up; the reliability flag counts all of them."""
    rp = exp.variant('robust_parse')
    rows = []
    for item in exp.items(rp, limit=10):
        if item['set'] not in ('rot13dec_highprob', 'rot13dec_lowprob'):
            continue
        cut = item['source_index'] < 4
        right = item['source_index'] % 2 == 0
        out = {'content': item['expected'] if right else 'wrong', 'status': 'truncated' if cut else 'ok',
               'finish_reason': 'length' if cut else 'stop'}
        rows.append({'item': item, 'output': out, 'repeat': 0, 'score': exp.score(item, out, rp)})
    agg = exp.aggregate(rows, rp)
    for name in ('rot13dec_highprob', 'rot13dec_lowprob'):
        cell = agg['conditions'][name]
        assert cell['scored'] == 6 and cell['official_repo_correct']['accuracy'] == 0.5   # 3/6, as without the cut
    assert agg['headline_unparsed'] == 8 and agg['headline_reliable'] is False         # 8/20 cut off > 10%


def test_robust_variant_scores_cut_off_rows():
    """the panel robust run (2026-10-11): at the robust caps a cut-off row is a decoding failure, scored by the paper
    rule on its text; only robust_parse (paper 200-token cap) treats it as missing."""
    rb = exp.variant('robust')
    assert rb.options['truncated'] == 'score' and 'truncated' not in exp.variant('robust_parse').options
    rows = []
    for item in exp.items(rb, limit=10):
        if item['set'] not in ('rot13dec_highprob', 'rot13dec_lowprob'):
            continue
        cut = item['source_index'] < 4
        right = item['source_index'] % 2 == 0 and not cut
        out = {'content': item['expected'] if right else ('' if cut else 'wrong'),
               'status': 'truncated' if cut else 'ok', 'finish_reason': 'length' if cut else 'stop'}
        sc = exp.score(item, out, rb)
        assert not sc['unparsed'] and sc['truncated_scored'] == cut and (sc['correct'] == right)
        rows.append({'item': item, 'output': out, 'repeat': 0, 'score': sc})
    agg = exp.aggregate(rows, rb)
    for name in ('rot13dec_highprob', 'rot13dec_lowprob'):
        cell = agg['conditions'][name]
        assert cell['scored'] == 10 and cell['official_repo_correct']['accuracy'] == 0.3   # items 4, 6, 8
    assert agg['headline_unparsed'] == 0 and agg['truncated_scored'] == 8 and agg['headline_reliable'] is True


def test_robust_aggregate_excludes_cut_off_rows_and_is_tolerant():
    rp = exp.variant('robust_parse')
    rows = []
    for item in exp.items(rp, limit=20):
        if item['set'] == 'rot13dec_lowprob' and item['source_index'] < 2:
            out = {'content': 'V -> T\nq -> o', 'status': 'truncated', 'finish_reason': 'length'}
        else:
            out = {'content': item['expected'] if item['set'] != 'rot13dec_lowprob' else 'nope', 'status': 'ok'}
        rows.append({'item': item, 'output': out, 'repeat': 0, 'score': exp.score(item, out, rp)})
    agg = exp.aggregate(rows, rp)
    c = agg['contrasts']['rot13dec_highprob-rot13dec_lowprob:official_repo_correct']
    assert c['paired_n'] == 18 and c['difference'] == 1.0          # cut-off pairs are not scored wrong
    assert agg['headline_unparsed'] == 2 and agg['headline_rows'] == 40 and agg['headline_reliable'] is True
    paper = exp.aggregate([{**r, 'score': exp.score(r['item'], r['output'], exp.variant('paper_base'))} for r in rows],
                          exp.variant('paper_base'))
    assert paper['contrasts']['rot13dec_highprob-rot13dec_lowprob:official_repo_correct']['paired_n'] == 20
    for r in [r for r in rows if r['item']['set'] == 'rot13dec_highprob'][:3]:
        r['output'] = {'content': '', 'status': 'ok', 'finish_reason': 'refusal'}
        r['score'] = exp.score(r['item'], r['output'], rp)
    agg = exp.aggregate(rows, rp)
    assert agg['headline_refusals'] == 3 and agg['headline_reliable'] is False     # (3 + 2) / 40 > 10%


@pytest.mark.skipif(not R2.exists(), reason='earlier pipeline runs not present')
def test_robust_equals_paper_on_saved_panel20_outputs():
    from shelflife.replay import run_records
    base, rp = exp.variant('paper_base'), exp.variant('robust_parse')
    items = {x['id']: x for x in exp.items(base)}
    compared = 0
    for run in sorted(R2.glob('C057-paper_base-Lall-*/M*')):
        for rec in run_records(run):
            out = rec.get('output')
            if out is None or rec['item_id'] not in items:
                continue
            item = items[rec['item_id']]
            paper, robust = exp.score(item, out, base, distances=False), exp.score(item, out, rp, distances=False)
            assert robust['official_repo_correct_paper'] == paper['official_repo_correct']
            if robust['parse_path'] == 'paper':
                compared += 1
                assert robust['correct'] == paper['correct'] and robust['paper_exact'] == paper['paper_exact']
            elif robust['parse_path'] == 'robust':
                assert robust['correct'] and not paper['correct']
            else:
                assert robust['unparsed']
                assert out.get('finish_reason') == 'length' or out.get('status') == 'truncated' or not paper['correct']
    assert compared > 5000
