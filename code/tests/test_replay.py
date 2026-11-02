"""Core helpers that every new claim gets for free: stream merging, default rescore, generic multi-turn replay."""
import json

import pytest

from shelflife import stats
from shelflife.ledger import MockTransport
from shelflife.program import Program
from shelflife.replay import replay_shelflife_run, run_records
from shelflife.runner import run
from test_runner import Toy, adder


def _write(path, rows):
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))


def test_run_records_merge_rule(tmp_path):
    base = {'model': 'M1', 'repeat': 0}
    _write(tmp_path / 'responses.jsonl', [{**base, 'case_id': 'a', 'status': 'error'},
                                          {**base, 'case_id': 'b', 'status': 'ok', 'content': 'first'}])
    _write(tmp_path / 'responses_continuation_001.jsonl', [{**base, 'case_id': 'a', 'status': 'ok', 'content': 'recovered'},
                                                           {**base, 'case_id': 'b', 'status': 'error'}])
    got = {r['case_id']: r for r in run_records(tmp_path)}
    assert got['a']['content'] == 'recovered'          # usable recovery supersedes the failed original
    assert got['b']['content'] == 'first'              # a later failure never replaces a usable output
    assert len(run_records(tmp_path, streams=['responses.jsonl'])) == 2


def test_run_records_preserve_completed_runner_conversation(tmp_path):
    base = {'model': 'M001', 'item_id': 'a', 'repeat': 0}
    _write(tmp_path / 'responses.jsonl', [{**base, 'status': 'complete', 'output': {'content': 'A'}}])
    _write(tmp_path / 'responses_recovery.jsonl', [{**base, 'status': 'not_submitted', 'output': None}])
    assert run_records(tmp_path)[0]['output'] == {'content': 'A'}


def test_default_rescore_and_unmatched_records(tmp_path):
    exp = Toy()
    run(exp, 'base', ['M001'], tmp_path / 'run', ledger=tmp_path / 'l', ceiling_usd=1,
        transport=MockTransport(adder), progress_every=0)
    assert exp.rescore(tmp_path / 'run')['M001'] == {'n': 5, 'accuracy': 1.0}
    (tmp_path / 'legacy').mkdir()
    _write(tmp_path / 'legacy' / 'responses.jsonl',
           [{'model': 'M9', 'case_id': 'i1', 'repeat': 0, 'status': 'ok', 'content': '2'},
            {'model': 'M9', 'case_id': 'nope', 'repeat': 0, 'status': 'ok', 'content': '2'}])
    with pytest.raises(ValueError, match='match no item'):
        exp.rescore(tmp_path / 'legacy')
    assert exp.rescore(tmp_path / 'legacy', allow_unmatched=True)['M9']['accuracy'] == 1.0


class Challenge(Program):
    def __init__(self, follow_up='Are you sure?'):
        self.follow_up = follow_up

    def forward(self, item, lm):
        first = lm([{'role': 'user', 'content': f"Add {item['a']}+{item['b']}"}], {'max_tokens': 8}, step='initial')
        second = lm([{'role': 'user', 'content': f"Add {item['a']}+{item['b']}"}, {'role': 'assistant', 'content': first.content},
                     {'role': 'user', 'content': self.follow_up}], {'max_tokens': 8}, step='challenge')
        return {'initial': first.content, 'final': second.content}


def test_generic_multi_turn_replay_detects_prompt_drift(tmp_path):
    exp = Toy()
    exp.program = lambda variant: Challenge()
    exp.score = lambda item, output, variant: {}
    exp.aggregate = lambda rows, variant: {}
    run(exp, 'base', ['M001', 'M017'], tmp_path / 'run', ledger=tmp_path / 'l', ceiling_usd=1, repeats=2,
        transport=MockTransport(lambda p: str(len(p['messages']))), progress_every=0)
    out = replay_shelflife_run(exp, 'base', tmp_path / 'run')
    assert out['conversations'] == 20 and out['unmatched'] == [] and out['unused'] == 0
    exp.program = lambda variant: Challenge('Are you certain?')
    drift = replay_shelflife_run(exp, 'base', tmp_path / 'run')
    assert len(drift['unmatched']) == 20


def test_rate_summary():
    rows = [{'item': {'id': i}, 'score': {'ok': i % 2 == 0}} for i in range(10)] + [{'item': {'id': 99}, 'score': None}]
    out = stats.rate_summary(rows, lambda r: r['score']['ok'])
    assert (out['n'], out['scored'], out['k'], out['rate']) == (11, 10, 5, 0.5)
    assert out['bounds_with_missing'] == [5 / 11, 6 / 11]


# Hand-authored raw/trace fixtures avoid using the runner under test to construct replay's expected identities.
SUBJECT = 'openai-direct/gpt-4-0613'
JUDGE = 'openai-direct/gpt-4o-2024-11-20'


def _saved_call(key, *, model=SUBJECT, item='i0', repeat=0, step='first', prompt='Q', content='A', status='ok'):
    return {'cache_key': key, 'model': model, 'case_id': item, 'repeat': repeat, 'step': step,
            'policy': 'initial_openai_v1', 'status': status, 'content': content,
            'request': {'model': model.split('/', 1)[1], 'messages': [{'role': 'user', 'content': prompt}],
                        'max_tokens': 8}, 'charged_usd': 0}


def _trace(rec):
    return {k: rec[k] for k in ('cache_key', 'model', 'step', 'status', 'content')}


def _saved_run(path, records, rows, subjects=(SUBJECT,)):
    path.mkdir()
    (path / 'raw').mkdir()
    (path / 'configuration.json').write_text(json.dumps({'models': list(subjects), 'policy': 'initial_openai_v1'}))
    for rec in records:
        (path / 'raw' / (rec['cache_key'] + '.json')).write_text(json.dumps(rec))
    _write(path / 'responses.jsonl', rows)
    return path


def _row(records, *, model=SUBJECT, item='i0', repeat=0, status='complete'):
    return {'model': model, 'item_id': item, 'repeat': repeat, 'status': status, 'trace': records}


def _experiment(program):
    exp = Toy()
    exp.program = lambda variant: program
    return exp


class SavedSequence(Program):
    def __init__(self, *, first='Q', second_model=None, first_step='first', stop=False):
        self.first, self.second_model, self.first_step, self.stop = first, second_model, first_step, stop

    def forward(self, item, lm):
        initial = lm([{'role': 'user', 'content': self.first}], {'max_tokens': 8}, step=self.first_step)
        if self.stop:
            return {'content': initial.content}
        judge = lm.for_model(self.second_model) if self.second_model else lm
        final = judge([{'role': 'user', 'content': 'Follow ' + initial.content}], {'max_tokens': 8}, step='second')
        return {'content': final.content}


def test_identical_payloads_preserve_item_repeat_and_auxiliary_responses(tmp_path):
    records, rows = [], []
    for item in ('i0', 'i1'):
        for repeat in (0, 1):
            answer = f'{item}-{repeat}'
            first = _saved_call(answer, item=item, repeat=repeat, content=answer)
            second = _saved_call('judge-' + answer, model=JUDGE, item=item, repeat=repeat,
                                 step='second', prompt='Follow ' + answer)
            records.extend((first, second))
            rows.append(_row([_trace(first), _trace(second)], item=item, repeat=repeat))
    path = _saved_run(tmp_path / 'run', records, rows)
    out = replay_shelflife_run(_experiment(SavedSequence(second_model=JUDGE)), 'base', path)
    assert out['conversations'] == 4  # never four additional conversations for the judge
    assert out['matched'] == 8 and out['unmatched'] == [] and out['unused'] == 0
    limited = replay_shelflife_run(_experiment(SavedSequence(second_model=JUDGE)), 'base', path, limit=1)
    assert (limited['conversations'], limited['requests'], limited['matched'], limited['unused']) == (1, 2, 2, 0)
    empty = replay_shelflife_run(_experiment(SavedSequence()), 'base', path, limit=0)
    assert (empty['conversations'], empty['requests'], empty['unused']) == (0, 0, 0)


def test_shared_judge_is_not_a_phantom_subject_conversation(tmp_path):
    first = _saved_call('a')
    judge = _saved_call('b', model=JUDGE, step='second', prompt='Follow A')
    path = _saved_run(tmp_path / 'run', [first, judge], [_row([_trace(first), _trace(judge)])],
                      subjects=(SUBJECT, JUDGE))
    out = replay_shelflife_run(_experiment(SavedSequence(second_model=JUDGE)), 'base', path)
    assert out['conversations'] == 1 and out['matched'] == 2 and not out['unmatched']


def test_unsent_is_tolerated_only_at_exact_model_and_step(tmp_path):
    first = _saved_call('a')
    unsent = {'model': SUBJECT, 'step': 'second', 'status': 'not_submitted', 'cache_key': None}
    path = _saved_run(tmp_path / 'run', [first], [_row([_trace(first), unsent], status='not_submitted')])
    exact = replay_shelflife_run(_experiment(SavedSequence()), 'base', path)
    assert exact['matched'] == 1 and exact['unsent_tolerated'] == 1 and not exact['unmatched']
    # A changed earlier prompt must not disappear just because some later step was never sent.
    drift = replay_shelflife_run(_experiment(SavedSequence(first='Changed')), 'base', path)
    assert len(drift['unmatched']) == 1 and drift['unsent_tolerated'] == 0
    other_model = replay_shelflife_run(_experiment(SavedSequence(second_model=JUDGE)), 'base', path)
    assert len(other_model['unmatched']) == 1 and other_model['unsent_tolerated'] == 0


def test_entirely_unsent_conversation_is_replayed(tmp_path):
    trace = [{'model': SUBJECT, 'step': 'first', 'status': 'not_submitted', 'cache_key': None}]
    path = _saved_run(tmp_path / 'run', [], [_row(trace, status='not_submitted')])
    out = replay_shelflife_run(_experiment(SavedSequence()), 'base', path)
    assert (out['conversations'], out['matched'], out['unsent_tolerated']) == (1, 0, 1)
    assert not out['unmatched']


def test_partial_failure_stops_and_missing_raw_is_not_tolerated(tmp_path):
    first = _saved_call('a', status='error', content='')
    path = _saved_run(tmp_path / 'run', [first], [_row([_trace(first)], status='call_error')])
    out = replay_shelflife_run(_experiment(SavedSequence()), 'base', path)
    assert out['matched'] == 1 and not out['unmatched'] and out['unused'] == 0
    (path / 'raw/a.json').unlink()
    missing = replay_shelflife_run(_experiment(SavedSequence()), 'base', path)
    assert len(missing['unmatched']) == 1 and missing['unsent_tolerated'] == 0
    assert 'missing' in missing['unmatched_details'][0]['reason']


def test_raw_only_interrupted_conversation_uses_exact_step_identity(tmp_path):
    first = _saved_call('a')
    second = _saved_call('b', step='second', prompt='Follow A')
    path = _saved_run(tmp_path / 'run', [first, second], [])
    out = replay_shelflife_run(_experiment(SavedSequence()), 'base', path)
    assert out['matched'] == 2 and not out['unmatched'] and out['unused'] == 0
    changed = replay_shelflife_run(_experiment(SavedSequence(first_step='renamed')), 'base', path)
    assert len(changed['unmatched']) == 1 and changed['matched'] == 0 and changed['unused'] == 2


def test_removed_program_step_and_missing_item_are_visible(tmp_path):
    first = _saved_call('a')
    second = _saved_call('b', step='second', prompt='Follow A')
    path = _saved_run(tmp_path / 'run', [first, second], [_row([_trace(first), _trace(second)])])
    stopped = replay_shelflife_run(_experiment(SavedSequence(stop=True)), 'base', path)
    assert stopped['matched'] == 1 and stopped['unused'] == 1
    exp = _experiment(SavedSequence())
    exp.items = lambda variant: []
    missing = replay_shelflife_run(exp, 'base', path)
    assert len(missing['unmatched']) == 1 and missing['unused'] == 2


def test_partial_trace_does_not_allow_extra_steps(tmp_path):
    first = _saved_call('a')
    path = _saved_run(tmp_path / 'run', [first], [_row([_trace(first)])])
    out = replay_shelflife_run(_experiment(SavedSequence()), 'base', path)
    assert out['matched'] == 1 and len(out['unmatched']) == 1
    assert 'beyond' in out['unmatched_details'][0]['reason']


def test_payload_comparison_preserves_json_value_types(tmp_path):
    first = _saved_call('a')
    first['request']['max_tokens'] = 8.0  # Python dict equality alone would equate this with integer 8.
    path = _saved_run(tmp_path / 'run', [first], [_row([_trace(first)])])
    out = replay_shelflife_run(_experiment(SavedSequence(stop=True)), 'base', path)
    assert out['matched'] == 0 and len(out['unmatched']) == 1


def test_omitted_unsent_step_is_protocol_drift(tmp_path):
    first = _saved_call('a')
    unsent = {'model': SUBJECT, 'step': 'second', 'status': 'not_submitted', 'cache_key': None}
    path = _saved_run(tmp_path / 'run', [first], [_row([_trace(first), unsent], status='not_submitted')])
    out = replay_shelflife_run(_experiment(SavedSequence(stop=True)), 'base', path)
    assert out['unused'] == 0 and len(out['unmatched']) == 1
    assert out['unmatched_details'][0]['steps'] == ['second']


def test_raw_only_interruption_reports_unverifiable_tail_separately(tmp_path):
    first = _saved_call('a')
    path = _saved_run(tmp_path / 'run', [first], [])
    out = replay_shelflife_run(_experiment(SavedSequence()), 'base', path)
    assert out['matched'] == 1 and not out['unmatched'] and out['unused'] == 0
    assert out['incomplete_tail'] == [{'conversation': (SUBJECT, 'i0', '0'), 'model': SUBJECT, 'step': 'second'}]
    assert out['unsent_tolerated'] == 0  # this is unknown coverage, not a recorded not_submitted call


def test_unresolved_resume_checks_original_raw_request_and_preserves_stop(tmp_path):
    first = _saved_call('a', status='error', content='')
    trace = [{'model': SUBJECT, 'step': 'first', 'status': 'not_submitted', 'cache_key': None}]
    path = _saved_run(tmp_path / 'run', [first], [_row(trace, status='not_submitted')])
    out = replay_shelflife_run(_experiment(SavedSequence()), 'base', path)
    assert out['matched'] == 1 and out['unused'] == 0 and not out['unmatched']
    assert out['unsent_tolerated'] == 0  # the original request was available to verify
    drift = replay_shelflife_run(_experiment(SavedSequence(first='Changed')), 'base', path)
    assert len(drift['unmatched']) == 1 and drift['unused'] == 1


def test_root_model_disambiguates_raw_only_auxiliary_calls(tmp_path):
    records = []
    for model in (SUBJECT, JUDGE):
        answer = model.split('/', 1)[1]
        first = _saved_call('first-' + answer, model=model, content=answer)
        second = _saved_call('second-' + answer, model=JUDGE, step='second', prompt='Follow ' + answer)
        records.extend([{**r, 'root_model': model, 'cache_schema': 'subject_v2'} for r in (first, second)])
    path = _saved_run(tmp_path / 'run', records, [], subjects=(SUBJECT, JUDGE))
    out = replay_shelflife_run(_experiment(SavedSequence(second_model=JUDGE)), 'base', path)
    assert out['conversations'] == 2 and out['matched'] == 4
    assert not out['unmatched'] and not out['unused'] and not out['incomplete_tail']
