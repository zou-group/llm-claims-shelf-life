"""Offline request traversal, cost semantics and provider-usage regressions."""
import json

import pytest

from shelflife.ledger import LedgerClient, MockTransport
from shelflife.planning import plan_program
from shelflife.program import Completion, LM, Program, SingleTurn, normalize_usage


class ToyPolicy:
    name = 'toy'

    def accepts(self, model):
        return model in {'subject', 'judge'}

    def adapt(self, model, case):
        cap = max(4, case['params'].get('max_tokens', 4))
        return ({'model': model, 'messages': case['messages'], 'max_tokens': cap},
                {'omitted_unsupported': {'temperature': case['params']['temperature']}
                 if 'temperature' in case['params'] else {}, 'effective_cap': cap})

    def estimate(self, model, payload):
        return (sum(len(m['content']) for m in payload['messages']) + payload['max_tokens']) / 1000

    def endpoint(self, model):
        pytest.fail('offline planning must never prepare a network endpoint')


class BranchAndJudge(Program):
    auxiliary_models = ('judge',)

    def forward(self, item, lm):
        first = lm([{'role': 'user', 'content': item['question']}], {'max_tokens': 2}, step='answer')
        if first.content == 'retry':
            first = lm([{'role': 'user', 'content': 'try again'}], {'max_tokens': 4}, step='retry')
        verdict = lm.for_model('judge')([{'role': 'user', 'content': 'Grade ' + first.content}],
                                       {'max_tokens': 4}, step='grade')
        return {'answer': first.content, 'grade': verdict.content}


def test_actual_forward_counts_repeats_branches_auxiliary_and_payloads(monkeypatch):
    monkeypatch.setattr(LedgerClient, '__init__', lambda *a, **kw: pytest.fail('no temporary ledger'))

    def reply(request):
        if request['step'] == 'answer':
            return 'retry' if request['repeat'] == 0 else 'B'
        return 'A' if request['step'] == 'retry' else 'correct'

    report = plan_program(BranchAndJudge(), [{'id': 'i', 'question': 'Q'}], ['subject'],
                          repeats=2, policy=ToyPolicy(), responder=reply)
    assert [(c['model'], c['step'], c['repeat']) for c in report['calls']] == [
        ('subject', 'answer', 0), ('subject', 'retry', 0), ('judge', 'grade', 0),
        ('subject', 'answer', 1), ('judge', 'grade', 1)]
    assert report['calls'][2]['payload']['messages'] == [{'role': 'user', 'content': 'Grade A'}]
    assert report['calls'][4]['payload']['messages'] == [{'role': 'user', 'content': 'Grade B'}]
    assert report['calls'][2]['root_model'] == 'subject'
    assert report['totals']['calls_by_model'] == {'subject': 3, 'judge': 2}
    assert report['totals']['scenario_bound_usd'] == pytest.approx(.045)
    assert report['totals']['whole_run_bound_usd'] is None
    assert report['totals']['completed_paths'] == 2
    assert report['scope'] == 'scenario_only'
    json.dumps(report)


def test_single_turn_complete_cost_bound_and_adaptations_include_repeats():
    program = SingleTurn(lambda item: [{'role': 'user', 'content': 'Q'}],
                         {'max_tokens': 2, 'temperature': .7}, case_extra={'min_cap_reasoning': 100})
    report = plan_program(program, [{'id': 'i'}], ['subject'], repeats=3, policy=ToyPolicy())
    assert report['totals']['whole_run_bound_usd'] == pytest.approx(.015)
    assert report['totals']['calls'] == 3
    assert report['calls'][0]['effective_output_cap'] == 4
    assert report['calls'][0]['adaptations']['omitted_unsupported'] == {'temperature': .7}
    assert report['calls'][0]['case']['min_cap_reasoning'] == 100
    assert report['scope'] == 'single_turn_requests'


def test_empty_placeholder_scenario_does_not_claim_all_paths():
    report = plan_program(BranchAndJudge(), [{'id': 'i', 'question': 'Q'}], ['subject'], policy=ToyPolicy())
    assert report['response_source'] == 'empty_placeholder'
    assert report['totals']['calls'] == 2
    assert report['totals']['whole_run_bound_usd'] is None
    assert any('placeholder' in warning for warning in report['warnings'])


def test_loop_guard_keeps_partial_plan():
    class Loop(Program):
        def forward(self, item, lm):
            i = 0
            while True:
                lm([{'role': 'user', 'content': 'Q'}], step=f'turn{i}')
                i += 1

    report = plan_program(Loop(), [{'id': 'i'}], ['subject'], policy=ToyPolicy(), max_calls_per_item=3)
    assert report['totals']['calls'] == 3
    assert report['totals']['completed_paths'] == 0
    assert 'max_calls_per_item=3' in report['paths'][0]['error']
    assert report['totals']['whole_run_bound_usd'] is None


def test_invalid_response_parse_keeps_first_request_and_marks_incomplete():
    class Parse(Program):
        def forward(self, item, lm):
            return {'number': int(lm([{'role': 'user', 'content': 'Q'}]).content)}

    report = plan_program(Parse(), [{'id': 'i'}], ['subject'], policy=ToyPolicy())
    assert report['paths'][0]['status'] == 'incomplete'
    assert 'ValueError' in report['paths'][0]['error']
    assert report['totals']['calls'] == 1


def test_duplicate_steps_use_same_validation_as_live_lm():
    class Duplicate(Program):
        def forward(self, item, lm):
            lm([{'role': 'user', 'content': 'Q'}])
            lm([{'role': 'user', 'content': 'Q'}])

    report = plan_program(Duplicate(), [{'id': 'i'}], ['subject'], policy=ToyPolicy())
    assert 'duplicate step name' in report['paths'][0]['error']
    assert report['totals']['calls'] == 1


def test_policy_fallback_and_case_extra_match_live_request(tmp_path):
    class Judge(Program):
        auxiliary_models = ('openai-direct/gpt-4-0613',)
        case_extra = {'min_cap_reasoning': 5000}

        def forward(self, item, lm):
            answer = lm([{'role': 'user', 'content': 'Q'}], {'max_tokens': 2}, step='answer')
            return {'grade': lm.for_model(self.auxiliary_models[0])(
                [{'role': 'user', 'content': answer.content}], {'max_tokens': 4}, step='grade').content}

    program = Judge()
    report = plan_program(program, [{'id': 'i'}], ['M206'], policy='panel_v2', responder=lambda r: 'A')
    client = LedgerClient(tmp_path / 'run', tmp_path / 'ledger', ceiling_usd=5,
                          policy='panel_v2', transport=MockTransport(lambda payload: 'A'), pacing=False)
    program.forward({'id': 'i'}, LM(client, 'M206', 'i', case_extra=program.case_extra))
    records = [json.loads(p.read_text()) for p in (tmp_path / 'run/raw').glob('*.json')]
    by_step = {record['step']: record for record in records}
    for call in report['calls']:
        actual = by_step[call['step']]
        assert call['payload'] == actual['request']
        assert call['adaptations'] == actual['adaptations']
        assert call['bound_usd'] == actual['reserve_usd']
    assert report['calls'][1]['policy'] == 'initial_openai_v1'


def test_scenario_completion_failure_stops_later_calls():
    report = plan_program(BranchAndJudge(), [{'id': 'i', 'question': 'Q'}], ['subject'],
                          policy=ToyPolicy(), responder=lambda r: Completion(step='scenario', status='error', error='fixture'))
    assert report['totals']['calls'] == 1
    assert report['totals']['completed_paths'] == 0
    assert 'CallFailed' in report['paths'][0]['error']


def test_undeclared_auxiliary_is_visible_in_plan():
    program = BranchAndJudge()
    program.auxiliary_models = ()
    report = plan_program(program, [{'id': 'i', 'question': 'Q'}], ['subject'], policy=ToyPolicy())
    assert any('Declare observed auxiliary models' in warning and 'judge' in warning for warning in report['warnings'])


@pytest.mark.parametrize('raw, expected', [
    ({'prompt_tokens': 20, 'completion_tokens': 10, 'total_tokens': 30,
      'prompt_tokens_details': {'cached_tokens': 5}, 'completion_tokens_details': {'reasoning_tokens': 6}},
     {'input_tokens': 20, 'output_tokens': 10, 'total_tokens': 30, 'cached_input_tokens': 5,
      'cache_creation_input_tokens': None, 'reasoning_tokens': 6}),
    ({'input_tokens': 20, 'output_tokens': 10, 'cache_read_input_tokens': 5, 'cache_creation_input_tokens': 7},
     {'input_tokens': 32, 'output_tokens': 10, 'total_tokens': 42, 'cached_input_tokens': 5,
      'cache_creation_input_tokens': 7, 'reasoning_tokens': None}),
    ({'input_tokens': 20, 'output_tokens': 10, 'input_tokens_details': {'cached_tokens': 5}},
     {'input_tokens': 20, 'output_tokens': 10, 'total_tokens': 30, 'cached_input_tokens': 5,
      'cache_creation_input_tokens': None, 'reasoning_tokens': None}),
])
def test_usage_provider_conventions(raw, expected):
    assert normalize_usage(raw) == expected


def test_usage_missing_values_are_unknown_and_reported_zero_is_zero():
    assert all(value is None for value in normalize_usage({}).values())
    assert normalize_usage({'prompt_tokens': 0, 'completion_tokens': 0})['total_tokens'] == 0
    assert normalize_usage({'prompt_tokens': 2})['total_tokens'] is None
    assert normalize_usage({'prompt_tokens': True, 'completion_tokens': -3})['total_tokens'] is None


def test_completion_reads_real_ledger_usage_and_summary_round_trip(tmp_path):
    raw = {'prompt_tokens': 9, 'completion_tokens': 7, 'completion_tokens_details': {'reasoning_tokens': 4}}
    response = {'model': 'gpt-4-0613', 'choices': [{'message': {'content': 'A'}, 'finish_reason': 'stop'}], 'usage': raw}
    client = LedgerClient(tmp_path / 'run', tmp_path / 'ledger', ceiling_usd=1,
                          transport=MockTransport(lambda p: response), pacing=False)
    completion = LM(client, 'openai-direct/gpt-4-0613', 'i')(
        [{'role': 'user', 'content': 'Q'}], {'max_tokens': 10})
    assert completion.usage['output_tokens'] == 7
    assert completion.usage['reasoning_tokens'] == 4
    assert completion.raw_usage == raw
    summary = completion.summary()
    round_trip = Completion.from_record(json.loads(json.dumps(summary)), step='main')
    assert round_trip.usage == completion.usage
    assert round_trip.raw_usage == raw
    summary['raw_usage']['prompt_tokens'] = 100
    assert completion.raw_usage['prompt_tokens'] == 9


def test_old_completion_summary_remains_readable():
    completion = Completion.from_record({'status': 'ok', 'content': 'A'}, step='main')
    assert completion.content == 'A'
    assert completion.raw_usage == {}
    assert completion.usage['output_tokens'] is None


def test_single_turn_exposes_usage_without_custom_program(tmp_path):
    response = {'model': 'gpt-4-0613', 'choices': [{'message': {'content': 'A'}, 'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 9, 'completion_tokens': 7}}
    client = LedgerClient(tmp_path / 'run', tmp_path / 'ledger', ceiling_usd=1,
                          transport=MockTransport(lambda p: response), pacing=False)
    program = SingleTurn(lambda i: [{'role': 'user', 'content': 'Q'}], {'max_tokens': 10})
    output = program.forward({'id': 'i'}, LM(client, 'openai-direct/gpt-4-0613', 'i'))
    assert output['usage']['output_tokens'] == 7
    assert output['raw_usage'] == response['usage']


def test_completion_exposes_detached_provider_response_without_summary_duplication(tmp_path):
    response = {'model': 'gpt-4-0613',
                'choices': [{'message': {'content': 'A'}, 'finish_reason': 'stop',
                             'logprobs': {'content': [{'token': 'A', 'logprob': -0.25}]}}],
                'usage': {'prompt_tokens': 9, 'completion_tokens': 1}}
    client = LedgerClient(tmp_path / 'run', tmp_path / 'ledger', ceiling_usd=1,
                          transport=MockTransport(lambda p: response), pacing=False)
    completion = LM(client, 'openai-direct/gpt-4-0613', 'i')(
        [{'role': 'user', 'content': 'Q'}], {'max_tokens': 10, 'logprobs': True})
    assert completion.raw_response['choices'][0]['logprobs']['content'][0]['logprob'] == -0.25
    assert 'raw_response' not in completion.summary()
    completion.raw_response['choices'][0]['logprobs']['content'][0]['logprob'] = -100
    assert response['choices'][0]['logprobs']['content'][0]['logprob'] == -0.25


@pytest.mark.parametrize('return_completion', [False, True])
def test_planning_scenario_supplies_provider_fields_without_file_io(return_completion):
    raw = {'choices': [{'logprobs': {'content': [{'token': 'A', 'logprob': -0.25}]}}],
           'usage': {'prompt_tokens': 5, 'completion_tokens': 1}}
    seen = []

    class LogprobProgram(Program):
        def forward(self, item, lm):
            completion = lm([{'role': 'user', 'content': 'Q'}], step='answer')
            seen.append(completion.raw_response['choices'][0]['logprobs']['content'][0]['logprob'])
            completion.raw_response['choices'][0]['logprobs']['content'][0]['logprob'] = -100
            return {'score': seen[-1]}

    def responder(request):
        fields = {'status': 'ok', 'content': 'A', 'raw_response': raw}
        return Completion.from_record(fields, step='answer') if return_completion else fields

    report = plan_program(LogprobProgram(), [{'id': 'i'}], ['subject'], policy=ToyPolicy(), responder=responder)
    assert seen == [-0.25]
    assert report['totals']['completed_paths'] == 1
    assert raw['choices'][0]['logprobs']['content'][0]['logprob'] == -0.25
    assert report['calls'][0]['scenario_response']['usage']['output_tokens'] == 1


def test_missing_provider_fields_keep_planning_explicitly_incomplete():
    class LogprobProgram(Program):
        def forward(self, item, lm):
            completion = lm([{'role': 'user', 'content': 'Q'}])
            return {'score': completion.raw_response['choices'][0]['logprobs']}

    report = plan_program(LogprobProgram(), [{'id': 'i'}], ['subject'], policy=ToyPolicy())
    assert report['paths'][0]['status'] == 'incomplete'
    assert 'KeyError' in report['paths'][0]['error']


def test_empty_models_rejected():
    with pytest.raises(ValueError, match='at least one model'):
        plan_program(SingleTurn(lambda i: []), [{'id': 'i'}], [])


@pytest.mark.parametrize('repeats', [0, -1, True, 1.5])
def test_invalid_repeat_counts_fail_before_planning(repeats):
    with pytest.raises(ValueError, match='repeats'):
        plan_program(SingleTurn(lambda i: []), [], [], repeats=repeats)
