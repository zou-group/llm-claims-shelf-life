"""Execute real programs offline with supplied scenario responses.

The planner has no transport, ledger or filesystem output. It adapts and estimates
the calls reached by ``Program.forward``. A scenario is evidence about one path,
never proof of a bound on response-dependent branches or prompt lengths.
"""
import copy
from collections import Counter

from .policies import get_policy, policy_for
from .program import Completion, LM, SingleTurn


class PlanningClient:
    """In-memory implementation of the small client interface consumed by LM.

    ``responder(request)`` returns text, a Completion, or a completion-summary dict
    (content/status/reasoning/finish_reason/usage/raw_usage/raw_response). Supply
    raw_response for provider-specific fields such as token logprobs. The request includes
    model, root_model, case_id, repeat, step, original case, adapted payload and
    policy. The callback can vary replies by item, model, repeat and named step.
    Without a responder every call receives empty content, explicitly a placeholder.
    """

    def __init__(self, *, policy=None, responder=None, max_calls_per_item=1000):
        if not isinstance(max_calls_per_item, int) or isinstance(max_calls_per_item, bool) or max_calls_per_item < 1:
            raise ValueError('max_calls_per_item must be a positive integer')
        self.fixed_policy = get_policy(policy) if isinstance(policy, str) else policy
        self.responder = responder
        self.max_calls_per_item = max_calls_per_item
        self.calls = []
        self.root_model = None
        self.path_start = 0
        self._policies = {}

    def policy(self, model):
        if model not in self._policies:
            self._policies[model] = (self.fixed_policy if self.fixed_policy is not None
                                     and self.fixed_policy.accepts(model) else policy_for(model))
        return self._policies[model]

    def call(self, model, case, repeat=0, step='main'):
        if len(self.calls) - self.path_start >= self.max_calls_per_item:
            raise ValueError(f'planning stopped at max_calls_per_item={self.max_calls_per_item}; path is incomplete')
        policy = self.policy(model)
        payload, adaptations = policy.adapt(model, copy.deepcopy(case))
        request = {'model': model, 'root_model': self.root_model, 'case_id': case['id'],
                   'repeat': repeat, 'step': step, 'policy': policy.name, 'case': copy.deepcopy(case),
                   'payload': payload, 'adaptations': adaptations,
                   'effective_output_cap': payload.get('max_tokens', payload.get('max_completion_tokens')),
                   'bound_usd': policy.estimate(model, payload),
                   'bound_scope': 'this adapted request, with this scenario prompt'}
        self.calls.append(request)
        response = '' if self.responder is None else self.responder(copy.deepcopy(request))
        if isinstance(response, Completion):
            response = {**response.summary(), 'raw_response': copy.deepcopy(response.raw_response)}
        if isinstance(response, str):
            response = {'content': response}
        if not isinstance(response, dict):
            raise TypeError('planning responder must return str, Completion, or completion-summary dict')
        record = {'status': 'ok' if response.get('content') else 'empty', 'finish_reason': 'stop', **response}
        # Planning has no paid calls or cache. Do not let scenario metadata suggest otherwise.
        record.update(cache_key=None, charged_usd=0.0)
        request['scenario_response'] = Completion.from_record(record, step=step, model=model).summary()
        return record


def plan_program(program, items, models, *, repeats=1, policy=None, responder=None, max_calls_per_item=1000):
    """Return calls, path outcomes and conservative costs for the supplied scenario.

    Every (item, model, repeat) executes ``forward``. Exceptions are retained as
    incomplete paths with their calls, so failed parsing/unsupported requests are
    visible. Policy and program validation failures raise before traversing paths.
    Use several responders to probe meaningful branches; totals from those separate
    scenarios must not be added as if they were independent paid jobs.

    Only an unmodified SingleTurn.forward has a complete request bound. For all
    other programs ``whole_run_bound_usd`` is None even when every scenario path
    returns successfully. Policy estimates are the same conservative estimates used
    to reserve the live ledger, not guarantees about provider billing.
    """
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
        raise ValueError('repeats must be a positive integer')
    models, items = list(models), list(items)
    if not models:
        raise ValueError('at least one model is required')
    if len({item['id'] for item in items}) != len(items):
        raise ValueError('duplicate item ids')
    if len(set(models)) != len(models):
        raise ValueError('duplicate model keys')
    client = PlanningClient(policy=policy, responder=responder, max_calls_per_item=max_calls_per_item)
    for model in models + list(getattr(program, 'auxiliary_models', ()) or ()):
        client.policy(model)
    if hasattr(program, 'validate'):
        program.validate(models)
    paths = []
    for repeat in range(repeats):
        for item in items:
            for model in models:
                client.root_model = model
                client.path_start = len(client.calls)
                lm = LM(client, model, item['id'], repeat,
                        case_extra=getattr(program, 'case_extra', None))
                path = {'model': model, 'case_id': item['id'], 'repeat': repeat, 'status': 'complete'}
                try:
                    program.forward(copy.deepcopy(item), lm)
                except Exception as error:
                    path.update(status='incomplete', error=f'{type(error).__name__}: {error}')
                path['call_indices'] = list(range(client.path_start, len(client.calls)))
                path['calls'] = len(path['call_indices'])
                paths.append(path)
    complete = all(path['status'] == 'complete' for path in paths)
    single_turn = isinstance(program, SingleTurn) and type(program).forward is SingleTurn.forward
    total = sum(call['bound_usd'] for call in client.calls)
    warnings = []
    if not single_turn:
        warnings.append('Scenario traversal does not bound unvisited branches or longer responses used in later prompts.')
    if responder is None and not single_turn:
        warnings.append('Empty placeholder responses were used; provide a responder to exercise meaningful branches.')
    if not complete:
        warnings.append('Some paths are incomplete; costs include only calls adapted before the failure.')
    declared = set(getattr(program, 'auxiliary_models', ()) or ())
    undeclared = sorted({call['model'] for call in client.calls
                         if call['model'] != call['root_model'] and call['model'] not in declared})
    if undeclared:
        warnings.append('Declare observed auxiliary models in program.auxiliary_models for live preflight: '
                        + ', '.join(undeclared))
    return {'scope': 'single_turn_requests' if single_turn else 'scenario_only',
            'response_source': 'empty_placeholder' if responder is None else 'provided_responder',
            'calls': client.calls, 'paths': paths, 'warnings': warnings,
            'totals': {'items': len(items), 'models': len(models), 'repeats': repeats,
                       'paths': len(paths), 'completed_paths': sum(p['status'] == 'complete' for p in paths),
                       'calls': len(client.calls), 'calls_by_model': dict(Counter(c['model'] for c in client.calls)),
                       'scenario_bound_usd': total,
                       'whole_run_bound_usd': total if single_turn and complete else None}}
