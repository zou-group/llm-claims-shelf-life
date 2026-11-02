"""Prompting strategies as programs.

A Program maps one dataset item to one or more model calls and returns an output dict. It plays the role of a DSPy
module's ``forward``, but every call goes through ``LM``, which sends the program's messages verbatim through the
ledgered client, so faithful variants keep the original prompt byte for byte.

  SingleTurn       one request built from the item (most claims)
  Program subclass anything else: multi-turn challenges ("Are you sure?"), generate-then-judge pipelines,
                   response-dependent follow-ups, agent loops. Call ``lm(messages, params, step=...)`` once per turn.

Each call gets a distinct ``step`` name, which enters the cache key. Repeated samples (``repeat``) are independent
calls. If a call errors, ``LM`` raises CallFailed so a later turn never runs on a missing earlier turn; the runner
records the partial trace.
"""
import copy
from dataclasses import dataclass, field


def normalize_usage(raw):
    """Comparable token counts, with ``None`` for unreported quantities.

    ``input_tokens`` includes cache reads/writes. Anthropic reports those separately
    from ordinary input; OpenAI-style prompt_tokens already includes cached input.
    ``output_tokens`` includes hidden reasoning when the provider includes it.
    Never infer visible-output tokens by subtracting reasoning: provider conventions
    differ. Keep ``Completion.raw_usage`` for paper-specific accounting.
    """
    raw = raw if isinstance(raw, dict) else {}

    def count(value):
        return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None

    input_tokens = count(raw.get('prompt_tokens', raw.get('input_tokens')))
    output_tokens = count(raw.get('completion_tokens', raw.get('output_tokens')))
    prompt_details = raw.get('prompt_tokens_details') or raw.get('input_tokens_details') or {}
    output_details = raw.get('completion_tokens_details') or raw.get('output_tokens_details') or {}
    prompt_details = prompt_details if isinstance(prompt_details, dict) else {}
    output_details = output_details if isinstance(output_details, dict) else {}
    cached = count(raw.get('cache_read_input_tokens', prompt_details.get('cached_tokens')))
    created = count(raw.get('cache_creation_input_tokens'))
    if ('prompt_tokens' not in raw and input_tokens is not None
            and ('cache_read_input_tokens' in raw or 'cache_creation_input_tokens' in raw)):
        input_tokens += (cached or 0) + (created or 0)
    total = count(raw.get('total_tokens'))
    if total is None and input_tokens is not None and output_tokens is not None:
        total = input_tokens + output_tokens
    return {'input_tokens': input_tokens, 'output_tokens': output_tokens, 'total_tokens': total,
            'cached_input_tokens': cached, 'cache_creation_input_tokens': created,
            'reasoning_tokens': count(output_details.get('reasoning_tokens', raw.get('reasoning_tokens')))}


class CallFailed(RuntimeError):
    def __init__(self, completion):
        super().__init__(f'{completion.step}: {completion.status} {completion.error or ""}'.strip())
        self.completion = completion


@dataclass
class Completion:
    step: str
    status: str            # ok | truncated | empty | error | not_submitted
    model: str = None
    content: str = ''
    reasoning: str = ''
    finish_reason: str = None
    returned_model: str = None
    cache_key: str = None
    charged_usd: float = 0.0
    error: str = None
    messages: list = field(default_factory=list)
    params: dict = field(default_factory=dict)
    usage: dict = field(default_factory=lambda: normalize_usage({}))
    raw_usage: dict = field(default_factory=dict)
    # Full provider body for protocol-specific fields (e.g. token logprobs).
    # Deliberately omitted from summary(): the ledger already persists this body.
    raw_response: dict = field(default_factory=dict)

    @classmethod
    def from_record(cls, record, *, step, model=None, messages=None, params=None):
        """Read a ledger record or saved completion summary without reopening raw files.

        ``raw_response`` is a detached copy of the provider body, or a scenario's
        explicit raw_response. Summary-only records cannot reconstruct it and
        return an empty dict; use ledger-backed replay when a program needs it.
        """
        response = record.get('raw_response', record.get('response')) or {}
        response = response if isinstance(response, dict) else {}
        raw = record.get('raw_usage', response.get('usage', {}) if isinstance(response, dict) else {})
        raw = copy.deepcopy(raw) if isinstance(raw, dict) else {}
        return cls(step=step, model=model if model is not None else record.get('model'),
                   status=record['status'], content=record.get('content') or '',
                   reasoning=record.get('reasoning') or '', finish_reason=record.get('finish_reason'),
                   returned_model=record.get('returned_model'), cache_key=record.get('cache_key'),
                   charged_usd=record.get('charged_usd') or 0.0,
                   error=record.get('error') or record.get('error_body'),
                   messages=copy.deepcopy(messages if messages is not None else record.get('messages', [])),
                   params=copy.deepcopy(params if params is not None else record.get('params', {})),
                   usage=copy.deepcopy(record.get('usage') or normalize_usage(raw)), raw_usage=raw,
                   raw_response=copy.deepcopy(response))

    @property
    def usable(self):
        return self.status in {'ok', 'truncated', 'empty'}

    def summary(self):
        return {'step': self.step, 'model': self.model, 'status': self.status, 'content': self.content, 'reasoning': self.reasoning,
                'finish_reason': self.finish_reason, 'returned_model': self.returned_model, 'cache_key': self.cache_key,
                'charged_usd': self.charged_usd, 'error': self.error,
                'usage': copy.deepcopy(self.usage), 'raw_usage': copy.deepcopy(self.raw_usage)}


class LM:
    """Callable handed to Program.forward. Bound to one (root model, item, repeat).

    ``model`` is the current call's model; ``root_model`` always identifies the
    model under test, including through nested auxiliary handles.
    """

    def __init__(self, client, model, item_id, repeat=0, case_extra=None, raise_on_error=True, trace=None,
                 root_model=None):
        self.client, self.model, self.item_id, self.repeat = client, model, item_id, repeat
        self.root_model = model if root_model is None else root_model
        self.case_extra = dict(case_extra or {})
        self.raise_on_error = raise_on_error
        self.trace = [] if trace is None else trace

    def for_model(self, model, case_extra=None):
        """Handle for another model (judge, grader, reformulator) sharing this item's repeat index and trace.

        The judge's calls go through the same ledger; its policy is the run's policy when that policy accepts the
        model, else the model's default policy (recorded per call in the raw record)."""
        return LM(self.client, model, self.item_id, self.repeat, case_extra=case_extra,
                  raise_on_error=self.raise_on_error, trace=self.trace, root_model=self.root_model)

    def __call__(self, messages, params=None, step='main'):
        if any(c.step == step and c.model == self.model for c in self.trace):
            raise ValueError(f'duplicate step name {step!r}: each call in a program needs a distinct step')
        messages = copy.deepcopy(list(messages))
        for m in messages:
            if set(m) - {'role', 'content', 'name'} or m['role'] not in {'system', 'user', 'assistant'}:
                raise ValueError(f'unsupported message {m}')
        case = {'id': self.item_id, 'messages': messages, 'params': dict(params or {}), **self.case_extra}
        try:
            # Old replay/planning adapters keep their original call signature.
            identity = {'root_model': self.root_model} if getattr(self.client, 'cache_schema', None) == 'subject_v2' else {}
            record = self.client.call(self.model, case, repeat=self.repeat, step=step, **identity)
        except RuntimeError as e:  # budget, guard or unresolved ledger entry: nothing was submitted
            completion = Completion(step=step, model=self.model, status='not_submitted', error=str(e), messages=messages, params=case['params'])
        else:
            completion = Completion.from_record(record, step=step, model=self.model,
                                                messages=messages, params=case['params'])
        self.trace.append(completion)
        if not completion.usable and self.raise_on_error:
            raise CallFailed(completion)
        return completion


class Program:
    """Base class. Subclasses implement forward(item, lm) -> dict.

    Declare every model the program calls besides the model under test in ``auxiliary_models``; the runner checks
    they are routable before submitting anything. Optional ``validate(models)`` may raise to stop a run up front.
    """
    name = 'program'
    auxiliary_models = ()

    def forward(self, item, lm):
        raise NotImplementedError

    def preview(self, item):
        """Messages of the first request (for prompt-fidelity tests and dry runs); None if response-dependent."""
        return None


class SingleTurn(Program):
    """One request per item. ``build(item) -> messages``; params are the original generation parameters.

    ``params_for(item)`` may be overridden when parameters vary per item (e.g. per-task output caps).
    """
    name = 'single_turn'

    def __init__(self, build, params=None, case_extra=None):
        self.build = build
        self.params = dict(params or {})
        self.case_extra = dict(case_extra or {})

    def params_for(self, item):
        return dict(self.params)

    def preview(self, item):
        return self.build(item)

    def forward(self, item, lm):
        c = lm(self.build(item), self.params_for(item), step='main')
        return {'content': c.content, 'finish_reason': c.finish_reason, 'status': c.status,
                'usage': copy.deepcopy(c.usage), 'raw_usage': copy.deepcopy(c.raw_usage)}

    def case(self, item):
        """Legacy-format case (``{"id","messages","params",...}``) as written to cases.jsonl by earlier runs."""
        return {'id': item['id'], 'messages': self.build(item), 'params': self.params_for(item), **self.case_extra}
