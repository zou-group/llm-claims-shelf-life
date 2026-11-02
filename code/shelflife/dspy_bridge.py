"""Optional DSPy front end over the ledgered transport (pinned to dspy==3.4.0).

Why a bridge rather than plain ``dspy.LM``:
  * DSPy's ChatAdapter/JSONAdapter rewrite prompts into signature formats ([[ ## field ## ]]); faithful reproduction
    needs the original text, so faithful variants use ``VerbatimAdapter``, which sends ``messages`` untouched and
    returns the raw completion;
  * DSPy's response cache would return one completion for repeated identical requests; independent repeated samples
    need distinct calls, so the bridge LM is built with cache=False and our ledger handles resume;
  * every call still goes through LedgerClient (budget, raw records, payload adaptation policy), so a DSPy program and
    a native Program with the same messages produce byte-identical provider payloads (tests/test_dspy_bridge.py).

Use DSPy-native prompting (dspy.Predict/ChainOfThought with the default adapters) only in variants marked
faithful=False, e.g. to test whether a modern prompting strategy changes a claim's outcome.

    class MyDSPyProgram(DSPyProgram):
        def build_module(self):
            return dspy.Predict('messages -> completion', adapter=...)   # or any dspy.Module
        def inputs(self, item):
            return {'messages': [...]}
"""
import threading

import dspy
from dspy.lm15 import Message, Response, TextPart, ThinkingPart, Usage

from .program import Program

_bound = threading.local()


class LedgerEngine:
    """lm15 Engine: complete(Request) -> Response, routed through the shelflife LM handle bound to this thread."""

    PARAMS = ('max_tokens', 'temperature', 'top_p', 'top_k', 'stop', 'seed', 'frequency_penalty', 'presence_penalty')

    def complete(self, request):
        lm = getattr(_bound, 'lm', None)
        if lm is None:
            raise RuntimeError('No shelflife LM bound to this thread: call DSPy modules through DSPyProgram.forward')
        messages = []
        if request.system is not None:
            system = request.system if isinstance(request.system, str) else ''.join(p.text for p in request.system)
            messages.append({'role': 'system', 'content': system})
        for m in request.messages:
            if m.role not in ('user', 'assistant'):
                raise ValueError(f'unsupported role {m.role}: only leading system + user/assistant turns map verbatim')
            messages.append({'role': m.role, 'content': m.text or ''})
        params = {}
        cfg = request.config
        if getattr(_bound, 'params', None) is not None:
            # Verbatim programs pass their source parameters through unchanged (DSPy would coerce 1 -> 1.0).
            params = dict(_bound.params)
        elif cfg is not None:
            for k in self.PARAMS:
                v = getattr(cfg, k, None)
                if v is None or (k == 'stop' and not v):
                    continue
                params[k] = list(v) if k == 'stop' and isinstance(v, tuple) else v
        _bound.counter += 1
        step = f'{_bound.prefix}{_bound.counter}'
        c = lm(messages, params, step=step)
        parts = [TextPart(c.content or '')]
        if c.reasoning:
            parts.insert(0, ThinkingPart(c.reasoning))
        finish = {'length': 'length', 'max_tokens': 'length'}.get(c.finish_reason, 'stop')
        return Response(id=c.cache_key, model=c.returned_model or request.model, message=Message.assistant(parts),
                        finish_reason=finish, usage=Usage(input_tokens=0, output_tokens=0, total_tokens=0))

    def stream(self, request):
        raise NotImplementedError('streaming is not supported by the ledgered bridge')

    def close(self):
        pass


def make_lm(model_label='shelflife', **kwargs):
    """A dspy.LM that routes through the thread-bound ledgered client. DSPy cache disabled."""
    return dspy.LM(f'openai/{model_label}', engine=LedgerEngine(), cache=False, **kwargs)


class VerbatimAdapter(dspy.Adapter):
    """Sends inputs['messages'] exactly as given; output field 'completion' gets the raw text."""

    def format(self, signature, demos, inputs):
        if demos:
            raise ValueError('VerbatimAdapter does not add demos: put any few-shot turns in the messages yourself')
        return [dict(m) for m in inputs['messages']]

    def parse(self, signature, completion):
        return {name: completion for name in signature.output_fields}


class DSPyProgram(Program):
    """Run a DSPy module inside the shelflife runner. Subclasses define build_module() and inputs(item).

    Each LM call made by the module becomes one ledgered call with step '<prefix><n>'. The module runs synchronously
    in the runner's worker thread; parallelism comes from the runner, not dspy.Parallel.
    """
    name = 'dspy_program'
    step_prefix = 'dspy'
    verbatim_params = None   # when set, sent exactly instead of DSPy's normalized generation config

    def __init__(self, adapter=None, lm_kwargs=None):
        self.adapter = adapter
        self.lm = make_lm(**(lm_kwargs or {}))
        self.module = self.build_module()

    def build_module(self):
        raise NotImplementedError

    def inputs(self, item):
        raise NotImplementedError

    def outputs(self, prediction):
        return {k: prediction[k] for k in prediction.keys()}

    def forward(self, item, lm):
        _bound.lm, _bound.counter, _bound.prefix, _bound.params = lm, 0, self.step_prefix, self.verbatim_params
        try:
            with dspy.context(lm=self.lm, adapter=self.adapter):
                prediction = self.module(**self.inputs(item))
        finally:
            _bound.lm = None
        return self.outputs(prediction)


class VerbatimPredict(DSPyProgram):
    """Single-turn faithful program expressed in DSPy: messages built by ``build(item)``, sent verbatim."""

    def __init__(self, build, params=None):
        self.build, self.params = build, dict(params or {})
        self.verbatim_params = self.params
        super().__init__(adapter=VerbatimAdapter())

    def build_module(self):
        return dspy.Predict('messages -> completion')

    def preview(self, item):
        return self.build(item)

    def inputs(self, item):
        return {'messages': self.build(item)}

    def outputs(self, prediction):
        return {'content': prediction.completion}
