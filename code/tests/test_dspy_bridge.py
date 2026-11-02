"""The DSPy front end sends byte-identical payloads to the native path and never collapses repeats."""
import pytest

dspy = pytest.importorskip('dspy')

from shelflife.dspy_bridge import VerbatimPredict  # noqa: E402
from shelflife.ledger import LedgerClient, MockTransport  # noqa: E402
from shelflife.program import LM, SingleTurn  # noqa: E402

BUILDS = {
    'system_user': lambda x: [{'role': 'system', 'content': 'You are X.'}, {'role': 'user', 'content': 'Q: ' + x['q']}],
    'multi_turn': lambda x: [{'role': 'user', 'content': x['q']}, {'role': 'assistant', 'content': 'A'},
                             {'role': 'user', 'content': 'Are you sure?'}],
}
PARAMS = [{}, {'temperature': 0, 'max_tokens': 512}, {'temperature': 1, 'top_p': 0.9, 'stop': ['\n']}]


@pytest.mark.parametrize('build', list(BUILDS))
@pytest.mark.parametrize('params', PARAMS, ids=str)
@pytest.mark.parametrize('model', ['M001', 'M017', 'M206'])
def test_payload_identity(tmp_path, build, params, model):
    payloads = {}
    for name, prog in (('native', SingleTurn(BUILDS[build], params)), ('dspy', VerbatimPredict(BUILDS[build], params))):
        t = MockTransport(lambda p: 'out')
        c = LedgerClient(tmp_path / name, tmp_path / 'l.sqlite3', 10, transport=t)
        out = prog.forward({'id': 'i', 'q': 'hi {braces}'}, LM(c, model, 'i', 0))
        assert out['content'] == 'out'
        payloads[name] = t.payloads
    assert payloads['native'] == payloads['dspy']


def test_repeats_are_distinct_calls(tmp_path):
    t = MockTransport(lambda p: 'out')
    c = LedgerClient(tmp_path / 'r', tmp_path / 'l.sqlite3', 10, transport=t)
    prog = VerbatimPredict(BUILDS['system_user'], {'temperature': 1})
    for r in range(3):
        prog.forward({'id': 'i', 'q': 'x'}, LM(c, 'M017', 'i', r))
    assert len(t.payloads) == 3
