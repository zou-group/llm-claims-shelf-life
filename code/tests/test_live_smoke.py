"""Paid end-to-end smoke test of the HTTP path (opt-in: pytest --live -m live). Costs well under $0.01.

One tiny request per route (OpenAI direct, Anthropic direct, OpenRouter) through LedgerClient with a $0.05 ceiling.
"""
import pytest

from shelflife.ledger import LedgerClient

CASE = {'id': 'smoke', 'messages': [{'role': 'user', 'content': 'Reply with the single word: pong'}],
        'params': {'temperature': 0, 'max_tokens': 5}}


@pytest.mark.live
@pytest.mark.parametrize('model', ['M015', 'M119', 'M061'])
def test_live_route(tmp_path, model):
    client = LedgerClient(tmp_path / 'run', tmp_path / 'ledger.sqlite3', 0.05)
    rec = client.call(model, CASE)
    assert rec['status'] in {'ok', 'truncated'}, rec.get('error_body') or rec.get('error')
    assert 'pong' in rec['content'].lower()
    assert rec['charged_usd'] < 0.01
