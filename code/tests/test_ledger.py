import io
import json
import urllib.error

import pytest

from shelflife.ledger import BudgetExceeded, LedgerClient, MockTransport

CASE = {'id': 'c1', 'messages': [{'role': 'user', 'content': 'hi'}], 'params': {'temperature': 0, 'max_tokens': 16}}


def client(tmp_path, transport, ceiling=1.0, **kw):
    return LedgerClient(tmp_path / 'run', tmp_path / 'ledger.sqlite3', ceiling, transport=transport, sleep=lambda s: None, **kw)


def test_record_saved_and_cached(tmp_path):
    t = MockTransport(lambda p: 'answer')
    c = client(tmp_path, t)
    r1 = c.call('M001', CASE)
    r2 = c.call('M001', CASE)
    assert r1['content'] == 'answer' and r1['status'] == 'ok'
    assert r2 == json.loads((tmp_path / 'run/raw' / f"{r1['cache_key']}.json").read_text())
    assert len(t.payloads) == 1  # resume never resubmits


def test_repeats_and_steps_are_independent_calls(tmp_path):
    t = MockTransport(lambda p: 'x')
    c = client(tmp_path, t)
    keys = {c.call('M001', CASE, repeat=r, step=s)['cache_key'] for r in range(3) for s in ('a', 'b')}
    assert len(keys) == 6 and len(t.payloads) == 6


def test_no_secrets_in_records(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-SECRET-123')
    c = client(tmp_path, MockTransport(lambda p: 'x'))
    rec = c.call('M001', CASE)
    assert 'SECRET' not in (tmp_path / 'run/raw' / f"{rec['cache_key']}.json").read_text()


def test_budget_ceiling_blocks_before_submission(tmp_path):
    t = MockTransport(lambda p: 'x')
    c = client(tmp_path, t, ceiling=1e-9)
    with pytest.raises(BudgetExceeded):
        c.call('M001', CASE)
    assert t.payloads == []


def test_task_budget(tmp_path):
    t = MockTransport(lambda p: {'model': 'm', 'choices': [{'message': {'content': 'x'}, 'finish_reason': 'stop'}],
                                 'usage': {'prompt_tokens': 10**6, 'completion_tokens': 0}})
    c = client(tmp_path, t, ceiling=100, task='T', task_budget_usd=5.0001)
    c.call('M001', CASE)  # gpt-4o-2024-05-13: 1M prompt tokens = $5
    with pytest.raises(BudgetExceeded):
        c.call('M001', {**CASE, 'id': 'c2'})


class Flaky:
    def __init__(self, codes):
        self.codes, self.payloads = list(codes), []

    def __call__(self, url, headers, payload):
        self.payloads.append(payload)
        if self.codes:
            code = self.codes.pop(0)
            if code == 'timeout':
                raise TimeoutError('timed out')
            raise urllib.error.HTTPError(url, code, 'err', {}, io.BytesIO(b'{"error":"x"}'))
        return {'model': 'm', 'choices': [{'message': {'content': 'ok'}, 'finish_reason': 'stop'}], 'usage': {}}


def test_explicit_rejections_retried_with_identical_payload(tmp_path):
    t = Flaky([429, 503])
    rec = client(tmp_path, t).call('M001', CASE)
    assert rec['status'] == 'ok' and len(rec['retries']) == 2
    assert t.payloads[0] == t.payloads[1] == t.payloads[2]


def test_timeouts_not_replayed_and_keep_reserve(tmp_path):
    t = Flaky(['timeout'])
    rec = client(tmp_path, t).call('M001', CASE)
    assert rec['status'] == 'error' and len(t.payloads) == 1
    assert rec['charged_usd'] == rec['reserve_usd'] and rec['transport_failure_kind'] == 'timeout'


def test_circuit_breaker_after_three_transport_failures(tmp_path):
    c = client(tmp_path, Flaky(['timeout'] * 3))
    for i in range(3):
        c.call('M001', {**CASE, 'id': f'c{i}'})
    with pytest.raises(RuntimeError, match='circuit open'):
        c.call('M001', {**CASE, 'id': 'c9'})


def test_4xx_validation_error_unbilled_on_panel_policy(tmp_path):
    rec = client(tmp_path, Flaky([400])).call('M001', CASE)
    assert rec['status'] == 'error' and rec['charged_usd'] == 0.0
