"""Auxiliary draws are independent across subjects while legacy keys remain exact."""
import hashlib
import json

import pytest

from shelflife.ledger import LedgerClient, MockTransport
from shelflife.program import LM


MESSAGES = [{'role': 'user', 'content': 'Grade the identical answer'}]
PARAMS = {'max_tokens': 8}


def test_auxiliary_calls_independent_across_subjects_and_repeats_but_resume_same_job(tmp_path):
    transport = MockTransport(lambda payload: 'correct')
    client = LedgerClient(tmp_path / 'run', tmp_path / 'ledger', 5, transport=transport,
                          pacing=False, cache_schema='subject_v2')
    keys = set()
    for subject in ('M001', 'M017'):
        for repeat in (0, 1):
            lm = LM(client, subject, 'same-item', repeat)
            judge = lm.for_model('openai-direct/gpt-4-0613')
            completion = judge(MESSAGES, PARAMS, step='grade')
            keys.add(completion.cache_key)
            assert judge.root_model == subject
            raw = json.loads((tmp_path / 'run/raw' / f'{completion.cache_key}.json').read_text())
            assert raw['root_model'] == subject and raw['cache_schema'] == 'subject_v2'
            assert raw['request']['messages'] == MESSAGES
            resumed = LM(client, subject, 'same-item', repeat).for_model('openai-direct/gpt-4-0613')(
                MESSAGES, PARAMS, step='grade')
            assert resumed.cache_key == completion.cache_key
    assert len(keys) == 4
    assert len(transport.payloads) == 4
    assert all(payload == transport.payloads[0] for payload in transport.payloads)
    from shelflife.replay import run_records
    # Raw-only recovery must not collapse auxiliary samples from different subjects.
    rows = run_records(tmp_path / 'run')
    assert {(r['root_model'], r['repeat']) for r in rows} == {
        (subject, repeat) for subject in ('M001', 'M017') for repeat in (0, 1)}
    assert len(rows) == 4


def test_nested_auxiliary_handles_keep_original_subject(tmp_path):
    client = LedgerClient(tmp_path / 'run', tmp_path / 'ledger', 5, transport=MockTransport(lambda p: 'ok'),
                          pacing=False, cache_schema='subject_v2')
    lm = LM(client, 'M001', 'item')
    nested = lm.for_model('M017').for_model('openai-direct/gpt-4-0613')
    completion = nested(MESSAGES, PARAMS)
    raw = json.loads((tmp_path / 'run/raw' / f'{completion.cache_key}.json').read_text())
    assert raw['root_model'] == 'M001'
    assert raw['model'] == 'openai-direct/gpt-4-0613'


def test_legacy_cache_key_matches_frozen_algorithm_and_ignores_optional_root(tmp_path):
    client = LedgerClient(tmp_path / 'run', tmp_path / 'ledger', 5, transport=MockTransport(lambda p: 'ok'), pacing=False)
    case = {'id': 'item', 'messages': MESSAGES, 'params': PARAMS}
    payload, _, _ = client.plan('M001', case)
    identity = {'run': str((tmp_path / 'run').resolve()), 'case_id': 'item', 'repeat': 0,
                'step': 'main', 'model': 'M001', 'payload': payload}
    expected = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    assert client.cache_key('M001', 'item', 0, 'main', payload) == expected
    assert client.cache_key('M001', 'item', 0, 'main', payload, root_model='M017') == expected
    raw = client.call('M001', case)
    assert raw['cache_key'] == expected
    assert 'cache_schema' not in raw and 'root_model' not in raw


def test_new_schema_never_reuses_legacy_cache_entry(tmp_path):
    transport = MockTransport(lambda p: 'ok')
    legacy = LedgerClient(tmp_path / 'run', tmp_path / 'ledger', 5, transport=transport, pacing=False)
    new = LedgerClient(tmp_path / 'run', tmp_path / 'ledger', 5, transport=transport, pacing=False, cache_schema='subject_v2')
    old_completion = LM(legacy, 'M001', 'item')(MESSAGES, PARAMS)
    new_completion = LM(new, 'M001', 'item')(MESSAGES, PARAMS)
    assert old_completion.cache_key != new_completion.cache_key
    assert len(transport.payloads) == 2


def test_new_schema_requires_explicit_subject_for_direct_client_calls(tmp_path):
    transport = MockTransport(lambda p: 'ok')
    client = LedgerClient(tmp_path / 'run', tmp_path / 'ledger', 5, transport=transport,
                          pacing=False, cache_schema='subject_v2')
    with pytest.raises(ValueError, match='explicit root_model'):
        client.call('M001', {'id': 'i', 'messages': MESSAGES, 'params': PARAMS})
    assert transport.payloads == []
    assert client.totals() == []


def test_legacy_adapter_signature_still_supported():
    class OldAdapter:
        def call(self, model, case, repeat=0, step='main'):
            return {'status': 'ok', 'content': 'ok', 'cache_key': 'old'}

    lm = LM(OldAdapter(), 'subject', 'item').for_model('judge')
    assert lm(MESSAGES, PARAMS).content == 'ok'
    assert lm.root_model == 'subject'


def test_unknown_schema_fails_before_creating_files(tmp_path):
    with pytest.raises(ValueError, match='unknown cache_schema'):
        LedgerClient(tmp_path / 'run', tmp_path / 'ledger', 5, cache_schema='typo')
    assert not (tmp_path / 'run').exists()
