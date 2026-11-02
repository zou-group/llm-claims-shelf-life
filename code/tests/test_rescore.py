"""Scoring revisions preserve generation evidence and the intended denominator."""
import json

import pytest

from shelflife.ledger import MockTransport
from shelflife.rescore import rescore_run
from shelflife.runner import run
from test_runner import Toy, adder


def saved_run(tmp_path):
    exp = Toy()
    path = tmp_path / 'run'
    run(exp, 'base', ['M001'], path, repeats=2, ledger=tmp_path / 'ledger', ceiling_usd=1,
        transport=MockTransport(adder), progress_every=0)
    return exp, path


def test_rescore_uses_saved_items_and_preserves_source(tmp_path):
    exp, path = saved_run(tmp_path)
    before = {p.name: p.read_bytes() for p in path.iterdir() if p.is_file()}
    exp.items = lambda *a, **k: pytest.fail('must use saved items')
    exp.score = lambda item, output, variant: {'correct': False}
    result = rescore_run(exp, path, tmp_path / 'scoring-v2')
    assert result['metrics']['M001'] == {'n': 10, 'accuracy': 0}
    assert result['provenance']['expected_rows'] == 10
    assert result['provenance']['absent_rows'] == 0
    assert before == {p.name: p.read_bytes() for p in path.iterdir() if p.is_file()}
    with pytest.raises(FileExistsError):
        rescore_run(exp, path, tmp_path / 'scoring-v2')


def test_interrupted_rows_are_not_dropped(tmp_path):
    exp, path = saved_run(tmp_path)
    response_path = path / 'responses.jsonl'
    lines = response_path.read_text().splitlines()
    response_path.write_text('\n'.join(lines[:3]) + '\n')
    exp.aggregate = lambda rows, v: {'n': len(rows), 'missing': sum(r['output'] is None for r in rows)}
    out = rescore_run(exp, path, tmp_path / 'revision')
    assert out['metrics']['M001'] == {'n': 10, 'missing': 7}
    assert out['provenance']['absent_rows'] == 7


def test_raw_calls_do_not_become_program_outputs(tmp_path):
    exp, path = saved_run(tmp_path)
    (path / 'responses.jsonl').write_text('')
    exp.aggregate = lambda rows, v: {'n': len(rows), 'missing': sum(r['output'] is None for r in rows)}
    out = rescore_run(exp, path, tmp_path / 'revision')
    assert out['metrics']['M001'] == {'n': 10, 'missing': 10}


def test_modified_items_or_variant_rejected_before_writing(tmp_path):
    exp, path = saved_run(tmp_path)
    items_path = path / 'items.jsonl'
    items_path.write_text(items_path.read_text().replace('"b": 1', '"b": 2'))
    with pytest.raises(ValueError, match='items_sha256'):
        rescore_run(exp, path, tmp_path / 'revision')
    assert not (tmp_path / 'revision').exists()


def test_changed_variant_rejected_before_writing(tmp_path):
    exp, path = saved_run(tmp_path)
    exp.variants['base'] = exp.variant().override('base', params={'max_tokens': 16})
    with pytest.raises(ValueError, match='variant changed'):
        rescore_run(exp, path, tmp_path / 'revision')
    assert not (tmp_path / 'revision').exists()


@pytest.mark.parametrize('update', [dict(models=['M001', 'M001']), dict(repeats=0),
                                    dict(repeats=True), dict(n_items=999)])
def test_invalid_saved_grid_rejected(tmp_path, update):
    exp, path = saved_run(tmp_path)
    config_path = path / 'configuration.json'
    config_path.write_text(json.dumps({**json.loads(config_path.read_text()), **update}))
    with pytest.raises(ValueError):
        rescore_run(exp, path, tmp_path / 'revision')
    assert not (tmp_path / 'revision').exists()


def test_new_response_stream_during_rescore_rejected(tmp_path):
    exp, path = saved_run(tmp_path)
    def aggregate(rows, variant):
        (path / 'responses_resume_000001.jsonl').write_text('')
        return {'n': len(rows)}
    exp.aggregate = aggregate
    with pytest.raises(RuntimeError, match='source run changed'):
        rescore_run(exp, path, tmp_path / 'revision')
    assert not (tmp_path / 'revision').exists()


def test_rescore_wrong_claim_output_inside_run_and_scoring_errors(tmp_path):
    exp, path = saved_run(tmp_path)
    with pytest.raises(ValueError, match='outside'):
        rescore_run(exp, path, path / 'revision')
    exp.claim_id = 'OTHER'
    with pytest.raises(ValueError, match='runner run'):
        rescore_run(exp, path, tmp_path / 'revision')
    exp.claim_id = 'TOY'
    def broken(*a):
        raise RuntimeError('scorer failed')
    exp.score = broken
    with pytest.raises(RuntimeError, match='scorer failed'):
        rescore_run(exp, path, tmp_path / 'revision')
    assert not (tmp_path / 'revision').exists()
