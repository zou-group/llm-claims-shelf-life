"""Exercise public agent workflows through the CLI, without a provider."""
import json

import pytest

from shelflife import cli, registry
from shelflife.program import Program
from test_runner import Toy


class Pipeline(Program):
    auxiliary_models = ('openai-direct/gpt-4-0613',)

    def forward(self, item, lm):
        answer = lm([{'role': 'user', 'content': 'Answer'}], {'max_tokens': 8}, step='answer')
        verdict = lm.for_model(self.auxiliary_models[0])(
            [{'role': 'user', 'content': 'Grade ' + answer.content}], {'max_tokens': 4}, step='judge')
        return {'content': verdict.content}


def test_plan_cli_follows_two_steps_and_repeats(tmp_path, monkeypatch, capsys):
    exp = Toy()
    exp.program = lambda v: Pipeline()
    monkeypatch.setattr(registry, 'get', lambda cid: exp)
    from shelflife.ledger import HTTPTransport
    monkeypatch.setattr(HTTPTransport, '__call__', lambda *a: pytest.fail('network call'))
    replies = tmp_path / 'scenario.json'
    replies.write_text(json.dumps({'answer': '42', 'judge': 'CORRECT'}))
    out = tmp_path / 'plan.json'
    cli.main(['plan', 'TOY', '--models', 'M001', '--limit', '2', '--repeats', '3',
              '--responses-json', str(replies), '--out', str(out), '--show', '0'])
    plan = json.loads(out.read_text())
    assert plan['totals']['calls'] == 12
    assert plan['totals']['whole_run_bound_usd'] is None
    assert len([c for c in plan['calls'] if c['payload']['messages'][0]['content'] == 'Grade 42']) == 6
    assert 'scenario_only' in capsys.readouterr().out
    with pytest.raises(FileExistsError):
        cli.main(['plan', 'TOY', '--models', 'M001', '--out', str(out)])


def test_incomplete_plan_cli_retains_diagnostics_and_fails(tmp_path, monkeypatch):
    exp = Toy()
    exp.program = lambda v: Pipeline()
    monkeypatch.setattr(registry, 'get', lambda cid: exp)
    replies = tmp_path / 'scenario.json'
    replies.write_text('{"answer": "42"}')
    out = tmp_path / 'plan.json'
    with pytest.raises(SystemExit) as exc:
        cli.main(['plan', 'TOY', '--models', 'M001', '--limit', '1',
                  '--responses-json', str(replies), '--out', str(out)])
    assert exc.value.code == 1
    report = json.loads(out.read_text())
    assert report['paths'][0]['status'] == 'incomplete'
    assert 'no synthetic response' in report['paths'][0]['error']


def test_rescore_cli_writes_revision(tmp_path, monkeypatch, capsys):
    from test_rescore import saved_run
    exp, path = saved_run(tmp_path)
    monkeypatch.setattr(registry, 'get', lambda cid: exp)
    dest = tmp_path / 'v2'
    cli.main(['rescore', 'TOY', '--run-dir', str(path), '--out-dir', str(dest)])
    result = json.loads(capsys.readouterr().out)
    assert result['provenance']['network_calls'] == 0
    assert (dest / 'code_snapshot/shelflife/rescore.py').exists()


def test_index_cli_preserves_exclusive_output_and_resolves_explicit_roots(tmp_path, monkeypatch):
    from shelflife import run_manifest
    exp = Toy()
    monkeypatch.setattr(registry, 'get', lambda cid: exp)
    def build(received_exp, roots):
        assert received_exp is exp and roots == [tmp_path.resolve()]
        return {'version': 1, 'claim_id': 'TOY', 'prior_runs': [], 'excluded_runs': {}}
    monkeypatch.setattr(run_manifest, 'build_manifest', build)
    out = tmp_path / 'manifest.json'
    cli.main(['index-runs', 'TOY', '--roots', str(tmp_path), '--out', str(out)])
    before = out.read_bytes()
    assert json.loads(before)['claim_id'] == 'TOY'
    with pytest.raises(FileExistsError):
        cli.main(['index-runs', 'TOY', '--roots', str(tmp_path), '--out', str(out)])
    assert out.read_bytes() == before


def test_show_includes_resolved_run_metadata(monkeypatch, capsys):
    exp = Toy()
    exp.run_manifest = 'engineering/example.json'
    exp.all_prior_runs = lambda: (registry.PriorRun('experiments/example', 'base'),)
    exp.all_excluded_runs = lambda: {'experiments/fixture': 'Explicit offline fixture'}
    monkeypatch.setattr(registry, 'get', lambda cid: exp)
    cli.main(['show', 'TOY'])
    shown = json.loads(capsys.readouterr().out)
    assert shown['run_manifest'] == exp.run_manifest
    assert shown['prior_runs'][0]['path'] == 'experiments/example'
    assert shown['excluded_runs'] == exp.all_excluded_runs()


@pytest.mark.parametrize('option', ['--limit', '--show'])
def test_plan_rejects_negative_prefix_or_display_count(option):
    with pytest.raises(ValueError, match='nonnegative'):
        cli.main(['plan', 'TOY', '--models', 'M001', option, '-1'])


@pytest.mark.parametrize('kwargs', [{'workers': 0}, {'repeats': 0}, {'repeats': -1}, {'limit': -2}])
def test_invalid_execution_grid_rejected_before_calls(tmp_path, kwargs):
    from shelflife.runner import run
    from shelflife.ledger import MockTransport
    transport = MockTransport(lambda p: pytest.fail('should fail before calls'))
    with pytest.raises(ValueError):
        run(Toy(), 'base', ['M001'], tmp_path / 'run', ledger=tmp_path / 'ledger',
            ceiling_usd=1, transport=transport, **kwargs)
    assert not transport.payloads
