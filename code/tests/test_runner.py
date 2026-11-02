import json

import pytest

from shelflife.ledger import MockTransport
from shelflife.program import Program, SingleTurn
from shelflife.prompts import PromptTemplate
from shelflife.registry import Experiment, Variant
from shelflife.runner import run


class Toy(Experiment):
    claim_id = 'TOY'
    default_variant = 'base'

    def build_variants(self):
        return {'base': Variant('base', 'toy', prompts={'user': PromptTemplate('user', 'Add {a}+{b}')},
                                params={'temperature': 0, 'max_tokens': 8})}

    def items(self, variant, limit=None):
        rows = [{'id': f'i{a}', 'a': a, 'b': 1} for a in range(5)]
        return rows[:limit] if limit else rows

    def program(self, variant):
        return SingleTurn(lambda x: [{'role': 'user', 'content': variant.prompts['user'].render(a=x['a'], b=x['b'])}],
                          variant.params)

    def score(self, item, output, variant):
        return {'correct': output['content'].strip() == str(item['a'] + item['b'])}

    def aggregate(self, rows, variant):
        scored = [r for r in rows if r['score']]
        return {'n': len(rows), 'accuracy': sum(r['score']['correct'] for r in scored) / len(scored)}


def adder(payload):
    a, b = payload['messages'][-1]['content'].removeprefix('Add ').split('+')
    return str(int(a) + int(b))


def test_run_resume_and_metrics(tmp_path):
    exp = Toy()
    t = MockTransport(adder)
    kw = dict(ledger=tmp_path / 'l.sqlite3', ceiling_usd=1, transport=t, workers=4, progress_every=0)
    out = run(exp, 'base', ['M001', 'M017'], tmp_path / 'run', repeats=2, **kw)
    assert out['metrics']['M001'] == {'n': 10, 'accuracy': 1.0}
    assert len(t.payloads) == 20
    original = (tmp_path / 'run/responses.jsonl').read_bytes()
    config_original = (tmp_path / 'run/configuration.json').read_bytes()
    run(exp, 'base', ['M001', 'M017'], tmp_path / 'run', repeats=2, **kw)
    assert len(t.payloads) == 20  # resumed entirely from the ledger
    assert (tmp_path / 'run/responses.jsonl').read_bytes() == original
    assert (tmp_path / 'run/configuration.json').read_bytes() == config_original
    assert (tmp_path / 'run/responses_resume_000001.jsonl').exists()
    cfg = json.loads((tmp_path / 'run/configuration.json').read_text())
    assert cfg['faithful'] and (tmp_path / 'run/code_snapshot/shelflife/runner.py').exists()
    assert cfg['cache_schema'] == 'subject_v2'


def test_crashing_resume_preserves_saved_evidence(tmp_path):
    exp = Toy()
    kw = dict(ledger=tmp_path / 'l', ceiling_usd=1, transport=MockTransport(adder),
              workers=1, progress_every=0)
    run_dir = tmp_path / 'run'
    run(exp, 'base', ['M001'], run_dir, **kw)
    original = {name: (run_dir / name).read_bytes() for name in
                ('responses.jsonl', 'items.jsonl', 'configuration.json', 'metrics.json')}
    program = exp.program(exp.variant())
    def crash(item, lm):
        raise ValueError('unexpected program bug')
    program.forward = crash
    exp.program = lambda variant: program
    with pytest.raises(ValueError, match='unexpected program bug'):
        run(exp, 'base', ['M001'], run_dir, **kw)
    assert all((run_dir / name).read_bytes() == data for name, data in original.items())
    assert (run_dir / 'responses_resume_000001.jsonl').exists()


def test_changed_variant_refuses_same_run_dir(tmp_path):
    exp = Toy()
    kw = dict(ledger=tmp_path / 'l.sqlite3', ceiling_usd=1, transport=MockTransport(adder), progress_every=0)
    run(exp, 'base', ['M001'], tmp_path / 'run', **kw)
    changed = exp.add_variant(exp.variant('base').override('alt', prompts={'user': 'Add {a} + {b}'}))
    assert not changed.faithful
    with pytest.raises(RuntimeError, match='configuration changed'):
        run(exp, 'alt', ['M001'], tmp_path / 'run', **kw)


class Challenge(Program):
    """Two-turn response-dependent program (like 'Are you sure?')."""

    def forward(self, item, lm):
        first = lm([{'role': 'user', 'content': 'Q'}], {'max_tokens': 8}, step='initial')
        second = lm([{'role': 'user', 'content': 'Q'}, {'role': 'assistant', 'content': first.content},
                     {'role': 'user', 'content': 'Are you sure?'}], {'max_tokens': 8}, step='challenge')
        return {'initial': first.content, 'final': second.content}


def test_multi_turn_program_and_failures(tmp_path):
    exp = Toy()
    exp.program = lambda variant: Challenge()
    exp.score = lambda item, output, variant: {'flipped': output['initial'] != output['final']}
    exp.aggregate = lambda rows, variant: {'n': len(rows)}
    seen = []

    def responder(payload):
        seen.append(payload['messages'])
        return 'A' if len(payload['messages']) == 1 else 'B'
    out = run(exp, 'base', ['M001'], tmp_path / 'run', ledger=tmp_path / 'l', ceiling_usd=1,
              transport=MockTransport(responder), limit=2, progress_every=0)
    assert all(r['score'] == {'flipped': True} for r in out['rows'])
    challenges = [m for m in seen if len(m) == 3]   # items run concurrently: request order is not fixed
    assert len(challenges) == 2 and all(m[1] == {'role': 'assistant', 'content': 'A'} for m in challenges)

    def broken(payload):
        if len(payload['messages']) == 1:
            raise TimeoutError('timed out')
        return 'B'
    out = run(exp, 'base', ['M001'], tmp_path / 'run2', ledger=tmp_path / 'l2', ceiling_usd=1,
              transport=MockTransport(broken), limit=1, progress_every=0)
    row = out['rows'][0]
    assert row['status'] == 'call_error' and len(row['trace']) == 1 and row['output'] is None


class GenerateThenJudge(Program):
    def forward(self, item, lm):
        answer = lm([{'role': 'user', 'content': 'Q'}], {'max_tokens': 8}, step='answer')
        judge = lm.for_model('openai-direct/gpt-4-0613')
        verdict = judge([{'role': 'user', 'content': 'Grade: ' + answer.content}], {'temperature': 0, 'max_tokens': 4},
                        step='grade')
        return {'answer': answer.content, 'grade': verdict.content, 'status': 'complete' if verdict.content else 'no_grade'}


def test_judge_on_its_own_model_and_policy(tmp_path):
    exp = Toy()
    exp.program = lambda variant: GenerateThenJudge()
    exp.score = lambda item, output, variant: {}
    exp.aggregate = lambda rows, variant: {}
    t = MockTransport(lambda p: 'A' if p['model'] != 'gpt-4-0613' else 'CORRECT')
    out = run(exp, 'base', ['M206'], tmp_path / 'run', ledger=tmp_path / 'l', ceiling_usd=5, transport=t, limit=1,
              policy='panel_v1', progress_every=0)
    row = out['rows'][0]
    assert [(c['step'], c['model']) for c in row['trace']] == [('answer', 'M206'), ('grade', 'openai-direct/gpt-4-0613')]
    assert t.payloads[1] == {'model': 'gpt-4-0613', 'messages': [{'role': 'user', 'content': 'Grade: A'}],
                             'temperature': 0, 'max_tokens': 4}   # initial_openai_v1 policy for the judge
    assert row['output_status'] == 'complete'
    done = json.loads((tmp_path / 'run/completion.json').read_text())
    assert done['output_status_counts'] == {'complete': 1}


def test_preflight_blocks_unroutable_auxiliary_model_before_any_call(tmp_path):
    exp = Toy()
    prog = GenerateThenJudge()
    prog.auxiliary_models = ('openai/gpt-4-0314-not-a-real-route',)
    exp.program = lambda variant: prog
    t = MockTransport(lambda p: 'x')
    with pytest.raises(RuntimeError, match='preflight failed'):
        run(exp, 'base', ['M206'], tmp_path / 'run', ledger=tmp_path / 'l', ceiling_usd=5, transport=t, limit=1,
            progress_every=0)
    assert t.payloads == []
