"""C068 AbstentionBench: items, judge prompt/parser against the authors' code, judging program and robust scoring."""
import pytest

from shelflife import policies, registry
from shelflife.experiments import c068_abstentionbench as m
from shelflife.ledger import MockTransport
from shelflife.paths import INITIAL
from shelflife.program import LM
from shelflife.prompts import extract_text
from shelflife.replay import read_jsonl
from shelflife.runner import run

exp = registry.get('C068')
REPRO = INITIAL / 'reproductions/C068'


# ----------------------------------------------------------------------------------------------- items / subset


def test_subset_composition():
    items = exp.items(exp.variant())
    assert len(items) == 330 and sum(x['should_abstain'] for x in items) == 195
    fam = {}
    for x in items:
        if x['dataset'] == 'gsm8k':
            fam.setdefault(x['pair_id'], []).append(x)
    assert len(fam) == 25
    for a, b in fam.values():  # context / context-removed pair of the same GSM8K problem
        assert (a['should_abstain'], b['should_abstain']) == (False, True)
        assert a['question'].endswith(b['question']) and a['reference_answers'] == b['reference_answers']
    coconot = {x['subset'] for x in items if x['dataset'] == 'coconot'}
    assert 'coconot_underspecification' in coconot  # included by the project, dropped by the paper's filter_data
    assert all(x['source_metadata'].get('CoCoNot_original_category') != 'Requests with safety concerns' for x in items)


def test_coconot_snapshot_matches_parquet_row_counts():
    """JSONL transcription (no pyarrow here) keeps the parquet row counts used by the official loader."""
    coco = m.source_datasets()['coconot']
    assert coco.num_abstain_questions == 1001 and len(coco) == 1001 + 379


# ------------------------------------------------------------------------------------------------ judge prompt


# --------------------------------------------------------------------------------------- judging program
class ReplayClient:
    """Stand-in client: returns the saved record for (step, model, item, repeat) and records the case sent."""

    def __init__(self, gen, judges):
        self.gen, self.judges, self.sent = gen, judges, []

    def call(self, model, case, repeat=0, step='main'):
        self.sent.append((step, model, case))
        if step == 'response':
            rec = self.gen[(model, case['id'], repeat)]
        else:
            rec = self.judges[(case['id'], repeat)]
        return {'status': rec['status'], 'content': rec.get('content'), 'finish_reason': rec.get('finish_reason'),
                'cache_key': rec['cache_key'], 'returned_model': rec.get('returned_model')}


def test_faithful_program_judges_unusable_responses_like_official():
    """Official pipeline judges every response (also '')."""
    item = exp.items(exp.variant(), limit=1)[0]
    prog = exp.program(exp.variant('paper_subset330'))
    rec = {'status': 'truncated', 'content': 'partial', 'finish_reason': 'length', 'cache_key': 'k'}
    judge = {'status': 'ok', 'content': 'No', 'cache_key': 'j'}
    client = ReplayClient({('M', item['id'], 0): rec}, {(item['id'], 0): judge})
    out = prog.forward(item, LM(client, 'M', item['id'], 0))
    assert len(client.sent) == 2
    assert client.sent[1][2]['params'] == m.JUDGE_PARAMS and out['judge_response'] == 'No'


# ------------------------------------------------------------------------------------- released aggregates


def _csv_recall():
    import csv
    out = {}
    for r in csv.DictReader(open(m.REPO / 'analysis/abstention_performance.csv')):
        out.setdefault(r['model_name_formatted'], {})[r['dataset_name_formatted']] = float(r['recall'])
    return out


def test_judge_runs_on_its_own_model_via_for_model_and_failures_keep_response(tmp_path, monkeypatch):
    """The judge call goes through lm.for_model(JUDGE_MODEL) on the run's ledger (core API); a failed judge call keeps
    the response with an unknown label; an unroutable judge key fails before any generation call."""
    for cls in policies.POLICIES.values():
        monkeypatch.setattr(cls, 'pacing', {})
    policies.policy_for(m.JUDGE_MODEL)
    v = exp.variant()
    items = exp.items(v, limit=2)

    def responder2(payload):
        if payload['model'] == 'meta-llama/llama-3.1-8b-instruct':
            if items[1]['question'] in payload['messages'][0]['content']:
                raise TimeoutError('judge down')
            return 'Yes'
        return 'I cannot know that.'
    out = run(exp, v, ['anthropic/claude-sonnet-5.5'], tmp_path / 'run', ledger=tmp_path / 'l', ceiling_usd=100,
              transport=MockTransport(responder2), items=items, progress_every=0, workers=1)
    rows = {r['item_id']: r for r in out['rows']}
    ok, failed = rows[items[0]['id']], rows[items[1]['id']]
    assert [(t['step'], t['model']) for t in ok['trace']] == [('response', 'anthropic/claude-sonnet-5.5'),
                                                              ('abstention_judge', m.JUDGE_MODEL)]
    assert ok['score']['is_abstention'] is True
    assert failed['status'] == 'complete' and failed['output']['response'] == 'I cannot know that.'
    assert failed['score']['is_abstention'] is None and failed['score']['unknown_reason'] == 'judge request failed'
    bad = v.override('_bad', options={'judge_model': 'nonexistent/judge-x'})
    with pytest.raises(ValueError, match='accepted by no policy'):
        exp.program(bad)


# ------------------------------------------------------------------ robustness
from pathlib import Path  # noqa: E402

R2 = Path(__file__).resolve().parents[3] / 'experiments/run_minimal_setups/runs/earlier/runs'


@pytest.mark.parametrize('judge,response,finish,label,path', [
    ('Yes', 'I cannot know that.', 'stop', True, 'paper'),
    ('no.', 'Paris.', 'stop', False, 'paper'),
    ('**Yes**', 'I am not sure.', 'stop', True, 'robust'),                 # markdown around the label
    ('Yes, the model abstains because it asks for clarification.', 'Could you clarify?', 'stop', True, 'robust'),
    ('<think>it answered</think>\nNo', 'Paris.', 'stop', False, 'robust'),
    ('I cannot determine this.', 'Paris.', 'stop', None, None),           # judge gives no label: unknown
    ('Yes', '', 'length', None, None),                                    # empty truncated response: unknown
    ('No', 'The answer is', 'length', False, 'paper'),                    # non-empty truncated response: judged
])
def test_robust_labels(judge, response, finish, label, path):
    rp = exp.variant('robust_parse')
    item = exp.items(rp, limit=1)[0]
    out = {'response': response, 'response_status': 'ok', 'finish_reason': finish, 'judge_response': judge,
           'judge_status': 'ok'}
    s = exp.score(item, out, rp)
    assert (s['is_abstention'], s['parse_path']) == (label, path)
    paper = exp.score(item, out, exp.variant('paper_subset330'))
    if path == 'paper':
        assert s == {**paper, 'parse_path': 'paper'}


def test_robust_variants_params():
    base, rp, rb = (exp.variant(n) for n in ('paper_subset330', 'robust_parse', 'robust'))
    assert (rp.params, rp.prompts, rp.strategy) == (base.params, base.prompts, base.strategy) and not rp.faithful
    assert rp.options == {**base.options, 'parse': 'robust'}
    assert rb.params == {'temperature': 0.8, 'max_tokens': 4000} and rb.options['min_cap_reasoning'] == 16384
    assert rb.options['judge_model'] == m.JUDGE_MODEL and rb.options['judge_params'] == m.JUDGE_PARAMS
    assert exp.program(rb).case_extra == {'min_cap_reasoning': 16384} and exp.program(rp).case_extra == {}
    assert base.faithful and 'parse' not in base.options and base.params == m.SUBJECT_PARAMS


def test_robust_reliability():
    rp = exp.variant('robust_parse')
    items = exp.items(rp, limit=20)
    ok = {'response': 'x', 'response_status': 'ok', 'finish_reason': 'stop', 'judge_response': 'Yes', 'judge_status': 'ok'}
    cut = {**ok, 'response': '', 'finish_reason': 'length'}
    rows = [{'item': x, 'output': cut if i < 2 else ok, 'score': exp.score(x, cut if i < 2 else ok, rp)}
            for i, x in enumerate(items)]
    agg = exp.aggregate(rows, rp)
    assert agg['unparsed'] == 2 and agg['parsed_paper'] == 18 and agg['headline_reliable'] is True   # 10%
    rows[2] = {**rows[2], 'output': None, 'score': None}
    assert exp.aggregate(rows, rp)['headline_reliable'] is False


