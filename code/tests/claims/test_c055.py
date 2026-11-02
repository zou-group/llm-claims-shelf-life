"""C055 overthinking: prompts against the authors' input format and GSM8K, scorer against the authors' functions and
their released debug pipeline outputs, usage-token extraction, aggregation, Table 1 provenance from the LaTeX."""
import json
import math
import os
import re

import pytest

from shelflife import registry
from shelflife.experiments import c055_overthinking as m
from shelflife.ledger import LedgerClient, MockTransport
from shelflife.policies import policy_for
from shelflife.program import LM

exp = registry.get('C055')
pytestmark = pytest.mark.skipif(exp.maturity == 'draft' and os.environ.get('SHELFLIFE_INCLUDE_DRAFTS') != '1',
                                reason='draft claim: run `python -m shelflife.cli check C055`')


def _gsm8k():
    return [json.loads(line) for line in m.GSM8K.read_text().splitlines()]


def test_prompt_is_raw_question_for_every_item():
    v = exp.variant()
    items = exp.items(v)
    gsm = _gsm8k()
    assert len(items) == len(gsm) == 1319 and len({x['id'] for x in items}) == 1319
    prog = exp.program(v)
    for x in items:
        assert prog.build(x) == [{'role': 'user', 'content': gsm[x['index']]['question']}]
        assert x['expected_answer'] == gsm[x['index']]['answer'].split('####')[-1].strip().replace(',', '')
        assert re.fullmatch(r'-?\d+', x['expected_answer']), x['id']


def test_limit_is_a_seeded_uniform_sample_and_stable():
    v = exp.variant()
    first = exp.items(v, limit=m.MINIMAL_LIMIT)
    assert len(first) == 100 and first == exp.items(v, limit=100)
    assert [x['id'] for x in first] == [x['id'] for x in exp.items(v)[:100]]
    idx = sorted(x['index'] for x in first)
    assert idx != list(range(100)) and idx[-1] > 1000          # spread over the test set, not the head


def test_params_and_routability():
    v = exp.variant()
    assert v.params == {'temperature': 0, 'max_tokens': 16000} and v.repeats == 1
    for key in m.REPRO_MODELS:
        assert policy_for(key) is not None
    assert not exp.program(v).auxiliary_models
    assert exp.claim_set_key == 'C055' and m.RUN_VARIANT == exp.default_variant


@pytest.mark.parametrize('text,expected', [
    ('so the answer is \\boxed{18}.', ('18', 'boxed')),
    ('\\boxed{\\frac{1}{2}} and later \\boxed{18}', ('18', 'boxed')),       # authors take the last \boxed
    ('She makes $18 every day.', ('18', 'last_number')),
    ('Total: 1,234.', ('1,234', 'last_number')),
    ('I cannot solve this.', (None, 'none')),
    ('', (None, 'none')),
    ('\\boxed{18', ('18', 'last_number')),                                  # unbalanced box: authors return None
    ('The answer is 18.\n\nLet me know if step 2 is unclear.', ('18', 'answer_phrase')),   # audit probes
    ('Answer: 18 (from 3 x 6)', ('18', 'answer_phrase')),
    ('... so she earns 18 dollars.\n#### 18', ('18', 'answer_phrase')),
])
def test_extraction(text, expected):
    assert m.extract_answer(text) == expected


@pytest.mark.parametrize('pred,ref,ok', [
    ('18', '18', True), ('$18', '18', True), ('18.00', '18', True), ('1,000', '1000', True),
    ('18 dollars', '18', False), ('17', '18', False), ('50%', '50', True), ('0.5', '50', True),
    ('\\text{18}', '18', True), ('\\frac{36}{2}', '18', False),            # symbolic branch not reproduced
    (None, '18', False),
    ('\\$18', '18', True), ('x = 18', '18', True),                     # prefix tolerance
])
def test_judge_rule(pred, ref, ok):
    assert m.judge_correct(pred, ref) is ok


def test_usage_tokens_shapes():
    assert m.usage_tokens({'usage': {'prompt_tokens': 50, 'completion_tokens': 900,
                                     'completion_tokens_details': {'reasoning_tokens': 700}}}) == \
        {'output_tokens': 900, 'reasoning_tokens': 700, 'prompt_tokens': 50, 'reasoning_exceeds_output': False}
    anthropic = m.usage_tokens({'usage': {'input_tokens': 40, 'output_tokens': 263,
                                          'output_tokens_details': {'thinking_tokens': 180}}})
    assert anthropic['output_tokens'] == 263 and anthropic['reasoning_tokens'] == 180
    assert m.usage_tokens({'usage': {'input_tokens': 40, 'output_tokens': 300}})['reasoning_tokens'] is None
    assert m.usage_tokens({})['output_tokens'] is None
    assert m.usage_tokens({'usage': {'completion_tokens': None}})['output_tokens'] is None      # null stays missing
    r1 = m.usage_tokens({'usage': {'completion_tokens': 791, 'completion_tokens_details': {'reasoning_tokens': 1036}}})
    assert r1['reasoning_tokens'] == 791 and r1['reasoning_exceeds_output'] is True             # clipped and flagged


def test_program_reads_usage_from_ledger_raw_record(tmp_path):
    """Offline: MockTransport returns a response with usage; forward() must carry it into the output."""
    def responder(payload):
        return {'model': payload['model'], 'provider': 'mock',
                'choices': [{'message': {'content': 'The answer is \\boxed{18}.'}, 'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 60, 'completion_tokens': 1234, 'cost': 0.0,
                          'completion_tokens_details': {'reasoning_tokens': 1000}}}
    client = LedgerClient(tmp_path / 'run', tmp_path / 'ledger.sqlite3', 1.0, transport=MockTransport(responder))
    v = exp.variant()
    item = exp.items(v, limit=1)[0]
    lm = LM(client, 'deepseek/deepseek-r1', item['id'])
    out = exp.program(v).forward(item, lm)
    assert out['output_tokens'] == 1234 and out['reasoning_tokens'] == 1000 and out['status'] == 'ok'
    s = exp.score(dict(item, expected_answer='18'), out, v)
    assert s['correct'] and s['output_tokens'] == 1234 and not s['truncated']


def _client(tmp_path, response):
    return LedgerClient(tmp_path / 'run', tmp_path / 'ledger.sqlite3', 1.0, transport=MockTransport(lambda p: response))


@pytest.mark.parametrize('model,response,expected', [
    ('M206', {'model': 'claude-sonnet-5-5', 'content': [{'type': 'thinking', 'thinking': 'hmm'},
                                                       {'type': 'text', 'text': '\\boxed{18}'}],
              'stop_reason': 'end_turn', 'usage': {'input_tokens': 40, 'output_tokens': 263,
                                                   'output_tokens_details': {'thinking_tokens': 180}}},
     {'output_tokens': 263, 'reasoning_tokens': 180, 'status': 'ok'}),
    ('deepseek/deepseek-r1', {'model': 'r1', 'choices': [{'message': {'content': '\\boxed{18}'}, 'finish_reason': 'stop'}],
                              'usage': {'prompt_tokens': 50, 'completion_tokens': None}},
     {'output_tokens': None, 'status': 'ok'}),
    ('deepseek/deepseek-r1', {'model': 'r1', 'choices': [{'message': {'content': 'Let me think'}, 'finish_reason': 'length'}],
                              'usage': {'prompt_tokens': 50, 'completion_tokens': 16000}},
     {'output_tokens': 16000, 'status': 'truncated'}),
    ('deepseek/deepseek-r1', {'model': 'r1', 'choices': [{'message': {'content': '\\boxed{18}'}, 'finish_reason': 'stop'}]},
     {'output_tokens': None, 'status': 'ok'}),
])
def test_program_usage_edge_cases_through_ledger(tmp_path, model, response, expected):
    """Audit minor 7: Anthropic shape, null completion_tokens, truncation, no usage block."""
    v = exp.variant()
    item = exp.items(v, limit=1)[0]
    out = exp.program(v).forward(item, LM(_client(tmp_path, response), model, item['id']))
    for k, val in expected.items():
        assert out[k] == val, k
    s = exp.score(item, out, v)
    assert s['truncated'] == (expected['status'] == 'truncated')


def _rows(tokens, correct=True, model='M206', failed=0, reasoning=None):
    v = exp.variant()
    items = exp.items(v, limit=len(tokens) + failed)
    rows = []
    for i, (item, t) in enumerate(zip(items, tokens)):
        out = {'content': f"\\boxed{{{item['expected_answer'] if correct else 'x'}}}", 'status': 'ok',
               'finish_reason': 'stop', 'output_tokens': t,
               'reasoning_tokens': None if reasoning is None else reasoning[i]}
        rows.append({'item': item, 'output': out, 'score': exp.score(item, out, v), 'model': model, 'repeat': 0})
    for item in items[len(tokens):]:
        rows.append({'item': item, 'output': None, 'score': None, 'model': model, 'repeat': 0, 'status': 'call_error'})
    return rows


@pytest.fixture(autouse=True)
def _no_reference():
    exp.set_reference_rows(None)
    yield
    exp.set_reference_rows(None)


def test_shelflife_needs_reference_then_is_paired():
    rows = _rows([880.0] * 10)
    hl = exp.aggregate(rows, exp.variant())['shelflife']
    assert hl['value'] is None and hl['status'] == 'needs_reference_run'
    assert hl['log_ratio_vs_paper_llama_constant'] == pytest.approx(math.log(880 / 220.3))
    exp.set_reference_rows(_rows([220.0] * 10, model=m.REFERENCE_MODEL))
    hl = exp.aggregate(rows, exp.variant())['shelflife']
    assert hl['status'] == 'ok' and hl['n'] == 10 and hl['value'] == pytest.approx(math.log(4))
    assert hl['direction'] == 1 and hl['null_value'] == 0 and hl['paper_value'] == pytest.approx(math.log(1056.3 / 220.3))
    assert hl['accuracy_difference_vs_reference'] == 0 and hl['ratio_of_means_vs_live_reference'] == pytest.approx(4)
    assert set(hl) >= {'metric', 'value', 'ci', 'n', 'direction', 'null_value', 'paper_value', 'definition'}
    assert hl == exp.shelflife_from_runs(rows, _rows([220.0] * 10, model=m.REFERENCE_MODEL))


def test_reference_model_against_itself_is_null():
    ref = _rows([200, 300, 250], model=m.REFERENCE_MODEL)
    assert exp.shelflife_from_runs(ref, ref)['value'] == 0.0


def test_observed_reasoning_overrides_table_flag():
    """Audit major 1: a table 'reasoning' model that never reasons on its route is labelled from its own rows."""
    out = exp.aggregate(_rows([200] * 10, model='M090', reasoning=[0] * 10), exp.variant())
    assert out['reasoning_model_table'] is True and out['observed_reasoning'] == 'none'
    assert out['shelflife']['observed_reasoning'] == 'none' and out['reasoning_share'] == 0
    out = exp.aggregate(_rows([900] * 10, model='M207', reasoning=[0] * 7 + [600] * 3), exp.variant())
    assert out['observed_reasoning'] == 'occasional' and out['reasoning_share'] == pytest.approx(0.3)
    out = exp.aggregate(_rows([900] * 4, model='M206', reasoning=[500] * 4), exp.variant())
    assert out['observed_reasoning'] == 'reasoning' and out['mean_reasoning_tokens_when_reasoning'] == 500
    assert exp.aggregate(_rows([200] * 3, model='M015'), exp.variant())['observed_reasoning'] == 'unknown'


def test_missing_usage_and_failures_counted_not_imputed():
    rows = _rows([500, None, 700], failed=2)
    out = exp.aggregate(rows, exp.variant())
    assert out['with_usage'] == 2 and out['missing_usage'] == 1 and out['mean_output_tokens'] == 600
    assert out['shelflife']['missing_usage'] == 1 and out['shelflife']['failed_calls'] == 2
    assert out['accuracy_bounds_with_missing'] == [3 / 5, 1.0]


def test_paired_log_ratio_counts_zero_tokens():
    model, ref = _rows([880, 0]), _rows([220, 220])
    out = exp.paired_log_ratio(model, ref)
    assert out['n_pairs'] == 2 and out['zero_token_pairs'] == 1
    assert out['mean_log_ratio'] == pytest.approx((math.log(4) + math.log(1 / 220)) / 2)


def test_reference_rows_from_saved_run(tmp_path):
    """Runner-format run with the reference model: rows are rebuilt with usage from the saved output."""
    from shelflife import runner
    def responder(payload):
        return {'model': payload['model'], 'provider': 'mock',
                'choices': [{'message': {'content': 'The answer is \\boxed{1}.'}, 'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 60, 'completion_tokens': 222, 'cost': 0.0}}
    runner.run(exp, exp.variant(), [m.REFERENCE_MODEL], tmp_path / 'ref', ledger=tmp_path / 'l.sqlite3',
               ceiling_usd=1.0, limit=3, transport=MockTransport(responder))
    rows = exp.reference_rows_from_run(tmp_path / 'ref')
    assert len(rows) == 3 and all(r['score']['output_tokens'] == 222 for r in rows)


def test_empty_content_falls_back_to_reasoning_text():
    """Reproduction pass: Novita-served R1 returned content=None with the whole response (final answer included) in
    `reasoning` for 42/100 repro-r5 rows; the paper scores the whole response, so the answer must still be found."""
    v = exp.variants['gsm8k']
    item = {'expected_answer': '50'}
    out = {'content': None, 'reasoning': 'Thinking...\n\nGerald has 60 - 10 = 50 left.\n\n\\boxed{50}', 'output_tokens': 10}
    sc = exp.score(item, out, v)
    assert sc['correct'] and sc['answer_source'] == 'reasoning_fallback'
    sc2 = exp.score(item, {'content': '\\boxed{49}', 'reasoning': '\\boxed{50}', 'output_tokens': 5}, v)
    assert not sc2['correct'] and sc2['answer_source'] == 'content'
    assert exp.score(item, {'content': '', 'reasoning': '', 'output_tokens': 1}, v)['extraction'] == 'none'
    assert exp.variants['gsm8k'].faithful


# ------------------------------------------------------------------ robustness
RP, RB, G = exp.variant('robust_parse'), exp.variant('robust'), exp.variant('gsm8k')
PANEL20 = ('M015 M017 M025 M045 M049 M066 M090 M102 M119 M143 M144 M171 M183 M190 M196 M199 M204 M206 '
           'M207').split()
ITEM = {'expected_answer': '50'}


def _rs(content, finish='stop'):
    return exp.score(ITEM, {'content': content, 'finish_reason': finish, 'status': 'ok', 'output_tokens': 10}, RP)


@pytest.mark.parametrize('text,extracted,path,correct', [
    ('so the total is \\boxed{50}.', '50', 'paper', True),
    ('The answer is 50.', '50', 'paper', True),
    # bold final answer followed by a check with other numbers (the last-number stand-in reads 40)
    ('He is out $90 - $40 = **$50**.\n\nThis reads "sells them for 40" as $40 for all five CDs.', '50', 'robust', True),
    ('**3. Calculate the difference:**\n$60 - $10 = $50\n\n**Gerald had $50 left.**', '50', 'robust', True),
    ('Difference: $21 - 13 = \\mathbf{50}$ eggs.', '50', 'robust', True),
    ('**Answer:** In 4 weeks she spent **50 hours** in total.', '50', 'robust', True),   # bold number after marker
    ('<think>try 7</think>Total: 1,250 - 1,200 = 50', '50', 'robust', True),
    ('I cannot solve this.', None, None, False),
])
def test_robust_extraction_modern_formats(text, extracted, path, correct):
    s = _rs(text)
    assert (s['parse_path'], s['correct']) == (path, correct), s
    if path == 'robust':
        assert s['extracted_answer'] == extracted


def test_robust_truncated_without_paper_answer_is_unparsed():
    s = _rs('Step 1: 30 cars ... then 20 more, so', finish='length')
    assert s['unparsed'] and s['extracted_answer'] is None and not s['correct']
    assert exp.score(ITEM, {'content': 'Step 1: 30 cars ... then 20', 'finish_reason': 'length'}, G)['extraction'] == \
        'last_number'                      # paper variant unchanged
    assert 'parse_path' not in exp.score(ITEM, {'content': '**50**'}, G)


def test_robust_aggregate_counts_and_reliability():
    rows = _rows([500.0] * 20)
    for r in rows:
        r['score'] = exp.score(r['item'], r['output'], RP)
    rows[0]['score'] = exp.score(rows[0]['item'], {'content': 'no idea', 'output_tokens': 5}, RP)
    agg = exp.aggregate(rows, RP)
    assert (agg['parsed_paper'], agg['parsed_robust'], agg['unparsed']) == (19, 0, 1)
    assert agg['headline_reliable'] is True and agg['shelflife']['headline_reliable'] is True
    rows[1]['score'] = rows[2]['score'] = rows[0]['score']
    assert exp.aggregate(rows, RP)['headline_reliable'] is False             # 3/20 > 10%
    assert 'headline_reliable' not in exp.aggregate(_rows([500.0] * 20), G)  # paper aggregate unchanged


def _saved_runs():
    root = m.PROJECT / 'experiments/run_minimal_setups/runs/earlier/runs'
    return [d for tag in ('C055-gsm8k-L100-r4', 'C055-gsm8k-L100-r5', 'C055-gsm8k-L100-w2') for mm in PANEL20
            for d in [root / tag / mm] if (d / 'responses.jsonl').exists()]


def test_truncated_row_scored_under_robust_unparsed_under_robust_parse():
    """truncated='score' (robust, the paper's 16000 budget): a row cut off before a boxed / explicit answer is wrong
    (no answer within the budget), not unparsed; robust_parse (truncated='missing'): unparsed."""
    assert RB.options['truncated'] == 'score' and 'truncated' not in RP.options
    for text in ('Step 1: 30 cars ... then 20 more, so 50', ''):
        out = {'content': text, 'finish_reason': 'length', 'status': 'truncated', 'output_tokens': 16000}
        s_rp, s_rb = exp.score(ITEM, out, RP), exp.score(ITEM, out, RB)
        assert s_rp['unparsed'] and s_rp['parse_path'] is None
        assert not s_rb['unparsed'] and s_rb['truncated_scored'] and s_rb['correct'] is False
        assert s_rb['extracted_answer'] is None             # the mid-reasoning number is still not the answer
    # a boxed answer before the cut keeps its verdict in both
    boxed = {'content': 'so \\boxed{50}. Let me double-check', 'finish_reason': 'length', 'output_tokens': 9}
    assert exp.score(ITEM, boxed, RB)['correct'] and exp.score(ITEM, boxed, RP)['correct']
    rows = _rows([500.0] * 20)
    for r in rows[:5]:
        r['output'] = {**r['output'], 'content': 'Step 1: thinking', 'finish_reason': 'length'}
    rb = exp.aggregate([{**r, 'score': exp.score(r['item'], r['output'], RB)} for r in rows], RB)
    rp = exp.aggregate([{**r, 'score': exp.score(r['item'], r['output'], RP)} for r in rows], RP)
    assert rb['truncated_scored'] == 5 and rb['unparsed'] == 0 and rb['headline_reliable'] is True
    assert rb['accuracy'] == pytest.approx(15 / 20)
    assert rp['unparsed'] == 5 and rp['headline_reliable'] is False
