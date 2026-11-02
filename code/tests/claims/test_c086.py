"""C086 Lost in the middle: prompts, data, scorer against the authors' code."""
import importlib.util
import json
import sys
import types

import pytest

from shelflife import registry
from shelflife.experiments import c086_lost_in_the_middle as m
from shelflife.paths import INITIAL
from shelflife.replay import read_jsonl

exp = registry.get('C086')
REPRO = INITIAL / 'reproductions/C086'


@pytest.fixture(scope='module')
def official_prompting():
    """The authors' prompting module imported as-is (pydantic Document.from_dict included)."""
    path = m.REPO / 'src/lost_in_the_middle/prompting.py'
    spec = importlib.util.spec_from_file_location('_litm_prompting', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['_litm_prompting'] = mod
    spec.loader.exec_module(mod)
    return mod


def _official_prompt(mod, item):
    if item['condition'] == 'closedbook':
        return mod.get_closedbook_qa_prompt(item['question'])
    src = (m.oracle_rows() if item['condition'] == 'oracle'
           else m.position_rows(item['document_count'], item['gold_position']))[item['source_index']]
    docs = [mod.Document.from_dict(ctx) for ctx in src['ctxs']]
    return mod.get_qa_prompt(item['question'], docs, mention_random_ordering=False, query_aware_contextualization=False)


@pytest.mark.parametrize('variant', ['paper_qa20'])
def test_every_prompt_byte_equal_to_official_get_qa_prompt(official_prompting, variant):
    v = exp.variant(variant)
    prog = exp.program(v)
    items = exp.items(v)
    assert len(items) == 2655 * (len(m.POSITIONS[v.options['document_count']]) + 2)
    for item in items:
        assert prog.build(item) == [{'role': 'user', 'content': _official_prompt(official_prompting, item)}], item['id']


def test_loaded_official_functions_equal_imported_module(official_prompting):
    v = exp.variant('paper_qa20_subset100')
    for item in exp.items(v, limit=14):
        if item['condition'] == 'closedbook':
            assert m.official()['get_closedbook_qa_prompt'](item['question']) == _official_prompt(official_prompting, item)
        else:
            docs = [types.SimpleNamespace(**d) for d in item['documents']]
            assert m.official()['get_qa_prompt'](item['question'], docs, False, False) == _official_prompt(official_prompting, item)


def test_released_data_gold_moves_distractors_fixed_zero_based_positions():
    """Full population (all 2,655) for 20 docs: same question/answers, gold passage = oracle passage at the 0-based
    index named by the file, distractor order unchanged, no answer-bearing distractor (as the paper describes)."""
    oracle = m.oracle_rows()
    ref = m.position_rows(20, 0)
    assert len(oracle) == len(ref) == 2655
    for p in m.POSITIONS[20]:
        rows = m.position_rows(20, p)
        for i, (x, r0, o) in enumerate(zip(rows, ref, oracle)):
            assert x['question'] == o['question'] and x['answers'] == o['answers'], (p, i)
            assert len(x['ctxs']) == 20 and x['ctxs'][p]['isgold'] and x['ctxs'][p] == o['ctxs'][0], (p, i)
            assert [d for d in x['ctxs'] if not d['isgold']] == [d for d in r0['ctxs'] if not d['isgold']], (p, i)
            assert not any(d.get('hasanswer') for d in x['ctxs'] if not d['isgold']), (p, i)


def test_item_gold_index_matches_document_number_in_prompt():
    v = exp.variant('paper_qa20_subset100')
    for item in exp.items(v, limit=70):
        if item['gold_position'] is None:
            assert item['documents'] == []
            continue
        gold = item['documents'][item['gold_position']]
        assert gold['isgold'] and sum(d['isgold'] for d in item['documents']) == 1
        text = exp.program(v).build(item)[0]['content']
        assert f'Document [{item["gold_position"] + 1}](Title: {gold["title"]}) {gold["text"]}' in text


def test_variant_params():
    assert exp.variant('paper_qa20').params == {'temperature': 0, 'max_tokens': 100}
    assert exp.variant('paper_qa20_subset100').faithful


# ------------------------------------------------------------------ scorer
@pytest.mark.parametrize('content,answers,correct', [
    ('Merry Clayton', ['Merry Clayton'], True),
    ('merry  clayton.', ['Merry Clayton'], True),                       # case/punct/whitespace normalised
    ('\nMerry Clayton', ['Merry Clayton'], False),                      # empty first line (official quirk)
    ('The singer is\nMerry Clayton', ['Merry Clayton'], False),          # answer only after first line
    ('**Answer:** Merry Clayton', ['Merry Clayton'], True),             # markdown is punctuation
    ('<think>Merry Clayton</think>\nLisa Fischer', ['Merry Clayton'], True),  # inline think text is first-line content
    ('Russia', ['US'], True),                                            # substring quirk kept (paper metric)
    ('', ['Merry Clayton'], False),
    ('anything', ['The'], True),                                         # article-only gold normalises to ''
])
def test_scorer_edge_cases_follow_official(content, answers, correct):
    item = {'answers': answers}
    assert exp.score(item, {'content': content, 'status': 'ok', 'finish_reason': 'stop'})['correct'] is correct
    official, _ = m.official()['get_metrics_for_example']({'answers': answers, 'model_answer': content})
    assert bool(official['best_subspan_em']) is correct


@pytest.mark.parametrize('output,unknown,correct', [
    ({'content': 'X', 'status': 'ok', 'finish_reason': 'stop'}, False, True),
    ({'content': '', 'status': 'ok', 'finish_reason': 'stop'}, True, False),          # blank visible answer
    ({'content': 'X', 'status': 'ok', 'finish_reason': 'error'}, False, True),       # visible correct answer stands
    ({'content': 'Y', 'status': 'ok', 'finish_reason': 'error'}, True, False),       # interrupted stream, no answer
    ({'content': 'X', 'status': 'ok', 'finish_reason': 'content_filter'}, True, True),
    ({'content': '', 'status': 'error', 'finish_reason': None}, True, False),
    ({'content': '', 'status': 'not_submitted', 'finish_reason': None}, True, False),
    # ledger-realistic truncations (ledger.py: finish length/max_tokens -> status 'truncated'): scored like the
    # official evaluator, which never sees the stop reason
    ({'content': 'X\nmore text cut', 'status': 'truncated', 'finish_reason': 'length'}, False, True),
    ({'content': 'X is the answer because', 'status': 'truncated', 'finish_reason': 'length'}, False, True),
    ({'content': 'Y\nX more text cut', 'status': 'truncated', 'finish_reason': 'length'}, False, False),
    ({'content': 'Y is the answer because', 'status': 'truncated', 'finish_reason': 'max_tokens'}, False, False),
    ({'content': '', 'status': 'truncated', 'finish_reason': 'length'}, True, False),  # reasoning ate the cap
])
def test_missing_output_policy(output, unknown, correct):
    s = exp.score({'answers': ['X']}, output)
    assert (s['unknown'], s['correct']) == (unknown, correct)
    official, _ = m.official()['get_metrics_for_example']({'answers': ['X'], 'model_answer': output['content']})
    assert bool(official['best_subspan_em']) is correct
    assert s['truncated'] is (output['status'] == 'truncated')
    assert s['first_line_complete'] is (output['status'] != 'truncated' or '\n' in output['content'])


def test_truncation_counts_reported_by_condition():
    v = exp.variant('paper_qa20_subset100')
    items = exp.items(v, limit=7)
    outs = {'qa20_gold0': ('A\ncut', 'truncated', 'length'), 'qa20_gold4': ('long first line cut', 'truncated', 'length')}
    rows = []
    for x in items:
        content, status, finish = outs.get(x['condition'], (x['answers'][0], 'ok', 'stop'))
        rows.append({'item': x, 'repeat': 0, 'score': exp.score(x, {'content': content, 'status': status,
                                                                   'finish_reason': finish}, v)})
    c = exp.aggregate(rows, v)['conditions']
    assert (c['qa20_gold0']['truncated'], c['qa20_gold0']['truncated_first_line_incomplete'], c['qa20_gold0']['unknown']) == (1, 0, 0)
    assert (c['qa20_gold4']['truncated'], c['qa20_gold4']['truncated_first_line_incomplete'], c['qa20_gold4']['unknown']) == (1, 1, 0)
    assert c['qa20_gold9']['truncated'] == 0 and c['qa20_gold9']['accuracy'] == 1.0


def test_aggregate_handles_failed_calls_as_unknown():
    v = exp.variant('paper_qa20_subset100')
    items = exp.items(v, limit=7)
    rows = [{'item': x, 'repeat': 0, 'score': None if x['condition'] == 'qa20_gold9' else
             exp.score(x, {'content': x['answers'][0], 'status': 'ok', 'finish_reason': 'stop'})} for x in items]
    out = exp.aggregate(rows, v)
    assert out['conditions']['qa20_gold9']['unknown'] == 1 and out['conditions']['qa20_gold0']['accuracy'] == 1.0
    assert out['contrasts']['first_minus_middle']['requested_bounds'] == [0.0, 1.0]
    assert out['contrasts']['first_minus_middle']['complete_case'] is None


# ------------------------------------------------------------------ robustness (parse='robust', 2026-10-10)
def test_robust_variants_params_and_faithfulness():
    rp, rb, base = exp.variant('robust_parse'), exp.variant('robust'), exp.variant('paper_qa20')
    assert rp.params == base.params and rp.prompts == base.prompts and rp.strategy == base.strategy
    assert rp.options['parse'] == 'robust' and not rp.faithful and rp.deviations
    assert rb.params == {'temperature': 0, 'max_tokens': 1024} and not rb.faithful
    assert rb.options['parse'] == 'robust' and rb.prompts == base.prompts
    assert exp.variant('paper_qa20').faithful and 'parse' not in base.options


ROB = exp.variant('robust_parse')
ANS = ['Wilhelm Conrad Röntgen']


@pytest.mark.parametrize('content,finish,correct,unknown,path', [
    ('Wilhelm Conrad Röntgen', 'stop', True, False, 'paper'),
    ('\n\nThe first prize went to **Wilhelm Conrad Röntgen** in 1901.', 'stop', True, False, 'robust'),   # M017/M144
    ('# First Nobel Prize in Physics\n\nBased on the documents, **Wilhelm Conrad Röntgen** won it.', 'length',
     True, False, 'robust'),                                                                              # M119
    ('**Answer:**\nWilhelm Conrad Röntgen', 'stop', True, False, 'robust'),
    ('Based on the provided documents:\n\n* Wilhelm Conrad Röntgen (1901)', 'stop', True, False, 'robust'),
    ('<think>Röntgen? check docs</think>\n\nWilhelm Conrad Röntgen', 'stop', True, False, 'robust'),
    ('\\boxed{Wilhelm Conrad Röntgen}', 'stop', True, False, 'paper'),
    ('Final answer: Wilhelm Conrad Röntgen', 'stop', True, False, 'paper'),
    ('Marie Curie won the first Nobel Prize in Physics.', 'stop', False, False, 'paper'),   # content first line: wrong
    ('# Title\n\nMarie Curie won it.', 'stop', False, False, 'robust'),                    # fallback line: wrong
    ('# First Nobel Prize\n\nThe prize went to Wilhelm Con', 'length', False, False, 'robust'),  # cut line scored as is
    ('# First Nobel Prize\n\n', 'length', False, True, None),                              # no answer-bearing line
    ('I cannot answer that from these documents.', 'stop', False, False, 'paper'),         # refusal stays wrong
    ('', 'length', False, True, None),                                                     # reasoning ate the cap
])
def test_robust_parse_modern_formats(content, finish, correct, unknown, path):
    status = 'truncated' if finish == 'length' else 'ok'
    s = exp.score({'answers': ANS}, {'content': content, 'status': status, 'finish_reason': finish}, ROB)
    assert (s['correct'], s['unknown'], s['parse_path']) == (correct, unknown, path)
    assert s['unparsed'] is (path is None)
    p = exp.score({'answers': ANS}, {'content': content, 'status': status, 'finish_reason': finish})
    if path == 'paper':      # the paper scorer's result stands wherever it parses
        assert (p['correct'], p['unknown']) == (s['correct'], s['unknown'])


def test_robust_aggregate_reports_parse_paths_and_tolerant_reliability():
    v = ROB
    items = exp.items(v, limit=7 * 20)
    rows = []
    for i, x in enumerate(items):
        if i < 10:
            out = None                                        # failed calls: missing
        elif i < 15:
            out = {'content': '', 'status': 'truncated', 'finish_reason': 'length'}
        elif i % 2:
            out = {'content': '# T\n\n' + x['answers'][0], 'status': 'ok', 'finish_reason': 'stop'}
        else:
            out = {'content': x['answers'][0], 'status': 'ok', 'finish_reason': 'stop'}
        rows.append({'item': x, 'repeat': 0, 'score': None if out is None else exp.score(x, out, v)})
    agg = exp.aggregate(rows, v)
    assert agg['parse'] == {'parsed_paper': 62, 'parsed_robust': 63, 'unparsed': 5, 'missing': 10}
    assert agg['headline_reliable'] is (15 / 140 <= 0.10)
    assert 'parse' not in exp.aggregate(rows, exp.variant('paper_qa20'))


SAMPLE_MODELS = ('M017', 'M045', 'M119', 'M144')   # blank first lines, long truncations, '# title' openings


