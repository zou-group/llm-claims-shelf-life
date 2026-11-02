"""C043 FairEval: prompts and parsers against the authors' code, and rescoring of earlier runs."""
import ast
import json

import pytest

from shelflife import registry
from shelflife.experiments import c043_faireval as m

exp = registry.get('C043')


def _fastchat_gen_prompt():
    return m.load_functions(m.FASTCHAT / 'eval_gpt_review.py', ['gen_prompt'])['gen_prompt']


def _jsonl(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _faireval_wrapper_from_ast():
    """FairEval.py gen_prompt's local prompt_template, read independently with ast."""
    tree = ast.parse((m.REPO / 'FairEval.py').read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'gen_prompt')
    node = next(n for n in ast.walk(fn) if isinstance(n, ast.Assign) and n.targets[0].id == 'prompt_template')
    return ast.literal_eval(node.value)


def test_faithful_default_is_faireval_wrapper_with_table1_rubric():
    """the default faithful variant is the authors' own FairEval.py wrapper ('Answer' markers) filled
    with the Table 1 rubric (FastChat prompt_id 1 defaults.prompt)."""
    assert exp.default_variant == 'faireval_vanilla'
    v = exp.variant()
    assert v.faithful
    wrapper = _faireval_wrapper_from_ast()
    rubric = next(r for r in _jsonl(m.FASTCHAT / 'prompt.jsonl') if r['prompt_id'] == 1)['defaults']['prompt']
    assert m.FAIREVAL_WRAPPER.text == wrapper and m.FAIREVAL_WRAPPER.verify_source()
    for item in exp.items(v):
        user = exp.program(v).build(item)[1]['content']
        assert user == wrapper.format(question=item['question'], answer_1=item['answer_1'], answer_2=item['answer_2'],
                                      prompt=rubric)
        assert "[The Start of Assistant 1's Answer]" in user and "'s response]" not in user


def test_items_cover_80_questions_both_orders_and_swap_candidates():
    items = exp.items(exp.variant())
    assert len(items) == 160
    by_pair = {}
    for x in items:
        by_pair.setdefault(x['pair_id'], []).append(x)
    for a, b in by_pair.values():
        assert (a['answer_1'], a['answer_2']) == (b['answer_2'], b['answer_1'])
        assert a['assistant_models'] == b['assistant_models'][::-1]


@pytest.mark.parametrize('text,expected,expected_ws', [
    ('8 7\nexplanation', [8.0, 7.0], [8.0, 7.0]), ('8,7\nx', [8.0, 7.0], [8.0, 7.0]), ('8.5 9', [8.5, 9.0], [8.5, 9.0]),
    ('8  7\nx', None, [8.0, 7.0]), ('Scores: 8 7', None, None), ('735 6\nfused digits', None, None), ('', None, None),
    # whitespace/markdown artefacts : FastChat rejects them, the diagnostic parser does not
    ('\n\n8 9\nexplanation', None, [8.0, 9.0]), (' 9 8', None, [9.0, 8.0]), ('8 9  \nexplanation', None, [8.0, 9.0]),
    ('8\t7\nx', None, [8.0, 7.0]), ('**8 7**\nx', None, [8.0, 7.0]), ('  \n', None, None), ('78\nx', None, None),
])
def test_fastchat_parser_with_range_check(text, expected, expected_ws):
    v = exp.variant()
    assert exp.parse(text, v) == expected
    assert exp.parse(text, v, 'fastchat_ws') == expected_ws
    item = exp.items(v, limit=1)[0]
    s = exp.score(item, {'content': text}, v)
    assert s['format_only_failure'] == (expected is None and expected_ws is not None)


# ------------------------------------------------------------------ robustness
@pytest.mark.parametrize('text,expected', [
    ('**Scores:** 5 4  \n\n**Evaluation:** ...', [5, 4]),          # markdown label (M196)
    ('Scores: 7 8\n\n**Assistant 1:** ...', [7, 8]),                # labelled line (M206)
    ('Score line:\n5 5\n\nBoth responses ...', [5, 5]),             # bare label, scores on the next line
    ('Score line: 6 7\n\nAssistant 1 gives ...', [6, 7]),
    ('Scores: 5 and 6.\n\nAssistant 1 ...', [5, 6]),
    ('Assistant 1 / Assistant 2 scores: 4 and 7\n\nx', [4, 7]),
    ('Assistant 1 scores 7 and Assistant 2 scores 3.\n\n**Assistant 1:** ...', [7, 3]),
    ("Assistant 1's score is 4 and Assistant 2's is 4, though for different reasons.", [4, 4]),
    ('<think>Scores could be 3 4 ...</think>\n8 7\nAssistant 1 ...', [8, 7]),   # think block removed
    ('<think>long hidden reasoning cut off by the cap', None),                     # truncated before the answer
    ('\\boxed{8} \\boxed{7}\nExplanation', [8, 7]),                                # LaTeX
    ('Assistant 1: 8\nAssistant 2: 7\nBoth are fine.', [8, 7]),
    ('Scores: 8/10 7/10', [8, 7]),
    ('67\n\nBoth assistants ...', None),                              # fused digits stay unparsed
    ('456 7 8\nAssistant 1 is accurate', None),
    ('893 7\n\nAssistant 1 gives', None),
    ('11 3\nx', None),                                                # out of range
    ('Assistant 1 is clearly better.\n\nAssistant 1 takes ...', None),  # prose verdict without scores
    ('Assistant 1 offers five indicators, but 3 are questionable.', None),
    ("I'm sorry, but I can't help with that.", None),                 # refusal
    ('', None),
])
def test_robust_score_line_modern_formats(text, expected):
    got = m.robust_score_line(text)
    assert got == (None if expected is None else [float(x) for x in expected])


@pytest.mark.parametrize('text,expected', [
    ('Assistant 1 has serious internal inconsistencies.\n\nAssistant 2 is better.\n\n2 4', [2, 4]),   # M206 form
    ('Assistant 1 offers five indicators.\n\n7 7.5', [7, 7.5]),
    ('Scores: Assistant 1 = 9, Assistant 2 = 5.\nreasons\n9 5', [9, 5]),
    ('456 7 8\nAssistant 1 is accurate.\n7 8', None),       # fused first-line score attempt blocks the fallback
    ('Assistant 1 is clearly better.\n\n11 2', None),        # out of range
    ('Assistant 1 is clearly better.', None),                 # single line
    ('Assistant 1 is better.\n\n1. Assistant 1 is correct.', None),
])
def test_robust_score_line_last(text, expected):
    assert m.robust_score_line(text) is None
    got = m.robust_score_line_last(text)
    assert got == (None if expected is None else [float(x) for x in expected])


def test_robust_score_line_last_is_counted_and_only_after_first_line_rules():
    v, pv = exp.variant('robust_parse'), exp.variant('faireval_vanilla')
    item = exp.items(v, limit=1)[0]
    s = exp.score(item, {'content': 'Assistant 1 is vague.\n\n3 8', 'status': 'ok'}, v)
    assert s['parse_path'] == 'robust' and s['score_line_last'] and s['scores'] == [3.0, 8.0]
    assert exp.score(item, {'content': 'Assistant 1 is vague.\n\n3 8', 'status': 'ok'}, pv)['verdict'] is None
    s = exp.score(item, {'content': '8 7\nreasons\n3 8', 'status': 'ok'}, v)     # first line wins
    assert s['parse_path'] == 'paper' and s['scores'] == [8.0, 7.0] and 'score_line_last' not in s
    rows = [{'item': item, 'output': {'content': 'x\n\n3 8'}, 'repeat': 0, 'status': 'complete',
             'score': exp.score(item, {'content': 'x\n\n3 8', 'status': 'ok'}, v)}]
    assert exp.aggregate(rows, v)['score_line_last'] == 1


def test_robust_variants_params_and_faithful_unchanged():
    base, rp, rb = exp.variant('faireval_vanilla'), exp.variant('robust_parse'), exp.variant('robust')
    assert base.faithful and base.params == {'temperature': 0, 'max_tokens': 512}
    assert base.options.get('parse', 'paper') == 'paper' and 'min_cap_reasoning' not in base.options
    assert rp.params == base.params and rp.strategy == base.strategy and rp.prompts == base.prompts
    assert rp.options['parse'] == 'robust' and not rp.faithful
    assert rb.params == {'temperature': 0, 'max_tokens': 1024} and rb.options['min_cap_reasoning'] == 16384
    assert rb.options['parse'] == 'robust' and rb.prompts == base.prompts and not rb.faithful
    item = exp.items(base, limit=1)[0]
    assert exp.program(base).case(item) == exp.program(rp).case(item)
    assert 'min_cap_reasoning' not in exp.program(base).case(item)
    req = exp.program(rb).case(item)
    assert req['min_cap_reasoning'] == 16384 and req['params']['max_tokens'] == 1024
    assert req['messages'] == exp.program(base).case(item)['messages']


def _panel20_source_runs():
    from shelflife.paths import PROJECT
    root = PROJECT / 'experiments/run_minimal_setups/analysis/rescored/C043'
    runs = {p.parent.name: __import__('pathlib').Path(json.loads(p.read_text())['source_run'])
            for p in sorted(root.glob('*/provenance.json'))}
    if not runs:
        pytest.skip('saved the panel C043 outputs (experiments/run_minimal_setups/analysis/rescored/C043) not present')
    return runs


def test_robust_aggregate_counts_and_tolerant_reliability():
    v = exp.variant('robust_parse')
    items = exp.items(v, limit=3)          # 3 pairs, 6 orders
    texts = ['8 7\nx', '7 8\nx',           # pair 1: paper parses both, vicuna wins both -> no conflict
             'Scores: 6 6\nx', '9 5\nx',   # pair 2: robust (tie) + paper (chatgpt wins) -> conflict
             'Assistant 1 is better.', None]  # pair 3: unparsed + failed call
    rows = []
    for item, text in zip(items, texts):
        out = None if text is None else {'content': text, 'status': 'ok'}
        rows.append({'item': item, 'output': out, 'repeat': 0, 'score': None if out is None else exp.score(item, out, v),
                     'status': 'complete' if out else 'call_error'})
    agg = exp.aggregate(rows, v)
    assert (agg['parsed_paper'], agg['parsed_robust'], agg['unparsed'], agg['missing']) == (3, 1, 1, 1)
    assert (agg['complete_pairs'], agg['conflicts'], agg['pairs']) == (2, 1, 3)
    assert agg['paper']['complete_pairs'] == 1 and agg['validity_rate'] == pytest.approx(4 / 6)
    assert agg['headline_reliable'] is False                        # 2/6 > 10%
    assert agg['whitespace_tolerant']['conflict_rate'] == agg['conflict_rate'] == 0.5
    # the paper variant on the same outputs keeps its own rule and keys
    pv = exp.variant('faireval_vanilla')
    prow = [{**r, 'score': None if r['output'] is None else exp.score(r['item'], r['output'], pv)} for r in rows]
    pagg = exp.aggregate(prow, pv)
    assert 'parsed_robust' not in pagg and pagg['complete_pairs'] == 1
