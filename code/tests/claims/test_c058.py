"""C058 counterfactual arithmetic: prompts, data, labels and scorer against the authors' code."""
import json
import random
from collections import Counter

import pytest

from shelflife import registry
from shelflife.experiments import c058_counterfactual as m
from shelflife.replay import read_jsonl
from shelflife.sourcecode import load_functions

exp = registry.get('C058')
BASES = (9, 10)
ARCHIVE_CELLS = [(model, task, base, cot) for model in m.ARCHIVE_MODELS for task, cots in
                 (('addition', (False, True)), ('ccc', (False,))) for base in BASES for cot in cots]


def digit_carry_sum(expr, base):
    """Independent positional addition (no int() conversion), uppercase digits."""
    a, b = expr.split('+')
    a, b = a.rjust(max(len(a), len(b)), '0'), b.rjust(max(len(a), len(b)), '0')
    out, carry = '', 0
    for x, y in zip(reversed(a), reversed(b)):
        s = m.DIGITS.index(x) + m.DIGITS.index(y) + carry
        out, carry = m.DIGITS[s % base] + out, s // base
    return ('1' if carry else '') + out


# ------------------------------------------------------------------ prompts
def test_prompts_byte_equal_to_official_templatize_all_rows():
    o = m.official()
    for task, cots in (('addition', (False, True)), ('ccc', (False,))):
        v = exp.variant('paper_ccc' if task == 'ccc' else 'paper_addition')
        v = v.override(f'{task}_all', options={'bases': BASES})
        items = exp.items(v)
        assert len(items) == len(BASES) * len(cots) * (200 if task == 'ccc' else 1000)
        prog = exp.program(v)
        for x in items:
            expected = o['templatize_ccc'](x['expr'], x['base']) if task == 'ccc' else \
                o['templatize'](x['expr'], x['base'], cot=x['cot'])
            assert prog.build(x) == [{'role': 'user', 'content': expected}], x['id']


def test_default_variant_params_and_cells():
    v = exp.variant()
    assert v.faithful and v.params == {'temperature': 0} and v.repeats == 1
    items = exp.items(v)
    assert Counter(x['condition'] for x in items) == {'base9_plain': 1000, 'base9_cot': 1000,
                                                       'base10_plain': 1000, 'base10_cot': 1000}


# --------------------------------------------------------------------- data


def test_labels_equal_independent_digit_carry_and_counterfactual_property():
    for base in BASES:
        for expr in m._data('addition', base):
            lab = m.label(expr, base)
            assert lab == digit_carry_sum(expr, base)
            if base != 10:   # rejection sampling: answer differs from the decimal reading (or uses letter digits)
                assert any(c.isalpha() for c in expr) or lab != str(sum(int(t) for t in expr.split('+')))
        for expr in m._data('ccc', base):
            assert m.label(expr, base, 'ccc') == digit_carry_sum(f'{expr}+1', base)


def test_duplicate_expression_counts():
    """The sampler draws with replacement: REPRO_LOG states 900 unique base-9 and 932 unique base-10 expressions."""
    assert len(set(m._data('addition', 9))) == 900 and len(set(m._data('addition', 10))) == 932


# ------------------------------------------------------------------ scoring
@pytest.mark.parametrize('text,task,expected', [
    ('So 76+14 in base-9 is 100. Therefore, the result is \\boxed{100}.', 'addition', ('100', 'parsed')),
    ('\\boxed{1a}', 'addition', ('1A', 'parsed')),                       # eval.py upper-cases the prediction
    ('', 'addition', (None, 'failed')),                                  # empty -> FAILED
    ('**Answer:** \\boxed{121}', 'addition', ('121', 'parsed')),          # markdown around the box
    ('I cannot help with that.', 'addition', (None, 'unparseable')),     # official parser asserts
    ('\\boxed{121_{\\text{(base-9)}}}', 'addition', (None, 'unparseable')),
    ('First \\boxed{12}\nso \\boxed{13}', 'addition', ('13', 'parsed')),   # conflicting boxes: last-line rule
    ('First \\boxed{12}, then \\boxed{13}', 'addition', (None, 'unparseable')),  # conflict within the last line
    ('The next number after 18 in base-9 is 20.', 'ccc', ('20', 'parsed')),
    ('no idea', 'ccc', (None, 'unparseable')),
])
def test_official_parse_edge_cases(text, task, expected):
    assert m.official_parse(text, task) == expected


def test_reasoning_model_content_only():
    """Only final content is scored; a reasoning trace with other boxes must not be passed in. A '<think>' block left in
    content is parsed as text: boxes inside it count as conflicting boxes and the last line decides."""
    item = exp.items(exp.variant(), limit=1)[0]
    assert exp.score(item, {'content': f'\\boxed{{{item["expected"]}}}', 'reasoning': '\\boxed{0}'}, exp.variant())['correct']
    text = f'<think>maybe \\boxed{{0}}</think>\nFinal: \\boxed{{{item["expected"]}}}'
    assert exp.score(item, {'content': text}, exp.variant())['correct']


@pytest.mark.parametrize('text,task,pred,status', [
    ('101 ', 'addition', '101', 'parsed'),          # authors strip trailing spaces off the TSV line (audit minor 1)
    ('101  ', 'addition', '101', 'parsed'),
    ('101\n', 'addition', '101\n', 'parsed'),       # escaped newline survives their strip: parsed, scored wrong
    ('101 ', 'ccc', None, None),
])
def test_official_parse_mirrors_authors_line_strip(text, task, pred, status):
    """eval.py: line.strip("\\n ") on the unicode_escape-d TSV line; eval_ccc.py: line.strip()."""
    official = m.official()
    escaped_line = 'expr\t' + official['tsv_escape'](text)
    stripped = escaped_line.strip('\n ' if task == 'addition' else None).split('\t')[1]
    fn = official['parse_output_ccc' if task == 'ccc' else 'parse_output']
    try:
        authors = fn(official['eval_unescape'](stripped))
        authors = authors.upper() if task == 'addition' else authors
    except Exception:
        authors = None
    got, st = m.official_parse(text, task)
    assert got == authors
    if pred is not None:
        assert (got, st) == (pred, status)


def test_official_parse_catches_any_exception(monkeypatch):
    """Audit minor 4: any raise from the authors' parser is 'unparseable', never a crash."""
    def boom(text):
        raise TypeError('NoneType group')
    monkeypatch.setitem(m.official(), 'parse_output', boom)
    assert m.official_parse('x') == (None, 'unparseable')


def test_paper_ccc_keeps_all_deviations():
    assert exp.variant('paper_ccc').deviations == exp.variant('paper_addition').deviations


def test_accuracy_bounds_include_missing_outputs():
    """Audit minor 5: bounds over all requested rows treat missing outputs (unrecovered API errors) as unknown."""
    v = exp.variant('paper_addition')
    items = [x for x in exp.items(v) if x['condition'] == 'base9_plain'][:4]
    rows = [{'item': items[0], 'score': exp.score(items[0], {'content': items[0]['expected']}, v)},
            {'item': items[1], 'score': exp.score(items[1], {'content': 'nonsense'}, v)},
            {'item': items[2], 'score': exp.score(items[2], {'content': ''}, v)},
            {'item': items[3], 'score': None}]
    cell = exp.aggregate(rows, v)['base9_plain']
    assert cell['missing_outputs'] == 1 and cell['n_rows'] == 4
    assert cell['accuracy_bounds_unparsed'] == [pytest.approx(1 / 3), pytest.approx(cell['accuracy_bounds_unparsed'][1])]
    lo, hi = cell['accuracy_bounds_all_rows']
    assert lo == pytest.approx(1 / 4) and hi == pytest.approx((1 + (4 - cell['parsed'])) / 4)


# ------------------------------------------------------------------ earlier pipeline subset variant
S50 = 'paper_addition_plain_s50'


def test_s50_variant_is_option_only_subset_of_paper_addition():
    """Faithful only because prompts, params and strategy equal paper_addition's; deviations inherited unchanged."""
    v, parent = exp.variant(S50), exp.variant('paper_addition')
    assert v.parent == 'paper_addition' and v.faithful
    assert (v.prompts, v.params, v.strategy, v.deviations, v.repeats) == (
        parent.prompts, parent.params, parent.strategy, parent.deviations, parent.repeats)
    assert v.options == {**parent.options, 'cots': (False,), 'subset': ('seeded', 200, 50), 'order': 'interleave_cells'}


def test_s50_items_are_50_seeded_plain_rows_per_base():
    """50 plain rows per base: the 50 lowest source indices of prepare.py's sorted(Random(20260929 + base).sample(
    range(1000), 200))."""
    items = exp.items(exp.variant(S50))
    assert len(items) == 100 and len({x['id'] for x in items}) == 100
    assert Counter(x['condition'] for x in items) == {'base9_plain': 50, 'base10_plain': 50}
    assert not any(x['cot'] for x in items)
    for base in (9, 10):
        idx = sorted(x['source_index'] for x in items if x['base'] == base)
        assert idx == sorted(random.Random(20260929 + base).sample(range(1000), 200))[:50]


def test_s50_order_keeps_limit_balanced():
    """Rows alternate base9/base10, so --limit 2k sends k rows per base (an odd limit gives base 9 one extra row);
    the blocked order of paper_addition would give base-9 rows only for any limit <= 1000."""
    v = exp.variant(S50)
    items = exp.items(v)
    assert [x['base'] for x in items] == [9, 10] * 50
    for k in (1, 5, 10, 25, 50):
        assert Counter(x['base'] for x in exp.items(v, 2 * k)) == {9: k, 10: k}
    assert exp.items(v, 20) == items[:20]
    assert {x['base'] for x in exp.items(exp.variant('paper_addition'), 100)} == {9}   # why the variant exists


def test_s50_requests_equal_parent_on_shared_items():
    """Every s50 request (messages + params) equals paper_addition's request for the same id."""
    ours = exp.cases(exp.variant(S50))
    parent = {c['id']: c for c in exp.cases(exp.variant('paper_addition'))}
    assert len(ours) == 100
    for c in ours:
        assert (c['messages'], c['params']) == (parent[c['id']]['messages'], parent[c['id']]['params']), c['id']
        assert c['params'] == {'temperature': 0}


def test_s50_aggregate_reports_plain_gap():
    v = exp.variant(S50)
    rows = []
    for it in exp.items(v):
        out = {'content': f'\\boxed{{{it["expected"] if it["base"] == 10 else "0"}}}'}
        rows.append({'item': it, 'output': out, 'repeat': 0, 'score': exp.score(it, out, v)})
    agg = exp.aggregate(rows, v)
    assert agg['base10_plain']['accuracy'] == 1.0 and agg['base9_plain']['accuracy'] == 0.0
    assert agg['gap_base10_minus_base9_plain'] == 1.0 and 'gap_base10_minus_base9_cot' not in agg


# ------------------------------------------------------------------ robustness
RP, RB = exp.variant('robust_parse'), exp.variant('robust')
PANEL20 = ('M015 M017 M025 M045 M049 M066 M090 M102 M119 M143 M144 M171 M183 M190 M196 M199 M204 M206 '
           'M207').split()


def _item(base=9, expr='36+71'):
    return {'task': 'addition', 'base': base, 'expected': m.label(expr, base), 'cot': False}


@pytest.mark.parametrize('text,base,pred,path', [
    ('\\boxed{117}', 9, '117', 'paper'),
    ('Thus the result is \\(\\boxed{117_9}\\).', 9, '117', 'paper'),
    ('**Final answer:** 117₉', 9, '117', 'robust'),               # no box: official FAILED/raises -> marker
    ('The answer is **117**.', 9, '117', 'robust'),
    ('<think>36+71 ...</think>\nFinal answer: 117', 9, '117', 'paper'),
    ('Answer: 119', 9, '119', 'paper'),                           # the official parser's decision stands
    ('The answer is **119**.', 9, None, None),                    # fallback: 9 is not a base-9 digit
    ('Let me add the units column: 6 + 1 = 7, then 3 + 7 = 10 which', 9, None, None),  # mid-reasoning, no marker
    ("I can't help with that.", 9, None, None),
    ('', 9, None, None),
])
def test_robust_parse_modern_formats(text, base, pred, path):
    it = _item(base)
    s = exp.score(it, {'content': text, 'finish_reason': 'stop'}, RP)
    assert (s['pred'] if s['parsed'] else None, s['parse_path']) == (pred, path), s
    assert s['unparsed'] == (path is None) and s['correct'] == (pred == it['expected'])


def test_robust_truncated_mid_reasoning_is_unparsed_and_missing_not_incorrect():
    v = RP
    items = exp.items(v)
    rows = []
    for k, it in enumerate(items):
        cut = it['base'] == 9 and k < 8                          # 4 base-9 rows cut off
        out = ({'content': 'Step 1: add the units column: 6 + 1 =', 'finish_reason': 'length'} if cut else
               {'content': f'\\boxed{{{it["expected"]}}}', 'finish_reason': 'stop'})
        rows.append({'item': it, 'output': out, 'repeat': 0, 'score': exp.score(it, out, v)})
    rows.append({'item': items[0], 'output': None, 'repeat': 1, 'score': None})   # a missing call
    paper = exp.aggregate([{**r, 'score': None if r['output'] is None else exp.score(r['item'], r['output'],
                                                                                         exp.variant(S50))}
                           for r in rows], exp.variant(S50))
    rob = exp.aggregate(rows, v)
    assert paper['base9_plain']['accuracy'] == pytest.approx(46 / 50) and 'headline_reliable' not in paper
    b9 = rob['base9_plain']
    assert b9['accuracy'] == 1.0 and b9['accuracy_paper_rule'] == pytest.approx(46 / 50)
    assert (b9['unparsed'], b9['truncated'], b9['n_rows'], b9['missing_outputs']) == (4, 4, 51, 1)
    assert rob['gap_base10_minus_base9_plain'] == 0 and rob['headline_reliable'] is True    # 5/51 <= 10%
    assert (rob['parsed_paper'], rob['parsed_robust'], rob['unparsed']) == (96, 0, 4)
    many = [dict(r, output={'content': '', 'finish_reason': 'length'}) if r['item']['base'] == 9 and i < 20 else r
            for i, r in enumerate(rows) if r['output'] is not None]
    many = [{**r, 'score': exp.score(r['item'], r['output'], v)} for r in many]
    assert exp.aggregate(many, v)['headline_reliable'] is False                  # 10/50 > 10%


def test_robust_variant_params():
    s50 = exp.variant(S50)
    assert (RP.params, RP.prompts, RP.strategy) == (s50.params, s50.prompts, s50.strategy)
    assert RP.options == {**s50.options, 'parse': 'robust'} and not RP.faithful and s50.faithful
    assert exp.cases(RP) == exp.cases(s50)
    assert RB.params == {'temperature': 0, 'max_tokens': 2048} and not RB.faithful
    assert RB.options['min_cap_reasoning'] == 16384 and RB.options['parse'] == 'robust'
    assert exp.program(RB).case_extra == {'min_cap_reasoning': 16384}
    assert exp.program(RP).case_extra == {} and exp.program(s50).case_extra == {}


def _saved_runs():
    from shelflife.paths import PROJECT
    root = PROJECT / 'experiments/run_minimal_setups/runs/earlier/runs'
    return [d for tag in ('C058-paper_addition_plain_s50-Lall-r4', 'C058-paper_addition_plain_s50-Lall-r5')
            for mod in PANEL20 for d in [root / tag / mod] if (d / 'responses.jsonl').exists()]


def test_truncated_row_scored_under_robust_unparsed_under_robust_parse():
    """truncated='score' (robust, 2048 / 16384 caps): a row cut off before any answer is incorrect (paper rule) and
    stays in the robust accuracy; robust_parse (truncated='missing'): unparsed and missing."""
    assert RB.options['truncated'] == 'score' and 'truncated' not in RP.options
    it = _item()
    for text in ('Step 1: add the units column: 6 + 1 =', ''):
        out = {'content': text, 'finish_reason': 'length'}
        s_rp, s_rb = exp.score(it, out, RP), exp.score(it, out, RB)
        assert s_rp['unparsed'] and s_rp['parse_path'] is None
        assert not s_rb['unparsed'] and s_rb['truncated_scored'] and s_rb['correct'] is False
    assert exp.score(it, {'content': '', 'finish_reason': 'stop'}, RB)['unparsed']   # complete, empty: unparsed
    items = exp.items(RB)
    rows = []
    for k, x in enumerate(items):
        cut = x['base'] == 9 and k < 8
        out = ({'content': 'Step 1: add the units column: 6 + 1 =', 'finish_reason': 'length'} if cut else
               {'content': f'\\boxed{{{x["expected"]}}}', 'finish_reason': 'stop'})
        rows.append({'item': x, 'output': out, 'repeat': 0})
    rob = exp.aggregate([{**r, 'score': exp.score(r['item'], r['output'], RB)} for r in rows], RB)
    assert rob['base9_plain']['accuracy'] == pytest.approx(46 / 50) and rob['base9_plain']['unparsed'] == 0
    assert rob['truncated_scored'] == 4 and rob['headline_reliable'] is True
    rp = exp.aggregate([{**r, 'score': exp.score(r['item'], r['output'], RP)} for r in rows], RP)
    assert rp['base9_plain']['accuracy'] == 1.0 and rp['base9_plain']['unparsed'] == 4
