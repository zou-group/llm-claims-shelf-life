"""C054: prompts against the paper's Appendix A LaTeX, the multi-turn program, the reconstructed scorer, aggregates.

No code or outputs were released, so the prompt test re-parses the LaTeX listings (bold = user turns) independently of
the module, and the scorer is tested on the paper's own printed example responses plus constructed edge cases.
"""
import copy
import json
import os
import re

import pytest

from shelflife import registry
from shelflife.experiments import c054_self_correction as M
from shelflife.program import CallFailed, Completion

exp = registry.get('C054')
pytestmark = pytest.mark.skipif(exp.maturity == 'draft' and os.environ.get('SHELFLIFE_INCLUDE_DRAFTS') != '1',
                                reason='draft claim: run `python -m shelflife.cli check C054`')
EX = M.PAPER_SRC / 'src/examples'


def _unlatex(s):
    return (s.replace('$\\backslash$', '\\').replace('\\{', '{').replace('\\}', '}').replace('\\$', '$'))


def _bold_turns(path):
    return [_unlatex(m) for m in re.findall(r'^\|\\textbf\{(.*)\}\|$', path.read_text(), flags=re.M)]


def _model_turns(path):
    """Non-bold listing paragraphs (model responses), without the paper's (Correct)/(Incorrect) annotations."""
    body = path.read_text().split('\\begin{minted}', 1)[1].split('\\end{minted}', 1)[0].split('\n', 1)[1]
    paras = [p.strip() for p in body.split('\n\n') if p.strip() and not p.strip().startswith('|\\textbf')]
    return [re.sub(r'\s*\((Correct|Incorrect)\)$', '', p) for p in paras]


def _norm(s):
    return ' '.join(s.split())


class Stub:
    """Offline LM: returns scripted contents per step, records the messages it was sent."""
    def __init__(self, contents, fail_at=None):
        self.contents, self.fail_at, self.sent = contents, fail_at, []

    def __call__(self, messages, params=None, step='main'):
        self.sent.append((step, copy.deepcopy(messages), dict(params)))
        if step == self.fail_at:
            raise CallFailed(Completion(step=step, status='error', error='boom'))
        return Completion(step=step, status='ok', content=self.contents[step], finish_reason='stop')


def test_program_two_rounds_history_and_params():
    v = exp.variant('gpt35_review')
    item = exp.items(v, limit=1)[0]
    contents = {'initial': 'I think (C).', 'feedback1': 'Maybe (E).', 'revise1': '(E)', 'feedback2': 'ok',
                'revise2': 'Final (C)'}
    stub = Stub(contents)
    out = exp.program(v).forward(item, stub)
    assert out['complete'] and [s for s, _, _ in stub.sent] == ['initial', 'feedback1', 'revise1', 'feedback2', 'revise2']
    assert all(p == {'temperature': 1.0, 'max_tokens': 1024} for _, _, p in stub.sent)
    last = stub.sent[-1][1]
    assert [m['role'] for m in last] == ['user', 'assistant'] * 4 + ['user']
    assert last[2]['content'] == M.FEEDBACK['review'].render() and last[4]['content'] == M.CSQA_REVISE.render()
    assert last[6]['content'] == M.FEEDBACK['review'].render() and last[3]['content'] == 'Maybe (E).'
    s = exp.score(item, out, v)
    assert set(s['stages']) == {'initial', 'revise1', 'revise2'}


def test_program_failure_keeps_partial_trace():
    v = exp.variant('minimal')
    item = exp.items(v, limit=1)[0]
    stub = Stub({'initial': '(A)', 'feedback1': 'x'}, fail_at='revise1')
    out = exp.program(v).forward(item, stub)
    assert not out['complete'] and out['failed_step'] == 'revise1' and set(out['turns']) == {'initial', 'feedback1'}
    s = exp.score(item, out, v)
    assert set(s['stages']) == {'initial'} and not s['complete']


def test_items_and_variants():
    full = exp.variant('gpt35_review')
    items = exp.items(full)
    assert len(items) == 1221 + 1319 and len({x['id'] for x in items}) == len(items)
    assert [x['dataset'] for x in items[:4]] == ['csqa', 'gsm8k', 'csqa', 'gsm8k']
    mini = exp.variant(M.RUN_VARIANT)
    assert mini.faithful and mini.options['rounds'] == 1 and mini.options['datasets'] == ['csqa']
    m = exp.items(mini, limit=M.MINIMAL_LIMIT)
    assert len(m) == 70 and all(x['dataset'] == 'csqa' for x in m)
    assert [x['id'] for x in m] == [x['id'] for x in items if x['dataset'] == 'csqa'][:70]
    assert len(exp.items(mini)) == M.REPRO_LIMIT == 1221
    golds = {x['answer'] for x in items if x['dataset'] == 'csqa'}
    assert golds == set('ABCDE')


@pytest.mark.parametrize('text,gold,parsed,strict', [
    ('Therefore the answer is (E) puncture wound.', 'E', 'E', 'E'),
    ('(E) looks right but actually (D).', 'E', 'D', 'D'),              # no answer statement: last (X)
    ('**(E)**', 'E', 'E', 'E'),
    ('', 'E', None, None),
    ('I cannot answer that.', 'A', None, None),
    # audit M1 probes: the strict parser (pre-audit) gets each of these wrong
    ('My answer remains (E). I considered (A) injury but it is too general.', 'E', 'E', 'A'),
    ('Final Answer: (E) puncture wound\n\nNote: options (A) and (B) are less specific.', 'E', 'E', 'B'),
    ('Final answer: **E**', 'E', 'E', None),
    ('The answer is E.', 'E', 'E', None),
    ('Answer: (e)', 'E', 'E', None),
    ('The final answer is $\\boxed{E}$', 'E', 'E', None),
    ('(e)', 'E', 'E', None),
    # more anchoring cases
    ('I first thought (C), but the final answer is (E). (C) is too narrow.', 'E', 'E', 'C'),
    ('The answer is (B). On reflection, my final answer: **D**', 'D', 'D', 'B'),
    ('My final answer, as a single letter from A to E, is (C).', 'C', 'C', 'C'),
    ('Final answer (A-E): B.', 'B', 'B', None),
    ('The answer is A fencing thrust, i.e. (E).', 'E', 'E', 'E'),   # article 'A' is not a bare answer letter
    ('Reasoning...\n\n**E**', 'E', 'E', None),
    ('Reasoning...\n\nE', 'E', 'E', None),
    ('I stand by (E) puncture wound.', 'E', 'E', 'E'),
])
def test_csqa_parser(text, gold, parsed, strict):
    s = M.judge({'dataset': 'csqa', 'answer': gold}, text)
    assert (s['pred'], s['strict_pred']) == (parsed, strict), s
    assert s['correct'] is (parsed == gold)


@pytest.mark.parametrize('text,gold,correct', [
    ('\\boxed{18}', '18', True),
    ('$\\boxed{18.00}$', '18', True),
    ('\\boxed{1,250}', '1250', True),
    ('\\boxed{\\$18}', '18', True),
    ('first \\boxed{17} then \\boxed{18}', '18', True),
    ('\\boxed{18 \\text{ dollars}}', '18', True),
    ('The answer is 18.', '18', True),           # no \boxed{}: number after the answer statement
    ('So the final answer is $1,250.', '1250', True),
    ('I computed 17 apples and 18 pears.', '18', False),   # no boxed, no answer statement: unparsed
    ('\\boxed{x+1}', '18', False),
    ('\\boxed{', '18', False),
    ('', '18', False),
])
def test_gsm8k_scorer_edge_cases(text, gold, correct):
    assert M.judge({'dataset': 'gsm8k', 'answer': gold}, text)['correct'] is correct


def _chain_rows(spec, missing=0, revise_text=None, finish=None):
    """spec: list of (initial_correct, revised_correct, same_pred) for CSQA chains (minimal variant)."""
    rows = []
    for i, (a, b, same) in enumerate(spec):
        item = {'id': f'C054:csqa:{i}', 'dataset': 'csqa', 'answer': 'A'}
        pa = 'A' if a else 'B'
        pb = pa if same else ('A' if b else ('C' if pa == 'B' else 'B'))
        out = {'complete': True, 'turns': {'initial': f'({pa})', 'feedback1': 'f',
                                           'revise1': revise_text(i) if revise_text else f'({pb})'},
               'finish': finish(i) if finish else {}}
        rows.append({'item': item, 'output': out, 'model': 'm', 'repeat': 0, 'status': 'complete',
                     'score': exp.score(item, out, exp.variant('minimal'))})
    for j in range(missing):
        item = {'id': f'C054:csqa:m{j}', 'dataset': 'csqa', 'answer': 'A'}
        rows.append({'item': item, 'output': None, 'score': None, 'model': 'm', 'repeat': 0, 'status': 'call_error'})
    return rows


def test_aggregate_reproduces_table3_csqa_drop():
    """1,000 chains with initial 75.8% and round-1 38.1% correct -> drop 0.377 (Table 3 GPT-3.5 CSQA)."""
    spec = ([(True, True, True)] * 300 + [(True, False, False)] * 458 + [(False, True, False)] * 81
            + [(False, False, True)] * 100 + [(False, False, False)] * 61)
    v = exp.variant('minimal')
    agg = exp.aggregate(_chain_rows(spec, missing=4), v)
    p = agg['by_dataset']['csqa']['paired_initial_vs_revise1']
    assert p['n'] == 1000 and p['acc_initial'] == pytest.approx(0.758) and p['acc_revise1'] == pytest.approx(0.381)
    assert p['drop'] == pytest.approx(0.377) and p['drop_ci95'][0] < 0.377 < p['drop_ci95'][1]
    ch = agg['by_dataset']['csqa']['answer_changes_initial_to_final']
    assert ch == {'n': 1000, 'no_change': 400, 'correct_to_incorrect': 458, 'incorrect_to_correct': 81,
                  'incorrect_to_incorrect': 61}
    assert agg['rows_without_output'] == 4 and agg['by_dataset']['csqa']['n'] == 1004
    h = agg['shelflife']
    assert h['value'] == pytest.approx(0.377) and h['paper_value'] == pytest.approx(0.377)
    assert h['direction'] == 1 and h['null_value'] == 0 and h['n'] == 1000
    json.dumps(agg)


def test_no_auxiliary_models_and_call_count_targets():
    assert not exp.program(exp.variant()).auxiliary_models
    assert M._count_calls(exp, 'gpt35_review') == 5 and M._count_calls(exp, 'minimal') == 3


def test_reliability_flag_on_unparsed_asymmetry():
    """Audit M1/M2: 10% of revise1 turns unparsed (none at initial) -> headline_reliable False; 3% -> True."""
    spec = [(True, True, True)] * 100
    v = exp.variant('minimal')
    bad = exp.aggregate(_chain_rows(spec, revise_text=lambda i: 'I stand by my answer.' if i < 10 else '(A)'), v)
    pt = bad['by_dataset']['csqa']['per_turn']
    assert pt['initial']['unparsed_rate'] == 0 and pt['revise1']['unparsed_rate'] == 0.10
    assert bad['headline_reliable'] is False and bad['headline_caveats']
    ok = exp.aggregate(_chain_rows(spec, revise_text=lambda i: 'I stand by my answer.' if i < 3 else '(A)'), v)
    assert ok['headline_reliable'] is True


def test_reliability_flag_and_sensitivity_on_truncation():
    spec = [(True, True, True)] * 100
    v = exp.variant('minimal')
    agg = exp.aggregate(_chain_rows(spec, revise_text=lambda i: 'Upon review...' if i < 8 else '(A)',
                                    finish=lambda i: {'initial': 'stop', 'revise1': 'length' if i < 8 else 'stop'}), v)
    res = agg['by_dataset']['csqa']
    assert res['per_turn']['revise1']['truncated_rate'] == 0.08 and res['per_turn']['initial']['truncated_rate'] == 0
    assert agg['headline_reliable'] is False
    assert agg['shelflife']['value'] == pytest.approx(0.08)
    assert res['paired_initial_vs_revise1_excluding_truncated']['n'] == 92
    assert agg['shelflife']['secondary']['drop_excluding_truncated'] == 0


def test_strict_parser_drop_reported_as_secondary():
    spec = [(True, True, True)] * 20
    v = exp.variant('minimal')
    agg = exp.aggregate(_chain_rows(spec, revise_text=lambda i: 'Final answer: **A**'), v)
    sec = agg['shelflife']['secondary']
    assert agg['shelflife']['value'] == 0 and sec['drop_strict_parser'] == 1.0
    assert sec['paper_value_gpt4_same_protocol'] == pytest.approx(0.025)
    assert agg['by_dataset']['csqa']['per_turn']['revise1']['parser_disagreement_rate'] == 1.0


# ------------------------------------------------------------------ robustness
RP, RB, MIN = exp.variant('robust_parse'), exp.variant('robust'), exp.variant('minimal')
PANEL20 = ('M015 M017 M025 M045 M049 M066 M090 M102 M119 M143 M144 M171 M183 M190 M196 M199 M204 M206 '
           'M207').split()
CH = '(A) man post (B) attack (C) intimidation (D) prisoner (E) unprotected'


def _rj(text, gold='D', truncated=False):
    return M.judge({'dataset': 'csqa', 'answer': gold, 'choices': CH}, text, truncated, parse='robust')


@pytest.mark.parametrize('text,pred,path', [
    ('Reasoning ... so the answer is (D).', 'D', 'paper'),
    ('Final answer: **D**', 'D', 'paper'),
    ('\\boxed{D}', 'D', 'paper'),
    ('The guard watches the prisoner.\n\n(answer) D', 'D', 'robust'),           # echoed placeholder (M190, M171)
    ('Therefore, after re-evaluation:\n\n(answer) **D**', 'D', 'robust'),
    ('<think>maybe B</think>The guard is for the **prisoner**.\n\n(answer)', 'D', 'robust'),   # bold option text
    ('Among the choices, only “prisoner” fits.\n\n(answer)', 'D', 'robust'),         # one quoted option
    ('“attack” is too strong; “prisoner” fits.\n\n(answer)', None, None),  # two quoted: ambiguous
    ('A guard is a person who watches.', None, None),                             # bare article 'A' never guessed
    ("I can't help with that.", None, None),                                      # refusal stays unparsed
    ('', None, None),
])
def test_robust_csqa_modern_formats(text, pred, path):
    s = _rj(text)
    assert (s['pred'], s['parse_path'], s['unparsed']) == (pred, path, path is None), s


def test_robust_truncated_and_gsm8k_fallback():
    s = _rj('To decide, consider each option: (', truncated=True)
    assert s['unparsed'] and s['truncated']
    g = M.judge({'dataset': 'gsm8k', 'answer': '18'}, 'She makes **$18** every day.', parse='robust')
    assert g['parse_path'] == 'robust' and g['correct']
    g2 = M.judge({'dataset': 'gsm8k', 'answer': '18'}, 'so \\boxed{18}', parse='robust')
    assert g2['parse_path'] == 'paper' and g2['correct']
    # the paper path is untouched for paper variants
    assert 'parse_path' not in M.judge({'dataset': 'csqa', 'answer': 'D'}, '(answer) D')
    assert M.judge({'dataset': 'csqa', 'answer': 'D'}, '(answer) D')['pred'] is None


def _robust_rows(spec, revise_text):
    rows = _chain_rows(spec, revise_text=revise_text)
    for r in rows:
        r['score'] = exp.score(r['item'], r['output'], RP)
    return rows


def test_robust_aggregate_drops_unparsed_chains_and_tolerant_reliability():
    spec = [(True, True, True)] * 100
    rows = _robust_rows(spec, lambda i: 'I stand by my answer.' if i < 8 else '(A)')
    agg = exp.aggregate(rows, RP)
    paper = exp.aggregate(_chain_rows(spec, revise_text=lambda i: 'I stand by my answer.' if i < 8 else '(A)'), MIN)
    assert paper['shelflife']['value'] == pytest.approx(0.08) and paper['headline_reliable'] is False
    assert agg['shelflife']['value'] == 0 and agg['shelflife']['n'] == 92          # unparsed chains are missing
    assert agg['headline_reliable'] is True and (agg['parsed_paper'], agg['parsed_robust'], agg['unparsed']) == (192, 0, 8)
    assert agg['validity_rate'] == pytest.approx(192 / 200)
    bad = exp.aggregate(_robust_rows(spec, lambda i: 'I stand by my answer.' if i < 11 else '(A)'), RP)
    assert bad['headline_reliable'] is False
    rescued = exp.aggregate(_robust_rows(spec, lambda i: '(answer) A'), RP)
    assert rescued['shelflife']['value'] == 0 and rescued['parsed_robust'] == 100 and rescued['headline_reliable']


def test_robust_variant_params():
    assert RP.params == MIN.params and RP.prompts == MIN.prompts and RP.strategy == MIN.strategy
    assert {k: v for k, v in RP.options.items() if k != 'parse'} == MIN.options and RP.options['parse'] == 'robust'
    assert not RP.faithful and not RB.faithful and MIN.faithful
    assert RB.params == {'temperature': 1.0, 'max_tokens': 1024} and RB.options['min_cap_reasoning'] == 16384
    assert exp.program(RB).case_extra == {'min_cap_reasoning': 16384}
    assert getattr(exp.program(RP), 'case_extra', None) is None and getattr(exp.program(MIN), 'case_extra', None) is None


def _saved_runs():
    root = M.PROJECT / 'experiments/run_minimal_setups/runs/earlier/runs'
    return [d for tag in ('C054-minimal-L70-r4', 'C054-minimal-L70-r5') for m in PANEL20
            for d in [root / tag / m] if (d / 'responses.jsonl').exists()]


def test_truncated_turn_scored_under_robust_unparsed_under_robust_parse():
    """truncated='score' (robust, 1024 / 16384 caps): an answer turn cut off before any answer is wrong (paper rule)
    and its chain stays in the pair; robust_parse (truncated='missing'): it is unparsed and the chain is missing."""
    assert RB.options['truncated'] == 'score' and 'truncated' not in RP.options
    item = {'id': 'x', 'dataset': 'csqa', 'answer': 'D', 'choices': CH}
    for text in ('To decide, consider each option: (', ''):
        out = {'complete': True, 'turns': {'initial': '(D)', 'feedback1': 'f', 'revise1': text},
               'finish': {'revise1': 'length'}}
        s_rp, s_rb = exp.score(item, out, RP)['stages']['revise1'], exp.score(item, out, RB)['stages']['revise1']
        assert s_rp['unparsed'] and s_rp['parse_path'] is None
        assert not s_rb['unparsed'] and s_rb['correct'] is False and s_rb['truncated_scored']
    # a complete turn without an answer stays unparsed under robust
    assert exp.score(item, {'complete': True, 'turns': {'initial': '(D)', 'feedback1': 'f', 'revise1': ''},
                            'finish': {}}, RB)['stages']['revise1']['unparsed']
    spec = [(True, True, True)] * 100
    cut = lambda i: {'revise1': 'length'} if i < 20 else {}
    rows = _chain_rows(spec, revise_text=lambda i: 'Let me reconsider each option carefully' if i < 20 else '(A)',
                       finish=cut)
    rb_rows = [{**r, 'score': exp.score(r['item'], r['output'], RB)} for r in rows]
    rp_rows = [{**r, 'score': exp.score(r['item'], r['output'], RP)} for r in rows]
    agg = exp.aggregate(rb_rows, RB)
    assert agg['shelflife']['n'] == 100 and agg['shelflife']['value'] == pytest.approx(0.20)
    assert agg['truncated_scored'] == 20 and agg['unparsed'] == 0 and agg['headline_reliable'] is True
    rp_agg = exp.aggregate(rp_rows, RP)
    assert rp_agg['shelflife']['n'] == 80 and rp_agg['headline_reliable'] is False
