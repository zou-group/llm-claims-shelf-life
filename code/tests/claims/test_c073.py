"""C073 MCQ selection bias: prompts, selection, permutation maps and scorers against the authors' code and released
gpt-3.5 outputs."""
import gzip
import json
import os

import pytest

from shelflife import registry
from shelflife.experiments import c073_mcq_selection_bias as m
from shelflife.replay import read_jsonl

exp = registry.get('C073')
ALL_CONDITIONS = ('original',) + m.MOVES   # the conditions of the registered variants


def _program(condition):
    return exp.program(exp.variant('paper_answer_moving'))


def _full_messages(condition, item, data, program):
    return program.preview(item)


# ------------------------------------------------------------------------------------------ prompts
def test_system_string_comes_from_prepare_eval():
    consts = m.official_string_constants()
    assert 'The following are multiple choice questions about {}.' in consts
    assert ' You should directly answer the question by choosing the correct option.' in consts
    assert m.SYSTEM.render(subject='x') == 'The following are multiple choice questions about x.' \
        ' You should directly answer the question by choosing the correct option.'


# ------------------------------------------------------------------------------- selection and maps


# ----------------------------------------------------------------------------------------- scorers
def _materialize(tmp_path, condition):
    """Write the compact archive back into the original directory layout (load_results never reads 'prompt')."""
    d = tmp_path / '0s_ichat' / m.ARCHIVE_DIRS[condition]
    d.mkdir(parents=True)
    by_subject = {}
    for rec in m.archive(condition):
        by_subject.setdefault(rec['subject'], []).append({'type': rec['type'], 'data': rec['data']})
    for s, recs in by_subject.items():
        (d / f'{s}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in recs))
    return str(d)


@pytest.mark.parametrize('text,pred', [('C.', 'C'), (' C.', ' '), ('**C**', '*'), ('Answer: C', 'A'),
                                       ('All of the above', 'A'), ('', None), ('c.', 'c')])
def test_official_first_character_rule(text, pred):
    """The authors' sampled[0] rule, including its failure modes on verbose outputs (documented, not fixed)."""
    item = m.make_item(*m.official_selection()[0], 'original')
    assert m.official_prediction(text, item) == pred


def test_prior_v3_parser_latent_case_insensitivity():
    """prior_v3 matches IDs case-insensitively. No saved output is affected (checked over all 2,440), but prose such
    as 'the answer is a function of' parses as A."""
    item = m.make_item(*m.official_selection()[0], 'original')
    s = exp.score(item, {'content': 'I think the answer is a function of x', 'status': 'ok'},
                  exp.variant('paper_answer_moving'), 'prior_v3')
    assert s['prediction'] == 'A'


# ------------------------------------------------------------------- robustness
from shelflife.paths import EXPERIMENTS  # noqa: E402
from shelflife.replay import run_records  # noqa: E402

MINIMAL_RUNS = EXPERIMENTS / 'run_minimal_setups/runs'
ABCD = list('ABCD')


@pytest.mark.parametrize('text,truncated,expected', [
    ('C', False, 'C'), ('C.', True, 'C'), ('(B)', False, 'B'), ('**C**', False, 'C'), ('Answer: C', False, 'C'),
    ('# Answer: **B. An op-code fetch cycle**\n\n## Explanation:\n\nIn the Intel', True, 'B'),   # M119 cap 32
    ('The correct answer is:\n\n**C. An op-code fetch cycle**\n\nExplanation: In the', True, 'C'),  # M045 cap 32
    ('The first machine cycle of an instruction is:\n\nD. An op-code fetch cycle', True, 'D'),   # M049 lead-in
    ('At the peak the greatest threat is likely: D. Inflation Inflation tends to', True, 'D'),
    ('The answer is \\boxed{B}.', False, 'B'), ('Final answer: (D)', False, 'D'),
    ('<think>A looks wrong, B maybe</think>\nC', False, 'C'),
    ('Let me consider option A and then', True, None),          # truncated mid-explanation: no guess
    ('The energy', True, None), ('## Step 1: Understand the context', True, None),
    ('A determination cannot be made from the options given.', False, None),   # article, not option A
    ('I cannot answer this question.', False, None),                           # refusal stays unparsed
    ('Either A or B could be right.', False, None),                             # ambiguous
    ('The fallacy is **D. False dilemma**. It presents only two options.', False, 'D'),
])
def test_robust_choice_modern_formats(text, truncated, expected):
    assert m.robust_choice(text, ABCD, truncated=truncated) == expected


def test_robust_choice_reads_repeated_option_text_only_on_complete_outputs():
    opts = ['6.8 eV', '12.2 eV', '6.0 eV', '13.6 eV']
    text = 'E = 54.4/9 = 6.04 eV. The final answer is $\\boxed{\\text{6.0 eV}}$.'   # M090 cap-32 run, complete
    assert m.robust_choice(text, ABCD, options=opts) == 'C'
    assert m.robust_choice(text, ABCD, truncated=True, options=opts) is None
    assert m.robust_choice('112', ABCD, options=['100', '112', '5', '7']) == 'B'


@pytest.mark.parametrize('text,expected', [('C', 'C'), ('C.', 'C'), ('C. An op-code', 'C'), ('A', 'A'), ('B)', 'B'),
                                           ('Answer: C', None), ('Based on', None), ('A determination', None),
                                           ('**C**', None), (' C', None), ('', None)])
def test_standalone_first_id(text, expected):
    assert m.standalone_first_id(text) == expected


def test_robust_variants_params_and_requests():
    sam, cap32 = exp.variant('shuffled_answer_moving'), exp.variant('shuffled_answer_moving_cap32')
    rob = exp.variant('robust')
    assert [x['id'] for x in exp.items(rob, 25)] == [x['id'] for x in exp.items(sam, 25)]
    assert cap32.params == {'temperature': 0.0, 'max_tokens': 32} and cap32.options['parser'] == 'prior_v3'
    assert rob.params == {'temperature': 0.0, 'max_tokens': 1024} and rob.options == {**sam.options, 'parse': 'robust'}
    assert rob.prompts == sam.prompts and rob.strategy == sam.strategy and not rob.faithful
    assert any('max_tokens 1024' in d for d in rob.deviations)
    prog_r, prog_s = exp.program(rob), exp.program(sam)
    it = exp.items(sam, 5)[3]
    assert prog_r.preview(it) == prog_s.preview(it)
    assert prog_r.params_for(it) == {'temperature': 0.0, 'max_tokens': 1024}
    # the paper variants keep the released scorer (first character, even for 'Answer: C')
    assert exp.score(it, {'content': 'Answer: C', 'status': 'ok'}, sam)['prediction'] == 'A'
    assert 'parse_path' not in exp.score(it, {'content': 'C', 'status': 'ok'}, sam)


def test_robust_aggregate_reports_parse_paths_and_tolerant_reliability():
    v = exp.variant('robust')
    items = exp.items(v, 50)
    texts = ['C', 'Answer: B', 'The energy']           # paper, robust, unparsed
    rows = []
    for i, it in enumerate(items):
        out = {'content': texts[i % 3] if i < 6 else it['expected'], 'status': 'truncated' if i % 3 == 2 else 'ok'}
        rows.append({'item': it, 'output': out, 'repeat': 0, 'score': exp.score(it, out, v)})
    rows[-1] = {**rows[-1], 'output': None, 'score': None}
    agg = exp.aggregate(rows, v)
    kept = [r for r in rows if not r['item']['excluded']]
    assert agg['parse_paths']['missing'] == 1 and agg['parse_paths']['unparsed'] == 2
    assert agg['parse_paths']['parsed_robust'] == 2
    assert sum(agg['parse_paths'].values()) == len(kept)
    assert agg['headline']['parser'] == 'official+robust'
    # every condition has 10 rows; one condition with 2 of 10 unparsed/missing -> unreliable under the 10% rule
    per = agg['conditions']
    assert agg['headline_reliable'] == all((c['unparsed'] + c['missing']) / c['n'] <= 0.10 for c in per.values())


def _saved_runs(run_name):
    root = MINIMAL_RUNS / run_name
    if not root.exists():
        pytest.skip(f'saved panel outputs not present: {root}')
    return sorted(d for d in root.iterdir() if d.is_dir() and (d / 'items.jsonl').exists())


@pytest.mark.parametrize('text,expected', [('Answer: A or B', None), ('Answer: B/D', None), ('Answer: A, B', None),
                                           ('Answer: (B)', 'B'), ('The answer is C, because A is wrong', 'C')])
def test_robust_choice_marker_ambiguity_audit(text, expected):
    from shelflife.experiments import c073_mcq_selection_bias as m
    assert m.robust_choice(text, 'ABCD') == expected
