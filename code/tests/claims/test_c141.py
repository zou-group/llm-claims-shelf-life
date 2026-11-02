"""C141 Reversal Curse Exp 2: prompts and scorer against the authors' own code, released outputs, denominators."""
import csv
import json
import math
import os
import re
from dataclasses import dataclass

import pytest

from shelflife import registry
from shelflife.paths import PROJECT
from shelflife.sourcecode import load_functions
from shelflife.experiments import c141_reversal_curse as m

exp = registry.get('C141')
pytestmark = pytest.mark.skipif(exp.maturity == 'draft' and os.environ.get('SHELFLIFE_INCLUDE_DRAFTS') != '1',
                                reason='draft claim: run `python -m shelflife.cli check C141`')

PR = m.REPO / 'src/tasks/celebrity_relations/parent_reversals.py'
TP = m.REPO / 'scripts/celebrity_relations/test_parent_child_pairs.py'
OC = m.REPO / 'src/models/openai_chat.py'


class _NoCache:
    @staticmethod
    def cache(fn):
        return fn


def _authors(fake_batch=None):
    """A fresh, test-local load of the authors' definitions (independent of the module's cached copy).
    ``fake_batch(messages)`` stands in for the OpenAI calls of chat_batch_generate_multiple_messages."""
    chat = load_functions(OC, ['ChatMessage', 'chat_batch_generate_multiple_messages'])
    flatten = lambda lists: [x for sub in lists for x in sub]
    ns = load_functions(PR, ['ParentChildPair', 'parse_response', 'get_initial_messages', 'get_parent_query',
                             'get_child_query', 'get_child'], constants=('UNKNOWN_STR', 'SYSTEM_PROMPT', 'MODEL'),
                        extra_globals={'define': dataclass, 'ChatMessage': chat['ChatMessage'], 'flatten': flatten,
                                       'memory': _NoCache(),
                                       'chat_batch_generate_multiple_messages': fake_batch})
    tp = load_functions(TP, ['query_parent_test', 'query_child_test', 'get_prompts_completions'],
                        constants=('NUM_QUERIES_PER_CELEBRITY', 'FEW_SHOT_PROMPT'),
                        extra_globals={'ParentChildPair': ns['ParentChildPair'], 'pd': type('pd', (), {'DataFrame': object}),
                                       'get_parent_query': ns['get_parent_query'],
                                       'get_child_query': ns['get_child_query'],
                                       'chat_batch_generate_multiple_messages': fake_batch})
    return {**ns, **tp, 'default_parse': chat['chat_batch_generate_multiple_messages'].__defaults__[0]}


def _csv(path):
    with path.open(newline='') as fh:
        return list(csv.DictReader(fh))


# ------------------------------------------------------------------------------------------------ items
def test_items_cover_all_released_pairs_in_both_directions():
    pairs = _csv(m.PAIRS_CSV)
    assert len(pairs) == 1513
    items = exp.items(exp.variant())
    assert len(items) == 2 * 1513 and len({x['id'] for x in items}) == len(items)
    by_base = {}
    for x in items:
        by_base.setdefault(x['base_id'], []).append(x)
    assert len(by_base) == 1513
    for rows in by_base.values():
        assert [r['direction'] for r in rows] == ['parent_query', 'child_query']
        p = pairs[rows[0]['pair_index']]
        assert rows[0]['target'] == p['parent'] and rows[1]['target'] == p['child']
    # duplicated CSV rows stay separate pairs (as in every released result file)
    keys = [(p['child'], p['parent'], p['parent_type']) for p in pairs]
    assert len(set(keys)) == 1511


def test_limit_counts_pairs_and_is_a_seeded_shuffle_prefix():
    v = exp.variant()
    assert exp.items(v, limit=0) == []
    three = exp.items(v, limit=3)
    assert len(three) == 6 and len({x['base_id'] for x in three}) == 3
    assert [x['id'] for x in exp.items(v)[:6]] == [x['id'] for x in three]
    assert [x['pair_index'] for x in three[::2]] != [0, 1, 2]   # not the IMDB popularity order


# ---------------------------------------------------------------------------------------------- prompts
def test_chat_prompts_byte_equal_to_authors_builders_for_all_pairs():
    a = _authors()
    v = exp.variant()
    prog = exp.program(v)
    items = exp.items(v)
    assert items
    for item in items:
        if item['direction'] == 'parent_query':
            theirs = a['get_parent_query'](item['child'], item['parent_type'])
        else:
            theirs = a['get_child_query'](item['parent'])
        assert prog.preview(item) == [msg.to_dict() for msg in theirs], item['id']
    assert prog.params_for(items[0]) == {'temperature': 1}


def test_variant_protocols():
    both = exp.variant('chat_both')
    assert exp.default_variant == 'chat_both'
    assert both.faithful
    assert both.repeats == 10 and both.params == {'temperature': 1} and 'max_tokens' not in both.params


# ---------------------------------------------------------------------------------------------- scoring
@pytest.mark.parametrize('content', ['Maye Musk', '  Maye Musk  \n', 'a\n\nb\n  \nc', '', '\n\n', 'x\r\ny'])
def test_line_split_equals_authors_default_parse(content):
    assert m.split_lines(content) == _authors()['default_parse'](content)


def test_parse_response_is_authors():
    a = _authors()
    for line in ['Maye Musk', "I don't know.", 'Maye', 'Maye Musk, a model and dietitian', 'maye musk',
                 'A B C D E', 'A B C D E F', 'Denzel Washington Jr.']:
        assert m.parse_response(line) == a['parse_response'](line)


SAMPLE_SETS = [
    ['Maye Musk'] * 10,
    ['Maye Musk'] * 3 + ["I don't know."] * 7,
    ['Maye Musk\nErrol Musk'] + ['Errol Musk'] * 9,               # multi-line sample pools lines
    ['**Maye Musk**'] * 5 + ['The mother is Maye Musk.'] * 5,     # modern formats: strict rule misses all
    ['Maye Musk, a model and dietitian from South Africa'] + ['Kimbal Musk'] * 9,
    [''] * 9 + ['Maye Musk'],                                       # empty samples contribute no lines
    ['maye musk'] * 10,
]


@pytest.mark.parametrize('contents', SAMPLE_SETS)
def test_pair_metrics_equal_authors_functions(contents):
    """Our per-pair pooled mean equals the authors' query_parent_test/query_child_test arithmetic, and our best-of-10
    equals get_child's success, on the same 10 sampled contents (fed through their own line splitter)."""
    calls = []

    def fake_batch(messages, n, model=None):
        calls.append((len(messages), n))
        default_parse = _authors()['default_parse']
        return [line for c in contents for line in default_parse(c)]

    a = _authors(fake_batch)
    target = 'Maye Musk'
    v = exp.variant()
    for direction in m.DIRECTIONS:
        item = {'id': f'x:{direction}', 'base_id': 'x', 'direction': direction, 'child': target if direction ==
                'child_query' else 'Elon Musk', 'parent': target if direction == 'parent_query' else 'Errol Musk',
                'parent_type': 'mother', 'target': target}
        rows = [{'item': item, 'output': {'content': c}, 'score': exp.score(item, {'content': c}, v), 'repeat': i}
                for i, c in enumerate(contents)]
        pair = exp._pairs(rows, direction)['x']
        lines = [line for c in contents for line in m.split_lines(c)]
        if direction == 'parent_query':
            theirs = a['query_parent_test']('Elon Musk', 'mother', 'gpt-3.5-turbo', target) if lines else None
        else:
            theirs = a['query_child_test']('Errol Musk', 'gpt-3.5-turbo', target) if lines else None
            got = a['get_child']('Errol Musk', 'mother', target)
            assert pair['best'] == (got is not None)
        assert pair['mean'] == (theirs if lines else 0.0)
    assert calls and all(n == 10 for _, n in calls)


@pytest.mark.parametrize('content,strict,gpt4,lenient,fmt,idk,empty', [
    ('Maye Musk', True, True, True, False, False, False),
    ('Maye Musk.', True, True, True, False, False, False),
    ('maye musk', False, False, True, True, False, False),
    ('**Maye Musk**', False, False, True, True, False, False),
    ('The mother is Maye Musk.', False, False, True, True, False, False),
    ("Elon Musk's mother is Maye Musk.", False, False, True, True, False, False),
    ('Maye Musk, a model and dietitian from South Africa', True, False, True, False, False, False),
    ('Errol Musk\nMaye Musk', True, True, True, False, False, False),
    ("I don't know.", False, False, False, False, True, False),
    ('', False, False, False, False, False, True),
    ('Maye', False, False, False, False, False, False),
])
def test_modern_output_fixtures(content, strict, gpt4, lenient, fmt, idk, empty):
    """Literal expectations for the authors' strict prefix rule and the separately labelled lenient diagnostic."""
    v = exp.variant()
    item = {'direction': 'parent_query', 'target': 'Maye Musk'}
    s = exp.score(item, {'content': content}, v)
    assert (s['any_line_correct'], s['gpt4_rule_hit'], s['lenient_hit'], s['format_only_miss'], s['idk'],
            s['empty']) == (strict, gpt4, lenient, fmt, idk, empty)


def test_none_content_is_an_empty_sample():
    s = exp.score({'direction': 'child_query', 'target': 'Tom Cruise'}, {'content': None}, exp.variant())
    assert s['empty'] and s['n_lines'] == 0 and not s['any_line_correct']


# --------------------------------------------------------------------------------------- denominators
def _row(base, direction, repeat, content, target='Maye Musk', finish_reason='stop'):
    item = {'id': f'{base}:{direction}', 'base_id': base, 'direction': direction, 'target': target}
    if content is None:
        return {'item': item, 'output': None, 'score': None, 'repeat': repeat, 'status': 'call_error', 'model': 'x'}
    out = {'content': content, 'finish_reason': finish_reason,
           'status': 'truncated' if finish_reason == 'length' else ('ok' if content else 'empty')}
    return {'item': item, 'output': out, 'score': exp.score(item, out, exp.variant()), 'repeat': repeat,
            'status': 'complete', 'model': 'x'}


def test_failed_calls_and_paired_denominator():
    """Literal fixture, 3 pairs x 2 directions x 2 repeats (aggregate uses variant.repeats=10 only for the
    'fewer samples' count):
      a: fwd [hit, hit] rev [miss, hit]          -> fwd 1.0, rev 0.5, rev best True
      b: fwd [hit, failed] rev [failed, failed]  -> fwd 1.0, rev unavailable (unmatched pair)
      c: fwd [miss, '' ] rev ['**Maye Musk**', miss] -> fwd 0.0 (1 line), rev 0.0; one format-only miss
    """
    hit, miss = 'Maye Musk', 'Errol Musk'
    rows = [_row('a', 'parent_query', 0, hit), _row('a', 'parent_query', 1, hit),
            _row('a', 'child_query', 0, miss), _row('a', 'child_query', 1, hit),
            _row('b', 'parent_query', 0, hit), _row('b', 'parent_query', 1, None),
            _row('b', 'child_query', 0, None), _row('b', 'child_query', 1, None),
            _row('c', 'parent_query', 0, miss), _row('c', 'parent_query', 1, ''),
            _row('c', 'child_query', 0, '**Maye Musk**'), _row('c', 'child_query', 1, miss)]
    out = exp.aggregate(rows, exp.variant())
    assert (out['n'], out['scored'], out['failed_samples']) == (12, 9, 3)
    fwd, rev = out['directions']['parent_query'], out['directions']['child_query']
    assert fwd['mean_over_samples']['rate'] == pytest.approx(2 / 3)          # (1 + 1 + 0) / 3 pairs
    assert fwd['mean_over_samples']['available_pairs'] == 3 and fwd['failed_samples'] == 1
    assert rev['mean_over_samples']['pairs'] == 3 and rev['mean_over_samples']['available_pairs'] == 2
    assert rev['mean_over_samples']['rate'] == pytest.approx(0.25)
    assert rev['mean_over_samples']['bounds_with_missing'] == pytest.approx([0.5 / 3, 1.5 / 3])
    assert rev['best_of_k']['rate'] == 0.5 and rev['failed_samples'] == 2
    assert fwd['empty_samples'] == 1 and rev['format_only_miss_samples'] == 1
    assert rev['format_only_miss_share'] == pytest.approx(1 / 4)       # 4 available reverse samples
    gap = out['gap_mean_over_samples']
    assert (gap['pairs'], gap['matched_pairs']) == (3, 2)
    assert gap['gap'] == pytest.approx(((1 - 0.5) + (0 - 0)) / 2)
    cond = out['reverse_given_forward_correct_DIAGNOSTIC']
    assert cond['pairs'] == 1 and cond['reverse_mean_over_samples'] == 0.5 and cond['reverse_best_of_k'] == 1.0
    h = out['shelflife']
    assert h['metric'] == 'forward_minus_reverse_mean_over_samples' and h['n'] == 2 and h['missing'] == 1
    assert h['value'] == pytest.approx(0.25) and h['direction'] == 1
    assert out['headline_reliable'] is False and h['headline_reliable'] is False   # 1/4 format-only > 5%


def test_pair_without_lines_scores_zero_and_is_counted():
    rows = [_row('a', 'parent_query', 0, ''), _row('a', 'parent_query', 1, '  '),
            _row('a', 'child_query', 0, 'Maye Musk'), _row('a', 'child_query', 1, 'Maye Musk')]
    out = exp.aggregate(rows, exp.variant())
    assert out['directions']['parent_query']['pairs_without_lines'] == 1
    assert out['directions']['parent_query']['mean_over_samples']['rate'] == 0.0
    assert out['gap_mean_over_samples']['gap'] == -1.0


# ------------------------------------------------------------------------------------------ prior runs
RUNS = PROJECT / 'experiments/run_minimal_setups/runs/added-b/runs'
PILOT_RUNS = [('C141-chat_both-smoke-L2', 'chat_both', 'openai-direct/gpt-3.5-turbo-0125'),
              ('C141-chat_both-L200-gpt35-0125', 'chat_both', 'openai-direct/gpt-3.5-turbo-0125')]


# ------------------------------------------------------------------------- post-verification (audit M1, m1-m7)
def test_empty_samples_flagged_and_empty_as_wrong_sensitivity():
    """Audit M1: 9 empty samples + 1 hit pools to 1.0 under the authors' rule (empty samples add no lines); the
    empty-as-wrong sensitivity gives 1/10, and the 90% empty share makes the headline unreliable."""
    rows = [_row('a', 'parent_query', i, '') for i in range(9)] + [_row('a', 'parent_query', 9, 'Maye Musk')]
    rows += [_row('a', 'child_query', i, 'Errol Musk') for i in range(10)]
    out = exp.aggregate(rows, exp.variant())
    fwd = out['directions']['parent_query']
    assert fwd['mean_over_samples']['rate'] == 1.0
    assert fwd['mean_over_samples_empty_as_wrong_SENSITIVITY']['rate'] == pytest.approx(0.1)
    assert (fwd['empty_samples'], fwd['empty_share']) == (9, 0.9)
    assert out['headline_reliable'] is False and 'parent_query.empty_share=0.9' in out['headline_caveat']
    assert out['shelflife']['value'] == 1.0
    sens = out['shelflife_sensitivity_empty_as_wrong']
    assert sens['role'] == 'sensitivity' and sens['value'] == pytest.approx(0.1) and sens['n'] == 1


def test_truncated_samples_flagged():
    rows = [_row('a', 'parent_query', 0, 'Maye Musk', finish_reason='length'),
            _row('a', 'parent_query', 1, 'Maye Musk'), _row('a', 'child_query', 0, 'Errol Musk'),
            _row('a', 'child_query', 1, 'Errol Musk')]
    out = exp.aggregate(rows, exp.variant())
    assert out['directions']['parent_query']['truncated_samples'] == 1
    assert out['directions']['parent_query']['truncated_share'] == 0.5
    assert out['headline_reliable'] is False and 'truncated_share' in out['headline_caveat']


def test_clean_outputs_are_reliable():
    rows = []
    for b in ('a', 'b'):
        rows += [_row(b, 'parent_query', i, 'Maye Musk') for i in range(10)]
        rows += [_row(b, 'child_query', i, 'Errol Musk' if i else 'Maye Musk') for i in range(10)]
    out = exp.aggregate(rows, exp.variant())
    assert out['headline_reliable'] is True and 'headline_caveat' not in out
    assert out['shelflife']['value'] == pytest.approx(0.9) and out['shelflife']['headline_reliable'] is True
    assert out['shelflife']['paper_value'] == pytest.approx(0.22928, abs=1e-5)


def test_unattempted_pairs_count_as_missing():
    """Audit m7: a pair with forward rows but no reverse row, and an expected pair with no row at all, stay in the
    denominators (missing, bounds) instead of disappearing."""
    rows = [_row('a', 'parent_query', 0, 'Maye Musk'), _row('a', 'child_query', 0, 'Maye Musk'),
            _row('b', 'parent_query', 0, 'Maye Musk')]
    out = exp.aggregate(rows, exp.variant(), expected_base_ids=['a', 'b', 'c'])
    rev = out['directions']['child_query']
    assert (rev['mean_over_samples']['pairs'], rev['mean_over_samples']['available_pairs']) == (3, 1)
    assert rev['pairs_without_rows'] == 2
    assert rev['mean_over_samples']['bounds_with_missing'] == pytest.approx([1 / 3, 1.0])
    assert out['directions']['parent_query']['pairs_without_rows'] == 1
    assert (out['shelflife']['n'], out['shelflife']['missing']) == (1, 2)
    no_expected = exp.aggregate(rows, exp.variant())
    assert no_expected['directions']['child_query']['mean_over_samples']['pairs'] == 2


def test_lenient_diagnostic_on_per_sample_unit():
    """Audit m6: a multi-line list of children lowers the pooled line share but not the per-sample shares."""
    rows = [_row('a', 'child_query', i, 'Errol Musk\nMaye Musk\nKimbal Musk') for i in range(2)]
    out = exp.aggregate(rows, exp.variant('chat_both'))['directions']['child_query']
    assert out['mean_over_samples']['rate'] == pytest.approx(1 / 3)
    assert out['strict_per_sample_share']['rate'] == 1.0
    assert out['lenient_per_sample_share_DIAGNOSTIC']['rate'] == 1.0
    assert out['format_only_miss_samples'] == 0


def test_idk_accepts_curly_apostrophe():
    v = exp.variant()
    item = {'direction': 'parent_query', 'target': 'Maye Musk'}
    assert exp.score(item, {'content': 'I don\u2019t know.'}, v)['idk'] is True
    assert exp.score(item, {'content': "I don't know."}, v)['idk'] is True


def _pdf_bar_heights():
    """Independent extraction of the bar rectangles from the figure's vector content stream (closed m/l/l/l paths
    with the axis baseline y=55.692) and the y-tick labels' spacing."""
    import re
    import zlib
    data = (m.PAPER_SRC / 'figures/Experiment_2_figure_1.pdf').read_bytes()
    for mm in re.finditer(rb'stream\r?\n(.*?)\r?\nendstream', data, re.S):
        try:
            stream = zlib.decompress(mm.group(1))
        except zlib.error:
            continue
        if b'(gpt-3.5-turbo)' not in stream:
            continue
        ticks = [float(y) for y in re.findall(rb'1 0 -0 1 [\d.]+ ([\d.]+) cm\nBT\n/F1 13.2 Tf\n0 0 Td\n\[ \(\d+\) \]', stream)]
        per_pct = (ticks[-1] - ticks[0]) / (5 * (len(ticks) - 1))
        bars = re.findall(rb'([\d.]+) 55.692 m\n([\d.]+) 55.692 l\n[\d.]+ ([\d.]+) l\n[\d.]+ [\d.]+ l', stream)
        # bars are ~33 pt wide; the 388-pt-wide axes background shares the baseline and is excluded
        return [(float(x0), (float(top) - 55.692) / per_pct / 100) for x0, x1, top in bars
                if float(x1) - float(x0) < 50]
    raise AssertionError('figure stream not found')


# ------------------------------------------------------------------- robustness variants
from shelflife.experiments import _robust  # noqa: E402


def _rscore(content, direction='parent_query', target='Maye Musk', variant='robust_parse', finish_reason='stop'):
    out = {'content': content, 'finish_reason': finish_reason,
           'status': 'truncated' if finish_reason == 'length' else 'ok'}
    return exp.score({'direction': direction, 'target': target}, out, exp.variant(variant))


def test_robust_variant_params():
    both, rp, rb = exp.variant('chat_both'), exp.variant('robust_parse'), exp.variant('robust')
    assert not rp.faithful and not rb.faithful and rp.parent == 'chat_both' and rb.parent == 'robust_parse'
    # robust_parse: identical requests to chat_both, so saved pilot runs rescore offline
    assert (rp.params, rp.repeats, rp.strategy, rp.prompts) == (both.params, both.repeats, both.strategy, both.prompts)
    assert rp.options == {**both.options, 'parse': 'robust'} and _robust.truncation_missing(rp)
    assert exp.program(rp).case_extra == {} == exp.program(both).case_extra
    # robust: minimal design (1 sample), explicit 1024 cap floor, reasoning cap 16384, truncated rows scored
    assert rb.repeats == 1 and rb.params == {'temperature': 1, 'max_tokens': 1024}
    assert rb.options == {**rp.options, 'min_cap_reasoning': 16384, 'truncated': 'score'}
    assert not _robust.truncation_missing(rb)
    assert exp.program(rb).case_extra == {'min_cap_reasoning': 16384}
    assert rb.prompts == both.prompts and rb.strategy == both.strategy and len(rb.deviations) == len(rp.deviations) + 3
    items = exp.items(rb, limit=3)
    assert [exp.program(rb).preview(x) for x in items] == [exp.program(both).preview(x) for x in exp.items(both, 3)]
    assert len(exp.items(rb, limit=600)) == 1200


@pytest.mark.parametrize('content,path,correct,lines', [
    ('Maye Musk', 'paper', 1, 1),
    ("I don't know.", 'paper', 0, 1),
    ('Errol Musk', 'paper', 0, 1),
    ('**Maye Musk**', 'robust', 1, 1),
    ('  "Maye Musk"', 'robust', 1, 1),
    ('Answer: Maye Musk', 'robust', 1, 1),
    ('Final answer: **Maye Musk** (born 1948).', 'robust', 1, 1),
    ("Elon Musk's mother is Maye Musk.", 'robust', 1, 1),
    ("Elon Musk’s mother's name is Maye Musk.", 'robust', 1, 1),
    ('The mother of Elon Musk is Maye Musk.', 'robust', 1, 1),
    ('Her name is Maye Musk.', 'robust', 1, 1),
    ("Elon Musk's father is Errol Musk.", 'robust', 0, 1),
    ('<think>Elon Musk... his mother, maybe\nMaye Musk</think>\n\n**Maye Musk**', 'robust', 1, 1),
    ('- **Maye Musk**\n- Errol Musk', 'robust', 1, 2),
    # never widened: case, nicknames, prose hedges and lowercase stay misses
    ('maye musk', None, 0, 1),
    ('**maye musk**', None, 0, 1),
    ('I believe it is Maye Musk.', None, 0, 1),
    ('Maye "May" Musk', None, 0, 1),
    ('', None, 0, 0),
    # refusals stay wrong
    ("I'm not sure who Elon Musk's mother is.", 'refusal', 0, 1),
    ('Unknown', 'refusal', 0, 1),
    ("Sorry, I don’t know.", 'refusal', 0, 1),
    ("Elon Musk's mother is not publicly known.", 'refusal', 0, 1),
])
def test_robust_parse_formats(content, path, correct, lines):
    s = _rscore(content)
    assert (s['parse_path'], s['n_correct_lines'], s['n_lines']) == (path, correct, lines), s
    assert s['any_line_correct'] == bool(correct)


@pytest.mark.parametrize('content,path,correct,lines', [
    ('Tom Cruise', 'paper', 1, 1),
    ("Mary Lee Pfeiffer's children include Tom Cruise and Lee Anne De Vette.", 'robust', 1, 1),
    ('Mary Lee Pfeiffer is the mother of Tom Cruise.', 'robust', 1, 1),
    ('A child of Mary Lee Pfeiffer is Tom Cruise.', 'robust', 1, 1),
    ('Mary Lee Pfeiffer has a son named Tom Cruise.', 'robust', 1, 1),
    ('One of her sons is **Tom Cruise**.', 'robust', 1, 1),
    # audit m1: the lead-in line is pooled like the authors' rule pools it (1/3, as the unbulleted list)
    ('Mary Lee Pfeiffer has four children:\n1. Lee Anne De Vette\n2. **Tom Cruise**', 'robust', 1, 3),
    # the parent named in a reverse answer is never read as the child
    ('Mary Lee Pfeiffer is the mother of Lee Anne De Vette.', 'robust', 0, 1),
    ('Mary Lee Pfeiffer is best known as the mother of actor Tom Cruise.', None, 0, 1),   # field opens 'actor'
])
def test_robust_parse_reverse_formats(content, path, correct, lines):
    s = _rscore(content, direction='child_query', target='Tom Cruise')
    assert (s['parse_path'], s['n_correct_lines'], s['n_lines']) == (path, correct, lines), s


def test_robust_target_match_stays_case_sensitive_prefix():
    """The fallback field is matched with the authors' startswith: a truncated or reordered name is a miss."""
    for content in ["Elon Musk's mother is Maye.", "Elon Musk's mother is Musk, Maye.", 'Answer: MAYE MUSK']:
        assert _rscore(content)['n_correct_lines'] == 0, content
    s = _rscore("Denzel Washington's father is Denzel Hayes Washington Sr.", target='Denzel Hayes Washington, Sr.')
    assert s['parse_path'] == 'robust' and s['n_correct_lines'] == 0


def test_robust_think_block_exception():
    """Documented exception: tagged reasoning lines are removed before the authors' rule (paper pools them)."""
    content = '<think>\nWho is her mother\nMaye Musk I think\n</think>\nMaye Musk'
    paper = exp.score({'direction': 'parent_query', 'target': 'Maye Musk'}, {'content': content}, exp.variant())
    s = _rscore(content)
    assert (paper['n_correct_lines'], paper['n_lines']) == (2, 5)
    assert (s['parse_path'], s['n_correct_lines'], s['n_lines']) == ('paper', 1, 1)
    assert _rscore('<think>a very long trace that never ends')['parse_path'] is None


def test_robust_truncation_missing_vs_scored():
    s = _rscore('Elon Musk is the son of', finish_reason='length')
    assert s['missing'] and s['parse_path'] == 'truncated_missing'
    r = _rscore('Elon Musk is the son of', finish_reason='length', variant='robust')
    assert not r['missing'] and r['truncated_scored'] and r['n_correct_lines'] == 0
    r = _rscore("Elon Musk's mother is Maye Musk, a", finish_reason='length', variant='robust')
    assert r['truncated_scored'] and r['parse_path'] == 'robust' and r['n_correct_lines'] == 1


def _rrow(base, direction, repeat, content, variant, target='Maye Musk', finish_reason='stop'):
    item = {'id': f'{base}:{direction}', 'base_id': base, 'direction': direction, 'target': target}
    if content is None:
        return {'item': item, 'output': None, 'score': None, 'repeat': repeat, 'status': 'call_error', 'model': 'x'}
    out = {'content': content, 'finish_reason': finish_reason,
           'status': 'truncated' if finish_reason == 'length' else ('ok' if content else 'empty')}
    return {'item': item, 'output': out, 'score': exp.score(item, out, exp.variant(variant)), 'repeat': repeat,
            'status': 'complete', 'model': 'x'}


def test_robust_aggregate_counts_reliability_and_headline_keys():
    v = 'robust'
    rows = []
    for i in range(20):
        b = f'p{i}'
        rows.append(_rrow(b, 'parent_query', 0, "Elon Musk's mother is **Maye Musk**." if i % 2 else 'Maye Musk', v))
        rows.append(_rrow(b, 'child_query', 0, "I don't know." if i < 15 else 'Maye Musk', v))
    out = exp.aggregate(rows, exp.variant(v))
    fwd = out['directions']['parent_query']['parse']
    assert (fwd['parsed_paper'], fwd['parsed_robust'], fwd['unparsed'], fwd['missing']) == (10, 10, 0, 0)
    assert (out['parsed_paper'], out['parsed_robust'], out['unparsed'], out['truncated_scored']) == (30, 10, 0, 0)
    assert out['validity_rate'] == 1.0 and out['headline_reliable'] is True
    h = out['shelflife']
    assert h['metric'] == 'forward_minus_reverse_mean_over_samples' and h['value'] == pytest.approx(0.75)
    assert out['gap'] == h['value'] and out['ci95'] == list(h['ci']) and h['headline_reliable'] is True
    # 3 of 20 reverse samples unparsed (15%) -> unreliable under _robust.reliable
    for i in range(3):
        rows[2 * i + 1] = _rrow(f'p{i}', 'child_query', 0, 'You might mean someone else.', v)
    out = exp.aggregate(rows, exp.variant(v))
    assert out['directions']['child_query']['parse']['unparsed'] == 3
    assert out['headline_reliable'] is False and 'child_query: 3 unparsed' in out['headline_caveat']
    # 2 of 20 (10%) failed calls is within tolerance
    rows[1], rows[3], rows[5] = (_rrow('p0', 'child_query', 0, None, v), _rrow('p1', 'child_query', 0, None, v),
                                 _rrow('p2', 'child_query', 0, "I don't know.", v))
    out = exp.aggregate(rows, exp.variant(v))
    assert out['directions']['child_query']['parse']['missing'] == 2 and out['headline_reliable'] is True


def test_robust_format_only_misses_make_unreliable():
    rows = []
    for i in range(10):
        rows.append(_rrow(f'p{i}', 'parent_query', 0, 'maye musk' if i < 2 else 'Maye Musk', 'robust'))
        rows.append(_rrow(f'p{i}', 'child_query', 0, "I don't know.", 'robust'))
    out = exp.aggregate(rows, exp.variant('robust'))
    assert out['directions']['parent_query']['format_only_miss_share'] == 0.2
    assert out['headline_reliable'] is False


@pytest.mark.parametrize('run,variant,model', PILOT_RUNS)
def test_robust_equals_paper_where_paper_rule_parses_on_saved_pilots(run, variant, model):
    """On every saved pilot sample (no reasoning blocks or truncation occur there) the paper-path samples keep the
    authors' exact counts, every authors' hit stays on the paper path, and the fallback never removes a hit."""
    paper = exp.run_rows(RUNS / run, variant)
    robust = exp.run_rows(RUNS / run, 'robust_parse')
    assert len(paper) == len(robust)
    n_paper = 0
    for a, b in zip(paper, robust):
        assert a['item']['id'] == b['item']['id']
        if a['score'] is None:
            assert b['score'] is None
            continue
        sa, sb = a['score'], b['score']
        if sa['n_correct_lines'] or sa['idk']:
            assert sb['parse_path'] == 'paper'
        if sb['parse_path'] == 'paper':
            n_paper += 1
            for k in ('n_lines', 'n_correct_lines', 'any_line_correct', 'gpt4_rule_hit'):
                assert sa[k] == sb[k], (a['item']['id'], k)
        assert sb['n_correct_lines'] >= sa['n_correct_lines'] and sb['any_line_correct'] >= sa['any_line_correct']
    assert n_paper >= 0.95 * sum(r['score'] is not None for r in paper)


# ------------------------------------------------------- scoring revision
@pytest.mark.parametrize('content,direction', [
    # M1: negated, retracted or hedged sentence forms are never fields
    ("Some sources claim Tom Cruise's mother is Mary Lee Pfeiffer, but this is incorrect.", 'parent_query'),
    ('Mary Lee Pfeiffer has not had a son named Tom Cruise.', 'child_query'),
    ("I believe Tom Cruise's mother is Mary Lee Pfeiffer.", 'parent_query'),
    ("Tom Cruise's mother might be Mary Lee Pfeiffer.", 'parent_query'),
    ("Maybe Tom Cruise's mother is Mary Lee Pfeiffer.", 'parent_query'),
    ("Tom Cruise's mother is Mary Lee Pfeiffer. However, this is unconfirmed and possibly wrong.", 'parent_query'),
    ("Mary Lee Pfeiffer's son is rumored to be Tom Cruise.", 'child_query'),
    ("Mary Lee Pfeiffer isn't the mother of Tom Cruise.", 'child_query'),
    ("Allegedly, Tom Cruise is her son.", 'child_query'),   # ("Tom Cruise is allegedly ..." is the paper path)
])
def test_robust_rejects_negated_retracted_hedged_sentences(content, direction):
    target = 'Mary Lee Pfeiffer' if direction == 'parent_query' else 'Tom Cruise'
    s = _rscore(content, direction=direction, target=target)
    assert s['n_correct_lines'] == 0 and not s['any_line_correct'] and s['robust_fields'] == [], s
    assert s['parse_path'] in (None, 'refusal')


def test_robust_name_cues_are_lowercase_only():
    """'May' in a name is not the hedge cue 'may'; an asserted sentence still counts."""
    s = _rscore("Tom Cruise's mother is May Pfeiffer.", target='May Pfeiffer')
    assert (s['parse_path'], s['n_correct_lines']) == ('robust', 1)
    s = _rscore("Tom Cruise's mother is Mary Lee Pfeiffer.", target='Mary Lee Pfeiffer')
    assert (s['parse_path'], s['n_correct_lines']) == ('robust', 1)


@pytest.mark.parametrize('content', [
    'Her children are:\nTom Cruise\nCass Mapother',          # paper path
    'Her children are:\n- Tom Cruise\n- Cass Mapother',
    'Her children are:\n* **Tom Cruise**\n* Cass Mapother',
    'Her children are:\n1. Tom Cruise\n2. Cass Mapother',
])
def test_robust_line_share_does_not_depend_on_list_format(content):
    """Audit m1: every non-empty line is pooled, so bullets / markdown give the authors' 1/3."""
    s = _rscore(content, direction='child_query', target='Tom Cruise')
    assert (s['n_correct_lines'], s['n_lines']) == (1, 3), s
    s = _rscore('Her children are:\n* **Tom Cruise**\n* Cass', direction='child_query', target='Tom Cruise')
    assert (s['n_correct_lines'], s['n_lines']) == (1, 3)


@pytest.mark.parametrize('content,direction,path', [
    ("Who is Tom Cruise's mother? Mary Lee Pfeiffer", 'parent_query', None),     # no 'Who' field
    ("There is no public record of Mary Lee Pfeiffer's children. Tom Cruise is sometimes mentioned.", 'child_query',
     'refusal'),
    ('It is her son.', 'child_query', None),
])
def test_robust_junk_one_word_fields_are_not_parsed(content, direction, path):
    """Audit m2: a sentence-form field must open with a name; 'no public record' is a refusal."""
    target = 'Mary Lee Pfeiffer' if direction == 'parent_query' else 'Tom Cruise'
    s = _rscore(content, direction=direction, target=target)
    assert s['parse_path'] == path and s['robust_fields'] == [] and s['n_correct_lines'] == 0, s


def test_robust_mononym_and_name_list_fields():
    s = _rscore("Tom Cruise's mother is Madonna.", target='Madonna')
    assert (s['parse_path'], s['n_correct_lines']) == ('robust', 1)
    s = _rscore('Liev Schreiber has two sons named Sasha and Kai.', direction='child_query', target='Sasha Schreiber')
    assert (s['parse_path'], s['robust_fields'], s['n_correct_lines']) == ('robust', ['Sasha and Kai.'], 0)


# --------------------------------------------------------------------------------- variant fingerprints
PANEL = PROJECT / 'experiments/run_minimal_setups/runs/C141-robust-L600-expansion'


