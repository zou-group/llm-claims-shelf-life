"""C059: CC items, prompts against the paper's LaTeX tables, the reconstructed scorer, and the gap aggregate.

No official prompt code, scorer or outputs exist (ofirpress/self-ask releases data only), so the prompt test parses
Appendix Tables 4-5 from the arXiv v3 LaTeX independently of the module, and the aggregate test rebuilds the
Table 6 ALL row from synthetic units.
"""
import json
import os
import re
from collections import Counter, defaultdict

import pytest

from shelflife import registry
from shelflife.experiments import c059_compositionality_gap as M

exp = registry.get('C059')
pytestmark = pytest.mark.skipif(exp.maturity == 'draft' and os.environ.get('SHELFLIFE_INCLUDE_DRAFTS') != '1',
                                reason='draft claim: run `python -m shelflife.cli check C059`')
V = exp.variant('direct_2shot')
TEX = M.PAPER_SRC / 'src/tables/cc-prompts.tex'


def _latex_tables():
    """[(label, prompt text)] from cc-prompts.tex: rows between \\toprule and \\bottomrule, '\\\\' row ends removed,
    an empty row is an empty line."""
    text = TEX.read_text()
    out = []
    for block in re.findall(r'\\label\{(.*?)\}(.*?)\\end\{tabular\}', text, flags=re.S):
        label, body = block
        body = body.split('\\toprule', 1)[1].split('\\bottomrule', 1)[0]
        lines = []
        for raw in body.split('\n'):
            raw = raw.strip()
            if not raw:
                continue
            for cell in raw.split('\\\\')[:-1] if raw.endswith('\\\\') else [raw]:
                lines.append(cell.strip())
        out.append((label, '\n'.join(lines)))
    return dict(out)


def test_items_load_and_units():
    items = exp.items(V)
    assert len(items) == 3 * 8659 and len({x['id'] for x in items}) == len(items)
    units = defaultdict(list)
    for x in items:
        units[x['unit']].append(x['kind'])
    assert all(v == ['hop2', 'sub1', 'sub2'] for v in units.values())
    raw = M.load_cc()
    assert len(raw) == 8693
    # exemplar rows are not evaluated
    ex = M.exemplar_people()
    evaluated = {(x['category'], x['person']) for x in items}
    assert all((c, p) not in evaluated for c, ps in ex.items() for p in ps)
    assert len(raw) - len(units) == 34       # 17 categories x 2 exemplars


def test_exemplars():
    ex = M.exemplar_people()
    assert len(ex) == 17
    for c, people in ex.items():
        if c.startswith('birthplace'):
            assert people == ['Antonio José de Sucre', 'Martina Navratilova']
    date = [tuple(p) for c, p in ex.items() if not c.startswith('birthplace')]
    assert len(set(date)) == 1 and len(set(date[0])) == 2


def test_limit_prefix_is_stratified_minimal_design():
    items = exp.items(V, limit=M.MINIMAL_LIMIT)
    assert len(items) == 228
    units = Counter(x['unit'] for x in items)
    assert len(units) == 76 and set(units.values()) == {3}
    cats = Counter(x['category'] for x in items if x['kind'] == 'hop2')
    assert len(cats) == 17 and min(cats.values()) == 4 and max(cats.values()) == 5
    assert M.REPRO_LIMIT % 3 == 0 and M.RUN_VARIANT in exp.variants


def test_exemplar_answers_come_from_same_category_and_hop():
    raw = M.load_cc()
    by = {(r['category'], r['person']): r for r in raw}
    for x in exp.items(V, limit=600):
        qk, ak = M.HOPS[x['kind']]
        for (q, a), person in zip(x['exemplars'], M.exemplar_people()[x['category']]):
            r = by[(x['category'], person)]
            assert q == r[qk] and a == str(r[ak][0])


@pytest.mark.parametrize('text,golds,numeric,em,cover,match', [
    ('Kabul', ['Kabul'], False, True, True, True),
    (' Kabul\n\nQuestion: What is the capital of Peru?\nAnswer: Lima', ['Kabul'], False, True, True, True),
    ('Kabul\nQuestion: What is the capital of Peru?\nAnswer: Lima', ['Lima'], False, False, False, False),
    ('Answer: Kabul.', ['Kabul'], False, True, True, True),
    ('**Kabul**', ['Kabul'], False, True, True, True),
    ('The capital is Kabul', ['Kabul'], False, False, True, True),
    # short preamble / bridge entity / 2-line answers (audit M2): rescued by the headline extractor only
    ('Antonio José de Sucre was born in Venezuela, whose capital is Caracas.', ['Caracas'], False, False, True, True),
    ('Born in Venezuela.\nThe capital is Caracas.', ['Caracas'], False, False, False, True),
    ('Let me think.\nHe was born in Venezuela.\nSo the capital is Caracas.', ['Caracas'], False, False, False, False),
    ('Caracas ' + 'and more words ' * 12, ['Caracas'], False, False, True, False),   # >30 words: strict fallback
    ('', ['Kabul'], False, False, False, False),
    ("I'm sorry, I can't help with that.", ['Kabul'], False, False, False, False),
    ('Cape Town', ['Pretoria', 'Bloemfontein', 'Cape Town'], False, True, True, True),
    ('the Afghan afghani', ['Afghan afghani'], False, True, True, True),
    ('.af', ['.af'], False, True, True, True),
    ('af', ['.af'], False, True, True, True),
    ('+93', ['+93'], False, True, True, True),
    ('$', ['$'], False, True, True, True),
    ('€', ['$'], False, False, False, False),
    ('Kč', ['Kč'], False, True, True, True),
    ('アフガニスタン', ['アフガニスタン'], False, True, True, True),
    ('33', ['33'], True, True, True, True),
    ('33°', ['33'], True, True, True, True),
    ('33°N', ['33'], True, False, True, True),
    ('-66', ['-66'], True, True, True, True),
    ('−66', ['-66'], True, True, True, True),
    ('66°W', ['-66'], True, False, False, True),
    ('66', ['-66'], True, False, False, False),
    ('33.5', ['33'], True, False, True, False),
    ('Born in 1722 in Afghanistan; latitude 33', ['33'], True, False, False, True),   # audit m3: last number
    ('José María Olazábal', ['José María Olazábal'], False, True, True, True),
    ('Jose Maria Olazabal', ['José María Olazábal'], False, False, False, False),
])
def test_scorer_edge_cases(text, golds, numeric, em, cover, match):
    s = M.score_answer(text, golds, numeric)
    assert (s['em'], s['cover_em'], s['match']) == (em, cover, match), s


def test_score_flags_truncation_and_empty():
    s = exp.score({'answers': ['Kabul'], 'numeric': False}, {'content': '', 'finish_reason': 'length'}, V)
    assert s['truncated'] and s['empty'] and not s['match']


SCORE_KEYS = ('match', 'em', 'cover_em', 'cover_full')


def _unit_rows(unit, category, person, flags, truncated=None, score=None):
    rows = []
    for kind, ok in zip(('hop2', 'sub1', 'sub2'), flags):
        item = {'id': f'{unit}:{kind}', 'unit': unit, 'kind': kind, 'category': category, 'person': person}
        sc = score(kind, ok) if score else {**{k: ok for k in SCORE_KEYS}, 'truncated': False, 'empty': False}
        if truncated and kind in truncated:
            sc = {**sc, 'truncated': True}
        rows.append({'item': item, 'output': {'content': ''}, 'score': sc, 'model': 'x', 'repeat': 0,
                     'status': 'complete'})
    return rows


def _rows(counts, missing_units=0, category='birthplace_capital'):
    """Synthetic runner rows: counts = {(sub1, sub2, hop2): n_units}."""
    rows, k = [], 0
    for (s1, s2, h2), n in counts.items():
        for _ in range(n):
            rows += _unit_rows(f'U{k:05d}', category, f'P{k:05d}', (h2, s1, s2))
            k += 1
    for j in range(missing_units):
        unit = f'M{j:03d}'
        for kind in ('hop2', 'sub1', 'sub2'):
            item = {'id': f'{unit}:{kind}', 'unit': unit, 'kind': kind, 'category': category, 'person': unit}
            rows.append({'item': item, 'output': None, 'score': None, 'model': 'x', 'repeat': 0, 'status': 'call_error'})
    return rows


def _table6_units(mix):
    """Synthetic units per category following davinci-002's Table 6 category rows; mix = {category: n_units}."""
    rows, k = [], 0
    for cat, n in mix.items():
        rr, rw, wr, ww = M.table6_rows()[cat]
        tot = rr + rw + wr + ww
        cells = [((True, True, True), rr), ((False, True, True), rw), ((True, True, False), wr),
                 ((False, False, False), ww)]   # flags = (hop2, sub1, sub2)
        for flags, share in cells:
            for _ in range(round(n * share / tot)):
                rows += _unit_rows(f'U{k:06d}', cat, f'P{k:06d}', flags)
                k += 1
    return rows


def _kind_score(verbose_hop2=False, trunc_hop2=False):
    def f(kind, ok):
        s = {k: ok for k in SCORE_KEYS}
        s.update(truncated=False, empty=False)
        if kind == 'hop2' and verbose_hop2:
            s.update(match=False, em=False, cover_em=False, cover_full=True)
        if kind == 'hop2' and trunc_hop2:
            s.update(match=False, em=False, cover_em=False, cover_full=False, truncated=True)
        return s
    return f


def test_headline_reliable_is_gated_on_the_2hop_kind():
    """Audit M2/M3: 15% verbose (gold present, not extracted) 2-hop answers with terse subs -> unreliable, even
    though the pooled rate over all kinds is only 5%; 3% truncated 2-hop answers -> unreliable."""
    def build(n_bad, scorer):
        rows = []
        for k in range(100):
            rows += _unit_rows(f'U{k}', 'birthplace_capital', f'P{k}', (True, True, True),
                               score=scorer if k < n_bad else None)
        return rows
    ok = exp.aggregate(build(5, _kind_score(verbose_hop2=True)), V)
    assert ok['headline_reliable'] is True
    bad = exp.aggregate(build(15, _kind_score(verbose_hop2=True)), V)
    assert bad['headline_reliable'] is False and bad['by_kind']['hop2']['gold_in_output_not_extracted'] == 0.15
    assert bad['by_kind']['sub1']['gold_in_output_not_extracted'] == 0
    tr = exp.aggregate(build(3, _kind_score(trunc_hop2=True)), V)
    assert tr['headline_reliable'] is False and tr['by_kind']['hop2']['truncated_rate'] == 0.03


def test_bootstrap_clusters_on_person():
    rows = []
    for k in range(40):
        rows += _unit_rows(f'A{k}', 'birthplace_capital', f'P{k}', (k % 3 == 0, True, True))
        rows += _unit_rows(f'B{k}', 'birthplace_tld', f'P{k}', (k % 3 == 0, True, True))
    agg = exp.aggregate(rows, V)
    lo, hi = agg['gap_ci95_bootstrap']
    assert lo <= agg['compositionality_gap'] <= hi


def test_aggregate_no_both_right_units():
    agg = exp.aggregate(_rows({(False, False, False): 10}), V)
    assert agg['compositionality_gap'] is None and agg['shelflife']['value'] is None and agg['shelflife']['ci'] is None


def test_aggregate_on_real_items_with_scored_outputs():
    """End to end through score(): perfect answers give gap 0; answering every 2-hop wrongly gives gap 1."""
    items = exp.items(V, limit=60)
    good = [{'item': x, 'output': {'content': x['answers'][0], 'finish_reason': 'stop'}, 'model': 'm', 'repeat': 0,
             'status': 'complete'} for x in items]
    for r in good:
        r['score'] = exp.score(r['item'], r['output'], V)
    agg = exp.aggregate(good, V)
    assert agg['compositionality_gap'] == 0 and agg['headline_reliable'] is True
    bad = [dict(r, output={'content': 'zzz'}) if r['item']['kind'] == 'hop2' else r for r in good]
    for r in bad:
        r['score'] = exp.score(r['item'], r['output'], V)
    assert exp.aggregate(bad, V)['compositionality_gap'] == 1


def test_no_auxiliary_models():
    assert not getattr(exp.program(V), 'auxiliary_models', ())


# ------------------------------------------------------------------ robustness
RP, RB = exp.variant('robust_parse'), exp.variant('robust')
PANEL20 = ('M015 M017 M025 M045 M049 M066 M090 M102 M119 M143 M144 M171 M183 M190 M196 M199 M204 M206 '
           'M207').split()


def _rs(text, golds, numeric=False, finish='stop'):
    return M.score_answer(text, golds, numeric, finish, parse='robust')


@pytest.mark.parametrize('text,golds,numeric,match,path', [
    ('Caracas', ['Caracas'], False, True, 'paper'),
    ('**Caracas**', ['Caracas'], False, True, 'paper'),
    # long first lines: paper window None -> robust fallback
    ('Leif Erikson was most likely born in Iceland around 970 AD. His father, Erik the Red, was exiled from Iceland '
     'and later settled Greenland, but Leif is generally believed to have been born before that move.\n\n'
     '**Answer: Iceland**', ['Iceland'], False, True, 'robust'),
    ('**Samuel Beckett** won the Nobel Prize in Literature in 1969, cited "for his writing, which, in new forms for '
     'the novel and drama, in the destitution of modern man acquires its elevation" and much more besides here.\n\n'
     'One note on your earlier answers: Agnon shared the 1966 prize.', ['Samuel Beckett'], False, True, 'robust'),
    ('Tiger Woods won the 2002 Masters Tournament. It was his third Masters title, and he became the third player to '
     'win the tournament in back-to-back years after Nicklaus and Faldo, which is a remarkable achievement indeed.',
     ['Tiger Woods'], False, True, 'robust'),
    ('Reasoning about latitude of the city and the rounding convention used by the earlier examples in this prompt, '
     'which use the country level rather than the city, and so on for a while longer than thirty words.\n\n'
     'Final answer: $\\boxed{18}$', ['18'], True, True, 'robust'),
    ('<think>' + 'x ' * 40 + '</think>' + ' word' * 35 + '\n\n' + 'more ' * 35, ['Kabul'], False, False, None),
    ("I don't know.", ['Kabul'], False, False, 'paper'),          # refusal stays a (wrong) parsed answer
    ('', ['Kabul'], False, False, None),
])
def test_robust_parse_modern_formats(text, golds, numeric, match, path):
    s = _rs(text, golds, numeric)
    assert (s['match'], s['parse_path']) == (match, path), s
    assert s['unparsed'] == (path is None)


def test_robust_truncated_rows_are_unparsed_whatever_they_contain():
    cut = 'To answer this question, we need to identify the birthplace of Leif Erikson and then'
    s = _rs(cut, ['Reykjavik'], finish='length')
    assert s['parse_path'] is None and s['unparsed'] and not s['match']
    hit = _rs('Reykjavik\n\nLeif Erikson was born in', ['Reykjavik'], finish='length')
    assert hit['parse_path'] is None and hit['unparsed']          # every cut-off row, matched or not (outcome-independent)
    # the paper scorer is untouched: same text, paper parse, still scored wrong and not flagged
    assert 'parse_path' not in M.score_answer(cut, ['Reykjavik'], False, 'length')


def test_robust_aggregate_drops_unparsed_units_and_uses_tolerant_reliability():
    def build(n_trunc):
        rows = []
        for k in range(100):
            sc = lambda kind, ok, k=k: {**{x: ok for x in SCORE_KEYS}, 'truncated': False, 'empty': False,
                                        'parse_path': None if (kind == 'hop2' and k < n_trunc) else 'paper',
                                        'unparsed': kind == 'hop2' and k < n_trunc}
            rows += _unit_rows(f'U{k}', 'birthplace_capital', f'P{k}', (k >= n_trunc, True, True), score=sc)
        return rows
    paper = exp.aggregate(build(8), V)
    rob = exp.aggregate(build(8), RP)
    assert paper['compositionality_gap'] == pytest.approx(0.08) and paper['n_units'] == 100
    assert rob['compositionality_gap'] == 0 and rob['n_units'] == 92        # cut-off units are missing, not wrong
    assert rob['headline_reliable'] is True and rob['by_kind']['hop2']['unparsed'] == 8
    assert (rob['parsed_paper'], rob['parsed_robust'], rob['unparsed']) == (292, 0, 8)
    assert exp.aggregate(build(11), RP)['headline_reliable'] is False      # 11% > 10% tolerance


def test_robust_variant_params():
    assert RP.params == V.params and RP.prompts == V.prompts and RP.strategy == V.strategy
    assert RP.options['parse'] == 'robust' and not RP.faithful and 'min_cap_reasoning' not in RP.options
    assert RB.params == {'temperature': 0, 'max_tokens': 1024} and not RB.faithful
    assert RB.options['parse'] == 'robust' and RB.options['min_cap_reasoning'] == M.ROBUST_MIN_CAP_REASONING
    assert exp.program(RB).case_extra == {'min_cap_reasoning': 16384} and exp.program(RP).case_extra == {}
    assert exp.program(V).case_extra == {} and V.params == {'temperature': 0, 'max_tokens': 32} and V.faithful


def _saved_runs():
    root = M.PROJECT / 'experiments/run_minimal_setups/runs/earlier/runs'
    return [d for tag in ('C059-direct_2shot-L228-r4', 'C059-direct_2shot-L228-r5') for m in PANEL20
            for d in [root / tag / m] if (d / 'responses.jsonl').exists()]


def test_truncated_rows_scored_under_robust_unparsed_under_robust_parse():
    """truncated='score' (robust, generous cap): a cut-off output is scored by the paper rule on its text, never
    unparsed for being cut; robust_parse (truncated='missing'): unparsed whatever it contains."""
    assert RB.options['truncated'] == 'score' and 'truncated' not in RP.options
    item = {'answers': ['Reykjavik'], 'numeric': False}
    loop = {'content': 'To answer this question, we need to identify the birthplace of Leif Erikson and then ' * 40,
            'finish_reason': 'length'}
    hit = {'content': 'Reykjavik\n\nLeif Erikson was born in', 'finish_reason': 'length'}
    empty = {'content': '', 'finish_reason': 'length'}
    for out in (loop, hit, empty):
        assert exp.score(item, out, RP)['unparsed'] and exp.score(item, out, RP)['parse_path'] is None
        s = exp.score(item, out, RB)
        assert not s['unparsed'] and s['truncated_scored'] and s['parse_path'] in ('paper', 'robust')
    assert exp.score(item, hit, RB)['match'] is True
    assert exp.score(item, loop, RB)['match'] is False and exp.score(item, empty, RB)['match'] is False
    # an empty COMPLETE output stays unparsed under robust (empty is not a cap failure)
    assert exp.score(item, {'content': '', 'finish_reason': 'stop'}, RB)['unparsed']
    # aggregate: cut-off 2-hop rows stay in the headline table (wrong), counted as truncated_scored
    rows = []
    for k in range(100):
        cut = k < 8
        sc = lambda kind, ok, cut=cut: {**{x: ok for x in SCORE_KEYS}, 'truncated': kind == 'hop2' and cut,
                                        'empty': False, 'parse_path': 'paper', 'unparsed': False,
                                        'truncated_scored': kind == 'hop2' and cut}
        rows += _unit_rows(f'U{k}', 'birthplace_capital', f'P{k}', (not cut, True, True), score=sc)
    agg = exp.aggregate(rows, RB)
    assert agg['n_units'] == 100 and agg['compositionality_gap'] == pytest.approx(0.08)
    assert agg['truncated_scored'] == 8 and agg['unparsed'] == 0 and agg['headline_reliable'] is True


def test_gold_not_extracted_guard_measures_the_robust_scorer():
    """Under parse='robust' the 'gold in output but not extracted' guard reads the robust match: a row whose gold the
    fallback extracted (and scored correct) is not counted; a row whose extracted short answer misses a gold that
    appears later in the text is counted, because the robust scorer scores it wrong."""
    rescued = _rs('**Reykjavik** is the capital of Iceland, where ' + 'Leif Erikson was born. ' * 6, ['Reykjavik'])
    assert rescued['parse_path'] == 'robust' and rescued['match'] and rescued['cover_full']
    assert not (rescued['cover_full'] and not rescued['match'])
    wrong = _rs('Iceland\n\n' + 'Leif Erikson was born in Iceland around 970 AD, as the sagas tell us. ' * 3
                + 'Its capital today is Reykjavik.', ['Reykjavik'])
    assert wrong['parse_path'] == 'robust' and wrong['robust_window'] == 'Iceland'
    assert not wrong['match'] and wrong['cover_full']


# M066 / M025 shapes on the panel robust run (C059-robust-L228-robust): worked answers whose first line is a
# reasoning preamble. The fallback must take the conclusion, never the preamble.
M066_PUSKAS = ("To answer this question, we need to identify the birthplace of Ferenc Puskás and then determine its "
               "capital.\n\n1. **Identify Ferenc Puskás's birthplace**: Ferenc Puskás was a Hungarian footballer, and he "
               "was born in Budapest, Hungary.\n\n2. **Determine the capital of his birthplace**: Since Budapest is the "
               "city where Ferenc Puskás was born and it is also the capital of Hungary, the answer is straightforward."
               "\n\nTherefore, the capital of the birthplace of Ferenc Puskás is **Budapest**.")
M066_MNANGAGWA = ("To answer the question, we need to identify the birthplace of Emmerson Mnangagwa and provide its "
                  "Spanish name.\n\n1. **Identify the birthplace of Emmerson Mnangagwa**: Emmerson Mnangagwa is a "
                  "Zimbabwean politician who has been serving as the President of Zimbabwe since 2017. He was born in "
                  "Zvishavane, in what was then Southern Rhodesia.\n\n2. **Provide the Spanish name**: The country is "
                  "Zimbabwe, which in Spanish is Zimbabue.\n\nSo the final answer is: Zimbabue")
STEP_HEADINGS = ("## Step 1: Identify the birthplace of Leif Erikson\nLeif Erikson was born in Iceland, according to the "
                 "Norse sagas that describe the voyages of his family and their settlement of Greenland in "
                 "considerable detail.\n\n## Step 2: Determine "
                 "the capital\nThe capital of Iceland is Reykjavik.\n\nThe final answer is: Reykjavik")
MASTERS_LIST = ("To determine the champion, we need the birth year of Jon Favreau and then the Masters winner of that "
                "year.\n\n1. **Jon Favreau's birth year**: Jon Favreau was born on October 19, 1966.\n\n2. **Masters "
                "Tournament 1966**: The 1966 Masters was won by Jack Nicklaus, his second consecutive green jacket.\n\n"
                "Thus, the champion of the Masters Tournament in the year Jon Favreau was born is Jack Nicklaus.")


@pytest.mark.parametrize('text,golds,window', [
    (M066_PUSKAS, ['Budapest'], 'Therefore, the capital of the birthplace of Ferenc Puskás is Budapest'),
    (M066_MNANGAGWA, ['Zimbabue'], 'Zimbabue'),
    (STEP_HEADINGS, ['Reykjavik'], 'Reykjavik'),
    (MASTERS_LIST, ['Jack Nicklaus'], 'Thus, the champion of the Masters Tournament in the year Jon Favreau was born is '
                                      'Jack Nicklaus'),
])
def test_robust_window_skips_reasoning_preambles(text, golds, window):
    s = _rs(text, golds)
    assert s['window'] is None and s['parse_path'] == 'robust'
    assert s['robust_window'] == window and s['match']
    # the same shape with a WRONG conclusion is scored wrong (outcome-independent: the rule never reads the gold)
    assert not _rs(text, ['Kabul'])['match']


@pytest.mark.parametrize('line', ['## Step 1: Identify the birthplace', 'To answer the question, we need to know',
                                  "Let's think step by step", 'First, find the birth year', 'We need two facts',
                                  '1. Identify the birthplace of X', 'Reasoning:', 'If you need more details, ask',
                                  'Note: the city was renamed'])
def test_is_preamble(line):
    assert M.is_preamble(_robust_norm(line))


@pytest.mark.parametrize('line', ['Reykjavik', 'Therefore, the capital is Budapest', 'Jack Nicklaus won in 1966',
                                  'The answer is Euro', '+387', 'Samuel Beckett'])
def test_is_not_preamble(line):
    assert not M.is_preamble(_robust_norm(line))


def test_paper_window_still_stands_for_short_step_preambles():
    """Paper first, as always: when the first two lines fit the paper's 30-word window (M066's '## Step 1: Identify
    the birthplace of Antonio José de Sucre' + one sentence), the paper decision stands; the fallback is not used."""
    text = ('## Step 1: Identify the birthplace of Leif Erikson\nLeif Erikson was born in Iceland.\n\n'
            '## Step 2: Determine the capital\nThe capital of Iceland is Reykjavik.')
    s = _rs(text, ['Reykjavik'])
    assert s['window'] is not None and s['parse_path'] == 'paper' and not s['match'] and s['cover_full']


def _robust_norm(line):
    from shelflife.experiments import _robust
    return _robust.normalize(line)


def test_robust_window_marker_filler_and_short_first_line():
    # 'the answer is straightforward.' is not an answer marker; the conclusion line is used instead
    assert M.robust_window('Since Lima is the city, the answer is straightforward.\n\nTherefore it is Lima.') == \
        'Therefore it is Lima'
    # a short first line opening a long explanation is the answer
    long = 'Lima\n\n' + 'Mario Vargas Llosa was born in Arequipa, Peru, and the capital of Peru is Lima. ' * 3
    assert M.robust_window(long) == 'Lima'
    # a reply that is only reasoning preambles and long lines has no short answer: unparsed
    assert M.robust_window('To answer this, we need to recall facts.\n\n' + 'word ' * 40) is None
