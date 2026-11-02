"""C067 Large Legal Fictions: prompts against the authors' task files and released queries, the scorer against the
released correctness scores, the paper table, and the documented USDC Court inconsistency. No prior runs exist."""
import os
import re
from collections import Counter

import pytest

from shelflife import registry

pytest.importorskip('eyecite', reason='C067 scorer needs eyecite==2.5.2 (plus thefuzz, word2number, spaCy)')
pytest.importorskip('thefuzz')
pytest.importorskip('word2number')

from shelflife.experiments import c067_legal_hallucinations as m  # noqa: E402

exp = registry.get('C067')
pytestmark = pytest.mark.skipif(exp.maturity == 'draft' and os.environ.get('SHELFLIFE_INCLUDE_DRAFTS') != '1',
                                reason='draft claim: run `python -m shelflife.cli check C067`')


def _task_block(filename, var):
    """Independent text parse (not AST) of one Task(...) block of a tasks file."""
    text = (m.DV / filename).read_text()
    start = text.index(f'{var}: Task = Task(')
    end = text.index(f'{var}.do()', start)
    return text[start:end]


@pytest.mark.parametrize('cell', [c[0] for c in m.CELLS])
def test_prompts_and_params_equal_task_files(cell):
    key, task, court, fname, var, post, corr = m.CELL[cell]
    block = _task_block(fname, var)
    sysmsg = re.search(r"system_message=(['\"])(.*?)\1,\n", block).group(2)
    template = re.search(r'query_template="(.*?)",\n', block).group(1)
    v = exp.variant()
    assert v.prompts[f'system_{cell}'].text == sysmsg
    assert v.prompts[f'query_{cell}'].text == template
    cap = re.search(r'max_tokens=(\d+)', block)
    assert v.options['max_tokens_by_cell'][cell] == (int(cap.group(1)) if cap else 100)
    assert f'llm_answer_postprocess={post}' in block or (post == 'clean_simple' and 'llm_answer_postprocess' not in block)
    assert f'correctness_callback={corr}' in block


def _template_regex(template, sysmsg):
    rx = re.escape(template)
    rx = rx.replace(re.escape('{case_name}'), '(?P<name>.+)').replace(re.escape('{case_citation}'), '(?P<cite>.+)')
    rx = rx.replace(re.escape('{case_year}'), r'(?P<year>\d{4})').replace(re.escape('{system_message}'), re.escape(sysmsg))
    return re.compile(rx, re.S)


def test_items_interleave_cells_and_are_deterministic():
    v = exp.variant()
    first = exp.items(v, m.MINIMAL_LIMIT)
    assert Counter(x['cell'] for x in first) == {c[0]: 30 for c in m.CELLS}
    assert Counter(x['cell'] for x in exp.items(v, m.REPRO_LIMIT)) == {c[0]: 100 for c in m.CELLS}
    assert [x['id'] for x in first] == [x['id'] for x in exp.items(v, m.MINIMAL_LIMIT)]
    assert len({x['id'] for x in exp.items(v)}) == len(exp.items(v))


@pytest.mark.parametrize('llm', ['gpt4', 'gpt35'])
def test_scorer_reproduces_released_correctness(llm):
    """Our port of the authors' cleaners + callbacks on the released outputs. GPT-4: all 27,086 scores. GPT-3.5: the
    22,588 rows outside USDC Court (whose "GPT 3.5" rows are Llama 2 copies), all but one (spaCy NER fallback in
    clean_judge_name: 'Salmon P. Chase' sentence scores 23 vs 18 with en_core_web_sm 3.8.0 vs 3.5.0; both are
    hallucinations, so no verdict changes)."""
    r = m._released_cached(exp, llm)
    assert r['n'] == 27086 and r['n_compared'] == (27086 if llm == 'gpt4' else 27086 - 4498)
    misses = round((1 - r['correctness_agreement']) * r['n_compared'])
    assert misses == (0 if llm == 'gpt4' else 1)


def _first(cell):
    return next(x for x in exp.items(exp.variant()) if x['cell'] == cell)


@pytest.mark.parametrize('cell,text,expected', [
    ('court_coa', '**Eighth Circuit**', 'circuit_8'),
    ('court_coa', 'The case was decided by the U.S. Court of Appeals for the Eighth Circuit.', 'circuit_8'),
    ('court_coa', "I'm sorry, but I couldn't find information about that case.", 'abstain'),
    ('court_coa', '', 'missing'),
    ('citation_scotus', '', 'missing'),     # the authors' callback would score this 0 (hallucination): now missing
    ('citation_scotus', '   \n', 'missing'),
    ('citation_scotus', 'Unknown', 'abstain'),
    ('author_scotus', 'Justice Antonin Scalia', 'name'),
])
def test_scorer_edge_cases(cell, text, expected):
    v = exp.variant()
    item = _first(cell)
    s = exp.score(item, {'content': text}, v)
    if expected == 'missing':
        assert s['missing'] == 'empty' and s['hallucination'] is None and s['hallucination_corrected'] is None
    elif expected == 'abstain':
        assert s['abstained'] and not s['hallucination']
    elif expected == 'hallucination':
        assert s['hallucination'] and not s['abstained']
    elif expected.startswith('circuit_'):
        assert s['cleaned'] == expected.split('_')[1]
        assert s['hallucination'] == (item['truth'] != '8')
    elif expected == 'name':
        assert s['cleaned'] == 'Antonin Scalia'
        assert s['hallucination'] == (item['truth'] != 'Antonin Scalia')


def test_truncated_outputs_are_missing_and_counted():
    v = exp.variant()
    item = _first('citation_scotus')
    s = exp.score(item, {'content': 'The citation for this case is', 'finish_reason': 'length'}, v)
    assert s['missing'] == 'truncated' and s['hallucination'] is None and s['paper_verdict_if_scored'] is True
    s = exp.score(item, {'content': item['truth'], 'status': 'truncated'}, v)
    assert s['missing'] == 'truncated'


@pytest.mark.parametrize('text', ["I don't know.", 'I am not able to verify this case.',
                                  "I don't have a reliable citation for this case.", "I'm not certain who wrote it."])
def test_lenient_abstention_sensitivity(text):
    """Phrasings the code lexicon misses (audit M3): hallucinations under the code rule, abstentions under the
    lenient sensitivity rule."""
    v = exp.variant()
    cell = 'author_scotus' if 'wrote' in text else 'citation_scotus'
    s = exp.score(_first(cell), {'content': text}, v)
    assert s['hallucination'] and not s['abstained']
    assert s['abstained_lenient'] and not s['hallucination_abstain_lenient'] and s['no_answer']


def test_lenient_lexicon_does_not_fire_on_plain_answers():
    for text in ['Eighth Circuit', '449 U.S. 302', 'Antonin Scalia', 'United States District Court for the District '
                 'of Nevada', 'Justice Sandra Day O\'Connor']:
        assert not m.LENIENT_ABSTAIN.search(text)


def test_released_gpt35_usdc_court_rows_are_llama2_copies():
    """Audit M1: every released "GPT 3.5" USDC-Court output and score equals Llama 2's (asserted in _cases too), and
    the table shows GPT 3.5 = Llama 2 there; the cell is not a GPT-3.5 target."""
    other = m.released_other_rows()
    llama = other[other.llm == 'Llama 2'].set_index('query')
    g35 = m.released_rows()
    g35 = g35[(g35.llm == 'GPT 3.5') & (g35.task == 'court_id') & (g35.court_level == 'usdc')].set_index('query')
    assert len(llama) == len(g35) == 4498
    assert (g35.loc[llama.index, 'llm_output'] == llama['llm_output']).all()
    assert m.PAPER_TABLE['gpt35']['court_usdc'] == 0.815
    t = {x.name: x for x in exp.targets}
    assert 'gpt35_court_usdc_released' not in t and t['gpt35_court_usdc'].check == 'unverifiable'


def test_missing_share_above_10pct_makes_headline_unreliable():
    v = exp.variant()
    items = exp.items(v, 60)
    rows = []
    for i, it in enumerate(items):
        out = {'content': '' if i < 7 else it['truth'], 'status': 'ok'}
        rows.append({'item': it, 'output': out, 'score': exp.score(it, out, v), 'repeat': 0})
    agg = exp.aggregate(rows, v)
    assert agg['missing'] == 7 and agg['missing_empty'] == 7 and agg['scored'] == 53
    assert agg['missing_share'] > 0.10 and agg['headline_reliable'] is False
    rows = rows[5:]
    assert exp.aggregate(rows, v)['headline_reliable'] is True


def test_usdc_court_inconsistency_documented():
    """GPT-4 rows hold the state as truth (Dataverse code: case.state), Llama 2 / PaLM 2 rows the full district name. A correct
    district answer for a multi-district state is a hallucination under the code rule but not under the corrected one;
    the authors' slug->state map labels W.D. Va. as West Virginia while the district-name truth says Virginia."""
    v = exp.variant()
    items = [x for x in exp.items(v) if x['cell'] == 'court_usdc']
    ark = next(x for x in items if x['truth'] == 'Arkansas' and 'Eastern District of Arkansas' in x['truth_district'])
    s = exp.score(ark, {'content': 'United States District Court for the Eastern District of Arkansas'}, v)
    assert s['hallucination'] and not s['hallucination_corrected']
    wdva = [x for x in items if x['court_slug'] == 'wd-va']
    assert wdva and all(x['truth'] == 'West Virginia' and x['truth_district'].endswith('Western District of Virginia')
                        for x in wdva)
    single = next(x for x in items if x['truth'] == 'Nevada')
    s = exp.score(single, {'content': 'United States District Court for the District of Nevada'}, v)
    assert not s['hallucination'] and not s['hallucination_corrected']


def test_aggregate_shape_shelflife_and_failed_calls():
    v = exp.variant()
    items = exp.items(v, 12)
    rows = []
    for i, it in enumerate(items):
        out = None if i == 0 else {'content': it['truth'] if i % 2 else 'Nonsense answer Zzz', 'status': 'ok'}
        rows.append({'item': it, 'output': out, 'score': None if out is None else exp.score(it, out, v), 'repeat': 0})
    agg = exp.aggregate(rows, v)
    assert agg['n'] == 12 and agg['failed_calls'] == 1 and agg['scored'] == 11 and agg['missing'] == 0
    for k in ('headline_abstain_lenient', 'no_answer_share', 'abstention_rate_lenient', 'abstention_sensitive'):
        assert k in agg
    assert set(agg['cells']['court_usdc']) >= {'rule_paper', 'rule_corrected', 'rate_abstain_lenient', 'missing'}
    assert set(agg['cells']) == {c[0] for c in m.CELLS}
    h = agg['shelflife']
    assert set(h) == {'metric', 'value', 'ci', 'n', 'direction', 'null_value', 'paper_value', 'definition'}
    assert h['direction'] == 1 and h['paper_value'] == m.PAPER_VALUE_CORRECTED_GPT4 and h['n'] == 11
    assert 0 <= h['value'] <= 1


def test_no_auxiliary_models_and_run_constants():
    assert not exp.program(exp.variant()).auxiliary_models
    assert m.RUN_VARIANT in exp.variants and m.MINIMAL_LIMIT == 180
    assert exp.claim_set_key == m.claim_set_key == 'C067'


# --- earlier pipeline validation scoring deviations (experiments/run_minimal_setups/runs/earlier/validation/batch1.md, C067) -------------
def test_comma_citation_format_is_normalised_for_headline():
    """'330, U.S., 518' is the format the prompt asks for; eyecite cannot parse it. The code rule still calls it a
    hallucination; the corrected (headline) rule scores the normalised '330 U.S. 518'."""
    assert m.normalize_citation('330, U.S., 518') == '330 U.S. 518'
    assert m.normalize_citation('2, F. Supp. 2d, 123') == '2 F. Supp. 2d 123'
    assert m.normalize_citation('449 U.S. 302') == '449 U.S. 302'
    assert m.normalize_citation('The citation is 330, U.S., 518, decided in 1947.') == \
        'The citation is 330, U.S., 518, decided in 1947.'
    v = exp.variant()
    item = next(x for x in exp.items(v) if x['cell'] == 'citation_scotus' and x['truth'] == '330 U.S. 518') \
        if any(x['truth'] == '330 U.S. 518' for x in exp.items(v)) else _first('citation_scotus')
    vol, rep, page = re.match(r'(\d+) (.+) (\d+)$', item['truth']).groups()
    s = exp.score(item, {'content': f'{vol}, {rep}, {page}'}, v)
    assert s['hallucination'] and s['hallucination_corrected_v0']          # old rules: false failure
    assert s['comma_normalized'] and not s['hallucination_corrected']      # headline rule: correct
    s = exp.score(item, {'content': f'{int(vol) + 1}, {rep}, {page}'}, v)
    assert s['hallucination_corrected']                                    # a wrong comma citation stays wrong
    s = exp.score(item, {'content': '2005, WL, 12345'}, v)                 # fabricated (gpt-oss-120b)
    assert s['hallucination_corrected']


def test_lenient_abstention_without_answer_is_abstention_in_headline():
    """M119 author_scotus: missed by the code lexicon (hallucination under the code and v0 rules); no name, so an
    abstention in the corrected headline. A hedged answer that still names a justice stays scored."""
    v = exp.variant()
    item = _first('author_scotus')
    s = exp.score(item, {'content': "I don't have reliable information about who wrote the majority opinion in "
                                    "Savery v. Sypher."}, v)
    assert s['hallucination'] and s['hallucination_corrected_v0'] and not s['abstained']
    assert s['abstained_no_answer'] and not s['hallucination_corrected']
    wrong = 'Antonin Scalia' if item['truth'] != 'Antonin Scalia' else 'John Marshall'
    s = exp.score(item, {'content': f"I'm not certain, but I believe it was Justice {wrong}."}, v)
    assert s['abstained_lenient'] and not s['abstained_no_answer'] and s['hallucination_corrected']
    for text in ["I don't have access to that citation.", 'I cannot confirm the citation for this case.']:
        s = exp.score(_first('citation_coa'), {'content': text}, v)
        assert s['abstained_no_answer'] and not s['hallucination_corrected'] and s['hallucination']


def test_aggregate_reports_v0_and_scoring_deviations():
    v = exp.variant()
    items = exp.items(v, 12)
    rows = [{'item': it, 'output': {'content': "I don't have reliable information about this case."},
             'repeat': 0} for it in items]
    for r in rows:
        r['score'] = exp.score(r['item'], r['output'], v)
    agg = exp.aggregate(rows, v)
    assert agg['scoring_deviations'] == list(m.SCORING_DEVIATIONS) and len(m.SCORING_DEVIATIONS) == 2
    assert agg['shelflife']['value'] == 0.0 and agg['pooled_corrected_v0'] == 1.0 and agg['pooled_paper'] == 1.0
    assert agg['abstained_no_answer'] == 12


# ------------------------------------------------------------------ robustness
from pathlib import Path  # noqa: E402

R2 = Path(__file__).resolve().parents[3] / 'experiments/run_minimal_setups/runs/earlier/runs'


def test_robust_fallback_reads_unicode_spaces_and_markdown():
    rp = exp.variant('robust_parse')
    item = _first('citation_scotus')
    vol, rep, page = item['truth'].split(' ', 1)[0], 'U.S.', item['truth'].rsplit(' ', 1)[-1]
    raw = f'{vol} U.S. {page}'
    paper = exp.score(item, {'content': raw}, exp.variant())
    s = exp.score(item, {'content': raw}, rp)
    if paper['unparseable']:                      # eyecite cannot read the narrow no-break spaces
        assert s['parse_path'] == 'robust' and not s['unparseable']
    assert not s['hallucination_corrected'] or item['truth'] != f'{vol} {rep} {page}'
    # bold markdown around a correct citation
    s = exp.score(item, {'content': f'**{item["truth"]}**'}, rp)
    assert s['parse_path'] in ('paper', 'robust') and not s['unparseable']


@pytest.mark.parametrize('text,path', [
    ('Unknown', 'paper'),                                       # code-lexicon abstention stays an abstention
    ("I don't know the citation for this case.", 'paper'),     # lenient abstention with no answer
    ("I can't reliably recall the citation for this case, and I'd rather not guess.", None),   # lexicon miss: unparsed
    ('No. 2011AP002800', None),                                 # unreadable non-citation: unparsed, paper verdict kept
    ('', None), ('The citation for this case is', None),
])
def test_robust_paths_and_paper_verdicts_kept(text, path):
    rp, zero = exp.variant('robust_parse'), exp.variant()
    item = _first('citation_usdc')
    out = {'content': text, 'finish_reason': 'length' if text.endswith('is') else 'stop'}
    s, paper = exp.score(item, out, rp), exp.score(item, out, zero)
    assert s['parse_path'] == path
    assert {k: s[k] for k in paper} == paper                     # nothing the paper scorer decided changes


def test_robust_variants_params():
    zero, rp, rb = (exp.variant(n) for n in ('zero_shot', 'robust_parse', 'robust'))
    assert (rp.params, rp.prompts, rp.strategy) == (zero.params, zero.prompts, zero.strategy) and not rp.faithful
    assert rp.options == {**zero.options, 'parse': 'robust'}
    assert rb.options['max_tokens_by_cell'] == {k: max(v, 1024) for k, v in zero.options['max_tokens_by_cell'].items()}
    assert rb.options['min_cap_reasoning'] == 16384 and not rb.faithful
    item = exp.items(rb, limit=1)[0]
    assert exp.program(rb).case(item)['params']['max_tokens'] == 1024
    assert exp.program(rb).case(item)['min_cap_reasoning'] == 16384
    assert exp.program(rp).case(item) == exp.program(zero).case(item)
    assert zero.faithful and 'parse' not in zero.options


def test_truncated_answer_scored_under_robust_missing_under_robust_parse():
    """truncated='score' (robust, 1024 / 16384 caps): a non-empty cut-off answer is scored by the authors' callbacks;
    an empty one stays missing. robust_parse (truncated='missing'): every cut-off answer is missing."""
    rp, rb = exp.variant('robust_parse'), exp.variant('robust')
    assert rb.options['truncated'] == 'score' and 'truncated' not in rp.options
    item = _first('citation_scotus')
    for out, verdict in (({'content': 'The citation for this case is', 'finish_reason': 'length'}, True),
                         ({'content': item['truth'] + ' and the court further', 'status': 'truncated'}, False)):
        s_rp, s_rb = exp.score(item, out, rp), exp.score(item, out, rb)
        assert s_rp['missing'] == 'truncated' and s_rp['hallucination'] is None and s_rp['parse_path'] is None
        assert s_rb['missing'] is None and s_rb['truncated_scored'] and s_rb['hallucination'] is verdict
    # unparseable cut-off text is scored like an unparseable complete answer (a hallucination, parse_path None)
    # an EMPTY cut-off answer (reasoning budget exhausted) is scored with the authors' empty-answer mapping under
    # robust: Citation -> hallucination, Court/Author -> abstention; robust_parse: missing,
    # labelled 'truncated' (not 'empty'); an empty answer that was not cut off stays missing 'empty' everywhere
    for cell, halluc in (('citation_scotus', True), ('citation_usdc', True), ('court_coa', False),
                         ('court_usdc', False), ('author_scotus', False)):
        it = _first(cell)
        e = exp.score(it, {'content': '', 'finish_reason': 'length', 'status': 'truncated'}, rb)
        assert e['missing'] is None and e['empty_truncated'] and e['truncated_scored'] and e['parse_path'] == 'paper'
        assert e['hallucination_corrected'] is halluc and e['abstained'] is (not halluc)
        e_rp = exp.score(it, {'content': '', 'finish_reason': 'length', 'status': 'truncated'}, rp)
        assert e_rp['missing'] == 'truncated' and e_rp['hallucination'] is None
        for v in (rb, rp, exp.variant('zero_shot')):
            assert exp.score(it, {'content': '  ', 'finish_reason': 'stop'}, v)['missing'] == 'empty'
    assert exp.score(item, {'content': '', 'finish_reason': 'length'}, exp.variant('zero_shot'))['missing'] == 'truncated'
    erows = [{'item': item, 'output': {}, 'score': exp.score(item, {'content': '', 'finish_reason': 'length'}, rb)}]
    eagg = exp.aggregate(erows, rb)
    assert eagg['truncated_empty_scored'] == 1 and eagg['missing'] == 0 and eagg['missing_empty'] == 0
    rows = [{'item': item, 'output': {}, 'score': exp.score(item, {'content': 'The citation for this case is',
                                                                    'finish_reason': 'length'}, rb)}]
    agg = exp.aggregate(rows, rb)
    assert agg['truncated_scored'] == 1 and agg['missing'] == 0
