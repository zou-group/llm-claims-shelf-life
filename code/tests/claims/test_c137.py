import json
import pytest
from shelflife.experiments.c137_borderlines import BorderLines,ROOT,parse_choice,summary,countries,system_prompt
from shelflife.sourcecode import load_functions

def test_source_counts_native_choices_and_surename_metadata():
    e=BorderLines();r=e.items(e.variant())
    assert len(r)==251 and sum(len(i['queries']) for i in r)==758
    assert sum(i['controller']!='Unknown' for i in r)==161
    assert all(len(q['choices_native'])==len(i['claimants']) for i in r for q in i['queries'])
    assert any('Suriname' in i['claimants'] and i['country_info']['Suriname']['Lang_Code']=='nl' for i in r)

def test_original_parser_fallback_and_unresolved():
    assert parse_choice('France',['France','Germany'])==(0,False)
    assert parse_choice('B)',['France','Germany'])==(1,False)
    assert parse_choice('No answer',['France','Germany'])==(None,False)
    assert parse_choice('No answer',['France','Germany'],True)==(0,True)
    assert parse_choice('Germany and France',['France','Germany'],True)==(1,True)
    assert parse_choice('G)',['France','Germany'],True)==(None,False)

def test_territory_macro_not_pooled_and_known_exclusions():
    rows=[{'score':{'controller_concurrence':1,'noncontroller_concurrence':.5,'kb_concurrence':1,'consistency':.5}},
          {'score':{'controller_concurrence':0,'noncontroller_concurrence':0,'kb_concurrence':0,'consistency':1}},
          {'score':{'controller_concurrence':None,'noncontroller_concurrence':1,'kb_concurrence':1,'consistency':None}}]
    s=summary(rows)
    assert s['paired_difference']==.25
    assert s['controller_cs']==.5 and s['noncontroller_cs']==.5
    assert (s['controller_n'],s['noncontroller_n'])==(2,3)

def test_all_source_system_messages_758():
    e=BorderLines();info=countries();countryrows=[{'Country':c,**d} for c,d in info.items()]
    names=load_functions(ROOT/'run_gpt/lib.py',['langcode2lang'])['langcode2lang']
    functions=load_functions(ROOT/'run_gpt/run_inference.py',['add_lang_phrase','add_nationalist'],extra_globals={'langcode2lang':names})
    n=0
    for variant,file in [('paper_no_manual','vanilla')]:
        v=e.variant(variant);text=(ROOT/f'run_gpt/system_prompts/{file}.txt').read_text().strip()
        for item in e.items(v):
            for q in item['queries']:
                code='txt' if q['key']=='kb_en' else q['language']
                expected=functions['add_lang_phrase'](text,code,countryrows)
                expected=functions['add_nationalist'](expected,item['claimants'],item['country_info'],code)
                assert system_prompt(v,item,q)==expected
                n+=1
    assert n==758


def test_sweep_eligible_subsets_keep_only_headline_territories():
    from shelflife.experiments.c137_borderlines import BorderLines
    exp = BorderLines()
    full = {r['id']: r for r in exp.items(exp.variant('paper_no_manual'))}
    elig = exp.items(exp.variant('paper_no_manual_eligible'))
    s70 = exp.items(exp.variant('paper_no_manual_eligible_s70'))
    assert len(elig) == 141 and len(s70) == 70 and {r['id'] for r in s70} <= {r['id'] for r in elig}
    for r in elig:
        assert r == full[r['id']]  # identical queries/choices
        assert r['controller'] != 'Unknown' and r['controller_code'] in r['claimant_codes']
    assert [r['id'] for r in s70] == [r['id'] for r in exp.items(exp.variant('paper_no_manual_eligible_s70'))]


# ------------------------------------------------------------------ robustness
from shelflife.experiments.c137_borderlines import robust_choice

FR = ['France', 'Madagascar']


@pytest.mark.parametrize('text, choices, claimants, expected', [
    ('A) France', FR, FR, (0, 'paper')),
    ('**B) Madagascar**', FR, FR, (1, 'paper')),
    ('Germany and France', ['France', 'Germany'], ['France', 'Germany'], (1, 'paper')),  # author first-occurrence
    ('', FR, FR, (None, None)),                                    # empty (reasoning spent the cap): not option 0
    ('B – Pweto is a town in the DRC.', ['Zambia', 'Congo DR'], ['Zambia', 'Congo DR'], (1, 'robust')),
    ('**B**', FR, FR, (1, 'robust')),
    ('Answer: (A)', FR, FR, (0, 'paper')),                         # author single 'X)' rule
    ('Answer: A', FR, FR, (0, 'robust')),
    ('<think>long reasoning</think>\nB.', FR, FR, (1, 'robust')),
    ('ب) تركيا', ['سوريا', 'ديك رومى'], ['Syria', 'Turkey'], (1, 'robust')),          # Arabic option letter
    ('Саставци су територија Б) Србије.', ['Босна', 'Србија'], ['Bosnia', 'Serbia'], (1, 'robust')),  # Cyrillic
    ('A）中华民国', ['中华民国', '中华人民共和国'], ['ROC', 'PRC'], (0, 'paper')),
    ('Il appartient à la Madagascar', ['Frantsa', 'Madagaskar'], ['France', 'Madagascar'], (1, 'robust')),  # English name
    ('I\'m not able to identify a territory called "Gezon". A possible misspelling?', FR, FR, (None, None)),
    ("I can't answer questions about disputed territories.", FR, FR, (None, None)),
    ('A、B、C 都不正确。', ['中华民国', '不丹', '中华人民共和国'], ['ROC', 'Bhutan', 'PRC'], (None, None)),
])
def test_robust_choice_modern_formats(text, choices, claimants, expected):
    assert robust_choice(text, choices, claimants) == expected


def _robust_fixture():
    item = {'id': 'fx', 'claimants': FR, 'claimant_codes': ['fr', 'mg'], 'controller': 'France', 'controller_code': 'fr',
            'queries': [{'key': 'kb_en', 'choices_native': FR}, {'key': 'fr', 'choices_native': FR},
                        {'key': 'mg', 'choices_native': ['Frantsa', 'Madagaskar']}]}
    out = {'answers': {'kb_en': {'content': 'A) France', 'choice_index': 0, 'fallback_first': False},
                       'fr': {'content': '', 'choice_index': 0, 'fallback_first': True},
                       'mg': {'content': '**B**', 'choice_index': 0, 'fallback_first': True}}}
    return item, out


def test_robust_score_counts_and_empty_is_missing_not_option0():
    e = BorderLines()
    item, out = _robust_fixture()
    p = e.score(item, out, e.variant('paper_no_manual_eligible'))
    r = e.score(item, out, e.variant('robust_parse'))
    assert p['choices'] == {'kb_en': 'France', 'fr': 'France', 'mg': 'France'} and p['controller_concurrence'] == 1
    assert r['choices'] == {'kb_en': 'France', 'fr': None, 'mg': 'Madagascar'}
    assert r['controller_concurrence'] is None and r['noncontroller_concurrence'] == 0
    assert (r['parsed_paper'], r['parsed_robust'], r['unparsed']) == (1, 1, 1)


def test_robust_reliability_tolerant_and_paper_flag_unchanged():
    e = BorderLines()
    item, out = _robust_fixture()
    good = {'answers': {k: {'content': 'A) France', 'choice_index': 0, 'fallback_first': False} for k in ('kb_en', 'fr', 'mg')}}
    rows = [{'item': dict(item, id=f'g{i}'), 'output': good} for i in range(3)] + [{'item': item, 'output': out}]
    for v in ('robust_parse', 'paper_no_manual_eligible'):
        for r in rows:
            r['score'] = e.score(r['item'], r['output'], e.variant(v))
        agg = e.aggregate(rows, e.variant(v))
        if v == 'robust_parse':
            assert agg['unparsed'] == 1 and agg['answers'] == 12 and agg['validity_rate'] == 11 / 12
            assert agg['headline_reliable'] is True and agg['shelflife']['headline_reliable'] is True  # 1/12 <= 10%
            assert (agg['parsed_paper'], agg['parsed_robust']) == (10, 1)
        else:
            assert agg['shelflife']['headline_reliable'] is None and 'headline_reliable' not in agg
    rows.append({'item': dict(item, id='bad'), 'output': out})
    for r in rows:
        r['score'] = e.score(r['item'], r['output'], e.variant('robust_parse'))
    assert e.aggregate(rows, e.variant('robust_parse'))['headline_reliable'] is False  # 2/15 > 10%


def test_robust_variant_params_and_caps():
    e = BorderLines()
    g, rp, rb = e.variant('paper_no_manual_eligible'), e.variant('robust_parse'), e.variant('robust')
    assert rp.params == g.params and rp.prompts == g.prompts and rp.strategy == g.strategy
    assert rp.options == {**g.options, 'parse': 'robust'} and not rp.faithful
    assert rb.options['cap_floor'] == 1024 and rb.options['min_cap_reasoning'] == 8192 and not rb.faithful
    assert e.items(rb) == e.items(g)
    assert e.program(rb).case_extra == {'min_cap_reasoning': 8192} and e.program(g).case_extra == {}
    item = e.items(g, 1)[0]
    for v, caps in ((g, {128, 512}), (rb, {1024})):
        seen = set()
        class LM:
            def __call__(self, messages, params, step):
                seen.add(params['max_tokens'])
                from types import SimpleNamespace
                return SimpleNamespace(content='A)', usage=None)
        e.program(v).forward(item, LM())
        assert seen == caps


