"""C104 (HL-N01) package hallucinations: prompts against the authors' prompt builder and the paper appendix, the scorer
against the authors' package_search_python, the frozen PyPI snapshot, and the paper numbers used as targets.
No prior runs exist for this claim."""
import csv
import gzip
import hashlib
import io
import json
import os
import re
from collections import Counter

import pytest

from shelflife import registry
from shelflife.experiments import c104_package_hallucination as m

exp = registry.get('C104')
pytestmark = pytest.mark.skipif(exp.maturity == 'draft' and os.environ.get('SHELFLIFE_INCLUDE_DRAFTS') != '1',
                                reason='draft claim: run `python -m shelflife.cli check C104`')


def test_items_cover_both_datasets_and_prefix_is_balanced():
    items = exp.items(exp.variant())
    assert len(items) == 4892 + 4922 == 9814
    assert len({x['id'] for x in items}) == len(items)
    for name, fname in m.DATASETS:
        lines = (m.PYDATA / fname).read_text().splitlines()
        mine = {x['line']: x['problem'] for x in items if x['dataset'] == name}
        assert mine == {i: json.loads(l) for i, l in enumerate(lines)}
    for limit in (2, 100, m.MINIMAL_LIMIT, m.REPRO_LIMIT):
        c = Counter(x['recency'] for x in exp.items(exp.variant(), limit))
        assert c['recent'] == c['all_time'] == limit // 2
    # deterministic order
    assert [x['id'] for x in exp.items(exp.variant(), 50)] == [x['id'] for x in exp.items(exp.variant(), 50)]


class _Stub:
    """Stand-ins for transformers objects: capture the chat messages generate_packages builds."""
    captured = []

    class _T:
        shape = (1, 0)

        def to(self, device):
            return self

    class _Seq:
        def __getitem__(self, key):
            return []

    class _Out:
        def __init__(self):
            self.sequences = _Stub._Seq()

    @classmethod
    def from_pretrained(cls, *a, **k):
        return cls()

    device = 'cpu'
    eos_token_id = 0

    def apply_chat_template(self, messages, **k):
        _Stub.captured.append(messages)
        return _Stub._T()

    def generate(self, *a, **k):
        return _Stub._Out()

    def decode(self, *a, **k):
        return ''


def test_default_prompts_byte_equal_to_authors_generate_packages(tmp_path):
    """Run the authors' generate_packages(mode=2) with stubbed transformers on every prompt of both datasets and
    compare its chat messages with ours. The authors pass the merged *_Master.json (Prompts, Answers) and read row[0];
    we pass the prompt file itself, whose column 0 is the same text (pandas 3 dropped positional row[0] on named
    columns). The temperature/top_p/max_new_tokens it uses are checked against our params."""
    import pandas as pd
    fn = m.load_functions(m.REPO / 'generate_package_names.py', ['generate_packages'],
                          extra_globals={'pd': pd, 'tqdm': lambda x, **k: x, 'AutoTokenizer': _Stub,
                                         'AutoModelForCausalLM': _Stub})['generate_packages']
    v = exp.variant('h3_direct')
    prog = exp.program(v)
    items = exp.items(v)
    for name, fname in m.DATASETS:
        _Stub.captured = []
        fn(2, str(m.PYDATA / fname), str(tmp_path / f'{name}.json'), 'stub', language='Python')
        mine = [x for x in items if x['dataset'] == name]
        mine.sort(key=lambda x: x['line'])
        assert len(_Stub.captured) == len(mine) > 4800
        for theirs, item in zip(_Stub.captured, mine):
            assert prog.build(item) == theirs, item['id']
    import inspect
    sig = inspect.signature(fn)
    assert sig.parameters['temperature'].default == v.params['temperature'] == 0.01
    assert sig.parameters['top_p'].default == v.params['top_p']
    assert 'max_new_tokens=64' in (m.REPO / 'generate_package_names.py').read_text() and v.params['max_tokens'] == 64


EDGE = ['numpy, pandas, scikit-learn', 'numpy,pandas\n\nThese help', '1. numpy\n2. pandas', '`numpy`, `pandas`',
        'fhirclient, fhir.resources, fhirpy, fhir-kindling-zz', 'requests, beautifulsoup4, BeautifulSoup, lxml, bs4',
        'Django, django_rest_framework, djangorestframework, drf-yasg', 'numpy, numpy, NumPy, np',
        'none', 'N/A packages', 'tensorflow, torch, keras, jax, flax, optax, haiku, dm-haiku, trax, mxnet, chainer',
        '**pandas**, **numpy**', 'pandas (for data), numpy (for math)', 'oslo.service, oslo_config, oslo-log',
        '- numpy\n- scipy\n- matplotlib', 'langchain, langchain-openai, openai, tiktoken, chromadb, faiss-cpu',
        'aws-sdk, boto3, botocore, aws_cdk.aws_s3', 'pyfhirmodels, fhir-py-resources-x9, pydantic-fhir2',
        '3', 'numpy, pandas, requests, langchain-openai, scikit-le', 'a, ab, abc, abcd']
NA_EDGE = ['None', '', 'NULL', 'nan']


def test_scorer_matches_authors_package_search_python(tmp_path):
    """The authors' package_search_python + the detect_packages CSV round trip on a frame of edge cases, compared with
    our per-response classify(). NA-producing responses are checked separately below: under pandas 3 their
    astype(str) keeps NaN and their code would crash; under pandas 2.2.1 (environment.yml) NaN -> 'nan', skipped."""
    import pandas as pd
    fns = m.load_functions(m.REPO / 'package_detection.py', ['package_search_python', 'normalize_python',
                                                               'check_packages'], extra_globals={'pd': pd, 're': re})
    fns['custom_parse_python'] = type('M', (), {'delete_dupes_and_empty': staticmethod(m.official()['delete_dupes_and_empty'])})
    df = pd.DataFrame({'Prompts': ['p'] * len(EDGE), 'Test_1': ['x'] * len(EDGE), 'Test_2': EDGE})
    df.to_csv(tmp_path / 'r.csv', index=False)
    df = pd.read_csv(tmp_path / 'r.csv')
    out = fns['package_search_python'](df, str(m.PYDATA), False, False, '')
    for text, valid, hall in zip(EDGE, out['valid_2'], out['hallucinated_2']):
        ours = m.classify(text)
        assert sorted(ours[0]) == sorted(valid) and sorted(ours[1]) == sorted(hall), text
    nonempty = sum(bool(v) or bool(h) for v, h in zip(out['valid_2'], out['hallucinated_2']))
    assert nonempty >= 15
    assert any(out['hallucinated_2'])


@pytest.mark.parametrize('text', NA_EDGE)
def test_na_responses_yield_no_packages(text):
    assert m.csv_roundtrip(text) == 'nan'
    assert m.classify(text) == ([], [])


def test_master_list_read_like_authors():
    valid, fps = m.master_sets()
    assert 'numpy' in valid and 'scikit-learn' in valid
    # pandas read_csv turns the registry names 'nan' and 'null' into NaN (one NaN entry), exactly as in the authors' run
    names = [str(x) for x in valid if isinstance(x, str)]
    assert len(names) == 500498 and len(valid) == 500499 and 'null' not in valid and 'nan' not in valid
    raw = (m.PYDATA / 'pypi_package_names.csv').read_text().split()
    assert 'null' in raw and 'nan' in raw
    assert len(fps) > 3800


def test_live_snapshot_matches_provenance():
    raw = gzip.open(m.LIVE).read()
    assert hashlib.sha256(raw).hexdigest() == 'db2820e6e8901a2e78f1eff01162ab1ea8674852c189ed3936ca21e77b6bae7d'
    live = m.live_names()
    assert len(live) == 905981
    assert {'numpy', 'langchain-openai', 'fhir-resources'} <= live
    assert 'fhir-kindling-zz' not in live


def test_score_and_aggregate_shape_and_denominators():
    v = exp.variant('h3_direct')
    items = exp.items(v, 4)
    texts = ['numpy, pandas, fakepkgzzz', 'langchain-openai, requests', None, '1. numpy\n2. pandas']
    rows = []
    for it, t in zip(items, texts):
        out = None if t is None else {'content': t, 'finish_reason': 'stop', 'status': 'ok'}
        rows.append({'item': it, 'output': out, 'model': 'M015',
                     'score': None if out is None else exp.score(it, out, v), 'repeat': 0})
    s = rows[1]['score']
    assert s['hallucinated'] == ['langchain-openai'] and s['confirmed'] == []   # created after Jan 2024
    assert s['h_post2024'] == ['langchain-openai'] and s['h_confirmed'] == []
    agg = exp.aggregate(rows, v)
    assert agg['n_prompts'] == 4 and agg['failed_calls'] == 1
    assert agg['responses_without_packages_paper_rule'] == 1 and agg['responses_without_packages'] == 0
    assert agg['paper']['k'] == 2 and agg['paper']['packages'] == 5
    assert agg['headline_upper']['k'] == 2 and agg['headline_upper']['packages'] == 7
    assert agg['headline_lower']['k'] == 1
    assert agg['n_post2024_names'] == 1 and agg['release_date'] == '2024-07-18'
    h = agg['shelflife']
    assert {'metric', 'value', 'ci', 'n', 'direction', 'null_value', 'paper_value'} <= set(h)
    assert h['null_value'] == 0.01 and h['paper_value'] is None and h['paper_value_reported'] == 0.0598
    assert h['bounds'] == [pytest.approx(1 / 7), pytest.approx(2 / 7)] and h['direction'] == 1
    assert h['no_package_share'] == 0 and h['reliable'] is True
    if 'langchain-openai' not in m.first_uploads():
        assert h['value'] is None   # not frozen yet: only bounds


def test_asof_release_rule_with_frozen_dates(tmp_path, monkeypatch):
    """Post-run step: freeze first-upload dates (fake PyPI JSON) and rescore: a post-2024 name counts as real only
    for models released after its first upload; a project without files is not real."""
    monkeypatch.setattr(m, 'FIRST_UPLOAD', tmp_path / 'first_upload.json')
    m.first_uploads.cache_clear()
    meta = {'langchain-openai': (200, {'releases': {'0.0.1': [{'upload_time_iso_8601': '2024-01-26T10:00:00Z'}],
                                                    '0.1': [{'upload_time_iso_8601': '2024-03-01T00:00:00Z'}]}}),
            'emptyproj': (200, {'releases': {'0.1': []}})}
    new = m.freeze_first_uploads(['langchain-openai', 'emptyproj'], fetch=lambda n: meta[n], today='2026-10-05T00:00Z')
    assert new['langchain-openai']['first_upload'] == '2024-01-26' and new['emptyproj']['first_upload'] is None
    assert m.freeze_first_uploads(['langchain-openai'], fetch=lambda n: 1 / 0) == {}   # never refetched
    v = exp.variant('h3_direct')
    items = exp.items(v, 2)

    def agg_for(model):
        rows = [{'item': it, 'model': model, 'output': {}, 'repeat': 0,
                 'score': exp.score(it, {'content': 'numpy, langchain-openai, fakepkgzzz'}, v)} for it in items]
        return exp.aggregate(rows, v)['shelflife']
    old = agg_for('openai-direct/gpt-3.5-turbo-0125')    # released 2024-01-25: langchain-openai not yet on PyPI
    new_ = agg_for('M015')                                 # released 2024-07-18
    assert old['value'] == pytest.approx(2 / 3) and new_['value'] == pytest.approx(1 / 3)
    m.first_uploads.cache_clear()


def test_first_upload_from_json():
    assert m.first_upload_from_json({'releases': {}}) is None
    assert m.first_upload_from_json({'releases': {'1': [{'upload_time': '2023-05-01T00:00:00'}],
                                                  '0': [{'upload_time': '2021-02-03T00:00:00'}]}}) == '2021-02-03'


@pytest.mark.parametrize('text,valid,hall,artifacts', [
    ('**scipy**, **matplotlib**', ['scipy', 'matplotlib'], [], 0),
    ('scipy==1.11, matplotlib>=3', ['scipy', 'matplotlib'], [], 0),
    ('[scipy, matplotlib, seaborn]', ['scipy', 'matplotlib', 'seaborn'], [], 0),
    ('numpy|pandas', [], [], 1),
    ('<think>let me think, maybe fakezz</think>numpy, pandas', ['numpy', 'pandas'], [], 0),
    ('numpy, pandas <think>unterminated, fakezz', ['numpy', 'pandas'], [], 0),
    ('- scipy\n- matplotlib', ['scipy', 'matplotlib'], [], 0),
    ('requests (for HTTP), beautifulsoup4', ['requests', 'beautifulsoup4'], [], 0),
    ('numpy[extra], pandas - data frames, scikit-learn: ML', ['numpy', 'pandas', 'scikit-learn'], [], 0),
    ('fhirclient, fhir-kindling-zz', ['fhirclient'], ['fhir-kindling-zz'], 0),
])
def test_name_shaped_headline_filter(text, valid, hall, artifacts):
    """Audit M4/m5: formatting artefacts of new models must not count as hallucinations in the headline."""
    v, h, a = m.classify_headline(text)
    assert v == valid and h == hall and len(a) == artifacts


def test_headline_truncation_drops_last_entry():
    v, h, _ = m.classify_headline('numpy, pandas, scikit-le', truncated=True)
    assert v == ['numpy', 'pandas'] and h == []


def test_truncation_flag_and_headline_safe_count():
    v = exp.variant('h3_direct')
    it = exp.items(v, 1)[0]
    s = exp.score(it, {'content': 'numpy, pandas, scikit-le', 'finish_reason': 'length'}, v)
    assert s['truncated'] and s['confirmed'] == ['scikit-le']        # paper rule scores the cut-off name
    assert s['h_hallucinated'] == [] and s['h_n_packages'] == 2       # headline drops it


def _table_cell(model):
    tex = (m.ARXIV / 'total_results_Python.tex').read_text()
    row = next(l for l in tex.splitlines() if l.strip().startswith(model + ' '))
    return re.findall(r'(\d+\.\d+)\\%', row)


def test_no_auxiliary_models():
    for v in exp.variants.values():
        assert not exp.program(v).auxiliary_models


def test_run_constants():
    assert m.RUN_VARIANT in exp.variants and m.MINIMAL_LIMIT == m.REPRO_LIMIT == 400
    assert exp.claim_set_key == m.claim_set_key == 'HL-N01'


# ------------------------------------------------------------------ paper-table protocol (reproductions)
def test_default_is_paper_table_protocol():
    v = exp.variant()
    assert v.name == 'paper_table_llm' and v.faithful and v.strategy == 'code_then_h2_h3'
    assert {t.name: t.variant for t in exp.targets}['gpt35_turbo_llm_prompts_rate'] == 'paper_table_llm'


def test_h2_prompts_byte_equal_to_authors_mode1(tmp_path):
    """generate_packages(mode=1) reads row['Answers'] (generated code) from the merged master file."""
    import pandas as pd
    fn = m.load_functions(m.REPO / 'generate_package_names.py', ['generate_packages'],
                          extra_globals={'pd': pd, 'tqdm': lambda x, **k: x, 'AutoTokenizer': _Stub,
                                         'AutoModelForCausalLM': _Stub})['generate_packages']
    codes = ['import numpy as np\nprint(np.zeros(3))\n', '  None  ', '```python\nimport requests\n```', 'x = 1']
    master = tmp_path / 'master.json'
    master.write_text(''.join(json.dumps({'Prompts': 'p', 'Answers': c}) + '\n' for c in codes))
    _Stub.captured = []
    fn(1, str(master), str(tmp_path / 'out.json'), 'stub', language='Python')
    prog = exp.program(exp.variant('paper_table_llm'))
    assert len(_Stub.captured) == len(codes)
    assert [prog.h2_messages(c) for c in codes] == _Stub.captured


def test_h3_step_identical_to_h3_direct():
    prog = exp.program(exp.variant('paper_table_llm'))
    base = exp.program(exp.variant('h3_direct'))
    for it in exp.items(exp.variant(), 50):
        assert prog.h3_messages(it) == base.build(it)
    assert exp.variant('paper_table_llm').params == exp.variant('h3_direct').params


def _table_outputs():
    return [('pip install requests\nimport requests', 'requests, fakezzpkg', 'numpy, pandas, nonexistzz'),
            ('None', 'pandas, fakepandaszz', 'flask, flask-fake-zz'),
            ('import numpy', '1. numpy\n2. scipy', 'numpy, numpy, NumPy'),
            ('pip install fakepipzz==1.0\npip install -U numpy', 'langchain-openai', 'seaborn'),
            ('```python\nimport torch\n```', 'torch, torchvision, fhir-kindling-zz', 'torch')]


def test_failed_prompt_counts_missing_in_table_aggregate():
    v = exp.variant('paper_table_llm')
    it = exp.items(v, 2)
    out = {'code': 'x', 'h2': 'numpy', 'h3': 'pandas'}
    rows = [{'item': it[0], 'output': out, 'model': 'M015', 'repeat': 0, 'score': exp.score(it[0], out, v)},
            {'item': it[1], 'output': None, 'model': 'M015', 'repeat': 0, 'score': None}]
    agg = exp.aggregate(rows, v)
    assert agg['n_prompts'] == 2 and agg['failed_prompts'] == 1 and agg['paper_table']['packages'] == 2


# ------------------------------------------------------------------ robustness (parse='robust', 2026-10-10)
def test_robust_variants_params():
    base, rp, rb = exp.variant('h3_direct'), exp.variant('robust_parse'), exp.variant('robust')
    assert rp.params == base.params and rp.prompts == base.prompts and rp.strategy == base.strategy
    assert rp.options['parse'] == 'robust' and not rp.faithful and rp.deviations
    assert rb.params == {'temperature': 0.01, 'top_p': 0.9, 'max_tokens': 1024} and not rb.faithful
    item = exp.items(rb, 1)[0]
    assert exp.program(rb).case(item)['min_cap_reasoning'] == m.ROBUST_MIN_CAP_REASONING
    assert exp.program(rp).case(item) == exp.program(base).case(item)
    assert exp.variant('h3_direct').faithful and exp.variant('paper_table_llm').faithful


@pytest.mark.parametrize('text,truncated,valid,hall,path', [
    ('numpy, pandas, requests', False, ['numpy', 'pandas', 'requests'], [], 'paper'),
    ('numpy, pandas, totallyfakepkgxyz', False, ['numpy', 'pandas'], ['totallyfakepkgxyz'], 'paper'),
    ('numpy\npandas\ntotallyfakepkgxyz', False, ['numpy', 'pandas'], ['totallyfakepkgxyz'], 'robust'),   # newline list
    ('- **numpy**\n- `requests`', False, ['numpy', 'requests'], [], 'robust'),                             # markdown list
    ('<think>which ones?</think>\nnumpy\nrequests', False, ['numpy', 'requests'], [], 'robust'),
    ('numpy\nrequests\nscikit-le', True, ['numpy', 'requests'], [], 'robust'),       # cut last entry dropped
    ('None', False, [], [], 'paper'),                                                   # the requested "None" answer
    ('None.', False, ['none'], [], 'paper'),     # paper quirk kept: 'None.' survives the CSV NA rule; 'none' is on PyPI
    ('', False, [], [], None),                                                          # reasoning ate the cap
    ('<think>long reasoning cut off', True, [], [], None),
    ('I cannot recommend packages for that.', False, [], [], None),                    # refusal: nothing parsed
])
def test_robust_parse_modern_formats(text, truncated, valid, hall, path):
    v, h, p = m.classify_robust(text, truncated)
    assert (sorted(m.re.sub(r'[-_.]+', '-', x).lower() for x in v), sorted(h), p) == (valid, hall, p) and p == path
    s = exp.score({'id': 'x'}, {'content': text, 'finish_reason': 'length' if truncated else 'stop'},
                  exp.variant('robust_parse'))
    assert s['parse_path'] == path and s['unparsed'] is (path is None)
    if path == 'paper':      # paper result stands wherever the paper rule parses
        assert (sorted(s['r_valid']), sorted(s['r_hallucinated'])) == (sorted(s['valid']), sorted(s['hallucinated']))


def test_robust_aggregate_paper_key_parse_counts_and_reliability():
    v = exp.variant('robust_parse')
    items = exp.items(v, 20)
    texts = ['numpy, fakepkgzzz'] * 10 + ['numpy\nfakepkgzzz'] * 7 + ['', ''] + [None]
    rows = []
    for x, t in zip(items, texts):
        out = None if t is None else {'content': t, 'finish_reason': 'stop'}
        rows.append({'item': x, 'model': 'M015', 'score': None if out is None else exp.score(x, out, v)})
    agg = exp.aggregate(rows, v)
    assert agg['parse'] == {'parsed_paper': 10, 'parsed_robust': 7, 'unparsed': 2, 'missing': 1}
    assert agg['paper']['rate'] == 17 / 34 and agg['paper_rule_only']['rate'] == 10 / 20
    assert agg['headline_reliable'] is (3 / 20 <= 0.10)
    assert 'parse' not in exp.aggregate(rows, exp.variant('h3_direct'))


