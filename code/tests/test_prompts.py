import pytest

from shelflife.prompts import PromptTemplate, Source, extract_text
from shelflife.registry import Variant


def test_render_styles():
    assert PromptTemplate('a', 'Q: {q}').render(q='x') == 'Q: x'
    assert PromptTemplate('a', 'Q: [Q] {keep}', style='replace', tokens={'[Q]': 'q'}).render(q='x') == 'Q: x {keep}'
    assert PromptTemplate('a', 'Q: %(q)s', style='percent').render(q='x') == 'Q: x'
    with pytest.raises(KeyError):
        PromptTemplate('a', 'Q: {q}').render()


def test_extract_from_python_source(tmp_path):
    f = tmp_path / 'm.py'
    f.write_text("X = 'hello\\n'\nD = {'k': 'v'}\nimport os\nos.environ['A'] = 'b'\n")
    src = Source(repo=str(tmp_path), path='m.py')
    t = PromptTemplate.from_source('x', src, style='literal', kind='py_var', var='X')
    assert t.text == 'hello\n' and t.verify_source()
    assert extract_text(src, kind='py_dict', var="D['k']") == 'v'


def test_override_marks_variant_unfaithful():
    base = Variant('v', 'd', prompts={'user': PromptTemplate('user', 'Q: {q}')}, params={'temperature': 0})
    assert base.faithful
    assert base.override('o', options={'n': 5}).faithful
    assert not base.override('p', prompts={'user': 'Question: {q}'}).faithful
    assert not base.override('t', params={'temperature': 1}).faithful
    assert not base.override('s', strategy='cot').faithful
    assert base.override('same', params={'temperature': 0}).faithful
    assert base.fingerprint != base.override('p', prompts={'user': 'Question: {q}'}).fingerprint
