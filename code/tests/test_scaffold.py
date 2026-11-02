"""`shelflife new` + `shelflife check`: a scaffolded draft imports, is skipped by the shared suite, and is NOT READY."""
import importlib
import ast
import json
import subprocess

import pytest

from shelflife import registry
from shelflife.paths import CODE, PROJECT


@pytest.fixture
def scaffolded(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    (repo / 'prompts.py').write_text("PROMPT = 'Q: {question}'\n")
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    subprocess.run(['git', '-C', str(repo), '-c', 'user.email=t@t', '-c', 'user.name=t', 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(repo), '-c', 'user.email=t@t', '-c', 'user.name=t', 'commit', '-qm', 'x'], check=True)
    rec = tmp_path / 'claim.json'
    rec.write_text(json.dumps({'claim': 'Models do X.', 'paper': {'title': 'Paper X', 'url': 'https://example.org/x'},
                               'experiment': {'location_in_paper': 'Table 1', 'main_result': '42%'}}))
    from shelflife.scaffold import scaffold
    out = scaffold('C999', slug='demo', repo=str(repo), claim_json=str(rec))
    files = [PROJECT / f for f in out['files']]
    try:
        yield out
    finally:
        for f in files:
            f.unlink(missing_ok=True)
        registry.REGISTRY.pop('C999', None)


def test_scaffold_then_check(scaffolded):
    assert scaffolded['commit'] and scaffolded['claim_record_found']
    mod = importlib.import_module('shelflife.experiments.c999_demo')
    exp = registry.get('C999')
    assert exp.maturity == 'draft' and exp.claim == 'Models do X.' and exp.source.commit == scaffolded['commit']
    from shelflife.check import static_checks
    checks, _ = static_checks('C999')
    failing = {name for level, name, ok, _ in checks if level == 'required' and not ok}
    assert 'no TODO left in module' in failing and 'at least one paper target' in failing
    with pytest.raises(FileExistsError):
        from shelflife.scaffold import scaffold
        scaffold('C999', slug='demo', repo=str(CODE))
    # the shared suite skips the draft (its claim tests and the generic tests)
    proc = subprocess.run(['.venv/bin/python', '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                           'tests/claims/test_c999.py', 'tests/test_registry.py', '-k', 'C999 or c999'],
                          cwd=CODE, capture_output=True, text=True)
    assert proc.returncode == 0 and 'skipped' in proc.stdout and 'failed' not in proc.stdout, proc.stdout[-500:]


@pytest.fixture
def isolated_scaffold(tmp_path, monkeypatch):
    from shelflife import scaffold
    for name, path in {'PROJECT': tmp_path, 'CODE': tmp_path / 'code',
                       'FINDINGS': tmp_path / 'findings', 'INITIAL': tmp_path / 'initial',
                       'EARLIER_RECORDS': tmp_path / 'round2', 'CLAIMS_100': tmp_path / 'claims_100.json'}.items():
        monkeypatch.setattr(scaffold, name, path)
    monkeypatch.setattr(scaffold, 'find_repo', lambda cid: None)
    monkeypatch.setattr(scaffold, 'top_level_runs', lambda cid: [])
    return scaffold


@pytest.mark.parametrize('record, expected_claim, expected_title', [
    ({'claim_set_record': {'claim': 'A selected claim.', 'paper': 'Selected paper', 'url': 'https://example.org/2'}},
     'A selected claim.', 'Selected paper'),
    ({'claim': {'statement': 'Nested claim.', 'notes': 'Do not put the whole record in the claim'},
      'paper': {'title': 'Custom paper'}}, 'Nested claim.', 'Custom paper'),
    ({'claim': {'text': 'Text with """ and a newline\ninside.'}, 'paper': 'Paper """ title'},
     'Text with """ and a newline\ninside.', 'Paper """ title'),
])
def test_scaffold_structured_metadata(isolated_scaffold, tmp_path, record, expected_claim, expected_title):
    path = tmp_path / 'input.json'
    path.write_text(json.dumps(record))
    out = isolated_scaffold.scaffold('C998', claim_json=path)
    module = tmp_path / out['files'][0]
    tree = ast.parse(module.read_text())
    compile(tree, str(module), 'exec')
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    fields = {node.targets[0].id: ast.literal_eval(node.value) for node in cls.body
              if isinstance(node, ast.Assign) and node.targets[0].id in {'title', 'claim'}}
    assert fields == {'title': expected_title, 'claim': expected_claim}
    assert {node.name for node in cls.body if isinstance(node, ast.FunctionDef)} == {
        'build_variants', 'items', 'program', 'score', 'aggregate'}
    test_path = tmp_path / out['files'][1]
    compile(test_path.read_text(), str(test_path), 'exec')
    manifest = json.loads((tmp_path / out['files'][3]).read_text())
    assert manifest == {'version': 1, 'claim_id': 'C998', 'prior_runs': [], 'excluded_runs': {}}


def test_scaffold_discovers_round2_metadata(isolated_scaffold):
    isolated_scaffold.EARLIER_RECORDS.mkdir()
    record = {'claim_set_record': {'claim': 'Selected statement.', 'paper': 'Selected paper'}}
    (isolated_scaffold.EARLIER_RECORDS / 'C998.json').write_text(json.dumps(record))
    assert isolated_scaffold.claim_record('C998') == record
    assert isolated_scaffold.scaffold('C998')['claim_record_found']


def test_bad_slug_cannot_escape_module_directory(isolated_scaffold):
    with pytest.raises(ValueError, match='slug'):
        isolated_scaffold.scaffold('C998', slug='../../outside')


def test_scaffold_limit_contract_and_scientific_todos(isolated_scaffold, tmp_path):
    out = isolated_scaffold.scaffold('C998')
    source = (tmp_path / out['files'][0]).read_text()
    # Simulate the only change an author makes to the data stub: loading an explicit release fixture.
    source = source.replace('rows = []', "rows = [{'id': 'first'}, {'id': 'second'}]")
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef))
    item_method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'items')
    namespace = {}
    exec(compile(ast.Module(body=[item_method], type_ignores=[]), '<generated items>', 'exec'), namespace)
    items = namespace['items']
    assert items(None, None, 0) == []
    assert items(None, None, 1) == [{'id': 'first'}]
    assert items(None, None) == [{'id': 'first'}, {'id': 'second'}]
    assert 'paired unit:' in source and 'denominator:' in source
    test_source = (tmp_path / out['files'][1]).read_text()
    assert 'test_failed_calls_and_paired_denominator' in test_source
    assert 'test_modern_output_fixtures' in test_source


def test_integration_guide_python_examples_execute_offline():
    """Keep the short authoring guide executable as public APIs evolve; no model transport is constructed."""
    import re
    guide = (CODE / 'docs/INTEGRATING.md').read_text()
    blocks = re.findall(r'^```python\n(.*?)^```', guide, flags=re.MULTILINE | re.DOTALL)
    assert len(blocks) == 3
    namespace = {}
    for block in blocks:
        exec(compile(block, 'docs/INTEGRATING.md', 'exec'), namespace)
    assert namespace['result']['shelflife']['value'] == 0.5
