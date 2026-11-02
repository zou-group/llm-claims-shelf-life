import os
import sys

sys.dont_write_bytecode = True  # see shelflife/__init__.py: never write caches under experiments/
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import pytest


def pytest_addoption(parser):
    parser.addoption('--live', action='store_true', help='run tests that make paid API calls')




# --- Public release (release1) -------------------------------------------------------------------------------
# The research repository also held ~14 GB of saved model outputs (earlier runs). Many tests replay those outputs to
# prove that the registry still sends byte-identical requests and that scorers reproduce saved metrics. Those outputs
# are not part of this release, so a test (or a test module) that fails only because a file inside this repository
# is missing is reported as SKIPPED with the missing path, instead of as an error. Any other failure still fails.
# Set SHELFLIFE_STRICT_INPUTS=1 to turn these skips back into failures (e.g. once the outputs are installed).
import re
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parents[2]
_STRICT = os.environ.get('SHELFLIFE_STRICT_INPUTS') == '1'


_RUN_PATH = re.compile(r"(?:/[^\s'\"]*?/|\b)((?:experiments|engineering)/[^\s'\":,)]+)")


def _missing_saved_output(exc):
    """A path to an earlier run (experiments/.../runs/..., engineering/...) named in the error and absent here."""
    for m in _RUN_PATH.finditer(str(exc)):
        rel = m.group(1).rstrip('.')
        if ('/runs/' in rel or rel.startswith('engineering/')) and not (_REPO / rel).exists():
            return rel
    return None


def _missing_repo_file(exc):
    seen = exc
    while exc is not None:
        if isinstance(exc, FileNotFoundError) and exc.filename:
            try:
                rel = _Path(exc.filename).resolve().relative_to(_REPO)
            except ValueError:
                return None
            return str(rel)
        exc = exc.__cause__ or exc.__context__
    return _missing_saved_output(seen)


def _kind(rel):
    """Input files can be downloaded (DOWNLOADS.md); everything else missing is our saved outputs."""
    import json
    manifest = _REPO / 'scripts' / 'inputs.json'
    if manifest.exists():
        paths = {e['path'] for e in json.loads(manifest.read_text())['files']}
        if rel in paths or any(rel.startswith(p.rsplit('/', 1)[0] + '/') for p in paths):
            return 'input file, run scripts/install_inputs.py'
    return 'saved outputs'


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    outcome = yield
    if _STRICT or outcome.excinfo is None:
        return
    rel = _missing_repo_file(outcome.excinfo[1])
    if rel:
        outcome.force_exception(pytest.skip.Exception(f'needs a file not in this release ({_kind(rel)}): {rel}',
                                                      _use_item_location=True))


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector):
    outcome = yield
    rep = outcome.get_result()
    if _STRICT or not rep.failed or not isinstance(collector, pytest.Module):
        return
    try:
        collector.obj  # re-raise the import error to inspect it
    except BaseException as e:  # noqa: BLE001
        rel = _missing_repo_file(e)
        if rel:
            rep.outcome = 'skipped'
            rep.longrepr = (str(collector.path), 0, f'Skipped: module needs a file not in this release ({_kind(rel)}): {rel}')


def pytest_collection_modifyitems(config, items):
    if not config.getoption('--live'):
        skip_live = pytest.mark.skip(reason='paid API test: pass --live (and -m live) to run')
        for item in items:
            if 'live' in item.keywords:
                item.add_marker(skip_live)
