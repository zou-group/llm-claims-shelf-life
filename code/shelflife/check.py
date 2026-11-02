"""`shelflife check CNNN`: one command that says what is left before a claim is ready.

Static checklist (no API calls) followed by the claim's test selection with drafts included. Exit code 0 only when
everything required passes; warnings (e.g. a paper judge with no route, audit not yet written) do not block.
"""
import inspect
import os
import re
import subprocess
import sys
from pathlib import Path

from . import registry
from .paths import CODE, PROJECT



def _todo_lines(path):
    if not path or not Path(path).exists():
        return []
    return [i + 1 for i, line in enumerate(Path(path).read_text().splitlines()) if 'TODO' in line]


def static_checks(cid):
    out = []   # (level, name, ok, detail); level 'required' | 'warn'
    add = lambda level, name, ok, detail='': out.append((level, name, bool(ok), detail))
    registry.load_all()
    module_errors = {k: v for k, v in registry.IMPORT_ERRORS.items() if k.startswith(cid.lower())}
    add('required', 'module imports', not module_errors, '; '.join(module_errors.values()))
    if cid not in registry.REGISTRY:
        add('required', 'registered with @register', False, f'no experiment registered as {cid}')
        return out, None
    exp = registry.get(cid)
    module_file = Path(inspect.getsourcefile(type(exp)))
    test_file = CODE / 'tests/claims' / f'test_{cid.lower()}.py'
    todos = _todo_lines(module_file)
    add('required', 'no TODO left in module', not todos, f'{module_file.name} lines {todos[:10]}' if todos else '')
    add('required', 'claim test file exists', test_file.exists(), str(test_file.relative_to(PROJECT)))
    t_todos = _todo_lines(test_file)
    add('required', 'no TODO left in claim tests', not t_todos, f'lines {t_todos[:10]}' if t_todos else '')
    try:
        variants = exp.variants
    except Exception as e:
        add('required', 'variants build', False, f'{type(e).__name__}: {e}')
        return out, exp
    default = variants.get(exp.default_variant)
    add('required', 'default variant exists and is faithful', default is not None and default.faithful,
        exp.default_variant)
    for v in variants.values():
        if not v.faithful:
            continue
        for t in v.prompts.values():
            add('required', f'{v.name}/{t.name}: prompt has provenance', t.source is not None)
            verified = t.verify_source()
            if verified is not None:
                add('required', f'{v.name}/{t.name}: prompt matches its source file', verified)
            else:
                add('warn', f'{v.name}/{t.name}: prompt re-extractable from its source', False,
                    'no extract spec: verbatim text is not re-verified automatically; a claim test must compare it '
                    'with the source (code output, or the paper\'s LaTeX/PDF text)')
        try:
            items = exp.items(v, limit=5)
            add('required', f'{v.name}: items load with unique ids',
                items and len({x['id'] for x in items}) == len(items), f'{len(items)} of first 5')
        except Exception as e:
            add('required', f'{v.name}: items load', False, f'{type(e).__name__}: {e}')
            continue
        try:
            prog = exp.program(v)
            from .policies import policy_for
            blocked = []
            for m in getattr(prog, 'auxiliary_models', ()) or ():
                try:
                    policy_for(m)
                except KeyError:
                    blocked.append(m)
            add('warn', f'{v.name}: auxiliary models routable', not blocked,
                f'no transport policy for {blocked}: runs are stopped by the preflight' if blocked else '')
        except Exception as e:
            add('required', f'{v.name}: program builds', False, f'{type(e).__name__}: {e}')
    add('required', 'at least one paper target', len(exp.targets) > 0)
    for t in exp.targets:
        if t.check in registry.OFFLINE_CHECKS:
            try:
                value = t.compute(exp)
                within = abs(value - t.value) <= t.tolerance
                ok = within != bool(t.known_discrepancy)
                detail = f'{value:.4g} vs paper {t.value:.4g}' + (' (documented discrepancy)' if t.known_discrepancy else '')
            except Exception as e:
                ok, detail = False, f'{type(e).__name__}: {e}'
            add('required', f'target {t.name}', ok, detail)
    scoring_anchor = sum(t.check in {'archived_outputs', 'prior_run'} for t in exp.targets)
    data_anchor = sum(t.check == 'released_data' for t in exp.targets)
    add('warn', 'scoring checked offline against a paper number (released outputs or prior run)', scoring_anchor > 0,
        'no archived_outputs/prior_run target: model scoring is not anchored to a paper number'
        + (' (released_data targets anchor the data only)' if data_anchor else '') if not scoring_anchor else '')
    return out, exp
    default = variants.get(exp.default_variant)
    add('required', 'default variant exists and is faithful', default is not None and default.faithful,
        exp.default_variant)
    for v in variants.values():
        if not v.faithful:
            continue
        for t in v.prompts.values():
            add('required', f'{v.name}/{t.name}: prompt has provenance', t.source is not None)
            verified = t.verify_source()
            if verified is not None:
                add('required', f'{v.name}/{t.name}: prompt matches its source file', verified)
            else:
                add('warn', f'{v.name}/{t.name}: prompt re-extractable from its source', False,
                    'no extract spec: verbatim text is not re-verified automatically; a claim test must compare it '
                    'with the source (code output, or the paper\'s LaTeX/PDF text)')
        try:
            items = exp.items(v, limit=5)
            add('required', f'{v.name}: items load with unique ids',
                items and len({x['id'] for x in items}) == len(items), f'{len(items)} of first 5')
        except Exception as e:
            add('required', f'{v.name}: items load', False, f'{type(e).__name__}: {e}')
            continue
        try:
            prog = exp.program(v)
            from .policies import policy_for
            blocked = []
            for m in getattr(prog, 'auxiliary_models', ()) or ():
                try:
                    policy_for(m)
                except KeyError:
                    blocked.append(m)
            add('warn', f'{v.name}: auxiliary models routable', not blocked,
                f'no transport policy for {blocked}: runs are stopped by the preflight' if blocked else '')
        except Exception as e:
            add('required', f'{v.name}: program builds', False, f'{type(e).__name__}: {e}')
    add('required', 'at least one paper target', len(exp.targets) > 0)
    for t in exp.targets:
        if t.check in registry.OFFLINE_CHECKS:
            try:
                value = t.compute(exp)
                within = abs(value - t.value) <= t.tolerance
                ok = within != bool(t.known_discrepancy)
                detail = f'{value:.4g} vs paper {t.value:.4g}' + (' (documented discrepancy)' if t.known_discrepancy else '')
            except Exception as e:
                ok, detail = False, f'{type(e).__name__}: {e}'
            add('required', f'target {t.name}', ok, detail)
    scoring_anchor = sum(t.check in {'archived_outputs', 'prior_run'} for t in exp.targets)
    data_anchor = sum(t.check == 'released_data' for t in exp.targets)
    add('warn', 'scoring checked offline against a paper number (released outputs or prior run)', scoring_anchor > 0,
        'no archived_outputs/prior_run target: model scoring is not anchored to a paper number'
        + (' (released_data targets anchor the data only)' if data_anchor else '') if not scoring_anchor else '')
    return out, exp


def run_tests(cid, extra=()):
    tests = [f'tests/claims/test_{cid.lower()}.py', 'tests/test_registry.py', 'tests/test_targets.py']
    tests = [t for t in tests if (CODE / t).exists()]
    env = {**os.environ, 'SHELFLIFE_INCLUDE_DRAFTS': '1', 'PYTHONDONTWRITEBYTECODE': '1'}
    proc = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '-rfE', *tests, '-k', cid, *extra],
                          cwd=CODE, env=env, capture_output=True, text=True)
    lines = proc.stdout.strip().splitlines()
    summary = next((l for l in reversed(lines) if re.search(r'\d+ (passed|failed|error)', l)), lines[-1] if lines else '')
    failures = [l for l in lines if l.startswith(('FAILED', 'ERROR'))]
    return proc.returncode, summary, failures


def main(cid, run_pytest=True, verbose=False):
    checks, exp = static_checks(cid)
    print(f'shelflife check {cid}' + (f'  (maturity: {exp.maturity})' if exp else ''))
    shown = checks if verbose else [c for c in checks if not c[2]]
    passed = sum(ok for _, _, ok, _ in checks)
    print(f'  {passed}/{len(checks)} checks pass' + ('' if verbose else ' (use --verbose to list them all)'))
    # identical warnings for many variants are collapsed into one line
    seen = {}
    for level, name, ok, detail in shown:
        key = (level, ok, name.split(': ', 1)[-1], detail)
        seen.setdefault(key, []).append(name.split(':', 1)[0] if ': ' in name else None)
    for (level, ok, name, detail), owners in seen.items():
        mark = 'ok  ' if ok else ('FAIL' if level == 'required' else 'warn')
        where = '' if owners == [None] else f" [{', '.join(o for o in owners if o)}]"
        print(f'  [{mark}] {name}{where}' + (f'  {detail}' if detail and not ok else ''))
    required_ok = all(ok for level, _, ok, _ in checks if level == 'required')
    tests_ok = True
    if run_pytest and exp is not None:
        code, summary, failures = run_tests(cid)
        tests_ok = code == 0
        print(f'\n  tests: {summary}')
        for f in failures[:15]:
            print('    ' + f[:200])
    ready = required_ok and tests_ok
    if not ready:
        print('\nNOT READY')
    elif not run_pytest:
        print('\nSTATIC CHECKS PASS (tests not run; not a readiness verdict)')
    else:
        print('\nTECHNICAL CHECKS PASS (scientific fidelity still requires independent review)')
        if exp is not None and exp.maturity == 'draft':
            print(f"Set maturity = 'ready' in {Path(inspect.getsourcefile(type(exp))).name} only after the independent review.")
    return 0 if ready else 1
