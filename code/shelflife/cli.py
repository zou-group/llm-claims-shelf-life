"""Command line: new, check, list, show, plan, run, rescore, index-runs, targets.

  python -m shelflife.cli new C101 [--clone auto|URL] [--repo PATH] [--slug name]   # scaffold a draft claim
  python -m shelflife.cli check C101                                              # readiness checklist + tests

  python -m shelflife.cli list
  python -m shelflife.cli show C043 [--variant faireval_vanilla]
  python -m shelflife.cli plan C043 --models M001 M206 --limit 3          # offline scenario traversal and costs
  python -m shelflife.cli plan C043 --models panel20 --limit 3            # panel names: panel20, all
  python -m shelflife.cli run C043 --variant faireval_vanilla --models panel20 --run-dir RUN --ledger L --ceiling 20
  python -m shelflife.cli rescore C043 --run-dir RUN --out-dir NEW        # offline immutable scoring revision
  python -m shelflife.cli index-runs C043 --roots RUN_ROOT --out NEW.json # reviewable run metadata
  python -m shelflife.cli targets [C043 ...]                              # recompute paper targets offline
"""
import argparse
import json
import sys
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

from . import registry


def cmd_new(a):
    from .scaffold import scaffold
    out = scaffold(a.claim, slug=a.slug, repo=a.repo, clone_url=a.clone, commit=a.commit, claim_json=a.claim_json,
                   force=a.force)
    print(json.dumps(out, indent=2))
    print(f"\nNext: fill the TODOs (see code/docs/INTEGRATING.md), then "
          f"`python -m shelflife.cli check {a.claim}`.")


def cmd_check(a):
    from .check import main as check_main
    sys.exit(check_main(a.claim, run_pytest=not a.no_tests, verbose=a.verbose))


def cmd_list(a):
    for cid in registry.list_experiments():
        exp = registry.get(cid)
        print(f'{cid}\t{exp.maturity}\t{exp.default_variant}\t{len(exp.variants)} variants\t{exp.title}')
    for name, err in registry.IMPORT_ERRORS.items():
        print(f'IMPORT ERROR {name}: {err}')


def cmd_show(a):
    from dataclasses import asdict
    exp = registry.get(a.claim)
    m = exp.manifest()
    # Show resolved metadata without making it part of the execution fingerprint.
    m.update(run_manifest=exp.run_manifest, prior_runs=[asdict(p) for p in exp.all_prior_runs()],
             excluded_runs=exp.all_excluded_runs())
    if a.variant:
        m['variants'] = {a.variant: m['variants'][a.variant]}
    print(json.dumps(m, indent=2, ensure_ascii=False, default=str))


def _panel_args(a):
    """Expand panel-set names in --models; a run named by a panel set defaults to the panel_v2 policy."""
    from . import models as M
    named = [m for m in a.models if m in M.PANEL_SETS]
    a.models = M.panel_models(a.models)
    if a.policy is None and named and all(n in ('all', 'panel20') for n in named):
        a.policy = 'panel_v2'
    return a


def cmd_plan(a):
    from .planning import plan_program
    if a.limit is not None and a.limit < 0:
        raise ValueError('limit must be nonnegative')
    if a.show < 0:
        raise ValueError('show must be nonnegative')
    a = _panel_args(a)
    exp = registry.get(a.claim)
    variant = exp.variant(a.variant)
    responder = None
    if a.responses_json:
        responses = json.loads(Path(a.responses_json).read_text())
        if not isinstance(responses, dict):
            raise ValueError('responses JSON must map step names to synthetic response text or completion summaries')
        def responder(request):
            step = request['step']
            if step in responses:
                return responses[step]
            if '*' in responses:
                return responses['*']
            raise ValueError(f'no synthetic response for step {step!r}; add it or an explicit "*" default')
    report = plan_program(exp.program(variant), exp.items(variant, a.limit), a.models,
                          repeats=variant.repeats if a.repeats is None else a.repeats, policy=a.policy,
                          responder=responder, max_calls_per_item=a.max_calls_per_item)
    if a.out:
        with Path(a.out).open('x') as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
    for call in report['calls'][:a.show]:
        print(json.dumps(call, ensure_ascii=False))
    print(json.dumps({k: report[k] for k in ('scope', 'response_source', 'totals', 'warnings')}, indent=2))
    failures = [path for path in report['paths'] if path['status'] != 'complete']
    if failures:
        for failure in failures[:5]:
            print(json.dumps(failure), file=sys.stderr)
        raise SystemExit(1)


def cmd_rescore(a):
    from .rescore import rescore_run
    out = rescore_run(registry.get(a.claim), a.run_dir, a.out_dir)
    print(json.dumps(out, indent=2, ensure_ascii=False, default=str))


def cmd_index_runs(a):
    from .run_manifest import build_manifest
    manifest = build_manifest(registry.get(a.claim), [Path(root).resolve() for root in a.roots])
    with Path(a.out).open('x') as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
        fh.write('\n')
    print(f'Wrote {a.out}; review it before selecting it as the experiment run_manifest.')


def cmd_run(a):
    from .runner import run
    a = _panel_args(a)
    exp = registry.get(a.claim)
    out = run(exp, a.variant, a.models, a.run_dir, ledger=a.ledger, ceiling_usd=a.ceiling,
              task_budget_usd=a.task_budget, repeats=a.repeats, workers=a.workers, limit=a.limit, policy=a.policy,
              retry_errors=a.retry_errors, breaker_threshold=a.breaker_threshold)
    print(json.dumps(out['metrics'], indent=2, default=str))


def cmd_targets(a):
    claims = a.claims or registry.list_experiments()
    rows = []
    for cid in claims:
        exp = registry.get(cid)
        for t in exp.targets:
            row = {'claim': cid, 'target': t.name, 'paper': t.value, 'tolerance': t.tolerance, 'check': t.check,
                   'where': t.where}
            if t.compute is not None and t.check in registry.OFFLINE_CHECKS:
                try:
                    value = t.compute(exp)
                    row.update(reproduced=value, ok=abs(value - t.value) <= t.tolerance)
                except Exception as e:
                    row.update(error=f'{type(e).__name__}: {e}')
            rows.append(row)
            print(json.dumps(row, default=str))





def main(argv=None):
    p = argparse.ArgumentParser(prog='shelflife')
    sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('new', help='scaffold a draft claim module, tests and findings file')
    s.add_argument('claim'); s.add_argument('--slug'); s.add_argument('--repo', help='existing local clone')
    s.add_argument('--clone', help="git URL to clone into shelflife/data/sources, or 'auto' to use the claim record's repo_url")
    s.add_argument('--commit'); s.add_argument('--claim-json', help='claim record (JSON file)')
    s.add_argument('--force', action='store_true'); s.set_defaults(f=cmd_new)
    s = sub.add_parser('check', help='readiness checklist and the claim test selection')
    s.add_argument('claim'); s.add_argument('--no-tests', action='store_true'); s.add_argument('--verbose', action='store_true')
    s.set_defaults(f=cmd_check)
    sub.add_parser('list').set_defaults(f=cmd_list)
    s = sub.add_parser('show'); s.add_argument('claim'); s.add_argument('--variant'); s.set_defaults(f=cmd_show)
    s = sub.add_parser('plan'); s.add_argument('claim'); s.add_argument('--variant'); s.add_argument('--models', nargs='+', required=True)
    s.add_argument('--limit', type=int, default=3); s.add_argument('--policy'); s.add_argument('--show', type=int, default=2)
    s.add_argument('--repeats', type=int, help='defaults to the variant repeat count')
    s.add_argument('--responses-json', help='offline scenario: JSON map of named steps to replies; optional "*" default')
    s.add_argument('--max-calls-per-item', type=int, default=1000, help='stop unbounded offline program loops')
    s.add_argument('--out', help='save full offline plan to a new JSON file')
    s.set_defaults(f=cmd_plan)
    s = sub.add_parser('rescore', help='score saved runner outputs into a new artifact, offline')
    s.add_argument('claim'); s.add_argument('--run-dir', required=True); s.add_argument('--out-dir', required=True)
    s.set_defaults(f=cmd_rescore)
    s = sub.add_parser('index-runs', help='build a reviewable manifest from explicit runner roots, offline')
    s.add_argument('claim'); s.add_argument('--roots', nargs='+', required=True); s.add_argument('--out', required=True)
    s.set_defaults(f=cmd_index_runs)
    s = sub.add_parser('run'); s.add_argument('claim'); s.add_argument('--variant'); s.add_argument('--models', nargs='+', required=True)
    s.add_argument('--run-dir', required=True); s.add_argument('--ledger', required=True); s.add_argument('--ceiling', type=float, required=True)
    s.add_argument('--task-budget', type=float); s.add_argument('--repeats', type=int); s.add_argument('--workers', type=int, default=8)
    s.add_argument('--limit', type=int); s.add_argument('--policy')
    s.add_argument('--retry-errors', action='store_true', help='resubmit cached transient errors (429/5xx/transport); '
                   'the failed record is archived as raw_retried/<key>.errN.json and keeps its ledger charge')
    s.add_argument('--breaker-threshold', type=int, default=3, help='transport failures within 60 s that open the '
                   'sticky circuit (default 3)')
    s.set_defaults(f=cmd_run)
    s = sub.add_parser('targets'); s.add_argument('claims', nargs='*'); s.set_defaults(f=cmd_targets)
    a = p.parse_args(argv)
    try:
        a.f(a)
    except FileNotFoundError as e:
        if e.filename and _is_large_input(e.filename):
            print(f'Missing large input file: {e.filename}\n'
                  'It is not stored in git; download it as described in DOWNLOADS.md.', file=sys.stderr)
            return 1
        raise


def _is_large_input(filename):
    """True if the file is one of the inputs listed in scripts/inputs.json."""
    from .paths import PROJECT
    manifest = PROJECT / 'scripts' / 'inputs.json'
    if not manifest.exists():
        return False
    try:
        rel = str(Path(filename).resolve().relative_to(PROJECT.resolve()))
    except ValueError:
        return False
    return rel in {e['path'] for e in json.loads(manifest.read_text())['files']}


if __name__ == '__main__':
    sys.exit(main())
