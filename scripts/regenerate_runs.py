#!/usr/bin/env python3
"""Regenerate the model outputs behind the post's 1,000 claim x model values.

Our saved outputs are not part of this repository. claims/run_manifest.csv records, for every plotted value, the
config, variant, model, policy, item limit and repeats of the run that produced it, with the hashes of its items and
requests. This script turns those rows into `shelflife.cli` commands.

    python scripts/regenerate_runs.py --claims C4 C11              # print `plan` commands (offline, free)
    python scripts/regenerate_runs.py --claims C4 --execute plan     # run the plans
    python scripts/regenerate_runs.py --claims C4 --execute run --ceiling 20   # paid: needs API keys

Offline plans of multi-step programs can stop early on empty placeholder replies and are reported as failed at the
end; that does not affect paid runs.

Rows marked same_protocol_today=False were generated with an earlier revision of the variant and then rescored, so
a new run uses today's protocol rather than byte-identical requests. Rows with origin "earlier pipeline" came from
the pre-shelflife harness; they are regenerated with the variant that reproduces them and its default settings.
After a run, compare each model's configuration.json (items_sha256, requests_sha256) with the manifest.

Twelve claims also read input files larger than 5 MB that are not in git; download them first (DOWNLOADS.md).
"""
import argparse
import csv
import shlex
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--claims', nargs='*', help='blog IDs (C1-C50) or config keys (C043); default: all')
    ap.add_argument('--models', nargs='*', help='model IDs (M015 ...); default: all in the manifest')
    ap.add_argument('--execute', choices=['plan', 'run'], help='run the commands instead of printing them')
    ap.add_argument('--ceiling', type=float, help='USD ceiling for --execute run (required for paid runs)')
    ap.add_argument('--out', default='runs/regenerated', help='run directories, relative to the repository root')
    ap.add_argument('--python', default='.venv/bin/python', help='Python with shelflife installed, relative to code/ (default .venv/bin/python)')
    a = ap.parse_args()
    if a.execute == 'run' and a.ceiling is None:
        ap.error('--execute run needs --ceiling')

    rows = list(csv.DictReader(open(ROOT / 'claims/run_manifest.csv')))
    want = set(a.claims or [])
    rows = [r for r in rows if (not want or r['blog_id'] in want or r['config'] in want)
            and (not a.models or r['model'] in a.models)]
    if not rows:
        sys.exit('no manifest rows match')

    groups = OrderedDict()   # one command per identical protocol; the runner writes one directory per model
    for r in rows:
        k = (r['blog_id'], r['config'], r['variant'], r['policy'], r['limit'], r['repeats'])
        groups.setdefault(k, []).append(r['model'])

    out = (ROOT / a.out).resolve()
    failed = []
    for (blog_id, config, variant, policy, limit, repeats), models in groups.items():
        cmd = [a.python, '-m', 'shelflife.cli', a.execute or 'plan', config, '--variant', variant, '--models', *models]
        if policy:
            cmd += ['--policy', policy]
        if repeats:
            cmd += ['--repeats', repeats]
        if (a.execute or 'plan') == 'plan':
            cmd += ['--limit', '1' if not limit or limit == 'all' else limit]   # plan prints a sample; cost scales
        else:
            if limit and limit != 'all':
                cmd += ['--limit', limit]
            run_dir = out / f'{blog_id}-{config}-{variant}'
            cmd += ['--run-dir', str(run_dir), '--ledger', str(out / 'ledger.sqlite3'), '--ceiling', str(a.ceiling)]
        if a.execute:
            print('+', shlex.join(cmd), flush=True)
            if subprocess.run(cmd, cwd=ROOT / 'code').returncode:
                failed.append(f'{blog_id} {config} {variant}')
                if a.execute == 'run':
                    break   # stop paid runs at the first failure
        else:
            print(f'(cd code && {shlex.join(cmd)})')
    if failed:
        # Offline plans of multi-step programs (dialogues, judges, agents) can stop early on empty placeholder
        # replies; the plan JSON says "Some paths are incomplete". Pass --responses-json to shelflife.cli plan for those.
        sys.exit('failed: ' + '; '.join(failed))


if __name__ == '__main__':
    main()
