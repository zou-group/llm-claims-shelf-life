"""Parallel runner: (model x item x repeat) jobs through a registered experiment's program.

Run directory layout
  configuration.json   experiment, variant manifest + fingerprint, models, repeats, items hash, code hash, policy
  code_snapshot/       shelflife package source and the experiment module at run time
  items.jsonl          the dataset rows used
  raw/<key>.json       one file per API call (request, adaptations, full response, cost)
  responses.jsonl      one row per (model, item, repeat): output, score, call trace
  responses_resume_*.jsonl  new response stream for each resume; earlier evidence is never truncated
  metrics.json         experiment.aggregate per model
  completion.json      counts and cost totals

Resume is deterministic: rerunning the same command reuses every finished call from the ledger. A run directory is
bound to its configuration; any change in variant, items, models or package code requires a new directory.
"""
import hashlib
import inspect
import json
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .ledger import LedgerClient, now
from .program import LM, CallFailed

PACKAGE = Path(__file__).resolve().parent


def _sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def package_hash(exp=None):
    """Core package files plus this experiment's own module (and its private _cNNN helpers).

    Other claims' modules are excluded so editing one claim never blocks resuming another claim's run.
    """
    h = hashlib.sha256()
    own = set()
    if exp is not None:
        module_file = Path(inspect.getsourcefile(type(exp))).resolve()
        own.add(module_file)
        prefix = module_file.stem.split('_')[0]
        own.update(p.resolve() for p in module_file.parent.glob(f'_{prefix}_*.py'))
    for f in sorted(PACKAGE.rglob('*.py')):
        if '__pycache__' in f.parts:
            continue
        if f.parent.name == 'experiments' and f.name != '__init__.py' and f.resolve() not in own:
            continue
        h.update(str(f.relative_to(PACKAGE)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def requests_hash(program, items):
    """Hash of every first request the program would send (covers prompts built by the authors' own code)."""
    previews = [program.preview(x) for x in items]
    params = [program.params_for(x) if hasattr(program, 'params_for') else None for x in items]
    return _sha([previews, params])


def _open_response_stream(run_dir):
    """Exclusive creation preserves evidence even if a resumed program crashes."""
    path = run_dir / 'responses.jsonl'
    index = 0
    while True:
        try:
            return path.open('x')
        except FileExistsError:
            index += 1
            path = run_dir / f'responses_resume_{index:06d}.jsonl'


def run_one(exp, variant, program, client, model, item, repeat):
    lm = LM(client, model, item['id'], repeat, case_extra=getattr(program, 'case_extra', None))
    row = {'claim_id': exp.claim_id, 'variant': variant.name, 'model': model, 'item_id': item['id'], 'repeat': repeat}
    try:
        output = program.forward(item, lm)
        row['status'] = 'complete'
    except CallFailed as e:
        output = None
        row['status'] = 'not_submitted' if e.completion.status == 'not_submitted' else 'call_error'
        row['error'] = str(e)
    row['output'] = output
    if isinstance(output, dict) and output.get('status') is not None:
        row['output_status'] = output['status']   # protocol-level outcome reported by the program itself
    row['trace'] = [c.summary() for c in lm.trace]
    # A program may catch CallFailed to preserve earlier outputs. "complete"
    # then describes Python execution, not successful calls or a valid metric.
    row['calls_complete'] = all(c.usable for c in lm.trace)
    row['charged_usd'] = sum(c.charged_usd or 0 for c in lm.trace)
    if output is not None:
        try:
            row['score'] = exp.score(item, output, variant)
        except Exception as e:  # scoring bugs must be visible, never silently dropped
            row['scoring_error'] = f'{type(e).__name__}: {e}'
    return row


def preflight(program, models, client):
    """Fail before any paid call: every model under test and every auxiliary model the program declares
    (``program.auxiliary_models``: judges, graders, coders, reformulators) must be routable by a policy, and the
    program's own ``validate(models)`` hook (if any) must pass."""
    problems = []
    for m in list(models) + list(getattr(program, 'auxiliary_models', ()) or ()):
        try:
            client.policy(m)
        except KeyError as e:
            problems.append(str(e))
    if hasattr(program, 'validate'):
        try:
            program.validate(list(models))
        except Exception as e:
            problems.append(f'{type(program).__name__}.validate: {e}')
    if problems:
        raise RuntimeError('preflight failed, nothing submitted: ' + '; '.join(problems))


def run(exp, variant, models, run_dir, *, ledger, ceiling_usd, task_budget_usd=None, repeats=None, workers=8,
        limit=None, policy=None, transport=None, items=None, progress_every=50, retry_errors=False, breaker_threshold=3):
    """Execute a registered experiment. Returns {'rows': [...], 'metrics': {...}}.

    variant: Variant or its name. models: model keys accepted by the chosen policy (or each model's default).
    transport: inject shelflife.ledger.MockTransport for offline tests.
    """
    variant = exp.variant(variant) if isinstance(variant, str) or variant is None else variant
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    repeats = variant.repeats if repeats is None else repeats
    for name, value in (('repeats', repeats), ('workers', workers)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 0):
        raise ValueError('limit must be a nonnegative integer or None')
    models = list(models)
    if not models or len(set(models)) != len(models):
        raise ValueError('models must be nonempty with no duplicate keys')
    items = exp.items(variant, limit) if items is None else items
    if len({x['id'] for x in items}) != len(items):
        raise ValueError('duplicate item ids')
    program = exp.program(variant)
    config = {'claim_id': exp.claim_id, 'experiment': type(exp).__qualname__, 'variant': variant.name,
              'variant_fingerprint': variant.fingerprint, 'variant_manifest': variant.manifest(),
              'faithful': variant.faithful, 'models': list(models), 'repeats': repeats, 'limit': limit,
              'items_sha256': _sha(items), 'n_items': len(items), 'requests_sha256': requests_hash(program, items),
              'package_sha256': package_hash(exp),
              'cache_schema': 'subject_v2',
              'policy': policy if isinstance(policy, (str, type(None))) else policy.name,
              'program': type(program).__qualname__}
    path = run_dir / 'configuration.json'
    if path.exists():
        previous = json.loads(path.read_text())
        old = {k: v for k, v in previous.items() if k not in {'created', 'ceiling_usd', 'task_budget_usd'}}
        if old != config:
            diff = sorted(k for k in set(old) | set(config) if old.get(k) != config.get(k))
            raise RuntimeError(f'configuration changed ({diff}); use a new run directory for generation. '
                               'For score/aggregate edits, use shelflife rescore with a new --out-dir; '
                               'first-request parity alone cannot prove a multi-step protocol unchanged.')
    else:
        path.write_text(json.dumps({**config, 'created': now(), 'ceiling_usd': ceiling_usd,
                                    'task_budget_usd': task_budget_usd}, indent=2, ensure_ascii=False))
    snap = run_dir / 'code_snapshot'
    if not snap.exists():
        shutil.copytree(PACKAGE, snap / 'shelflife', ignore=shutil.ignore_patterns('__pycache__', 'data'))
        module_file = inspect.getsourcefile(type(exp))
        shutil.copy2(module_file, snap / Path(module_file).name)
    items_path = run_dir / 'items.jsonl'
    if items_path.exists():
        saved_items = [json.loads(line) for line in items_path.read_text().splitlines() if line.strip()]
        if _sha(saved_items) != config['items_sha256']:
            raise RuntimeError('saved items changed; refusing to overwrite existing evidence')
    else:
        items_path.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in items))

    client = LedgerClient(run_dir, ledger, ceiling_usd, task=exp.claim_id, task_budget_usd=task_budget_usd,
                          transport=transport, policy=policy, retry_errors=retry_errors,
                          breaker_threshold=breaker_threshold, cache_schema='subject_v2')
    preflight(program, models, client)
    jobs = [(m, x, r) for r in range(repeats) for x in items for m in models]  # interleave models on matched items
    rows, lock, done = [], threading.Lock(), 0
    with ThreadPoolExecutor(max_workers=workers) as pool, _open_response_stream(run_dir) as fh:
        futures = [pool.submit(run_one, exp, variant, program, client, m, x, r) for m, x, r in jobs]
        for f in as_completed(futures):
            row = f.result()
            with lock:
                rows.append(row)
                fh.write(json.dumps(row, ensure_ascii=False) + '\n')
                fh.flush()
                done += 1
                if progress_every and done % progress_every == 0:
                    print(json.dumps({'completed': done, 'total': len(jobs)}), flush=True)
    by_id = {x['id']: x for x in items}
    metrics = {}
    for m in models:
        model_rows = [{'item': by_id[r['item_id']], 'output': r['output'], 'score': r.get('score'), 'model': m,
                       'repeat': r['repeat'], 'status': r['status']} for r in rows if r['model'] == m]
        metrics[m] = exp.aggregate(model_rows, variant)
    (run_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False, default=str))
    status_counts, output_status_counts = {}, {}
    for r in rows:
        status_counts[r['status']] = status_counts.get(r['status'], 0) + 1
        if 'output_status' in r:
            key = str(r['output_status'])
            output_status_counts[key] = output_status_counts.get(key, 0) + 1
    (run_dir / 'completion.json').write_text(json.dumps({
        'finished': now(), 'expected': len(jobs), 'completed': len(rows), 'status_counts': status_counts, 'output_status_counts': output_status_counts,
        'rows_with_failed_calls': sum(not r['calls_complete'] for r in rows),
        'rows_with_scoring_errors': sum('scoring_error' in r for r in rows),
        'response_stream': Path(fh.name).name,
        'charged_usd_this_run': sum(r['charged_usd'] for r in rows), 'ledger_totals': client.totals()}, indent=2))
    return {'rows': rows, 'metrics': metrics}


def plan(exp, variant, models, *, limit=None, policy=None):
    """Offline request plan for single-turn programs: payloads, adaptations and the conservative cost bound."""
    import tempfile
    variant = exp.variant(variant) if isinstance(variant, str) or variant is None else variant
    cases = exp.cases(variant, limit)
    with tempfile.TemporaryDirectory() as tmp:
        client = LedgerClient(Path(tmp) / 'run', Path(tmp) / 'ledger.sqlite3', 1e9, policy=policy)
        out = []
        for m in models:
            for case in cases:
                payload, adaptations, bound = client.plan(m, case)
                out.append({'model': m, 'case_id': case['id'], 'payload': payload, 'adaptations': adaptations,
                            'bound_usd': bound})
    return out
