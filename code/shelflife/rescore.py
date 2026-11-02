"""Offline, immutable scoring revisions of runner output.

Saved items and the configured model × item × repeat grid are authoritative. A
missing response is an explicit unscored row, including jobs interrupted before
the runner could write them. Original outputs, metrics and configuration are
never rewritten. Legacy formats continue to use Experiment.rescore().
"""
import hashlib
import json
import shutil
from pathlib import Path

from .ledger import now
from .replay import run_records
from .runner import PACKAGE, package_hash


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rescore_run(exp, run_dir, out_dir):
    """Write a new scoring artifact, returning its metrics and provenance.

    Both paths follow normal filesystem/CWD semantics (like ``run``). The output
    directory must not exist and must be outside the source run. The current
    variant must equal the generation variant; scoring changes belong in score /
    aggregate, not in a silently changed protocol.
    """
    run_dir, out_dir = Path(run_dir).resolve(), Path(out_dir).resolve()
    if out_dir == run_dir or run_dir in out_dir.parents or out_dir in run_dir.parents:
        raise ValueError('rescore output must be outside the original run and its ancestors')
    if out_dir.exists():
        raise FileExistsError(f'{out_dir} exists; choose a new scoring revision')
    config_path, items_path = run_dir / 'configuration.json', run_dir / 'items.jsonl'
    config = json.loads(config_path.read_text())
    if config.get('claim_id') != exp.claim_id or 'variant_fingerprint' not in config:
        raise ValueError('expected a shelflife.runner run of ' + exp.claim_id)
    variant = exp.variant(config['variant'])
    if variant.fingerprint != config['variant_fingerprint']:
        raise ValueError('variant changed since generation; restore the saved protocol before rescoring')
    items = [json.loads(line) for line in items_path.read_text().splitlines() if line.strip()]
    from .runner import _sha
    if _sha(items) != config['items_sha256']:
        raise ValueError('saved items do not match configuration items_sha256')
    ids = [item['id'] for item in items]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate saved item ids')
    models, repeats = config['models'], config['repeats']
    if (not isinstance(models, list) or not models or any(not isinstance(m, str) or not m for m in models)
            or len(set(models)) != len(models)):
        raise ValueError('saved models must be a nonempty list of unique model keys')
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError('saved repeats must be a positive integer')
    if config.get('n_items') != len(items):
        raise ValueError('saved item count differs from configuration n_items')
    expected = {(model, item_id, repeat) for model in models for item_id in ids for repeat in range(repeats)}
    saved = {}
    source_files = [config_path, items_path]
    from .replay import stream_files
    streams = stream_files(run_dir)
    source_files.extend(streams)
    source_hashes = {str(p.relative_to(run_dir)): _digest(p) for p in source_files}
    # Raw calls are not program outputs: never reinterpret a multi-turn call as
    # a complete item if the process died before writing responses.jsonl.
    for record in run_records(run_dir, streams=[p.name for p in streams]) if streams else []:
        if 'item_id' not in record or 'trace' not in record:
            raise ValueError('response stream contains a legacy/raw row; use Experiment.rescore for legacy data')
        key = (record['model'], record['item_id'], record.get('repeat', 0))
        if key not in expected:
            raise ValueError(f'response outside the saved experiment grid: {key}')
        saved[key] = record
    rows, metrics = [], {}
    for model in models:
        model_rows = []
        for item in items:
            for repeat in range(repeats):
                record = saved.get((model, item['id'], repeat))
                output = record.get('output') if record is not None else None
                status = record.get('status', 'call_error') if record is not None else 'not_submitted'
                score = None if output is None else exp.score(item, output, variant)
                row = {'item': item, 'output': output, 'score': score, 'model': model,
                       'repeat': repeat, 'status': status}
                model_rows.append(row)
                rows.append({'claim_id': exp.claim_id, 'variant': variant.name, 'item_id': item['id'],
                             'model': model, 'repeat': repeat, 'status': status, 'score': score,
                             'source_response_present': record is not None})
        metrics[model] = exp.aggregate(model_rows, variant)
    provenance = {'format_version': 1, 'created': now(), 'claim_id': exp.claim_id,
                  'source_run': str(run_dir), 'variant': variant.name,
                  'variant_fingerprint': variant.fingerprint,
                  'generation_package_sha256': config['package_sha256'],
                  'scoring_package_sha256': package_hash(exp),
                  'source_files': source_hashes,
                  'expected_rows': len(expected), 'saved_rows': len(saved),
                  'absent_rows': len(expected) - len(saved), 'network_calls': 0}
    # Evaluate and serialize everything before creating the destination, so a
    # scorer failure leaves no artifact that might be mistaken for complete.
    artifacts = {'provenance.json': json.dumps(provenance, indent=2, ensure_ascii=False),
                 'metrics.json': json.dumps(metrics, indent=2, ensure_ascii=False, default=str),
                 'scores.jsonl': ''.join(json.dumps(row, ensure_ascii=False, default=str) + '\n' for row in rows)}
    if (streams != stream_files(run_dir)
            or source_hashes != {str(p.relative_to(run_dir)): _digest(p) for p in source_files}):
        raise RuntimeError('source run changed during rescoring; retry after generation has stopped')
    out_dir.mkdir(parents=True, exist_ok=False)
    shutil.copytree(PACKAGE, out_dir / 'code_snapshot/shelflife',
                    ignore=shutil.ignore_patterns('__pycache__', 'data'))
    for name, content in artifacts.items():
        (out_dir / name).write_text(content)
    return {'metrics': metrics, 'provenance': provenance}
