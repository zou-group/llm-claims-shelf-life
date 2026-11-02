"""Explicit run registrations kept separately from experiment source.

A claim opts in with ``run_manifest = 'code/run-manifests/C123.json'``.
Files have ``version: 1``, ``claim_id``, ``prior_runs`` (PriorRun objects as JSON),
and ``excluded_runs`` (path -> reason). Loading never discovers or rewrites files.
``build_manifest`` is an explicit, metadata-only helper for creating a candidate.
"""
import dataclasses
import json
import os
from pathlib import Path, PurePosixPath

from .paths import PROJECT
from .registry import PriorRun

_SKIP_DIRS = {'.git', '.venv', '__pycache__', 'code_snapshot', 'node_modules', 'raw'}


def _relative_path(value, project, label):
    """Require canonical portable project-relative paths, including symlink containment."""
    if not isinstance(value, str) or not value or '\\' in value:
        raise ValueError(f'{label}: expected a canonical project-relative path')
    path = PurePosixPath(value)
    if path.is_absolute() or str(path) != value or any(p in {'.', '..'} for p in path.parts):
        raise ValueError(f'{label}: expected a canonical project-relative path: {value!r}')
    if value == '.' or not (project / value).resolve().is_relative_to(project.resolve()):
        raise ValueError(f'{label}: path escapes the project: {value!r}')
    return value


def _validate(data, exp, project):
    if not isinstance(data, dict):
        raise ValueError('run manifest must be a JSON object')
    unknown = set(data) - {'version', 'claim_id', 'prior_runs', 'excluded_runs'}
    if unknown:
        raise ValueError(f'unknown run manifest fields: {sorted(unknown)}')
    if type(data.get('version')) is not int or data['version'] != 1:
        raise ValueError('run manifest version must be 1')
    if data.get('claim_id') != exp.claim_id:
        raise ValueError(f'run manifest claim_id must be {exp.claim_id!r}')
    entries = data.get('prior_runs', [])
    excluded = data.get('excluded_runs', {})
    if not isinstance(entries, list) or not isinstance(excluded, dict):
        raise ValueError('prior_runs must be a list and excluded_runs must be an object')
    fields = {f.name for f in dataclasses.fields(PriorRun)}
    runs, paths = [], set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) - fields:
            raise ValueError(f'prior run has unknown fields or is not an object: {entry!r}')
        path = _relative_path(entry.get('path'), project, 'prior run path')
        if path in paths:
            raise ValueError(f'duplicate prior run path: {path}')
        variant = entry.get('variant')
        if not isinstance(variant, str) or variant not in exp.variants:
            raise ValueError(f'{path}: unknown variant {variant!r}')
        if 'description' in entry and not isinstance(entry['description'], str):
            raise ValueError(f'{path}: description must be a string')
        if entry.get('cases') is not None:
            _relative_path(entry['cases'], project, f'{path}: cases')
        deviations = entry.get('known_deviations', [])
        if not isinstance(deviations, list) or any(not isinstance(s, str) or not s.strip() for s in deviations):
            raise ValueError(f'{path}: known_deviations must be a list of nonempty strings')
        if 'no_requests' in entry and type(entry['no_requests']) is not bool:
            raise ValueError(f'{path}: no_requests must be a boolean')
        runs.append(PriorRun(**{**entry, 'known_deviations': tuple(deviations)}))
        paths.add(path)
    for path, reason in excluded.items():
        _relative_path(path, project, 'excluded run path')
        if not isinstance(reason, str) or len(reason.strip()) <= 10:
            raise ValueError(f'exclusion of {path} needs a descriptive reason (more than 10 characters)')
    _reject_overlap(runs, excluded)
    return tuple(runs), dict(excluded)


def _covers(parent, child):
    return child == parent or child.startswith(parent + '/')


def _reject_overlap(runs, excluded):
    for i, run in enumerate(runs):
        run_path = run.path.rstrip('/')
        for other in runs[i + 1:]:
            other_path = other.path.rstrip('/')
            if run.variant != other.variant and (_covers(run_path, other_path) or _covers(other_path, run_path)):
                raise ValueError(f'overlapping registrations have different variants: {run.path} / {other.path}')
        for path in excluded:
            path = path.rstrip('/')
            if _covers(path, run_path) or _covers(run_path, path):
                raise ValueError(f'run is both registered and excluded: {run.path} / {path}')


def load_manifest(exp, path=None, *, project=PROJECT):
    """Load and validate one explicit manifest. A missing declared file is an error."""
    project = Path(project)
    path = exp.run_manifest if path is None else path
    if path is None:
        return (), {}
    path = _relative_path(path, project, 'run manifest')
    return _validate(json.loads((project / path).read_text()), exp, project)


def merged_runs(exp, *, project=PROJECT):
    """Merge opt-in metadata with unchanged legacy declarations; reject ambiguity.

    Claims that have not opted in retain their historical declarations exactly.
    Identical manifest/legacy entries are deduplicated; conflicting entries fail.
    """
    if exp.run_manifest is None:
        return tuple(exp.prior_runs), dict(exp.excluded_runs)
    extra, exclusions = load_manifest(exp, project=project)
    by_path = {}
    for prior in exp.prior_runs:
        path = prior.path.rstrip('/')
        if path in by_path and dataclasses.replace(by_path[path], path=path) != dataclasses.replace(prior, path=path):
            raise ValueError(f'conflicting legacy registration for {path}')
        by_path.setdefault(path, prior)
    excluded = {p.rstrip('/'): reason for p, reason in exp.excluded_runs.items()}
    for run in extra:
        old = by_path.get(run.path)
        if old is not None and dataclasses.replace(old, path=old.path.rstrip('/')) != run:
            raise ValueError(f'conflicting registration for {run.path}')
        by_path.setdefault(run.path, run)
    for path, reason in exclusions.items():
        if path in excluded and excluded[path] != reason:
            raise ValueError(f'conflicting exclusion for {path}')
        excluded[path] = reason
    runs = tuple(by_path.values())
    _reject_overlap(runs, excluded)
    return runs, excluded


def manifest_saved_run_dirs(exp, *, project=PROJECT):
    """Saved-call directories inside explicitly registered manifest paths.

    Supplement the historical C###-in-path coverage scan for new runs whose
    directory names need not encode the claim ID. Inspect filenames only.
    """
    project = Path(project)
    runs, excluded = load_manifest(exp, project=project)
    found = set()
    for relative in {r.path for r in runs} | set(excluded):
        for directory, children, files in os.walk(project / relative, followlinks=False):
            path = Path(directory)
            has_raw = 'raw' in children and any(p.is_file() for p in (path / 'raw').glob('*.json'))
            has_responses = any(f.startswith('responses') and f.endswith('.jsonl') for f in files)
            children[:] = [d for d in children if d not in _SKIP_DIRS and not (path / d).is_symlink()]
            if has_raw or has_responses:
                found.add(path.relative_to(project).as_posix())
    return sorted(found)


def build_manifest(exp, roots, *, project=PROJECT):
    """Return a candidate JSON manifest from runner configurations under explicit roots.

    ``roots`` are project-relative directories or absolute directories inside the
    project. Only configuration.json contents and raw file presence are
    inspected. No API calls, dataset loading, source edits or implicit discovery.
    Existing inline registrations/exclusions are respected. A declared manifest's
    entries and hand-written annotations are preserved, with new discoveries added
    to the returned candidate. Nothing is written or overwritten. Review before
    replacing the declared file. Legacy runs without configurations need manual
    entries. Each per-model runner directory receives its own registration.
    Missing raw files are flagged for review, never treated as proof that no calls
    were submitted. Set ``no_requests`` manually only with evidence of no calls.
    """
    project = Path(project).resolve()
    if not roots or isinstance(roots, (str, Path)):
        raise ValueError('provide at least one explicit run root')
    configs = set()
    for root in roots:
        root = Path(root)
        root = (root if root.is_absolute() else project / root).resolve()
        if not root.is_relative_to(project) or root == project:
            raise ValueError(f'run root must be a directory strictly inside the project: {root}')
        if not root.is_dir():
            raise FileNotFoundError(f'run root is not a directory: {root}')
        for directory, children, files in os.walk(root, followlinks=False):
            children[:] = sorted(d for d in children if d not in _SKIP_DIRS
                                 and not Path(directory, d).is_symlink())
            if 'configuration.json' in files:
                config = Path(directory, 'configuration.json')
                if not config.resolve().is_relative_to(root):
                    raise ValueError(f'configuration escapes the supplied run root: {config}')
                configs.add(config)
    existing, explicit_exclusions = load_manifest(exp, project=project)
    all_runs, all_exclusions = merged_runs(exp, project=project)
    entries = [dataclasses.asdict(p) for p in existing]
    registered = {p.path.rstrip('/'): p for p in all_runs}
    excluded = {p.rstrip('/') for p in all_exclusions}
    for path in sorted(configs):
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            raise ValueError(f'{path}: configuration must be an object')
        if data.get('claim_id') != exp.claim_id:
            continue
        relative = path.parent.relative_to(project).as_posix()
        if any(_covers(p, relative) for p in excluded):
            continue
        variant = data.get('variant')
        if not isinstance(variant, str) or variant not in exp.variants:
            raise ValueError(f'{relative}: unknown variant {variant!r}')
        covered = [p for p in registered if _covers(p, relative)]
        if covered:
            if any(registered[p].variant != variant for p in covered):
                raise ValueError(f'{relative}: configuration conflicts with registered variant')
            continue
        models = data.get('models', [])
        if not isinstance(models, list) or any(not isinstance(m, str) for m in models):
            raise ValueError(f'{relative}: models must be a list of subject-model keys')
        has_raw = any(p.is_file() for p in (path.parent / 'raw').glob('*.json'))
        entries.append(dataclasses.asdict(PriorRun(
            path=relative, variant=variant,
            description=('Runner configuration' + (f"; subject models: {', '.join(models)}" if models else '')
                         + ('; no raw files found: investigate missing archives or verify no calls were submitted'
                            if not has_raw else '')))))
    # asdict preserves tuples; JSON serialization naturally turns them into lists.
    data = json.loads(json.dumps({'version': 1, 'claim_id': exp.claim_id,
                                 'prior_runs': entries, 'excluded_runs': explicit_exclusions}))
    _validate(data, exp, project)
    return data


SKIP_DIRS = {'.venv', '.git', 'code_snapshot', '__pycache__', 'node_modules'}


def claim_run_dirs(claim_id, root=None):
    """Directories under experiments/ that hold saved API calls for a claim: a non-empty raw/ or responses*.jsonl,
    with the claim id in the path. Excludes snapshots, venvs and the official-code clones."""
    import os
    import re
    from .paths import EXPERIMENTS
    root = EXPERIMENTS if root is None else Path(root)
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        rel = Path(dirpath).relative_to(root.parent)
        parts = rel.parts
        if 'claims' in parts and 'code' in parts:   # official repository clones
            dirnames[:] = []
            continue
        if claim_id not in re.findall(r'C\d{3}', str(rel)):
            continue
        has_raw = 'raw' in dirnames and any(Path(dirpath, 'raw').glob('*.json'))
        has_resp = any(f.startswith('responses') and f.endswith('.jsonl') for f in filenames)
        if has_raw or has_resp:
            out.append(str(rel))
    return sorted(out)
