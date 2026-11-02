"""Filesystem anchors. Experiments are read-only inputs to this package."""
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
PROJECT = CODE.parent
EXPERIMENTS = PROJECT / 'experiments'
INITIAL = EXPERIMENTS / 'initial-reproduction'
HELPERS = Path(__file__).resolve().parent / 'experiments' / 'helpers'   # scorer helpers for C010, C014, C077
DATA = Path(__file__).resolve().parent / 'data'


def claim_dir(claim_id):
    """initial-reproduction/claims/<ID>_<slug>/ (holds claim.json and the official code clone)."""
    matches = sorted((INITIAL / 'claims').glob(f'{claim_id}_*'))
    if len(matches) != 1:
        raise FileNotFoundError(f'expected one claim directory for {claim_id}, found {matches}')
    return matches[0]


def source_repo(claim_id, name=None):
    """Official repository clone for a claim (initial-reproduction/claims/<ID>_*/code/<repo>)."""
    repos = sorted(p for p in (claim_dir(claim_id) / 'code').iterdir() if p.is_dir())
    if name is not None:
        repos = [p for p in repos if p.name == name]
    if len(repos) != 1:
        raise FileNotFoundError(f'expected one source repository for {claim_id} ({name}), found {repos}')
    return repos[0]
