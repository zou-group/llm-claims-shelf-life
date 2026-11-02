"""C143 similarity math and the offline embedding cache (pure numpy; no API calls).

The functions mirror the authors' analysis scripts (repo @ ff3b875,
src/model_calibration_analysis/main_analysis_within_model.py):

- ``within_pairs_mean``: ``get_pairwise_similarity`` (cosine, upper triangle k=1, so self-pairs EXCLUDED), averaged.
- ``paper_histogram``: ``np.histogram(means, bins=10, range=(0, 1))`` as percentages of ``len(means)``.

The embedding cache is a directory of ``.npz`` files (or a single ``.npz``), each holding ``sha256`` (str array, sha256 of
the exact UTF-8 response text) and ``embedding`` (float array, one row per text), optionally ``model`` (str). It is
written by experiments/run_minimal_setups/runs/added-b/scripts/embed_hivemind.py and only read here.
"""
import functools
import hashlib
from pathlib import Path

import numpy as np

# Paper bin labels, highest first (the order of Fig. 2 / App. Table within_model_similarity_full).
BIN_LABELS = ('0.9-1.0', '0.8-0.9', '0.7-0.8', '0.6-0.7', '0.5-0.6', '0.4-0.5', '0.3-0.4', '0.2-0.3', '0.1-0.2',
              '0.0-0.1')


def text_sha256(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _unit(E):
    E = np.asarray(E, dtype=np.float64)
    if E.ndim != 2:
        raise ValueError('embeddings must be a 2-D array (n, d)')
    norms = np.linalg.norm(E, axis=1)
    if np.any(norms == 0):
        raise ValueError('zero-norm embedding')
    return E / norms[:, None]


def cosine_matrix(A, B=None):
    A = _unit(A)
    B = A if B is None else _unit(B)
    return A @ B.T


def within_pairs_mean(E):
    """Mean cosine over the n(n-1)/2 distinct pairs (self-pairs excluded). None for n < 2."""
    E = np.asarray(E)
    if len(E) < 2:
        return None
    S = cosine_matrix(E)
    return float(S[np.triu_indices_from(S, k=1)].mean())


def with_self_pairs(pairs_mean, n):
    """Convert an off-diagonal mean over n samples to the self-pairs-included mean: 1/n + (1 - 1/n) * mean."""
    return 1.0 / n + (1.0 - 1.0 / n) * pairs_mean


def paper_histogram(means):
    """Percent of queries per paper bin, as the authors computed it: np.histogram(bins=10, range=(0, 1)), divided by
    len(means) (values outside [0, 1], e.g. a float overshoot 1.0000000002 of an all-identical pool, are not counted
    in any bin but stay in the denominator; reported as ``out_of_range``). Bins are [a, b) except the last, [0.9, 1.0]."""
    means = np.asarray([m for m in means if m is not None], dtype=np.float64)
    if len(means) == 0:
        return {'n': 0, 'percent': {b: None for b in BIN_LABELS}, 'out_of_range': 0}
    counts, _ = np.histogram(means, bins=10, range=(0, 1))
    pct = counts / len(means) * 100
    return {'n': int(len(means)), 'percent': {BIN_LABELS[9 - i]: float(pct[i]) for i in range(10)},
            'out_of_range': int(len(means) - counts.sum())}


# ------------------------------------------------------------------------------------------------ embedding cache
def _cache_files(path):
    path = Path(path)
    if path.is_file():
        return (path,)
    if path.is_dir():
        return tuple(sorted(path.glob('*.npz')))
    return ()


@functools.lru_cache(maxsize=8)
def _load_cache(files_and_mtimes, expected_model):
    vectors, conflicts, wrong_model = {}, 0, []
    for name, _mtime in files_and_mtimes:
        with np.load(name, allow_pickle=False) as z:
            if 'model' in z.files and expected_model is not None and str(z['model']) != expected_model:
                wrong_model.append(name)
                continue
            shas, emb = z['sha256'], np.asarray(z['embedding'], dtype=np.float32)
            if len(shas) != len(emb):
                raise ValueError(f'{name}: {len(shas)} hashes but {len(emb)} vectors')
            for sha, vec in zip(shas.tolist(), emb):
                if sha in vectors:
                    conflicts += int(not np.allclose(vectors[sha], vec, atol=1e-4))
                    continue
                vectors[sha] = vec
    return vectors, conflicts, tuple(wrong_model)


def load_embedding_cache(path, expected_model='text-embedding-3-small'):
    """{sha256: float32 vector} from every .npz under ``path``. Returns (vectors, info). A missing path gives an
    empty mapping and ``info['exists'] = False`` (the caller reports every response as missing an embedding).
    The first file (sorted by name) wins for a repeated hash; differing duplicates are counted as conflicts."""
    files = _cache_files(path)
    key = tuple((str(f), f.stat().st_mtime_ns) for f in files)
    vectors, conflicts, wrong = _load_cache(key, expected_model)
    dims = sorted({len(v) for v in vectors.values()})
    return vectors, {'path': str(path), 'exists': bool(files), 'files': len(files), 'vectors': len(vectors),
                     'dims': dims, 'duplicate_conflicts': conflicts, 'skipped_files_wrong_model': list(wrong)}


# --------------------------------------------------------------------------------------------- sampling noise (n)
def jackknife_pairs_se(E):
    """Leave-one-response-out jackknife SE of the mean over distinct pairs (a U-statistic of order 2). None for n < 3.
    Used to count queries whose per-query mean is within 2 SE of the threshold (20- vs 50-sample noise, audit m6)."""
    E = np.asarray(E)
    n = len(E)
    if n < 3:
        return None
    S = cosine_matrix(E)
    np.fill_diagonal(S, 0.0)
    total = S.sum() / 2
    loo = (total - S.sum(axis=1)) / ((n - 1) * (n - 2) / 2)
    return float(np.sqrt((n - 1) / n * ((loo - loo.mean()) ** 2).sum()))

