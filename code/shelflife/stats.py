"""Statistics shared across claims (project METHODOLOGY.md conventions).

Repeated outputs for one item are clustered observations: resample items (clusters), not calls. A zero observed
error rate is bounded with the one-sided exact upper limit, never reported as zero.
"""
import math
import random
from collections import defaultdict

from scipy import stats as _st


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (float('nan'), float('nan'))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def clopper_pearson(k, n, alpha=0.05):
    if n == 0:
        return (float('nan'), float('nan'))
    lo = 0.0 if k == 0 else _st.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else _st.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return (float(lo), float(hi))


def zero_event_upper(n, alpha=0.05):
    """One-sided (1-alpha) upper bound on a rate when 0 of n independent clusters show the event: 1 - alpha**(1/n)."""
    return 1 - alpha ** (1 / n) if n else 1.0


def cluster_bootstrap(rows, cluster, statistic, n_boot=2000, seed=0, alpha=0.05):
    """Percentile CI resampling clusters with replacement.

    rows: list; cluster(row) -> hashable cluster id; statistic(list_of_rows) -> float.
    Returns (point, lo, hi).
    """
    groups = defaultdict(list)
    for r in rows:
        groups[cluster(r)].append(r)
    keys = sorted(groups, key=str)
    point = statistic(rows)
    if not keys:
        return point, float('nan'), float('nan')
    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        sample = []
        for k in (keys[rng.randrange(len(keys))] for _ in keys):
            sample.extend(groups[k])
        value = statistic(sample)
        if value is not None and not (isinstance(value, float) and math.isnan(value)):
            draws.append(value)
    if not draws:
        return point, float('nan'), float('nan')
    draws.sort()
    lo = draws[int(math.floor(alpha / 2 * (len(draws) - 1)))]
    hi = draws[int(math.ceil((1 - alpha / 2) * (len(draws) - 1)))]
    return point, lo, hi


def bounds_with_missing(known_success, known_failure, unresolved):
    """Lower/upper bounds on a success rate when some outputs are unusable (METHODOLOGY.md 'Uncertainty and absence')."""
    n = known_success + known_failure + unresolved
    if n == 0:
        return (float('nan'), float('nan'))
    return (known_success / n, (known_success + unresolved) / n)


def rate_summary(rows, success, cluster=lambda r: r['item']['id'], usable=lambda r: r.get('score') is not None,
                 n_boot=2000, seed=0):
    """The standard binary-rate aggregate in one call.

    rows: runner rows ({'item','output','score',...}); success(row) -> bool for usable rows.
    Returns n (all intended), scored, unscored, k, rate (scored denominator), ci95 (cluster bootstrap over items),
    and bounds treating every unscored row as failure / success. Use the paper's denominator: if the paper counts
    failed outputs as wrong, pass usable=lambda r: True and make success() return False for them.
    """
    scored = [r for r in rows if usable(r)]
    k = sum(bool(success(r)) for r in scored)
    rate = lambda rs: sum(bool(success(r)) for r in rs) / len(rs) if rs else float('nan')
    _, lo, hi = cluster_bootstrap(scored, cluster, rate, n_boot=n_boot, seed=seed)
    return {'n': len(rows), 'scored': len(scored), 'unscored': len(rows) - len(scored), 'k': k,
            'rate': k / len(scored) if scored else None, 'ci95': [lo, hi],
            'bounds_with_missing': list(bounds_with_missing(k, len(scored) - k, len(rows) - len(scored)))}
