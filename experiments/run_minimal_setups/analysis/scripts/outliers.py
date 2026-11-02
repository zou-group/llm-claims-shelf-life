"""Outlier mask: which panel models each claim's fit leaves out (step 2 of the analysis).

For each claim, a Theil-Sen line of the normalized effect e against release date is fitted; points whose residual
exceeds 3 x max(1.4826 MAD, 0.1) are flagged, at most 20% of the points, largest first. Flagged points are left out
of the anchored fits (fits_anchored.py) but still plotted. TOY-01 (strawberry) keeps every reliable point.

Also provides the date helpers fits_anchored.py uses (t_of, date_of, death_time, DEAD_E).

Usage: python experiments/run_minimal_setups/analysis/scripts/outliers.py
Reads analysis/results_long.csv and the model classes; writes analysis/outliers/{points.json, summary.csv}.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import theilslopes

sys.path.insert(0, str(Path(__file__).resolve().parent))
import curves as C  # noqa: E402  (T0, CLIP)

EXP = C.EXP
OUT = EXP / 'analysis/outliers'
KEEP_ALL_VALID = {'TOY-01'}   # strawberry: every reliable observation is kept
DEAD_E = 0.2                  # expiration threshold on e


def t_of(date):
    return (pd.Timestamp(date) - C.T0).days / 365.25


def date_of(t):
    return None if t is None or not np.isfinite(t) else str((C.T0 + pd.to_timedelta(t * 365.25, unit='D')).date())


def outliers(x, y):
    """Boolean mask of outliers from a Theil-Sen line (see module docstring)."""
    n = len(x)
    if n < 6 or np.ptp(x) == 0:
        return np.zeros(n, bool)
    slope, icpt, _, _ = theilslopes(y, x)
    r = y - (icpt + slope * x)
    s = 1.4826 * np.median(np.abs(r - np.median(r)))
    thr = 3 * max(s, 0.1)
    bad = np.abs(r) > thr
    cap = int(np.floor(0.2 * n))
    if bad.sum() > cap:
        keep = np.argsort(-np.abs(r))[:cap]
        bad = np.zeros(n, bool)
        bad[keep] = True
    return bad


def death_time(grid, curve):
    """First grid point from which the curve stays <= DEAD_E to the end. -inf if dead from the start, None if never."""
    below = curve <= DEAD_E
    if not below[-1]:
        return None
    i = len(below) - 1
    while i > 0 and below[i - 1]:
        i -= 1
    return -np.inf if i == 0 else float(grid[i])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(EXP / 'analysis/results_long.csv')
    cls = json.loads((C.PROJ / 'experiments/run_minimal_setups/analysis/inputs/model_classes.json').read_text())
    df['open'] = df.model.map(lambda m: bool(cls.get(m, {}).get('open')))
    df['size'] = df.model.map(lambda m: cls.get(m, {}).get('size'))
    df['t'] = (pd.to_datetime(df.release) - C.T0).dt.days / 365.25
    df = df[df.e.notna() & df.reliable].copy()
    for c in ('e', 'e_lo', 'e_hi'):
        df[c] = df[c].clip(*C.CLIP)
    points, rows = {}, []
    for key, d in df.groupby('key', sort=False):
        d = d.copy()
        d['out_t'] = False if key in KEEP_ALL_VALID else outliers(d.t.values, d.e.values)
        pts = d[['model', 'name', 'release', 't', 'cls', 'open', 'size', 'value', 'e', 'e_lo', 'e_hi',
                 'out_t', 'dead_consistent']].to_dict('records')
        points[key] = dict(points=pts, null=float(d.null.iloc[0]), paper=float(d.paper.iloc[0]))
        rows.append(dict(key=key, n=len(d), n_out_time=int(d.out_t.sum())))
        print(f'{key:8} n={len(d):2} outliers={int(d.out_t.sum())}')
    (OUT / 'points.json').write_text(json.dumps(points, default=lambda o: None if isinstance(o, float) and not np.isfinite(o)
                                                else (bool(o) if isinstance(o, np.bool_) else float(o))))
    pd.DataFrame(rows).to_csv(OUT / 'summary.csv', index=False)


if __name__ == '__main__':
    main()
