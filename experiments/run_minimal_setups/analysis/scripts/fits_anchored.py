"""Anchored fits: the blog's curves, categories and death dates.

Fit (the same for every claim):
  * Anchor: our reproduction on the paper's model, e = repro_e at that model's release date; the curve passes through
    it exactly. C143 has no reproduction: the paper's own value, e = 1, at the paper date.
  * Points: the inlier panel models of outliers.py (Theil-Sen outlier mask out_t; --all-points keeps every model) released
    on or after the anchor date.
  * Shapes rewritten through the anchor (free parameters): constant e = e_a (0); linear e_a + b (t - t_a) (1);
    exponential f + (e_a - f) exp(-k (t - t_a)) (2); logistic lo + (e_a - lo) (1 + exp((t_a - t50)/s)) /
    (1 + exp((t - t50)/s)) (3). The lowest-AICc shape is used.
  * 500 bootstrap draws resample the panel models and draw the anchor from N(repro_e, ((hi - lo)/3.92)^2).
  * Off-trend check: F-test of the anchored shape against the same shape fitted freely (a constant is tested as a line).

Category:
  * dead: at least half of the panel models released since RECENT_FROM (the last 6 months; 8 models) have e <= 0.2
    (point estimates, outliers included). The curve plays no part.
  * otherwise the anchored curve at T_END: strengthened e >= 1.5 (or CI above 1 and rising); persists e >= 0.8 and CI
    above 0.2 (and not CI below 1 while falling); weakened 0.2 < e < 0.8; uncertain otherwise.
Death date (dead claims only):
  * 'fitted'  first date the curve reaches e <= 0.2 and stays there, on or after the first fitted model; bootstrap CI.
  * 'window'  the curve is below 0.2 before the first fitted model: midpoint of max(anchor date, paper date) .. that
              model.
  * 'doa'     a window of at most 3 months, or a claim already dead (by the curve or the count) at its paper date,
              dated at the paper date. No date is ever before the paper (the anchor model often predates the paper).
  * 'count'   the curve never reaches 0.2 (C22/C51, C50/C166): the first model release date from which, at every later
              release date through T_END, at least half of the models released in the 6 months before it (and at least
              COUNT_MIN_N of them) have e <= 0.2. No CI.

Anchors are read from analysis/post/results.json (repro_e, repro_e_lo/hi, repro_release, paper_date). They do not
depend on any fit, so the order is: post_data.py, this script (twice: default and --all-points), post_data.py.

Class fits (--classes): the same fit and category on each model class (open / closed / small / large; inlier models,
at least CLASS_MIN_N per class), with the class's own recent-model count (used only when the class has at least 3 recent models). Writes analysis/curves_anchored/class_fits.json.

Usage: python experiments/run_minimal_setups/analysis/scripts/fits_anchored.py [--all-points | --classes] [--claim=KEY]
A single-claim refresh preserves all other saved entries and uses the same per-claim bootstrap seed.
Writes analysis/curves_anchored/{fits.json, summary.csv} (or analysis/curves_anchored_all_models/ with --all-points).
"""
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import f as fdist

import zlib

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import curves as C  # noqa: E402
import outliers as F  # noqa: E402

N_BOOT = 500
CENTER = 1.5
PERSIST_E = 0.8        # persists = still close to the paper's effect (within 20%)


def shapes(ta, ea):
    """Anchored shapes: name -> (function of (p, t), k free params, starts, bounds) for a given anchor."""
    def lin(p, t):
        return ea + p[0] * (t - ta)

    def exp_(p, t):
        return p[0] + (ea - p[0]) * np.exp(-p[1] * (t - ta))

    def logit(p, t):
        lo, t50, s = p
        return lo + (ea - lo) * (1 + np.exp(np.clip((ta - t50) / s, -50, 50))) / (1 + np.exp(np.clip((t - t50) / s, -50, 50)))

    return {'constant': (lambda p, t: np.full_like(t, ea, dtype=float), 0),
            'linear': (lin, 1), 'exponential': (exp_, 2), 'logistic': (logit, 3)}


def starts(name, t, y, ta):
    span = float(np.ptp(np.r_[ta, t]) + 1e-6)
    if name == 'linear':
        return [[0.0]], ([-np.inf], [np.inf])
    if name == 'exponential':
        S = [[f0, k / span] for f0 in (min(y), max(y), float(np.mean(y))) for k in (-1.5, 1.0, 4.0)]
        return S, ([-10, -6 / span], [10, 6 / span])
    S = [[lo, t50, s * span] for lo in (min(y), max(y), float(np.mean(y)))
         for t50 in np.quantile(np.r_[ta, t], [0.25, 0.5, 0.75]) for s in (0.07, 0.25)]
    return S, ([-10, ta - 0.2 * span, 0.02 * span], [10, t.max() + 0.2 * span, 1.5 * span])


def fit_anchored(name, t, y, ta, ea):
    g, k = shapes(ta, ea)[name]
    if k == 0:
        return (np.array([]), float(np.sum((g(None, t) - y) ** 2))), g
    S, bounds = starts(name, t, y, ta)
    best = None
    for s0 in S:
        s0 = np.clip(s0, np.array(bounds[0]) + 1e-6, np.array(bounds[1]) - 1e-6)
        try:
            r = least_squares(lambda p: g(p, t) - y, s0, bounds=bounds, method='trf')
        except Exception:
            continue
        rss = float(np.sum(r.fun ** 2))
        if best is None or rss < best[1]:
            best = (r.x, rss)
    return best, g


def select(t, y, ta, ea):
    fits = {}
    for name, (_, k) in shapes(ta, ea).items():
        b, g = fit_anchored(name, t, y, ta, ea)
        if b is not None:
            fits[name] = dict(params=b[0], rss=b[1], k=k, aicc=C.aicc(b[1], len(t), k) if k else
                              len(t) * math.log(max(b[1], 1e-12) / len(t)), g=g)
    return min(fits, key=lambda n: fits[n]['aicc']), fits   # lowest AICc


def anchor_test(name, t, y, rss_a, k):
    """F-test: anchored shape vs the same shape with the anchor freed (one extra parameter), on the same points.
    A constant is tested as a line (anchored slope vs free line): a free constant cannot see a trend that starts
    far from the anchor (C22/C051 before 2026-10-21)."""
    if name == 'constant':
        name = 'linear'
        rss_a = fit_anchored('linear', t, y, *ANCHOR_NOW)[0][1]
    b, _ = C.fit_shape(name, t, y, CENTER)
    if b is None:
        return None
    free_rss, kf = b[1], C.SHAPES[name][1]
    dfd = len(t) - kf
    if dfd <= 0 or free_rss <= 0:
        return None
    Fv = max(rss_a - free_rss, 0) / (free_rss / dfd)
    return float(fdist.sf(Fv, 1, dfd))


def verdict_close(est, lo, hi, rising, falling, pe=PERSIST_E):
    """Category of the anchored curve at T_END (dead is decided by the recent-model count, not here)."""
    if est is None or not np.isfinite(est) or (lo <= 0.2 and hi >= 1.0 and est < pe):
        return 'uncertain'
    if est >= 1.5 or (lo > 1 and rising):
        return 'strengthened'
    if est <= 0.2 and hi < 0.67:
        return 'dead'
    if est >= pe and lo > 0.2 and not (hi < 1 and falling):
        return 'persists'
    if 0.2 < est < pe or (est >= pe and hi < 1 and falling):
        return 'attenuated'
    return 'uncertain'


def run(key, pts, anchor, rng):
    ta, ea, esd = anchor['t'], anchor['e'], anchor['sd']
    post = [p for p in pts if not p['out_t'] and p['t'] >= ta - 1e-9]
    t = np.array([p['t'] for p in post])
    y = np.clip(np.array([p['e'] for p in post]), *C.CLIP)
    n_before = sum(1 for p in pts if not p['out_t'] and p['t'] < ta - 1e-9)
    if len(t) < 3:
        return dict(n=len(t), n_before=n_before, shape=None)
    global ANCHOR_NOW
    ANCHOR_NOW = (ta, ea)
    shape, fits = select(t, y, ta, ea)
    p, g, k = fits[shape]['params'], fits[shape]['g'], fits[shape]['k']
    grid = np.linspace(ta, max(t.max(), C.T_END), 160)
    curve = g(p, grid)
    ends, bands, deaths = [], [], []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(t), len(t))
        eb = float(np.clip(ea + esd * rng.standard_normal(), *C.CLIP)) if esd else ea
        b, gb = fit_anchored(shape, t[idx], y[idx], ta, eb)
        if b is None:
            continue
        cb = gb(b[0], grid)
        bands.append(cb)
        ends.append(float(gb(b[0], np.array([C.T_END]))[0]))
        deaths.append(F.death_time(grid, cb))
    end = float(g(p, np.array([C.T_END]))[0])
    lo, hi = float(np.percentile(ends, 2.5)), float(np.percentile(ends, 97.5))
    start = float(g(p, np.array([max(ta, 0.55)]))[0])
    blo, bhi = np.percentile(bands, 2.5, axis=0), np.percentile(bands, 97.5, axis=0)
    # rising / falling: change from the anchor to T_END, bootstrap CI
    ch = np.array(ends) - np.array([b[0] for b in bands])
    clo, chi = np.percentile(ch, [2.5, 97.5])
    verdict_08 = verdict_close(end, lo, hi, bool(clo > 0), bool(chi < 0))
    dt = F.death_time(grid, curve)
    t1 = float(t.min())
    death = kind = dci = None
    if dt is not None:
        dt = ta if dt == -np.inf else dt
        if dt < t1:
            kind, death, dci = 'window', (ta + t1) / 2, [ta, t1]
        else:
            fin = [d for d in deaths if d is not None]
            kind, death = 'fitted', dt
            if len(fin) >= 0.5 * len(deaths):
                fin = [max(d, ta) for d in fin]
                dci = [float(np.percentile(fin, 2.5)), float(np.percentile(fin, 97.5))]
    return dict(n=len(t), n_before=n_before, shape=shape, k=k, end=end, end_ci=[lo, hi],
                verdict_close=verdict_08, falling=bool(chi < 0), rising=bool(clo > 0),
                anchor_p=anchor_test(shape, t, y, fits[shape]['rss'], k), start=start,
                dates=[F.date_of(v) for v in grid], curve=curve.tolist(), band_lo=blo.tolist(), band_hi=bhi.tolist(),
                death=F.date_of(death) if death is not None else None, death_kind=kind,
                death_ci=[F.date_of(v) for v in dci] if dci else None)


RECENT_FROM = '2026-03-31'   # the panel's last 6 months (T_END 2026-09-30): 8 models for every claim
COUNT_MIN_N = 4              # a 'count' death date needs at least this many models in the 6-month window
DOA_YEARS = 0.25
PROJ = C.PROJ
OUT = {False: C.EXP / 'analysis/curves_anchored', True: C.EXP / 'analysis/curves_anchored_all_models'}


def count_death(points):
    """'count' death date (module docstring). points: dicts with date (YYYY-MM-DD) and e. None if the rule never holds."""
    dates = sorted({p['date'] for p in points})
    def holds(d):
        lo = F.date_of(F.t_of(d) - 0.5)
        w = [p for p in points if lo < p['date'] <= d]
        return len(w) >= COUNT_MIN_N and 2 * sum(p['e'] <= F.DEAD_E for p in w) >= len(w)
    ok = [holds(d) for d in dates]
    if not ok or not ok[-1]:
        return None
    i = len(ok) - 1
    while i > 0 and ok[i - 1]:
        i -= 1
    return dates[i]


def death_date(a, points, anchor, tp):
    """Death date of a dead claim (module docstring), never before the paper date tp: the anchor (the reproduction
    model's release) often predates the paper, but a claim cannot die before it is published. A claim whose curve
    (or newest-model count) is already dead by the paper date is dead on arrival, dated at the paper."""
    doa = dict(death=F.date_of(tp), death_lo=F.date_of(tp), death_hi=F.date_of(tp), death_kind='doa')   # a zero-length window
    if not a.get('death'):   # the curve never reaches 0.2: date from the newest-model count
        d = count_death(points)
        return doa if F.t_of(d) <= tp else dict(death=d, death_lo=None, death_hi=None, death_kind='count')
    if a['death_kind'] == 'window':
        t0, t1 = max(anchor['t'], tp), F.t_of(a['death_ci'][1])
        if t1 <= tp:
            return doa
        return dict(death=F.date_of((t0 + t1) / 2), death_lo=F.date_of(t0), death_hi=a['death_ci'][1],
                    death_kind='doa' if t1 - t0 <= DOA_YEARS else 'window')
    if F.t_of(a['death']) <= tp:
        return doa
    ci = a['death_ci']
    return dict(death=a['death'], death_lo=ci and F.date_of(max(F.t_of(ci[0]), tp)), death_hi=ci and ci[1],
                death_kind='fitted')


def categorize(a, recent):
    """Category from the anchored fit a (run()) and the recent-model counts (n, n dead)."""
    n, k = recent
    if n and 2 * k >= n:
        return 'dead'
    v = a.get('verdict_close') or 'uncertain'
    return 'attenuated' if v == 'dead' else v   # the curve says dead but most recent models are not: weakened


CLASS_MIN_N = 5
GROUPS = {'open': lambda p: p['open'], 'closed': lambda p: not p['open'],
          'small': lambda p: p['size'] == 'small', 'large': lambda p: p['size'] == 'large'}


def selected_claim():
    return next((x.split('=', 1)[1] for x in sys.argv if x.startswith('--claim=')), None)


def class_fits():
    fits = json.loads((F.OUT / 'points.json').read_text())
    anchors = json.loads((OUT[False] / 'fits.json').read_text())
    res = {}
    for key in sorted(anchors):
        if selected_claim() and key != selected_claim():
            continue
        anchor = anchors[key]['anchor']
        res[key] = {}
        for g, sel in GROUPS.items():
            pts = [p for p in fits[key]['points'] if sel(p)]
            fitted = [p for p in pts if not p['out_t'] and p['t'] >= anchor['t'] - 1e-9]
            if len(fitted) < CLASS_MIN_N:
                res[key][g] = None
                continue
            rng = np.random.default_rng([20261021, zlib.crc32(f'{key}/{g}'.encode())])
            a = run(key, pts, anchor, rng)
            rec = [p for p in pts if p['release'] >= RECENT_FROM]
            nd = sum(p['e'] <= F.DEAD_E for p in rec)
            res[key][g] = dict(n=a['n'], shape=a['shape'], end=a['end'], end_ci=a['end_ci'], recent_n=len(rec),
                               recent_dead=nd, category=categorize(a, (len(rec), nd) if len(rec) >= 3 else (0, 0)))
        print(key, {g: v and v['category'] for g, v in res[key].items()})
    if selected_claim():
        saved = json.loads((OUT[False] / 'class_fits.json').read_text())
        saved.update(res)
        res = saved
    (OUT[False] / 'class_fits.json').write_text(json.dumps(res))


def main():
    if '--classes' in sys.argv:
        return class_fits()
    all_points = '--all-points' in sys.argv
    out = OUT[all_points]
    out.mkdir(parents=True, exist_ok=True)
    fits = json.loads((F.OUT / 'points.json').read_text())
    res = json.loads((PROJ / 'experiments/run_minimal_setups/analysis/post/results.json').read_text())
    allfits, rows = {}, []
    for c in sorted(res['claims'], key=lambda c: c['key']):
        key = c['key']
        if selected_claim() and key != selected_claim():
            continue
        pts = [dict(p, out_t=False) for p in fits[key]['points']] if all_points else fits[key]['points']
        if c['repro_e'] is not None:
            sd = ((c['repro_e_hi'] - c['repro_e_lo']) / 3.92) if c.get('repro_e_lo') is not None else 0.0
            anchor = dict(t=F.t_of(c['repro_release']), e=float(np.clip(c['repro_e'], *C.CLIP)), sd=sd,
                          date=c['repro_release'], source='repro', lo=c.get('repro_e_lo'), hi=c.get('repro_e_hi'))
        else:
            anchor = dict(t=F.t_of(c['paper_date']), e=1.0, sd=0.0, date=c['paper_date'], source='paper', lo=None, hi=None)
        rng = np.random.default_rng([20261021, zlib.crc32(key.encode())])   # per-claim stream: order-independent
        a = run(key, pts, anchor, rng)
        pe = [dict(date=p['release'], e=p['e'], e_hi=p['e_hi']) for p in fits[key]['points']]   # all points
        rec = [p for p in pe if p['date'] >= RECENT_FROM]
        n_dead = sum(p['e'] <= F.DEAD_E for p in rec)
        n_clear = sum(p['e'] <= F.DEAD_E and (p['e_hi'] if p['e_hi'] is not None else p['e']) < 0.67 for p in rec)
        cat = categorize(a, (len(rec), n_dead))
        death = dict(death=None, death_lo=None, death_hi=None, death_kind=None)
        if cat == 'dead':
            death = death_date(a, pe, anchor, F.t_of(c['paper_date']))
        # keep the curve's own crossing apart from the reported death date
        a['curve_death'], a['curve_death_kind'], a['curve_death_ci'] = a.pop('death'), a.pop('death_kind'), a.pop('death_ci')
        a.update(category=cat, anchor=anchor, recent_n=len(rec), recent_dead=n_dead, recent_clear_dead=n_clear, **death)
        allfits[key] = a
        rows.append(dict(key=key, category=cat, curve_verdict=a.get('verdict_close'), shape=a.get('shape'), n=a['n'],
                         n_before_anchor=a['n_before'], end=a.get('end'), end_lo=a.get('end_ci', [None])[0],
                         end_hi=a.get('end_ci', [None, None])[1], recent_n=len(rec), recent_dead=n_dead,
                         recent_clear_dead=n_clear, anchor_date=anchor['date'], anchor_e=anchor['e'],
                         anchor_source=anchor['source'], anchor_p=a.get('anchor_p'), **death))
        print(f"{key:9} {cat:12} shape={a.get('shape'):11} end={a.get('end', float('nan')):5.2f} recent={n_dead}/{len(rec)} "
              f"death={death['death']} ({death['death_kind']})")
    if selected_claim():
        if selected_claim() not in allfits:
            raise ValueError(f'Unknown claim: {selected_claim()}')
        saved = json.loads((out / 'fits.json').read_text())
        saved.update(allfits)
        allfits = saved
        with open(out / 'summary.csv', newline='') as f:
            prior = list(csv.DictReader(f))
        rows = [rows[0] if r['key'] == selected_claim() else r for r in prior]
    (out / 'fits.json').write_text(json.dumps(allfits, default=lambda o: None if isinstance(o, float) and not np.isfinite(o)
                                              else (bool(o) if isinstance(o, np.bool_) else float(o))))
    with open(out / 'summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    print(Counter(r['category'] for r in rows))


if __name__ == '__main__':
    main()
