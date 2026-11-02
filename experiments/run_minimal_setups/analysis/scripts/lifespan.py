"""Claim lifespans and the paper-date confound.

Questions:
  1. How long do claims last from publication to time of death? (Kaplan-Meier over all counted claims; claims that
     are not dead are censored at T_END = 2026-09-30.)
  2. Does the data support "most published LLM findings are no longer valid after ~6 months"? (KM at 6 months, plus
     the fitted effect e at paper date + 6 / 12 / 24 months, read off each claim's robust trend curve.)
  3. Is "dead" confounded with paper date? (Category by paper year, Fisher test, and the same KM by paper era.)

Death dates and curves come from the anchored fits (fits_anchored.py, 2026-10-21: the curve starts at our reproduction
on the paper's model; dead = at least half of the newest models at e <= 0.2; death = first date the curve reaches
e <= 0.2 and stays there, never before the paper date). Deaths of kind 'window' / 'doa' (dead by the first fitted model
after the paper, or already at the paper date) are interval-censored between the paper date and that model; we report
the earliest bound (lifespan 0), the window midpoint (the estimate shown on the blog) and the latest bound. Deaths of
kind 'count' (curve never crosses; dated from the newest-model count) are treated like fitted dates.

Reads analysis/post/results.json (the counted claims, their categories, death dates and anchored curves).
Writes analysis/lifespan/{lifespan.csv, summary.md}.
Usage (standard library + numpy/scipy): python experiments/run_minimal_setups/analysis/scripts/lifespan.py
"""
import csv
import json
from datetime import date
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact, spearmanr

EXP = Path(__file__).resolve().parents[2]
PROJ = EXP.parents[1]
OUT = EXP / 'analysis/lifespan'
T0 = date(2024, 1, 1)
T_END = date(2026, 9, 30)
FIRST_MODEL = date(2024, 7, 18)
DEAD_E, PERSIST_E = 0.2, 0.8
MO = 365.25 / 12


def d(s):
    return date.fromisoformat(s[:10])


def months(a, b):
    return (b - a).days / MO


def t_years(x):
    return (x - T0).days / 365.25


def km(times, events, at):
    """Kaplan-Meier survival S(t) at each t in `at` (times in months; events 1 = death, 0 = censored)."""
    order = np.argsort(times, kind='stable')
    t, e = np.asarray(times)[order], np.asarray(events)[order]
    out = []
    for a in at:
        s, n = 1.0, len(t)
        for ti in np.unique(t[(t <= a)]):
            deaths = int(((t == ti) & (e == 1)).sum())
            at_risk = int((t >= ti).sum())
            if at_risk:
                s *= 1 - deaths / at_risk
        out.append(s)
    return out


def km_median(times, events):
    grid = np.linspace(0, 60, 601)
    s = km(times, events, grid)
    below = [g for g, v in zip(grid, s) if v <= 0.5]
    return below[0] if below else None


def main():
    claims = json.loads((PROJ / 'experiments/run_minimal_setups/analysis/post/results.json').read_text())['claims']
    rows = []
    for c in claims:
        p = d(c['paper_date'])
        cat = c['category']
        r = dict(key=c['key'], title=c['title'], group=c['group'], category=cat, paper_date=str(p), death=c['death'])
        obs = months(p, T_END)
        r['death_kind'] = c.get('death_kind')
        if cat == 'dead' and c.get('death_kind') in ('window', 'doa'):
            hi = max(0.0, months(p, d(c['death_hi'] or c['death'])))
            r.update(life_lo=0.0, life_mid=round(months(p, d(c['death'])), 1), life_hi=round(hi, 1), event=1,
                     note='dead on arrival' if c['death_kind'] == 'doa' else 'death window: paper to first post-paper model')
        elif cat == 'dead' and c['death']:
            life = round(months(p, d(c['death'])), 1)
            r.update(life_lo=life, life_mid=life, life_hi=life, event=1, note='')
        elif cat == 'dead':
            r.update(life_lo=None, life_mid=None, life_hi=round(obs, 1), event=1, note='dead-now rule, no fitted death date')
        else:
            obs = round(obs, 1)
            r.update(life_lo=obs, life_mid=obs, life_hi=obs, event=0, note='censored (not dead at Sep 2026)')
        f = c['fit']
        grid, curve = np.array([t_years(d(x)) for x in f['dates']]), np.array(f['curve'], dtype=float)
        for h in (6, 12, 24):
            tq = t_years(p) + h / 12
            r[f'e_{h}m'] = (round(float(np.interp(tq, grid, curve)), 3)
                            if grid[0] <= tq <= t_years(T_END) else None)
        r['e_now'] = c['end']
        rows.append(r)

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / 'lifespan.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    L = []
    for bound in ('life_lo', 'life_mid', 'life_hi'):
        t = [r[bound] if r[bound] is not None else r['life_hi'] for r in rows]
        e = [r['event'] for r in rows]
        s6, s12, s24 = km(t, e, [6, 12, 24])
        L.append((bound, 1 - s6, 1 - s12, 1 - s24, km_median(t, e)))
    dead = [r for r in rows if r['category'] == 'dead']
    dated = sorted(r['life_lo'] for r in dead if r['note'] == '')

    lines = ['# Claim lifespans and the paper-date confound', '',
             f'{n} counted claims; {len(dead)} dead. Generated by analysis/scripts/lifespan.py.', '',
             '## Kaplan-Meier: share of claims dead within h months of publication', '',
             '| window deaths counted at | 6 mo | 12 mo | 24 mo | median lifespan (mo) |', '|---|---|---|---|---|']
    for bound, a, b, c_, med in L:
        lab = {'life_lo': 'paper date (earliest)', 'life_mid': 'window midpoint (blog estimate)',
               'life_hi': 'first post-paper model (latest)'}[bound]
        lines.append(f'| {lab} | {a:.0%} | {b:.0%} | {c_:.0%} | {med if med is None else round(med, 1)} |')
    win = sorted(r['life_mid'] for r in dead if r['death_kind'] in ('window', 'doa'))
    alld = sorted(r['life_mid'] for r in dead if r['life_mid'] is not None)
    lines += ['', f'Fitted deaths (months from paper): {", ".join(f"{x:.0f}" for x in dated)} '
              f'(median {np.median(dated):.1f}).',
              f'Window deaths, midpoints (months from paper): {", ".join(f"{x:.1f}" for x in win)}.',
              f'All dated deaths: n = {len(alld)}, median {np.median(alld):.1f}, range {min(alld):.1f}-{max(alld):.1f}.', '']

    lines += ['## Fitted effect h months after publication (claims whose date + h falls inside the panel window)', '',
              '| h | n | dead (e<=0.2) | attenuated | persists/strengthened (e>=0.8) |', '|---|---|---|---|---|']
    for h in (6, 12, 24):
        v = [r[f'e_{h}m'] for r in rows if r[f'e_{h}m'] is not None]
        if v:
            v = np.array(v)
            lines.append(f'| {h} mo | {len(v)} | {(v <= DEAD_E).mean():.0%} | '
                         f'{((v > DEAD_E) & (v < PERSIST_E)).mean():.0%} | {(v >= PERSIST_E).mean():.0%} |')

    lines += ['', '## Category by paper year', '', '| paper year | n | dead | attenuated | persists | strengthened |',
              '|---|---|---|---|---|---|']
    era = lambda r: '2022-23' if r['paper_date'] < '2024' else r['paper_date'][:4]
    for y in sorted({era(r) for r in rows}):
        g = [r for r in rows if era(r) == y]
        cnt = {k: sum(r['category'] == k for r in g) for k in ('dead', 'attenuated', 'persists', 'strengthened')}
        lines.append(f'| {y} | {len(g)} | ' + ' | '.join(f'{cnt[k]} ({cnt[k] / len(g):.0%})' for k in cnt) + ' |')
    old = [r for r in rows if r['paper_date'] < '2024']
    mid = [r for r in rows if '2024' <= r['paper_date'] < '2025']
    tab = [[sum(r['category'] == 'dead' for r in g), sum(r['category'] != 'dead' for r in g)] for g in (old, mid)]
    _, pf = fisher_exact(tab)
    rho, pr = spearmanr([t_years(d(r['paper_date'])) for r in rows], [r['e_now'] for r in rows])
    lines += ['', f'Dead share, papers 2022-23 vs 2024: {tab[0][0]}/{len(old)} vs {tab[1][0]}/{len(mid)}, '
              f'Fisher p = {pf:.2f}.', f'Spearman(paper date, fitted e at Sep 2026) = {rho:.2f} (p = {pr:.3f}).', '']
    (OUT / 'summary.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
