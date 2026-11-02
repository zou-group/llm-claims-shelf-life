"""The data the blog post shows: anchors, curves, verdicts, death dates and counts.

Standard library only. Run from anywhere:  python3 experiments/run_minimal_setups/analysis/scripts/post_data.py
Reads (never writes) the experiment folders; writes analysis/post/results.json (everything the figures plot)
and analysis/post/numbers.json (the numbers quoted in the text). The post itself is docs/index.html.

Sources (all relative to the project root):
  experiments/run_minimal_setups/analysis/curves_anchored/{summary.csv,fits.json}  anchored fits: curves, categories, death
                                                                               dates (fits_anchored.py, 2026-10-21)
  experiments/run_minimal_setups/analysis/curves_anchored_all_models/summary.csv    the same with no outlier exclusion
                                                                               (sensitivity only; fits_anchored.py --all-points)
  experiments/run_minimal_setups/analysis/outliers/{summary.csv,points.json}  per-model points and the outlier mask
  experiments/run_minimal_setups/analysis/claim_ids.json                       frozen reader-facing claim numbers (C1, C2, ...)
  experiments/run_minimal_setups/analysis/claim_meta.json                      titles, metrics, paper/repro values, dates
  experiments/run_minimal_setups/analysis/results_long.csv                     per-model values (incl. unreliable rows)
  experiments/run_minimal_setups/analysis/inputs/claim_set_search.json                 field group and one-line label per claim
  experiments/run_minimal_setups/analysis/inputs/claim_set.json                    the same for the expansion claims (C141-C143, C157-C166, C079)
  experiments/run_minimal_setups/analysis/inputs/reproduction_verdicts.csv        reproduction on the papers' own models
  experiments/run_minimal_setups/analysis/inputs/panel20.csv     the panel
  experiments/run_minimal_setups/analysis/inputs/model_classes.json                         open/closed, small/large
  experiments/run_minimal_setups/analysis/inputs/C101_panel/*/metrics.json     strawberry answers per model
  experiments/run_minimal_setups/analysis/inputs/C101_reproduction/*/metrics.json            strawberry anchors (gpt-4o 2024-05-13 / 08-06)
"""
from public_data import public_results, public_numbers

import csv
import datetime as dt
import json
import re
from pathlib import Path

from paper_dates import apply_release_dates

HERE = Path(__file__).resolve().parents[1] / 'post'   # analysis/post/: the data the post shows
PROJ = Path(__file__).resolve().parents[4]
EXP = PROJ / 'experiments'
RMS = EXP / 'run_minimal_setups'
T0 = dt.date(2024, 1, 1)

# Release dates of the reproduction models (first model named when two were used). From the 207-record catalog
# (the model catalogue) where listed;
# the three OpenAI snapshots it lacks are dated by their snapshot IDs.
REPRO_RELEASE = {
    'gpt-4-0613': '2023-06-13',              # snapshot ID
    'gpt-3.5-turbo-0125': '2024-01-25',      # snapshot ID
    'gpt-4-turbo-2024-04-09': '2024-04-09',  # snapshot ID
    'gpt-4o-2024-05-13': '2024-05-13', 'gpt-4o-mini-2024-07-18': '2024-07-18', 'llama-3.1-8b-instruct': '2024-07-23',
    'gpt-4o-2024-08-06': '2024-08-06', 'deepseek-r1': '2025-01-20', 'gpt-4.1': '2025-04-14',
    'gpt-4o-2024-11-20': '2024-11-20',       # snapshot ID
    # the five added claims reproductions (2026-10-17)
    'GPT-4': '2023-03-14',                   # unversioned GPT-4 (C162 archive answers): GPT-4 launch date
    'gpt-4-1106-preview': '2023-11-06',      # snapshot ID (C157 archive tallies)
    'Qwen2.5-72B-Instruct': '2024-09-19',    # panel release date (M025)
}
# Expansion claims added 2026-10-12 (claim_meta group 'expansion'). These two are not counted
# (experiment decision): C141-gap is the superseded C141 metric, C142-AITA reuses C026's posts, prompt and scorer.
NOT_COUNTED = {'C141-gap', 'C142-AITA'}
# Reproduction on the paper's model for the expansion claims (they are not in the reproduction pass);
# experiments/run_minimal_setups/runs/added-b/report.md. C143's20sample panel-completion runs are not
# faithful50sample paper reproductions; the authors' raw outputs remain unreleased.
EXPANSION_MATCH = {
    'C141': 'same-family-substitute (gpt-3.5-turbo-0125 for the retired -0613)',
    'C142-OEQ': 'proxy (GPT-4o 2024-11-20, one of the paper\'s 11 models, against the 11-model mean)',
    'C142-PAS': 'proxy (GPT-4o 2024-11-20, one of the paper\'s 11 models, against the 11-model mean)',
    # Expansion 5 (experiments/run_minimal_setups/runs/added-a/results.md, added to the post 2026-10-17)
    'C161': 'exact (gpt-4-0613 = 2023 gpt-4 alias target), rescored with the panel scorer',
    'C162': "exact (authors' released GPT-4 zero-shot answers, rescored; no fresh run)",
    'C157': "exact (authors' released GPT-4-1106 selector tallies; no fresh run)",
    'C166': 'exact (Qwen2.5-72B-Instruct, fresh run; GPT-6 Luna judge instead of the paper\'s)',
    'C079': 'exact (gpt-4-0613 for the paper\'s unversioned GPT-4), rescored with the panel scorer',
}
# Claim wording shown on a card when the register's wording covers more than we test. C079's register claim has two
# limbs; only the implicit word-association limb was ported (claim_meta note).
CLAIM_FIX = {
    'C079': 'Models that look unbiased on explicit benchmarks still produce stereotype-consistent pairings on the '
            'IAT-style word-assignment prompt (mean implicit bias score > 0). (The paper\'s second limb, relative '
            'decision bias, is not tested here.)',
}
# Current blog taxonomy; historical claim registers retain their original groups.
DISPLAY_GROUP_OVERRIDES = {
    'C161': 'Jailbreaks & safety robustness',
    'PI-N06': 'Jailbreaks & safety robustness',
    'C093': 'Factuality, hallucination & calibration',
}
# Categories come from the anchored fits (fits_anchored.py, 2026-10-21); a split between the anchored class fits is
# shown on each card through class_split.
CATEGORIES = ['strengthened', 'persists', 'attenuated', 'dead']
# Reader-facing claim numbers (2026-10-14). Internal keys (C010, SH-N02, TOY-01, ...) stay the data keys; readers see
# C1..C50, numbered once by paper date. The numbers are frozen in claim_ids.json: a new claim gets the next number,
# and no claim is ever renumbered.
CLAIM_IDS = HERE.parent / 'claim_ids.json'


def claim_ids(claims):
    ids = json.loads(CLAIM_IDS.read_text()) if CLAIM_IDS.exists() else {}
    new = sorted((c for c in claims if c['key'] not in ids), key=lambda c: (c['paper_date'], c['key']))
    for c in new:
        ids[c['key']] = f'C{len(ids) + 1}'
    if new:
        CLAIM_IDS.write_text(json.dumps(ids, indent=1) + '\n')
    return ids


def reproduction_tier(e, lo, hi):
    """Apply the existing tolerance rule to the current normalized reproduction."""
    if e is None or lo is None or hi is None or lo <= 0:
        return 'not_reproduced'
    if lo <= 1 <= hi or abs(e - 1) <= .25:
        return 'two_sided'
    return 'one_sided' if e > 1.25 else 'not_reproduced'


def dead_strength(a, all_points):
    """Descriptive support for an existing dead verdict; never changes its category.

    Both checks = strong, one = moderate, neither = weak. Recent counts include
    outliers. Trend support requires both outlier-excluding and all-points fits.
    Use full-precision summary values for decisions, not rounded display values.
    """
    n, nd = int(a['recent_n']), int(a['recent_dead'])
    assert n == int(all_points['recent_n']) and nd == int(all_points['recent_dead'])
    def interval(row):
        e, lo, hi = (float(row[k]) for k in ('end', 'end_lo', 'end_hi'))
        return f'e = {e:.3f} [{lo:.3f}, {hi:.3f}]'
    checks = [
        dict(ok=n > 0 and 2 * nd > n,
             text=f'Recent-model agreement: {nd} of {n} reliable models at e ≤ 0.2 '
                  '(last six months, including outliers; strict majority required)'),
        dict(ok=float(a['end_hi']) <= 0.2 and float(all_points['end_hi']) <= 0.2,
             text='Robust trend evidence: both 95% upper bounds must be ≤ 0.2. '
                  f'Main fit: {interval(a)}; including outliers: {interval(all_points)} '
                  '(September 30, 2026)'),
    ]
    k = sum(c['ok'] for c in checks)
    return dict(level='strong' if k == 2 else 'moderate' if k == 1 else 'weak', checks=checks)


def apply_display_fit_overrides(claims):
    """Retain blackmail outliers in its displayed fit/date.

    Keep the original fits and the two-fit support assessment for sensitivity.
    """
    all_fits = json.loads((RMS / 'analysis/curves_anchored_all_models/fits.json').read_text())
    for c in claims:
        if c['key'] != 'C028':
            continue
        af = all_fits[c['key']]
        assert af['category'] == c['category'] == 'dead'
        c['fit_source'] = 'all_points'
        c.update(end=num(af['end']), end_lo=num(af['end_ci'][0]), end_hi=num(af['end_ci'][1]),
                 death=af['death'], death_lo=af['death_lo'], death_hi=af['death_hi'],
                 death_kind=af['death_kind'], n_outliers=0)
        c['fit'] = dict(shape=af['shape'], n=af['n'], n_before_anchor=af['n_before'],
                        end=num(af['end']), end_lo=num(af['end_ci'][0]), end_hi=num(af['end_ci'][1]),
                        anchored=True, anchor_p=num(af.get('anchor_p')), dates=af['dates'],
                        curve=[num(v) for v in af['curve']],
                        band_lo=[num(v) for v in af['band_lo']], band_hi=[num(v) for v in af['band_hi']])
        for point in c['points']:
            point['outlier'] = False
        # Class-specific fits remain the original sensitivity analyses, labelled in the UI.
        for check in c['dead_strength']['checks']:
            check['text'] = check['text'].replace('Main fit:', 'Fit excluding outliers:')


def class_split(cf):
    """Anchored class fits that disagree: one class persists (or strengthened) while the other is dead."""
    out = []
    for a, b in (('open', 'closed'), ('small', 'large')):
        va, vb = (cf.get(a) or {}).get('category'), (cf.get(b) or {}).get('category')
        alive = ('persists', 'strengthened')
        if (va in alive and vb == 'dead') or (vb in alive and va == 'dead'):
            out.append(f'{a} {LABEL[va]}, {b} {LABEL[vb]}')
    return '; '.join(out)


LABEL = {'strengthened': 'strengthened', 'persists': 'persists', 'attenuated': 'weakened', 'dead': 'expired',
         'uncertain': 'uncertain', 'model-specific': 'model-specific'}


def rcsv(p):
    with open(p, newline='') as f:
        return list(csv.DictReader(f))


def num(x, nd=4):
    if x is None or x == '':
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if v != v:
        return None
    return round(v, nd)


def t2date(t):
    return (T0 + dt.timedelta(days=float(t) * 365.25)).isoformat()


def unit_of(label):
    m = re.search(r'\(([^()]*)\)\s*$', label or '')
    if not m:
        return ''
    u = m.group(1)
    if u.startswith('%'):
        return '%'
    if u.startswith('pp'):
        return ' pp'
    if 'pts' in u:
        return ' pts'
    return ''


# Paper titles for the claim cards, from the citation strings in claim_meta.json; two need a hand fix.
TITLE_FIX = {
    'SH-N03': "Wait, but Tylenol is Acetaminophen... Investigating and Improving Language Models' Ability to Resist Requests for Misinformation",
    'C026': 'Social Sycophancy: A Broader Understanding of LLM Sycophancy',
}


def paper_title(key, citation):
    if key in TITLE_FIX:
        return TITLE_FIX[key]
    m = re.search(r'\(\d{4}\)\.\s*(.+?[?!.])(?:\s|$)', citation or '')
    return m.group(1).rstrip('.') if m else ''


def short_cite(c):
    """'Walters & Wilder (2023)' for two authors, 'Cheng et al. (2026)' for more. A citation that starts with a note
    ending in 'Closest published evidence:' (TOY-01) is cited by that evidence."""
    c = (c or '').split('Closest published evidence:')[-1]
    m = re.match(r'\s*(.+?)\s*\((\d{4})\)', c)
    if not m:
        return c[:60]
    authors, year = m.groups()
    first = authors.split(',')[0].strip()
    names = re.findall(r"([A-Z][\w'’-]+), (?:[A-Z]\.\s?-?)+", authors)
    if ' et al' in authors or len(names) > 2:
        return f'{first} et al. ({year})'
    if len(names) == 2:
        return f'{names[0]} & {names[1]} ({year})'
    return f'{first} ({year})'


def main():
    summary = {r['key']: r for r in rcsv(RMS / 'analysis/outliers/summary.csv')}
    fits = json.loads((RMS / 'analysis/outliers/points.json').read_text())
    # anchored fits (2026-10-21): every category, curve and death date on the page comes from these
    anch = {r['key']: r for r in rcsv(RMS / 'analysis/curves_anchored/summary.csv')}
    anch_fits = json.loads((RMS / 'analysis/curves_anchored/fits.json').read_text())
    anch_cls = json.loads((RMS / 'analysis/curves_anchored/class_fits.json').read_text())
    p_all = RMS / 'analysis/curves_anchored_all_models/summary.csv'
    summary_all = {r['key']: r for r in rcsv(p_all)} if p_all.exists() else {}
    meta = apply_release_dates(json.loads((RMS / 'analysis/claim_meta.json').read_text()))
    long_rows = rcsv(RMS / 'analysis/results_long.csv')
    claim_set = {c['key']: c for c in json.loads((EXP / 'run_minimal_setups/analysis/inputs/claim_set_search.json').read_text())}
    claim_set_exp = {c['key']: c for c in json.loads((EXP / 'run_minimal_setups/analysis/inputs/claim_set.json').read_text())}
    tiered = {r['key']: r for r in rcsv(EXP / 'run_minimal_setups/analysis/inputs/reproduction_verdicts.csv')}
    pilot = {r['record_id']: r for r in rcsv(EXP / 'run_minimal_setups/analysis/inputs/panel20.csv')}
    classes = json.loads((EXP / 'run_minimal_setups/analysis/inputs/model_classes.json').read_text())

    # ---------------------------------------------------------------- models (the panel)
    models = {}
    for r in long_rows:
        m = r['model']
        if m in models or m not in pilot:
            continue
        p = pilot[m]
        c = classes.get(m, {})
        models[m] = dict(
            id=m, name=r['name'], developer=r['developer'], release=r['release'], cls=r['cls'],
            open=bool(c.get('open')), size=c.get('size'),
            params_b=num(p['total_params_b'], 1), active_b=num(p['active_params_b'], 1), weights=p['weights'],
            price_in=num(p['input_usd_per_1m'], 4), price_out=num(p['output_usd_per_1m'], 4),
            role=p['role'], provider=p['provider'], model_id=p['model_id'],
        )
    assert len(models) == 20, len(models)
    model_list = sorted(models.values(), key=lambda m: m['release'])

    # ---------------------------------------------------------------- claims
    claims = []
    # The 41-claim set plus the expansion claims (claim_meta group 'expansion': C141, C142-OEQ, C142-PAS, C143,
    # added to the post 2026-10-13; C161, C162, C157, C166, C079, added 2026-10-17); NOT_COUNTED keys are left out.
    for key, s in summary.items():
        if key in NOT_COUNTED:
            continue
        f = fits[key]
        m = meta[key]
        added = m.get('group') == 'expansion'
        cs = claim_set_exp.get(m.get('code', key), {}) if added else claim_set.get(key, {})
        tv = tiered.get(key, {})
        rows = [r for r in long_rows if r['key'] == key]
        reliable_pts = {p['model']: p for p in f['points']}
        anchor_date = anch[key]['anchor_date']   # the reproduction model's release (C143: the paper date)
        nulls = {num(r['null'], 6) for r in rows}
        per_model_null = len(nulls) > 1
        points = []
        for r in rows:
            fp = reliable_pts.get(r['model'])
            rel = r['reliable'] == 'True'
            points.append(dict(
                model=r['model'], name=r['name'], release=r['release'], cls=r['cls'],
                open=bool(classes.get(r['model'], {}).get('open')), size=classes.get(r['model'], {}).get('size'),
                value=num(r['value']), lo=num(r['lo']), hi=num(r['hi']), n=num(r['n'], 0),
                null=num(r['null'], 6),
                e=num(r['e']), e_lo=num(r['e_lo']), e_hi=num(r['e_hi']),
                reliable=rel,
                outlier=bool(fp and fp.get('out_t')),
                before_anchor=bool(anchor_date and r['release'] < anchor_date),   # not in the anchored fit
                dead_consistent=(r['dead_consistent'] == 'True'),
                note=r['note'][:160],
            ))
        a, af = anch[key], anch_fits[key]
        fit = None
        if af.get('curve'):
            # the anchored curve starts at the reproduction (the diamond) and runs to Sep 2026 (fits_anchored.py)
            fit = dict(
                shape=af['shape'], n=af['n'], n_before_anchor=af['n_before'], end=num(af['end']),
                end_lo=num(af['end_ci'][0]), end_hi=num(af['end_ci'][1]), anchored=True,
                anchor_p=num(af.get('anchor_p')),
                dates=af['dates'],
                curve=[num(v) for v in af['curve']],
                band_lo=[num(v) for v in af['band_lo']],
                band_hi=[num(v) for v in af['band_hi']],
            )
        death = a.get('death') or None
        claims.append(dict(
            key=key, code=m.get('code', key),
            title=m['title'], label=cs.get('label', m['title']), claim=CLAIM_FIX.get(key, cs.get('claim', '')).replace('summarises', 'summarizes'),
            group=DISPLAY_GROUP_OVERRIDES.get(key, cs.get('group', '')), claim_type=cs.get('claim_type', ''), added=added,
            category=a['category'], class_split=class_split(anch_cls[key]),
            metric_label=m['metric_label'], metric_scale=num(m.get('metric_scale'), 0) or 1, unit=unit_of(m['metric_label']),
            paper_value=num(m.get('paper_value'), 6), paper_value_note=m.get('paper_value_note', '').replace('behaviour', 'behavior').replace('dead value', 'expiration value'),
            paper_models=m.get('paper_models', ''),
            repro_value=num(m.get('repro_value'), 6), repro_ci=[num(x, 6) for x in (m.get('repro_ci') or [])] or None,
            repro_model=m.get('repro_model') or '',
            paper_date=m['paper_date'], paper_date_note=m.get('paper_date_note', ''),
            paper_date_source=m['paper_date_source'], paper_date_kind=m['paper_date_kind'],
            paper_date_status=m['paper_date_status'],
            citation=m.get('citation', ''), short_cite=short_cite(m.get('citation', '')), url=m.get('url', ''),
            paper_title=paper_title(key, m.get('citation', '')), paper_is_proxy=(key == 'TOY-01'),
            null=num(f['null'], 6), per_model_null=per_model_null, paper=num(f['paper'], 6),
            n=int(s['n']), n_outliers=int(s['n_out_time']), n_rows=len(points), n_unreliable=sum(1 for p in points if not p['reliable']),
            end=num(a['end']), end_lo=num(a['end_lo']), end_hi=num(a['end_hi']),
            death=death, death_lo=a.get('death_lo') or None, death_hi=a.get('death_hi') or None,
            death_kind=a.get('death_kind') or None,
            recent_n=int(a['recent_n']), recent_dead=int(a['recent_dead']),
            class_ends={g: num((anch_cls[key][g] or {}).get('end')) for g in ('open', 'closed', 'small', 'large')},
            class_verdicts={g: (anch_cls[key][g] or {}).get('category') for g in ('open', 'closed', 'small', 'large')},
            repro_tier=tv.get('tier', ''), repro_verdict=tv.get('final_class', ''), repro_model_match=tv.get('model_match', ''),
            fit=fit, points=points,
            category_all_models=summary_all.get(key, {}).get('category'),
            dead_strength=dead_strength(a, summary_all[key]) if a['category'] == 'dead' else None,
        ))
    assert len(claims) == 50, len(claims)
    ids = claim_ids(claims)
    for c in claims:
        c['id'] = ids[c['key']]

    # ---------------------------------------------------------------- reproduction on the paper's model, on the e scale
    # e = (repro - dead) / (paper - dead) with the same dead value and paper anchor as the panel curves, so e = 1 is the
    # paper. Exceptions: C044's reproduction used the paper's binary self-preference estimator (null 0.5), not the panel's
    # own-minus-neutral excess, so its e is the reproduction pass's (final_verdicts_tiered.csv); TOY-01 has no paper number, its
    # anchor is our original-model reproduction (40% wrong).
    for c in claims:
        n, pa, r, ci = c['null'], c['paper'], c['repro_value'], c['repro_ci']
        tv = tiered.get(c['key'], {})
        note = ''
        if r is not None and pa != n:
            e = (r - n) / (pa - n)
            lohi = sorted((x - n) / (pa - n) for x in ci) if ci and None not in ci else [None, None]
        elif c['added']:
            e, lohi = None, [None, None]
        else:
            e, lohi = num(tv.get('e'), 6), [num(tv.get('e_lo'), 6), num(tv.get('e_hi'), 6)]
            note = "the reproduction's own estimator (paper's binary choice vs a 0.5 null), from the reproduction pass"
        # TOY-01 (strawberry) has no paper number; it is shown as the illustrative figure at the top of the post instead
        c['in_repro_check'] = c['key'] != 'TOY-01' and e is not None
        c['repro_e'], c['repro_e_lo'], c['repro_e_hi'], c['repro_e_note'] = num(e), num(lohi[0]), num(lohi[1]), note
        rm = c['repro_model'].split(' ')[0]
        c['repro_model_short'] = rm
        c['repro_release'] = REPRO_RELEASE.get(rm)
        if c['added'] and e is not None:
            # the reproduction tier rule: significant, and the paper
            # value inside our 95% CI or |e - 1| <= 0.25 -> two-sided; significant and e > 1.25 -> one-sided
            c['repro_verdict'] = reproduction_tier(e, *lohi)
            c['repro_model_match'] = EXPANSION_MATCH[c['key']]
        if c['in_repro_check']:
            # Earlier reproduction labels can predate the metric now shown on the page.
            # Keep the historical verdict, but derive the displayed tier consistently.
            c['repro_tier'] = reproduction_tier(e, *lohi)
        assert c['repro_e'] is not None or (c['added'] and not c['in_repro_check']), c['key']
        assert c['repro_e'] is None or c['repro_release'], c['key']
    cat_order = {c: i for i, c in enumerate(CATEGORIES)}
    claims.sort(key=lambda c: (cat_order[c['category']], -(c['end'] or 0)))

    # ---------------------------------------------------------------- strawberry (C101 / TOY-01)
    straw = []
    strawberry_sources = {r['model']: r['source'] for r in long_rows if r['key'] == 'TOY-01'}
    assert set(strawberry_sources) == set(models), 'strawberry source grid must match the selected panel'
    for mid in sorted(strawberry_sources):
        d = RMS / 'analysis/inputs/C101_panel' / mid
        j = json.load(open(d / 'metrics.json'))[mid]
        pp = j['per_prompt']
        mm = models[mid]
        straw.append(dict(
            model=mid, name=mm['name'], developer=mm['developer'], release=mm['release'], open=mm['open'], anchor=False,
            r_toy=dict(correct=pp['r_toy']['k_correct'], scored=pp['r_toy']['scored'], answers=pp['r_toy']['answer_counts']),
            r_wording=dict(correct=pp['r_wording']['k_correct'], scored=pp['r_wording']['scored'], answers=pp['r_wording']['answer_counts']),
            s_control=dict(correct=pp['s_control']['k_correct'], scored=pp['s_control']['scored'], answers=pp['s_control']['answer_counts']),
        ))
    anchors = {'openai-direct_gpt-4o-2024-05-13': ('GPT-4o (2024-05-13)', '2024-05-13'),
               'openai-direct_gpt-4o-2024-08-06': ('GPT-4o (2024-08-06)', '2024-08-06')}
    for d in sorted((EXP / 'run_minimal_setups/analysis/inputs/C101_reproduction').glob('openai-direct_*')):
        j = json.load(open(d / 'metrics.json'))
        k = list(j)[0]
        pp = j[k]['per_prompt']
        name, rel = anchors[d.name]
        straw.append(dict(
            model=d.name, name=name, developer='OpenAI', release=rel, open=False, anchor=True,
            r_toy=dict(correct=pp['r_toy']['k_correct'], scored=pp['r_toy']['scored'], answers=pp['r_toy']['answer_counts']),
            r_wording=dict(correct=pp['r_wording']['k_correct'], scored=pp['r_wording']['scored'], answers=pp['r_wording']['answer_counts']),
            s_control=dict(correct=pp['s_control']['k_correct'], scored=pp['s_control']['scored'], answers=pp['s_control']['answer_counts']),
        ))
    straw.sort(key=lambda s: s['release'])
    assert len(straw) == len(models) + 2, len(straw)   # every panel model plus the two GPT-4o anchors

    # ---------------------------------------------------------------- reproduction summary (from the tiered verdicts)
    def tier_counts():
        out = {}
        for r in tiered.values():
            out[r['final_class']] = out.get(r['final_class'], 0) + 1
        return out
    sweep = dict(
        n_claims=len(tiered), classes=tier_counts(),
        excluded=[dict(key=r['key'], code=r['code'], label=claim_set.get(r['key'], {}).get('label', r['key']),
                       paper=claim_set.get(r['key'], {}).get('paper', ''), note=r['tier_note'],
                       paper_value=num(r['paper_value'], 4), our_value=num(r['our_value'], 4), our_ci=r['our_ci95'],
                       e=num(r['e'], 3), e_lo=num(r['e_lo'], 3), e_hi=num(r['e_hi'], 3),
                       final_class=r['final_class'], reason=r['reason'])
                  for r in tiered.values() if r['tier'] == 'excluded'],
        match_counts={},
    )
    for r in tiered.values():
        mm = r['model_match'].split(' ')[0].split('(')[0] if r['model_match'] else 'none'
        sweep['match_counts'][mm] = sweep['match_counts'].get(mm, 0) + 1

    apply_display_fit_overrides(claims)
    generated = dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')
    data = dict(generated=generated, categories=CATEGORIES, models=model_list, claims=claims, strawberry=straw, sweep=sweep)
    HERE.mkdir(exist_ok=True)
    (HERE / 'results.json').write_text(json.dumps(public_results(data), separators=(',', ':')))

    # ---------------------------------------------------------------- numbers quoted in the prose ({{name}} in the .md files)
    def mon(iso):
        return dt.date.fromisoformat(iso).strftime('%b %Y')
    sc = sweep['classes']
    may = next(s for s in straw if '2024-05-13' in s['model'])
    numbers = {f'n_{c}': sum(1 for x in claims if x['category'] == c) for c in CATEGORIES}
    numbers.update(
        n_claims=len(claims), n_models=len(model_list),
        n_pairs_expected=len(claims) * len(model_list),
        n_pairs_recorded=sum(len(c['points']) for c in claims),
        n_pairs_missing=len(claims) * len(model_list) - sum(len(c['points']) for c in claims),
        n_scored_min=min(sum(p['reliable'] for p in c['points']) for c in claims),
        n_scored_max=max(sum(p['reliable'] for p in c['points']) for c in claims),
        n_selected=sum(1 for c in claims if not c['added']), n_added=sum(1 for c in claims if c['added']),
        n_label_changes_all=sum(1 for c in claims if c['category_all_models'] and c['category_all_models'] != c['category']),
        n_dead_strong=sum(1 for c in claims if c['dead_strength'] and c['dead_strength']['level'] == 'strong'),
        n_dead_moderate=sum(1 for c in claims if c['dead_strength'] and c['dead_strength']['level'] == 'moderate'),
        n_dead_weak=sum(1 for c in claims if c['dead_strength'] and c['dead_strength']['level'] == 'weak'),
        n_open=sum(1 for m in model_list if m['open']), n_closed=sum(1 for m in model_list if not m['open']),
        model_span=f"{mon(model_list[0]['release'])} to {mon(model_list[-1]['release'])}",
        sw_reproduced=sc.get('reproduced', 0), sw_partial=sc.get('partially_reproduced', 0),
        sw_not=sc.get('not_reproduced', 0), sw_nt=sc.get('not_testable', 0),
        straw_4o_may_wrong=may['r_toy']['scored'] - may['r_toy']['correct'],
        generated=generated[:10],
        n_repro_check=sum(1 for c in claims if c['in_repro_check']),
        repro_within25=sum(1 for c in claims if c['in_repro_check'] and abs(c['repro_e'] - 1) <= 0.25),
        repro_ci_has_paper=sum(1 for c in claims if c['in_repro_check'] and c['repro_e_lo'] is not None and c['repro_e_lo'] <= 1 <= c['repro_e_hi']),
        repro_exact=sum(1 for c in claims if c['in_repro_check'] and c['repro_model_match'].startswith('exact')),
        repro_two_sided=sum(1 for c in claims if c['in_repro_check'] and c['repro_tier'] == 'two_sided'),
        repro_one_sided=sum(1 for c in claims if c['in_repro_check'] and c['repro_tier'] == 'one_sided'),
        # "paper vs ours" for claims dropped before the post (public_numbers removes it from data/numbers.json)
        dropped={e['key']: f"{e['paper_value']} vs {e['our_value']} {e['our_ci'] or ''}".strip() for e in sweep['excluded']},
    )
    (HERE / 'numbers.json').write_text(json.dumps(public_numbers(numbers), indent=1) + '\n')
    print('wrote analysis/post/results.json and analysis/post/numbers.json')


if __name__ == '__main__':
    main()
