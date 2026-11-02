"""Curve shapes and AICc used by fits_anchored.py, plus the time axis (T0, T_END) and the effect clip range.

Shapes: constant / linear / exponential / logistic, fitted by least squares from several starts.
"""
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

warnings.filterwarnings('ignore')

EXP = Path(__file__).resolve().parents[2]
PROJ = EXP.parents[1]
T0 = pd.Timestamp('2024-01-01')
T_END = (pd.Timestamp('2026-09-30') - T0).days / 365.25
CLIP = (-1.0, 3.0)


def f_const(p, t):
    return np.full_like(t, p[0], dtype=float)


def f_lin(p, t):
    return p[0] + p[1] * t


def f_exp(p, t, c=1.5):
    return p[0] + p[1] * np.exp(-p[2] * (t - c))


def f_logit(p, t):
    lo, hi, t50, s = p
    return lo + (hi - lo) / (1 + np.exp(np.clip((t - t50) / s, -50, 50)))


SHAPES = {'constant': (f_const, 1), 'linear': (f_lin, 2), 'exponential': (f_exp, 3), 'logistic': (f_logit, 4)}


def starts(name, t, y):
    ym, yr = float(np.mean(y)), float(np.ptp(y) + 1e-6)
    if name == 'constant':
        return [[ym]], ([-np.inf], [np.inf])
    if name == 'linear':
        return [[ym, 0.0]], ([-np.inf, -np.inf], [np.inf, np.inf])
    span = float(np.ptp(t) + 1e-6)
    if name == 'exponential':
        S = [[f, a, k / span] for f in (min(y), max(y), ym) for a in (yr, -yr) for k in (-1.5, 1.0, 4.0)]
        return S, ([-10, -50, -6 / span], [10, 50, 6 / span])
    S = [[lo, hi, t50, s * span] for lo, hi in ((min(y), max(y)), (max(y), min(y)))
         for t50 in np.quantile(t, [0.25, 0.5, 0.75]) for s in (0.07, 0.25)]
    return S, ([-10, -10, t.min() - 0.2 * span, 0.02 * span], [10, 10, t.max() + 0.2 * span, 1.5 * span])


def fit_shape(name, t, y, center):
    f, _ = SHAPES[name]
    g = (lambda p, x: f_exp(p, x, center)) if name == 'exponential' else f
    S, bounds = starts(name, t, y)
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


def aicc(rss, n, k):
    if n - k - 1 <= 0:
        return np.inf
    return n * math.log(max(rss, 1e-12) / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1)
