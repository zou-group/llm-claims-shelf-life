"""AlpacaEval length-controlled win rate (metrics/glm_winrate.py, ``length_controlled_v1``) in numpy/scipy.

The official implementation needs sklearn + patsy (not installed here). This module re-implements exactly what
``get_length_controlled_winrate(annotations, glm_name='length_controlled_v1', baseline=None)`` computes:

  features   np.tanh(std_delta_len), instruction_difficulty, not_gamed_baseline (no intercept), where
             std_delta_len = (len(output_1) - len(output_2)) / std (pandas ddof=1) over the model's annotations,
             output_1 = baseline, output_2 = model, preference -> (pref with 0 replaced by 1.5) - 1 in [0, 1]
  training   the model's annotations with non-NaN preference (weight 1) concatenated after the two released gamed
             baselines in df_gamed.csv (HF tatsu-lab/alpaca_eval @ 2edc6fa; not_gamed_baseline False, weight
             regularize_to_baseline_lambda / 2 = 0.1)
  fit        fit_LogisticRegressionCV(is_ytrue_proba=True): soft labels via duplicated rows (label 1 weighted p*w,
             label 0 weighted (1-p)*w), L1 penalty, no intercept, LogisticRegressionCV(Cs=10 -> logspace(-4, 4, 10),
             GroupKFold(5) over original rows, scoring = -logloss_continuous with the true sample weights), best C by
             the fold-summed score, refit on all rows with that C
  LC         mean over ALL the model's annotations (NaN preferences included) of logistic(w . x) with
             std_delta_len = 0, i.e. mean_i logistic(w_not_gamed + w_difficulty * difficulty_i), times 100.

Deviation: liblinear (coordinate descent, tol 1e-4) is replaced by an exact L-BFGS-B solve of the same objective
(||w||_1 + C * sum_i s_i * logloss_i, w split into non-negative parts). Fold assignment reproduces sklearn's
GroupKFold algorithm with this machine's numpy argsort. Validated against the released weights and leaderboard LC
values in tests/claims/test_c046.py.
"""
import functools

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..paths import INITIAL

DF_GAMED = INITIAL / 'reproductions/C046/df_gamed.csv'      # HF tatsu-lab/alpaca_eval @ 2edc6fa (download_manifest.json)
CS = np.logspace(-4, 4, 10)
N_SPLITS = 5
LAMBDA = 0.2
COLUMNS = ('np.tanh(std_delta_len)', 'instruction_difficulty', 'not_gamed_baseline.astype(float)')


@functools.lru_cache(maxsize=None)
def gamed():
    g = pd.read_csv(DF_GAMED)
    return g.drop(columns=['model'])


def _softplus(x):
    return np.logaddexp(0, x)


def _logistic(x):
    return np.exp(-np.logaddexp(0, -x))


def _fit_l1(X, p, w, C):
    """argmin ||beta||_1 + C * sum_i w_i [p_i softplus(-z_i) + (1 - p_i) softplus(z_i)], z = X beta (no intercept)."""
    d = X.shape[1]

    def f(theta):
        beta = theta[:d] - theta[d:]
        z = X @ beta
        loss = np.sum(w * (p * _softplus(-z) + (1 - p) * _softplus(z)))
        g_beta = X.T @ (w * (_logistic(z) - p))
        obj = theta.sum() / C + loss
        grad = np.concatenate([g_beta + 1 / C, -g_beta + 1 / C])
        return obj, grad
    res = minimize(f, np.zeros(2 * d), jac=True, method='L-BFGS-B', bounds=[(0, None)] * (2 * d),
                   options={'ftol': 1e-15, 'gtol': 1e-10, 'maxiter': 10000})
    return res.x[:d] - res.x[d:]


def group_kfold(n_groups, n_splits=N_SPLITS):
    """sklearn GroupKFold fold of each group when every group has 2 samples (the duplicated soft-label rows)."""
    sizes = np.full(n_groups, 2, dtype=np.int64)
    order = np.argsort(sizes)[::-1]
    per_fold = np.zeros(n_splits)
    fold = np.zeros(n_groups, dtype=int)
    for g in order:
        k = int(np.argmin(per_fold))
        per_fold[k] += 2
        fold[g] = k
    return fold


def _cv_logloss(X, p, w, fold, C):
    score = 0.0
    for k in range(N_SPLITS):
        tr, te = fold != k, fold == k
        beta = _fit_l1(X[tr], p[tr], w[tr], C)
        y = np.clip(_logistic(X[te] @ beta), 1e-15, 1 - 1e-15)
        score += -np.mean(w[te] * (p[te] * np.log(y) + (1 - p[te]) * np.log(1 - y)))
    return score


def design(preference, len_baseline, len_model, difficulty):
    """Official featurisation of one model's annotations (all rows, NaN preferences kept for prediction)."""
    pref = pd.Series(preference, dtype=float).replace({0.0: 1.5}) - 1
    delta = pd.Series(np.asarray(len_baseline, float) - np.asarray(len_model, float))
    std_delta = delta / delta.std()
    X = np.column_stack([np.tanh(std_delta.to_numpy()), np.asarray(difficulty, float), np.ones(len(pref))])
    return X, pref.to_numpy()


def fit_length_controlled(preference, len_baseline, len_model, difficulty, regularize=True):
    """length_controlled_v1 on one model's annotations (preference in [1, 2] = official scale, 2 = model wins).

    Returns {'length_controlled_winrate', 'weights' {column: value}, 'C', 'n_train_model_rows'}."""
    X, p = design(preference, len_baseline, len_model, difficulty)
    keep = ~np.isnan(p)
    Xt, pt, wt = X[keep], p[keep], np.ones(keep.sum())
    if regularize:
        g = gamed()
        Xg = np.column_stack([np.tanh(g['std_delta_len'].to_numpy()), g['instruction_difficulty'].to_numpy(),
                              g['not_gamed_baseline'].astype(float).to_numpy()])
        Xt = np.vstack([Xg, Xt])
        pt = np.concatenate([g['preference'].to_numpy(float), pt])
        wt = np.concatenate([np.full(len(g), LAMBDA / 2), wt])
    fold = group_kfold(len(pt))
    scores = [_cv_logloss(Xt, pt, wt, fold, C) for C in CS]
    best = int(np.argmin(scores))           # sklearn: argmax of the fold-summed (negative-loss) score, first on ties
    beta = _fit_l1(Xt, pt, wt, CS[best])
    X_test = X.copy()
    X_test[:, 0] = 0.0                      # np.tanh(0): no length difference
    return {'length_controlled_winrate': float(np.mean(_logistic(X_test @ beta)) * 100),
            'weights': dict(zip(COLUMNS, map(float, beta))), 'C': float(CS[best]), 'n_train_model_rows': int(keep.sum())}
