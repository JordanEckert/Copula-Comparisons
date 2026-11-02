from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance, norm, t as student_t
from sklearn.tree import DecisionTreeClassifier

TARGET = 'target'
_NUM_DTYPES = (np.float64, np.float32, np.int64, np.int32, float, int)


def _num_cat_cols(df: pd.DataFrame, exclude_target: bool = True) -> Tuple[List[str], List[str]]:
    cols = [c for c in df.columns if not (exclude_target and c == TARGET)]
    num = [c for c in cols if df[c].dtype in _NUM_DTYPES]
    cat = [c for c in cols if c not in num]
    return num, cat


def _zscore(real: pd.DataFrame, other: pd.DataFrame, cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    R = np.nan_to_num(real[cols].values.astype(float), nan=0.0)
    O = np.nan_to_num(other[cols].values.astype(float), nan=0.0)
    mu = R.mean(axis=0)
    sd = R.std(axis=0)
    sd[sd == 0] = 1.0
    return (R - mu) / sd, (O - mu) / sd


def standardized_wasserstein(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> float:
    num, _ = _num_cat_cols(real_df)
    if not num:
        return float('nan')
    R, S = _zscore(real_df, synth_df, num)
    return float(np.mean([wasserstein_distance(R[:, j], S[:, j]) for j in range(len(num))]))


def _dcr_vector(A: np.ndarray, B: np.ndarray, rng: np.random.Generator,
                max_n: int = 5000) -> np.ndarray:
    if len(A) > max_n:
        A = A[rng.choice(len(A), max_n, replace=False)]
    if len(B) > max_n:
        B = B[rng.choice(len(B), max_n, replace=False)]
    return cdist(A, B, metric='euclidean').min(axis=1)


def dcr_stats(train_df: pd.DataFrame, synth_df: pd.DataFrame,
              test_df: pd.DataFrame, seed: int = 0) -> Dict[str, float]:
    num, _ = _num_cat_cols(train_df)
    nan = float('nan')
    if not num:
        return {'dcr': nan, 'dcr_ratio': nan, 'dcr_p5_ratio': nan}
    rng = np.random.default_rng(seed)
    R, S = _zscore(train_df, synth_df, num)
    _, T = _zscore(train_df, test_df, num)
    d_synth = _dcr_vector(S, R, rng)
    d_ref = _dcr_vector(T, R, rng)
    mean_s, p5_s = float(d_synth.mean()), float(np.percentile(d_synth, 5))
    mean_r, p5_r = float(d_ref.mean()), float(np.percentile(d_ref, 5))
    return {
        'dcr': mean_s,
        'dcr_ratio': mean_s / mean_r if mean_r > 0 else nan,
        'dcr_p5_ratio': p5_s / p5_r if p5_r > 0 else nan,
    }


def _cramers_v(a: pd.Series, b: pd.Series) -> float:
    ct = pd.crosstab(a, b).values.astype(float)
    n = ct.sum()
    if n == 0 or min(ct.shape) < 2:
        return 0.0
    exp = np.outer(ct.sum(1), ct.sum(0)) / n
    with np.errstate(divide='ignore', invalid='ignore'):
        chi2 = np.nansum(np.where(exp > 0, (ct - exp) ** 2 / exp, 0.0))
    return float(np.sqrt(chi2 / (n * (min(ct.shape) - 1))))


def _correlation_ratio(cat: pd.Series, num: pd.Series) -> float:
    y = np.nan_to_num(num.values.astype(float), nan=0.0)
    codes = pd.Categorical(cat).codes
    ss_tot = ((y - y.mean()) ** 2).sum()
    if ss_tot == 0:
        return 0.0
    ss_between = 0.0
    for k in np.unique(codes):
        yk = y[codes == k]
        ss_between += len(yk) * (yk.mean() - y.mean()) ** 2
    return float(np.sqrt(ss_between / ss_tot))


def _association_matrix(df: pd.DataFrame, num: List[str], cat: List[str]) -> np.ndarray:
    cols = num + cat
    d = len(cols)
    A = np.eye(d)
    if num:
        A[:len(num), :len(num)] = np.nan_to_num(df[num].corr().values, nan=0.0)
    for i in range(len(num), d):
        for j in range(i + 1, d):
            A[i, j] = A[j, i] = _cramers_v(df[cols[i]], df[cols[j]])
        for j in range(len(num)):
            A[i, j] = A[j, i] = _correlation_ratio(df[cols[i]], df[cols[j]])
    return A


def association_l2(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> float:
    num, cat = _num_cat_cols(real_df)
    if len(num) + len(cat) < 2:
        return float('nan')
    A_r = _association_matrix(real_df, num, cat)
    A_s = _association_matrix(synth_df, num, cat)
    return float(np.linalg.norm(A_r - A_s, ord='fro'))


def _design(real_df: pd.DataFrame, other_df: pd.DataFrame,
            include_target: bool) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    num, cat = _num_cat_cols(real_df, exclude_target=not include_target)
    parts_r, parts_o, names = [], [], []
    if num:
        R, O = _zscore(real_df, other_df, num)
        parts_r.append(R)
        parts_o.append(O)
        names.extend(num)
    for c in cat:
        levels = list(pd.Categorical(real_df[c].astype(str)).categories)
        for lv in levels[1:]:
            parts_r.append((real_df[c].astype(str).values == lv).astype(float)[:, None])
            parts_o.append((other_df[c].astype(str).values == lv).astype(float)[:, None])
            names.append(f'{c}={lv}')
    X_r = np.hstack(parts_r) if parts_r else np.empty((len(real_df), 0))
    X_o = np.hstack(parts_o) if parts_o else np.empty((len(other_df), 0))
    return X_r, X_o, names


def _pmse(real_df: pd.DataFrame, other_df: pd.DataFrame) -> float:
    r, o = real_df, other_df
    if TARGET in r.columns and r[TARGET].dtype not in _NUM_DTYPES:
        r = r.copy(); o = o.copy()
        r[TARGET] = r[TARGET].astype(str); o[TARGET] = o[TARGET].astype(str)
    X_r, X_o, _ = _design(r, o, include_target=True)
    X = np.vstack([X_r, X_o])
    y = np.concatenate([np.zeros(len(X_r)), np.ones(len(X_o))])
    c = len(X_o) / len(y)
    X = X[:, X.std(axis=0) > 0]
    if X.shape[1] == 0:
        return float('nan')
    clf = DecisionTreeClassifier(min_samples_leaf=20, random_state=0)
    clf.fit(X, y)
    p = clf.predict_proba(X)[:, 1]
    return float(np.mean((p - c) ** 2))


def pmse_ratio(train_df: pd.DataFrame, synth_df: pd.DataFrame,
               test_df: pd.DataFrame, seed: int = 0,
               max_n: int = 5000) -> float:
    rng = np.random.default_rng(seed)
    n = min(len(synth_df), len(test_df), max_n)
    s = synth_df.iloc[rng.choice(len(synth_df), n, replace=False)]
    t = test_df.iloc[rng.choice(len(test_df), n, replace=False)]
    ref = _pmse(train_df, t)
    obs = _pmse(train_df, s)
    return obs / ref if ref > 0 else float('nan')


def _ols(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    n, p = X.shape
    XtX = X.T @ X + 1e-8 * np.eye(p)
    beta = np.linalg.solve(XtX, X.T @ y)
    resid = y - X @ beta
    dof = max(n - p, 1)
    sigma2 = float(resid @ resid) / dof
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(XtX)))
    return beta, se, dof


def _logit_irls(X: np.ndarray, y: np.ndarray, max_iter: int = 50,
                tol: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, int]:
    n, p = X.shape
    beta = np.zeros(p)
    ridge = 1e-8 * np.eye(p)
    for _ in range(max_iter):
        eta = X @ beta
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = mu * (1 - mu)
        H = (X * w[:, None]).T @ X + ridge
        g = X.T @ (y - mu)
        step = np.linalg.solve(H, g)
        ll_old = float(np.sum(y * eta - np.logaddexp(0, eta)))
        for _h in range(10):
            b_new = beta + step
            eta_new = X @ b_new
            if float(np.sum(y * eta_new - np.logaddexp(0, eta_new))) >= ll_old - 1e-12:
                break
            step *= 0.5
        beta = b_new
        if np.max(np.abs(step)) < tol:
            break
    eta = X @ beta
    mu = 1.0 / (1.0 + np.exp(-eta))
    w = mu * (1 - mu)
    H = (X * w[:, None]).T @ X + ridge
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    return beta, se, 0


def _analyst_y(real_df: pd.DataFrame, other_df: pd.DataFrame,
               task_type: str) -> Tuple[np.ndarray, np.ndarray, str]:
    if task_type == 'regression':
        return (real_df[TARGET].values.astype(float),
                other_df[TARGET].values.astype(float), 'ols')
    yr = real_df[TARGET].astype(str).values
    yo = other_df[TARGET].astype(str).values
    if task_type == 'binclass':
        levels = sorted(set(yr))
        pos = levels[-1]
    else:
        vals, counts = np.unique(yr, return_counts=True)
        pos = vals[np.argmax(counts)]
    return (yr == pos).astype(float), (yo == pos).astype(float), 'logit'


def inference_fidelity(real_df: pd.DataFrame, synth_df: pd.DataFrame,
                       task_type: str, alpha: float = 0.05,
                       min_count: int = 5) -> Dict[str, float]:
    nan = float('nan')
    out = {'ci_overlap': nan, 'sig_agree': nan, 'coverage': nan}
    if TARGET not in real_df.columns:
        return out
    X_r, X_s, names = _design(real_df, synth_df, include_target=False)
    y_r, y_s, family = _analyst_y(real_df, synth_df, task_type)
    if family == 'logit' and (y_s.sum() == 0 or y_s.sum() == len(y_s)):
        return out

    ok = []
    for j in range(X_r.shape[1]):
        cr, cs = X_r[:, j], X_s[:, j]
        if cr.std() == 0 or cs.std() == 0:
            continue
        if '=' in names[j] and (cr.sum() < min_count or cs.sum() < min_count):
            continue
        ok.append(j)
    if not ok:
        return out
    Xr = np.column_stack([np.ones(len(X_r)), X_r[:, ok]])
    Xs = np.column_stack([np.ones(len(X_s)), X_s[:, ok]])

    fit = _ols if family == 'ols' else _logit_irls
    try:
        b_r, se_r, dof_r = fit(Xr, y_r)
        b_s, se_s, dof_s = fit(Xs, y_s)
    except np.linalg.LinAlgError:
        return out

    def crit(dof):
        return student_t.ppf(1 - alpha / 2, dof) if dof > 0 else norm.ppf(1 - alpha / 2)

    q_r, q_s = crit(dof_r), crit(dof_s)
    L_r, U_r = b_r - q_r * se_r, b_r + q_r * se_r
    L_s, U_s = b_s - q_s * se_s, b_s + q_s * se_s

    overlaps, agree, cover = [], [], []
    for k in range(1, Xr.shape[1]):
        inter = min(U_r[k], U_s[k]) - max(L_r[k], L_s[k])
        w_r, w_s = U_r[k] - L_r[k], U_s[k] - L_s[k]
        if w_r <= 0 or w_s <= 0 or not np.isfinite(inter):
            continue
        overlaps.append(max(0.0, 0.5 * (inter / w_r + inter / w_s)))
        sig_r = (L_r[k] > 0) or (U_r[k] < 0)
        sig_s = (L_s[k] > 0) or (U_s[k] < 0)
        same_sign = np.sign(b_r[k]) == np.sign(b_s[k])
        agree.append(float((sig_r == sig_s) and (same_sign or not sig_r)))
        cover.append(float(L_s[k] <= b_r[k] <= U_s[k]))
    if not overlaps:
        return out
    return {'ci_overlap': float(np.mean(overlaps)),
            'sig_agree': float(np.mean(agree)),
            'coverage': float(np.mean(cover))}


def _knn_radius(D: np.ndarray, k: int) -> np.ndarray:
    return np.partition(D, k - 1, axis=1)[:, k - 1]


def _support_curve(D_ab: np.ndarray, radii_a: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    order = np.argsort(radii_a)
    inside = D_ab[order] <= radii_a[order][:, None]
    hit = inside.any(axis=0)
    first = np.where(hit, inside.argmax(axis=0), len(order))
    n_a = len(order)
    return np.array([np.mean(first < max(1, int(np.ceil(a * n_a)))) if a > 0 else 0.0
                     for a in alphas])


def alpha_beta_authenticity(real_df: pd.DataFrame, synth_df: pd.DataFrame,
                            seed: int = 0, max_n: int = 5000, k: int = 5,
                            n_alphas: int = 30) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    r, s = real_df, synth_df
    if TARGET in r.columns and r[TARGET].dtype not in _NUM_DTYPES:
        r = r.copy(); s = s.copy()
        r[TARGET] = r[TARGET].astype(str); s[TARGET] = s[TARGET].astype(str)
    X_r_full, X_s, _ = _design(r, s, include_target=True)
    if len(X_r_full) > max_n:
        X_r_full = X_r_full[rng.choice(len(X_r_full), max_n, replace=False)]
    if len(X_s) > max_n:
        X_s = X_s[rng.choice(len(X_s), max_n, replace=False)]
    nan = float('nan')
    if min(len(X_r_full), len(X_s)) <= k + 1:
        return {'alpha_precision': nan, 'beta_recall': nan, 'authenticity': nan}

    D_rs_full = cdist(X_r_full, X_s)
    D_rr_full = cdist(X_r_full, X_r_full); np.fill_diagonal(D_rr_full, np.inf)
    real_nn = D_rr_full.min(axis=1)
    synth_to_real = D_rs_full.min(axis=0)
    synth_to_real_idx = D_rs_full.argmin(axis=0)
    authentic = real_nn[synth_to_real_idx] < synth_to_real

    n = min(len(X_r_full), len(X_s))
    X_r = X_r_full[rng.choice(len(X_r_full), n, replace=False)]
    X_s = X_s[rng.choice(len(X_s), n, replace=False)]
    D_rs = cdist(X_r, X_s)
    D_rr = cdist(X_r, X_r); np.fill_diagonal(D_rr, np.inf)
    D_ss = cdist(X_s, X_s); np.fill_diagonal(D_ss, np.inf)
    rad_r = _knn_radius(D_rr, k)
    rad_s = _knn_radius(D_ss, k)

    alphas = np.linspace(0, 1, n_alphas)
    prec = _support_curve(D_rs, rad_r, alphas)
    rec = _support_curve(D_rs.T, rad_s, alphas)

    da = alphas[1] - alphas[0]
    return {
        'alpha_precision': float(1 - 2 * np.sum(np.abs(alphas - prec)) * da),
        'beta_recall': float(1 - 2 * np.sum(np.abs(alphas - rec)) * da),
        'authenticity': float(np.mean(authentic)),
    }
