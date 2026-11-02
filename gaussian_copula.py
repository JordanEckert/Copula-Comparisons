from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

from copula_utils import reconstruct, is_categorical


def _encode_full(train_df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    cat_cols = [c for c in train_df.columns if is_categorical(train_df[c])]
    num_cols = [c for c in train_df.columns if c not in cat_cols]

    parts: List[np.ndarray] = []
    col_names: List[str] = []

    for c in num_cols:
        parts.append(train_df[c].values.astype(np.float64).reshape(-1, 1))
        col_names.append(c)

    for c in cat_cols:
        dummies = pd.get_dummies(train_df[c], prefix=str(c), prefix_sep='_')
        parts.append(dummies.values.astype(np.float64))
        col_names.extend(dummies.columns.tolist())

    M = np.hstack(parts) if parts else np.empty((len(train_df), 0))
    return M, col_names


def _nearest_psd(S: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    w, V = np.linalg.eigh(S)
    w = np.clip(w, eps, None)
    S = (V * w) @ V.T
    d = np.sqrt(np.diag(S))
    return S / np.outer(d, d)


def fit_gaussian_copula(train_df: pd.DataFrame,
                        seed: Optional[int] = None) -> dict:
    M, col_names = _encode_full(train_df)
    n = M.shape[0]

    R = np.apply_along_axis(lambda x: rankdata(x, method='average'), 0, M)
    Z = norm.ppf(R / (n + 1))

    with np.errstate(invalid='ignore', divide='ignore'):
        Sigma = np.corrcoef(Z, rowvar=False)
    Sigma = np.atleast_2d(Sigma)
    Sigma[np.isnan(Sigma)] = 0.0
    np.fill_diagonal(Sigma, 1.0)
    Sigma = _nearest_psd(Sigma)

    n_num = sum(1 for c in train_df.columns if not is_categorical(train_df[c]))
    return {
        'method': 'gaussian_copula',
        'M': M,
        'Sigma': Sigma,
        'columns': col_names,
        'n_num': n_num,
        'template': train_df.iloc[:0].copy(),
    }


def sample_gaussian_copula(fit: dict, n: int,
                           seed: Optional[int] = None) -> pd.DataFrame:
    if seed is not None:
        np.random.seed(seed)

    M = fit['M']
    Sigma = fit['Sigma']
    col_names = fit['columns']
    n_num = fit['n_num']
    d = M.shape[1]

    Z = np.random.multivariate_normal(np.zeros(d), Sigma, size=n)
    U = norm.cdf(Z)
    X_new = np.zeros((n, d))
    for j in range(n_num):
        X_new[:, j] = np.quantile(M[:, j], U[:, j])

    if n_num < d:
        p = M[:, n_num:].mean(axis=0)
        X_new[:, n_num:] = U[:, n_num:] - (1.0 - p)

    D_full = pd.DataFrame(X_new, columns=col_names)
    return reconstruct(D_full, fit['template'])


def generate_gaussian_copula(train_df: pd.DataFrame,
                             n: int,
                             seed: Optional[int] = None,
                             task_type: Optional[str] = None) -> pd.DataFrame:
    fit = fit_gaussian_copula(train_df, seed=seed)
    return sample_gaussian_copula(fit, n, seed=seed)
