from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from copula_utils import reconstruct, is_categorical


def _encode_full(train_df: pd.DataFrame) -> Tuple[np.ndarray, List[str], int]:
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
    return M, col_names, len(num_cols)


def _rbetacop(n_synth: int, ranks: np.ndarray,
              rng: np.random.Generator) -> np.ndarray:
    n = ranks.shape[0]
    idx = rng.integers(0, n, size=n_synth)
    R = ranks[idx].astype(np.float64)
    return rng.beta(R, n + 1.0 - R)


def _quantile_pp(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    xs = np.sort(x)
    n = xs.shape[0]
    pos = np.arange(1, n + 1) / (n + 1.0)
    return np.interp(u, pos, xs)


def fit_beta_copula(train_df: pd.DataFrame,
                    seed: Optional[int] = None,
                    jitter_eps: float = 1e-10) -> dict:
    rng = np.random.default_rng(seed)

    M, col_names, n_num = _encode_full(train_df)
    Mj = M + rng.uniform(-jitter_eps, jitter_eps, size=M.shape)
    ranks = np.apply_along_axis(
        lambda x: rankdata(x, method='ordinal'), axis=0, arr=Mj
    )
    return {
        'method': 'beta_copula',
        'M': M,
        'ranks': ranks,
        'columns': col_names,
        'n_num': n_num,
        'template': train_df.iloc[:0].copy(),
    }


def sample_beta_copula(fit: dict, n: int, seed: Optional[int] = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    M = fit['M']
    ranks = fit['ranks']
    col_names = fit['columns']
    n_num = fit['n_num']
    d = M.shape[1]

    U = _rbetacop(n, ranks, rng)
    X_new = np.zeros((n, d))
    for j in range(n_num):
        X_new[:, j] = _quantile_pp(M[:, j], U[:, j])

    if n_num < d:
        p = M[:, n_num:].mean(axis=0)
        X_new[:, n_num:] = U[:, n_num:] - (1.0 - p)

    D_full = pd.DataFrame(X_new, columns=col_names)
    return reconstruct(D_full, fit['template'])


def generate_beta_copula(train_df: pd.DataFrame,
                         n: int,
                         seed: Optional[int] = None,
                         task_type: Optional[str] = None,
                         jitter_eps: float = 1e-10) -> pd.DataFrame:
    fit = fit_beta_copula(train_df, seed=seed, jitter_eps=jitter_eps)
    return sample_beta_copula(fit, n, seed=seed)
