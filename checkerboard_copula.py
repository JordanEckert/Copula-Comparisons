from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from copula_utils import reconstruct
from beta_copula import _encode_full, _quantile_pp


def _rcheckerboard(n_synth: int, ranks: np.ndarray,
                   rng: np.random.Generator) -> np.ndarray:
    n = ranks.shape[0]
    idx = rng.integers(0, n, size=n_synth)
    R = ranks[idx].astype(np.float64)
    V = rng.uniform(0.0, 1.0, size=R.shape)
    return (R - V) / n


def fit_checkerboard_copula(train_df: pd.DataFrame,
                            seed: Optional[int] = None,
                            jitter_eps: float = 1e-10) -> dict:
    rng = np.random.default_rng(seed)

    M, col_names, n_num = _encode_full(train_df)
    Mj = M + rng.uniform(-jitter_eps, jitter_eps, size=M.shape)
    ranks = np.apply_along_axis(
        lambda x: rankdata(x, method='ordinal'), axis=0, arr=Mj
    )
    return {
        'method': 'checkerboard_copula',
        'M': M,
        'ranks': ranks,
        'columns': col_names,
        'n_num': n_num,
        'template': train_df.iloc[:0].copy(),
    }


def sample_checkerboard_copula(fit: dict, n: int,
                               seed: Optional[int] = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    M = fit['M']
    ranks = fit['ranks']
    col_names = fit['columns']
    n_num = fit['n_num']
    d = M.shape[1]

    U = _rcheckerboard(n, ranks, rng)
    X_new = np.zeros((n, d))
    for j in range(n_num):
        X_new[:, j] = _quantile_pp(M[:, j], U[:, j])

    if n_num < d:
        p = M[:, n_num:].mean(axis=0)
        X_new[:, n_num:] = U[:, n_num:] - (1.0 - p)

    D_full = pd.DataFrame(X_new, columns=col_names)
    return reconstruct(D_full, fit['template'])


def generate_checkerboard_copula(train_df: pd.DataFrame,
                                 n: int,
                                 seed: Optional[int] = None,
                                 task_type: Optional[str] = None,
                                 jitter_eps: float = 1e-10) -> pd.DataFrame:
    fit = fit_checkerboard_copula(train_df, seed=seed, jitter_eps=jitter_eps)
    return sample_checkerboard_copula(fit, n, seed=seed)
