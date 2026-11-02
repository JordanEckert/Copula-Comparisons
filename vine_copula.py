from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import pyvinecopulib as pv

from copula_utils import is_categorical
from beta_copula import _quantile_pp

FAMILY_SET = [
    pv.BicopFamily.indep, pv.BicopFamily.gaussian, pv.BicopFamily.clayton,
    pv.BicopFamily.gumbel, pv.BicopFamily.frank, pv.BicopFamily.joe,
]


def fit_vine_copula(train_df: pd.DataFrame,
                    seed: Optional[int] = None,
                    jitter_eps: float = 1e-10,
                    num_threads: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    n = len(train_df)

    cols: List[str] = list(train_df.columns)
    var_types: List[str] = []
    U_cols: List[np.ndarray] = []
    U_minus_cols: List[np.ndarray] = []
    num_values = {}
    cat_levels = {}
    cat_cum = {}

    for c in cols:
        s = train_df[c]
        if is_categorical(s):
            cat = pd.Categorical(s)
            codes = cat.codes.astype(np.int64)
            K = len(cat.categories)
            counts = np.bincount(codes, minlength=K).astype(np.float64)
            cum = np.cumsum(counts) / (n + 1.0)
            low = np.concatenate([[0.0], cum[:-1]])
            var_types.append('d')
            U_cols.append(cum[codes])
            U_minus_cols.append(low[codes])
            cat_levels[c] = list(cat.categories)
            cat_cum[c] = cum
        else:
            x = s.values.astype(np.float64)
            xj = x + rng.uniform(-jitter_eps, jitter_eps, size=n)
            var_types.append('c')
            U_cols.append(rankdata(xj, method='ordinal') / (n + 1.0))
            num_values[c] = x

    data = np.column_stack(U_cols + U_minus_cols)
    controls = pv.FitControlsVinecop(
        family_set=FAMILY_SET,
        parametric_method='itau',
        selection_criterion='mbicv',
        select_trunc_lvl=True,
        num_threads=num_threads,
        seeds=[int(seed)] if seed is not None else [],
    )
    vine = pv.Vinecop.from_data(np.asfortranarray(data),
                                var_types=var_types, controls=controls)

    return {
        'method': 'vine_copula',
        'vine': vine,
        'columns': cols,
        'var_types': var_types,
        'num_values': num_values,
        'cat_levels': cat_levels,
        'cat_cum': cat_cum,
        'template': train_df.iloc[:0].copy(),
    }


def sample_vine_copula(fit: dict, n: int,
                       seed: Optional[int] = None) -> pd.DataFrame:
    vine = fit['vine']
    U = vine.simulate(n, seeds=[int(seed)] if seed is not None else [])

    out = {}
    for j, c in enumerate(fit['columns']):
        if fit['var_types'][j] == 'd':
            cum = fit['cat_cum'][c]
            k = np.minimum(np.searchsorted(cum, U[:, j], side='left'),
                           len(cum) - 1)
            levels = fit['cat_levels'][c]
            tmpl = fit['template'][c]
            out[c] = pd.Categorical([levels[i] for i in k],
                                    categories=levels,
                                    ordered=bool(getattr(tmpl.dtype, 'ordered', False)))
        else:
            out[c] = _quantile_pp(fit['num_values'][c], U[:, j])

    return pd.DataFrame(out, columns=fit['columns'])


def generate_vine_copula(train_df: pd.DataFrame,
                         n: int,
                         seed: Optional[int] = None,
                         task_type: Optional[str] = None,
                         jitter_eps: float = 1e-10) -> pd.DataFrame:
    fit = fit_vine_copula(train_df, seed=seed, jitter_eps=jitter_eps)
    return sample_vine_copula(fit, n, seed=seed)
