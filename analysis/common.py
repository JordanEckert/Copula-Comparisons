import json
import math
import os

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON = os.path.join(ROOT, 'results', 'benchmarks', 'benchmark_results.json')
TRAIN_CSV = os.path.join(ROOT, 'results', 'training_cost', 'train_metrics.csv')
SAMPLING_CSV = os.path.join(ROOT, 'results', 'profiling', 'sampling_params.csv')

OUT_DIR = os.path.join(ROOT, 'results', 'paper')
FACTS_DIR = os.path.join(OUT_DIR, 'facts')
FIG_DIR = os.path.join(OUT_DIR, 'figures')
os.makedirs(FACTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

TASK_ABBR = {'regression': 'reg', 'binclass': 'bin', 'multiclass': 'multi'}

N_TRAIN = {
    'abalone': 2672, 'adult': 26048, 'buddy': 12053, 'california': 13209,
    'cardio': 44800, 'churn2': 6400, 'diabetes': 491, 'fb-comments': 157638,
    'gesture': 6318, 'higgs-small': 62751, 'house': 14581, 'insurance': 856,
    'king': 13832, 'miniboone': 83240, 'wilt': 3096,
}

DISPLAY = {
    'real': 'Real',
    'beta_copula': 'BetaCopula',
    'checkerboard_copula': 'Checkerboard',
    'gaussian_copula': 'GaussCopula',
    'vine_copula': 'VineCopula',
    'tabddpm': 'TabDDPM',
    'ctabgan': 'CTAB-GAN',
    'ctabgan_plus': 'CTAB-GAN+',
    'tvae': 'TVAE',
}
COST_KEY = {'ctabgan_plus': 'ctabgan-plus'}

RANK_METHODS = ['beta_copula', 'checkerboard_copula', 'gaussian_copula', 'vine_copula',
                'tabddpm', 'ctabgan_plus', 'tvae']
TABLE_METHODS = ['tabddpm', 'ctabgan', 'ctabgan_plus', 'tvae', 'vine_copula', 'gaussian_copula',
                 'checkerboard_copula', 'beta_copula']
METRICS = ['p1', 'p2', 'corr_l2', 'assoc_l2', 'wd', 'js', 'pmse_ratio',
           'alpha_precision', 'beta_recall', 'authenticity', 'dcr_ratio', 'dcr_p5_ratio',
           'ci_overlap', 'coverage', 'sig_agree']
ALL_STORED_METRICS = METRICS + ['dcr']

HIGHER_IS_BETTER = {
    'p1': True, 'p2': True, 'dcr': True, 'dcr_ratio': True, 'dcr_p5_ratio': True,
    'alpha_precision': True, 'beta_recall': True, 'authenticity': True,
    'ci_overlap': True, 'coverage': True, 'sig_agree': True,
    'wd': False, 'js': False, 'corr_l2': False, 'assoc_l2': False,
    'pmse_ratio': False,
}

METRIC_LABEL = {
    'p1': 'P1', 'p2': 'P2', 'corr_l2': 'CorrL2', 'wd': 'WD', 'js': 'JS', 'dcr': 'DCR',
    'assoc_l2': 'AssocL2', 'dcr_ratio': 'DCR ratio', 'dcr_p5_ratio': 'DCR$_{5\\%}$ ratio',
    'pmse_ratio': 'pMSE ratio', 'ci_overlap': 'CI overlap', 'sig_agree': 'Sig.\\ agree',
    'coverage': 'Coverage', 'alpha_precision': r'$\alpha$-precision',
    'beta_recall': r'$\beta$-recall', 'authenticity': 'Authenticity',
}


def load_results(path, keep=None):
    with open(path) as f:
        res = json.load(f)
    out = {}
    for ds, r in res.items():
        if keep is not None and ds not in keep:
            continue
        out[ds] = r
    return out


def get_val(res, method, metric, stat='mean'):
    v = res.get(f'{method}_{metric}_{stat}')
    if v is None:
        return float('nan')
    try:
        v = float(v)
    except (TypeError, ValueError):
        return float('nan')
    return v if math.isfinite(v) else float('nan')


def value_matrix(results, methods, metric, stat='mean'):
    rows = {ds: {m: get_val(r, m, metric, stat) for m in methods} for ds, r in results.items()}
    df = pd.DataFrame.from_dict(rows, orient='index')[methods]
    df.index.name = 'dataset'
    return df


def real_series(results, metric):
    return pd.Series({ds: get_val(r, 'real', metric) for ds, r in results.items()}, name='real')


def rank_matrix(df, metric):
    if metric == 'pmse_ratio':
        scored = df.apply(lambda col: np.abs(np.log(col)))
        ascending = True
    else:
        scored = df
        ascending = not HIGHER_IS_BETTER[metric]
    return scored.rank(axis=1, ascending=ascending, method='average')


def nemenyi_cd(k, n, alpha=0.05):
    q_alpha = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    return float(q_alpha * np.sqrt(k * (k + 1) / (6.0 * n)))


def complete_case(df):
    return df.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='any')


def wins_split_ties(ranks):
    w = pd.Series(0.0, index=ranks.columns)
    for _, row in ranks.iterrows():
        best = row[row == row.min()].index
        for m in best:
            w[m] += 1.0 / len(best)
    return w


def friedman_block(results, methods, metric):
    df = value_matrix(results, methods, metric)
    cc = complete_case(df)
    dropped = sorted(set(df.index) - set(cc.index))
    n, k = cc.shape
    out = {'values': cc, 'n': int(n), 'k': int(k), 'dropped_datasets': dropped}
    if n < 2 or k < 2:
        out.update({'ranks': None, 'mean_rank': pd.Series(np.nan, index=methods),
                    'wins': pd.Series(np.nan, index=methods),
                    'chi2': float('nan'), 'p': float('nan'), 'cd': float('nan')})
        return out
    ranks = rank_matrix(cc, metric)
    chi2, p = stats.friedmanchisquare(*[ranks[c].values for c in ranks.columns])
    out.update({'ranks': ranks, 'mean_rank': ranks.mean(axis=0), 'wins': wins_split_ties(ranks),
                'chi2': float(chi2), 'p': float(p), 'cd': nemenyi_cd(k, n)})
    return out


def is_better(a, b, metric):
    if metric == 'pmse_ratio':
        return abs(math.log(a)) < abs(math.log(b))
    return a > b if HIGHER_IS_BETTER[metric] else a < b


def human_time(s):
    if s is None or not np.isfinite(s):
        return '---'
    if s < 60:
        return f'{s:.2f} s'
    if s < 3600:
        return f'{s / 60:.1f} min'
    if s < 86400:
        return f'{s / 3600:.2f} h'
    return f'{s / 86400:.2f} d'


def human_bytes(n):
    if n is None or not np.isfinite(n):
        return '---'
    for unit in ('B', 'KB', 'MB', 'GB'):
        if n < 1024.0:
            return f'{n:.1f} {unit}'
        n /= 1024.0
    return f'{n:.1f} TB'


def human_count(n):
    if n is None or not np.isfinite(n):
        return '---'
    if n < 1e3:
        return f'{n:.0f}'
    if n < 1e6:
        return f'{n / 1e3:.1f} K'
    if n < 1e9:
        return f'{n / 1e6:.2f} M'
    return f'{n / 1e9:.2f} B'


def fmt_mean(v, nd=3, big_nd=2, big=10.0):
    if v is None or not np.isfinite(v):
        return '---'
    s = f'{v:.{big_nd}f}' if abs(v) >= big else f'{v:.{nd}f}'
    return s.replace('-', '$-$', 1) if s.startswith('-') else s


def fmt_mean_std(m, s, nd=3):
    if m is None or not np.isfinite(m):
        return '---'
    if s is None or not np.isfinite(s):
        s = 0.0
    if s > 1e-4:
        return f'{m:.{nd}f}$\\pm${s:.{nd}f}'
    return f'{m:.{nd}f}'


def tex_escape(s):
    return s.replace('_', r'\_').replace('&', r'\&').replace('%', r'\%')


def tex_sci(p, nd=1):
    if p is None or not np.isfinite(p):
        return '---'
    if p >= 0.001:
        return f'{p:.3f}'
    mant, expo = f'{p:.{nd}e}'.split('e')
    return f'${mant}\\times10^{{{int(expo)}}}$'


def bold(s):
    return r'\textbf{' + s + '}'


def geomean(x):
    x = np.asarray([v for v in x if np.isfinite(v)], dtype=float)
    if x.size == 0:
        return float('nan')
    if np.any(x <= 0):
        return 0.0 if np.all(x >= 0) else float('nan')
    return float(np.exp(np.mean(np.log(x))))


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, pd.DataFrame):
        return {'__dataframe__': True, 'index': [str(i) for i in o.index],
                'columns': [str(c) for c in o.columns],
                'data': _jsonable(o.values.tolist())}
    if isinstance(o, pd.Series):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def write_facts(name, obj):
    os.makedirs(FACTS_DIR, exist_ok=True)
    path = os.path.join(FACTS_DIR, f'{name}.json')
    with open(path, 'w') as f:
        json.dump(_jsonable(obj), f, indent=1)
    return path


def read_facts(name):
    with open(os.path.join(FACTS_DIR, f'{name}.json')) as f:
        return json.load(f)


def write_text(name, text):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    with open(path, 'w') as f:
        f.write(text)
    print(f'wrote {path}')
    return path


def df_to_text(df, float_fmt='{:.3f}'):
    cols = [str(c) for c in df.columns]
    lines = ['| ' + str(df.index.name or '') + ' | ' + ' | '.join(cols) + ' |',
             '|' + '---|' * (len(cols) + 1)]
    for idx, row in df.iterrows():
        cells = []
        for v in row.values:
            if isinstance(v, (float, np.floating)):
                cells.append('nan' if not np.isfinite(v) else float_fmt.format(v))
            else:
                cells.append(str(v))
        lines.append('| ' + str(idx) + ' | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)
