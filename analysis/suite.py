import math
import os
import numpy as np
import pandas as pd

from common import (JSON, ROOT, RANK_METHODS, TABLE_METHODS, METRICS, ALL_STORED_METRICS,
                    DISPLAY, TASK_ABBR, METRIC_LABEL, HIGHER_IS_BETTER, load_results,
                    value_matrix, real_series, friedman_block, is_better, geomean, fmt_mean,
                    fmt_mean_std, bold, tex_sci, write_facts, write_text, OUT_DIR)

results = load_results(JSON)
DATASETS = list(results.keys())
assert len(DATASETS) == 15, DATASETS
TASKS = {ds: results[ds]['task'] for ds in DATASETS}

def r2(v):
    from decimal import Decimal, ROUND_HALF_UP
    return str(Decimal(repr(float(v))).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))

COMPLETE = list(DATASETS)
facts = {'datasets': DATASETS, 'tasks': TASKS, 'rank_methods': RANK_METHODS, 'table_methods': TABLE_METHODS,
         'json': os.path.relpath(JSON, ROOT)}

for k in ('p1', 'p2'):
    a = real_series(results, k)
    assert all(np.isfinite(a[ds]) for ds in COMPLETE), k

blocks = {}
rank_rows, long_rows = [], []
for metric in ALL_STORED_METRICS:
    b = friedman_block(results, RANK_METHODS, metric)
    blocks[metric] = b
    for m in RANK_METHODS:
        rank_rows.append({'metric': metric, 'method': DISPLAY[m], 'method_key': m,
                          'mean_rank': b['mean_rank'][m], 'wins': b['wins'][m],
                          'n_datasets': b['n'], 'k_methods': b['k'],
                          'friedman_chi2': b['chi2'], 'p_value': b['p'], 'nemenyi_CD': b['cd'],
                          'dropped_datasets': ';'.join(b['dropped_datasets'])})
    if b['ranks'] is not None:
        for ds, row in b['ranks'].iterrows():
            for m in RANK_METHODS:
                long_rows.append({'metric': metric, 'dataset': ds, 'method': DISPLAY[m],
                                  'value': b['values'].loc[ds, m], 'rank': row[m]})
mean_ranks = pd.DataFrame(rank_rows)
mean_ranks.to_csv(f'{OUT_DIR}/mean_ranks.csv', index=False)
pd.DataFrame(long_rows).to_csv(f'{OUT_DIR}/ranks_per_dataset.csv', index=False)
print(f'wrote {OUT_DIR}/mean_ranks.csv')
facts['friedman'] = {
    metric: {'n': b['n'], 'k': b['k'], 'chi2': b['chi2'], 'p': b['p'], 'cd': b['cd'],
             'dropped_datasets': b['dropped_datasets'],
             'mean_rank': {DISPLAY[m]: b['mean_rank'][m] for m in RANK_METHODS},
             'wins': {DISPLAY[m]: b['wins'][m] for m in RANK_METHODS},
             'used_datasets': list(b['values'].index)}
    for metric, b in blocks.items()}
assert blocks['dcr']['ranks'].equals(blocks['dcr_ratio']['ranks']), 'dcr vs dcr_ratio ranks differ'
facts['dcr_equals_dcr_ratio_ranks'] = True


def nemenyi_pairs(metric):
    b = blocks[metric]
    out = []
    for i, m1 in enumerate(RANK_METHODS):
        for m2 in RANK_METHODS[i + 1:]:
            d = abs(b['mean_rank'][m1] - b['mean_rank'][m2])
            out.append({'a': DISPLAY[m1], 'b': DISPLAY[m2], 'diff': float(d), 'separated': bool(d > b['cd'])})
    return out


facts['nemenyi_pairs'] = {metric: nemenyi_pairs(metric) for metric in METRICS}
facts['beta_separated_from'] = {
    metric: sorted([p['b'] if p['a'] == 'BetaCopula' else p['a'] for p in facts['nemenyi_pairs'][metric]
                    if p['separated'] and 'BetaCopula' in (p['a'], p['b'])])
    for metric in METRICS}
facts['beta_not_separated_from'] = {
    metric: sorted([p['b'] if p['a'] == 'BetaCopula' else p['a'] for p in facts['nemenyi_pairs'][metric]
                    if not p['separated'] and 'BetaCopula' in (p['a'], p['b'])])
    for metric in METRICS}

GEO_METRICS = ('pmse_ratio', 'dcr_ratio', 'dcr_p5_ratio')
val_rows = []
means = {}
for metric in ALL_STORED_METRICS:
    vals = value_matrix(results, TABLE_METHODS, metric)
    recs = {}
    for m in TABLE_METHODS:
        v = vals[m].dropna()
        rec = {'n_datasets': int(v.size), 'mean': v.mean() if v.size else np.nan,
               'geomean': geomean(v.values) if metric in GEO_METRICS else np.nan,
               'median': v.median() if v.size else np.nan,
               'datasets': ';'.join(v.index)}
        recs[m] = rec
        row = {'metric': metric, 'method': DISPLAY[m], **rec}
        for ds in DATASETS:
            row[ds] = vals.loc[ds, m]
        val_rows.append(row)
    means[metric] = pd.DataFrame.from_dict(recs, orient='index')
mean_values = pd.DataFrame(val_rows)
with open(f'{OUT_DIR}/mean_values.csv', 'w') as f:
    f.write('# Unified suite. mean = arithmetic mean over the datasets with a finite value of the per-dataset means '
            '(15 for the ranked methods, 9 classification datasets for CTAB-GAN); geomean only for pmse_ratio, dcr_ratio, dcr_p5_ratio\n')
    mean_values.to_csv(f, index=False)
print(f'wrote {OUT_DIR}/mean_values.csv')
facts['means'] = {metric: {DISPLAY[m]: {k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                                        for k, v in means[metric].loc[m].items()}
                           for m in TABLE_METHODS} for metric in ALL_STORED_METRICS}
means14 = {}
for metric in ALL_STORED_METRICS:
    vals = value_matrix(results, TABLE_METHODS, metric).loc[COMPLETE]
    means14[metric] = {DISPLAY[m]: {'n': int(vals[m].notna().sum()), 'mean': float(vals[m].mean()),
                                    'geomean': (geomean(vals[m].dropna().values) if metric in GEO_METRICS else None),
                                    'median': float(vals[m].median())}
                       for m in TABLE_METHODS}
facts['means_14'] = means14

gap_rows = []
gap_facts = {}
p2 = value_matrix(results, TABLE_METHODS, 'p2')
real_p2 = real_series(results, 'p2')
for metric in ['p2', 'p1']:
    vals = value_matrix(results, TABLE_METHODS, metric)
    real = real_series(results, metric)
    gap = vals.sub(real, axis=0)
    gap_facts[metric] = {}
    for m in TABLE_METHODS:
        g = gap[m].dropna()
        gap_rows.append({'metric': metric, 'method': DISPLAY[m], 'n_datasets': int(g.size),
                         'mean_gap': g.mean(), 'median_gap': g.median(),
                         'min_gap': g.min(), 'max_gap': g.max(),
                         'n_within_0.02': int((g.abs() <= 0.02).sum()),
                         'n_above_real': int((g > 0).sum()), 'datasets': ';'.join(g.index)})
        gap_facts[metric][DISPLAY[m]] = {
            'n': int(g.size), 'mean_gap': float(g.mean()), 'median_gap': float(g.median()),
            'min_gap': float(g.min()), 'min_gap_dataset': str(g.idxmin()),
            'max_gap': float(g.max()), 'max_gap_dataset': str(g.idxmax()),
            'n_within_0.02': int((g.abs() <= 0.02).sum()),
            'n_above_real': int((g > 0).sum()),
            'per_dataset': {str(k): float(v) for k, v in g.items()}}
gap_df = pd.DataFrame(gap_rows)
gap14 = {}
for metric in ['p2', 'p1']:
    vals = value_matrix(results, TABLE_METHODS, metric).loc[COMPLETE]
    real = real_series(results, metric).loc[COMPLETE]
    gap = vals.sub(real, axis=0)
    gap14[metric] = {}
    for m in TABLE_METHODS:
        g = gap[m].dropna()
        gap14[metric][DISPLAY[m]] = {'n': int(g.size), 'mean_gap': float(g.mean()), 'median_gap': float(g.median()),
                                     'min_gap': float(g.min()), 'min_gap_dataset': str(g.idxmin()),
                                     'max_gap': float(g.max()), 'max_gap_dataset': str(g.idxmax()),
                                     'n_within_0.02': int((g.abs() <= 0.02).sum()), 'n_above_real': int((g > 0).sum())}
facts['utility_gap_14'] = gap14
beta_gap = (p2['beta_copula'] - real_p2).dropna()
facts['beta_p2_within_0.02'] = {
    'n': int((beta_gap.abs() <= 0.02).sum()), 'of': int(beta_gap.size),
    'datasets': sorted(beta_gap[beta_gap.abs() <= 0.02].index.tolist()),
    'outside': {str(k): float(v) for k, v in beta_gap[beta_gap.abs() > 0.02].items()},
    'per_dataset_gap': {str(k): float(v) for k, v in beta_gap.items()},
}
deep = ['tabddpm', 'ctabgan', 'ctabgan_plus', 'tvae']
loses_all = []
for ds in COMPLETE:
    others = [p2.loc[ds, m] for m in deep if np.isfinite(p2.loc[ds, m])]
    if others and all(p2.loc[ds, 'beta_copula'] < v for v in others):
        loses_all.append(ds)
facts['beta_loses_to_every_deep_model_on_p2'] = loses_all
TIE = 0.001
h2h_rows = []
h2h_facts = {}
for m in TABLE_METHODS:
    if m == 'beta_copula':
        continue
    d = (p2['beta_copula'] - p2[m]).dropna()
    wins = int((d > TIE).sum()); losses = int((d < -TIE).sum()); ties = int((d.abs() <= TIE).sum())
    h2h_rows.append({'metric': 'p2', 'comparison': f'BetaCopula vs {DISPLAY[m]}',
                     'n_datasets': int(d.size), 'beta_wins': wins, 'beta_losses': losses,
                     'ties': ties, 'tie_threshold': TIE, 'mean_diff_beta_minus_other': d.mean(),
                     'median_diff_beta_minus_other': d.median(),
                     'loss_datasets': ';'.join(sorted(d[d < -TIE].index)),
                     'tie_datasets': ';'.join(sorted(d[d.abs() <= TIE].index))})
    h2h_facts[DISPLAY[m]] = {'n': int(d.size), 'wins': wins, 'losses': losses, 'ties': ties,
                             'mean_diff': float(d.mean()), 'median_diff': float(d.median()),
                             'loss_datasets': sorted(d[d < -TIE].index.tolist()),
                             'tie_datasets': sorted(d[d.abs() <= TIE].index.tolist()),
                             'per_dataset_diff': {str(k): float(v) for k, v in d.items()}}
h2h_df = pd.DataFrame(h2h_rows)
with open(f'{OUT_DIR}/utility_gap.csv', 'w') as f:
    f.write('# Utility gap to Real: (method value - real value) per dataset, over datasets where both are finite\n')
    gap_df.to_csv(f, index=False)
    f.write('\n# Head-to-head on P2, BetaCopula minus other; ties = |diff| <= 0.001\n')
    h2h_df.to_csv(f, index=False)
print(f'wrote {OUT_DIR}/utility_gap.csv')
facts['utility_gap'] = gap_facts
facts['head_to_head_p2'] = h2h_facts

h2h_rows = []
h2h_all = {}
for other in [m for m in TABLE_METHODS if m != 'beta_copula']:
    h2h_all[DISPLAY[other]] = {}
    for metric in ALL_STORED_METRICS:
        vals = value_matrix(results, ['beta_copula', other], metric).dropna()
        n = int(len(vals))
        if n == 0:
            continue
        bw = bl = ties = 0
        for ds, r in vals.iterrows():
            a, b_ = r['beta_copula'], r[other]
            if a == b_:
                ties += 1
            elif is_better(a, b_, metric):
                bw += 1
            else:
                bl += 1
        diff = vals['beta_copula'] - vals[other]
        rec = {'n_datasets': n, 'beta_wins': bw, 'other_wins': bl, 'ties_exact': ties,
               'mean_diff_beta_minus_other': float(diff.mean()),
               'median_diff_beta_minus_other': float(diff.median()),
               'mean_beta': float(vals['beta_copula'].mean()), 'mean_other': float(vals[other].mean()),
               'loss_datasets': sorted(vals.index[[not is_better(a, b_, metric) and a != b_
                                                   for a, b_ in zip(vals['beta_copula'], vals[other])]].tolist())}
        if metric == 'pmse_ratio':
            rec['mean_abs_log_beta'] = float(np.abs(np.log(vals['beta_copula'])).mean())
            rec['mean_abs_log_other'] = float(np.abs(np.log(vals[other])).mean())
        h2h_rows.append({'comparison': f'BetaCopula vs {DISPLAY[other]}', 'metric': metric,
                         'direction': ('|log| closest to 0' if metric == 'pmse_ratio'
                                       else ('higher' if HIGHER_IS_BETTER[metric] else 'lower')),
                         **{k: (';'.join(v) if isinstance(v, list) else v) for k, v in rec.items()}})
        h2h_all[DISPLAY[other]][metric] = rec
h2h = pd.DataFrame(h2h_rows)
with open(f'{OUT_DIR}/head_to_head.csv', 'w') as f:
    f.write('# BetaCopula vs each other method, per metric; wins by the metric direction over the datasets where both '
            'are finite; ties = exactly equal stored means\n')
    h2h.to_csv(f, index=False)
print(f'wrote {OUT_DIR}/head_to_head.csv')
facts['head_to_head'] = h2h_all

SHORT = {'real': 'Real', 'tabddpm': 'DDPM', 'ctabgan': 'CTAB', 'ctabgan_plus': 'CTAB+', 'tvae': 'TVAE',
         'vine_copula': 'Vine', 'gaussian_copula': 'Gauss', 'checkerboard_copula': 'Chkb', 'beta_copula': 'Beta'}
P2_COLS = ['real'] + TABLE_METHODS
CORR_COLS = TABLE_METHODS
p2_std = value_matrix(results, TABLE_METHODS, 'p2', stat='std')
corr = value_matrix(results, TABLE_METHODS, 'corr_l2')


def best_synthetic(row_vals, metric):
    finite = {m: v for m, v in row_vals.items() if np.isfinite(v)}
    if not finite:
        return set()
    best = None
    for m, v in finite.items():
        if best is None or is_better(v, finite[best], metric):
            best = m
    return {m for m, v in finite.items() if v == finite[best]}


def dataset_label(ds):
    return f'{ds} & {TASK_ABBR[TASKS[ds]]}'


n_p2 = len(P2_COLS); n_corr = len(CORR_COLS)
lines = [r'\begingroup\setlength{\tabcolsep}{1.2pt}', r'\begin{tabular}{ll' + 'r' * n_p2 + 'r' * n_corr + '}', r'\toprule',
         r'\multicolumn{2}{l}{} & \multicolumn{' + str(n_p2) + r'}{c}{P2 (macro $F_1$ / $R^2$, $\uparrow$)} & '
         r'\multicolumn{' + str(n_corr) + r'}{c}{CorrL2 ($\downarrow$)} \\',
         r'\cmidrule(lr){3-' + str(2 + n_p2) + r'} \cmidrule(lr){' + str(3 + n_p2) + '-' + str(2 + n_p2 + n_corr) + '}',
         'Dataset & Task & ' + ' & '.join(SHORT[m] for m in P2_COLS) + ' & '
         + ' & '.join(SHORT[m] for m in CORR_COLS) + r' \\', r'\midrule']
for ds in DATASETS:
    cells = [dataset_label(ds)]
    row_p2 = {m: p2.loc[ds, m] for m in P2_COLS if m != 'real'}
    best_p2 = best_synthetic(row_p2, 'p2')
    for m in P2_COLS:
        v = real_p2[ds] if m == 'real' else p2.loc[ds, m]
        s = fmt_mean(v)
        if m in best_p2:
            s = bold(s)
        cells.append(s)
    row_c = {m: corr.loc[ds, m] for m in CORR_COLS}
    best_c = best_synthetic(row_c, 'corr_l2')
    for m in CORR_COLS:
        s = fmt_mean(corr.loc[ds, m])
        if m in best_c:
            s = bold(s)
        cells.append(s)
    lines.append(' & '.join(cells) + r' \\')
lines += [r'\bottomrule', r'\end{tabular}', r'\endgroup']
write_text('table_p2_corr.tex',
           '% A5: per-dataset P2 and CorrL2. Use inside \\begin{table*} at \\scriptsize. Header abbreviations: '
           'DDPM = TabDDPM, CTAB = CTAB-GAN, CTAB+ = CTAB-GAN+, Vine = VineCopula, Gauss = GaussCopula, '
           'Chkb = Checkerboard, Beta = BetaCopula. Bold = best synthetic method per block; '
           '--- = not available.\n' + '\n'.join(lines) + '\n')

lines = [r'\begingroup\setlength{\tabcolsep}{1.2pt}', r'\begin{tabular}{l' + 'r' * n_p2 + '}', r'\toprule',
         'Dataset & ' + ' & '.join(DISPLAY[m] for m in P2_COLS) + r' \\', r'\midrule']
for ds in DATASETS:
    cells = [f'{ds} ({TASK_ABBR[TASKS[ds]]})']
    row_p2 = {m: p2.loc[ds, m] for m in P2_COLS if m != 'real'}
    best_p2 = best_synthetic(row_p2, 'p2')
    for m in P2_COLS:
        if m == 'real':
            s = fmt_mean(real_p2[ds])
        else:
            s = fmt_mean_std(p2.loc[ds, m], p2_std.loc[ds, m])
        if m in best_p2:
            s = bold(s)
        cells.append(s)
    lines.append(' & '.join(cells) + r' \\')
lines += [r'\bottomrule', r'\end{tabular}', r'\endgroup']
write_text('table_p2_std.tex', '% A5 (appendix): per-dataset P2 mean$\\pm$std (population SD over generation seeds)\n'
           + '\n'.join(lines) + '\n')

RANK_ORDER = ['beta_copula', 'checkerboard_copula', 'gaussian_copula', 'vine_copula', 'tabddpm', 'ctabgan_plus', 'tvae']
GROUPS = [('Utility', ['p1', 'p2']),
          ('Statistical fidelity', ['corr_l2', 'assoc_l2', 'wd', 'js', 'pmse_ratio']),
          ('Faithfulness', ['alpha_precision', 'beta_recall', 'authenticity']),
          ('Privacy proxies', ['dcr_ratio', 'dcr_p5_ratio']),
          ('Inference fidelity', ['ci_overlap', 'coverage', 'sig_agree'])]
ARROW = {k: (r'$\to 1$' if k == 'pmse_ratio' else (r'$\uparrow$' if HIGHER_IS_BETTER[k] else r'$\downarrow$')) for k in METRICS}
lines = [r'\begin{tabular}{l' + 'r' * 3 + 'r' * len(RANK_ORDER) + '}', r'\toprule',
         'Metric & $n$ & CD & $p$ & ' + ' & '.join(SHORT[m] for m in RANK_ORDER) + r' \\', r'\midrule']
for gname, keys in GROUPS:
    lines.append(r'\multicolumn{' + str(4 + len(RANK_ORDER)) + r'}{l}{\emph{' + gname + r'}} \\')
    for metric in keys:
        b = blocks[metric]
        cells = [METRIC_LABEL[metric] + ' ' + ARROW[metric], str(b['n']), f"{b['cd']:.2f}", tex_sci(b['p'])]
        for m in RANK_ORDER:
            v = b['mean_rank'][m]
            s = r2(v)
            if v == b['mean_rank'].min():
                s = bold(s)
            cells.append(s)
        lines.append(' & '.join(cells) + r' \\')
lines += [r'\bottomrule', r'\end{tabular}']
write_text('table_ranks.tex', '% A5: mean ranks (1 = best) over complete-case datasets, 7 methods, metrics as rows; '
           'n = datasets used, CD = Nemenyi critical difference (alpha = 0.05), p = Friedman p-value. '
           'Header abbreviations as in table_p2_corr.tex.\n' + '\n'.join(lines) + '\n')

MEAN_COLS = [('p2', 'P2$\\uparrow$', False), ('corr_l2', 'CorrL2$\\downarrow$', False), ('assoc_l2', 'AssocL2$\\downarrow$', False),
             ('wd', 'WD$\\downarrow$', False), ('js', 'JS$\\times10^{3}\\downarrow$', False), ('pmse_ratio', 'pMSE$\\to$1', True),
             ('alpha_precision', '$\\alpha$-prec.$\\uparrow$', False), ('beta_recall', '$\\beta$-rec.$\\uparrow$', False),
             ('authenticity', 'Auth.$\\uparrow$', False), ('dcr_ratio', 'DCR$\\uparrow$', True),
             ('dcr_p5_ratio', 'DCR$_{5\\%}\\uparrow$', True), ('ci_overlap', 'CI$\\uparrow$', False),
             ('coverage', 'Cov.$\\uparrow$', False), ('sig_agree', 'Sig.$\\uparrow$', False)]
ROW_ORDER = ['tabddpm', 'ctabgan', 'ctabgan_plus', 'tvae', 'vine_copula', 'gaussian_copula', 'checkerboard_copula', 'beta_copula']


def mean_cell(metric, m, use_geo):
    rec = means14[metric][DISPLAY[m]]
    v = rec['geomean'] if use_geo else rec['mean']
    if metric == 'js':
        return fmt_mean(v * 1000.0, nd=2, big_nd=1)
    return fmt_mean(v)


def best_mean(metric, use_geo):
    vals = {m: (means14[metric][DISPLAY[m]]['geomean'] if use_geo else means14[metric][DISPLAY[m]]['mean']) for m in RANK_METHODS}
    if metric in ('pmse_ratio', 'dcr_ratio', 'dcr_p5_ratio') and metric == 'pmse_ratio':
        key = {m: abs(math.log(v)) if v and v > 0 else float('inf') for m, v in vals.items()}
        best = min(key.values())
        return {m for m, k in key.items() if k == best}
    if HIGHER_IS_BETTER[metric]:
        best = max(vals.values())
    else:
        best = min(vals.values())
    return {m for m, v in vals.items() if v == best}


lines = [r'\begingroup\setlength{\tabcolsep}{2pt}', r'\begin{tabular}{l' + 'r' * len(MEAN_COLS) + '}', r'\toprule',
         'Method & ' + ' & '.join(c[1] for c in MEAN_COLS) + r' \\', r'\midrule']
for m in ROW_ORDER:
    name = DISPLAY[m] + (r'$^{\dagger}$' if m == 'ctabgan' else '')
    cells = [name]
    for metric, _, geo in MEAN_COLS:
        s = mean_cell(metric, m, geo)
        if m in RANK_METHODS and m in best_mean(metric, geo):
            s = bold(s)
        cells.append(s)
    lines.append(' & '.join(cells) + r' \\')
lines.append(r'\midrule')
real14 = {k: float(real_series(results, k).loc[COMPLETE].mean()) for k in ('p1', 'p2')}
lines.append('Real & ' + f"{real14['p2']:.3f}" + ' & ' + ' & '.join('---' for _ in MEAN_COLS[1:]) + r' \\')
lines += [r'\bottomrule', r'\end{tabular}', r'\endgroup']
write_text('table_means.tex',
           '% A5: means over the 15 complete datasets of the per-dataset means (geometric means for pMSE ratio, DCR ratio, '
           'DCR 5th-percentile ratio; JS multiplied by 1000); CTAB-GAN (dagger) over its 9 classification datasets '
           '(JS: 4); bold = best among the 7 ranked methods; Real = train-on-real P2 over the same 15 datasets. '
           'For a table* at \\footnotesize or \\scriptsize.\n'
           + '\n'.join(lines) + '\n')
facts['real_means_14'] = real14
facts['means_table_best'] = {metric: sorted(DISPLAY[m] for m in best_mean(metric, geo)) for metric, _, geo in MEAN_COLS}

def p_compact(p):
    if p is None or not np.isfinite(p):
        return '---'
    if p >= 0.001:
        return f'{p:.3f}'
    return r'$<\!10^{' + str(int(math.ceil(math.log10(p)))) + '}$'


RM_COLS = [('p1', 'P1$\\uparrow$', False)] + [(k, ('JS$\\downarrow$' if k == 'js' else lab), g) for k, lab, g in MEAN_COLS]
RM_GROUPS = [('Utility', 2), ('Statistical fidelity', 5), ('Faithfulness', 3), ('Privacy proxies', 2), ('Inference fidelity', 3)]
assert sum(g[1] for g in RM_GROUPS) == len(RM_COLS)
ncol = 1 + len(RM_COLS)
lines = [r'\begingroup\setlength{\tabcolsep}{1.5pt}', r'\begin{tabular}{l' + 'r' * len(RM_COLS) + '}', r'\toprule']
head = ' & ' + ' & '.join(r'\multicolumn{' + str(n) + '}{c}{' + g + '}' for g, n in RM_GROUPS) + r' \\'
lines.append(head)
c0 = 2
cmid = []
for g, n in RM_GROUPS:
    cmid.append(r'\cmidrule(lr){' + str(c0) + '-' + str(c0 + n - 1) + '}')
    c0 += n
lines.append(' '.join(cmid))
lines.append('Method & ' + ' & '.join(c[1] for c in RM_COLS) + r' \\')
lines.append(r'\midrule')
RM_ROW_ORDER = ['tabddpm', 'ctabgan_plus', 'tvae', 'vine_copula', 'gaussian_copula', 'checkerboard_copula', 'beta_copula']
FAMILY_GAP_BEFORE = ('vine_copula', 'checkerboard_copula')
rank_lines = [r'\multicolumn{' + str(ncol) + r'}{l}{\emph{Mean rank over datasets (1 is best)}} \\']
for m in RM_ROW_ORDER:
    if m in FAMILY_GAP_BEFORE:
        rank_lines.append(r'\addlinespace[2pt]')
    cells = [DISPLAY[m]]
    for metric, _, _ in RM_COLS:
        b = blocks[metric]
        v = b['mean_rank'][m]
        s_ = r2(v)
        if v == b['mean_rank'].min():
            s_ = bold(s_)
        cells.append(s_)
    rank_lines.append(' & '.join(cells) + r' \\')
rank_lines.append(r'\addlinespace[2pt]')
rank_lines.append('$n$ datasets & ' + ' & '.join(str(blocks[k]['n']) for k, _, _ in RM_COLS) + r' \\')
rank_lines.append('Nemenyi CD & ' + ' & '.join(f"{blocks[k]['cd']:.2f}" for k, _, _ in RM_COLS) + r' \\')
mean_lines = [r'\multicolumn{' + str(ncol) + r'}{l}{\emph{Mean over datasets}} \\']
for m in ROW_ORDER:
    if m in FAMILY_GAP_BEFORE:
        mean_lines.append(r'\addlinespace[2pt]')
    name = DISPLAY[m] + (r'$^{\dagger}$' if m == 'ctabgan' else '')
    cells = [name]
    for metric, _, geo in RM_COLS:
        if metric == 'p1':
            cells.append('---')
            continue
        s_ = mean_cell(metric, m, geo)
        if m in RANK_METHODS and m in best_mean(metric, geo):
            s_ = bold(s_)
        cells.append(s_)
    mean_lines.append(' & '.join(cells) + r' \\')
mean_lines.append(r'\addlinespace[2pt]')
mean_lines.append('Real & --- & ' + f"{real14['p2']:.3f}" + ' & ' + ' & '.join('---' for _ in RM_COLS[2:]) + r' \\')
lines += mean_lines + [r'\midrule'] + rank_lines
lines += [r'\bottomrule', r'\end{tabular}', r'\endgroup']
write_text('table_ranks_means.tex',
           '% A5: merged results table for a table* at \\scriptsize. Top block: means over datasets; bottom block: mean ranks (1 = best) over the complete-case '
           'datasets for the 7 ranked methods, with n and the Nemenyi CD (alpha = 0.05); Friedman p-values are in facts/suite.json '
           '(friedman_p_compact) for the caption. '
           'Means block: means over the 15 complete datasets of the per-dataset means (geometric means for pMSE ratio and '
           'the two DCR ratios; JS x 1000; P1 omitted because its mean over datasets is dominated by negative R^2 values on '
           'regression tasks); CTAB-GAN (dagger) over its 9 classification datasets (JS: 4); bold = best among the 7 ranked '
           'methods; Real = train-on-real P2 over the same 15 datasets.\n' + '\n'.join(lines) + '\n')
facts['friedman_p_compact'] = {k: p_compact(blocks[k]['p']) for k, _, _ in RM_COLS}

ci = value_matrix(results, TABLE_METHODS, 'ci_overlap')
ci_std = value_matrix(results, TABLE_METHODS, 'ci_overlap', stat='std')
COL_ORDER = ROW_ORDER
lines = [r'\begingroup\setlength{\tabcolsep}{2pt}', r'\begin{tabular}{l' + 'r' * len(COL_ORDER) + '}', r'\toprule',
         'Dataset & ' + ' & '.join(DISPLAY[m] + (r'$^{\dagger}$' if m == 'ctabgan' else '') for m in COL_ORDER) + r' \\', r'\midrule']
for ds in DATASETS:
    row = ci.loc[ds]
    best = row[row == row.max()].index
    cells = [f'{ds} ({TASK_ABBR[TASKS[ds]]})']
    for m in COL_ORDER:
        s = fmt_mean_std(ci.loc[ds, m], ci_std.loc[ds, m])
        if m in best:
            s = bold(s)
        cells.append(s)
    lines.append(' & '.join(cells) + r' \\')
lines.append(r'\midrule')
lines.append('Mean & ' + ' & '.join(f"{ci[m].loc[COMPLETE].mean():.3f}" for m in COL_ORDER) + r' \\')
lines.append('Mean rank & ' + ' & '.join(('---' if m not in RANK_METHODS else r2(blocks['ci_overlap']['mean_rank'][m])) for m in COL_ORDER) + r' \\')
lines += [r'\bottomrule', r'\end{tabular}', r'\endgroup']
write_text('table_ci_per_dataset.tex',
           '% A5 (appendix): per-dataset analyst-GLM 95% CI overlap, mean$\\pm$std (population SD over generation seeds); '
           'task type in parentheses; bold = best per dataset; Mean over the 15 complete datasets (CTAB-GAN: its 9); '
           'mean rank from the complete-case ranking. For a table at \\footnotesize in the one-column appendix.\n'
           + '\n'.join(lines) + '\n')

degenerate = []
for ds in DATASETS:
    r = results[ds]
    for m in TABLE_METHODS:
        notes = []
        p1m, p2m = r.get(f'{m}_p1_mean'), r.get(f'{m}_p2_mean')
        stds = [r.get(f'{m}_{k}_std') for k in ALL_STORED_METRICS]
        if p2m is not None and isinstance(p2m, float) and math.isfinite(p2m):
            if all((s == 0.0) for s in stds if s is not None) and any(s is not None for s in stds) and m != 'ctabgan':
                notes.append('all *_std == 0 (single surviving generation seed)')
            if p1m == p2m:
                notes.append(f'p1 == p2 == {p1m:.6f}')
            ap, br = r.get(f'{m}_alpha_precision_mean'), r.get(f'{m}_beta_recall_mean')
            if ap is not None and ap < 0:
                notes.append(f'alpha_precision = {ap:.4f} (< 0: precision curve identically 0)')
            if br is not None and br < 0:
                notes.append(f'beta_recall = {br:.4f} (< 0)')
            au = r.get(f'{m}_authenticity_mean')
            if au == 1.0:
                notes.append('authenticity == 1.0')
            cio = r.get(f'{m}_ci_overlap_mean')
            if cio == 0.0:
                notes.append('ci_overlap == 0.0')
            dr = r.get(f'{m}_dcr_ratio_mean')
            if dr is not None and dr > 10:
                notes.append(f'dcr_ratio = {dr:.2f}')
        if notes:
            degenerate.append({'dataset': ds, 'method': DISPLAY[m], 'notes': notes})
facts['degenerate'] = degenerate
js = value_matrix(results, RANK_METHODS, 'js')
facts['js_all_nan_datasets'] = sorted(js.index[js.isna().all(axis=1)].tolist())
facts['ctabgan_available_datasets'] = sorted(p2['ctabgan'].dropna().index.tolist())
facts['per_dataset'] = {metric: {ds: {DISPLAY[m]: (None if not np.isfinite(v) else float(v))
                                      for m, v in value_matrix(results, TABLE_METHODS, metric).loc[ds].items()}
                                 for ds in DATASETS} for metric in ALL_STORED_METRICS}
facts['per_dataset_std'] = {metric: {ds: {DISPLAY[m]: (None if not np.isfinite(v) else float(v))
                                          for m, v in value_matrix(results, TABLE_METHODS, metric, stat='std').loc[ds].items()}
                                     for ds in DATASETS} for metric in ALL_STORED_METRICS}
facts['real_values'] = {k: {ds: (None if not np.isfinite(v) else float(v)) for ds, v in real_series(results, k).items()} for k in ('p1', 'p2', 'dcr')}
d5 = value_matrix(results, TABLE_METHODS, 'dcr_p5_ratio').loc[COMPLETE]
facts['checkerboard_dcr_p5'] = {'n_exactly_zero': int((d5['checkerboard_copula'] == 0).sum()),
                                'zero_datasets': sorted(d5.index[d5['checkerboard_copula'] == 0].tolist()),
                                'max': float(d5['checkerboard_copula'].max()),
                                'max_dataset': str(d5['checkerboard_copula'].idxmax()),
                                'per_dataset': {ds: float(v) for ds, v in d5['checkerboard_copula'].items()}}
facts['beta_dcr_ratio_below_1'] = {'n': int((value_matrix(results, ['beta_copula'], 'dcr_ratio').loc[COMPLETE]['beta_copula'] < 1).sum()), 'of': len(COMPLETE)}
facts['beta_dcr_p5_ratio_below_1'] = {'n': int((d5['beta_copula'] < 1).sum()), 'of': len(COMPLETE),
                                      'above_1': sorted(d5.index[d5['beta_copula'] >= 1].tolist())}
cb = corr['beta_copula'].loc[COMPLETE].sort_values(ascending=False)
cbs = value_matrix(results, TABLE_METHODS, 'corr_l2', stat='std')['beta_copula']
facts['beta_corr_l2_largest'] = [{'dataset': ds, 'mean': float(v), 'std': float(cbs[ds])} for ds, v in cb.head(4).items()]
write_facts('suite', facts)
print('degenerate:', degenerate)
print(mean_ranks.pivot(index='method', columns='metric', values='mean_rank').round(2).to_string())
print({k: (blocks[k]['n'], round(blocks[k]['cd'], 3), blocks[k]['p']) for k in METRICS})
