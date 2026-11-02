import numpy as np
import pandas as pd

from common import (TRAIN_CSV, SAMPLING_CSV, JSON, N_TRAIN, DISPLAY, COST_KEY, TABLE_METHODS,
                    load_results, value_matrix, real_series, human_time, human_count,
                    write_facts, write_text, OUT_DIR)

COST_METHODS = ['tabddpm', 'ctabgan', 'ctabgan-plus', 'tvae', 'beta_copula',
                'gaussian_copula', 'checkerboard_copula', 'vine_copula']
TABLE_ORDER = ['tabddpm', 'ctabgan', 'ctabgan-plus', 'tvae', 'vine_copula',
               'gaussian_copula', 'checkerboard_copula', 'beta_copula']
LABEL = {m: DISPLAY[m.replace('-', '_')] for m in COST_METHODS}
DATASETS = list(N_TRAIN.keys())

train = pd.read_csv(TRAIN_CSV)
samp = pd.read_csv(SAMPLING_CSV)
train_status_counts = train['status'].value_counts().to_dict()
train = train[train['dataset'].isin(DATASETS) & train['method'].isin(COST_METHODS)]
train_ok = train[train['status'].isin(['success', 'already_trained'])].copy()
samp = samp[samp['dataset'].isin(DATASETS) & samp['method'].isin(COST_METHODS)]
samp_ok = samp[samp['status'] == 'success'].copy()

rows = []
for ds in DATASETS:
    for m in COST_METHODS:
        t = train_ok[(train_ok['dataset'] == ds) & (train_ok['method'] == m)]
        s = samp_ok[(samp_ok['dataset'] == ds) & (samp_ok['method'] == m)]
        tt = float(t['train_time_s'].iloc[0]) if len(t) else np.nan
        sp = float(t['train_space_bytes'].iloc[0]) if len(t) else np.nan
        st = float(s['sample5_time_s'].iloc[0]) if len(s) else np.nan
        npar = float(s['n_params'].iloc[0]) if len(s) else np.nan
        t_status = train[(train['dataset'] == ds) & (train['method'] == m)]['status']
        s_status = samp[(samp['dataset'] == ds) & (samp['method'] == m)]['status']
        rows.append({'dataset': ds, 'method': LABEL[m], 'method_key': m, 'n_train': N_TRAIN[ds],
                     'train_status': t_status.iloc[0] if len(t_status) else 'absent',
                     'train_time_s': tt, 'sample_status': s_status.iloc[0] if len(s_status) else 'absent',
                     'sample5_time_s': st,
                     'cost_per_draw_s': tt + st / 5.0 if np.isfinite(tt) and np.isfinite(st) else np.nan,
                     'train_space_bytes': sp, 'n_params': npar})
cpd = pd.DataFrame(rows)
with open(f'{OUT_DIR}/cost_per_dataset.csv', 'w') as f:
    f.write('# cost_per_draw_s = train_time_s + sample5_time_s/5 (NaN where the sampling profile is missing).\n')
    cpd.to_csv(f, index=False)
print(f'wrote {OUT_DIR}/cost_per_dataset.csv')

wide_tt = cpd.pivot(index='dataset', columns='method_key', values='train_time_s')
wide_cpd = cpd.pivot(index='dataset', columns='method_key', values='cost_per_draw_s')
wide_sp = cpd.pivot(index='dataset', columns='method_key', values='train_space_bytes')
wide_np = cpd.pivot(index='dataset', columns='method_key', values='n_params')
wide_st = cpd.pivot(index='dataset', columns='method_key', values='sample5_time_s')

with np.errstate(divide='ignore', invalid='ignore'):
    log_tt_ratio = np.log10(wide_tt.div(wide_tt['beta_copula'], axis=0))
    log_cpd_ratio = np.log10(wide_cpd.div(wide_cpd['beta_copula'], axis=0))
log_tt_ratio = log_tt_ratio.replace([np.inf, -np.inf], np.nan)
log_cpd_ratio = log_cpd_ratio.replace([np.inf, -np.inf], np.nan)

sum_rows = []
for m in COST_METHODS:
    tt = wide_tt[m].dropna()
    rec = {'method': LABEL[m], 'method_key': m, 'n_datasets_train': int(tt.size),
           'train_datasets': ';'.join(tt.index),
           'train_time_mean_s': tt.mean(), 'train_time_median_s': tt.median(),
           'train_time_min_s': tt.min(), 'train_time_min_dataset': tt.idxmin() if tt.size else '',
           'train_time_max_s': tt.max(), 'train_time_max_dataset': tt.idxmax() if tt.size else '',
           'train_time_total_s': tt.sum(), 'train_time_total_human': human_time(tt.sum()),
           'train_time_median_human': human_time(tt.median()),
           'log10_train_ratio_vs_beta_median': log_tt_ratio[m].median(),
           'log10_train_ratio_vs_beta_min': log_tt_ratio[m].min(),
           'log10_train_ratio_vs_beta_max': log_tt_ratio[m].max(),
           'n_datasets_sampling': int(wide_st[m].notna().sum()),
           'sample5_time_median_s': wide_st[m].median(), 'sample5_time_mean_s': wide_st[m].mean(),
           'sample5_time_max_s': wide_st[m].max(),
           'sample5_time_max_dataset': (wide_st[m].idxmax() if wide_st[m].notna().any() else ''),
           'cost_per_draw_median_s': wide_cpd[m].median(), 'cost_per_draw_mean_s': wide_cpd[m].mean(),
           'log10_cost_per_draw_ratio_vs_beta_median': log_cpd_ratio[m].median(),
           'log10_cost_per_draw_ratio_vs_beta_min': log_cpd_ratio[m].min(),
           'log10_cost_per_draw_ratio_vs_beta_max': log_cpd_ratio[m].max(),
           'train_space_mean_bytes': wide_sp[m].mean(), 'train_space_median_bytes': wide_sp[m].median(),
           'n_params_mean': wide_np[m].mean(), 'n_params_median': wide_np[m].median()}
    sum_rows.append(rec)
summary = pd.DataFrame(sum_rows)
with open(f'{OUT_DIR}/cost_summary.csv', 'w') as f:
    f.write('# Per-method cost summary over the 15 datasets. log10 ratios are per dataset vs '
            'beta_copula, then median/min/max over datasets.\n')
    summary.to_csv(f, index=False)
print(f'wrote {OUT_DIR}/cost_summary.csv')

tab = wide_tt['tabddpm'].dropna()
headline = {
    'fb_comments_train_time_s': {LABEL[m]: (None if not np.isfinite(wide_tt.loc['fb-comments', m]) else float(wide_tt.loc['fb-comments', m])) for m in COST_METHODS},
    'fb_comments_train_time_human': {LABEL[m]: human_time(wide_tt.loc['fb-comments', m]) for m in COST_METHODS},
    'adult_train_time_s': {LABEL[m]: (None if not np.isfinite(wide_tt.loc['adult', m]) else float(wide_tt.loc['adult', m])) for m in COST_METHODS},
    'adult_train_time_human': {LABEL[m]: human_time(wide_tt.loc['adult', m]) for m in COST_METHODS},
    'ratio_ctabgan_plus_over_beta_fb_comments': float(wide_tt.loc['fb-comments', 'ctabgan-plus'] / wide_tt.loc['fb-comments', 'beta_copula']),
    'ratio_tvae_over_beta_fb_comments': float(wide_tt.loc['fb-comments', 'tvae'] / wide_tt.loc['fb-comments', 'beta_copula']),
    'ratio_tabddpm_over_beta_fb_comments': float(wide_tt.loc['fb-comments', 'tabddpm'] / wide_tt.loc['fb-comments', 'beta_copula']),
    'ratio_vine_over_beta_fb_comments': float(wide_tt.loc['fb-comments', 'vine_copula'] / wide_tt.loc['fb-comments', 'beta_copula']),
    'tabddpm_train_time_min_s': float(tab.min()), 'tabddpm_train_time_min_dataset': str(tab.idxmin()),
    'tabddpm_train_time_max_s': float(tab.max()), 'tabddpm_train_time_max_dataset': str(tab.idxmax()),
    'vine_copula_fb_comments_fit_time_s': float(wide_tt.loc['fb-comments', 'vine_copula']),
    'vine_copula_fb_comments_fit_time_human': human_time(wide_tt.loc['fb-comments', 'vine_copula']),
    'beta_copula_train_time_max_s': float(wide_tt['beta_copula'].max()),
    'beta_copula_train_time_max_dataset': str(wide_tt['beta_copula'].idxmax()),
    'beta_copula_train_time_total_s': float(wide_tt['beta_copula'].sum()),
    'beta_copula_sample5_median_s': float(wide_st['beta_copula'].median()),
    'beta_copula_sample5_max_s': float(wide_st['beta_copula'].max()),
    'beta_copula_sample5_max_dataset': str(wide_st['beta_copula'].idxmax()),
    'copula_train_time_max_s': {LABEL[m]: float(wide_tt[m].max()) for m in ('beta_copula', 'gaussian_copula', 'checkerboard_copula')},
    'train_status_counts_all_rows': {str(k): int(v) for k, v in train_status_counts.items()},
    'ctabgan_train_datasets': sorted(wide_tt['ctabgan'].dropna().index.tolist()),
    'ctabgan_no_config_datasets': sorted(train[(train['method'] == 'ctabgan') & (train['status'] == 'no_config')]['dataset'].tolist()),
}
headline['sample5_time_s'] = {LABEL[m]: {ds: (None if not np.isfinite(wide_st.loc[ds, m]) else float(wide_st.loc[ds, m]))
                                         for ds in DATASETS} for m in COST_METHODS if wide_st[m].notna().any()}
headline['train_time_s'] = {LABEL[m]: {ds: (None if not np.isfinite(wide_tt.loc[ds, m]) else float(wide_tt.loc[ds, m]))
                                       for ds in DATASETS} for m in COST_METHODS}
headline['summary'] = {r['method']: {k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                                     for k, v in r.items() if k not in ('method',)} for r in sum_rows}

lines = [r'\begin{tabular}{lrrrrrr}', r'\toprule',
         r'Method & $n$ & Median train & Total train & Median sample ($5\times$) & Median size (MiB) & Median \#params \\',
         r'\midrule']
for m in TABLE_ORDER:
    tt = wide_tt[m].dropna()
    st = wide_st[m].dropna()
    sp = wide_sp[m].dropna()
    npar = wide_np[m].dropna()
    lines.append(' & '.join([LABEL[m] + (r'$^{\dagger}$' if m == 'ctabgan' else ''), str(tt.size),
                             human_time(tt.median()), human_time(tt.sum()),
                             human_time(st.median()) if st.size else '---',
                             f'{sp.median() / 1024 ** 2:.1f}' if sp.size else '---',
                             human_count(npar.median()) if npar.size else '---']) + r' \\')
lines += [r'\bottomrule', r'\end{tabular}']
write_text('table_cost.tex',
           '% C4: wall-clock cost. $n$ = number of datasets with a training measurement; medians over those datasets. '
           'Total train = sum over the $n$ datasets ($^{\\dagger}$CTAB-GAN: 9 classification datasets only). '
           'Sample ($5\\times$) = time to draw 5 synthetic datasets of n_train rows (profiled for every method).\n' + '\n'.join(lines) + '\n')

results = load_results(JSON)
p2 = value_matrix(results, TABLE_METHODS, 'p2')
real_p2 = real_series(results, 'p2')
cu_rows = []
for ds in DATASETS:
    beta_cost = wide_cpd.loc[ds, 'beta_copula']
    for m in TABLE_METHODS:
        ck = COST_KEY.get(m, m)
        c = wide_cpd.loc[ds, ck]
        gap = p2.loc[ds, m] - real_p2[ds] if np.isfinite(real_p2[ds]) else np.nan
        cu_rows.append({'dataset': ds, 'method': DISPLAY[m], 'method_key': m, 'n_train': N_TRAIN[ds],
                        'train_time_s': wide_tt.loc[ds, ck], 'sample5_time_s': wide_st.loc[ds, ck],
                        'cost_per_draw_s': c, 'p2': p2.loc[ds, m], 'p2_real': real_p2[ds],
                        'p2_gap_to_real': gap,
                        'log10_cost_ratio_vs_beta': (np.log10(c / beta_cost) if np.isfinite(c) and np.isfinite(beta_cost) and c > 0 else np.nan)})
cu = pd.DataFrame(cu_rows)
with open(f'{OUT_DIR}/cost_utility.csv', 'w') as f:
    f.write('# cost_per_draw_s = train_time_s + sample5_time_s/5; p2_gap_to_real = p2_method - p2_real\n')
    cu.to_csv(f, index=False)
print(f'wrote {OUT_DIR}/cost_utility.csv')
cu_both = cu.dropna(subset=['cost_per_draw_s', 'p2_gap_to_real'])
med = cu_both.groupby('method').agg(n=('cost_per_draw_s', 'size'),
                                    median_cost_per_draw_s=('cost_per_draw_s', 'median'),
                                    median_p2_gap=('p2_gap_to_real', 'median'),
                                    mean_p2_gap=('p2_gap_to_real', 'mean'),
                                    median_log10_cost_ratio=('log10_cost_ratio_vs_beta', 'median'))
headline['cost_utility_medians'] = {m: {k: (None if not np.isfinite(v) else float(v)) for k, v in r.items()} for m, r in med.iterrows()}
write_facts('cost', headline)
print(summary[['method', 'n_datasets_train', 'train_time_median_human', 'train_time_total_human',
               'log10_train_ratio_vs_beta_median', 'cost_per_draw_median_s',
               'log10_cost_per_draw_ratio_vs_beta_median']].to_string(index=False))
print(med)
