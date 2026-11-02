import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from common import (JSON, RANK_METHODS, DISPLAY, load_results, friedman_block, write_facts, FIG_DIR, OUT_DIR)

os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral', 'STIX Two Text', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 6.5, 'legend.frameon': False, 'legend.handletextpad': 0.4, 'legend.columnspacing': 0.8,
    'axes.linewidth': 0.5, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.minor.width': 0.3, 'ytick.minor.width': 0.3, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.minor.size': 1.5, 'ytick.minor.size': 1.5, 'xtick.direction': 'out', 'ytick.direction': 'out',
    'axes.grid': True, 'grid.color': '#e6e6e3', 'grid.linewidth': 0.5, 'grid.linestyle': '-',
    'axes.axisbelow': True, 'axes.edgecolor': '#444444', 'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 300,
})
SURFACE = '#ffffff'
INK = '#111111'
GRAY = '#8c8c8c'

STYLE = {
    'beta_copula':         ('#D55E00', 'o', True),
    'checkerboard_copula': ('#E69F00', 's', True),
    'gaussian_copula':     ('#CC79A7', '^', True),
    'vine_copula':         ('#009E73', 'v', True),
    'tabddpm':             ('#0072B2', 'D', True),
    'ctabgan_plus':        ('#56B4E9', 'p', True),
    'ctabgan':             ('#56B4E9', 'p', False),
    'tvae':                ('#8c8c8c', 'h', True),
}
MS = {'o': 1.0, 's': 0.92, '^': 1.1, 'v': 1.1, 'D': 0.85, 'p': 1.15, 'h': 1.1, 'x': 1.05}


def mk(ax, x, y, m, size, zorder=3, alpha=1.0, **kw):
    hue, marker, filled = STYLE[m]
    s = (size * MS[marker]) ** 2
    if marker == 'x':
        return ax.scatter(x, y, s=s, marker=marker, c=hue, linewidths=0.9, zorder=zorder, alpha=alpha, **kw)
    if filled:
        return ax.scatter(x, y, s=s, marker=marker, c=hue, edgecolors=SURFACE, linewidths=0.5,
                          zorder=zorder, alpha=alpha, **kw)
    return ax.scatter(x, y, s=s, marker=marker, facecolors=SURFACE, edgecolors=hue, linewidths=0.9,
                      zorder=zorder, alpha=alpha, **kw)


def handle(m, size=5.5, line=False):
    hue, marker, filled = STYLE[m]
    if marker == 'x':
        return Line2D([0], [0], marker=marker, color=hue if line else 'none', markeredgecolor=hue,
                      markerfacecolor=hue, markersize=size * MS[marker], markeredgewidth=0.9, lw=0.9 if line else 0)
    return Line2D([0], [0], marker=marker, color=hue if line else 'none', lw=0.9 if line else 0,
                  markerfacecolor=hue if filled else SURFACE, markeredgecolor=SURFACE if filled else hue,
                  markeredgewidth=0.5 if filled else 0.9, markersize=size * MS[marker])


def save(fig, name, tight=False):
    kw = {'bbox_inches': 'tight', 'pad_inches': 0.02} if tight else {}
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG_DIR, f'{name}.{ext}'), facecolor=SURFACE, **kw)
    plt.close(fig)
    print(f'wrote {FIG_DIR}/{name}.pdf / .png')


fig_facts = {}

cu = pd.read_csv(os.path.join(OUT_DIR, 'cost_utility.csv'), comment='#')
D1_METHODS = ['tvae', 'ctabgan', 'ctabgan_plus', 'tabddpm', 'beta_copula']
YLIM = (-0.6, 0.05)
sub = cu[cu['method_key'].isin(D1_METHODS)].dropna(subset=['cost_per_draw_s', 'p2_gap_to_real'])
clipped = sub[(sub['p2_gap_to_real'] < YLIM[0]) | (sub['p2_gap_to_real'] > YLIM[1])]
fig_facts['D1'] = {
    'n_points_total': int(len(sub)),
    'n_points_per_method': {DISPLAY[m]: int((sub['method_key'] == m).sum()) for m in D1_METHODS},
    'n_clipped': int(len(clipped)),
    'clipped_points': [{'dataset': r['dataset'], 'method': r['method'], 'cost_per_draw_s': float(r['cost_per_draw_s']),
                        'p2_gap_to_real': float(r['p2_gap_to_real'])} for _, r in clipped.iterrows()],
    'ylim': list(YLIM),
    'medians': {}}

fig, ax = plt.subplots(figsize=(3.25, 2.2), layout='constrained')
fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01)
ax.set_xscale('log')
ax.axhline(0, color=INK, lw=0.6, ls=(0, (4, 3)), zorder=1)
shown = sub[(sub['p2_gap_to_real'] >= YLIM[0]) & (sub['p2_gap_to_real'] <= YLIM[1])]
for m in D1_METHODS:
    d = shown[shown['method_key'] == m]
    mk(ax, d['cost_per_draw_s'], d['p2_gap_to_real'], m, size=3.6, zorder=3, alpha=0.75)
for m in D1_METHODS:
    d = sub[sub['method_key'] == m]
    mc, mg = d['cost_per_draw_s'].median(), d['p2_gap_to_real'].median()
    fig_facts['D1']['medians'][DISPLAY[m]] = {'n': int(len(d)), 'median_cost_per_draw_s': float(mc),
                                              'median_p2_gap': float(mg)}
    mk(ax, [mc], [mg], m, size=9.5, zorder=5)
ax.set_ylim(*YLIM)
ax.set_xlim(5e-3, 3e6)
ax.set_xlabel('Cost per synthetic draw, train + sample (s, log)')
ax.set_ylabel(r'P2 gap to Real ($\mathrm{P2}_{\rm synth} - \mathrm{P2}_{\rm real}$)')
ax.set_yticks([0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6])
ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
n_clip = len(clipped)
if n_clip:
    who = '; '.join(f"{r['method']}/{r['dataset']}" for _, r in clipped.iterrows())
    ax.text(0.02, 0.03, f"{n_clip} point{'s' if n_clip != 1 else ''} below $-0.6$ not shown ({who})",
            transform=ax.transAxes,
            fontsize=6, color='#555555', ha='left', va='bottom')
legend_order = ['beta_copula', 'tabddpm', 'ctabgan_plus', 'ctabgan', 'tvae']
handles = [handle(m, size=5.0) for m in legend_order]
labels = [DISPLAY[m] for m in legend_order]
leg = ax.legend(handles, labels, loc='lower left', bbox_to_anchor=(0.0, 0.16), ncol=2, handlelength=1.0,
                borderaxespad=0.2, labelspacing=0.3)
ax.text(0.02, 0.10 if n_clip else 0.03, 'small: one dataset;  large: per-method median', transform=ax.transAxes,
        fontsize=6, color='#555555', ha='left', va='bottom')
save(fig, 'cost_utility_pareto')


def cd_panel(ax, mean_r, cd, labels, lo_hi=None, label_fs=7, tick_fs=6.5):
    order = mean_r.sort_values()
    methods, values = list(order.index), order.values
    k = len(methods)
    lo, hi = (1.0, float(k)) if lo_hi is None else lo_hi
    y_axis = 0.0
    ax.plot([lo, hi], [y_axis, y_axis], color=INK, lw=0.7, solid_capstyle='butt')
    for t in range(int(np.ceil(lo)), int(np.floor(hi)) + 1):
        ax.plot([t, t], [y_axis, y_axis + 0.045], color=INK, lw=0.6)
        ax.text(t, y_axis + 0.07, f'{t}', ha='center', va='bottom', fontsize=tick_fs, color=INK)
    y_cd = y_axis + 0.30
    ax.plot([lo, lo + cd], [y_cd, y_cd], color=INK, lw=0.9, solid_capstyle='butt')
    for xx in (lo, lo + cd):
        ax.plot([xx, xx], [y_cd - 0.03, y_cd + 0.03], color=INK, lw=0.7)
    ax.text(lo + cd / 2, y_cd + 0.05, f'CD = {cd:.2f}', ha='center', va='bottom', fontsize=tick_fs, color=INK)
    raw = []
    for i in range(k):
        j = i
        while j + 1 < k and values[j + 1] - values[i] <= cd:
            j += 1
        if j > i:
            raw.append((i, j))
    cliques = [c for c in raw if not any(o != c and o[0] <= c[0] and o[1] >= c[1] for o in raw)]
    row_end, rows = [], []
    for i, j in cliques:
        placed = False
        for r, end in enumerate(row_end):
            if values[i] > end + 0.05:
                row_end[r] = values[j]; rows.append(r); placed = True; break
        if not placed:
            row_end.append(values[j]); rows.append(len(row_end) - 1)
    for (i, j), r in zip(cliques, rows):
        y_bar = y_axis - 0.06 - 0.055 * r
        ax.plot([min(values[i], values[j]) - 0.05, max(values[i], values[j]) + 0.05], [y_bar, y_bar], color=INK, lw=2.2, solid_capstyle='butt', zorder=2)
    n_left = (k + 1) // 2
    step = 0.16
    stem_top = y_axis - 0.06 - 0.055 * max(len(row_end) - 1, 0) - 0.03
    pad = 0.08 * (hi - lo)
    texts = []
    for i in range(k):
        left = i < n_left
        j = i if left else i - n_left
        y_stem = stem_top - step * (j + 1)
        x_end = lo - pad if left else hi + pad
        ax.plot([values[i], values[i]], [y_axis, y_stem], color='#777777', lw=0.5, zorder=1)
        ax.plot([values[i], x_end], [y_stem, y_stem], color='#777777', lw=0.5, zorder=1)
        texts.append(ax.text(x_end - 0.02 * (hi - lo) if left else x_end + 0.02 * (hi - lo), y_stem,
                             f'{labels[methods[i]]}', ha='right' if left else 'left',
                             va='center', fontsize=label_fs, color=INK))
    ax.set_ylim(stem_top - step * (n_left + 0.4), y_cd + 0.20)
    ax.axis('off')
    ax.set_xlim(lo - pad - 0.5 * (hi - lo), hi + pad + 0.5 * (hi - lo))
    fig = ax.figure
    for _ in range(5):
        fig.canvas.draw()
        inv = ax.transData.inverted()
        xs = []
        for t in texts:
            bb = t.get_window_extent()
            xs += [inv.transform((bb.x0, bb.y0))[0], inv.transform((bb.x1, bb.y1))[0]]
        x0, x1 = min(xs + [lo - pad]), max(xs + [hi + pad])
        m = 0.015 * (x1 - x0)
        ax.set_xlim(x0 - m, x1 + m)
    return cliques, methods, values


def cd_figure(name, results, methods, panels, figsize, label_fs=7, nrows=1):
    ncols = int(np.ceil(len(panels) / nrows))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()
    out = {}
    for ax, (metric, tag) in zip(axes, panels):
        b = friedman_block(results, methods, metric)
        cliques, order, vals = cd_panel(ax, b['mean_rank'], b['cd'], DISPLAY, label_fs=label_fs)
        ax.text(0.0, 1.0, tag, transform=ax.transAxes, fontsize=7.5, ha='left', va='top', color=INK)
        out[metric] = {'n': b['n'], 'k': b['k'], 'cd': b['cd'], 'p': b['p'], 'chi2': b['chi2'],
                       'order_best_first': [DISPLAY[m] for m in order],
                       'mean_ranks': {DISPLAY[m]: float(v) for m, v in zip(order, vals)},
                       'cliques': [[DISPLAY[m] for m in order[i:j + 1]] for i, j in cliques]}
    fig.subplots_adjust(wspace=0.06, hspace=0.12, left=0.0, right=1.0, top=0.97, bottom=0.02)
    save(fig, name)
    return out


results = load_results(JSON)
fig_facts['D2'] = cd_figure('cd_diagrams', results, RANK_METHODS,
                            [('p2', '(a) P2'), ('corr_l2', '(b) CorrL2'),
                             ('ci_overlap', '(c) CI overlap'), ('assoc_l2', '(d) AssocL2')],
                            figsize=(6.75, 3.9), label_fs=6.5, nrows=2)

cpd = pd.read_csv(os.path.join(OUT_DIR, 'cost_per_dataset.csv'), comment='#')
D4_METHODS = ['ctabgan_plus', 'ctabgan', 'tvae', 'tabddpm', 'vine_copula', 'gaussian_copula',
              'checkerboard_copula', 'beta_copula']
fig, ax = plt.subplots(figsize=(3.25, 2.85), layout='constrained')
fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01)
ax.set_xscale('log'); ax.set_yscale('log')
d4 = {}
for m in D4_METHODS:
    key = m.replace('_plus', '-plus')
    d = cpd[cpd['method_key'] == key].dropna(subset=['train_time_s']).sort_values('n_train')
    d = d[d['train_time_s'] > 0]
    hue, marker, filled = STYLE[m]
    ax.plot(d['n_train'], d['train_time_s'], color=hue, lw=0.8, zorder=2, alpha=0.9)
    mk(ax, d['n_train'], d['train_time_s'], m, size=3.8, zorder=3)
    d4[DISPLAY[m]] = {'n_points': int(len(d)),
                      'slope_log_log': float(np.polyfit(np.log10(d['n_train']), np.log10(d['train_time_s']), 1)[0])
                      if len(d) > 1 else None}
fig_facts['D4'] = {'methods_drawn': [DISPLAY[m] for m in D4_METHODS], 'log_log_slopes': d4}
ax.set_xlabel(r'Training rows $n_{\rm train}$ (log)')
ax.set_ylabel('Training wall-clock time (s, log)')
ax.set_ylim(5e-4, 2e6)
ax.set_yticks([1e-3, 1e-1, 1e1, 1e3, 1e5])
ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.set_xlim(300, 4e5)
legend_order = ['ctabgan_plus', 'ctabgan', 'tvae', 'tabddpm', 'vine_copula', 'gaussian_copula',
                'checkerboard_copula', 'beta_copula']
handles = [handle(m, size=4.6, line=True) for m in legend_order]
labels = [DISPLAY[m] for m in legend_order]
ax.legend(handles, labels, loc='lower left', bbox_to_anchor=(-0.02, 1.0), ncol=4, handlelength=1.5,
          borderaxespad=0.0, labelspacing=0.25, columnspacing=0.6, mode=None)
save(fig, 'train_time_vs_n')

write_facts('figures', fig_facts)
