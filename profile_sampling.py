import os
import sys
import gc
import csv
import time
import pickle
import argparse
import warnings

N_INSTANCES = 5
SEED0 = 0
CSV_OUT = os.path.join('results', 'profiling', 'sampling_params.csv')

SUBPROCESS_KEYS = {
    'tabddpm': 'tabddpm',
    'ctabgan': 'ctabgan',
    'ctabgan-plus': 'ctabgan_plus',
    'tvae': 'tvae',
}
INPROCESS = ['beta_copula', 'gaussian_copula', 'checkerboard_copula',
             'vine_copula']
ALL_METHODS = list(SUBPROCESS_KEYS) + INPROCESS

ARTIFACTS = {
    'tabddpm': ('ddpm_cb_best', 'model.pt'),
    'ctabgan': ('ctabgan', 'ctabgan.obj'),
    'ctabgan-plus': ('ctabgan-plus', 'ctabgan.obj'),
    'tvae': ('tvae', 'tvae.obj'),
}


def count_params(obj, _seen=None, _depth=0) -> float:
    if _seen is None:
        _seen = set()
    if _depth > 12 or id(obj) in _seen:
        return 0.0
    _seen.add(id(obj))

    import numpy as np
    try:
        import torch
        if isinstance(obj, torch.nn.Module):
            return float(sum(p.numel() for p in obj.parameters()))
        if isinstance(obj, torch.Tensor):
            return float(obj.numel())
    except ImportError:
        pass

    if isinstance(obj, np.ndarray):
        return float(obj.size)

    cls = type(obj).__module__

    total = 0.0
    if isinstance(obj, dict):
        for v in obj.values():
            total += count_params(v, _seen, _depth + 1)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            total += count_params(v, _seen, _depth + 1)
    elif hasattr(obj, '__dict__') and not cls.startswith(('builtins',)):
        for v in vars(obj).values():
            total += count_params(v, _seen, _depth + 1)
    return total


def count_artifact_params(exp_dir, ds, method) -> float:
    subdir, fname = ARTIFACTS[method]
    path = os.path.join(exp_dir, ds, subdir, fname)
    if not os.path.exists(path):
        return float('nan')

    for extra in ('CTAB-GAN', 'CTAB-GAN-Plus', 'CTGAN'):
        p = os.path.abspath(extra)
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)

    try:
        if fname.endswith('.pt'):
            import torch
            state = torch.load(path, map_location='cpu')
        else:
            with open(path, 'rb') as f:
                state = pickle.load(f)
        return count_params(state)
    except Exception as e:
        warnings.warn(f"[{ds}/{method}] could not count params from {path}: {e}")
        return float('nan')


def profile_subprocess(ds, method, task_type, train_df, exp_dir, data_dir):
    from benchmark import resample_baseline

    n_train = len(train_df)
    t0 = time.time()
    for seed in range(SEED0, SEED0 + N_INSTANCES):
        df = resample_baseline(exp_dir, ds, SUBPROCESS_KEYS[method], seed,
                               data_dir, task_type, n_train=n_train)
        if df is None:
            return {'status': f'sample_failed(seed={seed})',
                    'sample5_time_s': float('nan'), 'n_params': float('nan')}
    sample_time = time.time() - t0

    n_params = count_artifact_params(exp_dir, ds, method)
    return {'status': 'success', 'sample5_time_s': sample_time,
            'n_params': n_params}


def profile_inprocess(ds, method, task_type, train_df):
    n = len(train_df)

    if method == 'beta_copula':
        from beta_copula import fit_beta_copula, sample_beta_copula
        state = fit_beta_copula(train_df, seed=SEED0)
        sample = lambda s: sample_beta_copula(state, n, seed=s)
        sizeable = {k: v for k, v in state.items() if k != 'template'}
    elif method == 'gaussian_copula':
        from gaussian_copula import fit_gaussian_copula, sample_gaussian_copula
        state = fit_gaussian_copula(train_df, seed=SEED0)
        sample = lambda s: sample_gaussian_copula(state, n, seed=s)
        sizeable = {k: v for k, v in state.items() if k != 'template'}
    elif method == 'checkerboard_copula':
        from checkerboard_copula import (fit_checkerboard_copula,
                                         sample_checkerboard_copula)
        state = fit_checkerboard_copula(train_df, seed=SEED0)
        sample = lambda s: sample_checkerboard_copula(state, n, seed=s)
        sizeable = {k: v for k, v in state.items() if k != 'template'}
    elif method == 'vine_copula':
        from vine_copula import fit_vine_copula, sample_vine_copula
        state = fit_vine_copula(train_df, seed=SEED0)
        sample = lambda s: sample_vine_copula(state, n, seed=s)
        sizeable = {k: v for k, v in state.items()
                    if k not in ('template', 'vine')}
        vine_params = float(state['vine'].npars) + float(state['vine'].matrix.size)
    else:
        return {'status': 'unknown_method',
                'sample5_time_s': float('nan'), 'n_params': float('nan')}

    t0 = time.time()
    for seed in range(SEED0, SEED0 + N_INSTANCES):
        sample(seed)
    sample_time = time.time() - t0

    n_params = count_params(sizeable)
    if method == 'vine_copula':
        n_params += vine_params

    return {'status': 'success', 'sample5_time_s': sample_time,
            'n_params': n_params}


FIELDS = ['dataset', 'method', 'status', 'sample5_time_s', 'n_params']


def read_prior(path):
    prior = {}
    if not os.path.exists(path):
        return prior
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            if row.get('dataset') and row.get('method'):
                prior[(row['dataset'], row['method'])] = row
    return prior


def write_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in FIELDS})


def _fmt(rec):
    def fnum(v):
        try:
            return f'{float(v):.2f}'
        except (TypeError, ValueError):
            return '---'
    return f"sample5 {fnum(rec.get('sample5_time_s'))}s, params {fnum(rec.get('n_params'))}"


def main():
    parser = argparse.ArgumentParser(
        description='Profile sampling time (5 full synthetic datasets) and '
                    'parameter counts for every generator.')
    parser.add_argument('--datasets', nargs='+', default=None)
    parser.add_argument('--methods', nargs='+', default=None,
                        help=f'Subset of: {ALL_METHODS}')
    parser.add_argument('--data_dir', default='data/')
    parser.add_argument('--exp_dir', default='exp/')
    parser.add_argument('--force', action='store_true',
                        help='Recompute pairs already present in the CSV')
    args = parser.parse_args()

    from benchmark import DATASETS, load_dataset_as_dataframes

    datasets = args.datasets or list(DATASETS)
    methods = args.methods or ALL_METHODS
    for m in methods:
        if m not in ALL_METHODS:
            sys.exit(f"ERROR: unknown method '{m}'. Valid: {ALL_METHODS}")

    prior = {} if args.force else read_prior(CSV_OUT)
    rows, jobs = [], [(ds, m) for ds in datasets for m in methods]
    job_set = set(jobs)

    for i, (ds, method) in enumerate(jobs, 1):
        prev = prior.get((ds, method))
        if prev is not None and prev.get('status') == 'success':
            print(f"[{i}/{len(jobs)}] {ds}/{method}: RESUMED ({_fmt(prev)})")
            rows.append(prev)
            write_csv(rows, CSV_OUT)
            continue

        task_type = DATASETS.get(ds, {}).get('task')
        if task_type is None:
            print(f"[{i}/{len(jobs)}] {ds}: unknown dataset, skipping")
            continue

        print(f"[{i}/{len(jobs)}] {ds}/{method}: profiling ...")
        try:
            train_df, _, _ = load_dataset_as_dataframes(args.data_dir, ds, task_type)
            if method in SUBPROCESS_KEYS:
                rec = profile_subprocess(ds, method, task_type, train_df,
                                         args.exp_dir, args.data_dir)
            else:
                rec = profile_inprocess(ds, method, task_type, train_df)
        except Exception as e:
            warnings.warn(f"[{ds}/{method}] failed: {e}")
            rec = {'status': f'failed({type(e).__name__})',
                   'sample5_time_s': float('nan'), 'n_params': float('nan')}

        rec.update({'dataset': ds, 'method': method})
        print(f"[{i}/{len(jobs)}] {ds}/{method}: {rec['status'].upper()} ({_fmt(rec)})")
        rows.append(rec)
        write_csv(rows, CSV_OUT)
        gc.collect()

    for key, prev in prior.items():
        if key not in job_set:
            rows.append(prev)
    write_csv(rows, CSV_OUT)
    print(f"\nCSV written to: {CSV_OUT}")


if __name__ == '__main__':
    main()
