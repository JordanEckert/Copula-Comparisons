import os
import sys
import gc
import csv
import time
import pickle
import argparse
import warnings
import subprocess


DATA_DIR = 'data'
EXP_DIR = 'exp'
CSV_OUT = os.path.join('results', 'training_cost', 'train_metrics.csv')


METHODS = {
    'tabddpm': {
        'kind': 'subprocess', 'exp_subdir': 'ddpm_cb_best',
        'pipeline': 'scripts_tabddpm/pipeline.py', 'artifact': 'model.pt',
    },
    'ctabgan': {
        'kind': 'subprocess', 'exp_subdir': 'ctabgan',
        'pipeline': 'CTAB-GAN/pipeline_ctabgan.py', 'artifact': 'ctabgan.obj',
    },
    'ctabgan-plus': {
        'kind': 'subprocess', 'exp_subdir': 'ctabgan-plus',
        'pipeline': 'CTAB-GAN-Plus/pipeline_ctabganp.py', 'artifact': 'ctabgan.obj',
    },
    'tvae': {
        'kind': 'subprocess', 'exp_subdir': 'tvae',
        'pipeline': 'CTGAN/pipeline_tvae.py', 'artifact': 'tvae.obj',
    },
    'beta_copula': {'kind': 'inprocess'},
    'gaussian_copula': {'kind': 'inprocess'},
    'checkerboard_copula': {'kind': 'inprocess'},
    'vine_copula': {'kind': 'inprocess'},
}

DATASETS = [
    'abalone', 'adult', 'buddy', 'california', 'cardio', 'churn2',
    'diabetes', 'fb-comments', 'gesture', 'higgs-small',
    'house', 'insurance', 'king', 'miniboone', 'wilt',
]

SEED = 0


def _path_bytes(path) -> float:
    if path and os.path.exists(path):
        return float(os.path.getsize(path))
    return float('nan')


def _human_bytes(n: float) -> str:
    if n is None or (isinstance(n, float) and n != n):
        return '---'
    for unit in ('B', 'KB', 'MB', 'GB'):
        if n < 1024.0:
            return f'{n:.1f} {unit}'
        n /= 1024.0
    return f'{n:.1f} TB'


def _human_time(s: float) -> str:
    if s is None or (isinstance(s, float) and s != s):
        return '---'
    return f'{s:.2f}s' if s < 60 else f'{s / 60:.2f}m'


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and v == v


def cleanup_memory():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _rec(ds, method, status, t=float('nan'), b=float('nan')):
    return {'dataset': ds, 'method': method, 'status': status,
            'train_time_s': t, 'train_space_bytes': b}


def train_subprocess(ds, method, info, force):
    exp_subdir = os.path.join(EXP_DIR, ds, info['exp_subdir'])
    config_path = os.path.join(exp_subdir, 'config.toml')
    artifact_path = os.path.join(exp_subdir, info['artifact'])

    if not os.path.exists(config_path):
        return _rec(ds, method, 'no_config')
    if os.path.exists(artifact_path) and not force:
        return _rec(ds, method, 'already_trained', b=_path_bytes(artifact_path))

    env = {**os.environ, 'PYTHONPATH': '.'}
    t0 = time.time()
    try:
        subprocess.run(
            ['python3', info['pipeline'], '--config', config_path, '--train'],
            check=True, capture_output=True, text=True, env=env,
        )
    except subprocess.CalledProcessError as e:
        warnings.warn(f"[{ds}/{method}] train failed: {e.stderr[:300]}")
        return _rec(ds, method, f'train_failed(exit={e.returncode})',
                    t=time.time() - t0)
    train_time = time.time() - t0
    return _rec(ds, method, 'success', t=train_time, b=_path_bytes(artifact_path))


def _pickle_size(obj) -> float:
    try:
        return float(len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)))
    except Exception as e:
        warnings.warn(f"could not measure serialized size: {e}")
        return float('nan')


def train_inprocess(ds, method, train_df, task_type):
    t0 = time.time()
    try:
        if method == 'beta_copula':
            from beta_copula import fit_beta_copula
            state = fit_beta_copula(train_df, seed=SEED)
        elif method == 'gaussian_copula':
            from gaussian_copula import fit_gaussian_copula
            state = fit_gaussian_copula(train_df, seed=SEED)
        elif method == 'checkerboard_copula':
            from checkerboard_copula import fit_checkerboard_copula
            state = fit_checkerboard_copula(train_df, seed=SEED)
        elif method == 'vine_copula':
            from vine_copula import fit_vine_copula
            state = fit_vine_copula(train_df, seed=SEED)
        else:
            return _rec(ds, method, 'unknown_method')
    except Exception as e:
        warnings.warn(f"[{ds}/{method}] fit failed: {e}")
        return _rec(ds, method, f'train_failed({type(e).__name__})',
                    t=time.time() - t0)
    train_time = time.time() - t0

    return _rec(ds, method, 'success', t=train_time, b=_pickle_size(state))


DONE_STATUSES = ('success', 'already_trained')


def read_prior(path):
    prior = {}
    if not os.path.exists(path):
        return prior
    try:
        with open(path, newline='') as f:
            for row in csv.DictReader(f):
                ds, method = row.get('dataset'), row.get('method')
                if not ds or not method:
                    continue
                tt, ts = row.get('train_time_s', ''), row.get('train_space_bytes', '')
                prior[(ds, method)] = {
                    'dataset': ds, 'method': method,
                    'status': row.get('status', ''),
                    'train_time_s': float(tt) if tt else float('nan'),
                    'train_space_bytes': float(ts) if ts else float('nan'),
                }
    except Exception as e:
        warnings.warn(f"could not read prior results from {path}: {e}")
    return prior


def write_csv(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dataset', 'method', 'status',
                    'train_time_s', 'train_space_bytes'])
        for r in results:
            w.writerow([
                r['dataset'], r['method'], r['status'],
                f"{r['train_time_s']:.4f}" if _is_num(r['train_time_s']) else '',
                int(r['train_space_bytes']) if _is_num(r['train_space_bytes']) else '',
            ])


def main():
    parser = argparse.ArgumentParser(
        description='Train all generative models sequentially and record '
                    'train time + disk space.'
    )
    parser.add_argument('--force', action='store_true',
                        help='Re-train even if a model artifact already exists')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Subset of datasets (default: all 15)')
    parser.add_argument('--methods', nargs='+', default=None,
                        help='Subset of methods (default: all): '
                             + ' '.join(METHODS))
    args = parser.parse_args()

    datasets = args.datasets if args.datasets else DATASETS
    methods = args.methods if args.methods else list(METHODS)

    for m in methods:
        if m not in METHODS:
            print(f"ERROR: unknown method '{m}'. Valid: {list(METHODS)}")
            sys.exit(1)

    prior = {} if args.force else read_prior(CSV_OUT)

    results = []
    jobs = [(ds, m) for ds in datasets for m in methods]
    job_set = set(jobs)
    total = len(jobs)

    carry = [prev for key, prev in prior.items() if key not in job_set]

    for i, (ds, method) in enumerate(jobs, 1):
        info = METHODS[method]

        prev = prior.get((ds, method))
        if prev is not None and prev['status'] in DONE_STATUSES:
            print(f"\n[{i}/{total}] {ds} / {method}: RESUMED "
                  f"(already done: {prev['status']}, "
                  f"train {_human_time(prev['train_time_s'])})")
            results.append(prev)
            write_csv(results + carry, CSV_OUT)
            continue

        print(f"\n[{i}/{total}] {ds} / {method}: TRAINING ({info['kind']})")
        if info['kind'] == 'subprocess':
            rec = train_subprocess(ds, method, info, args.force)
        else:
            from benchmark import (DATASETS as BENCH_DATASETS,
                                   load_dataset_as_dataframes)
            task_type = BENCH_DATASETS.get(ds, {}).get('task')
            if task_type is None:
                rec = _rec(ds, method, 'unknown_dataset')
            else:
                try:
                    train_df, _, _ = load_dataset_as_dataframes(
                        DATA_DIR, ds, task_type)
                    rec = train_inprocess(ds, method, train_df, task_type)
                except Exception as e:
                    warnings.warn(f"[{ds}] could not load: {e}")
                    rec = _rec(ds, method, f'load_failed({type(e).__name__})')

        print(f"[{i}/{total}] {ds} / {method}: {rec['status'].upper()} "
              f"(train {_human_time(rec['train_time_s'])}, "
              f"{_human_bytes(rec['train_space_bytes'])})")
        results.append(rec)
        write_csv(results + carry, CSV_OUT)
        cleanup_memory()

    results.extend(carry)
    write_csv(results, CSV_OUT)

    print(f"\n\n{'=' * 70}\nTRAINING SUMMARY\n{'=' * 70}")
    print(f"{'Dataset':<13} {'Method':<16} {'Status':<22} "
          f"{'TrainTime':>10} {'TrainSpace':>11}")
    print('-' * 74)
    for r in results:
        print(f"{r['dataset']:<13} {r['method']:<16} {r['status']:<22} "
              f"{_human_time(r['train_time_s']):>10} "
              f"{_human_bytes(r['train_space_bytes']):>11}")

    n_ok = sum(1 for r in results if r['status'] == 'success')
    n_skip = sum(1 for r in results
                 if r['status'] in ('no_config', 'already_trained'))
    n_fail = sum(1 for r in results if 'failed' in r['status'])
    print(f"\nTrained: {n_ok} | Skipped: {n_skip} | Failed: {n_fail}")
    print(f"CSV written to: {CSV_OUT}")


if __name__ == '__main__':
    main()
