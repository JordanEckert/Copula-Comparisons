import os
for _var in (
    'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
    'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
    'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
):
    os.environ.setdefault(_var, '1')

import json
import argparse
import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Tuple, Optional, Any
from copy import deepcopy
import time
import subprocess
import tempfile
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from threadpoolctl import threadpool_limits as _threadpool_limits
except ImportError:
    _threadpool_limits = None

from sklearn.metrics import f1_score, r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings(
    'ignore',
    message='Found unknown categories',
    category=UserWarning,
)

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    warnings.warn("CatBoost not installed. Protocol 2 evaluation will be skipped.")

from beta_copula import generate_beta_copula
from eval_metrics import (standardized_wasserstein, dcr_stats, association_l2,
                          pmse_ratio, inference_fidelity, alpha_beta_authenticity)
from gaussian_copula import generate_gaussian_copula
from checkerboard_copula import generate_checkerboard_copula
try:
    from vine_copula import fit_vine_copula, sample_vine_copula
    VINE_AVAILABLE = True
except ImportError:
    VINE_AVAILABLE = False


DATASETS = {
    'abalone':       {'abbr': 'AB',  'task': 'regression'},
    'adult':         {'abbr': 'AD',  'task': 'binclass'},
    'buddy':         {'abbr': 'BU',  'task': 'multiclass'},
    'california':    {'abbr': 'CA',  'task': 'regression'},
    'cardio':        {'abbr': 'CAR', 'task': 'binclass'},
    'churn2':        {'abbr': 'CH',  'task': 'binclass'},
    'diabetes':      {'abbr': 'DI',  'task': 'binclass'},
    'fb-comments':   {'abbr': 'FB',  'task': 'regression'},
    'gesture':       {'abbr': 'GE',  'task': 'multiclass'},
    'higgs-small':   {'abbr': 'HI',  'task': 'binclass'},
    'house':         {'abbr': 'HO',  'task': 'regression'},
    'insurance':     {'abbr': 'IN',  'task': 'regression'},
    'king':          {'abbr': 'KI',  'task': 'regression'},
    'miniboone':     {'abbr': 'MI',  'task': 'binclass'},
    'wilt':          {'abbr': 'WI',  'task': 'binclass'},
}

BASELINE_METHODS = {
    'tabddpm':       'ddpm_cb_best',
    'ctabgan':       'ctabgan',
    'ctabgan_plus':  'ctabgan-plus',
    'tvae':          'tvae',
}

BASELINE_PIPELINES = {
    'tabddpm':       ('scripts_tabddpm/pipeline.py', '--sample'),
    'ctabgan':       ('CTAB-GAN/pipeline_ctabgan.py', '--sample'),
    'ctabgan_plus':  ('CTAB-GAN-Plus/pipeline_ctabganp.py', '--sample'),
    'tvae':          ('CTGAN/pipeline_tvae.py', '--sample'),
}


def load_info(data_dir: str, ds_name: str) -> dict:
    info_path = os.path.join(data_dir, ds_name, 'info.json')
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            return json.load(f)
    return {}


def load_split_numpy(data_dir: str, ds_name: str, split: str) -> Tuple[
    Optional[np.ndarray], Optional[np.ndarray], np.ndarray
]:
    path = os.path.join(data_dir, ds_name)

    num_path = os.path.join(path, f'X_num_{split}.npy')
    X_num = np.load(num_path).astype(np.float64) if os.path.exists(num_path) else None

    cat_path = os.path.join(path, f'X_cat_{split}.npy')
    X_cat = np.load(cat_path, allow_pickle=True) if os.path.exists(cat_path) else None

    y = np.load(os.path.join(path, f'y_{split}.npy'), allow_pickle=True)

    return X_num, X_cat, y


def numpy_to_dataframe(
    X_num: Optional[np.ndarray],
    X_cat: Optional[np.ndarray],
    y: np.ndarray,
    info: dict,
    task_type: str,
    include_target: bool = True
) -> pd.DataFrame:
    parts = {}

    if X_num is not None:
        for j in range(X_num.shape[1]):
            parts[f'num_{j}'] = X_num[:, j].astype(np.float64)

    if X_cat is not None:
        for j in range(X_cat.shape[1]):
            raw_str = [str(v) for v in X_cat[:, j]]
            unique_vals = sorted(set(raw_str))
            parts[f'cat_{j}'] = pd.Categorical(raw_str, categories=unique_vals)

    if include_target:
        if task_type in ('binclass', 'multiclass'):
            y_str = [str(v) for v in y]
            unique_y = sorted(set(y_str))
            parts['target'] = pd.Categorical(y_str, categories=unique_y)
        else:
            parts['target'] = y.astype(np.float64)

    return pd.DataFrame(parts)


def load_dataset_as_dataframes(data_dir: str, ds_name: str, task_type: str) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    info = load_info(data_dir, ds_name)
    frames = {}
    for split in ['train', 'val', 'test']:
        X_num, X_cat, y = load_split_numpy(data_dir, ds_name, split)
        frames[split] = numpy_to_dataframe(X_num, X_cat, y, info, task_type)
    return frames['train'], frames['val'], frames['test']


try:
    import tomli
except ImportError:
    import tomllib as tomli
import tomli_w


def resample_baseline(
    exp_dir: str, ds_name: str, method_key: str, seed: int,
    data_dir: str, task_type: str, n_train: int = 0
) -> Optional[pd.DataFrame]:
    method_dir = BASELINE_METHODS[method_key]
    source_dir = os.path.join(exp_dir, ds_name, method_dir)
    config_path = os.path.join(source_dir, 'config.toml')

    if not os.path.exists(config_path):
        return None

    with open(config_path, 'rb') as f:
        config = tomli.load(f)

    pipeline_script, _ = BASELINE_PIPELINES[method_key]
    model_files = {
        'tabddpm':      ['model.pt'],
        'ctabgan':       ['ctabgan.obj'],
        'ctabgan_plus':  ['ctabgan.obj'],
        'tvae':          ['tvae.obj'],
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        config['parent_dir'] = tmp_dir
        config['sample']['seed'] = seed
        config['seed'] = seed
        config['device'] = 'cpu'
        if n_train > 0 and 'num_samples' in config.get('sample', {}):
            config['sample']['num_samples'] = n_train

        tmp_config_path = os.path.join(tmp_dir, 'config.toml')
        with open(tmp_config_path, 'wb') as f:
            tomli_w.dump(config, f)

        for mf in model_files.get(method_key, []):
            src = os.path.join(source_dir, mf)
            if os.path.exists(src):
                shutil.copy2(src, tmp_dir)
            else:
                warnings.warn(f"Model file not found: {src}")
                return None

        cmd = ['python3', pipeline_script, '--config', tmp_config_path, '--sample']
        try:
            result = subprocess.run(
                cmd, check=True, capture_output=True, text=True,
                env={**os.environ, 'PYTHONPATH': '.'}
            )
        except subprocess.CalledProcessError as e:
            warnings.warn(f"Baseline sampling failed for {method_key} seed {seed}: "
                          f"{e.stderr[:500]}")
            return None

        y_path = os.path.join(tmp_dir, 'y_train.npy')
        if not os.path.exists(y_path):
            warnings.warn(f"No y_train.npy produced by {method_key} seed {seed}")
            return None

        y = np.load(y_path, allow_pickle=True)

        num_path = os.path.join(tmp_dir, 'X_num_train.npy')
        X_num = np.load(num_path).astype(np.float64) if os.path.exists(num_path) else None

        cat_path = os.path.join(tmp_dir, 'X_cat_train.npy')
        X_cat = np.load(cat_path, allow_pickle=True) if os.path.exists(cat_path) else None

        info = load_info(data_dir, ds_name)
        return numpy_to_dataframe(X_num, X_cat, y, info, task_type)

def load_baseline_synthetic(
    exp_dir: str, ds_name: str, method_dir: str, data_dir: str, task_type: str
) -> Optional[pd.DataFrame]:
    synth_path = os.path.join(exp_dir, ds_name, method_dir)

    y_path = os.path.join(synth_path, 'y_train.npy')
    if not os.path.exists(y_path):
        return None

    y = np.load(y_path, allow_pickle=True)

    num_path = os.path.join(synth_path, 'X_num_train.npy')
    X_num = np.load(num_path).astype(np.float64) if os.path.exists(num_path) else None

    cat_path = os.path.join(synth_path, 'X_cat_train.npy')
    X_cat = np.load(cat_path, allow_pickle=True) if os.path.exists(cat_path) else None

    info = load_info(data_dir, ds_name)
    return numpy_to_dataframe(X_num, X_cat, y, info, task_type)


def check_baseline_available(exp_dir: str, ds_name: str, method_dir: str) -> bool:
    return os.path.exists(os.path.join(exp_dir, ds_name, method_dir, 'y_train.npy'))


def load_catboost_params(tuned_models_dir: str, ds_name: str, task_type: str) -> dict:
    search_paths = [
        os.path.join(tuned_models_dir, 'catboost', f'{ds_name}_cv.json'),
        os.path.join(tuned_models_dir, ds_name, 'catboost', 'best_params.json'),
        os.path.join(tuned_models_dir, ds_name, 'catboost_config.json'),
    ]

    for p in search_paths:
        if os.path.exists(p):
            with open(p, 'r') as f:
                params = json.load(f)
            params.pop('cat_features', None)
            params.pop('task_type', None)
            params.pop('thread_count', None)
            return params

    warnings.warn(
        f"No tuned CatBoost params found for {ds_name}, using defaults"
    )
    return {
        'iterations': 2000,
        'learning_rate': 0.05,
        'depth': 6,
        'l2_leaf_reg': 3.0,
    }


def prepare_Xy_pair(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    task_type: str,
    real_train_df: Optional[pd.DataFrame] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target_col = 'target'
    feature_cols = [c for c in train_df.columns if c != target_col]

    num_cols = [c for c in feature_cols
                if train_df[c].dtype in (np.float64, np.float32, np.int64, np.int32, float, int)]
    cat_cols = [c for c in feature_cols if c not in num_cols]

    fit_df = real_train_df if real_train_df is not None else train_df

    parts_train = []
    parts_test = []

    if num_cols:
        parts_train.append(train_df[num_cols].values.astype(np.float64))
        parts_test.append(test_df[num_cols].values.astype(np.float64))

    if cat_cols:
        enc = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
        enc.fit(fit_df[cat_cols].astype(str))
        parts_train.append(enc.transform(train_df[cat_cols].astype(str)))
        parts_test.append(enc.transform(test_df[cat_cols].astype(str)))

    X_train = np.column_stack(parts_train) if parts_train else np.empty((len(train_df), 0))
    X_test = np.column_stack(parts_test) if parts_test else np.empty((len(test_df), 0))

    le = LabelEncoder()
    if task_type in ('binclass', 'multiclass'):
        le.fit(list(train_df[target_col].astype(str)) + list(test_df[target_col].astype(str)))
        y_train = le.transform(train_df[target_col].astype(str))
        y_test = le.transform(test_df[target_col].astype(str))
    else:
        y_train = train_df[target_col].values.astype(np.float64)
        y_test = test_df[target_col].values.astype(np.float64)

    return X_train, y_train, X_test, y_test


def eval_protocol1(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    task_type: str, seed: int = 42
) -> float:
    scores = []

    if task_type in ('binclass', 'multiclass'):
        models = [
            DecisionTreeClassifier(max_depth=28, random_state=seed),
            RandomForestClassifier(max_depth=28, random_state=seed),
            LogisticRegression(max_iter=500, random_state=seed),
            MLPClassifier(max_iter=100, random_state=seed),
        ]
        avg = 'macro'
        X_train_clean = np.nan_to_num(X_train, nan=0.0)
        X_test_clean = np.nan_to_num(X_test, nan=0.0)
        for m in models:
            try:
                m.fit(X_train_clean, y_train)
                y_pred = m.predict(X_test_clean)
                scores.append(f1_score(y_test, y_pred, average=avg, zero_division=0))
            except Exception as e:
                warnings.warn(f"Protocol 1 model {type(m).__name__} failed: {e}")
                scores.append(0.0)
    else:
        models = [
            DecisionTreeRegressor(max_depth=28, random_state=seed),
            RandomForestRegressor(max_depth=28, random_state=seed),
            Ridge(max_iter=500),
            MLPRegressor(max_iter=100, random_state=seed),
        ]
        X_train_clean = np.nan_to_num(X_train, nan=0.0)
        X_test_clean = np.nan_to_num(X_test, nan=0.0)
        for m in models:
            try:
                m.fit(X_train_clean, y_train)
                y_pred = m.predict(X_test_clean)
                scores.append(r2_score(y_test, y_pred))
            except Exception as e:
                warnings.warn(f"Protocol 1 model {type(m).__name__} failed: {e}")
                scores.append(0.0)

    return float(np.mean(scores))


def eval_protocol2_catboost(
    synth_df: pd.DataFrame, test_df: pd.DataFrame,
    task_type: str, cb_params: dict, seed: int = 42
) -> float:
    if not CATBOOST_AVAILABLE:
        return float('nan')

    target_col = 'target'
    feature_cols = [c for c in synth_df.columns if c != target_col]

    cat_indices = [i for i, c in enumerate(feature_cols)
                   if hasattr(synth_df[c], 'cat') or synth_df[c].dtype == object]

    X_synth = synth_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()

    for i in cat_indices:
        col = feature_cols[i]
        X_synth[col] = X_synth[col].astype(str)
        X_test[col] = X_test[col].astype(str)

    if task_type in ('binclass', 'multiclass'):
        le = LabelEncoder()
        all_y = list(synth_df[target_col].astype(str)) + list(test_df[target_col].astype(str))
        le.fit(all_y)
        y_synth = le.transform(synth_df[target_col].astype(str))
        y_test = le.transform(test_df[target_col].astype(str))
    else:
        y_synth = synth_df[target_col].values.astype(np.float64)
        y_test = test_df[target_col].values.astype(np.float64)

    params = deepcopy(cb_params)
    params['random_seed'] = seed
    params['verbose'] = 0
    if cat_indices:
        params['cat_features'] = cat_indices

    try:
        if task_type in ('binclass', 'multiclass'):
            model = CatBoostClassifier(**params)
            model.fit(X_synth, y_synth)
            y_pred = model.predict(X_test).flatten()
            try:
                y_pred = y_pred.astype(int)
            except (ValueError, TypeError):
                y_pred = le.transform(y_pred.astype(str))
            return f1_score(y_test, y_pred, average='macro', zero_division=0)
        else:
            model = CatBoostRegressor(**params)
            model.fit(X_synth, y_synth)
            y_pred = model.predict(X_test)
            return r2_score(y_test, y_pred)
    except Exception as e:
        warnings.warn(f"CatBoost eval failed: {e}")
        return float('nan')


def jensen_shannon_categorical(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> float:
    cat_cols = [c for c in real_df.columns
                if hasattr(real_df[c], 'cat') and c != 'target']
    if not cat_cols:
        return float('nan')
    divergences = []
    for col in cat_cols:
        real_counts = real_df[col].value_counts(normalize=True)
        synth_counts = synth_df[col].value_counts(normalize=True)
        all_cats = sorted(set(real_counts.index) | set(synth_counts.index))
        p = np.array([real_counts.get(c, 0) for c in all_cats], dtype=float)
        q = np.array([synth_counts.get(c, 0) for c in all_cats], dtype=float)
        eps = 1e-12
        p, q = p + eps, q + eps
        p, q = p / p.sum(), q / q.sum()
        m = 0.5 * (p + q)
        jsd = 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))
        divergences.append(jsd)
    return float(np.mean(divergences)) if divergences else float('nan')


def correlation_matrix_l2(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> float:
    num_cols = [c for c in real_df.columns
                if real_df[c].dtype in (np.float64, np.float32, np.int64, np.int32, float, int)
                and c != 'target']
    if len(num_cols) < 2:
        return float('nan')
    corr_real = np.nan_to_num(real_df[num_cols].corr().values, nan=0.0)
    corr_synth = np.nan_to_num(synth_df[num_cols].corr().values, nan=0.0)
    return float(np.linalg.norm(corr_real - corr_synth, ord='fro'))


def evaluate_synthetic(
    synth_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    task_type: str,
    cb_params: dict,
    n_eval_seeds: int = 10,
) -> dict:
    wd = standardized_wasserstein(train_df, synth_df)
    js = jensen_shannon_categorical(train_df, synth_df)
    corr_l2 = correlation_matrix_l2(train_df, synth_df)
    assoc_l2 = association_l2(train_df, synth_df)
    dcr = dcr_stats(train_df, synth_df, test_df)
    pmse = pmse_ratio(train_df, synth_df, test_df)
    infer = inference_fidelity(train_df, synth_df, task_type)
    aba = alpha_beta_authenticity(train_df, synth_df)

    X_synth, y_synth, X_test, y_test = prepare_Xy_pair(
        synth_df, test_df, task_type, real_train_df=train_df
    )

    p1_scores = []
    p2_scores = []
    for eval_seed in range(n_eval_seeds):
        p1_scores.append(eval_protocol1(
            X_synth, y_synth, X_test, y_test, task_type, seed=eval_seed
        ))
        p2_scores.append(eval_protocol2_catboost(
            synth_df, test_df, task_type, cb_params, seed=eval_seed
        ))

    return {
        'p1': float(np.mean(p1_scores)),
        'p2': float(np.nanmean(p2_scores)),
        'wd': wd, 'js': js, 'corr_l2': corr_l2, 'assoc_l2': assoc_l2,
        'dcr': dcr['dcr'], 'dcr_ratio': dcr['dcr_ratio'],
        'dcr_p5_ratio': dcr['dcr_p5_ratio'],
        'pmse_ratio': pmse,
        'ci_overlap': infer['ci_overlap'], 'sig_agree': infer['sig_agree'],
        'coverage': infer['coverage'],
        'alpha_precision': aba['alpha_precision'],
        'beta_recall': aba['beta_recall'],
        'authenticity': aba['authenticity'],
    }


METRICS_KEYS = ['p1', 'p2', 'wd', 'js', 'corr_l2', 'assoc_l2',
                'dcr', 'dcr_ratio', 'dcr_p5_ratio', 'pmse_ratio',
                'ci_overlap', 'sig_agree', 'coverage',
                'alpha_precision', 'beta_recall', 'authenticity']


def run_single_method_task(task_args: dict) -> Tuple[str, str, dict]:
    if _threadpool_limits is not None:
        _threadpool_limits(1)

    ds_name = task_args['ds_name']
    task_type = task_args['task_type']
    method = task_args['method']
    data_dir = task_args['data_dir']
    exp_dir = task_args['exp_dir']
    tuned_models_dir = task_args['tuned_models_dir']
    n_gen_seeds = task_args['n_gen_seeds']
    n_eval_seeds = task_args['n_eval_seeds']

    metrics_keys = METRICS_KEYS
    result: Dict[str, Any] = {}

    try:
        train_df, _val_df, test_df = load_dataset_as_dataframes(
            data_dir, ds_name, task_type
        )
    except FileNotFoundError as e:
        warnings.warn(f"[{ds_name}/{method}] data not found: {e}")
        return ds_name, method, result

    cb_params = load_catboost_params(tuned_models_dir, ds_name, task_type)
    cb_params['thread_count'] = 1

    t0 = time.time()

    if method == 'real':
        r = evaluate_synthetic(
            train_df, train_df, test_df, task_type, cb_params, n_eval_seeds,
        )
        result['real_p1_mean'] = r['p1']
        result['real_p2_mean'] = r['p2']
        result['real_dcr_mean'] = r['dcr']
        print(f"  [{ds_name}/real] {time.time() - t0:5.1f}s  "
              f"P1={r['p1']:.3f}  P2={r['p2']:.3f}  DCR={r['dcr']:.3f}",
              flush=True)
        return ds_name, method, result

    if method in BASELINE_METHODS:
        method_dir = BASELINE_METHODS[method]
        config_path = os.path.join(exp_dir, ds_name, method_dir, 'config.toml')
        if not os.path.exists(config_path):
            result[f'{method}_p2_mean'] = float('nan')
            result[f'{method}_p2_std'] = 0.0
            print(f"  [{ds_name}/{method}] SKIPPED (no config.toml)", flush=True)
            return ds_name, method, result

        gen_results = []
        for gen_seed in range(n_gen_seeds):
            synth_df = resample_baseline(
                exp_dir, ds_name, method, gen_seed, data_dir, task_type,
                n_train=len(train_df),
            )
            if synth_df is None:
                continue
            gen_results.append(evaluate_synthetic(
                synth_df, train_df, test_df, task_type, cb_params, n_eval_seeds,
            ))

        if gen_results:
            for metric in metrics_keys:
                vals = [r[metric] for r in gen_results]
                result[f'{method}_{metric}'] = float(np.mean(vals))
                result[f'{method}_{metric}_mean'] = float(np.mean(vals))
                result[f'{method}_{metric}_std'] = float(np.std(vals))
            print(f"  [{ds_name}/{method}] {time.time() - t0:5.1f}s  "
                  f"P1={result[f'{method}_p1_mean']:.3f}±{result[f'{method}_p1_std']:.3f}  "
                  f"P2={result[f'{method}_p2_mean']:.3f}±{result[f'{method}_p2_std']:.3f}",
                  flush=True)
        else:
            result[f'{method}_p2_mean'] = float('nan')
            result[f'{method}_p2_std'] = 0.0
            print(f"  [{ds_name}/{method}] ALL SEEDS FAILED", flush=True)
        return ds_name, method, result

    if method == 'beta_copula':
        def gen_fn(seed):
            return generate_beta_copula(train_df, len(train_df), seed, task_type)
    elif method == 'gaussian_copula':
        def gen_fn(seed):
            return generate_gaussian_copula(train_df, len(train_df), seed, task_type)
    elif method == 'checkerboard_copula':
        def gen_fn(seed):
            return generate_checkerboard_copula(train_df, len(train_df), seed, task_type)
    elif method == 'vine_copula':
        if not VINE_AVAILABLE:
            warnings.warn(f"[{ds_name}/{method}] pyvinecopulib not installed, skipping")
            return ds_name, method, result
        try:
            vine_fit = fit_vine_copula(train_df, seed=0)
        except Exception as e:
            warnings.warn(f"[{ds_name}/{method}] vine fit failed: {e}")
            return ds_name, method, result

        def gen_fn(seed, _fit=vine_fit):
            return sample_vine_copula(_fit, len(train_df), seed)
    else:
        warnings.warn(f"Unknown method: {method}")
        return ds_name, method, result

    gen_results = []
    for gen_seed in range(n_gen_seeds):
        try:
            synth_df = gen_fn(gen_seed)
        except Exception as e:
            warnings.warn(f"[{ds_name}/{method}] gen_seed {gen_seed} failed: {e}")
            continue
        gen_results.append(evaluate_synthetic(
            synth_df, train_df, test_df, task_type, cb_params, n_eval_seeds,
        ))

    if gen_results:
        for metric in metrics_keys:
            vals = [r[metric] for r in gen_results]
            result[f'{method}_{metric}'] = float(np.mean(vals))
            result[f'{method}_{metric}_mean'] = float(np.mean(vals))
            result[f'{method}_{metric}_std'] = float(np.std(vals))
        print(f"  [{ds_name}/{method}] {time.time() - t0:5.1f}s  "
              f"P1={result[f'{method}_p1_mean']:.3f}±{result[f'{method}_p1_std']:.3f}  "
              f"P2={result[f'{method}_p2_mean']:.3f}±{result[f'{method}_p2_std']:.3f}",
              flush=True)
    else:
        print(f"  [{ds_name}/{method}] ALL SEEDS FAILED", flush=True)

    return ds_name, method, result


def _method_done(method: str, ds_results: dict) -> bool:
    sentinel = 'real_p2_mean' if method == 'real' else f'{method}_p2_mean'
    return sentinel in ds_results


def run_benchmark(
    data_dir: str,
    exp_dir: str,
    output_dir: str,
    tuned_models_dir: str,
    datasets: Optional[List[str]] = None,
    n_gen_seeds: int = 5,
    n_eval_seeds: int = 10,
    run_beta_copula: bool = True,
    run_gaussian_copula: bool = True,
    run_checkerboard_copula: bool = True,
    run_vine_copula: bool = True,
    run_baselines: bool = True,
    baseline_methods: Optional[List[str]] = None,
    n_workers: int = 1,
    resume: bool = True,
    dry_run: bool = False,
):
    os.makedirs(output_dir, exist_ok=True)

    if datasets is None:
        datasets = list(DATASETS.keys())
    if baseline_methods is None:
        baseline_methods = list(BASELINE_METHODS.keys())

    snapshot_path = os.path.join(output_dir, 'benchmark_results.json')
    existing_results: Dict[str, dict] = {}
    if resume and os.path.exists(snapshot_path):
        try:
            with open(snapshot_path, 'r') as f:
                existing_results = json.load(f)
            print(f"Resuming from {snapshot_path} "
                  f"({len(existing_results)} dataset(s) already on disk)")
        except (json.JSONDecodeError, IOError) as e:
            warnings.warn(
                f"Could not parse existing snapshot {snapshot_path}: {e}. "
                f"Starting fresh."
            )
            existing_results = {}

    expected: Dict[str, set] = {}
    methods_per_ds: Dict[str, List[str]] = {}
    for ds_name in datasets:
        ds_info = DATASETS.get(ds_name)
        if ds_info is None:
            print(f"Unknown dataset: {ds_name}, skipping")
            continue
        methods: List[str] = []
        if run_baselines:
            methods.extend(baseline_methods)
        if run_beta_copula:
            methods.append('beta_copula')
        if run_gaussian_copula:
            methods.append('gaussian_copula')
        if run_checkerboard_copula:
            methods.append('checkerboard_copula')
        if run_vine_copula:
            methods.append('vine_copula')
        methods.append('real')
        expected[ds_name] = set(methods)
        methods_per_ds[ds_name] = methods

    tasks: List[dict] = []
    completed: Dict[str, set] = {ds: set() for ds in expected}
    n_skipped = 0
    skip_summary: Dict[str, int] = {}
    for ds_name, methods in methods_per_ds.items():
        task_type = DATASETS[ds_name]['task']
        ds_existing = existing_results.get(ds_name, {})
        for method in methods:
            if _method_done(method, ds_existing):
                completed[ds_name].add(method)
                n_skipped += 1
                skip_summary[ds_name] = skip_summary.get(ds_name, 0) + 1
                continue
            tasks.append({
                'ds_name': ds_name,
                'task_type': task_type,
                'method': method,
                'data_dir': data_dir,
                'exp_dir': exp_dir,
                'tuned_models_dir': tuned_models_dir,
                'n_gen_seeds': n_gen_seeds,
                'n_eval_seeds': n_eval_seeds,
            })

    n_workers = max(1, int(n_workers))
    print(f"\n{'='*70}")
    if n_skipped:
        fully_done = [ds for ds, n in skip_summary.items()
                      if n == len(expected[ds])]
        if fully_done:
            print(f"Fully cached datasets (skipping entirely): "
                  f"{', '.join(sorted(fully_done))}")
        partials = [(ds, n) for ds, n in skip_summary.items()
                    if ds not in fully_done]
        for ds, n in sorted(partials):
            print(f"Partially cached: {ds} ({n}/{len(expected[ds])} methods)")
    print(f"Running {len(tasks)} (dataset, method) tasks on {n_workers} "
          f"worker{'s' if n_workers != 1 else ''} "
          f"({n_skipped} skipped from snapshot)")
    print(f"{'='*70}")

    if dry_run:
        print("\nDRY RUN — tasks that would be executed, in order:")
        if not tasks:
            print("  (nothing to do: every requested pair is in the snapshot)")
        for t in tasks:
            print(f"  {t['ds_name']:13s} {t['method']}")
        print("\nNothing was written.")
        return existing_results

    all_results: Dict[str, dict] = {
        ds: dict(res) for ds, res in existing_results.items()
    }
    for ds_name in expected:
        if ds_name in existing_results:
            all_results[ds_name] = dict(existing_results[ds_name])
        else:
            all_results[ds_name] = {
                'dataset': ds_name,
                'task': DATASETS[ds_name]['task'],
            }
        all_results[ds_name].setdefault('dataset', ds_name)
        all_results[ds_name].setdefault('task', DATASETS[ds_name]['task'])

    def _record(ds_name: str, method: str, partial: dict):
        all_results[ds_name].update(partial)
        completed[ds_name].add(method)

        done, total = len(completed[ds_name]), len(expected[ds_name])
        print(f"    progress: {ds_name} {done}/{total} methods done",
              flush=True)

        with open(snapshot_path, 'w') as f:
            json.dump(all_results, f, indent=2)

        if completed[ds_name] == expected[ds_name]:
            print(f"\n=== {ds_name} complete — snapshotting CSVs ===\n",
                  flush=True)
            write_metric_csvs(all_results, output_dir)

    if n_workers == 1:
        for t in tasks:
            try:
                ds_name, method, partial = run_single_method_task(t)
            except Exception as e:
                warnings.warn(f"Task {t['ds_name']}/{t['method']} crashed: {e}")
                continue
            _record(ds_name, method, partial)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_task = {
                executor.submit(run_single_method_task, t): t for t in tasks
            }
            for fut in as_completed(future_to_task):
                t = future_to_task[fut]
                try:
                    ds_name, method, partial = fut.result()
                except Exception as e:
                    warnings.warn(
                        f"Task {t['ds_name']}/{t['method']} crashed: {e}"
                    )
                    completed[t['ds_name']].add(t['method'])
                    continue
                _record(ds_name, method, partial)

    with open(snapshot_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    write_metric_csvs(all_results, output_dir)
    print_summary_table(all_results)
    return all_results


ALL_METHODS = [
    'real',
    'tabddpm', 'ctabgan', 'ctabgan_plus', 'tvae', 'beta_copula',
    'gaussian_copula', 'checkerboard_copula', 'vine_copula',
]

METHOD_LABELS = {
    'real':              'Real',
    'tabddpm':           'TabDDPM',
    'ctabgan':           'CTABGAN',
    'ctabgan_plus':      'CTABGAN+',
    'tvae':              'TVAE',
    'beta_copula':       'BetaCopula',
    'gaussian_copula':   'GaussCopula',
    'checkerboard_copula': 'CheckerboardCopula',
    'vine_copula':       'VineCopula',
}

METRIC_FILES = [
    ('p1',      'p1_table.csv',       True,  'Protocol 1 (weak models, macro F1 / R2)'),
    ('p2',      'p2_table.csv',       True,  'Protocol 2 (tuned CatBoost, macro F1 / R2)'),
    ('dcr',     'dcr_table.csv',      True,  'Privacy — mean Distance to Closest Record, z-scored numerics (higher = more private)'),
    ('dcr_ratio', 'dcr_ratio_table.csv', True, 'Privacy — mean DCR / mean DCR of real test rows (~1 = like fresh real data)'),
    ('dcr_p5_ratio', 'dcr_p5_ratio_table.csv', True, 'Privacy — 5th-percentile DCR / same for real test rows'),
    ('wd',      'wd_table.csv',       False, 'Statistical fidelity — Wasserstein distance (numerical, z-scored)'),
    ('js',      'js_table.csv',       False, 'Statistical fidelity — Jensen-Shannon divergence (categorical)'),
    ('corr_l2', 'corr_l2_table.csv',  False, 'Statistical fidelity — Correlation matrix L2 distance (numeric Pearson)'),
    ('assoc_l2', 'assoc_l2_table.csv', False, 'Statistical fidelity — mixed-type association matrix L2 (Pearson / Cramer V / correlation ratio)'),
    ('pmse_ratio', 'pmse_ratio_table.csv', False, 'General utility — pMSE vs train relative to real test rows (Snoke et al. 2018; ideal = 1, <1 memorisation, >1 shift)'),
    ('ci_overlap', 'ci_overlap_table.csv', True, 'Inference fidelity — analyst-GLM 95% CI overlap (Karr et al. 2006)'),
    ('sig_agree', 'sig_agree_table.csv', True, 'Inference fidelity — sign-and-significance agreement rate of analyst-GLM coefficients'),
    ('coverage', 'coverage_table.csv', True, 'Inference fidelity — fraction of real coefficients inside the synthetic 95% CI'),
    ('alpha_precision', 'alpha_precision_table.csv', True, 'Faithfulness — alpha-precision (Alaa et al. 2022, k-NN support; 1 = on the diagonal)'),
    ('beta_recall', 'beta_recall_table.csv', True, 'Faithfulness — beta-recall / coverage of the real support (Alaa et al. 2022; 1 = on the diagonal)'),
    ('authenticity', 'authenticity_table.csv', True, 'Faithfulness — share of synthetic rows that are not near-copies of a training row (Alaa et al. 2022)'),
]


def _format_cell(mean_val: float, std_val: float) -> str:
    if mean_val is None or (isinstance(mean_val, float) and np.isnan(mean_val)):
        return '---'
    if std_val is None or (isinstance(std_val, float) and np.isnan(std_val)):
        std_val = 0.0
    if std_val > 1e-4:
        return f'{mean_val:.3f}±{std_val:.3f}'
    return f'{mean_val:.3f}'


def write_metric_csvs(results: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    if not results:
        return

    for metric_key, filename, higher_is_better, description in METRIC_FILES:
        rows = []
        for ds_name, res in results.items():
            row: Dict[str, Any] = {'dataset': ds_name}
            for m in ALL_METHODS:
                if m == 'real':
                    mean_val = res.get(f'real_{metric_key}_mean', float('nan'))
                    row[METHOD_LABELS[m]] = _format_cell(mean_val, 0.0)
                else:
                    mean_val = res.get(f'{m}_{metric_key}_mean', float('nan'))
                    std_val = res.get(f'{m}_{metric_key}_std', 0.0)
                    row[METHOD_LABELS[m]] = _format_cell(mean_val, std_val)
            rows.append(row)

        df = pd.DataFrame(rows).set_index('dataset')
        path = os.path.join(output_dir, filename)
        direction = '↑ higher is better' if higher_is_better else '↓ lower is better'
        with open(path, 'w') as f:
            f.write(f'# {description} ({direction})\n')
            df.to_csv(f)
        print(f'  wrote {path}')

    write_aggregate_table(results, output_dir)


def write_aggregate_table(results: dict, output_dir: str):
    if not results:
        return

    metric_specs = [
        ('corr_l2', 'CorrL2'),
        ('p1', 'P1'),
        ('p2', 'P2'),
    ]

    rows = []
    for m in ALL_METHODS:
        row: Dict[str, Any] = {'method': METHOD_LABELS[m]}
        for metric_key, label in metric_specs:
            key = (f'real_{metric_key}_mean' if m == 'real'
                   else f'{m}_{metric_key}_mean')
            vals = []
            for res in results.values():
                v = res.get(key, None)
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    vals.append(float(v))
            if vals:
                row[f'{label}_mean'] = round(float(np.mean(vals)), 4)
                row[f'{label}_std'] = round(float(np.std(vals)), 4)
                row[f'{label}_n'] = len(vals)
            else:
                row[f'{label}_mean'] = float('nan')
                row[f'{label}_std'] = float('nan')
                row[f'{label}_n'] = 0
        rows.append(row)

    df = pd.DataFrame(rows).set_index('method')
    path = os.path.join(output_dir, 'aggregate_table.csv')
    with open(path, 'w') as f:
        f.write('# Aggregated across datasets (mean/std over datasets, NaNs '
                'excluded). CorrL2 = correlation-matrix L2 distance '
                '(lower is better); P1/P2 = ML efficiency (higher is better).\n')
        df.to_csv(f)
    print(f'  wrote {path}')


def print_summary_table(results: dict):
    for protocol, label in [
        ('p2', 'Protocol 2 (Tuned CatBoost)'),
        ('p1', 'Protocol 1 (Weak Models)'),
        ('dcr', 'Privacy — Distance to Closest Record (↑ = more private)'),
    ]:
        col_w = 20
        width = 12 + col_w * len(ALL_METHODS)
        print(f"\n{'='*width}")
        if protocol == 'dcr':
            print(f"SUMMARY: {label}")
        else:
            print(f"SUMMARY: ML Efficiency — {label} (Macro F1 / R2)")
        print(f"{'='*width}")

        header_parts = [f"{'Dataset':<12}"]
        for m in ALL_METHODS:
            header_parts.append(f"{METHOD_LABELS[m]:>{col_w}}")
        print(' '.join(header_parts))
        print('-' * width)

        for ds_name, res in results.items():
            row_parts = [f"{ds_name:<12}"]
            for m in ALL_METHODS:
                if m == 'real':
                    val = res.get(f'real_{protocol}_mean', float('nan'))
                    if np.isnan(val):
                        row_parts.append(f"{'---':>{col_w}}")
                    else:
                        row_parts.append(f"{val:>{col_w}.3f}")
                else:
                    mean_val = res.get(f'{m}_{protocol}_mean', float('nan'))
                    std_val = res.get(f'{m}_{protocol}_std', 0.0)
                    if np.isnan(mean_val):
                        row_parts.append(f"{'---':>{col_w}}")
                    elif std_val > 0.0001:
                        cell = f"{mean_val:.3f}±{std_val:.3f}"
                        row_parts.append(f"{cell:>{col_w}}")
                    else:
                        row_parts.append(f"{mean_val:>{col_w}.3f}")
            print(' '.join(row_parts))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Benchmark of copula generators against deep tabular generators"
    )
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--exp_dir', type=str, default='exp/',
                        help='Path to exp/ directory with baseline synthetic data')
    parser.add_argument('--output_dir', type=str, default='results/benchmarks/',
                        help='Where the CSV tables and JSON dump are written')
    parser.add_argument('--tuned_models_dir', type=str, default='tuned_models/')
    parser.add_argument('--datasets', nargs='+', default=None)
    parser.add_argument('--n_gen_seeds', type=int, default=5)
    parser.add_argument('--n_eval_seeds', type=int, default=10)
    parser.add_argument('--no_beta_copula', action='store_true',
                        help='Skip the standalone BetaCopula baseline')
    parser.add_argument('--no_gaussian_copula', action='store_true',
                        help='Skip the standalone Gaussian copula baseline')
    parser.add_argument('--no_checkerboard_copula', action='store_true',
                        help='Skip the checkerboard (unsmoothed beta) copula ablation')
    parser.add_argument('--no_vine_copula', action='store_true',
                        help='Skip the vine copula baseline (needs pyvinecopulib)')
    parser.add_argument('--no_baselines', action='store_true',
                        help='Skip all external baselines (TabDDPM, CTABGAN, '
                             'CTABGAN+, TVAE)')
    parser.add_argument('--baselines', nargs='+', default=None,
                        help='Specific baselines (e.g. tabddpm tvae). '
                             'Default: all 4.')
    parser.add_argument(
        '--n_workers', type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help='Number of parallel workers for (dataset, method) tasks. '
             'Default: os.cpu_count() - 1. Use 1 to disable parallelism '
             '(handy for debugging).',
    )
    parser.add_argument(
        '--no_resume', action='store_true',
        help='Force a full rerun even if benchmark_results.json exists in '
             '--output_dir. Without this flag, any (dataset, method) pair '
             'already present in the snapshot is skipped.',
    )
    parser.add_argument(
        '--dry_run', action='store_true',
        help='Print the (dataset, method) plan — cached vs. to-run — and '
             'exit without computing or writing anything.',
    )

    args = parser.parse_args()

    run_benchmark(
        data_dir=args.data_dir,
        exp_dir=args.exp_dir,
        output_dir=args.output_dir,
        tuned_models_dir=args.tuned_models_dir,
        datasets=args.datasets,
        n_gen_seeds=args.n_gen_seeds,
        n_eval_seeds=args.n_eval_seeds,
        run_beta_copula=not args.no_beta_copula,
        run_gaussian_copula=not args.no_gaussian_copula,
        run_checkerboard_copula=not args.no_checkerboard_copula,
        run_vine_copula=not args.no_vine_copula,
        run_baselines=not args.no_baselines,
        baseline_methods=args.baselines,
        n_workers=args.n_workers,
        resume=not args.no_resume,
        dry_run=args.dry_run,
    )
