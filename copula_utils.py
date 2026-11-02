import numpy as np
import pandas as pd


def is_categorical(series: pd.Series) -> bool:
    return pd.api.types.is_categorical_dtype(series) or \
           pd.api.types.is_object_dtype(series) or \
           isinstance(series.dtype, pd.CategoricalDtype)


def clean_encoding(Z: pd.DataFrame, original_data: pd.DataFrame) -> pd.DataFrame:
    Z = Z.copy()
    col_names = Z.columns.tolist()

    categorical_vars = [col for col in original_data.columns
                        if is_categorical(original_data[col])]

    dummy_map = {}
    for v in categorical_vars:
        matching_cols = [col for col in col_names if col.startswith(f"{v}_")]
        if matching_cols:
            dummy_map[v] = matching_cols

    for v, cols in dummy_map.items():
        if len(cols) < 2:
            continue

        Z_subset = Z[cols].values

        Z_group = np.zeros_like(Z_subset)
        for row_idx in range(len(Z_subset)):
            row = Z_subset[row_idx]
            row_bin = (row >= 0.5).astype(int)

            if row_bin.sum() != 1:
                row_bin = np.zeros(len(row), dtype=int)
                row_bin[np.argmax(row)] = 1

            Z_group[row_idx] = row_bin

        Z[cols] = Z_group

    return Z


def reconstruct(D_full: pd.DataFrame, original_data: pd.DataFrame) -> pd.DataFrame:
    D_full = D_full.copy()

    D_full = clean_encoding(D_full, original_data)

    col_names = D_full.columns.tolist()
    categorical_vars = [col for col in original_data.columns
                        if is_categorical(original_data[col])]

    recovered_data = {}
    used_cols = set()

    for v in categorical_vars:
        matching_cols = [col for col in col_names if col.startswith(f"{v}_")]

        if not matching_cols:
            continue

        D_var = D_full[matching_cols].values

        level_names = [col.replace(f"{v}_", "") for col in matching_cols]

        if isinstance(original_data[v].dtype, pd.CategoricalDtype):
            orig_levels = original_data[v].cat.categories.tolist()
            is_ordered = original_data[v].cat.ordered
        else:
            orig_levels = sorted(original_data[v].unique().tolist())
            is_ordered = False

        if D_var.shape[1] >= 2:
            chosen_indices = np.argmax(D_var, axis=1)
            picked = [level_names[idx] for idx in chosen_indices]
        else:
            represented = level_names[0]
            baseline = [lv for lv in orig_levels if lv != represented]
            baseline = baseline[0] if baseline else orig_levels[0]
            picked = [represented if val >= 0.5 else baseline for val in D_var[:, 0]]

        if is_ordered:
            recovered_data[v] = pd.Categorical(picked, categories=orig_levels, ordered=True)
        else:
            recovered_data[v] = pd.Categorical(picked, categories=orig_levels, ordered=False)

        used_cols.update(matching_cols)

    continuous_cols = [col for col in col_names if col not in used_cols]
    for col in continuous_cols:
        if col in original_data.columns:
            recovered_data[col] = D_full[col].values

    result = pd.DataFrame({col: recovered_data[col] for col in original_data.columns})

    return result
