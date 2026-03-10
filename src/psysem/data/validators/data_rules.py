from __future__ import annotations

import pandas as pd

from ..contracts import ESEMSpec


def validate_data_reference_columns(
    spec: ESEMSpec,
    data: pd.DataFrame,
    item_set: set[str],
    structural_vars: set[str],
    errors: list[str],
) -> None:
    if not isinstance(data, pd.DataFrame):
        errors.append("`data` must be a pandas.DataFrame.")
        return

    columns = set(data.columns.tolist())
    for item in sorted(item_set):
        if item not in columns:
            errors.append(f"Item column `{item}` is missing from data.")

    for optional_name in [spec.group, spec.weight, spec.cluster, spec.id]:
        if optional_name is not None and optional_name not in columns:
            errors.append(f"Optional column `{optional_name}` is missing from data.")

    for structural_var in sorted(structural_vars):
        if structural_var not in columns:
            errors.append(f"Structural variable `{structural_var}` is missing from data.")


def validate_ordinal_columns(
    spec: ESEMSpec,
    data: pd.DataFrame,
    errors: list[str],
) -> None:
    for var_name, var_type in spec.variable_types.items():
        if var_type != "ordinal" or var_name not in data.columns:
            continue
        series = data[var_name]
        if not _is_ordinal_compatible(series):
            errors.append(
                f"Variable `{var_name}` is marked ordinal but contains non-ordinal values."
            )


def _is_ordinal_compatible(series: pd.Series) -> bool:
    non_null = series.dropna()
    if non_null.empty:
        return True

    if isinstance(non_null.dtype, pd.CategoricalDtype):
        return bool(getattr(non_null.dtype, "ordered", False))

    if pd.api.types.is_integer_dtype(non_null.dtype):
        return True

    if pd.api.types.is_numeric_dtype(non_null.dtype):
        numeric = pd.to_numeric(non_null, errors="coerce")
        if numeric.isna().any():
            return False
        return bool(((numeric % 1).abs() < 1e-8).all())

    return False
