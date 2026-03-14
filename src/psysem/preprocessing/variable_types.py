from __future__ import annotations

import numpy as np
import pandas as pd


SUPPORTED_VARIABLE_TYPES = frozenset({"continuous", "ordinal"})


def validate_declared_variable_types(
    item_names: tuple[str, ...],
    declared_variable_types: dict[str, str] | None,
) -> dict[str, str]:
    if declared_variable_types is None:
        return {}

    invalid_names = [name for name in declared_variable_types if name not in item_names]
    if invalid_names:
        joined = ", ".join(sorted(invalid_names))
        raise ValueError(f"`variable_types` contains items not present in `items`: {joined}.")

    normalized: dict[str, str] = {}
    invalid_type_names: list[str] = []
    for name, kind in declared_variable_types.items():
        if not isinstance(kind, str):
            invalid_type_names.append(str(name))
            continue
        normalized[name] = kind.strip().lower()

    invalid_type_names.extend(
        name for name, kind in normalized.items() if kind not in SUPPORTED_VARIABLE_TYPES
    )
    if invalid_type_names:
        joined = ", ".join(sorted(invalid_type_names))
        raise ValueError(
            f"`variable_types` entries must be `continuous` or `ordinal`; invalid entries for: {joined}."
        )
    return normalized


def resolve_variable_types(
    item_frame: pd.DataFrame,
    *,
    declared_variable_types: dict[str, str] | None,
) -> dict[str, str]:
    resolved: dict[str, str] = {}
    declared = validate_declared_variable_types(
        tuple(item_frame.columns),
        declared_variable_types,
    )

    for column_name in item_frame.columns:
        if column_name in declared:
            resolved[column_name] = declared[column_name]
            continue
        resolved[column_name] = infer_variable_type(item_frame[column_name])

    return resolved


def infer_variable_type(series: pd.Series) -> str:
    values = series.dropna()
    if values.empty:
        return "continuous"

    n_obs = int(values.shape[0])
    unique_count = int(values.nunique(dropna=True))
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all() and 2 <= unique_count <= 8 and unique_count < n_obs:
        rounded = np.round(numeric.to_numpy(dtype=float))
        if np.allclose(numeric.to_numpy(dtype=float), rounded, atol=1e-8):
            return "ordinal"
    return "continuous"


def build_preprocessing_recommendations(
    *,
    resolved_variable_types: dict[str, str],
    correlation_method: str,
    declared_variable_types: dict[str, str] | None,
) -> tuple[str, ...]:
    ordinal_items = [name for name, kind in resolved_variable_types.items() if kind == "ordinal"]
    if not ordinal_items:
        return ()

    source = "declared" if declared_variable_types else "inferred"
    if correlation_method == "pearson":
        return (
            f'{source.capitalize()} ordinal-like items detected ({", ".join(ordinal_items)}); '
            'consider `correlation_method="spearman"` now or `polychoric` for ordinal analysis.',
        )
    if correlation_method == "spearman":
        return (
            f"{source.capitalize()} ordinal-like items detected ({', '.join(ordinal_items)}); "
            "Spearman is a lightweight fallback compared with `polychoric` for ordinal analysis.",
        )
    return ()
