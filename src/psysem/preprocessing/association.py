from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import AssociationMatrixConfig, AssociationMatrixResult
from .polychoric import build_polychoric_matrix
from .stabilization import stabilize_association_matrix
from .variable_types import (
    build_preprocessing_recommendations,
    resolve_variable_types,
)


SUPPORTED_MISSING_STRATEGIES = frozenset({"pairwise", "dropna"})
SUPPORTED_CORRELATION_METHODS = frozenset({"pearson", "spearman", "polychoric"})


def normalize_missing_strategy(strategy: str) -> str:
    return strategy.strip().lower()


def normalize_correlation_method(method: str) -> str:
    return method.strip().lower()


def build_association_matrix(
    data: pd.DataFrame,
    config: AssociationMatrixConfig,
) -> AssociationMatrixResult:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("`data` must be a pandas.DataFrame.")

    item_names = tuple(config.items)
    if not item_names:
        raise ValueError("`items` cannot be empty.")

    missing = [name for name in item_names if name not in data.columns]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Missing item columns: {joined}.")

    missing_strategy = normalize_missing_strategy(config.missing_strategy)
    if missing_strategy not in SUPPORTED_MISSING_STRATEGIES:
        raise ValueError("`missing_strategy` must be one of: pairwise, dropna.")

    correlation_method = normalize_correlation_method(config.correlation_method)
    if correlation_method not in SUPPORTED_CORRELATION_METHODS:
        raise ValueError("`correlation_method` must be one of: pearson, spearman, polychoric.")

    item_frame = data.loc[:, list(item_names)]
    resolved_variable_types = resolve_variable_types(
        item_frame,
        declared_variable_types=config.variable_types,
    )
    _validate_correlation_method_inputs(
        item_names=item_names,
        correlation_method=correlation_method,
        resolved_variable_types=resolved_variable_types,
    )

    corr, pairwise_n, dropped_rows, n_complete_rows, warnings = _compute_association_matrix(
        item_frame,
        missing_strategy=missing_strategy,
        correlation_method=correlation_method,
    )
    recommendations = build_preprocessing_recommendations(
        resolved_variable_types=resolved_variable_types,
        correlation_method=correlation_method,
        declared_variable_types=config.variable_types,
    )
    stabilization_applied = False
    if config.stabilize:
        corr, stabilization_applied = stabilize_association_matrix(
            corr,
            min_eigenvalue=config.min_eigenvalue,
        )
    matrix = pd.DataFrame(corr, index=list(item_names), columns=list(item_names))
    if not config.include_pairwise_counts:
        pairwise_n = None

    return AssociationMatrixResult(
        matrix=matrix,
        item_names=item_names,
        correlation_method=correlation_method,
        missing_strategy=missing_strategy,
        resolved_variable_types=resolved_variable_types,
        pairwise_n=pairwise_n,
        n_complete_rows=n_complete_rows,
        dropped_rows=dropped_rows,
        stabilization_applied=stabilization_applied,
        warnings=tuple(dict.fromkeys((*warnings, *recommendations))),
    )


def _compute_association_matrix(
    item_frame: pd.DataFrame,
    *,
    missing_strategy: str,
    correlation_method: str,
) -> tuple[np.ndarray, pd.DataFrame | None, int, int | None, tuple[str, ...]]:
    if correlation_method == "polychoric":
        return build_polychoric_matrix(item_frame, missing_strategy=missing_strategy)

    item_names = list(item_frame.columns)
    warnings: list[str] = []

    if missing_strategy == "dropna":
        analysis_frame = item_frame.dropna(axis=0, how="any")
        n_complete_rows = int(analysis_frame.shape[0])
        dropped_rows = int(item_frame.shape[0] - analysis_frame.shape[0])
        corr = analysis_frame.corr(method=correlation_method).to_numpy(dtype=float)
        if dropped_rows > 0:
            warnings.append(
                f"Input preprocessing dropped {dropped_rows} row(s) with missing values under dropna strategy."
            )
        pairwise_n = pd.DataFrame(
            np.full((len(item_names), len(item_names)), n_complete_rows, dtype=int),
            index=item_names,
            columns=item_names,
        )
    else:
        pairwise_n = item_frame.notna().astype(int).T @ item_frame.notna().astype(int)
        corr = item_frame.corr(method=correlation_method).to_numpy(dtype=float)
        n_complete_rows = int(item_frame.dropna(axis=0, how="any").shape[0])
        dropped_rows = 0
        unique_counts = np.unique(pairwise_n.to_numpy(dtype=int))
        if unique_counts.size > 1:
            warnings.append(
                "Pairwise missing strategy used variable-specific observation counts when building the correlation matrix."
            )

    if correlation_method == "spearman":
        warnings.append("Input preprocessing used Spearman rank correlation.")

    return corr, pairwise_n, dropped_rows, n_complete_rows, tuple(warnings)


def _validate_correlation_method_inputs(
    *,
    item_names: tuple[str, ...],
    correlation_method: str,
    resolved_variable_types: dict[str, str],
) -> None:
    if correlation_method != "polychoric":
        return

    non_ordinal = [name for name in item_names if resolved_variable_types.get(name) != "ordinal"]
    if non_ordinal:
        joined = ", ".join(non_ordinal)
        raise ValueError(
            "`correlation_method='polychoric'` currently requires all analysis items to resolve to `ordinal`; "
            f"non-ordinal items: {joined}."
        )
