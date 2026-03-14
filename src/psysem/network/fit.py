from __future__ import annotations

import numpy as np
import pandas as pd

from ..preprocessing import AssociationMatrixConfig, build_association_matrix
from .contracts import NetworkConfig, NetworkResult
from .metrics import build_edge_table, build_node_table

_SUPPORTED_NETWORK_ESTIMATORS = frozenset({"ggm"})


def fit_network(data: pd.DataFrame, config: NetworkConfig) -> NetworkResult:
    """Fit a static undirected item network via Gaussian graphical modeling."""
    _validate_inputs(data, config)
    prepared = build_association_matrix(
        data,
        AssociationMatrixConfig(
            items=tuple(config.items),
            missing_strategy=config.missing_strategy,
            correlation_method=config.correlation_method,
            variable_types=config.variable_types,
            stabilize=True,
            min_eigenvalue=1e-8,
            include_pairwise_counts=True,
        ),
    )

    association = prepared.matrix.to_numpy(dtype=float)
    precision, inversion_method, estimation_warnings = _estimate_precision_matrix(
        association,
        ridge=float(config.ridge),
    )
    partial = _precision_to_partial_correlation(precision)
    adjacency = _threshold_partial_correlation(partial, min_abs_edge=float(config.min_abs_edge))

    item_names = prepared.item_names
    edge_table = build_edge_table(adjacency, item_names=item_names)
    warnings = list(prepared.warnings)
    warnings.extend(estimation_warnings)
    if config.ridge > 0:
        warnings.append(
            f"Network estimation applied ridge regularization ({config.ridge:.6g}) before matrix inversion."
        )
    if edge_table.empty:
        warnings.append("No edges survived the configured `min_abs_edge` threshold.")

    return NetworkResult(
        item_names=item_names,
        estimator=config.estimator.strip().lower(),
        inversion_method=inversion_method,
        correlation_method=prepared.correlation_method,
        missing_strategy=prepared.missing_strategy,
        resolved_variable_types=prepared.resolved_variable_types,
        association_matrix=prepared.matrix.copy(),
        precision_matrix=pd.DataFrame(precision, index=list(item_names), columns=list(item_names)),
        partial_correlation_matrix=pd.DataFrame(
            partial,
            index=list(item_names),
            columns=list(item_names),
        ),
        adjacency_matrix=pd.DataFrame(
            adjacency,
            index=list(item_names),
            columns=list(item_names),
        ),
        edge_table=edge_table,
        node_table=build_node_table(adjacency, item_names=item_names),
        pairwise_n=prepared.pairwise_n,
        n_complete_rows=prepared.n_complete_rows,
        dropped_rows=prepared.dropped_rows,
        stabilization_applied=prepared.stabilization_applied,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _validate_inputs(data: pd.DataFrame, config: NetworkConfig) -> None:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("`data` must be a pandas.DataFrame.")
    if not config.items:
        raise ValueError("`items` cannot be empty.")
    if len(config.items) < 2:
        raise ValueError("`items` must contain at least 2 variables for network analysis.")
    estimator = config.estimator.strip().lower()
    if estimator not in _SUPPORTED_NETWORK_ESTIMATORS:
        supported = ", ".join(sorted(_SUPPORTED_NETWORK_ESTIMATORS))
        raise ValueError(f"Unsupported network estimator `{estimator}`. Available: {supported}.")
    if config.ridge < 0:
        raise ValueError("`ridge` must be >= 0.")
    if not (0.0 <= config.min_abs_edge < 1.0):
        raise ValueError("`min_abs_edge` must be in [0, 1).")


def _estimate_precision_matrix(
    association: np.ndarray,
    *,
    ridge: float,
) -> tuple[np.ndarray, str, tuple[str, ...]]:
    estimation_matrix = np.asarray(association, dtype=float).copy()
    if ridge > 0:
        estimation_matrix = estimation_matrix + np.eye(estimation_matrix.shape[0], dtype=float) * ridge

    try:
        precision = np.linalg.inv(estimation_matrix)
        method = "inverse"
        warnings: tuple[str, ...] = ()
    except np.linalg.LinAlgError:
        precision = np.linalg.pinv(estimation_matrix, hermitian=True)
        method = "pinv"
        warnings = (
            "Association matrix inversion fell back to pseudo-inverse during network estimation.",
        )

    precision = (precision + precision.T) / 2.0
    return precision, method, warnings


def _precision_to_partial_correlation(precision: np.ndarray) -> np.ndarray:
    diag = np.clip(np.diag(precision), 1e-12, None)
    scale = np.sqrt(np.outer(diag, diag))
    partial = np.divide(
        -precision,
        scale,
        out=np.zeros_like(precision),
        where=scale > 0,
    )
    partial = np.clip(partial, -1.0, 1.0)
    np.fill_diagonal(partial, 0.0)
    return (partial + partial.T) / 2.0


def _threshold_partial_correlation(
    partial: np.ndarray,
    *,
    min_abs_edge: float,
) -> np.ndarray:
    adjacency = np.asarray(partial, dtype=float).copy()
    adjacency[np.abs(adjacency) < min_abs_edge] = 0.0
    np.fill_diagonal(adjacency, 0.0)
    return (adjacency + adjacency.T) / 2.0
