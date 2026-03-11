from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import EFAEvaluationConfig, EFAEvaluationResult
from .fit import EFAResult


def evaluate_efa_model(result: EFAResult, config: EFAEvaluationConfig) -> EFAEvaluationResult:
    """Score one EFA solution for automated factor-count comparison."""
    _validate_evaluation_config(config)
    loadings = np.abs(result.loadings.to_numpy(dtype=float))
    n_items, _ = loadings.shape

    salient_by_item = np.max(loadings, axis=1)
    salient_items = int(np.sum(salient_by_item >= config.salient_loading))
    cross_loaded_items = int(np.sum(np.sum(loadings >= config.cross_loading, axis=1) >= 2))
    communalities = result.communalities.to_numpy(dtype=float)
    low_h2_items = int(np.sum(communalities < config.min_h2))

    explained_total = float(np.sum(result.explained_variance.to_numpy(dtype=float)) / max(n_items, 1))
    simple_structure_ratio = float(
        np.mean(np.sum(loadings >= config.cross_loading, axis=1) <= 1)
    )
    mean_h2 = float(np.mean(communalities))
    mean_max_loading = float(np.mean(salient_by_item))

    salient_per_factor = np.sum(loadings >= config.salient_loading, axis=0).astype(float)
    factor_balance = 0.0
    warnings: list[str] = []
    if np.any(salient_per_factor == 0):
        warnings.append("At least one factor has no salient items.")
        factor_balance = 1.0
    elif float(np.mean(salient_per_factor)) > 0:
        factor_balance = float(np.std(salient_per_factor) / np.mean(salient_per_factor))

    cross_ratio = float(cross_loaded_items / max(n_items, 1))
    score = (
        config.variance_weight * explained_total
        + config.simplicity_weight * simple_structure_ratio
        + config.communality_weight * mean_h2
        + 0.25 * mean_max_loading
        - config.cross_loading_penalty * cross_ratio
        - config.factor_balance_penalty * factor_balance
    )
    return EFAEvaluationResult(
        n_factors=int(result.loadings.shape[1]),
        score=float(score),
        explained_total=explained_total,
        simple_structure_ratio=simple_structure_ratio,
        mean_h2=mean_h2,
        mean_max_loading=mean_max_loading,
        cross_loaded_items=cross_loaded_items,
        low_h2_items=low_h2_items,
        salient_items=salient_items,
        warnings=tuple(warnings),
    )


def evaluation_to_series(result: EFAEvaluationResult) -> pd.Series:
    """Convert evaluation result to one row for comparison tables."""
    return pd.Series(
        {
            "n_factors": result.n_factors,
            "score": result.score,
            "explained_total": result.explained_total,
            "simple_structure_ratio": result.simple_structure_ratio,
            "mean_h2": result.mean_h2,
            "mean_max_loading": result.mean_max_loading,
            "cross_loaded_items": result.cross_loaded_items,
            "low_h2_items": result.low_h2_items,
            "salient_items": result.salient_items,
            "warnings": "; ".join(result.warnings),
        }
    )


def _validate_evaluation_config(config: EFAEvaluationConfig) -> None:
    if config.salient_loading <= 0:
        raise ValueError("`salient_loading` must be > 0.")
    if config.cross_loading <= 0:
        raise ValueError("`cross_loading` must be > 0.")
    if config.cross_loading < config.salient_loading:
        raise ValueError("`cross_loading` must be >= `salient_loading`.")
    if not (0.0 <= config.min_h2 <= 1.0):
        raise ValueError("`min_h2` must be between 0 and 1.")
