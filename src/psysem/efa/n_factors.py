from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd

from .contracts import FactorSelectionConfig, FactorSelectionResult
from .diagnostics import build_efa_correlation_matrix


def suggest_n_factors(data: pd.DataFrame, config: FactorSelectionConfig) -> FactorSelectionResult:
    """Suggest factor count using multiple EFA heuristics."""
    corr, items, n_obs, warnings = build_efa_correlation_matrix(
        data=data,
        items=config.items,
        dropna=config.dropna,
    )
    n_items = len(items)
    n_min, n_max = _resolve_factor_range(config, n_items=n_items)
    _validate_selection_config(config, n_items=n_items, n_min=n_min, n_max=n_max)

    eigenvalues = _sorted_eigenvalues(corr)
    suggestions: dict[str, int] = {}
    warning_list = list(warnings)

    scree_elbow: int | None = None
    if config.enable_scree:
        scree_elbow = _clip_n_factors(_scree_elbow(eigenvalues), n_min=n_min, n_max=n_max)
        suggestions["scree"] = scree_elbow

    parallel_thresholds: pd.Series | None = None
    if config.enable_pa:
        pa_threshold = _parallel_analysis_thresholds(
            n_obs=n_obs,
            n_items=n_items,
            n_iter=config.pa_iter,
            percentile=config.pa_percentile,
            random_state=config.random_state,
        )
        parallel_thresholds = pd.Series(
            pa_threshold,
            index=_factor_index(n_items),
            name="pa_threshold",
        )
        pa_suggestion = int(np.sum(eigenvalues > pa_threshold))
        suggestions["parallel_analysis"] = _clip_n_factors(pa_suggestion, n_min=n_min, n_max=n_max)

    map_values: pd.Series | None = None
    if config.enable_map:
        map_curve = _map_values(corr, n_max=n_max)
        map_values = pd.Series(
            map_curve,
            index=pd.Index(range(0, n_max + 1), name="n_factors"),
            name="map",
        )
        map_range = map_curve[n_min : n_max + 1]
        suggestions["map"] = int(np.argmin(map_range) + n_min)

    if config.enable_kaiser:
        kaiser_suggestion = int(np.sum(eigenvalues > 1.0))
        suggestions["kaiser"] = _clip_n_factors(kaiser_suggestion, n_min=n_min, n_max=n_max)

    if len(set(suggestions.values())) > 1:
        warning_list.append("Factor-count methods disagree; review per-method suggestions.")

    consensus_n_factors = _aggregate_consensus(
        suggestions=suggestions,
        n_min=n_min,
        n_max=n_max,
        strategy=config.consensus_strategy,
    )
    return FactorSelectionResult(
        items=items,
        n_obs=n_obs,
        n_items=n_items,
        n_min=n_min,
        n_max=n_max,
        eigenvalues=pd.Series(eigenvalues, index=_factor_index(n_items), name="eigenvalue"),
        scree_elbow=scree_elbow,
        parallel_thresholds=parallel_thresholds,
        map_values=map_values,
        suggestions_by_method=suggestions,
        consensus_n_factors=consensus_n_factors,
        warnings=tuple(warning_list),
        correlation_matrix=pd.DataFrame(corr, index=list(items), columns=list(items)),
    )


def _validate_selection_config(
    config: FactorSelectionConfig,
    *,
    n_items: int,
    n_min: int,
    n_max: int,
) -> None:
    if n_items < 2:
        raise ValueError("At least 2 items are required for factor-count suggestion.")
    if n_min < 1:
        raise ValueError("`n_min` must be >= 1.")
    if n_max >= n_items:
        raise ValueError("`n_max` must be smaller than number of items.")
    if config.pa_iter <= 0:
        raise ValueError("`pa_iter` must be > 0.")
    if not (0.0 < config.pa_percentile <= 1.0):
        raise ValueError("`pa_percentile` must be in (0, 1].")

    enabled = [config.enable_pa, config.enable_map, config.enable_kaiser, config.enable_scree]
    if not any(enabled):
        raise ValueError("At least one factor-count method must be enabled.")


def _resolve_factor_range(config: FactorSelectionConfig, *, n_items: int) -> tuple[int, int]:
    n_min = int(config.n_min)
    max_allowed = max(1, n_items - 1)
    n_max = max_allowed if config.n_max is None else int(config.n_max)
    if n_max < n_min:
        raise ValueError("`n_max` must be >= `n_min`.")
    return n_min, n_max


def _sorted_eigenvalues(corr: np.ndarray) -> np.ndarray:
    eigvals = np.linalg.eigvalsh(corr)
    return np.sort(np.clip(eigvals, 0.0, None))[::-1]


def _factor_index(n_items: int) -> pd.Index:
    return pd.Index([f"F{i + 1}" for i in range(n_items)], name="factor")


def _parallel_analysis_thresholds(
    *,
    n_obs: int,
    n_items: int,
    n_iter: int,
    percentile: float,
    random_state: int | None,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    simulated = np.empty((n_iter, n_items), dtype=float)
    for i in range(n_iter):
        random_data = rng.standard_normal(size=(n_obs, n_items))
        corr = np.corrcoef(random_data, rowvar=False)
        corr = _stabilize_correlation(corr)
        simulated[i, :] = _sorted_eigenvalues(corr)
    return np.quantile(simulated, percentile, axis=0)


def _map_values(corr: np.ndarray, *, n_max: int) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(corr)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.clip(eigvals[order], 0.0, None)
    eigvecs = eigvecs[:, order]
    p = corr.shape[0]
    upper = np.triu_indices(p, k=1)
    map_curve = np.zeros(n_max + 1, dtype=float)

    for k in range(0, n_max + 1):
        if k == 0:
            reproduced = np.zeros_like(corr)
        else:
            loadings = eigvecs[:, :k] * np.sqrt(eigvals[:k])
            reproduced = loadings @ loadings.T
        residual = corr - reproduced
        np.fill_diagonal(residual, 0.0)
        off_diag = residual[upper]
        map_curve[k] = float(np.mean(off_diag * off_diag))
    return map_curve


def _scree_elbow(eigenvalues: Iterable[float]) -> int:
    values = np.asarray(tuple(eigenvalues), dtype=float)
    n = values.shape[0]
    if n == 1:
        return 1

    x = np.arange(1, n + 1, dtype=float)
    x1, y1 = x[0], values[0]
    x2, y2 = x[-1], values[-1]
    denom = np.hypot(y2 - y1, x2 - x1)
    if denom <= 0:
        return 1

    distance = np.abs((y2 - y1) * x - (x2 - x1) * values + x2 * y1 - y2 * x1) / denom
    return int(x[np.argmax(distance)])


def _aggregate_consensus(
    *,
    suggestions: dict[str, int],
    n_min: int,
    n_max: int,
    strategy: str,
) -> int:
    if not suggestions:
        raise ValueError("No factor-count suggestion available.")
    if strategy != "majority_min_tie":
        raise ValueError(f"Unsupported consensus strategy `{strategy}`.")

    counts = Counter(suggestions.values())
    top_votes = max(counts.values())
    winners = [k for k, v in counts.items() if v == top_votes]
    return _clip_n_factors(min(winners), n_min=n_min, n_max=n_max)


def _clip_n_factors(value: int, *, n_min: int, n_max: int) -> int:
    return int(min(max(value, n_min), n_max))


def _stabilize_correlation(corr: np.ndarray) -> np.ndarray:
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr, 1.0)
    return corr
