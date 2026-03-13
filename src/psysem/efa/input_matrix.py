from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar
from scipy.stats import multivariate_normal, norm

if TYPE_CHECKING:
    from .fit import EFAConfig


@dataclass(frozen=True)
class _EFAInputMatrix:
    """Internal prepared input for EFA fitting.

    The first version intentionally preserves the current behavior of
    ``fit_efa()``: select the configured item columns, compute the default
    pandas Pearson correlation matrix, then stabilize the resulting matrix for
    downstream eigendecomposition and optimization.
    """

    corr: NDArray[np.float64]
    item_names: tuple[str, ...]
    warnings: tuple[str, ...] = ()


def normalize_missing_strategy(strategy: str) -> str:
    """Normalize missing-data strategy names for EFA input preparation."""

    return strategy.strip().lower()


def normalize_correlation_method(method: str) -> str:
    """Normalize correlation-method names for EFA input preparation."""

    return method.strip().lower()


def build_efa_input_matrix(data: pd.DataFrame, config: EFAConfig) -> _EFAInputMatrix:
    """Build the correlation-matrix input used by EFA extraction.

    This module is the staging point for future input-preprocessing features
    such as explicit missing-data strategies and alternative correlation types.
    """

    item_names = tuple(config.items)
    item_frame = data.loc[:, list(item_names)]
    missing_strategy = normalize_missing_strategy(config.missing_strategy)
    correlation_method = normalize_correlation_method(config.correlation_method)
    resolved_variable_types = _resolve_variable_types(
        item_frame,
        declared_variable_types=config.variable_types,
    )
    _validate_correlation_method_inputs(
        item_names=item_names,
        correlation_method=correlation_method,
        resolved_variable_types=resolved_variable_types,
    )
    corr, warnings = _compute_correlation_matrix(
        item_frame,
        missing_strategy=missing_strategy,
        correlation_method=correlation_method,
        resolved_variable_types=resolved_variable_types,
    )
    recommendations = _build_preprocessing_recommendations(
        resolved_variable_types=resolved_variable_types,
        correlation_method=correlation_method,
        declared_variable_types=config.variable_types,
    )
    corr = _stabilize_correlation_matrix(corr)
    return _EFAInputMatrix(
        corr=corr,
        item_names=item_names,
        warnings=tuple(dict.fromkeys((*warnings, *recommendations))),
    )


def _compute_correlation_matrix(
    item_frame: pd.DataFrame,
    *,
    missing_strategy: str,
    correlation_method: str,
    resolved_variable_types: dict[str, str],
) -> tuple[NDArray[np.float64], tuple[str, ...]]:
    if correlation_method == "polychoric":
        return _compute_polychoric_matrix(
            item_frame,
            missing_strategy=missing_strategy,
        )

    if missing_strategy == "dropna":
        complete = item_frame.dropna(axis=0, how="any")
        corr = complete.corr(method=correlation_method).to_numpy(dtype=float)
        dropped_rows = int(item_frame.shape[0] - complete.shape[0])
        warnings: list[str] = []
        if dropped_rows > 0:
            warnings.append(
                f"Input preprocessing dropped {dropped_rows} row(s) with missing values under dropna strategy."
            )
        if correlation_method == "spearman":
            warnings.append("Input preprocessing used Spearman rank correlation.")
        return corr, tuple(warnings)

    pairwise_counts = item_frame.notna().astype(int).T @ item_frame.notna().astype(int)
    corr = item_frame.corr(method=correlation_method).to_numpy(dtype=float)
    warnings = []
    unique_counts = np.unique(pairwise_counts.to_numpy(dtype=int))
    if unique_counts.size > 1:
        warnings.append(
            "Pairwise missing strategy used variable-specific observation counts when building the correlation matrix."
        )
    if correlation_method == "spearman":
        warnings.append("Input preprocessing used Spearman rank correlation.")
    return corr, tuple(warnings)


def _stabilize_correlation_matrix(corr: NDArray[np.float64]) -> NDArray[np.float64]:
    """Ensure the correlation matrix is numeric, symmetric, and usable."""

    stabilized = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    stabilized = (stabilized + stabilized.T) / 2.0
    np.fill_diagonal(stabilized, 1.0)
    eigenvalues, eigenvectors = np.linalg.eigh(stabilized)
    min_eigenvalue = float(np.min(eigenvalues))
    if min_eigenvalue < 1e-8:
        clipped = np.clip(eigenvalues, 1e-8, None)
        stabilized = eigenvectors @ np.diag(clipped) @ eigenvectors.T
        scales = np.sqrt(np.clip(np.diag(stabilized), 1e-12, None))
        stabilized = stabilized / np.outer(scales, scales)
        stabilized = (stabilized + stabilized.T) / 2.0
        np.fill_diagonal(stabilized, 1.0)
    return stabilized


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


def _compute_polychoric_matrix(
    item_frame: pd.DataFrame,
    *,
    missing_strategy: str,
) -> tuple[NDArray[np.float64], tuple[str, ...]]:
    item_names = list(item_frame.columns)
    n_items = len(item_names)
    corr = np.eye(n_items, dtype=float)
    warnings: list[str] = ["Input preprocessing used polychoric correlation for ordinal items."]
    pairwise_counts = np.zeros((n_items, n_items), dtype=int)

    if missing_strategy == "dropna":
        analysis_frame = item_frame.dropna(axis=0, how="any")
        dropped_rows = int(item_frame.shape[0] - analysis_frame.shape[0])
        if dropped_rows > 0:
            warnings.append(
                f"Input preprocessing dropped {dropped_rows} row(s) with missing values under dropna strategy."
            )
    else:
        analysis_frame = item_frame

    for i, left_name in enumerate(item_names):
        pairwise_counts[i, i] = int(analysis_frame[left_name].dropna().shape[0])
        for j in range(i + 1, n_items):
            right_name = item_names[j]
            if missing_strategy == "dropna":
                pair_frame = analysis_frame.loc[:, [left_name, right_name]]
            else:
                pair_frame = item_frame.loc[:, [left_name, right_name]].dropna(axis=0, how="any")
            pairwise_counts[i, j] = pairwise_counts[j, i] = int(pair_frame.shape[0])

            rho, pair_warnings = _estimate_polychoric_correlation(
                pair_frame[left_name],
                pair_frame[right_name],
                pair_label=f"{left_name}~{right_name}",
            )
            corr[i, j] = corr[j, i] = rho
            warnings.extend(pair_warnings)

    if missing_strategy == "pairwise":
        unique_counts = np.unique(pairwise_counts[np.triu_indices(n_items, k=1)])
        if unique_counts.size > 1:
            warnings.append(
                "Pairwise missing strategy used variable-specific observation counts when building the polychoric correlation matrix."
            )

    return corr, tuple(dict.fromkeys(warnings))


def _estimate_polychoric_correlation(
    left: pd.Series,
    right: pd.Series,
    *,
    pair_label: str,
) -> tuple[float, tuple[str, ...]]:
    if left.empty or right.empty:
        return 0.0, (f"Polychoric estimation for {pair_label} had no complete observations; using 0.0.",)

    left_codes, left_thresholds = _ordinal_codes_and_thresholds(left)
    right_codes, right_thresholds = _ordinal_codes_and_thresholds(right)
    n_left = len(left_thresholds) - 1
    n_right = len(right_thresholds) - 1
    if n_left < 2 or n_right < 2:
        return (
            0.0,
            (f"Polychoric estimation for {pair_label} requires at least two observed categories per item; using 0.0.",),
        )

    contingency = pd.crosstab(left_codes, right_codes, dropna=False)
    contingency = contingency.reindex(index=range(n_left), columns=range(n_right), fill_value=0)
    counts = contingency.to_numpy(dtype=float)
    if not np.any(counts):
        return 0.0, (f"Polychoric estimation for {pair_label} had no usable contingency counts; using 0.0.",)

    start = float(np.corrcoef(left_codes.to_numpy(dtype=float), right_codes.to_numpy(dtype=float))[0, 1])
    if not np.isfinite(start):
        start = 0.0
    start = float(np.clip(start, -0.95, 0.95))

    def objective(rho: float) -> float:
        probs = _polychoric_cell_probabilities(left_thresholds, right_thresholds, rho)
        return float(-np.sum(counts * np.log(np.clip(probs, 1e-12, None))))

    best_rho = start
    best_value = objective(start)
    result = minimize_scalar(objective, bounds=(-0.995, 0.995), method="bounded", options={"xatol": 1e-4})
    if result.success and np.isfinite(result.fun):
        best_rho = float(result.x)
        best_value = float(result.fun)

    grid = np.linspace(-0.95, 0.95, num=9)
    for rho in grid:
        value = objective(float(rho))
        if value < best_value:
            best_rho = float(rho)
            best_value = float(value)

    if not result.success:
        warning = f"Polychoric estimation for {pair_label} did not fully converge; using best available estimate."
        return float(np.clip(best_rho, -0.999, 0.999)), (warning,)
    return float(np.clip(best_rho, -0.999, 0.999)), ()


def _ordinal_codes_and_thresholds(series: pd.Series) -> tuple[pd.Series, NDArray[np.float64]]:
    values = series.dropna()
    if isinstance(values.dtype, pd.CategoricalDtype):
        categorical = values.astype("category")
        categories = list(categorical.cat.categories)
    else:
        categories = sorted(values.unique().tolist())
        categorical = pd.Categorical(values, categories=categories, ordered=True)

    codes = pd.Series(categorical.codes, index=values.index, dtype=int)
    counts = np.bincount(codes.to_numpy(dtype=int), minlength=len(categories)).astype(float)
    proportions = counts / max(float(np.sum(counts)), 1.0)
    cumulative = np.cumsum(proportions)[:-1]
    if cumulative.size:
        cumulative = np.clip(cumulative, 1e-6, 1.0 - 1e-6)
        inner = norm.ppf(cumulative)
    else:
        inner = np.array([], dtype=float)
    thresholds = np.concatenate(([-np.inf], inner, [np.inf])).astype(float)
    return codes, thresholds


def _polychoric_cell_probabilities(
    left_thresholds: NDArray[np.float64],
    right_thresholds: NDArray[np.float64],
    rho: float,
) -> NDArray[np.float64]:
    n_left = len(left_thresholds) - 1
    n_right = len(right_thresholds) - 1
    probs = np.zeros((n_left, n_right), dtype=float)
    bounded_rho = float(np.clip(rho, -0.999, 0.999))
    for i in range(n_left):
        for j in range(n_right):
            probs[i, j] = _bivariate_normal_rectangle_probability(
                x_low=left_thresholds[i],
                x_high=left_thresholds[i + 1],
                y_low=right_thresholds[j],
                y_high=right_thresholds[j + 1],
                rho=bounded_rho,
            )
    probs = np.clip(probs, 1e-12, None)
    probs /= float(np.sum(probs))
    return probs


def _bivariate_normal_rectangle_probability(
    *,
    x_low: float,
    x_high: float,
    y_low: float,
    y_high: float,
    rho: float,
) -> float:
    upper_upper = _bivariate_normal_cdf(x_high, y_high, rho)
    lower_upper = _bivariate_normal_cdf(x_low, y_high, rho)
    upper_lower = _bivariate_normal_cdf(x_high, y_low, rho)
    lower_lower = _bivariate_normal_cdf(x_low, y_low, rho)
    return float(max(upper_upper - lower_upper - upper_lower + lower_lower, 1e-12))


def _bivariate_normal_cdf(x: float, y: float, rho: float) -> float:
    if np.isneginf(x) or np.isneginf(y):
        return 0.0
    if np.isposinf(x) and np.isposinf(y):
        return 1.0
    if np.isposinf(x):
        return float(norm.cdf(y))
    if np.isposinf(y):
        return float(norm.cdf(x))

    cov = np.array([[1.0, rho], [rho, 1.0]], dtype=float)
    return float(multivariate_normal(mean=np.zeros(2), cov=cov, allow_singular=False).cdf([x, y]))


def _resolve_variable_types(
    item_frame: pd.DataFrame,
    *,
    declared_variable_types: dict[str, str] | None,
) -> dict[str, str]:
    resolved: dict[str, str] = {}
    declared = {
        name: kind.strip().lower()
        for name, kind in (declared_variable_types or {}).items()
    }

    for column_name in item_frame.columns:
        if column_name in declared:
            resolved[column_name] = declared[column_name]
            continue
        resolved[column_name] = _infer_variable_type(item_frame[column_name])

    return resolved


def _infer_variable_type(series: pd.Series) -> str:
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


def _build_preprocessing_recommendations(
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
            f'{source.capitalize()} ordinal-like items detected ({", ".join(ordinal_items)}); consider `correlation_method="spearman"` now or `polychoric` when available.',
        )
    if correlation_method == "spearman":
        return (
            f"{source.capitalize()} ordinal-like items detected ({', '.join(ordinal_items)}); Spearman is a lightweight fallback until `polychoric` is available.",
        )
    return ()
