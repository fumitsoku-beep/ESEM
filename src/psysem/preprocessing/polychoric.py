from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar
from scipy.stats import multivariate_normal, norm


def build_polychoric_matrix(
    item_frame: pd.DataFrame,
    *,
    missing_strategy: str,
) -> tuple[NDArray[np.float64], pd.DataFrame, int, int | None, tuple[str, ...]]:
    item_names = list(item_frame.columns)
    n_items = len(item_names)
    corr = np.eye(n_items, dtype=float)
    warnings: list[str] = ["Input preprocessing used polychoric correlation for ordinal items."]
    pairwise_counts = np.zeros((n_items, n_items), dtype=int)
    dropped_rows = 0
    n_complete_rows: int | None = None

    if missing_strategy == "dropna":
        analysis_frame = item_frame.dropna(axis=0, how="any")
        dropped_rows = int(item_frame.shape[0] - analysis_frame.shape[0])
        n_complete_rows = int(analysis_frame.shape[0])
        if dropped_rows > 0:
            warnings.append(
                f"Input preprocessing dropped {dropped_rows} row(s) with missing values under dropna strategy."
            )
    else:
        analysis_frame = item_frame
        n_complete_rows = int(item_frame.dropna(axis=0, how="any").shape[0])

    for i, left_name in enumerate(item_names):
        pairwise_counts[i, i] = int(analysis_frame[left_name].dropna().shape[0])
        for j in range(i + 1, n_items):
            right_name = item_names[j]
            if missing_strategy == "dropna":
                pair_frame = analysis_frame.loc[:, [left_name, right_name]]
            else:
                pair_frame = item_frame.loc[:, [left_name, right_name]].dropna(axis=0, how="any")
            pairwise_counts[i, j] = pairwise_counts[j, i] = int(pair_frame.shape[0])

            rho, pair_warnings = estimate_polychoric_correlation(
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

    pairwise_n = pd.DataFrame(pairwise_counts, index=item_names, columns=item_names, dtype=int)
    return corr, pairwise_n, dropped_rows, n_complete_rows, tuple(dict.fromkeys(warnings))


def estimate_polychoric_correlation(
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
    result = minimize_scalar(
        objective,
        bounds=(-0.995, 0.995),
        method="bounded",
        options={"xatol": 1e-4},
    )
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
