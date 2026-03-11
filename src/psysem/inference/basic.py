from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.stats import norm

from ..parameter_index import ParameterIndexMap


@dataclass(frozen=True)
class ParameterInferenceEntry:
    """One parameter-level inference row."""

    parameter: str
    parameter_index: int
    vector_position: int
    estimate: float
    standard_error: float | None
    z_value: float | None
    p_value: float | None
    ci_lower: float | None
    ci_upper: float | None


@dataclass(frozen=True)
class InferenceResult:
    """Numerical inference result for SEM free parameters."""

    entries: tuple[ParameterInferenceEntry, ...]
    hessian_matrix: np.ndarray | None
    covariance_matrix: np.ndarray | None
    warnings: tuple[str, ...] = field(default_factory=tuple)


def estimate_parameter_inference(
    *,
    objective_fn: Callable[[np.ndarray], float],
    parameter_vector: np.ndarray,
    parameter_index_map: ParameterIndexMap,
    alpha: float = 0.05,
    step_scale: float = 1e-4,
) -> InferenceResult:
    """Estimate SE/z/p/CI from numerical Hessian at optimized parameters."""
    if not (0.0 < alpha < 1.0):
        raise ValueError("`alpha` must be in (0, 1).")
    if step_scale <= 0:
        raise ValueError("`step_scale` must be positive.")

    x = np.asarray(parameter_vector, dtype=float)
    if x.shape != (parameter_index_map.n_free,):
        raise ValueError(
            f"`parameter_vector` shape {x.shape} does not match n_free={parameter_index_map.n_free}."
        )
    if parameter_index_map.n_free == 0:
        return InferenceResult(entries=tuple(), hessian_matrix=None, covariance_matrix=None)

    warnings: list[str] = []
    try:
        hessian = _numerical_hessian(objective_fn, x, step_scale=step_scale)
    except Exception as exc:
        warnings.append(f"Numerical Hessian failed: {exc}")
        fallback_entries = _empty_entries(parameter_index_map=parameter_index_map, estimates=x)
        return InferenceResult(
            entries=fallback_entries,
            hessian_matrix=None,
            covariance_matrix=None,
            warnings=tuple(warnings),
        )

    covariance: np.ndarray | None
    try:
        covariance = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        covariance = np.linalg.pinv(hessian)
        warnings.append("Hessian is singular; pseudo-inverse used for covariance approximation.")

    covariance = (covariance + covariance.T) / 2.0
    diag = np.diag(covariance)
    se = np.full(parameter_index_map.n_free, np.nan, dtype=float)
    positive = diag > 0
    se[positive] = np.sqrt(diag[positive])
    if not np.all(positive):
        warnings.append("Some covariance diagonal entries are non-positive; related SE values set to nan.")

    z_values = np.full(parameter_index_map.n_free, np.nan, dtype=float)
    valid_se = np.isfinite(se) & (se > 0.0)
    z_values[valid_se] = x[valid_se] / se[valid_se]
    p_values = np.full(parameter_index_map.n_free, np.nan, dtype=float)
    p_values[valid_se] = 2.0 * (1.0 - norm.cdf(np.abs(z_values[valid_se])))
    critical = float(norm.ppf(1.0 - alpha / 2.0))

    rows: list[ParameterInferenceEntry] = []
    for entry in parameter_index_map.entries:
        pos = entry.vector_position
        estimate = float(x[pos])
        se_value = float(se[pos]) if np.isfinite(se[pos]) else None
        z_value = float(z_values[pos]) if np.isfinite(z_values[pos]) else None
        p_value = float(p_values[pos]) if np.isfinite(p_values[pos]) else None
        ci_lower: float | None = None
        ci_upper: float | None = None
        if se_value is not None:
            ci_lower = estimate - critical * se_value
            ci_upper = estimate + critical * se_value
        rows.append(
            ParameterInferenceEntry(
                parameter=entry.parameter,
                parameter_index=entry.parameter_index,
                vector_position=entry.vector_position,
                estimate=estimate,
                standard_error=se_value,
                z_value=z_value,
                p_value=p_value,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
            )
        )

    return InferenceResult(
        entries=tuple(rows),
        hessian_matrix=hessian,
        covariance_matrix=covariance,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _empty_entries(
    *,
    parameter_index_map: ParameterIndexMap,
    estimates: np.ndarray,
) -> tuple[ParameterInferenceEntry, ...]:
    rows: list[ParameterInferenceEntry] = []
    for entry in parameter_index_map.entries:
        estimate = float(estimates[entry.vector_position])
        rows.append(
            ParameterInferenceEntry(
                parameter=entry.parameter,
                parameter_index=entry.parameter_index,
                vector_position=entry.vector_position,
                estimate=estimate,
                standard_error=None,
                z_value=None,
                p_value=None,
                ci_lower=None,
                ci_upper=None,
            )
        )
    return tuple(rows)


def _numerical_hessian(
    objective_fn: Callable[[np.ndarray], float],
    x: np.ndarray,
    *,
    step_scale: float,
) -> np.ndarray:
    n = int(x.shape[0])
    hessian = np.zeros((n, n), dtype=float)
    steps = step_scale * np.maximum(1.0, np.abs(x))
    f0 = float(objective_fn(x))

    for i in range(n):
        hi = float(steps[i])
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += hi
        x_minus[i] -= hi
        f_plus = float(objective_fn(x_plus))
        f_minus = float(objective_fn(x_minus))
        hessian[i, i] = (f_plus - 2.0 * f0 + f_minus) / (hi * hi)

        for j in range(i + 1, n):
            hj = float(steps[j])
            x_pp = x.copy()
            x_pm = x.copy()
            x_mp = x.copy()
            x_mm = x.copy()
            x_pp[i] += hi
            x_pp[j] += hj
            x_pm[i] += hi
            x_pm[j] -= hj
            x_mp[i] -= hi
            x_mp[j] += hj
            x_mm[i] -= hi
            x_mm[j] -= hj
            f_pp = float(objective_fn(x_pp))
            f_pm = float(objective_fn(x_pm))
            f_mp = float(objective_fn(x_mp))
            f_mm = float(objective_fn(x_mm))
            value = (f_pp - f_pm - f_mp + f_mm) / (4.0 * hi * hj)
            hessian[i, j] = value
            hessian[j, i] = value

    return hessian
