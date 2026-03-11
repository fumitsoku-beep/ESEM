from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..measurement import MeasurementDesign
from ..parameter_index import ParameterIndexMap
from ..structural import StructuralDesign
from .contracts import MLEstimationContext, MLOptimizationResult


def gaussian_ml_discrepancy(
    sample_covariance: np.ndarray,
    implied_covariance: np.ndarray,
) -> float:
    """Compute Gaussian ML discrepancy function F_ml(S, Sigma)."""
    sample = np.asarray(sample_covariance, dtype=float)
    implied = np.asarray(implied_covariance, dtype=float)
    if sample.ndim != 2 or implied.ndim != 2:
        raise ValueError("Covariance inputs must be 2D arrays.")
    if sample.shape[0] != sample.shape[1]:
        raise ValueError("`sample_covariance` must be square.")
    if implied.shape != sample.shape:
        raise ValueError("`implied_covariance` must match sample covariance shape.")

    sign_sample, logdet_sample = np.linalg.slogdet(sample)
    sign_implied, logdet_implied = np.linalg.slogdet(implied)
    if sign_sample <= 0 or sign_implied <= 0:
        raise ValueError("Covariance matrix is not positive definite.")

    trace_term = float(np.trace(sample @ np.linalg.inv(implied)))
    p = float(sample.shape[0])
    value = logdet_implied + trace_term - logdet_sample - p
    if value < 0 and math.isclose(value, 0.0, rel_tol=0.0, abs_tol=1e-10):
        return 0.0
    return float(value)


def build_ml_context(
    data: object,
    *,
    observed_variables: tuple[str, ...],
) -> MLEstimationContext:
    """Prepare a basic ML context from observed data columns."""
    if not isinstance(data, pd.DataFrame):
        return MLEstimationContext(
            observed_variables=tuple(),
            sample_covariance=None,
            objective_at_sample_cov=None,
            warnings=("ML context skipped: `data` is not a pandas DataFrame.",),
        )

    available_columns = [name for name in observed_variables if name in data.columns]
    if not available_columns:
        return MLEstimationContext(
            observed_variables=tuple(),
            sample_covariance=None,
            objective_at_sample_cov=None,
            warnings=("ML context skipped: no observed variables found in DataFrame.",),
        )

    warnings: list[str] = []
    if len(available_columns) < len(observed_variables):
        missing = [name for name in observed_variables if name not in available_columns]
        warnings.append(
            "ML context uses partial observed set; missing columns: " + ", ".join(missing)
        )

    numeric = data.loc[:, available_columns].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna(axis=0, how="any")
    if len(numeric) < 2:
        warnings.append("ML context skipped: fewer than 2 complete rows after numeric filtering.")
        return MLEstimationContext(
            observed_variables=tuple(available_columns),
            sample_covariance=None,
            objective_at_sample_cov=None,
            warnings=tuple(warnings),
        )

    raw_sample_cov = numeric.cov()
    sample_cov_array, was_adjusted = _ensure_positive_definite(raw_sample_cov.to_numpy(dtype=float))
    sample_cov = pd.DataFrame(sample_cov_array, index=available_columns, columns=available_columns)
    if was_adjusted:
        warnings.append("Sample covariance was adjusted to positive-definite for ML prototype.")

    objective: float | None = None
    try:
        objective = gaussian_ml_discrepancy(sample_cov_array, sample_cov_array)
    except ValueError as exc:
        warnings.append(f"ML objective placeholder unavailable: {exc}")

    return MLEstimationContext(
        observed_variables=tuple(available_columns),
        sample_covariance=sample_cov,
        objective_at_sample_cov=objective,
        warnings=tuple(warnings),
    )


def build_implied_covariance(
    measurement_design: MeasurementDesign,
    parameter_vector: np.ndarray,
    parameter_index_map: ParameterIndexMap,
    *,
    structural_design: StructuralDesign | None = None,
    observed_variables: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Construct implied observed covariance from current parameter vector."""
    observed = (
        tuple(measurement_design.observed_variables)
        if observed_variables is None
        else tuple(observed_variables)
    )
    missing_observed = [name for name in observed if name not in measurement_design.lambda_matrix.index]
    if missing_observed:
        raise ValueError(
            "Observed variables missing from measurement design: " + ", ".join(missing_observed)
        )

    vector_by_index = _parameter_vector_by_index(parameter_vector, parameter_index_map)
    latent_order = tuple(measurement_design.latent_variables)

    lambda_df = measurement_design.lambda_matrix.loc[list(observed), list(latent_order)]
    lambda_index_df = measurement_design.lambda_parameter_index.loc[list(observed), list(latent_order)]
    lambda_array = _resolve_matrix(
        lambda_df,
        parameter_index=lambda_index_df,
        vector_by_index=vector_by_index,
        nan_default=0.0,
    )

    theta_df = measurement_design.theta_matrix.loc[list(observed), list(observed)]
    theta_array = _resolve_matrix(theta_df, parameter_index=None, vector_by_index=vector_by_index, nan_default=1.0)
    theta_array = np.diag(np.diag(theta_array))

    phi_array, _ = _build_latent_covariance(
        latent_order=latent_order,
        structural_design=structural_design,
        vector_by_index=vector_by_index,
    )
    sigma = lambda_array @ phi_array @ lambda_array.T + theta_array
    sigma = (sigma + sigma.T) / 2.0
    sigma += np.eye(sigma.shape[0], dtype=float) * 1e-8
    return pd.DataFrame(sigma, index=observed, columns=observed)


def optimize_ml_parameters(
    data: object,
    *,
    measurement_design: MeasurementDesign,
    structural_design: StructuralDesign | None,
    parameter_index_map: ParameterIndexMap,
    parameter_table: tuple[dict[str, Any], ...],
    max_iter: int = 200,
) -> MLOptimizationResult:
    """Run prototype ML optimization for SEM parameters."""
    if parameter_index_map.n_free == 0:
        return MLOptimizationResult(
            success=True,
            status="skipped_no_free_parameters",
            n_iter=0,
            objective=0.0,
            observed_variables=tuple(),
            parameter_vector=tuple(),
        )

    context = build_ml_context(data, observed_variables=measurement_design.observed_variables)
    warnings = list(context.warnings)
    if context.sample_covariance is None:
        return MLOptimizationResult(
            success=False,
            status="skipped_no_sample_covariance",
            n_iter=0,
            objective=None,
            observed_variables=context.observed_variables,
            parameter_vector=tuple(),
            warnings=tuple(warnings),
        )

    sample_cov = context.sample_covariance
    x0 = build_start_vector(parameter_index_map, parameter_table=parameter_table)
    bounds = _build_bounds(parameter_index_map, parameter_table=parameter_table)
    latent_warnings = _structural_prototype_warnings(structural_design)
    warnings.extend(latent_warnings)

    def objective_fn(x: np.ndarray) -> float:
        try:
            implied = build_implied_covariance(
                measurement_design,
                x,
                parameter_index_map,
                structural_design=structural_design,
                observed_variables=context.observed_variables,
            )
            return gaussian_ml_discrepancy(sample_cov.to_numpy(dtype=float), implied.to_numpy(dtype=float))
        except Exception:
            return 1e12

    raw_result = minimize(
        objective_fn,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": int(max_iter)},
    )
    best_vector = np.asarray(raw_result.x, dtype=float)
    objective_value: float | None
    implied_covariance: pd.DataFrame | None
    try:
        implied_covariance = build_implied_covariance(
            measurement_design,
            best_vector,
            parameter_index_map,
            structural_design=structural_design,
            observed_variables=context.observed_variables,
        )
        objective_value = gaussian_ml_discrepancy(
            sample_cov.to_numpy(dtype=float),
            implied_covariance.to_numpy(dtype=float),
        )
    except Exception as exc:
        warnings.append(f"Failed to evaluate implied covariance at optimizer solution: {exc}")
        implied_covariance = None
        objective_value = None

    success = bool(raw_result.success) and objective_value is not None and math.isfinite(objective_value)
    status = "converged" if success else f"failed: {raw_result.message}"
    parameter_values = parameter_vector_to_named_values(best_vector, parameter_index_map)

    return MLOptimizationResult(
        success=success,
        status=status,
        n_iter=int(getattr(raw_result, "nit", 0)),
        objective=objective_value,
        observed_variables=context.observed_variables,
        parameter_vector=tuple(float(item) for item in best_vector.tolist()),
        parameter_values=parameter_values,
        sample_covariance=sample_cov,
        implied_covariance=implied_covariance,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def build_start_vector(
    parameter_index_map: ParameterIndexMap,
    *,
    parameter_table: tuple[dict[str, Any], ...],
) -> np.ndarray:
    """Build a deterministic prototype start vector by parameter role."""
    start_by_index: dict[int, float] = {}
    for row in parameter_table:
        if not bool(row["is_free"]):
            continue
        parameter_index = row.get("parameter_index")
        if not isinstance(parameter_index, int):
            continue
        if parameter_index in start_by_index:
            continue
        operator = row.get("operator")
        lhs = row.get("lhs")
        rhs = row.get("rhs")
        if operator == "=~":
            start_by_index[parameter_index] = 0.6
        elif operator == "~":
            start_by_index[parameter_index] = 0.15
        elif operator == "~~" and isinstance(lhs, str) and isinstance(rhs, str) and lhs == rhs:
            start_by_index[parameter_index] = 1.0
        else:
            start_by_index[parameter_index] = 0.2

    vector = np.zeros(parameter_index_map.n_free, dtype=float)
    for entry in parameter_index_map.entries:
        vector[entry.vector_position] = start_by_index.get(entry.parameter_index, 0.2)
    return vector


def parameter_vector_to_named_values(
    parameter_vector: np.ndarray,
    parameter_index_map: ParameterIndexMap,
) -> dict[str, float]:
    """Convert optimization vector to public parameter-name mapping."""
    vector = np.asarray(parameter_vector, dtype=float)
    if vector.shape != (parameter_index_map.n_free,):
        raise ValueError(
            f"`parameter_vector` shape {vector.shape} does not match n_free={parameter_index_map.n_free}."
        )
    values: dict[str, float] = {}
    for entry in parameter_index_map.entries:
        values[entry.parameter] = float(vector[entry.vector_position])
    return values


def _parameter_vector_by_index(
    parameter_vector: np.ndarray,
    parameter_index_map: ParameterIndexMap,
) -> dict[int, float]:
    vector = np.asarray(parameter_vector, dtype=float)
    if vector.shape != (parameter_index_map.n_free,):
        raise ValueError(
            f"`parameter_vector` shape {vector.shape} does not match n_free={parameter_index_map.n_free}."
        )
    return {
        entry.parameter_index: float(vector[entry.vector_position])
        for entry in parameter_index_map.entries
    }


def _resolve_matrix(
    matrix: pd.DataFrame,
    *,
    parameter_index: pd.DataFrame | None,
    vector_by_index: dict[int, float],
    nan_default: float,
) -> np.ndarray:
    resolved = matrix.to_numpy(dtype=float, copy=True)
    if parameter_index is not None:
        index_array = parameter_index.to_numpy(dtype=int, copy=False)
        nonzero_positions = np.argwhere(index_array > 0)
        for row, col in nonzero_positions:
            param_idx = int(index_array[row, col])
            resolved[row, col] = float(vector_by_index.get(param_idx, nan_default))
    resolved = np.nan_to_num(resolved, nan=nan_default)
    return resolved


def _build_latent_covariance(
    *,
    latent_order: tuple[str, ...],
    structural_design: StructuralDesign | None,
    vector_by_index: dict[int, float],
) -> tuple[np.ndarray, list[str]]:
    warnings: list[str] = []
    n_latent = len(latent_order)
    if structural_design is None or n_latent == 0:
        return np.eye(n_latent, dtype=float), warnings

    phi = np.eye(n_latent, dtype=float)
    latent_to_idx = {name: idx for idx, name in enumerate(latent_order)}
    endogenous = tuple(
        name for name in structural_design.endogenous_latent_variables if name in latent_to_idx
    )
    exogenous = tuple(
        name for name in structural_design.exogenous_latent_variables if name in latent_to_idx
    )
    if not endogenous:
        return phi, warnings

    beta_df = structural_design.beta_matrix.loc[list(endogenous), list(endogenous)]
    beta_idx_df = structural_design.beta_parameter_index.loc[list(endogenous), list(endogenous)]
    beta = _resolve_matrix(
        beta_df,
        parameter_index=beta_idx_df,
        vector_by_index=vector_by_index,
        nan_default=0.0,
    )

    gamma_columns = [name for name in structural_design.gamma_matrix.columns if name in exogenous]
    if len(gamma_columns) < len(structural_design.exogenous_latent_variables):
        if structural_design.observed_predictor_variables:
            warnings.append(
                "Structural observed predictors are not included in ML prototype implied covariance."
            )
    gamma = np.zeros((len(endogenous), len(exogenous)), dtype=float)
    for col_name in gamma_columns:
        src = structural_design.gamma_matrix.loc[list(endogenous), col_name]
        src_idx = structural_design.gamma_parameter_index.loc[list(endogenous), col_name]
        col_array = _resolve_matrix(
            src.to_frame(),
            parameter_index=src_idx.to_frame(),
            vector_by_index=vector_by_index,
            nan_default=0.0,
        )[:, 0]
        exo_position = exogenous.index(col_name)
        gamma[:, exo_position] = col_array

    psi_df = structural_design.psi_matrix.loc[list(endogenous), list(endogenous)]
    psi_idx_df = structural_design.psi_parameter_index.loc[list(endogenous), list(endogenous)]
    psi = _resolve_matrix(
        psi_df,
        parameter_index=psi_idx_df,
        vector_by_index=vector_by_index,
        nan_default=1.0,
    )
    psi = np.diag(np.maximum(np.diag(psi), 1e-6))

    try:
        transform = np.linalg.inv(np.eye(len(endogenous), dtype=float) - beta)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"Failed to invert (I-Beta): {exc}") from exc

    cov_exo = np.eye(len(exogenous), dtype=float)
    cov_endo = transform @ (gamma @ cov_exo @ gamma.T + psi) @ transform.T
    cov_exo_endo = np.zeros((len(exogenous), len(endogenous)), dtype=float)
    if len(exogenous) > 0:
        cov_exo_endo = (transform @ gamma).T

    for exo_i, exo_name in enumerate(exogenous):
        phi[latent_to_idx[exo_name], latent_to_idx[exo_name]] = 1.0
        for endo_j, endo_name in enumerate(endogenous):
            cov_value = cov_exo_endo[exo_i, endo_j]
            phi[latent_to_idx[exo_name], latent_to_idx[endo_name]] = cov_value
            phi[latent_to_idx[endo_name], latent_to_idx[exo_name]] = cov_value
    for endo_i, endo_name_i in enumerate(endogenous):
        for endo_j, endo_name_j in enumerate(endogenous):
            phi[latent_to_idx[endo_name_i], latent_to_idx[endo_name_j]] = cov_endo[endo_i, endo_j]

    phi = (phi + phi.T) / 2.0
    phi, _ = _ensure_positive_definite(phi)
    return phi, warnings


def _build_bounds(
    parameter_index_map: ParameterIndexMap,
    *,
    parameter_table: tuple[dict[str, Any], ...],
) -> list[tuple[float | None, float | None]]:
    lower_by_index: dict[int, float | None] = {
        entry.parameter_index: None for entry in parameter_index_map.entries
    }
    upper_by_index: dict[int, float | None] = {
        entry.parameter_index: None for entry in parameter_index_map.entries
    }
    for row in parameter_table:
        if not bool(row["is_free"]):
            continue
        parameter_index = row.get("parameter_index")
        if not isinstance(parameter_index, int):
            continue
        operator = row.get("operator")
        lhs = row.get("lhs")
        rhs = row.get("rhs")
        if operator == "~~" and isinstance(lhs, str) and isinstance(rhs, str) and lhs == rhs:
            current = lower_by_index.get(parameter_index)
            new_lower = 1e-6
            lower_by_index[parameter_index] = (
                new_lower if current is None else max(current, new_lower)
            )

    bounds: list[tuple[float | None, float | None]] = []
    for entry in parameter_index_map.entries:
        bounds.append(
            (
                lower_by_index.get(entry.parameter_index),
                upper_by_index.get(entry.parameter_index),
            )
        )
    return bounds


def _structural_prototype_warnings(structural_design: StructuralDesign | None) -> list[str]:
    if structural_design is None:
        return []
    if structural_design.observed_predictor_variables:
        return [
            "ML prototype currently ignores observed structural predictors in implied covariance."
        ]
    return []


def _ensure_positive_definite(
    matrix: np.ndarray,
    *,
    min_eigenvalue: float = 1e-6,
) -> tuple[np.ndarray, bool]:
    sym = (matrix + matrix.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(sym)
    adjusted = np.maximum(eigenvalues, min_eigenvalue)
    was_adjusted = bool(np.any(adjusted != eigenvalues))
    rebuilt = (eigenvectors * adjusted) @ eigenvectors.T
    rebuilt = (rebuilt + rebuilt.T) / 2.0
    return rebuilt, was_adjusted

