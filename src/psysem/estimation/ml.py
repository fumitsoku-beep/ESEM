from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .contracts import MLEstimationContext


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

    sample_cov = numeric.cov()
    objective: float | None = None
    try:
        objective = gaussian_ml_discrepancy(sample_cov.to_numpy(), sample_cov.to_numpy())
    except ValueError as exc:
        warnings.append(f"ML objective placeholder unavailable: {exc}")

    return MLEstimationContext(
        observed_variables=tuple(available_columns),
        sample_covariance=sample_cov,
        objective_at_sample_cov=objective,
        warnings=tuple(warnings),
    )
