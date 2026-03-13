from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import minimize

if TYPE_CHECKING:
    from .fit import EFAConfig


@dataclass(frozen=True)
class _RotationOptimizationResult:
    params: NDArray[np.float64]
    objective_value: float
    converged: bool
    attempt_index: int


def _rotate_none(loadings: NDArray[np.float64], _: EFAConfig) -> NDArray[np.float64]:
    return loadings


def _rotate_varimax(loadings: NDArray[np.float64], _: EFAConfig) -> NDArray[np.float64]:
    return _varimax(loadings)


def _rotate_promax(
    loadings: NDArray[np.float64],
    _: EFAConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply Promax oblique rotation."""

    orthogonal = _varimax(loadings)
    power = 4.0
    target = np.sign(orthogonal) * np.abs(orthogonal) ** power

    transform = np.linalg.pinv(orthogonal) @ target
    pattern, phi = _compute_oblique_pattern_and_phi(orthogonal, transform)
    return pattern, phi


def _rotate_oblimin(
    loadings: NDArray[np.float64],
    config: EFAConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64], tuple[str, ...]]:
    """Apply a direct oblimin-family rotation using a quartimin-style criterion."""

    base = loadings.copy()
    n_factors = base.shape[1]

    def objective(offdiag_params: NDArray[np.float64]) -> float:
        transform = _offdiag_params_to_transform(offdiag_params, n_factors=n_factors)
        sign, logdet = np.linalg.slogdet(transform)
        if sign == 0 or not np.isfinite(logdet):
            return 1e12
        pattern, _ = _compute_oblique_pattern_and_phi(base, transform)
        row_sq = pattern * pattern
        criterion = 0.5 * np.sum(np.square(np.sum(row_sq, axis=1)) - np.sum(row_sq * row_sq, axis=1))
        penalty = 1e-4 * float(np.sum(offdiag_params * offdiag_params))
        return float(criterion + penalty)

    optimization = _optimize_oblique_rotation(
        objective=objective,
        config=config,
        n_factors=n_factors,
        criterion_name="Oblimin",
    )
    transform = _offdiag_params_to_transform(optimization.params, n_factors=n_factors)
    pattern, phi = _compute_oblique_pattern_and_phi(base, transform)
    warnings = _build_rotation_optimization_warnings(
        optimization=optimization,
        criterion_name="Oblimin",
    )
    return pattern, phi, warnings


def _rotate_geomin(
    loadings: NDArray[np.float64],
    config: EFAConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64], tuple[str, ...]]:
    """Apply an oblique geomin rotation.

    Geomin encourages approximate simple structure by shrinking most loadings in
    each row toward zero while allowing a small subset to remain salient.
    """

    base = _varimax(loadings)
    n_factors = base.shape[1]
    epsilon = 0.01

    def objective(offdiag_params: NDArray[np.float64]) -> float:
        transform = _offdiag_params_to_transform(offdiag_params, n_factors=n_factors)
        sign, logdet = np.linalg.slogdet(transform)
        if sign == 0 or not np.isfinite(logdet):
            return 1e12

        pattern, _ = _compute_oblique_pattern_and_phi(base, transform)
        row_geometric_mean = np.exp(np.mean(np.log(pattern * pattern + epsilon), axis=1))
        penalty = 1e-4 * float(np.sum(offdiag_params * offdiag_params))
        return float(np.sum(row_geometric_mean) + penalty)

    optimization = _optimize_oblique_rotation(
        objective=objective,
        config=config,
        n_factors=n_factors,
        criterion_name="Geomin",
    )
    transform = _offdiag_params_to_transform(optimization.params, n_factors=n_factors)
    pattern, phi = _compute_oblique_pattern_and_phi(base, transform)
    warnings = _build_rotation_optimization_warnings(
        optimization=optimization,
        criterion_name="Geomin",
    )
    return pattern, phi, warnings


def _rotate_target(
    loadings: NDArray[np.float64],
    config: EFAConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64], tuple[str, ...]]:
    """Apply an oblique target rotation.

    Finite cells in ``rotation_target`` are treated as target values to be
    approximated, while ``NaN`` cells are left free.
    """

    base = _varimax(loadings)
    _, n_factors = base.shape
    target = _coerce_rotation_target(
        config.rotation_target,
        item_names=list(config.items),
        n_factors=n_factors,
    )
    weights = _coerce_rotation_target_weights(
        config.rotation_target_weights,
        item_names=list(config.items),
        n_factors=n_factors,
        target=target,
    )
    mask = weights > 0.0

    def objective(offdiag_params: NDArray[np.float64]) -> float:
        transform = _offdiag_params_to_transform(offdiag_params, n_factors=n_factors)
        sign, logdet = np.linalg.slogdet(transform)
        if sign == 0 or not np.isfinite(logdet):
            return 1e12

        pattern, _ = _compute_oblique_pattern_and_phi(base, transform)
        diff = pattern[mask] - target[mask]
        penalty = 1e-4 * float(np.sum(offdiag_params * offdiag_params))
        return float(np.sum(weights[mask] * diff * diff) + penalty)

    optimization = _optimize_oblique_rotation(
        objective=objective,
        config=config,
        n_factors=n_factors,
        criterion_name="Target",
    )
    transform = _offdiag_params_to_transform(optimization.params, n_factors=n_factors)
    pattern, phi = _compute_oblique_pattern_and_phi(base, transform)
    warnings = _build_rotation_optimization_warnings(
        optimization=optimization,
        criterion_name="Target",
    )
    return pattern, phi, warnings


def _optimize_oblique_rotation(
    *,
    objective,
    config: EFAConfig,
    n_factors: int,
    criterion_name: str,
) -> _RotationOptimizationResult:
    starts = _build_rotation_start_vectors(
        n_factors=n_factors,
        restarts=config.rotation_restarts,
        random_state=config.random_state,
    )
    best_success: _RotationOptimizationResult | None = None
    best_any: _RotationOptimizationResult | None = None

    for attempt_index, start in enumerate(starts):
        optimization = minimize(
            objective,
            x0=start,
            method="BFGS",
            options={"maxiter": config.max_iter, "gtol": config.tol},
        )
        params = np.asarray(optimization.x if hasattr(optimization, "x") else start, dtype=float)
        value = float(objective(params))
        result = _RotationOptimizationResult(
            params=params,
            objective_value=value,
            converged=bool(optimization.success),
            attempt_index=attempt_index,
        )

        if np.isfinite(result.objective_value):
            if best_any is None or result.objective_value < best_any.objective_value:
                best_any = result
            if result.converged and (
                best_success is None or result.objective_value < best_success.objective_value
            ):
                best_success = result

    if best_success is not None:
        return best_success
    if best_any is not None:
        return best_any

    n_params = n_factors * n_factors - n_factors
    fallback = np.zeros(n_params, dtype=float)
    return _RotationOptimizationResult(
        params=fallback,
        objective_value=float("inf"),
        converged=False,
        attempt_index=0,
    )


def _build_rotation_start_vectors(
    *,
    n_factors: int,
    restarts: int,
    random_state: int | None,
) -> list[NDArray[np.float64]]:
    n_params = n_factors * n_factors - n_factors
    starts = [np.zeros(n_params, dtype=float)]
    if restarts <= 0:
        return starts

    rng = np.random.default_rng(random_state)
    for _ in range(restarts):
        starts.append(rng.normal(loc=0.0, scale=0.10, size=n_params))
    return starts


def _build_rotation_optimization_warnings(
    *,
    optimization: _RotationOptimizationResult,
    criterion_name: str,
) -> tuple[str, ...]:
    warnings: list[str] = []
    if optimization.converged and optimization.attempt_index > 0:
        warnings.append(
            f"{criterion_name} rotation converged after restart attempt #{optimization.attempt_index}."
        )
    if not optimization.converged:
        warnings.append(
            f"{criterion_name} rotation did not fully converge; returning best available transform."
        )
    return tuple(warnings)


def _coerce_rotation_target(
    target: pd.DataFrame | NDArray[np.float64] | None,
    *,
    item_names: list[str],
    n_factors: int,
) -> NDArray[np.float64]:
    """Validate and coerce a target-pattern matrix.

    Rows correspond to observed items and columns correspond to factors.
    Finite entries are targeted values; ``NaN`` entries are unconstrained.
    """

    if target is None:
        raise ValueError("`rotation_target` is required when rotation is `target`.")

    if isinstance(target, pd.DataFrame):
        if set(item_names).issubset(target.index):
            array = target.loc[item_names, :].to_numpy(dtype=float)
        else:
            array = target.to_numpy(dtype=float)
    else:
        array = np.asarray(target, dtype=float)

    expected_shape = (len(item_names), n_factors)
    if array.shape != expected_shape:
        raise ValueError(
            f"`rotation_target` must have shape {expected_shape}, got {array.shape}."
        )

    if not np.isfinite(array).any():
        raise ValueError("`rotation_target` must contain at least one finite target value.")

    return array


def _coerce_rotation_target_weights(
    weights: pd.DataFrame | NDArray[np.float64] | None,
    *,
    item_names: list[str],
    n_factors: int,
    target: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Validate and coerce target-rotation weights."""

    if weights is None:
        resolved = np.where(np.isfinite(target), 1.0, 0.0)
    elif isinstance(weights, pd.DataFrame):
        if set(item_names).issubset(weights.index):
            resolved = weights.loc[item_names, :].to_numpy(dtype=float)
        else:
            resolved = weights.to_numpy(dtype=float)
    else:
        resolved = np.asarray(weights, dtype=float)

    expected_shape = (len(item_names), n_factors)
    if resolved.shape != expected_shape:
        raise ValueError(
            f"`rotation_target_weights` must have shape {expected_shape}, got {resolved.shape}."
        )
    if np.any(~np.isfinite(resolved)):
        raise ValueError("`rotation_target_weights` must contain only finite values.")
    if np.any(resolved < 0.0):
        raise ValueError("`rotation_target_weights` must be >= 0.")

    resolved = np.where(np.isfinite(target), resolved, 0.0)
    if not np.any(resolved > 0.0):
        raise ValueError(
            "`rotation_target_weights` must contain at least one positive weight on a finite target cell."
        )
    return resolved


def _offdiag_params_to_transform(
    offdiag_params: NDArray[np.float64],
    *,
    n_factors: int,
) -> NDArray[np.float64]:
    """Build an oblique transformation matrix with fixed unit diagonal."""

    transform = np.eye(n_factors, dtype=float)
    mask = ~np.eye(n_factors, dtype=bool)
    transform[mask] = np.asarray(offdiag_params, dtype=float)
    return transform


def _compute_oblique_pattern_and_phi(
    base_loadings: NDArray[np.float64],
    transform: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert an oblique transformation matrix into pattern loadings and ``Phi``."""

    pattern = base_loadings @ transform
    phi = np.linalg.pinv(transform.T @ transform)
    phi = _stabilize_factor_correlation(phi)
    return pattern, phi


def _varimax(
    loadings: NDArray[np.float64],
    gamma: float = 1.0,
    q: int = 50,
    tol: float = 1e-6,
) -> NDArray[np.float64]:
    p, k = loadings.shape
    rotation = np.eye(k)
    previous = 0.0

    for _ in range(q):
        rotated = loadings @ rotation
        gram = rotated.T @ rotated
        diagonal = np.diag(np.diag(gram))
        transformed = loadings.T @ (rotated**3 - (gamma / p) * rotated @ diagonal)
        u, s, vh = np.linalg.svd(transformed)
        rotation = u @ vh
        current = np.sum(s)
        if previous > 0 and current - previous < tol:
            break
        previous = current

    return loadings @ rotation


def _stabilize_factor_correlation(factor_corr: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return a symmetric correlation-like factor matrix with unit diagonal."""

    phi = np.asarray(factor_corr, dtype=float)
    phi = np.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0)
    phi = (phi + phi.T) / 2.0
    diag = np.sqrt(np.clip(np.diag(phi), 1e-12, None))
    phi = phi / np.outer(diag, diag)
    phi = (phi + phi.T) / 2.0
    np.fill_diagonal(phi, 1.0)
    return phi