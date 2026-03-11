from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class MLEstimationContext:
    """Prepared input state for ML estimation."""

    observed_variables: tuple[str, ...]
    sample_covariance: pd.DataFrame | None
    objective_at_sample_cov: float | None
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class MLOptimizationResult:
    """Output bundle for ML prototype optimization."""

    success: bool
    status: str
    n_iter: int
    objective: float | None
    observed_variables: tuple[str, ...]
    parameter_vector: tuple[float, ...]
    parameter_values: dict[str, float] = field(default_factory=dict)
    sample_covariance: pd.DataFrame | None = None
    implied_covariance: pd.DataFrame | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)
