from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class LoadingParameter:
    """One loading entry with parameter metadata for optimization mapping."""

    observed: str
    latent: str
    is_free: bool
    parameter: str | None
    parameter_index: int | None
    fixed_value: float | None
    relation_index: int
    term_index: int
    block_name: str | None = None


@dataclass(frozen=True)
class MeasurementDesign:
    """Measurement-layer matrix representation for SEM."""

    observed_variables: tuple[str, ...]
    latent_variables: tuple[str, ...]
    lambda_matrix: pd.DataFrame
    theta_matrix: pd.DataFrame
    loading_parameters: tuple[LoadingParameter, ...] = field(default_factory=tuple)
    block_latent_pairs: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    free_loadings: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    fixed_loadings: tuple[tuple[str, str, float], ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)
