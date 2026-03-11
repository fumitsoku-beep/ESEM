from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class MeasurementDesign:
    """Measurement-layer matrix representation for SEM."""

    observed_variables: tuple[str, ...]
    latent_variables: tuple[str, ...]
    lambda_matrix: pd.DataFrame
    theta_matrix: pd.DataFrame
    free_loadings: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    fixed_loadings: tuple[tuple[str, str, float], ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)
