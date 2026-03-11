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

