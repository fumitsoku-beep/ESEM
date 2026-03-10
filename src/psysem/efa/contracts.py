from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class EFADiagnosticsConfig:
    items: tuple[str, ...]
    dropna: bool = True
    min_sample_ratio: float = 5.0


@dataclass
class EFADiagnosticsResult:
    items: tuple[str, ...]
    n_obs: int
    n_items: int
    sample_ratio: float
    kmo_total: float
    kmo_per_item: pd.Series
    kmo_label: str
    bartlett_chi2: float
    bartlett_df: int
    bartlett_p: float
    warnings: tuple[str, ...] = field(default_factory=tuple)
    correlation_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass(frozen=True)
class FactorSelectionConfig:
    items: tuple[str, ...]
    n_min: int = 1
    n_max: int | None = None
    pa_iter: int = 500
    pa_percentile: float = 0.95
    random_state: int | None = None
    enable_pa: bool = True
    enable_map: bool = True
    enable_kaiser: bool = True
    enable_scree: bool = True
    dropna: bool = True
    consensus_strategy: str = "majority_min_tie"


@dataclass
class FactorSelectionResult:
    items: tuple[str, ...]
    n_obs: int
    n_items: int
    n_min: int
    n_max: int
    eigenvalues: pd.Series
    scree_elbow: int | None
    parallel_thresholds: pd.Series | None
    map_values: pd.Series | None
    suggestions_by_method: dict[str, int]
    consensus_n_factors: int
    warnings: tuple[str, ...] = field(default_factory=tuple)
    correlation_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
