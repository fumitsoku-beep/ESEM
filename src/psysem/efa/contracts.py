from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from .fit import EFAResult


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


@dataclass(frozen=True)
class EFAEvaluationConfig:
    salient_loading: float = 0.30
    cross_loading: float = 0.30
    min_h2: float = 0.20
    variance_weight: float = 1.00
    simplicity_weight: float = 0.75
    communality_weight: float = 0.50
    cross_loading_penalty: float = 1.00
    factor_balance_penalty: float = 0.25


@dataclass
class EFAEvaluationResult:
    n_factors: int
    score: float
    explained_total: float
    simple_structure_ratio: float
    mean_h2: float
    mean_max_loading: float
    cross_loaded_items: int
    low_h2_items: int
    salient_items: int
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class EFAWorkflowConfig:
    items: tuple[str, ...]
    selection: FactorSelectionConfig = field(default_factory=lambda: FactorSelectionConfig(items=()))
    diagnostics: EFADiagnosticsConfig = field(default_factory=lambda: EFADiagnosticsConfig(items=()))
    evaluation: EFAEvaluationConfig = field(default_factory=EFAEvaluationConfig)
    extraction: str = "paf"
    rotation: str = "varimax"
    max_iter: int = 200
    tol: float = 1e-6
    min_uniqueness: float = 0.005
    candidate_strategy: str = "selection_union"
    include_consensus: bool = True
    manual_candidates: tuple[int, ...] = ()


@dataclass
class EFAWorkflowResult:
    diagnostics: EFADiagnosticsResult
    selection: FactorSelectionResult
    candidate_results: dict[int, "EFAResult"]
    candidate_evaluations: dict[int, EFAEvaluationResult]
    comparison_table: pd.DataFrame
    best_n_factors: int
    best_model: "EFAResult"
    best_evaluation: EFAEvaluationResult
    warnings: tuple[str, ...] = field(default_factory=tuple)
