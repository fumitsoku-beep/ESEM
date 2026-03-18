from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

from ..sem.estimation import SEMFitConfig

if TYPE_CHECKING:
    from ..data import ESEMSpec
    from ..efa import EFAResult
    from ..sem.model import ModelSpec
    from ..sem.result import SEMResult


@dataclass(frozen=True)
class ESEMWorkflowConfig:
    """Config for the first runnable ESEM workflow."""

    generator_strategies: tuple[str, ...] = ("block_full",)
    enabled_judges: tuple[str, ...] = ("convergence", "fit_indices", "efa_bridge")
    selector_strategy: str = "best_score"
    include_sem_fit: bool = True
    include_efa_bridge: bool = True
    keep_all_candidates: bool = True
    judge_weights: dict[str, float] | None = None
    selector_weights: dict[str, float] | None = None
    fit_config: SEMFitConfig | None = None
    require_ml_estimator: bool = True
    efa_extraction: str = "paf"
    efa_rotation: str = "varimax"
    efa_max_iter: int = 200
    efa_tol: float = 1e-6
    efa_min_uniqueness: float = 0.005
    efa_missing_strategy: str = "pairwise"
    efa_correlation_method: str | None = None


@dataclass
class ESEMJudgeResult:
    """One judge output for one candidate."""

    judge: str
    score: float
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class ESEMCandidateResult:
    """One generated ESEM candidate with evaluation outputs."""

    candidate_id: str
    strategy: str
    model_spec: ModelSpec
    sem_result: SEMResult | None
    block_efa_results: dict[str, "EFAResult"] = field(default_factory=dict)
    judge_results: dict[str, ESEMJudgeResult] = field(default_factory=dict)
    total_score: float = float("nan")
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class ESEMWorkflowResult:
    """Top-level output bundle from :func:`run_esem_workflow`."""

    input_spec: "ESEMSpec"
    candidates: dict[str, ESEMCandidateResult]
    comparison_table: pd.DataFrame
    best_candidate_id: str
    best_candidate: ESEMCandidateResult
    warnings: tuple[str, ...] = field(default_factory=tuple)
