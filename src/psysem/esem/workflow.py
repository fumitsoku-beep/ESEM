from __future__ import annotations

import math
from typing import Any, Mapping

import pandas as pd

from ..core import SEMModel
from ..data import ESEMSpec, esem_spec_from_dict, validate_esem_spec
from ..efa import EFAConfig, EFAResult, fit_efa
from ..model import model_spec_from_esem_spec
from ..preprocessing import (
    SUPPORTED_CORRELATION_METHODS,
    SUPPORTED_MISSING_STRATEGIES,
    normalize_correlation_method,
    normalize_missing_strategy,
)
from ..result import SEMResult
from .contracts import (
    ESEMCandidateResult,
    ESEMJudgeResult,
    ESEMWorkflowConfig,
    ESEMWorkflowResult,
)

_SUPPORTED_GENERATORS = frozenset({"block_full"})
_SUPPORTED_JUDGES = frozenset({"convergence", "fit_indices", "efa_bridge"})
_SUPPORTED_SELECTOR_STRATEGIES = frozenset({"best_score"})


def run_esem_workflow(
    data: pd.DataFrame,
    spec: ESEMSpec | Mapping[str, Any],
    config: ESEMWorkflowConfig,
) -> ESEMWorkflowResult:
    """Run the first usable end-to-end ESEM workflow.

    Current implementation intentionally keeps one candidate strategy:
    ``block_full``. It still provides a stable workflow entry that can run
    one complete ESEM-like pass:
    spec validation -> (optional) EFA bridge -> SEM fitting -> scoring.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("`data` must be a pandas.DataFrame.")
    if not isinstance(config, ESEMWorkflowConfig):
        raise TypeError("`config` must be an ESEMWorkflowConfig instance.")

    _validate_config(config)
    resolved_spec = _coerce_spec(spec)
    validate_esem_spec(resolved_spec, data)

    if (
        config.include_sem_fit
        and config.require_ml_estimator
        and resolved_spec.estimator.lower() not in {"ml", "mlr"}
    ):
        raise ValueError(
            "Current `run_esem_workflow` requires `spec.estimator` in {ML, MLR} "
            "to run the SEM optimization path."
        )

    candidates: dict[str, ESEMCandidateResult] = {}
    workflow_warnings: list[str] = []
    rows: list[dict[str, Any]] = []

    for strategy in config.generator_strategies:
        candidate = _run_single_candidate(
            data=data,
            spec=resolved_spec,
            config=config,
            strategy=strategy,
        )
        candidates[candidate.candidate_id] = candidate
        workflow_warnings.extend(candidate.warnings)
        rows.append(_candidate_to_row(candidate))

    comparison = pd.DataFrame(rows).sort_values(
        by=["total_score", "candidate_id"],
        ascending=[False, True],
        kind="stable",
    )
    if comparison.empty:
        raise ValueError("No ESEM candidates were produced.")

    best_candidate_id = str(comparison.iloc[0]["candidate_id"])
    best_candidate = candidates[best_candidate_id]
    return ESEMWorkflowResult(
        input_spec=resolved_spec,
        candidates=candidates,
        comparison_table=comparison.reset_index(drop=True),
        best_candidate_id=best_candidate_id,
        best_candidate=best_candidate,
        warnings=tuple(dict.fromkeys(workflow_warnings)),
    )


def _run_single_candidate(
    *,
    data: pd.DataFrame,
    spec: ESEMSpec,
    config: ESEMWorkflowConfig,
    strategy: str,
) -> ESEMCandidateResult:
    if strategy != "block_full":
        raise ValueError(
            f"Unsupported generator strategy `{strategy}`. "
            "Currently only `block_full` is implemented."
        )

    candidate_id = strategy
    model_spec = model_spec_from_esem_spec(spec)

    candidate_warnings: list[str] = []
    block_efa_results: dict[str, EFAResult] = {}
    judge_results: dict[str, ESEMJudgeResult] = {}
    sem_result: SEMResult | None = None

    if config.include_efa_bridge:
        block_efa_results, bridge_warnings = _run_block_efa_bridge(data=data, spec=spec, config=config)
        candidate_warnings.extend(bridge_warnings)

    if config.include_sem_fit:
        sem_result = SEMModel().fit(data, spec=spec, fit_config=config.fit_config)
        candidate_warnings.extend(sem_result.warnings)

    for judge in config.enabled_judges:
        judge_result = _run_judge(
            judge=judge,
            sem_result=sem_result,
            block_efa_results=block_efa_results,
        )
        judge_results[judge] = judge_result
        candidate_warnings.extend(judge_result.warnings)

    total_score = _aggregate_score(judge_results, config.judge_weights)
    return ESEMCandidateResult(
        candidate_id=candidate_id,
        strategy=strategy,
        model_spec=model_spec,
        sem_result=sem_result,
        block_efa_results=block_efa_results,
        judge_results=judge_results,
        total_score=total_score,
        warnings=tuple(dict.fromkeys(candidate_warnings)),
    )


def _run_block_efa_bridge(
    *,
    data: pd.DataFrame,
    spec: ESEMSpec,
    config: ESEMWorkflowConfig,
) -> tuple[dict[str, EFAResult], list[str]]:
    results: dict[str, EFAResult] = {}
    warnings: list[str] = []
    for block in spec.blocks:
        block_variable_types = {
            item: spec.variable_types[item]
            for item in block.items
            if item in spec.variable_types
        }
        correlation_method = _resolve_block_efa_correlation_method(
            block_variable_types=block_variable_types,
            configured_method=config.efa_correlation_method,
        )
        efa_result = fit_efa(
            data,
            EFAConfig(
                items=block.items,
                n_factors=block.n_factors,
                extraction=config.efa_extraction,
                rotation=config.efa_rotation,
                max_iter=config.efa_max_iter,
                tol=config.efa_tol,
                min_uniqueness=config.efa_min_uniqueness,
                missing_strategy=normalize_missing_strategy(config.efa_missing_strategy),
                correlation_method=correlation_method,
                variable_types=block_variable_types,
            ),
        )
        results[block.name] = efa_result
        if efa_result.warnings:
            warnings.extend(
                [f"[{block.name}] {message}" for message in efa_result.warnings]
            )
    return results, warnings


def _run_judge(
    *,
    judge: str,
    sem_result: SEMResult | None,
    block_efa_results: dict[str, EFAResult],
) -> ESEMJudgeResult:
    if judge == "convergence":
        return _judge_convergence(sem_result)
    if judge == "fit_indices":
        return _judge_fit_indices(sem_result)
    if judge == "efa_bridge":
        return _judge_efa_bridge(block_efa_results)
    raise ValueError(
        f"Unsupported judge `{judge}`. Available: {', '.join(sorted(_SUPPORTED_JUDGES))}."
    )


def _judge_convergence(sem_result: SEMResult | None) -> ESEMJudgeResult:
    if sem_result is None:
        return ESEMJudgeResult(
            judge="convergence",
            score=0.0,
            passed=False,
            details={"reason": "sem_fit_disabled"},
            warnings=("Convergence judge skipped: SEM fit is disabled.",),
        )
    score = 1.0 if sem_result.converged else -1.0
    details = {"converged": sem_result.converged}
    warnings = () if sem_result.converged else ("SEM did not converge.",)
    return ESEMJudgeResult(
        judge="convergence",
        score=score,
        passed=sem_result.converged,
        details=details,
        warnings=warnings,
    )


def _judge_fit_indices(sem_result: SEMResult | None) -> ESEMJudgeResult:
    if sem_result is None:
        return ESEMJudgeResult(
            judge="fit_indices",
            score=0.0,
            passed=False,
            details={"reason": "sem_fit_disabled"},
            warnings=("Fit-indices judge skipped: SEM fit is disabled.",),
        )

    fit = sem_result.fit_indices or {}
    cfi = _finite_or_default(fit.get("cfi"), 0.0)
    tli = _finite_or_default(fit.get("tli"), 0.0)
    rmsea = _finite_or_default(fit.get("rmsea"), 1.0)
    srmr = _finite_or_default(fit.get("srmr"), 1.0)

    # Keep a simple bounded score for ranking prototype candidates.
    score = cfi + 0.5 * tli - rmsea - srmr
    passed = sem_result.converged and cfi >= 0.90 and rmsea <= 0.08 and srmr <= 0.08
    warnings: list[str] = []
    if not fit:
        warnings.append("Fit indices are empty.")
    elif any(key not in fit for key in ("cfi", "tli", "rmsea", "srmr")):
        warnings.append("Fit indices are partial; expected CFI/TLI/RMSEA/SRMR.")

    return ESEMJudgeResult(
        judge="fit_indices",
        score=float(score),
        passed=bool(passed),
        details={
            "cfi": cfi,
            "tli": tli,
            "rmsea": rmsea,
            "srmr": srmr,
        },
        warnings=tuple(warnings),
    )


def _judge_efa_bridge(block_efa_results: dict[str, EFAResult]) -> ESEMJudgeResult:
    if not block_efa_results:
        return ESEMJudgeResult(
            judge="efa_bridge",
            score=0.0,
            passed=False,
            details={"reason": "efa_bridge_disabled_or_empty"},
            warnings=("EFA bridge judge skipped: no block EFA result available.",),
        )

    all_h2: list[float] = []
    cross_loaded = 0
    rmsr_values: list[float] = []
    for result in block_efa_results.values():
        all_h2.extend([float(value) for value in result.communalities.values])
        cross_loaded += len(result.cross_loaded_items)
        rmsr_values.append(float(result.residual_summary.get("rmsr", 0.0)))

    mean_h2 = float(sum(all_h2) / len(all_h2)) if all_h2 else 0.0
    mean_rmsr = float(sum(rmsr_values) / len(rmsr_values)) if rmsr_values else 0.0
    score = mean_h2 - 0.05 * float(cross_loaded) - mean_rmsr
    passed = mean_h2 >= 0.20 and mean_rmsr <= 0.08

    warnings: list[str] = []
    if mean_h2 < 0.20:
        warnings.append("EFA bridge shows low average communality (< 0.20).")
    if mean_rmsr > 0.08:
        warnings.append("EFA bridge shows relatively high residual RMSR (> 0.08).")
    if cross_loaded > 0:
        warnings.append(f"EFA bridge detected {cross_loaded} cross-loaded items.")

    return ESEMJudgeResult(
        judge="efa_bridge",
        score=float(score),
        passed=passed,
        details={
            "mean_h2": mean_h2,
            "mean_rmsr": mean_rmsr,
            "cross_loaded_items": cross_loaded,
        },
        warnings=tuple(warnings),
    )


def _aggregate_score(
    judge_results: dict[str, ESEMJudgeResult],
    judge_weights: dict[str, float] | None,
) -> float:
    if not judge_results:
        return float("nan")
    weights = judge_weights or {}
    total = 0.0
    for name, result in judge_results.items():
        weight = float(weights.get(name, 1.0))
        total += weight * result.score
    return float(total)


def _coerce_spec(spec: ESEMSpec | Mapping[str, Any]) -> ESEMSpec:
    if isinstance(spec, ESEMSpec):
        return spec
    if isinstance(spec, Mapping):
        return esem_spec_from_dict(spec)
    raise TypeError("`spec` must be ESEMSpec or mapping payload.")


def _validate_config(config: ESEMWorkflowConfig) -> None:
    if not config.generator_strategies:
        raise ValueError("`generator_strategies` cannot be empty.")
    unknown_generators = [
        name for name in config.generator_strategies if name not in _SUPPORTED_GENERATORS
    ]
    if unknown_generators:
        raise ValueError(
            "Unsupported generator strategies: "
            + ", ".join(unknown_generators)
            + ". Available: "
            + ", ".join(sorted(_SUPPORTED_GENERATORS))
            + "."
        )

    if not config.enabled_judges:
        raise ValueError("`enabled_judges` cannot be empty.")
    unknown_judges = [name for name in config.enabled_judges if name not in _SUPPORTED_JUDGES]
    if unknown_judges:
        raise ValueError(
            "Unsupported judges: "
            + ", ".join(unknown_judges)
            + ". Available: "
            + ", ".join(sorted(_SUPPORTED_JUDGES))
            + "."
        )

    if config.selector_strategy not in _SUPPORTED_SELECTOR_STRATEGIES:
        raise ValueError(
            f"Unsupported `selector_strategy` `{config.selector_strategy}`. "
            "Available: best_score."
        )
    if config.efa_max_iter <= 0:
        raise ValueError("`efa_max_iter` must be > 0.")
    if config.efa_tol <= 0:
        raise ValueError("`efa_tol` must be > 0.")
    if not (0.0 < config.efa_min_uniqueness < 1.0):
        raise ValueError("`efa_min_uniqueness` must be between 0 and 1.")
    missing_strategy = normalize_missing_strategy(config.efa_missing_strategy)
    if missing_strategy not in SUPPORTED_MISSING_STRATEGIES:
        raise ValueError("`efa_missing_strategy` must be one of: pairwise, dropna.")
    if config.efa_correlation_method is not None:
        correlation_method = normalize_correlation_method(config.efa_correlation_method)
        if correlation_method not in SUPPORTED_CORRELATION_METHODS:
            raise ValueError(
                "`efa_correlation_method` must be one of: pearson, spearman, polychoric."
            )
    if config.judge_weights is not None:
        for name, value in config.judge_weights.items():
            if name not in _SUPPORTED_JUDGES:
                raise ValueError(f"Unknown judge weight key `{name}`.")
            if float(value) <= 0:
                raise ValueError(f"Judge weight for `{name}` must be > 0.")


def _resolve_block_efa_correlation_method(
    *,
    block_variable_types: dict[str, str],
    configured_method: str | None,
) -> str:
    if configured_method is not None:
        return normalize_correlation_method(configured_method)

    block_types = {kind.strip().lower() for kind in block_variable_types.values()}
    if block_types and block_types == {"ordinal"}:
        return "polychoric"
    if "ordinal" in block_types:
        return "spearman"
    return "pearson"


def _candidate_to_row(candidate: ESEMCandidateResult) -> dict[str, Any]:
    sem = candidate.sem_result
    fit = sem.fit_indices if sem is not None else {}
    converged = sem.converged if sem is not None else False
    return {
        "candidate_id": candidate.candidate_id,
        "strategy": candidate.strategy,
        "converged": converged,
        "total_score": candidate.total_score,
        "cfi": _finite_or_default(fit.get("cfi"), float("nan")),
        "tli": _finite_or_default(fit.get("tli"), float("nan")),
        "rmsea": _finite_or_default(fit.get("rmsea"), float("nan")),
        "srmr": _finite_or_default(fit.get("srmr"), float("nan")),
        "aic": _finite_or_default(fit.get("aic"), float("nan")),
        "bic": _finite_or_default(fit.get("bic"), float("nan")),
        "n_warnings": len(candidate.warnings),
    }


def _finite_or_default(value: Any, default: float) -> float:
    if not isinstance(value, (int, float)):
        return float(default)
    if not math.isfinite(float(value)):
        return float(default)
    return float(value)
