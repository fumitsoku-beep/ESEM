from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..preprocessing import (
    SUPPORTED_CORRELATION_METHODS,
    SUPPORTED_MISSING_STRATEGIES,
    normalize_correlation_method,
    normalize_missing_strategy,
)
from .contracts import (
    EFADiagnosticsConfig,
    EFAWorkflowConfig,
    EFAWorkflowResult,
    FactorSelectionConfig,
)
from .diagnostics import run_efa_diagnostics
from .evaluation import evaluate_efa_model, evaluation_to_series
from .fit import EFAConfig, EFAResult, fit_efa
from .interpretation import interpret_efa
from .n_factors import suggest_n_factors


@dataclass(frozen=True)
class _WorkflowPreprocessingConfig:
    missing_strategy: str
    correlation_method: str
    variable_types: dict[str, str] | None


def run_efa_workflow(data: pd.DataFrame, config: EFAWorkflowConfig) -> EFAWorkflowResult:
    """Run diagnostics -> factor suggestion -> candidate fitting -> scoring."""
    items = _normalize_items(config.items)
    preprocessing = _resolve_workflow_preprocessing(config)
    diagnostics_config = _merge_diagnostics_config(config.diagnostics, items, preprocessing)
    selection_config = _merge_selection_config(config.selection, items, preprocessing)

    diagnostics = run_efa_diagnostics(data, diagnostics_config)
    selection = suggest_n_factors(data, selection_config)
    candidates = _resolve_candidates(selection, config)

    candidate_results: dict[int, EFAResult] = {}
    candidate_evals = {}
    candidate_interpretations = {}
    rows: list[pd.Series] = []
    warning_list = list(selection.warnings)
    warning_list.extend(diagnostics.warnings)

    for n_factors in candidates:
        efa_config = EFAConfig(
            items=items,
            n_factors=n_factors,
            extraction=config.extraction,
            rotation=config.rotation,
            max_iter=config.max_iter,
            tol=config.tol,
            min_uniqueness=config.min_uniqueness,
            missing_strategy=preprocessing.missing_strategy,
            correlation_method=preprocessing.correlation_method,
            variable_types=preprocessing.variable_types,
        )
        efa_result = fit_efa(data, efa_config)
        eval_result = evaluate_efa_model(efa_result, config.evaluation)
        candidate_results[n_factors] = efa_result
        candidate_evals[n_factors] = eval_result
        if config.include_interpretation:
            candidate_interpretations[n_factors] = interpret_efa(
                efa_result, config.interpretation
            )
        row = evaluation_to_series(eval_result)
        row["n_factors"] = n_factors
        rows.append(row)
        warning_list.extend(eval_result.warnings)

    comparison = pd.DataFrame(rows).sort_values(
        by=["score", "n_factors"],
        ascending=[False, True],
        kind="stable",
    )
    if comparison.empty:
        raise ValueError("No candidate models were fitted.")

    best_n_factors = int(comparison.iloc[0]["n_factors"])
    best_interpretation = candidate_interpretations.get(best_n_factors)
    return EFAWorkflowResult(
        diagnostics=diagnostics,
        selection=selection,
        candidate_results=candidate_results,
        candidate_evaluations=candidate_evals,
        candidate_interpretations=candidate_interpretations,
        comparison_table=comparison.reset_index(drop=True),
        best_n_factors=best_n_factors,
        best_model=candidate_results[best_n_factors],
        best_evaluation=candidate_evals[best_n_factors],
        best_interpretation=best_interpretation,
        warnings=tuple(dict.fromkeys(warning_list)),
    )


def _normalize_items(items: tuple[str, ...]) -> tuple[str, ...]:
    if not items:
        raise ValueError("`items` cannot be empty in EFA workflow.")
    return items


def _merge_diagnostics_config(
    config: EFADiagnosticsConfig,
    items: tuple[str, ...],
    preprocessing: _WorkflowPreprocessingConfig,
) -> EFADiagnosticsConfig:
    if config.items and config.items != items:
        raise ValueError("`diagnostics.items` must match workflow `items`.")
    return EFADiagnosticsConfig(
        items=items,
        dropna=config.dropna,
        min_sample_ratio=config.min_sample_ratio,
        missing_strategy=preprocessing.missing_strategy,
        correlation_method=preprocessing.correlation_method,
        variable_types=preprocessing.variable_types,
    )


def _merge_selection_config(
    config: FactorSelectionConfig,
    items: tuple[str, ...],
    preprocessing: _WorkflowPreprocessingConfig,
) -> FactorSelectionConfig:
    if config.items and config.items != items:
        raise ValueError("`selection.items` must match workflow `items`.")
    return FactorSelectionConfig(
        items=items,
        n_min=config.n_min,
        n_max=config.n_max,
        pa_iter=config.pa_iter,
        pa_percentile=config.pa_percentile,
        random_state=config.random_state,
        enable_pa=config.enable_pa,
        enable_map=config.enable_map,
        enable_kaiser=config.enable_kaiser,
        enable_scree=config.enable_scree,
        dropna=config.dropna,
        consensus_strategy=config.consensus_strategy,
        consensus_weights=config.consensus_weights,
        missing_strategy=preprocessing.missing_strategy,
        correlation_method=preprocessing.correlation_method,
        variable_types=preprocessing.variable_types,
    )


def _resolve_workflow_preprocessing(config: EFAWorkflowConfig) -> _WorkflowPreprocessingConfig:
    missing_strategy = _resolve_workflow_missing_strategy(config)
    correlation_method = _resolve_workflow_correlation_method(config)
    variable_types = _resolve_workflow_variable_types(config)
    return _WorkflowPreprocessingConfig(
        missing_strategy=missing_strategy,
        correlation_method=correlation_method,
        variable_types=variable_types,
    )


def _resolve_workflow_missing_strategy(config: EFAWorkflowConfig) -> str:
    workflow_strategy = _normalize_optional_missing_strategy(config.missing_strategy)
    diagnostics_strategy, diagnostics_explicit = _resolve_subconfig_missing_strategy(config.diagnostics)
    selection_strategy, selection_explicit = _resolve_subconfig_missing_strategy(config.selection)

    if workflow_strategy is not None:
        if diagnostics_explicit and diagnostics_strategy != workflow_strategy:
            raise ValueError(
                "`diagnostics.missing_strategy` must match workflow preprocessing configuration."
            )
        if selection_explicit and selection_strategy != workflow_strategy:
            raise ValueError(
                "`selection.missing_strategy` must match workflow preprocessing configuration."
            )
        return workflow_strategy

    if diagnostics_strategy != selection_strategy:
        raise ValueError(
            "Workflow preprocessing is inconsistent: diagnostics and selection resolve to "
            "different missing-data strategies."
        )
    return diagnostics_strategy


def _resolve_workflow_correlation_method(config: EFAWorkflowConfig) -> str:
    workflow_method = _normalize_optional_correlation_method(config.correlation_method)
    diagnostics_method, diagnostics_explicit = _resolve_subconfig_correlation_method(
        config.diagnostics
    )
    selection_method, selection_explicit = _resolve_subconfig_correlation_method(config.selection)

    if workflow_method is not None:
        if diagnostics_explicit and diagnostics_method != workflow_method:
            raise ValueError(
                "`diagnostics.correlation_method` must match workflow preprocessing configuration."
            )
        if selection_explicit and selection_method != workflow_method:
            raise ValueError(
                "`selection.correlation_method` must match workflow preprocessing configuration."
            )
        return workflow_method

    if diagnostics_method != selection_method:
        raise ValueError(
            "Workflow preprocessing is inconsistent: diagnostics and selection resolve to "
            "different correlation methods."
        )
    return diagnostics_method


def _resolve_workflow_variable_types(
    config: EFAWorkflowConfig,
) -> dict[str, str] | None:
    workflow_variable_types = config.variable_types
    diagnostics_variable_types = config.diagnostics.variable_types
    selection_variable_types = config.selection.variable_types

    if workflow_variable_types is not None:
        if (
            diagnostics_variable_types is not None
            and diagnostics_variable_types != workflow_variable_types
        ):
            raise ValueError("`diagnostics.variable_types` must match workflow preprocessing config.")
        if selection_variable_types is not None and selection_variable_types != workflow_variable_types:
            raise ValueError("`selection.variable_types` must match workflow preprocessing config.")
        return dict(workflow_variable_types)

    if (
        diagnostics_variable_types is not None
        and selection_variable_types is not None
        and diagnostics_variable_types != selection_variable_types
    ):
        raise ValueError(
            "Workflow preprocessing is inconsistent: diagnostics and selection define "
            "different `variable_types`."
        )
    if diagnostics_variable_types is not None:
        return dict(diagnostics_variable_types)
    if selection_variable_types is not None:
        return dict(selection_variable_types)
    return None


def _resolve_subconfig_missing_strategy(
    config: EFADiagnosticsConfig | FactorSelectionConfig,
) -> tuple[str, bool]:
    if config.missing_strategy is not None:
        return _normalize_required_missing_strategy(config.missing_strategy), True
    return ("dropna" if config.dropna else "pairwise"), False


def _resolve_subconfig_correlation_method(
    config: EFADiagnosticsConfig | FactorSelectionConfig,
) -> tuple[str, bool]:
    if config.correlation_method is not None:
        return _normalize_required_correlation_method(config.correlation_method), True
    return "pearson", False


def _normalize_optional_missing_strategy(strategy: str | None) -> str | None:
    if strategy is None:
        return None
    return _normalize_required_missing_strategy(strategy)


def _normalize_required_missing_strategy(strategy: str) -> str:
    normalized = normalize_missing_strategy(strategy)
    if normalized not in SUPPORTED_MISSING_STRATEGIES:
        raise ValueError("`missing_strategy` must be one of: pairwise, dropna.")
    return normalized


def _normalize_optional_correlation_method(method: str | None) -> str | None:
    if method is None:
        return None
    return _normalize_required_correlation_method(method)


def _normalize_required_correlation_method(method: str) -> str:
    normalized = normalize_correlation_method(method)
    if normalized not in SUPPORTED_CORRELATION_METHODS:
        raise ValueError("`correlation_method` must be one of: pearson, spearman, polychoric.")
    return normalized


def _resolve_candidates(selection, config: EFAWorkflowConfig) -> list[int]:
    if config.candidate_strategy not in {"selection_union", "range"}:
        raise ValueError(
            f"Unsupported candidate strategy `{config.candidate_strategy}`. "
            "Use `selection_union` or `range`."
        )

    if config.candidate_strategy == "range":
        candidates = list(range(selection.n_min, selection.n_max + 1))
    else:
        candidates = sorted(set(selection.suggestions_by_method.values()))
        if config.include_consensus:
            candidates = sorted(set(candidates + [selection.consensus_n_factors]))

    if config.manual_candidates:
        for value in config.manual_candidates:
            if not isinstance(value, int):
                raise ValueError("`manual_candidates` must contain integers.")
            if value < selection.n_min or value > selection.n_max:
                raise ValueError(
                    f"Manual candidate `{value}` is outside [{selection.n_min}, {selection.n_max}]."
                )
        candidates = sorted(set(candidates + list(config.manual_candidates)))

    if not candidates:
        candidates = [selection.consensus_n_factors]
    return candidates
