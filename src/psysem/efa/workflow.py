from __future__ import annotations

import pandas as pd

from .contracts import (
    EFADiagnosticsConfig,
    EFAWorkflowConfig,
    EFAWorkflowResult,
    FactorSelectionConfig,
)
from .diagnostics import run_efa_diagnostics
from .evaluation import evaluate_efa_model, evaluation_to_series
from .fit import EFAConfig, EFAResult, fit_efa
from .n_factors import suggest_n_factors


def run_efa_workflow(data: pd.DataFrame, config: EFAWorkflowConfig) -> EFAWorkflowResult:
    """Run diagnostics -> factor suggestion -> candidate fitting -> scoring."""
    items = _normalize_items(config.items)
    diagnostics_config = _merge_diagnostics_config(config.diagnostics, items)
    selection_config = _merge_selection_config(config.selection, items)

    diagnostics = run_efa_diagnostics(data, diagnostics_config)
    selection = suggest_n_factors(data, selection_config)
    candidates = _resolve_candidates(selection, config)

    candidate_results: dict[int, EFAResult] = {}
    candidate_evals = {}
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
        )
        efa_result = fit_efa(data, efa_config)
        eval_result = evaluate_efa_model(efa_result, config.evaluation)
        candidate_results[n_factors] = efa_result
        candidate_evals[n_factors] = eval_result
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
    return EFAWorkflowResult(
        diagnostics=diagnostics,
        selection=selection,
        candidate_results=candidate_results,
        candidate_evaluations=candidate_evals,
        comparison_table=comparison.reset_index(drop=True),
        best_n_factors=best_n_factors,
        best_model=candidate_results[best_n_factors],
        best_evaluation=candidate_evals[best_n_factors],
        warnings=tuple(dict.fromkeys(warning_list)),
    )


def _normalize_items(items: tuple[str, ...]) -> tuple[str, ...]:
    if not items:
        raise ValueError("`items` cannot be empty in EFA workflow.")
    return items


def _merge_diagnostics_config(
    config: EFADiagnosticsConfig,
    items: tuple[str, ...],
) -> EFADiagnosticsConfig:
    if config.items and config.items != items:
        raise ValueError("`diagnostics.items` must match workflow `items`.")
    return EFADiagnosticsConfig(
        items=items,
        dropna=config.dropna,
        min_sample_ratio=config.min_sample_ratio,
    )


def _merge_selection_config(
    config: FactorSelectionConfig,
    items: tuple[str, ...],
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
    )


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
