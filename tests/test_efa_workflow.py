import numpy as np
import pandas as pd
import pytest

from psysem import (
    EFADiagnosticsConfig,
    EFAEvaluationConfig,
    EFAWorkflowConfig,
    FactorSelectionConfig,
    run_efa_workflow,
)


def _synthetic_efa_data(n: int = 500, seed: int = 99) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(n, 2))
    loadings = np.array(
        [
            [0.86, 0.06],
            [0.82, 0.11],
            [0.79, 0.09],
            [0.07, 0.84],
            [0.12, 0.80],
            [0.09, 0.77],
        ]
    )
    noise = rng.normal(scale=0.38, size=(n, 6))
    observed = factors @ loadings.T + noise
    return pd.DataFrame(observed, columns=[f"i{i}" for i in range(1, 7)])


def test_run_efa_workflow_selection_union_smoke() -> None:
    data = _synthetic_efa_data()
    config = EFAWorkflowConfig(
        items=("i1", "i2", "i3", "i4", "i5", "i6"),
        diagnostics=EFADiagnosticsConfig(items=()),
        selection=FactorSelectionConfig(items=(), n_min=1, n_max=4, pa_iter=120, random_state=42),
        evaluation=EFAEvaluationConfig(),
        extraction="paf",
        rotation="varimax",
        candidate_strategy="selection_union",
        include_consensus=True,
    )
    result = run_efa_workflow(data, config)

    assert result.best_n_factors in result.candidate_results
    assert result.best_model.loadings.shape[1] == result.best_n_factors
    assert not result.comparison_table.empty
    assert "score" in result.comparison_table.columns
    assert set(result.candidate_results) == set(result.candidate_evaluations)


def test_run_efa_workflow_range_strategy_uses_full_range() -> None:
    data = _synthetic_efa_data()
    config = EFAWorkflowConfig(
        items=("i1", "i2", "i3", "i4", "i5", "i6"),
        diagnostics=EFADiagnosticsConfig(items=()),
        selection=FactorSelectionConfig(items=(), n_min=1, n_max=3, pa_iter=80, random_state=42),
        evaluation=EFAEvaluationConfig(),
        candidate_strategy="range",
    )
    result = run_efa_workflow(data, config)
    assert set(result.candidate_results) == {1, 2, 3}


def test_run_efa_workflow_rejects_invalid_candidate_strategy() -> None:
    data = _synthetic_efa_data()
    config = EFAWorkflowConfig(
        items=("i1", "i2", "i3", "i4", "i5", "i6"),
        diagnostics=EFADiagnosticsConfig(items=()),
        selection=FactorSelectionConfig(items=(), n_min=1, n_max=3),
        evaluation=EFAEvaluationConfig(),
        candidate_strategy="unknown",
    )
    with pytest.raises(ValueError, match="Unsupported candidate strategy"):
        run_efa_workflow(data, config)


def test_run_efa_workflow_rejects_out_of_range_manual_candidate() -> None:
    data = _synthetic_efa_data()
    config = EFAWorkflowConfig(
        items=("i1", "i2", "i3", "i4", "i5", "i6"),
        diagnostics=EFADiagnosticsConfig(items=()),
        selection=FactorSelectionConfig(items=(), n_min=1, n_max=3),
        evaluation=EFAEvaluationConfig(),
        manual_candidates=(4,),
    )
    with pytest.raises(ValueError, match="outside"):
        run_efa_workflow(data, config)
