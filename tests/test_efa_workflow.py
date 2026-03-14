import numpy as np
import pandas as pd
import pytest

from psysem import (
    EFADiagnosticsConfig,
    EFAEvaluationConfig,
    EFAInterpretationConfig,
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


def _ordinal_efa_data(n: int = 500, seed: int = 99) -> pd.DataFrame:
    continuous = _synthetic_efa_data(n=n, seed=seed)
    ordinal = {
        column: pd.qcut(continuous[column], q=5, labels=False, duplicates="drop") + 1
        for column in continuous.columns
    }
    return pd.DataFrame(ordinal)


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
    assert set(result.candidate_results) == set(result.candidate_interpretations)
    assert result.best_interpretation is not None
    assert not result.best_interpretation.item_table.empty
    assert not result.best_interpretation.factor_table.empty


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


def test_run_efa_workflow_can_disable_interpretation() -> None:
    data = _synthetic_efa_data()
    config = EFAWorkflowConfig(
        items=("i1", "i2", "i3", "i4", "i5", "i6"),
        diagnostics=EFADiagnosticsConfig(items=()),
        selection=FactorSelectionConfig(items=(), n_min=1, n_max=3, pa_iter=80, random_state=42),
        evaluation=EFAEvaluationConfig(),
        include_interpretation=False,
    )
    result = run_efa_workflow(data, config)
    assert result.candidate_interpretations == {}
    assert result.best_interpretation is None


def test_run_efa_workflow_rejects_non_integer_manual_candidate() -> None:
    data = _synthetic_efa_data()
    config = EFAWorkflowConfig(
        items=("i1", "i2", "i3", "i4", "i5", "i6"),
        diagnostics=EFADiagnosticsConfig(items=()),
        selection=FactorSelectionConfig(items=(), n_min=1, n_max=3),
        evaluation=EFAEvaluationConfig(),
        manual_candidates=(2, "3"),  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match="must contain integers"):
        run_efa_workflow(data, config)


def test_run_efa_workflow_rejects_diagnostics_items_mismatch() -> None:
    data = _synthetic_efa_data()
    config = EFAWorkflowConfig(
        items=("i1", "i2", "i3", "i4", "i5", "i6"),
        diagnostics=EFADiagnosticsConfig(items=("i1", "i2")),
        selection=FactorSelectionConfig(items=(), n_min=1, n_max=3),
        evaluation=EFAEvaluationConfig(),
    )
    with pytest.raises(ValueError, match="diagnostics.items"):
        run_efa_workflow(data, config)


def test_run_efa_workflow_rejects_selection_items_mismatch() -> None:
    data = _synthetic_efa_data()
    config = EFAWorkflowConfig(
        items=("i1", "i2", "i3", "i4", "i5", "i6"),
        diagnostics=EFADiagnosticsConfig(items=()),
        selection=FactorSelectionConfig(items=("i1", "i2"), n_min=1, n_max=3),
        evaluation=EFAEvaluationConfig(),
    )
    with pytest.raises(ValueError, match="selection.items"):
        run_efa_workflow(data, config)


def test_run_efa_workflow_rejects_inconsistent_preprocessing_between_subconfigs() -> None:
    data = _synthetic_efa_data()
    config = EFAWorkflowConfig(
        items=("i1", "i2", "i3", "i4", "i5", "i6"),
        diagnostics=EFADiagnosticsConfig(items=(), missing_strategy="dropna"),
        selection=FactorSelectionConfig(items=(), n_min=1, n_max=3, missing_strategy="pairwise"),
        evaluation=EFAEvaluationConfig(),
    )
    with pytest.raises(ValueError, match="Workflow preprocessing is inconsistent"):
        run_efa_workflow(data, config)


def test_run_efa_workflow_top_level_preprocessing_propagates() -> None:
    data = _ordinal_efa_data()
    variable_types = {column: "ordinal" for column in data.columns}
    result = run_efa_workflow(
        data,
        EFAWorkflowConfig(
            items=("i1", "i2", "i3", "i4", "i5", "i6"),
            diagnostics=EFADiagnosticsConfig(items=()),
            selection=FactorSelectionConfig(items=(), n_min=1, n_max=3, enable_pa=False),
            evaluation=EFAEvaluationConfig(),
            missing_strategy="pairwise",
            correlation_method="polychoric",
            variable_types=variable_types,
        ),
    )
    assert any("polychoric" in msg.lower() for msg in result.diagnostics.warnings)
    assert any("polychoric" in msg.lower() for msg in result.selection.warnings)
    assert any("polychoric" in msg.lower() for msg in result.best_model.warnings)


def test_run_efa_workflow_best_interpretation_matches_best_model() -> None:
    data = _synthetic_efa_data()
    result = run_efa_workflow(
        data,
        EFAWorkflowConfig(
            items=("i1", "i2", "i3", "i4", "i5", "i6"),
            diagnostics=EFADiagnosticsConfig(items=()),
            selection=FactorSelectionConfig(items=(), n_min=1, n_max=4, pa_iter=120, random_state=42),
            evaluation=EFAEvaluationConfig(),
            interpretation=EFAInterpretationConfig(residual_top_n=5),
            include_interpretation=True,
        ),
    )
    assert result.best_interpretation is not None
    assert result.best_interpretation is result.candidate_interpretations[result.best_n_factors]
    assert result.best_interpretation.item_table.shape[0] == len(result.best_model.loadings.index)


def test_run_efa_workflow_comparison_table_sorted_by_score_desc_then_n_factors() -> None:
    data = _synthetic_efa_data()
    result = run_efa_workflow(
        data,
        EFAWorkflowConfig(
            items=("i1", "i2", "i3", "i4", "i5", "i6"),
            diagnostics=EFADiagnosticsConfig(items=()),
            selection=FactorSelectionConfig(items=(), n_min=1, n_max=4, pa_iter=120, random_state=42),
            evaluation=EFAEvaluationConfig(),
            candidate_strategy="range",
        ),
    )
    expected = result.comparison_table.sort_values(
        by=["score", "n_factors"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)
    assert result.comparison_table.reset_index(drop=True).equals(expected)
