from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from psysem import ESEMWorkflowConfig, SEMFitConfig, run_esem_workflow
from psysem.data import esem_spec_from_dict


def _make_demo_data(n_obs: int = 360, seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(n_obs, 2))
    loadings = np.array(
        [
            [0.80, 0.10],
            [0.75, 0.15],
            [0.70, 0.05],
            [0.10, 0.80],
            [0.15, 0.75],
            [0.05, 0.70],
        ],
        dtype=float,
    )
    noise = rng.normal(scale=0.45, size=(n_obs, 6))
    observed = factors @ loadings.T + noise
    columns = [f"i{i}" for i in range(1, 7)]
    return pd.DataFrame(observed, columns=columns)


def _make_demo_ordinal_data(n_obs: int = 360, seed: int = 123) -> pd.DataFrame:
    continuous = _make_demo_data(n_obs=n_obs, seed=seed)
    ordinal = {
        column: pd.qcut(continuous[column], q=5, labels=False, duplicates="drop") + 1
        for column in continuous.columns
    }
    return pd.DataFrame(ordinal)


def _spec_payload(estimator: str = "ML") -> dict[str, object]:
    items = [f"i{i}" for i in range(1, 7)]
    return {
        "blocks": [
            {
                "name": "internalizing",
                "items": items,
                "n_factors": 2,
            }
        ],
        "estimator": estimator,
        "variable_types": {item: "continuous" for item in items},
    }


def test_run_esem_workflow_smoke_with_mapping_spec() -> None:
    data = _make_demo_data()
    config = ESEMWorkflowConfig(
        fit_config=SEMFitConfig(
            max_iter=180,
            restarts=1,
            random_seed=42,
        ),
    )
    result = run_esem_workflow(data, _spec_payload(), config)

    assert result.best_candidate_id == "block_full"
    assert "block_full" in result.candidates
    assert not result.comparison_table.empty
    assert "total_score" in result.comparison_table.columns

    candidate = result.best_candidate
    assert candidate.sem_result is not None
    assert candidate.sem_result.measurement_design is not None
    assert candidate.sem_result.optimization_info.get("ml_optimized") is True
    assert "internalizing" in candidate.block_efa_results


def test_run_esem_workflow_accepts_esem_spec_input() -> None:
    data = _make_demo_data(seed=456)
    spec = esem_spec_from_dict(_spec_payload())

    result = run_esem_workflow(
        data,
        spec,
        ESEMWorkflowConfig(
            include_efa_bridge=False,
            enabled_judges=("convergence", "fit_indices"),
            fit_config=SEMFitConfig(max_iter=160, restarts=1, random_seed=7),
        ),
    )

    assert result.input_spec is spec
    assert result.best_candidate.sem_result is not None
    assert result.best_candidate.block_efa_results == {}


def test_run_esem_workflow_rejects_non_ml_estimator_when_required() -> None:
    data = _make_demo_data()
    with pytest.raises(ValueError, match="requires `spec.estimator` in \\{ML, MLR\\}"):
        run_esem_workflow(
            data,
            _spec_payload(estimator="WLSMV"),
            ESEMWorkflowConfig(require_ml_estimator=True),
        )


def test_run_esem_workflow_rejects_unknown_generator() -> None:
    data = _make_demo_data(n_obs=120)
    with pytest.raises(ValueError, match="Unsupported generator strategies"):
        run_esem_workflow(
            data,
            _spec_payload(),
            ESEMWorkflowConfig(generator_strategies=("unknown",)),
        )


def test_run_esem_workflow_ordinal_block_bridge_uses_polychoric() -> None:
    data = _make_demo_ordinal_data()
    payload = _spec_payload()
    payload["variable_types"] = {f"i{i}": "ordinal" for i in range(1, 7)}
    result = run_esem_workflow(
        data,
        payload,
        ESEMWorkflowConfig(
            include_sem_fit=False,
            enabled_judges=("efa_bridge",),
        ),
    )
    efa_result = result.best_candidate.block_efa_results["internalizing"]
    assert any("polychoric" in msg.lower() for msg in efa_result.warnings)
