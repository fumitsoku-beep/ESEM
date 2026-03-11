import numpy as np
import pandas as pd
import pytest

from psysem import EFAConfig, EFAEvaluationConfig, evaluate_efa_model, fit_efa


def _synthetic_efa_data(n: int = 450, seed: int = 21) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(n, 2))
    loadings = np.array(
        [
            [0.82, 0.10],
            [0.78, 0.14],
            [0.74, 0.07],
            [0.10, 0.84],
            [0.13, 0.79],
            [0.08, 0.75],
        ]
    )
    noise = rng.normal(scale=0.40, size=(n, 6))
    observed = factors @ loadings.T + noise
    return pd.DataFrame(observed, columns=[f"i{i}" for i in range(1, 7)])


def test_evaluate_efa_model_smoke() -> None:
    data = _synthetic_efa_data()
    efa_result = fit_efa(
        data,
        EFAConfig(
            items=("i1", "i2", "i3", "i4", "i5", "i6"),
            n_factors=2,
            extraction="paf",
            rotation="varimax",
        ),
    )
    evaluation = evaluate_efa_model(efa_result, EFAEvaluationConfig())

    assert evaluation.n_factors == 2
    assert evaluation.score == pytest.approx(float(evaluation.score))
    assert 0.0 <= evaluation.explained_total <= 1.0
    assert 0.0 <= evaluation.simple_structure_ratio <= 1.0
    assert 0.0 <= evaluation.mean_h2 <= 1.0
    assert evaluation.salient_items >= 0


def test_evaluate_efa_model_rejects_invalid_thresholds() -> None:
    data = _synthetic_efa_data()
    efa_result = fit_efa(
        data,
        EFAConfig(
            items=("i1", "i2", "i3", "i4", "i5", "i6"),
            n_factors=2,
        ),
    )
    with pytest.raises(ValueError, match="cross_loading"):
        evaluate_efa_model(
            efa_result,
            EFAEvaluationConfig(salient_loading=0.40, cross_loading=0.30),
        )
