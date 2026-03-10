import numpy as np
import pandas as pd
import pytest

from psysem import FactorSelectionConfig, suggest_n_factors


def _synthetic_efa_data(n: int = 600, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(n, 2))
    loadings = np.array(
        [
            [0.85, 0.05],
            [0.80, 0.10],
            [0.78, 0.08],
            [0.05, 0.82],
            [0.10, 0.78],
            [0.08, 0.76],
        ]
    )
    noise = rng.normal(scale=0.35, size=(n, 6))
    observed = factors @ loadings.T + noise
    return pd.DataFrame(observed, columns=[f"i{i}" for i in range(1, 7)])


def test_suggest_n_factors_smoke() -> None:
    data = _synthetic_efa_data()
    config = FactorSelectionConfig(
        items=("i1", "i2", "i3", "i4", "i5", "i6"),
        n_min=1,
        n_max=4,
        pa_iter=120,
        random_state=42,
    )
    result = suggest_n_factors(data, config)

    assert result.n_min == 1
    assert result.n_max == 4
    assert result.eigenvalues.shape[0] == 6
    assert result.consensus_n_factors in {1, 2, 3, 4}
    assert "parallel_analysis" in result.suggestions_by_method
    assert "map" in result.suggestions_by_method
    assert "kaiser" in result.suggestions_by_method
    assert "scree" in result.suggestions_by_method
    assert result.parallel_thresholds is not None
    assert result.map_values is not None


def test_suggest_n_factors_reproducible_parallel_analysis() -> None:
    data = _synthetic_efa_data()
    config = FactorSelectionConfig(
        items=("i1", "i2", "i3", "i4", "i5", "i6"),
        n_min=1,
        n_max=4,
        pa_iter=80,
        random_state=1234,
    )
    result_a = suggest_n_factors(data, config)
    result_b = suggest_n_factors(data, config)
    assert result_a.parallel_thresholds is not None
    assert result_b.parallel_thresholds is not None
    assert result_a.parallel_thresholds.equals(result_b.parallel_thresholds)
    assert result_a.suggestions_by_method == result_b.suggestions_by_method


def test_suggest_n_factors_only_kaiser() -> None:
    data = _synthetic_efa_data()
    config = FactorSelectionConfig(
        items=("i1", "i2", "i3", "i4", "i5", "i6"),
        n_min=1,
        n_max=4,
        enable_pa=False,
        enable_map=False,
        enable_scree=False,
        enable_kaiser=True,
    )
    result = suggest_n_factors(data, config)
    assert set(result.suggestions_by_method) == {"kaiser"}


def test_suggest_n_factors_rejects_invalid_range() -> None:
    data = _synthetic_efa_data()
    config = FactorSelectionConfig(
        items=("i1", "i2", "i3", "i4", "i5", "i6"),
        n_min=4,
        n_max=2,
    )
    with pytest.raises(ValueError, match="n_max"):
        suggest_n_factors(data, config)


def test_suggest_n_factors_requires_at_least_one_method() -> None:
    data = _synthetic_efa_data()
    config = FactorSelectionConfig(
        items=("i1", "i2", "i3", "i4", "i5", "i6"),
        n_min=1,
        n_max=4,
        enable_pa=False,
        enable_map=False,
        enable_kaiser=False,
        enable_scree=False,
    )
    with pytest.raises(ValueError, match="At least one factor-count method"):
        suggest_n_factors(data, config)
