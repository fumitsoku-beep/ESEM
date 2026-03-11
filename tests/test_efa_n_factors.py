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


def _strict_map_values(corr: np.ndarray, n_max: int) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(corr)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.clip(eigvals[order], 0.0, None)
    eigvecs = eigvecs[:, order]
    p = corr.shape[0]
    upper = np.triu_indices(p, k=1)
    map_curve = np.zeros(n_max + 1, dtype=float)

    for k in range(0, n_max + 1):
        if k == 0:
            partial = corr.copy()
        else:
            loadings = eigvecs[:, :k] * np.sqrt(eigvals[:k])
            residual = corr - (loadings @ loadings.T)
            residual = np.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0)
            residual = (residual + residual.T) / 2.0
            diag = np.clip(np.diag(residual), 1e-12, None)
            scale = np.sqrt(np.outer(diag, diag))
            partial = np.divide(residual, scale, out=np.zeros_like(residual), where=scale > 0)
            partial = np.clip(partial, -1.0, 1.0)
        np.fill_diagonal(partial, 0.0)
        values = partial[upper]
        map_curve[k] = float(np.mean(values * values))
    return map_curve


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


def test_suggest_n_factors_map_matches_partial_formula() -> None:
    data = _synthetic_efa_data()
    config = FactorSelectionConfig(
        items=("i1", "i2", "i3", "i4", "i5", "i6"),
        n_min=1,
        n_max=4,
        enable_pa=False,
        enable_map=True,
        enable_kaiser=False,
        enable_scree=False,
    )
    result = suggest_n_factors(data, config)
    assert result.map_values is not None

    corr = data.loc[:, list(config.items)].corr().to_numpy(dtype=float)
    expected = _strict_map_values(corr, n_max=4)
    for k in range(0, 5):
        assert result.map_values.loc[k] == pytest.approx(expected[k], rel=1e-9, abs=1e-9)


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
