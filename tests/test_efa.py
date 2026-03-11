import numpy as np
import pandas as pd
import pytest

from psysem import (
    EFAConfig,
    fit_efa,
    list_extraction_methods,
    list_rotation_methods,
    register_extraction_method,
    register_rotation_method,
)


def _synthetic_efa_data(n: int = 400, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(n, 2))
    loadings = np.array(
        [
            [0.80, 0.10],
            [0.75, 0.15],
            [0.70, 0.05],
            [0.10, 0.80],
            [0.15, 0.75],
            [0.05, 0.70],
        ]
    )
    noise = rng.normal(scale=0.45, size=(n, 6))
    observed = factors @ loadings.T + noise
    columns = [f"i{i}" for i in range(1, 7)]
    return pd.DataFrame(observed, columns=columns)


def test_fit_efa_paf_varimax_shapes() -> None:
    data = _synthetic_efa_data()
    config = EFAConfig(
        items=("i1", "i2", "i3", "i4", "i5", "i6"),
        n_factors=2,
        extraction="paf",
        rotation="varimax",
    )
    result = fit_efa(data, config)

    assert result.loadings.shape == (6, 2)
    assert result.communalities.shape[0] == 6
    assert result.uniquenesses.shape[0] == 6
    assert result.explained_variance.shape[0] == 2
    assert result.correlation_matrix.shape == (6, 6)
    assert result.residual_matrix.shape == (6, 6)
    assert result.factor_correlation.shape == (2, 2)
    assert result.complexity.shape[0] == 6
    assert set(result.residual_summary) == {
        "rmsr",
        "mean_abs_residual",
        "max_abs_residual",
        "n_abs_gt_0_05",
        "n_abs_gt_0_10",
    }
    assert isinstance(result.cross_loaded_items, tuple)
    assert isinstance(result.warnings, tuple)
    assert (result.communalities >= 0).all()
    assert (result.communalities <= 1).all()
    assert (result.complexity >= 1.0).all()
    assert (result.uniquenesses >= config.min_uniqueness).all()
    assert result.n_iter >= 1


def test_fit_efa_pca_none_rotation() -> None:
    data = _synthetic_efa_data()
    config = EFAConfig(
        items=("i1", "i2", "i3", "i4", "i5", "i6"),
        n_factors=2,
        extraction="pca",
        rotation="none",
    )
    result = fit_efa(data, config)
    assert result.extraction == "pca"
    assert result.rotation == "none"
    assert result.converged is True


def test_fit_efa_rejects_invalid_rotation() -> None:
    data = _synthetic_efa_data()
    config = EFAConfig(
        items=("i1", "i2", "i3", "i4", "i5", "i6"),
        n_factors=2,
        rotation="oblimin",
    )
    with pytest.raises(ValueError, match="Unsupported rotation method"):
        fit_efa(data, config)


def test_fit_efa_rejects_missing_items() -> None:
    data = _synthetic_efa_data().drop(columns=["i6"])
    config = EFAConfig(
        items=("i1", "i2", "i3", "i4", "i5", "i6"),
        n_factors=2,
    )
    with pytest.raises(ValueError, match="Missing item columns"):
        fit_efa(data, config)


def test_list_methods_contains_defaults() -> None:
    assert "paf" in list_extraction_methods()
    assert "pca" in list_extraction_methods()
    assert "varimax" in list_rotation_methods()
    assert "none" in list_rotation_methods()


def test_register_custom_methods_and_fit() -> None:
    def custom_extraction(corr: np.ndarray, config: EFAConfig):
        p = corr.shape[0]
        loadings = np.zeros((p, config.n_factors), dtype=float)
        communalities = np.zeros(p, dtype=float)
        return loadings, communalities, 1, True

    def custom_rotation(loadings: np.ndarray, _: EFAConfig):
        return loadings

    register_extraction_method("test_zero", custom_extraction, overwrite=True)
    register_rotation_method("test_identity", custom_rotation, overwrite=True)

    data = _synthetic_efa_data()
    config = EFAConfig(
        items=("i1", "i2", "i3", "i4", "i5", "i6"),
        n_factors=2,
        extraction="test_zero",
        rotation="test_identity",
    )
    result = fit_efa(data, config)
    assert (result.loadings.to_numpy() == 0.0).all()
    assert result.extraction == "test_zero"
    assert result.rotation == "test_identity"


def test_register_existing_method_requires_overwrite() -> None:
    def custom_rotation(loadings: np.ndarray, _: EFAConfig):
        return loadings

    with pytest.raises(ValueError, match="already registered"):
        register_rotation_method("varimax", custom_rotation, overwrite=False)


def test_fit_efa_rejects_non_dataframe_input() -> None:
    config = EFAConfig(items=("i1", "i2", "i3"), n_factors=1)
    with pytest.raises(TypeError, match="pandas.DataFrame"):
        fit_efa(data=[1, 2, 3], config=config)  # type: ignore[arg-type]


def test_fit_efa_rejects_invalid_n_factors() -> None:
    data = _synthetic_efa_data()
    with pytest.raises(ValueError, match="`n_factors` must be > 0"):
        fit_efa(data, EFAConfig(items=("i1", "i2", "i3"), n_factors=0))
    with pytest.raises(ValueError, match="smaller than number of items"):
        fit_efa(data, EFAConfig(items=("i1", "i2", "i3"), n_factors=3))


def test_fit_efa_rejects_invalid_optimization_settings() -> None:
    data = _synthetic_efa_data()
    with pytest.raises(ValueError, match="`max_iter` must be > 0"):
        fit_efa(data, EFAConfig(items=("i1", "i2", "i3"), n_factors=1, max_iter=0))
    with pytest.raises(ValueError, match="`tol` must be > 0"):
        fit_efa(data, EFAConfig(items=("i1", "i2", "i3"), n_factors=1, tol=0.0))
    with pytest.raises(ValueError, match="`min_uniqueness` must be between 0 and 1"):
        fit_efa(data, EFAConfig(items=("i1", "i2", "i3"), n_factors=1, min_uniqueness=1.0))


def test_fit_efa_accepts_case_insensitive_method_names() -> None:
    data = _synthetic_efa_data()
    result = fit_efa(
        data,
        EFAConfig(
            items=("i1", "i2", "i3", "i4", "i5", "i6"),
            n_factors=2,
            extraction="PAF",
            rotation="VARIMAX",
        ),
    )
    assert result.extraction == "paf"
    assert result.rotation == "varimax"


def test_fit_efa_residual_matrix_has_zero_diagonal_and_is_symmetric() -> None:
    data = _synthetic_efa_data()
    result = fit_efa(
        data,
        EFAConfig(items=("i1", "i2", "i3", "i4", "i5", "i6"), n_factors=2),
    )
    residual = result.residual_matrix.to_numpy(dtype=float)
    assert np.allclose(np.diag(residual), 0.0)
    assert np.allclose(residual, residual.T)


def test_fit_efa_warns_for_cross_loaded_and_boundary_uniqueness() -> None:
    def dense_extraction(corr: np.ndarray, config: EFAConfig):
        p = corr.shape[0]
        loadings = np.full((p, config.n_factors), 0.80, dtype=float)
        communalities = np.sum(loadings * loadings, axis=1)
        return loadings, communalities, 1, True

    def identity_rotation(loadings: np.ndarray, _: EFAConfig):
        return loadings

    register_extraction_method("test_dense_cross", dense_extraction, overwrite=True)
    register_rotation_method("test_identity_dense", identity_rotation, overwrite=True)

    data = _synthetic_efa_data()
    result = fit_efa(
        data,
        EFAConfig(
            items=("i1", "i2", "i3", "i4", "i5", "i6"),
            n_factors=2,
            extraction="test_dense_cross",
            rotation="test_identity_dense",
            min_uniqueness=0.01,
        ),
    )
    joined = " | ".join(result.warnings).lower()
    assert "boundary uniqueness" in joined
    assert "cross-loaded items detected" in joined
    assert len(result.cross_loaded_items) == 6


def test_register_method_rejects_invalid_names() -> None:
    def passthrough_extraction(corr: np.ndarray, config: EFAConfig):
        p = corr.shape[0]
        return np.zeros((p, config.n_factors)), np.zeros(p), 1, True

    def passthrough_rotation(loadings: np.ndarray, _: EFAConfig):
        return loadings

    with pytest.raises(TypeError, match="must be a string"):
        register_extraction_method(123, passthrough_extraction, overwrite=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="cannot be empty"):
        register_rotation_method("   ", passthrough_rotation, overwrite=True)
