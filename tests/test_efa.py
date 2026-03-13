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


def _synthetic_ordinal_efa_data(n: int = 1200, seed: int = 21) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    factor = rng.normal(size=(n, 1))
    loadings = np.array([[0.85], [0.80], [0.75], [0.70]])
    noise = rng.normal(scale=0.55, size=(n, 4))
    latent = factor @ loadings.T + noise
    thresholds = np.array([-1.0, -0.2, 0.25, 0.9])
    observed = np.column_stack([
        np.digitize(latent[:, column], thresholds) + 1 for column in range(latent.shape[1])
    ])
    return pd.DataFrame(observed, columns=[f"i{i}" for i in range(1, 5)])


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


def test_fit_efa_accepts_dropna_missing_strategy() -> None:
    data = _synthetic_efa_data()
    data.loc[0, "i1"] = np.nan
    data.loc[1, "i2"] = np.nan

    result = fit_efa(
        data,
        EFAConfig(
            items=("i1", "i2", "i3", "i4", "i5", "i6"),
            n_factors=2,
            extraction="paf",
            rotation="varimax",
            missing_strategy="dropna",
        ),
    )

    assert result.loadings.shape == (6, 2)
    assert any("dropna strategy" in warning for warning in result.warnings)


def test_fit_efa_accepts_spearman_correlation_method() -> None:
    data = _synthetic_efa_data()
    result = fit_efa(
        data,
        EFAConfig(
            items=("i1", "i2", "i3", "i4", "i5", "i6"),
            n_factors=2,
            extraction="paf",
            rotation="varimax",
            correlation_method="spearman",
        ),
    )

    assert result.loadings.shape == (6, 2)
    assert any("Spearman rank correlation" in warning for warning in result.warnings)


def test_fit_efa_accepts_polychoric_correlation_method() -> None:
    data = _synthetic_ordinal_efa_data()
    result = fit_efa(
        data,
        EFAConfig(
            items=("i1", "i2", "i3", "i4"),
            n_factors=1,
            extraction="paf",
            rotation="none",
            correlation_method="polychoric",
        ),
    )

    assert result.loadings.shape == (4, 1)
    assert result.correlation_matrix.shape == (4, 4)
    assert any("polychoric correlation" in warning for warning in result.warnings)


def test_fit_efa_warns_for_declared_ordinal_items_under_pearson() -> None:
    data = _synthetic_efa_data()
    result = fit_efa(
        data,
        EFAConfig(
            items=("i1", "i2", "i3", "i4", "i5", "i6"),
            n_factors=2,
            extraction="paf",
            rotation="varimax",
            correlation_method="pearson",
            variable_types={"i1": "ordinal", "i2": "ordinal"},
        ),
    )

    assert any("Declared ordinal-like items detected" in warning for warning in result.warnings)


def test_fit_efa_minres_varimax_shapes() -> None:
    data = _synthetic_efa_data()
    result = fit_efa(
        data,
        EFAConfig(
            items=("i1", "i2", "i3", "i4", "i5", "i6"),
            n_factors=2,
            extraction="minres",
            rotation="varimax",
        ),
    )

    assert result.extraction == "minres"
    assert result.loadings.shape == (6, 2)
    assert result.converged is True
    assert result.n_iter >= 0
    assert (result.uniquenesses >= 0.005).all()
    assert (result.communalities >= 0.0).all()
    assert (result.communalities <= 1.0).all()


def test_fit_efa_ml_varimax_shapes() -> None:
    data = _synthetic_efa_data()
    result = fit_efa(
        data,
        EFAConfig(
            items=("i1", "i2", "i3", "i4", "i5", "i6"),
            n_factors=2,
            extraction="ml",
            rotation="varimax",
        ),
    )

    assert result.extraction == "ml"
    assert result.loadings.shape == (6, 2)
    assert result.converged is True
    assert result.n_iter >= 0
    assert (result.uniquenesses >= 0.005).all()
    assert (result.communalities >= 0.0).all()
    assert (result.communalities <= 1.0).all()


def test_fit_efa_ml_reports_non_convergence_warning() -> None:
    data = _synthetic_efa_data()
    result = fit_efa(
        data,
        EFAConfig(
            items=("i1", "i2", "i3", "i4", "i5", "i6"),
            n_factors=2,
            extraction="ml",
            rotation="varimax",
            max_iter=1,
        ),
    )

    joined = " | ".join(result.warnings)
    assert result.converged is False
    assert "Extraction did not converge" in joined


def test_fit_efa_paf_promax_returns_factor_correlation() -> None:
    rng = np.random.default_rng(123)
    latent_cov = np.array([[1.0, 0.45], [0.45, 1.0]])
    factors = rng.multivariate_normal(mean=np.zeros(2), cov=latent_cov, size=500)
    loadings = np.array(
        [
            [0.82, 0.10],
            [0.77, 0.14],
            [0.72, 0.12],
            [0.09, 0.81],
            [0.15, 0.76],
            [0.11, 0.73],
        ]
    )
    noise = rng.normal(scale=0.40, size=(500, 6))
    observed = factors @ loadings.T + noise
    data = pd.DataFrame(observed, columns=[f"i{i}" for i in range(1, 7)])

    result = fit_efa(
        data,
        EFAConfig(
            items=("i1", "i2", "i3", "i4", "i5", "i6"),
            n_factors=2,
            extraction="paf",
            rotation="promax",
        ),
    )

    phi = result.factor_correlation.to_numpy(dtype=float)
    assert result.rotation == "promax"
    assert result.loadings.shape == (6, 2)
    assert phi.shape == (2, 2)
    assert np.allclose(np.diag(phi), 1.0)
    assert np.allclose(phi, phi.T)
    assert not np.allclose(phi, np.eye(2))


def test_fit_efa_paf_oblimin_returns_factor_correlation() -> None:
    rng = np.random.default_rng(321)
    latent_cov = np.array([[1.0, 0.35], [0.35, 1.0]])
    factors = rng.multivariate_normal(mean=np.zeros(2), cov=latent_cov, size=500)
    loadings = np.array(
        [
            [0.84, 0.08],
            [0.79, 0.10],
            [0.75, 0.11],
            [0.10, 0.82],
            [0.12, 0.78],
            [0.09, 0.74],
        ]
    )
    noise = rng.normal(scale=0.42, size=(500, 6))
    observed = factors @ loadings.T + noise
    data = pd.DataFrame(observed, columns=[f"i{i}" for i in range(1, 7)])

    result = fit_efa(
        data,
        EFAConfig(
            items=("i1", "i2", "i3", "i4", "i5", "i6"),
            n_factors=2,
            extraction="paf",
            rotation="oblimin",
        ),
    )

    phi = result.factor_correlation.to_numpy(dtype=float)
    assert result.rotation == "oblimin"
    assert result.loadings.shape == (6, 2)
    assert phi.shape == (2, 2)
    assert np.allclose(np.diag(phi), 1.0)
    assert np.allclose(phi, phi.T)
    assert not np.allclose(phi, np.eye(2))


def test_fit_efa_paf_geomin_returns_factor_correlation() -> None:
    rng = np.random.default_rng(456)
    latent_cov = np.array([[1.0, 0.40], [0.40, 1.0]])
    factors = rng.multivariate_normal(mean=np.zeros(2), cov=latent_cov, size=500)
    loadings = np.array(
        [
            [0.83, 0.09],
            [0.78, 0.13],
            [0.73, 0.10],
            [0.12, 0.80],
            [0.10, 0.77],
            [0.08, 0.72],
        ]
    )
    noise = rng.normal(scale=0.41, size=(500, 6))
    observed = factors @ loadings.T + noise
    data = pd.DataFrame(observed, columns=[f"i{i}" for i in range(1, 7)])

    result = fit_efa(
        data,
        EFAConfig(
            items=("i1", "i2", "i3", "i4", "i5", "i6"),
            n_factors=2,
            extraction="paf",
            rotation="geomin",
        ),
    )

    phi = result.factor_correlation.to_numpy(dtype=float)
    assert result.rotation == "geomin"
    assert result.loadings.shape == (6, 2)
    assert phi.shape == (2, 2)
    assert np.allclose(np.diag(phi), 1.0)
    assert np.allclose(phi, phi.T)
    assert not np.allclose(phi, np.eye(2))


def test_fit_efa_geomin_restarts_are_reproducible_with_random_state() -> None:
    data = _synthetic_efa_data(n=500, seed=91)
    config = EFAConfig(
        items=("i1", "i2", "i3", "i4", "i5", "i6"),
        n_factors=2,
        extraction="paf",
        rotation="geomin",
        rotation_restarts=3,
        random_state=123,
    )

    result_a = fit_efa(data, config)
    result_b = fit_efa(data, config)

    assert np.allclose(result_a.loadings.to_numpy(dtype=float), result_b.loadings.to_numpy(dtype=float))
    assert np.allclose(
        result_a.factor_correlation.to_numpy(dtype=float),
        result_b.factor_correlation.to_numpy(dtype=float),
    )


def test_fit_efa_paf_target_rotation_respects_zero_targets() -> None:
    data = _synthetic_efa_data(n=500, seed=77)
    target = pd.DataFrame(
        [
            [np.nan, 0.0],
            [np.nan, 0.0],
            [np.nan, 0.0],
            [0.0, np.nan],
            [0.0, np.nan],
            [0.0, np.nan],
        ],
        index=[f"i{i}" for i in range(1, 7)],
        columns=["F1", "F2"],
    )

    result = fit_efa(
        data,
        EFAConfig(
            items=("i1", "i2", "i3", "i4", "i5", "i6"),
            n_factors=2,
            extraction="paf",
            rotation="target",
            rotation_target=target,
        ),
    )

    loadings = result.loadings.to_numpy(dtype=float)
    phi = result.factor_correlation.to_numpy(dtype=float)
    off_target = np.abs(
        np.array([
            loadings[0, 1],
            loadings[1, 1],
            loadings[2, 1],
            loadings[3, 0],
            loadings[4, 0],
            loadings[5, 0],
        ])
    )
    primary = np.abs(
        np.array([
            loadings[0, 0],
            loadings[1, 0],
            loadings[2, 0],
            loadings[3, 1],
            loadings[4, 1],
            loadings[5, 1],
        ])
    )

    assert result.rotation == "target"
    assert loadings.shape == (6, 2)
    assert phi.shape == (2, 2)
    assert np.allclose(np.diag(phi), 1.0)
    assert np.allclose(phi, phi.T)
    assert off_target.mean() < primary.mean()


def test_fit_efa_target_rotation_requires_target_matrix() -> None:
    data = _synthetic_efa_data()
    with pytest.raises(ValueError, match="rotation_target"):
        fit_efa(
            data,
            EFAConfig(
                items=("i1", "i2", "i3", "i4", "i5", "i6"),
                n_factors=2,
                extraction="paf",
                rotation="target",
            ),
        )


def test_fit_efa_target_rotation_zero_weights_match_free_cells() -> None:
    data = _synthetic_efa_data(n=500, seed=78)
    weighted_target = pd.DataFrame(
        [
            [np.nan, 0.0],
            [np.nan, 0.0],
            [np.nan, 0.0],
            [0.0, np.nan],
            [0.0, np.nan],
            [0.0, np.nan],
        ],
        index=[f"i{i}" for i in range(1, 7)],
        columns=["F1", "F2"],
    )
    free_target = pd.DataFrame(
        [
            [np.nan, np.nan],
            [np.nan, np.nan],
            [np.nan, np.nan],
            [0.0, np.nan],
            [0.0, np.nan],
            [0.0, np.nan],
        ],
        index=[f"i{i}" for i in range(1, 7)],
        columns=["F1", "F2"],
    )
    weights = pd.DataFrame(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        index=weighted_target.index[::-1],
        columns=weighted_target.columns,
    )

    weighted_result = fit_efa(
        data,
        EFAConfig(
            items=("i1", "i2", "i3", "i4", "i5", "i6"),
            n_factors=2,
            extraction="paf",
            rotation="target",
            rotation_target=weighted_target,
            rotation_target_weights=weights,
            rotation_restarts=2,
            random_state=321,
        ),
    )
    free_result = fit_efa(
        data,
        EFAConfig(
            items=("i1", "i2", "i3", "i4", "i5", "i6"),
            n_factors=2,
            extraction="paf",
            rotation="target",
            rotation_target=free_target,
            rotation_restarts=2,
            random_state=321,
        ),
    )

    assert np.allclose(
        weighted_result.loadings.to_numpy(dtype=float),
        free_result.loadings.to_numpy(dtype=float),
    )
    assert np.allclose(
        weighted_result.factor_correlation.to_numpy(dtype=float),
        free_result.factor_correlation.to_numpy(dtype=float),
    )


def test_fit_efa_target_rotation_rejects_invalid_weights() -> None:
    data = _synthetic_efa_data()
    target = pd.DataFrame(
        [
            [np.nan, 0.0],
            [np.nan, 0.0],
            [np.nan, 0.0],
            [0.0, np.nan],
            [0.0, np.nan],
            [0.0, np.nan],
        ],
        index=[f"i{i}" for i in range(1, 7)],
        columns=["F1", "F2"],
    )

    with pytest.raises(ValueError, match="must be >= 0"):
        fit_efa(
            data,
            EFAConfig(
                items=("i1", "i2", "i3", "i4", "i5", "i6"),
                n_factors=2,
                extraction="paf",
                rotation="target",
                rotation_target=target,
                rotation_target_weights=np.full((6, 2), -1.0),
            ),
        )

    with pytest.raises(ValueError, match="positive weight"):
        fit_efa(
            data,
            EFAConfig(
                items=("i1", "i2", "i3", "i4", "i5", "i6"),
                n_factors=2,
                extraction="paf",
                rotation="target",
                rotation_target=target,
                rotation_target_weights=np.zeros((6, 2), dtype=float),
            ),
        )


def test_fit_efa_rejects_invalid_rotation() -> None:
    data = _synthetic_efa_data()
    config = EFAConfig(
        items=("i1", "i2", "i3", "i4", "i5", "i6"),
        n_factors=2,
        rotation="quartimin",
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
    assert "ml" in list_extraction_methods()
    assert "minres" in list_extraction_methods()
    assert "pca" in list_extraction_methods()
    assert "geomin" in list_rotation_methods()
    assert "oblimin" in list_rotation_methods()
    assert "promax" in list_rotation_methods()
    assert "target" in list_rotation_methods()
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


def test_register_custom_oblique_rotation_and_fit() -> None:
    def custom_oblique_rotation(loadings: np.ndarray, _: EFAConfig):
        phi = np.array([[1.0, 0.25], [0.25, 1.0]], dtype=float)
        return loadings, phi

    register_rotation_method("test_oblique", custom_oblique_rotation, overwrite=True)

    data = _synthetic_efa_data()
    result = fit_efa(
        data,
        EFAConfig(
            items=("i1", "i2", "i3", "i4", "i5", "i6"),
            n_factors=2,
            extraction="paf",
            rotation="test_oblique",
        ),
    )

    phi = result.factor_correlation.to_numpy(dtype=float)
    assert result.rotation == "test_oblique"
    assert np.allclose(np.diag(phi), 1.0)
    assert np.allclose(phi, phi.T)
    assert not np.allclose(phi, np.eye(2))


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
    with pytest.raises(ValueError, match="`rotation_restarts` must be >= 0"):
        fit_efa(data, EFAConfig(items=("i1", "i2", "i3"), n_factors=1, rotation_restarts=-1))
    with pytest.raises(ValueError, match="`min_uniqueness` must be between 0 and 1"):
        fit_efa(data, EFAConfig(items=("i1", "i2", "i3"), n_factors=1, min_uniqueness=1.0))
    with pytest.raises(ValueError, match="`missing_strategy` must be one of"):
        fit_efa(data, EFAConfig(items=("i1", "i2", "i3"), n_factors=1, missing_strategy="median"))
    with pytest.raises(ValueError, match="`correlation_method` must be one of"):
        fit_efa(data, EFAConfig(items=("i1", "i2", "i3"), n_factors=1, correlation_method="kendall"))
    with pytest.raises(ValueError, match="requires all analysis items to resolve to `ordinal`"):
        fit_efa(
            data,
            EFAConfig(
                items=("i1", "i2", "i3"),
                n_factors=1,
                correlation_method="polychoric",
                variable_types={"i1": "continuous", "i2": "ordinal", "i3": "ordinal"},
            ),
        )
    with pytest.raises(ValueError, match="`variable_types` entries must be `continuous` or `ordinal`"):
        fit_efa(
            data,
            EFAConfig(items=("i1", "i2", "i3"), n_factors=1, variable_types={"i1": "binary"}),
        )
    with pytest.raises(ValueError, match="`variable_types` contains items not present in `items`"):
        fit_efa(
            data,
            EFAConfig(items=("i1", "i2", "i3"), n_factors=1, variable_types={"i4": "ordinal"}),
        )


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
