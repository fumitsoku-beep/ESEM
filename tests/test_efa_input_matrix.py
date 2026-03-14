import numpy as np
import pandas as pd
import pytest

from psysem import EFAConfig
from psysem.efa.input_matrix import build_efa_input_matrix
from psysem.preprocessing import AssociationMatrixConfig, build_association_matrix


def _ordinal_likert_data(n: int = 1500, seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    latent = rng.multivariate_normal(
        mean=np.zeros(3),
        cov=np.array(
            [
                [1.0, 0.65, 0.55],
                [0.65, 1.0, 0.60],
                [0.55, 0.60, 1.0],
            ]
        ),
        size=n,
    )
    thresholds = np.array([-0.9, -0.2, 0.3, 0.9])
    observed = np.column_stack([
        np.digitize(latent[:, column], thresholds) + 1 for column in range(latent.shape[1])
    ])
    return pd.DataFrame(observed, columns=["i1", "i2", "i3"])


def test_build_efa_input_matrix_preserves_current_default_pipeline() -> None:
    data = pd.DataFrame(
        {
            "i1": [1.0, 2.0, 3.0, 4.0],
            "i2": [2.0, 3.0, 4.0, 5.0],
            "i3": [4.0, 3.0, 2.0, 1.0],
        }
    )
    config = EFAConfig(items=("i1", "i2", "i3"), n_factors=1)

    prepared = build_efa_input_matrix(data, config)
    expected = data.loc[:, ["i1", "i2", "i3"]].corr().to_numpy(dtype=float)
    expected = (expected + expected.T) / 2.0
    np.fill_diagonal(expected, 1.0)

    assert prepared.item_names == ("i1", "i2", "i3")
    assert prepared.warnings == ()
    assert np.allclose(prepared.corr, expected)


def test_build_efa_input_matrix_dropna_uses_complete_rows() -> None:
    data = pd.DataFrame(
        {
            "i1": [1.0, 2.0, np.nan, 4.0],
            "i2": [2.0, 3.0, 4.0, 5.0],
            "i3": [4.0, 3.0, 2.0, np.nan],
        }
    )
    config = EFAConfig(items=("i1", "i2", "i3"), n_factors=1, missing_strategy="dropna")

    prepared = build_efa_input_matrix(data, config)
    expected = data.dropna(axis=0, how="any").corr().to_numpy(dtype=float)
    expected = (expected + expected.T) / 2.0
    np.fill_diagonal(expected, 1.0)

    assert np.allclose(prepared.corr, expected)
    assert any("dropped 2 row(s)" in warning for warning in prepared.warnings)


def test_build_efa_input_matrix_pairwise_warns_for_variable_specific_counts() -> None:
    data = pd.DataFrame(
        {
            "i1": [1.0, 2.0, np.nan, 4.0],
            "i2": [2.0, 3.0, 4.0, 5.0],
            "i3": [4.0, np.nan, 2.0, 1.0],
        }
    )
    config = EFAConfig(items=("i1", "i2", "i3"), n_factors=1, missing_strategy="pairwise")

    prepared = build_efa_input_matrix(data, config)

    assert prepared.corr.shape == (3, 3)
    assert any("variable-specific observation counts" in warning for warning in prepared.warnings)


def test_build_efa_input_matrix_supports_spearman_correlation() -> None:
    data = pd.DataFrame(
        {
            "i1": [10.0, 20.0, 30.0, 40.0, 1000.0],
            "i2": [1.0, 2.0, 3.0, 4.0, 5.0],
            "i3": [5.0, 4.0, 3.0, 2.0, 1.0],
        }
    )
    config = EFAConfig(
        items=("i1", "i2", "i3"),
        n_factors=1,
        correlation_method="spearman",
    )

    prepared = build_efa_input_matrix(data, config)
    expected = data.corr(method="spearman").to_numpy(dtype=float)
    expected = (expected + expected.T) / 2.0
    np.fill_diagonal(expected, 1.0)

    assert np.allclose(prepared.corr, expected)
    assert any("Spearman rank correlation" in warning for warning in prepared.warnings)


def test_build_efa_input_matrix_supports_polychoric_correlation() -> None:
    data = _ordinal_likert_data()
    config = EFAConfig(
        items=("i1", "i2", "i3"),
        n_factors=1,
        correlation_method="polychoric",
    )

    prepared = build_efa_input_matrix(data, config)

    assert prepared.corr.shape == (3, 3)
    assert np.allclose(prepared.corr, prepared.corr.T)
    assert np.allclose(np.diag(prepared.corr), 1.0)
    assert prepared.corr[0, 1] > 0.5
    assert any("polychoric correlation" in warning for warning in prepared.warnings)


def test_build_efa_input_matrix_polychoric_rejects_non_ordinal_items() -> None:
    data = pd.DataFrame(
        {
            "i1": [0.1, 0.4, 0.2, 0.5, 0.3],
            "i2": [1, 2, 3, 4, 5],
            "i3": [2, 3, 4, 5, 1],
        }
    )
    config = EFAConfig(
        items=("i1", "i2", "i3"),
        n_factors=1,
        correlation_method="polychoric",
        variable_types={"i2": "ordinal", "i3": "ordinal"},
    )

    with pytest.raises(ValueError, match="requires all analysis items to resolve to `ordinal`"):
        build_efa_input_matrix(data, config)


def test_build_efa_input_matrix_recommends_non_pearson_for_ordinal_like_data() -> None:
    data = pd.DataFrame(
        {
            "i1": [1, 2, 3, 4, 5, 1, 2],
            "i2": [2, 3, 4, 5, 1, 2, 3],
            "i3": [5, 4, 3, 2, 1, 5, 4],
        }
    )
    config = EFAConfig(items=("i1", "i2", "i3"), n_factors=1, correlation_method="pearson")

    prepared = build_efa_input_matrix(data, config)

    assert any("ordinal-like items detected" in warning for warning in prepared.warnings)
    assert any("polychoric" in warning for warning in prepared.warnings)


def test_build_efa_input_matrix_uses_declared_variable_types_for_recommendations() -> None:
    data = pd.DataFrame(
        {
            "i1": [0.1, 0.4, 0.2, 0.5, 0.3],
            "i2": [0.2, 0.5, 0.3, 0.6, 0.4],
            "i3": [0.3, 0.4, 0.2, 0.5, 0.1],
        }
    )
    config = EFAConfig(
        items=("i1", "i2", "i3"),
        n_factors=1,
        variable_types={"i1": "ordinal", "i2": "continuous", "i3": "continuous"},
    )

    prepared = build_efa_input_matrix(data, config)

    assert any("Declared ordinal-like items detected (i1)" in warning for warning in prepared.warnings)


def test_build_efa_input_matrix_stabilizes_non_finite_entries() -> None:
    data = pd.DataFrame(
        {
            "i1": [1.0, 1.0, 1.0, 1.0],
            "i2": [1.0, 2.0, 3.0, 4.0],
            "i3": [4.0, 3.0, 2.0, 1.0],
        }
    )
    config = EFAConfig(items=("i1", "i2", "i3"), n_factors=1)

    prepared = build_efa_input_matrix(data, config)

    assert prepared.corr.shape == (3, 3)
    assert np.all(np.isfinite(prepared.corr))
    assert np.allclose(prepared.corr, prepared.corr.T)
    assert np.allclose(np.diag(prepared.corr), 1.0)


def test_build_efa_input_matrix_matches_shared_preprocessing_output() -> None:
    data = _ordinal_likert_data()
    config = EFAConfig(
        items=("i1", "i2", "i3"),
        n_factors=1,
        correlation_method="polychoric",
    )

    efa_prepared = build_efa_input_matrix(data, config)
    shared_prepared = build_association_matrix(
        data,
        AssociationMatrixConfig(
            items=("i1", "i2", "i3"),
            correlation_method="polychoric",
            include_pairwise_counts=False,
        ),
    )

    assert np.allclose(efa_prepared.corr, shared_prepared.matrix.to_numpy(dtype=float))
    assert efa_prepared.item_names == shared_prepared.item_names
    assert efa_prepared.warnings == shared_prepared.warnings
