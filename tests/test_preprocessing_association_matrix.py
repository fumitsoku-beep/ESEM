import numpy as np
import pandas as pd
import pytest

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


def test_build_association_matrix_default_pipeline_matches_current_efa_behavior() -> None:
    data = pd.DataFrame(
        {
            "i1": [1.0, 2.0, 3.0, 4.0],
            "i2": [2.0, 3.0, 4.0, 5.0],
            "i3": [4.0, 3.0, 2.0, 1.0],
        }
    )

    prepared = build_association_matrix(
        data,
        AssociationMatrixConfig(items=("i1", "i2", "i3")),
    )

    expected = data.loc[:, ["i1", "i2", "i3"]].corr().to_numpy(dtype=float)
    expected = (expected + expected.T) / 2.0
    np.fill_diagonal(expected, 1.0)

    assert prepared.item_names == ("i1", "i2", "i3")
    assert prepared.correlation_method == "pearson"
    assert prepared.missing_strategy == "pairwise"
    assert prepared.warnings == ()
    assert prepared.dropped_rows == 0
    assert prepared.n_complete_rows == 4
    assert prepared.stabilization_applied is True
    assert prepared.pairwise_n is not None
    assert np.allclose(prepared.matrix.to_numpy(dtype=float), expected)
    assert np.all(prepared.pairwise_n.to_numpy(dtype=int) == 4)


def test_build_association_matrix_dropna_tracks_complete_rows_and_pairwise_counts() -> None:
    data = pd.DataFrame(
        {
            "i1": [1.0, 2.0, np.nan, 4.0],
            "i2": [2.0, 3.0, 4.0, 5.0],
            "i3": [4.0, 3.0, 2.0, np.nan],
        }
    )

    prepared = build_association_matrix(
        data,
        AssociationMatrixConfig(
            items=("i1", "i2", "i3"),
            missing_strategy="dropna",
        ),
    )

    expected = data.dropna(axis=0, how="any").corr().to_numpy(dtype=float)
    expected = (expected + expected.T) / 2.0
    np.fill_diagonal(expected, 1.0)

    assert prepared.n_complete_rows == 2
    assert prepared.dropped_rows == 2
    assert prepared.pairwise_n is not None
    assert np.all(prepared.pairwise_n.to_numpy(dtype=int) == 2)
    assert np.allclose(prepared.matrix.to_numpy(dtype=float), expected)
    assert any("dropped 2 row(s)" in warning for warning in prepared.warnings)


def test_build_association_matrix_supports_polychoric_and_exposes_metadata() -> None:
    data = _ordinal_likert_data()

    prepared = build_association_matrix(
        data,
        AssociationMatrixConfig(
            items=("i1", "i2", "i3"),
            correlation_method="polychoric",
        ),
    )

    assert prepared.matrix.shape == (3, 3)
    assert prepared.pairwise_n is not None
    assert prepared.resolved_variable_types == {"i1": "ordinal", "i2": "ordinal", "i3": "ordinal"}
    assert np.allclose(prepared.matrix.to_numpy(dtype=float), prepared.matrix.to_numpy(dtype=float).T)
    assert np.allclose(np.diag(prepared.matrix.to_numpy(dtype=float)), 1.0)
    assert prepared.matrix.iloc[0, 1] > 0.5
    assert any("polychoric correlation" in warning for warning in prepared.warnings)


def test_build_association_matrix_polychoric_rejects_non_ordinal_items() -> None:
    data = pd.DataFrame(
        {
            "i1": [0.1, 0.4, 0.2, 0.5, 0.3],
            "i2": [1, 2, 3, 4, 5],
            "i3": [2, 3, 4, 5, 1],
        }
    )

    with pytest.raises(ValueError, match="requires all analysis items to resolve to `ordinal`"):
        build_association_matrix(
            data,
            AssociationMatrixConfig(
                items=("i1", "i2", "i3"),
                correlation_method="polychoric",
                variable_types={"i2": "ordinal", "i3": "ordinal"},
            ),
        )


def test_build_association_matrix_reports_stabilization_metadata() -> None:
    data = pd.DataFrame(
        {
            "i1": [1.0, 1.0, 1.0, 1.0],
            "i2": [1.0, 2.0, 3.0, 4.0],
            "i3": [4.0, 3.0, 2.0, 1.0],
        }
    )

    prepared = build_association_matrix(
        data,
        AssociationMatrixConfig(items=("i1", "i2", "i3")),
    )

    assert prepared.stabilization_applied is True
    assert np.all(np.isfinite(prepared.matrix.to_numpy(dtype=float)))
