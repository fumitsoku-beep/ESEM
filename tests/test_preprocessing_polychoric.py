import numpy as np
import pandas as pd

from psysem.preprocessing.polychoric import (
    build_polychoric_matrix,
    estimate_polychoric_correlation,
)


def _ordinal_pair_data(n: int = 1000, seed: int = 33) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    latent = rng.multivariate_normal(
        mean=np.zeros(2),
        cov=np.array([[1.0, 0.70], [0.70, 1.0]]),
        size=n,
    )
    thresholds = np.array([-0.8, -0.1, 0.4, 0.9])
    observed = np.column_stack([
        np.digitize(latent[:, column], thresholds) + 1 for column in range(latent.shape[1])
    ])
    return pd.DataFrame(observed, columns=["i1", "i2"])


def test_estimate_polychoric_correlation_returns_positive_signal_for_ordinal_pair() -> None:
    data = _ordinal_pair_data()

    rho, warnings = estimate_polychoric_correlation(
        data["i1"],
        data["i2"],
        pair_label="i1~i2",
    )

    assert rho > 0.5
    assert warnings == ()


def test_estimate_polychoric_correlation_handles_single_category_input() -> None:
    left = pd.Series([1, 1, 1, 1, 1], name="i1")
    right = pd.Series([1, 2, 2, 3, 3], name="i2")

    rho, warnings = estimate_polychoric_correlation(left, right, pair_label="i1~i2")

    assert rho == 0.0
    assert any("at least two observed categories" in warning for warning in warnings)


def test_build_polychoric_matrix_reports_pairwise_missing_counts() -> None:
    data = _ordinal_pair_data(n=200)
    data["i3"] = ((data["i1"] + data["i2"]) / 2.0).round().clip(1, 5)
    data.loc[0, "i1"] = np.nan
    data.loc[1:3, "i2"] = np.nan
    data.loc[4:7, "i3"] = np.nan

    corr, pairwise_n, dropped_rows, n_complete_rows, warnings = build_polychoric_matrix(
        data,
        missing_strategy="pairwise",
    )

    assert corr.shape == (3, 3)
    assert pairwise_n.loc["i1", "i2"] == int(data.loc[:, ["i1", "i2"]].dropna().shape[0])
    assert dropped_rows == 0
    assert n_complete_rows == int(data.dropna(axis=0, how="any").shape[0])
    assert any("variable-specific observation counts" in warning for warning in warnings)
