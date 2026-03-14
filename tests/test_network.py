from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from psysem import NetworkConfig, fit_network


def _ggm_demo_data(n: int = 5000, seed: int = 123) -> tuple[pd.DataFrame, np.ndarray]:
    precision = np.array(
        [
            [1.0, -0.45, 0.20],
            [-0.45, 1.0, -0.30],
            [0.20, -0.30, 1.0],
        ],
        dtype=float,
    )
    covariance = np.linalg.inv(precision)
    rng = np.random.default_rng(seed)
    observed = rng.multivariate_normal(mean=np.zeros(3), cov=covariance, size=n)
    return pd.DataFrame(observed, columns=["i1", "i2", "i3"]), precision


def _ordinal_network_data(n: int = 2500, seed: int = 321) -> pd.DataFrame:
    continuous, _ = _ggm_demo_data(n=n, seed=seed)
    ordinal = {
        column: pd.qcut(continuous[column], q=5, labels=False, duplicates="drop") + 1
        for column in continuous.columns
    }
    return pd.DataFrame(ordinal)


def _expected_partial_correlation(precision: np.ndarray) -> np.ndarray:
    diag = np.sqrt(np.diag(precision))
    scale = np.outer(diag, diag)
    partial = -precision / scale
    np.fill_diagonal(partial, 0.0)
    return partial


def test_fit_network_smoke_recovers_partial_correlation_structure() -> None:
    data, precision = _ggm_demo_data()
    result = fit_network(data, NetworkConfig(items=("i1", "i2", "i3")))

    expected = _expected_partial_correlation(precision)

    assert result.association_matrix.shape == (3, 3)
    assert result.precision_matrix.shape == (3, 3)
    assert result.partial_correlation_matrix.shape == (3, 3)
    assert result.adjacency_matrix.shape == (3, 3)
    assert np.allclose(np.diag(result.partial_correlation_matrix.to_numpy(dtype=float)), 0.0)
    assert np.allclose(
        result.partial_correlation_matrix.to_numpy(dtype=float),
        result.partial_correlation_matrix.to_numpy(dtype=float).T,
    )
    assert result.edge_table["abs_weight"].is_monotonic_decreasing
    assert set(result.node_table.columns) == {
        "node",
        "degree",
        "strength",
        "expected_influence",
        "positive_strength",
        "negative_strength",
    }
    assert result.partial_correlation_matrix.loc["i1", "i2"] == pytest.approx(
        expected[0, 1],
        abs=0.08,
    )
    assert result.partial_correlation_matrix.loc["i1", "i3"] == pytest.approx(
        expected[0, 2],
        abs=0.08,
    )
    assert result.partial_correlation_matrix.loc["i2", "i3"] == pytest.approx(
        expected[1, 2],
        abs=0.08,
    )


def test_fit_network_min_abs_edge_filters_adjacency_and_node_degree() -> None:
    data, _ = _ggm_demo_data()
    result = fit_network(
        data,
        NetworkConfig(
            items=("i1", "i2", "i3"),
            min_abs_edge=0.4,
        ),
    )

    assert result.edge_table.shape[0] == 1
    assert tuple(result.edge_table.loc[0, ["source", "target"]]) == ("i1", "i2")
    degrees = dict(zip(result.node_table["node"], result.node_table["degree"], strict=True))
    assert degrees == {"i1": 1, "i2": 1, "i3": 0}


def test_fit_network_supports_polychoric_for_ordinal_items() -> None:
    data = _ordinal_network_data()
    result = fit_network(
        data,
        NetworkConfig(
            items=("i1", "i2", "i3"),
            correlation_method="polychoric",
            variable_types={"i1": "ordinal", "i2": "ordinal", "i3": "ordinal"},
        ),
    )

    assert result.correlation_method == "polychoric"
    assert result.resolved_variable_types == {"i1": "ordinal", "i2": "ordinal", "i3": "ordinal"}
    assert result.edge_table.shape[0] >= 1
    assert any("polychoric correlation" in warning for warning in result.warnings)


def test_fit_network_propagates_pairwise_metadata() -> None:
    data, _ = _ggm_demo_data(n=400)
    data.loc[0, "i1"] = np.nan
    data.loc[1, "i2"] = np.nan

    result = fit_network(
        data,
        NetworkConfig(
            items=("i1", "i2", "i3"),
            missing_strategy="pairwise",
        ),
    )

    assert result.pairwise_n is not None
    assert result.pairwise_n.loc["i1", "i2"] == 398
    assert any("pairwise" in warning.lower() for warning in result.warnings)


def test_fit_network_rejects_invalid_config() -> None:
    data, _ = _ggm_demo_data(n=200)

    with pytest.raises(ValueError, match="Unsupported network estimator"):
        fit_network(
            data,
            NetworkConfig(items=("i1", "i2", "i3"), estimator="glasso"),
        )

    with pytest.raises(ValueError, match="`ridge` must be >= 0"):
        fit_network(
            data,
            NetworkConfig(items=("i1", "i2", "i3"), ridge=-1e-3),
        )

    with pytest.raises(ValueError, match="`min_abs_edge` must be in \\[0, 1\\)"):
        fit_network(
            data,
            NetworkConfig(items=("i1", "i2", "i3"), min_abs_edge=1.0),
        )
