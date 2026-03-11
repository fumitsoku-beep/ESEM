import math

import numpy as np
import pandas as pd

from psysem import SEMModel, compute_basic_fit_indices
from psysem.fit_indices import compute_fit_indices


def test_compute_basic_fit_indices_keeps_placeholder_mode() -> None:
    indices = compute_basic_fit_indices()
    for key in ("cfi", "tli", "rmsea", "srmr", "aic", "bic"):
        assert key in indices
        assert math.isnan(indices[key])


def test_compute_fit_indices_perfect_fit_values() -> None:
    covariance = np.array(
        [
            [1.0, 0.30, 0.20],
            [0.30, 1.10, 0.35],
            [0.20, 0.35, 0.90],
        ],
        dtype=float,
    )
    result = compute_fit_indices(
        sample_covariance=covariance,
        implied_covariance=covariance,
        n_obs=200,
        n_free_parameters=5,
        objective=0.0,
    )
    assert result.chi_square == 0.0
    assert result.df_model == 1
    assert result.indices["srmr"] == 0.0
    assert result.indices["rmsea"] == 0.0
    assert result.indices["cfi"] == 1.0
    assert result.indices["tli"] == 1.0
    assert result.indices["aic"] == 10.0
    assert result.indices["bic"] == 5.0 * math.log(200.0)


def test_compute_fit_indices_returns_nan_for_invalid_df_model() -> None:
    covariance = np.array(
        [
            [1.0, 0.25, 0.10],
            [0.25, 1.20, 0.30],
            [0.10, 0.30, 0.80],
        ],
        dtype=float,
    )
    result = compute_fit_indices(
        sample_covariance=covariance,
        implied_covariance=covariance,
        n_obs=150,
        n_free_parameters=6,  # p(p+1)/2 == 6, so df_model == 0
        objective=0.0,
    )
    assert result.df_model == 0
    assert math.isnan(result.indices["cfi"])
    assert math.isnan(result.indices["tli"])
    assert math.isnan(result.indices["rmsea"])
    assert any("degrees of freedom are invalid" in warning for warning in result.warnings)


def test_sem_model_fit_populates_fit_indices_for_estimable_model() -> None:
    rng = np.random.default_rng(123)
    n = 300
    eta = rng.normal(size=n)
    data = pd.DataFrame(
        {
            "x1": 0.80 * eta + rng.normal(scale=0.70, size=n),
            "x2": 0.75 * eta + rng.normal(scale=0.65, size=n),
            "x3": 0.70 * eta + rng.normal(scale=0.75, size=n),
            "x4": 0.85 * eta + rng.normal(scale=0.60, size=n),
        }
    )

    result = SEMModel("eta =~ x1 + x2 + x3 + x4").fit(data)
    assert set(result.fit_indices) == {"cfi", "tli", "rmsea", "srmr", "aic", "bic"}
    assert not math.isnan(result.fit_indices["srmr"])
    assert not math.isnan(result.fit_indices["aic"])
    assert not math.isnan(result.fit_indices["bic"])
    assert "chi_square" in result.optimization_info
    assert "df_model" in result.optimization_info
    assert "chi_square_baseline" in result.optimization_info
    assert "df_baseline" in result.optimization_info

