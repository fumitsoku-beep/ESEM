import numpy as np
import pandas as pd
import pytest

from psysem import build_ml_context, gaussian_ml_discrepancy


def test_gaussian_ml_discrepancy_zero_when_sample_equals_implied() -> None:
    cov = np.array([[1.0, 0.2], [0.2, 0.8]], dtype=float)
    value = gaussian_ml_discrepancy(cov, cov)
    assert value == pytest.approx(0.0, abs=1e-10)


def test_gaussian_ml_discrepancy_rejects_non_positive_definite() -> None:
    sample = np.array([[1.0, 2.0], [2.0, 4.0]], dtype=float)
    implied = np.eye(2, dtype=float)
    with pytest.raises(ValueError, match="positive definite"):
        gaussian_ml_discrepancy(sample, implied)


def test_build_ml_context_uses_observed_subset() -> None:
    data = pd.DataFrame(
        {
            "x1": [1.0, 2.0, 3.0, 4.0],
            "x2": [2.0, 2.5, 3.5, 4.0],
            "y": [0.8, 1.4, 2.1, 2.7],
        }
    )
    context = build_ml_context(data, observed_variables=("x1", "x2", "y"))
    assert context.sample_covariance is not None
    assert context.observed_variables == ("x1", "x2", "y")
    if context.objective_at_sample_cov is not None:
        assert context.objective_at_sample_cov == pytest.approx(0.0, abs=1e-10)


def test_build_ml_context_warns_for_non_dataframe() -> None:
    context = build_ml_context([1, 2, 3], observed_variables=("x1",))
    assert context.sample_covariance is None
    assert any("not a pandas DataFrame" in warning for warning in context.warnings)
