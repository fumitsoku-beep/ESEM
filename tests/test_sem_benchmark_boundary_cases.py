import math

import numpy as np
import pandas as pd

from psysem import SEMModel, to_markdown


def test_boundary_benchmark_just_identified_model_marks_partial_fit_indices() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=300)
    data = pd.DataFrame(
        {
            "x1": x,
            "x2": x + 1e-6 * rng.normal(size=300),
            "x3": x + 1e-6 * rng.normal(size=300),
        }
    )

    result = SEMModel("eta =~ x1 + x2 + x3").fit(data)
    summary = result.summary()
    report = to_markdown(result)

    assert result.converged is True
    assert result.optimization_info["fit_status"] == "partial"
    assert result.optimization_info["fit_failure_reason"] == "invalid_model_degrees_of_freedom"
    assert result.optimization_info["df_model"] == 0
    assert math.isnan(result.fit_indices["cfi"])
    assert math.isnan(result.fit_indices["tli"])
    assert math.isnan(result.fit_indices["rmsea"])
    assert any("degrees of freedom are invalid" in warning for warning in result.warnings)
    assert "Fit status: partial" in summary
    assert "Fit issue: invalid_model_degrees_of_freedom" in summary
    assert "Fit status: `partial`" in report
    assert "Fit issue: `invalid_model_degrees_of_freedom`" in report


def test_boundary_benchmark_small_sample_surfaces_skipped_optimization_warning() -> None:
    data = pd.DataFrame(
        {
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "x2": [1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1],
            "x3": [0.9, 1.9, 2.9, 3.9, 4.9, 5.9, 6.9, 7.9],
        }
    )

    result = SEMModel("eta =~ x1 + x2 + x3").fit(data)
    summary = result.summary()
    report = to_markdown(result)

    assert result.converged is True
    assert result.optimization_info["ml_optimized"] is False
    assert result.optimization_info["status"] == "placeholder"
    assert all(math.isnan(value) for value in result.fit_indices.values())
    assert any("sample size below prototype threshold" in warning for warning in result.warnings)
    assert "ml_optimized: False" in summary
    assert "ML optimization skipped: sample size below prototype threshold" in summary
    assert "- ml_optimized: `False`" in report
    assert "ML optimization skipped: sample size below prototype threshold" in report