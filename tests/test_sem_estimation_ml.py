import numpy as np
import pandas as pd
import pytest

from psysem import (
    SEMModel,
    build_implied_covariance,
    build_measurement_design,
    build_ml_context,
    build_parameter_index_map,
    build_start_vector,
    estimate_parameter_inference,
    gaussian_ml_discrepancy,
    optimize_ml_parameters,
    parameter_vector_to_named_values,
    parse_model,
)


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


def test_build_implied_covariance_measurement_only_smoke() -> None:
    spec = parse_model("eta =~ x1 + x2 + x3")
    parameter_table = (
        {
            "relation_index": 1,
            "term_index": 1,
            "lhs": "eta",
            "operator": "=~",
            "rhs": "x1",
            "is_free": True,
            "parameter": "p1",
            "parameter_index": 1,
            "vector_position": 0,
            "fixed_value": None,
        },
        {
            "relation_index": 1,
            "term_index": 2,
            "lhs": "eta",
            "operator": "=~",
            "rhs": "x2",
            "is_free": True,
            "parameter": "p2",
            "parameter_index": 2,
            "vector_position": 1,
            "fixed_value": None,
        },
        {
            "relation_index": 1,
            "term_index": 3,
            "lhs": "eta",
            "operator": "=~",
            "rhs": "x3",
            "is_free": True,
            "parameter": "p3",
            "parameter_index": 3,
            "vector_position": 2,
            "fixed_value": None,
        },
    )
    measurement_design = build_measurement_design(spec, parameter_table=parameter_table)
    parameter_index_map = build_parameter_index_map(parameter_table)
    start = build_start_vector(parameter_index_map, parameter_table=parameter_table)
    sigma = build_implied_covariance(measurement_design, start, parameter_index_map)
    assert sigma.shape == (3, 3)
    assert tuple(sigma.columns) == ("x1", "x2", "x3")
    assert float(np.min(np.diag(sigma.to_numpy(dtype=float)))) > 0.0


def test_optimize_ml_parameters_smoke() -> None:
    rng = np.random.default_rng(42)
    n = 300
    eta = rng.normal(size=n)
    data = pd.DataFrame(
        {
            "x1": 0.8 * eta + rng.normal(scale=0.7, size=n),
            "x2": 0.7 * eta + rng.normal(scale=0.8, size=n),
            "x3": 0.9 * eta + rng.normal(scale=0.6, size=n),
        }
    )

    spec = parse_model("eta =~ x1 + x2 + x3")
    parameter_table = (
        {
            "relation_index": 1,
            "term_index": 1,
            "lhs": "eta",
            "operator": "=~",
            "rhs": "x1",
            "is_free": True,
            "parameter": "p1",
            "parameter_index": 1,
            "vector_position": 0,
            "fixed_value": None,
        },
        {
            "relation_index": 1,
            "term_index": 2,
            "lhs": "eta",
            "operator": "=~",
            "rhs": "x2",
            "is_free": True,
            "parameter": "p2",
            "parameter_index": 2,
            "vector_position": 1,
            "fixed_value": None,
        },
        {
            "relation_index": 1,
            "term_index": 3,
            "lhs": "eta",
            "operator": "=~",
            "rhs": "x3",
            "is_free": True,
            "parameter": "p3",
            "parameter_index": 3,
            "vector_position": 2,
            "fixed_value": None,
        },
    )
    measurement_design = build_measurement_design(spec, parameter_table=parameter_table)
    parameter_index_map = build_parameter_index_map(parameter_table)
    result = optimize_ml_parameters(
        data,
        measurement_design=measurement_design,
        structural_design=None,
        parameter_index_map=parameter_index_map,
        parameter_table=parameter_table,
        max_iter=100,
    )
    assert isinstance(result.success, bool)
    assert len(result.parameter_vector) == 3
    assert set(result.parameter_values) == {"p1", "p2", "p3"}
    assert result.sample_covariance is not None
    assert result.implied_covariance is not None
    assert result.objective is None or result.objective >= 0.0


def test_parameter_vector_to_named_values_shape_validation() -> None:
    parameter_table = (
        {"is_free": True, "parameter": "p1", "parameter_index": 1},
        {"is_free": True, "parameter": "p2", "parameter_index": 2},
    )
    parameter_index_map = build_parameter_index_map(parameter_table)
    with pytest.raises(ValueError, match="does not match n_free"):
        parameter_vector_to_named_values(np.array([0.1]), parameter_index_map)


def test_sem_model_fit_runs_ml_optimizer_on_sufficient_sample_size() -> None:
    rng = np.random.default_rng(0)
    n = 240
    eta = rng.normal(size=n)
    data = pd.DataFrame(
        {
            "x1": 0.8 * eta + rng.normal(scale=0.7, size=n),
            "x2": 0.7 * eta + rng.normal(scale=0.8, size=n),
            "x3": 0.9 * eta + rng.normal(scale=0.6, size=n),
        }
    )
    result = SEMModel("eta =~ x1 + x2 + x3").fit(data)
    assert result.optimization_info["ml_optimized"] is True
    assert "ml_optimization_success" in result.optimization_info
    assert "ml_n_optimized_observed" in result.optimization_info
    assert "n_inference_parameters" in result.optimization_info
    assert result.parameter_inference


def test_estimate_parameter_inference_handles_failed_hessian() -> None:
    parameter_table = (
        {"is_free": True, "parameter": "p1", "parameter_index": 1},
    )
    index_map = build_parameter_index_map(parameter_table)

    def bad_objective(_: np.ndarray) -> float:
        raise RuntimeError("boom")

    result = estimate_parameter_inference(
        objective_fn=bad_objective,
        parameter_vector=np.array([0.3], dtype=float),
        parameter_index_map=index_map,
    )
    assert len(result.entries) == 1
    assert result.entries[0].standard_error is None
    assert any("Numerical Hessian failed" in warning for warning in result.warnings)
