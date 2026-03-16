import numpy as np
import pandas as pd
import pytest

from psysem.sem.estimation import ml as ml_module
from psysem import (
    ParameterBoundsConfig,
    SEMFitConfig,
    SEMModel,
    build_measurement_design,
    build_parameter_index_map,
    optimize_ml_parameters,
    parse_model,
)


def test_sem_fit_config_validation() -> None:
    with pytest.raises(ValueError, match="max_iter"):
        SEMFitConfig(max_iter=0)
    with pytest.raises(ValueError, match="tol"):
        SEMFitConfig(tol=0.0)
    with pytest.raises(ValueError, match="restarts"):
        SEMFitConfig(restarts=-1)
    with pytest.raises(ValueError, match="random_start_scale"):
        SEMFitConfig(random_start_scale=-0.1)
    with pytest.raises(ValueError, match="supports only"):
        SEMFitConfig(method="BFGS")
    with pytest.raises(ValueError, match="bounds require lower <= upper"):
        ParameterBoundsConfig(default_lower=1.0, default_upper=0.0)


def test_optimize_ml_parameters_supports_restarts_and_attempt_diagnostics() -> None:
    data = _make_measurement_data(seed=12, n_obs=260)
    parameter_table, measurement_design, parameter_index_map = _build_measurement_setup()
    config = SEMFitConfig(
        max_iter=50,
        tol=1e-7,
        restarts=2,
        random_seed=123,
        random_start_scale=0.05,
    )
    result = optimize_ml_parameters(
        data,
        measurement_design=measurement_design,
        structural_design=None,
        parameter_index_map=parameter_index_map,
        parameter_table=parameter_table,
        fit_config=config,
    )
    assert result.n_attempts == 3
    assert result.method == "L-BFGS-B"
    assert len(result.attempt_objectives) == 3
    assert result.best_attempt is not None
    assert 1 <= result.best_attempt <= 3


def test_sem_model_fit_surfaces_fit_config_in_optimization_info() -> None:
    data = _make_measurement_data(seed=7, n_obs=240)
    config = SEMFitConfig(max_iter=40, tol=1e-7, restarts=1, random_seed=9)
    result = SEMModel("eta =~ x1 + x2 + x3").fit(data, fit_config=config)
    info = result.optimization_info
    assert info["fit_method"] == "L-BFGS-B"
    assert info["fit_max_iter"] == 40
    assert info["fit_tol"] == pytest.approx(1e-7)
    assert info["fit_restarts"] == 1
    assert info["fit_random_start_scale"] == pytest.approx(0.15)
    assert info["ml_n_attempts"] == 2
    assert info["ml_method"] == "L-BFGS-B"
    assert isinstance(info["ml_best_attempt"], int)


def test_sem_model_fit_rejects_invalid_fit_config_type() -> None:
    data = _make_measurement_data(seed=17, n_obs=200)
    with pytest.raises(TypeError, match="SEMFitConfig"):
        SEMModel("eta =~ x1 + x2 + x3").fit(data, fit_config="invalid")  # type: ignore[arg-type]


def test_optimize_ml_parameters_reports_failure_category(monkeypatch: pytest.MonkeyPatch) -> None:
    data = _make_measurement_data(seed=21, n_obs=220)
    parameter_table, measurement_design, parameter_index_map = _build_measurement_setup()

    def _raise_implied(*_args: object, **_kwargs: object) -> pd.DataFrame:
        raise ValueError("forced implied covariance failure")

    monkeypatch.setattr(ml_module, "build_implied_covariance", _raise_implied)
    result = optimize_ml_parameters(
        data,
        measurement_design=measurement_design,
        structural_design=None,
        parameter_index_map=parameter_index_map,
        parameter_table=parameter_table,
        fit_config=SEMFitConfig(max_iter=20, restarts=1, random_seed=11),
    )
    assert result.success is False
    assert result.failure_category == "implied_covariance"
    assert result.n_attempts == 2
    assert any("category=implied_covariance" in warning for warning in result.warnings)


def _build_measurement_setup() -> tuple[
    tuple[dict[str, object], ...],
    object,
    object,
]:
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
    return parameter_table, measurement_design, parameter_index_map


def _make_measurement_data(*, seed: int, n_obs: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    eta = rng.normal(size=n_obs)
    return pd.DataFrame(
        {
            "x1": 0.80 * eta + rng.normal(scale=0.70, size=n_obs),
            "x2": 0.75 * eta + rng.normal(scale=0.65, size=n_obs),
            "x3": 0.70 * eta + rng.normal(scale=0.75, size=n_obs),
        }
    )
