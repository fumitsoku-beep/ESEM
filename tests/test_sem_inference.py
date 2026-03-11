import numpy as np
import pytest

from psysem import build_parameter_index_map, estimate_parameter_inference


def test_estimate_parameter_inference_quadratic_objective() -> None:
    parameter_table = (
        {"is_free": True, "parameter": "p1", "parameter_index": 1},
        {"is_free": True, "parameter": "p2", "parameter_index": 2},
    )
    parameter_index_map = build_parameter_index_map(parameter_table)
    x = np.array([1.0, -2.0], dtype=float)
    a = np.array([[0.25, 0.0], [0.0, 1.0 / 9.0]], dtype=float)

    def objective_fn(v: np.ndarray) -> float:
        return 0.5 * float(v.T @ a @ v)

    result = estimate_parameter_inference(
        objective_fn=objective_fn,
        parameter_vector=x,
        parameter_index_map=parameter_index_map,
    )
    assert len(result.entries) == 2
    assert result.covariance_matrix is not None
    se = [entry.standard_error for entry in result.entries]
    assert se[0] == pytest.approx(2.0, rel=1e-2, abs=1e-2)
    assert se[1] == pytest.approx(3.0, rel=1e-2, abs=1e-2)
    assert result.entries[0].z_value == pytest.approx(0.5, rel=1e-2, abs=1e-2)


def test_estimate_parameter_inference_rejects_shape_mismatch() -> None:
    parameter_table = (
        {"is_free": True, "parameter": "p1", "parameter_index": 1},
    )
    parameter_index_map = build_parameter_index_map(parameter_table)

    with pytest.raises(ValueError, match="does not match n_free"):
        estimate_parameter_inference(
            objective_fn=lambda v: float(np.sum(v**2)),
            parameter_vector=np.array([0.1, 0.2], dtype=float),
            parameter_index_map=parameter_index_map,
        )

