import pandas as pd
import pytest

from psysem import build_measurement_design, check_measurement_identification, parse_model


def test_build_measurement_design_smoke() -> None:
    spec = parse_model("eta =~ 1*x1 + x2 + x3")
    design = build_measurement_design(spec)
    assert design.observed_variables == ("x1", "x2", "x3")
    assert design.latent_variables == ("eta",)
    assert design.lambda_matrix.shape == (3, 1)
    assert design.lambda_matrix.loc["x1", "eta"] == pytest.approx(1.0)
    assert pd.isna(design.lambda_matrix.loc["x2", "eta"])
    assert pd.isna(design.lambda_matrix.loc["x3", "eta"])
    assert design.theta_matrix.shape == (3, 3)


def test_build_measurement_design_rejects_no_measurement_relations() -> None:
    spec = parse_model("y ~ x1 + x2")
    with pytest.raises(ValueError, match="No measurement relations"):
        build_measurement_design(spec)


def test_build_measurement_design_rejects_too_few_indicators() -> None:
    spec = parse_model("eta =~ x1")
    with pytest.raises(ValueError, match="fewer than 2 indicators"):
        build_measurement_design(spec)


def test_check_measurement_identification_warns_without_fixed_marker() -> None:
    spec = parse_model("eta =~ x1 + x2 + x3")
    design = build_measurement_design(spec)
    warnings = check_measurement_identification(design)
    assert any("no fixed loading marker" in warning.lower() for warning in warnings)
