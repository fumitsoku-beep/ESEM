import pandas as pd
import pytest

from psysem import (
    build_implied_covariance,
    build_measurement_design,
    build_parameter_index_map,
    build_start_vector,
    check_measurement_identification,
    parse_model,
)


def test_build_measurement_design_smoke() -> None:
    spec = parse_model("eta =~ 1*x1 + x2 + x3")
    design = build_measurement_design(spec)
    assert design.observed_variables == ("x1", "x2", "x3")
    assert design.latent_variables == ("eta",)
    assert design.lambda_matrix.shape == (3, 1)
    assert design.lambda_matrix.loc["x1", "eta"] == pytest.approx(1.0)
    assert pd.isna(design.lambda_matrix.loc["x2", "eta"])
    assert pd.isna(design.lambda_matrix.loc["x3", "eta"])
    assert int(design.lambda_parameter_index.loc["x1", "eta"]) == 0
    assert int(design.lambda_parameter_index.loc["x2", "eta"]) == 1
    assert int(design.lambda_parameter_index.loc["x3", "eta"]) == 2
    assert design.theta_matrix.shape == (3, 3)
    assert design.phi_matrix.shape == (1, 1)
    assert int(design.theta_parameter_index.loc["x1", "x1"]) == 3
    assert int(design.theta_parameter_index.loc["x2", "x2"]) == 4
    assert int(design.theta_parameter_index.loc["x3", "x3"]) == 5
    assert int(design.phi_parameter_index.loc["eta", "eta"]) == 6
    assert len(design.loading_parameters) == 3
    assert design.loading_parameters[0].is_free is False
    assert design.loading_parameters[1].is_free is True
    assert design.loading_parameters[0].vector_position is None
    assert design.loading_parameters[1].vector_position == 0


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


def test_build_measurement_design_with_parameter_table_uses_given_indices() -> None:
    spec = parse_model("eta =~ a*x1 + x2 + x3")
    parameter_table = (
        {
            "relation_index": 1,
            "term_index": 1,
            "is_free": True,
            "parameter": "a",
            "parameter_index": 5,
            "vector_position": 0,
            "fixed_value": None,
        },
        {
            "relation_index": 1,
            "term_index": 2,
            "is_free": True,
            "parameter": "p1",
            "parameter_index": 6,
            "vector_position": 1,
            "fixed_value": None,
        },
        {
            "relation_index": 1,
            "term_index": 3,
            "is_free": True,
            "parameter": "p2",
            "parameter_index": 7,
            "vector_position": 2,
            "fixed_value": None,
        },
        {
            "relation_index": 2,
            "term_index": 1,
            "lhs": "x1",
            "operator": "~~",
            "rhs": "x1",
            "is_free": True,
            "parameter": "th1",
            "parameter_index": 8,
            "vector_position": 3,
            "fixed_value": None,
        },
        {
            "relation_index": 3,
            "term_index": 1,
            "lhs": "x2",
            "operator": "~~",
            "rhs": "x2",
            "is_free": True,
            "parameter": "th2",
            "parameter_index": 9,
            "vector_position": 4,
            "fixed_value": None,
        },
        {
            "relation_index": 4,
            "term_index": 1,
            "lhs": "x3",
            "operator": "~~",
            "rhs": "x3",
            "is_free": True,
            "parameter": "th3",
            "parameter_index": 10,
            "vector_position": 5,
            "fixed_value": None,
        },
    )
    design = build_measurement_design(spec, parameter_table=parameter_table)
    indices = [item.parameter_index for item in design.loading_parameters]
    assert indices == [5, 6, 7]
    positions = [item.vector_position for item in design.loading_parameters]
    assert positions == [0, 1, 2]
    assert all(item.is_free for item in design.loading_parameters)
    assert int(design.lambda_parameter_index.loc["x1", "eta"]) == 5
    assert int(design.lambda_parameter_index.loc["x2", "eta"]) == 6
    assert int(design.lambda_parameter_index.loc["x3", "eta"]) == 7
    assert int(design.theta_parameter_index.loc["x1", "x1"]) == 8
    assert int(design.theta_parameter_index.loc["x2", "x2"]) == 9
    assert int(design.theta_parameter_index.loc["x3", "x3"]) == 10


def test_build_measurement_design_tracks_block_latent_pairs() -> None:
    spec = parse_model(
        "internalizing_f1 =~ i1 + i2 + i3\n"
        "externalizing_f1 =~ e1 + e2 + e3"
    )
    from psysem.sem.model import ModelSpec

    spec_with_blocks = ModelSpec(
        source=spec.source,
        syntax=spec.syntax,
        relations=spec.relations,
        observed_variables=spec.observed_variables,
        latent_variables=spec.latent_variables,
        estimator=spec.estimator,
        block_names=("internalizing", "externalizing"),
        warnings=spec.warnings,
        constraints=spec.constraints,
        parsed_constraints=spec.parsed_constraints,
    )
    design = build_measurement_design(spec_with_blocks)
    assert design.block_latent_pairs == (
        ("internalizing", "internalizing_f1"),
        ("externalizing", "externalizing_f1"),
    )


def test_build_measurement_design_preserves_latent_covariance_rows() -> None:
    spec = parse_model(
        "visual =~ 1*x1 + x2 + x3\n"
        "textual =~ 1*y1 + y2 + y3\n"
        "visual ~~ textual"
    )
    parameter_table = (
        {
            "relation_index": 1,
            "term_index": 1,
            "lhs": "visual",
            "operator": "=~",
            "rhs": "x1",
            "is_free": False,
            "parameter": None,
            "parameter_index": None,
            "vector_position": None,
            "fixed_value": 1.0,
        },
        {
            "relation_index": 1,
            "term_index": 2,
            "lhs": "visual",
            "operator": "=~",
            "rhs": "x2",
            "is_free": True,
            "parameter": "p1",
            "parameter_index": 1,
            "vector_position": 0,
            "fixed_value": None,
        },
        {
            "relation_index": 1,
            "term_index": 3,
            "lhs": "visual",
            "operator": "=~",
            "rhs": "x3",
            "is_free": True,
            "parameter": "p2",
            "parameter_index": 2,
            "vector_position": 1,
            "fixed_value": None,
        },
        {
            "relation_index": 2,
            "term_index": 1,
            "lhs": "textual",
            "operator": "=~",
            "rhs": "y1",
            "is_free": False,
            "parameter": None,
            "parameter_index": None,
            "vector_position": None,
            "fixed_value": 1.0,
        },
        {
            "relation_index": 2,
            "term_index": 2,
            "lhs": "textual",
            "operator": "=~",
            "rhs": "y2",
            "is_free": True,
            "parameter": "p3",
            "parameter_index": 3,
            "vector_position": 2,
            "fixed_value": None,
        },
        {
            "relation_index": 2,
            "term_index": 3,
            "lhs": "textual",
            "operator": "=~",
            "rhs": "y3",
            "is_free": True,
            "parameter": "p4",
            "parameter_index": 4,
            "vector_position": 3,
            "fixed_value": None,
        },
        {
            "relation_index": 3,
            "term_index": 1,
            "lhs": "visual",
            "operator": "~~",
            "rhs": "textual",
            "is_free": True,
            "parameter": "p5",
            "parameter_index": 5,
            "vector_position": 4,
            "fixed_value": None,
        },
    )
    design = build_measurement_design(spec, parameter_table=parameter_table)
    assert pd.isna(design.phi_matrix.loc["visual", "textual"])
    assert pd.isna(design.phi_matrix.loc["textual", "visual"])
    assert int(design.phi_parameter_index.loc["visual", "textual"]) == 5
    assert int(design.phi_parameter_index.loc["textual", "visual"]) == 5


def test_build_measurement_design_preserves_observed_residual_covariance() -> None:
    spec = parse_model("eta =~ 1*x1 + x2 + x3\nx1 ~~ x2")
    parameter_table = (
        {
            "relation_index": 1,
            "term_index": 1,
            "lhs": "eta",
            "operator": "=~",
            "rhs": "x1",
            "is_free": False,
            "parameter": None,
            "parameter_index": None,
            "vector_position": None,
            "fixed_value": 1.0,
        },
        {
            "relation_index": 1,
            "term_index": 2,
            "lhs": "eta",
            "operator": "=~",
            "rhs": "x2",
            "is_free": True,
            "parameter": "p1",
            "parameter_index": 1,
            "vector_position": 0,
            "fixed_value": None,
        },
        {
            "relation_index": 1,
            "term_index": 3,
            "lhs": "eta",
            "operator": "=~",
            "rhs": "x3",
            "is_free": True,
            "parameter": "p2",
            "parameter_index": 2,
            "vector_position": 1,
            "fixed_value": None,
        },
        {
            "relation_index": 2,
            "term_index": 1,
            "lhs": "x1",
            "operator": "~~",
            "rhs": "x2",
            "is_free": True,
            "parameter": "p3",
            "parameter_index": 3,
            "vector_position": 2,
            "fixed_value": None,
        },
        {
            "relation_index": 3,
            "term_index": 1,
            "lhs": "x1",
            "operator": "~~",
            "rhs": "x1",
            "is_free": True,
            "parameter": "p4",
            "parameter_index": 4,
            "vector_position": 3,
            "fixed_value": None,
        },
        {
            "relation_index": 4,
            "term_index": 1,
            "lhs": "x2",
            "operator": "~~",
            "rhs": "x2",
            "is_free": True,
            "parameter": "p5",
            "parameter_index": 5,
            "vector_position": 4,
            "fixed_value": None,
        },
        {
            "relation_index": 5,
            "term_index": 1,
            "lhs": "x3",
            "operator": "~~",
            "rhs": "x3",
            "is_free": True,
            "parameter": "p6",
            "parameter_index": 6,
            "vector_position": 5,
            "fixed_value": None,
        },
    )
    design = build_measurement_design(spec, parameter_table=parameter_table)
    assert pd.isna(design.theta_matrix.loc["x1", "x2"])
    assert pd.isna(design.theta_matrix.loc["x2", "x1"])
    assert int(design.theta_parameter_index.loc["x1", "x2"]) == 3
    assert int(design.theta_parameter_index.loc["x2", "x1"]) == 3


def test_build_implied_covariance_uses_observed_residual_covariance() -> None:
    spec = parse_model("eta =~ 1*x1 + x2 + x3\nx1 ~~ x2")
    parameter_table = (
        {
            "relation_index": 1,
            "term_index": 1,
            "lhs": "eta",
            "operator": "=~",
            "rhs": "x1",
            "is_free": False,
            "parameter": None,
            "parameter_index": None,
            "vector_position": None,
            "fixed_value": 1.0,
        },
        {
            "relation_index": 1,
            "term_index": 2,
            "lhs": "eta",
            "operator": "=~",
            "rhs": "x2",
            "is_free": True,
            "parameter": "p1",
            "parameter_index": 1,
            "vector_position": 0,
            "fixed_value": None,
        },
        {
            "relation_index": 1,
            "term_index": 3,
            "lhs": "eta",
            "operator": "=~",
            "rhs": "x3",
            "is_free": True,
            "parameter": "p2",
            "parameter_index": 2,
            "vector_position": 1,
            "fixed_value": None,
        },
        {
            "relation_index": 2,
            "term_index": 1,
            "lhs": "x1",
            "operator": "~~",
            "rhs": "x2",
            "is_free": True,
            "parameter": "p3",
            "parameter_index": 3,
            "vector_position": 2,
            "fixed_value": None,
        },
        {
            "relation_index": 3,
            "term_index": 1,
            "lhs": "x1",
            "operator": "~~",
            "rhs": "x1",
            "is_free": True,
            "parameter": "p4",
            "parameter_index": 4,
            "vector_position": 3,
            "fixed_value": None,
        },
        {
            "relation_index": 4,
            "term_index": 1,
            "lhs": "x2",
            "operator": "~~",
            "rhs": "x2",
            "is_free": True,
            "parameter": "p5",
            "parameter_index": 5,
            "vector_position": 4,
            "fixed_value": None,
        },
        {
            "relation_index": 5,
            "term_index": 1,
            "lhs": "x3",
            "operator": "~~",
            "rhs": "x3",
            "is_free": True,
            "parameter": "p6",
            "parameter_index": 6,
            "vector_position": 5,
            "fixed_value": None,
        },
    )
    parameter_index_map = build_parameter_index_map(parameter_table)
    design = build_measurement_design(spec, parameter_table=parameter_table)
    start_vector = build_start_vector(parameter_index_map, parameter_table=parameter_table)
    implied = build_implied_covariance(design, start_vector, parameter_index_map)
    assert implied.loc["x1", "x2"] > 0.0
    assert implied.loc["x2", "x1"] == pytest.approx(implied.loc["x1", "x2"])
