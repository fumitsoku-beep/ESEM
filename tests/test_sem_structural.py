import pandas as pd
import pytest

from psysem import build_structural_design, check_structural_validity, parse_model


def test_build_structural_design_smoke() -> None:
    spec = parse_model(
        "eta1 =~ 1*x1 + x2 + x3\n"
        "eta2 =~ 1*y1 + y2 + y3\n"
        "eta2 ~ b1*eta1 + z1"
    )
    design = build_structural_design(spec)
    assert len(design.path_table) == 2
    assert design.endogenous_latent_variables == ("eta2",)
    assert design.exogenous_latent_variables == ("eta1",)
    assert design.observed_predictor_variables == ("z1",)
    assert design.beta_matrix.shape == (1, 1)
    assert pd.isna(design.gamma_matrix.loc["eta2", "eta1"])
    assert pd.isna(design.gamma_matrix.loc["eta2", "z1"])
    assert int(design.gamma_parameter_index.loc["eta2", "eta1"]) == 1
    assert int(design.gamma_parameter_index.loc["eta2", "z1"]) == 2
    assert int(design.beta_parameter_index.loc["eta2", "eta2"]) == 0
    assert design.psi_matrix.shape == (1, 1)
    assert pd.isna(design.psi_matrix.loc["eta2", "eta2"])
    assert int(design.psi_parameter_index.loc["eta2", "eta2"]) == 3
    assert len(design.disturbance_parameters) == 1
    assert design.disturbance_parameters[0].latent == "eta2"


def test_build_structural_design_warns_cycle() -> None:
    spec = parse_model(
        "eta1 =~ x1 + x2 + x3\n"
        "eta2 =~ y1 + y2 + y3\n"
        "eta1 ~ eta2\n"
        "eta2 ~ eta1"
    )
    design = build_structural_design(spec)
    assert any("cycle" in warning.lower() for warning in design.warnings)


def test_build_structural_design_rejects_unknown_target_if_modelspec_inconsistent() -> None:
    spec = parse_model("eta =~ x1 + x2 + x3\ny ~ eta")
    from psysem.sem.model import ModelSpec

    broken = ModelSpec(
        source=spec.source,
        syntax=spec.syntax,
        relations=spec.relations,
        observed_variables=tuple(name for name in spec.observed_variables if name != "y"),
        latent_variables=spec.latent_variables,
        estimator=spec.estimator,
        block_names=spec.block_names,
        warnings=spec.warnings,
        constraints=spec.constraints,
        parsed_constraints=spec.parsed_constraints,
    )
    with pytest.raises(ValueError, match="Unknown structural target"):
        build_structural_design(broken)


def test_build_structural_design_with_parameter_table_uses_given_indices() -> None:
    spec = parse_model("eta =~ x1 + x2 + x3\ny ~ a*eta + 0.5*x1 + x2")
    parameter_table = (
        {
            "relation_index": 2,
            "term_index": 1,
            "lhs": "y",
            "operator": "~",
            "rhs": "eta",
            "is_free": True,
            "parameter": "a",
            "parameter_index": 10,
            "vector_position": 0,
            "fixed_value": None,
        },
        {
            "relation_index": 2,
            "term_index": 2,
            "lhs": "y",
            "operator": "~",
            "rhs": "x1",
            "is_free": False,
            "parameter": None,
            "parameter_index": None,
            "vector_position": None,
            "fixed_value": 0.5,
        },
        {
            "relation_index": 2,
            "term_index": 3,
            "lhs": "y",
            "operator": "~",
            "rhs": "x2",
            "is_free": True,
            "parameter": "p1",
            "parameter_index": 11,
            "vector_position": 1,
            "fixed_value": None,
        },
    )
    design = build_structural_design(spec, parameter_table=parameter_table)
    free_indices = [item.parameter_index for item in design.path_table if item.is_free]
    assert free_indices == [10, 11]
    free_positions = [item.vector_position for item in design.path_table if item.is_free]
    assert free_positions == [0, 1]
    assert len(design.disturbance_parameters) == 0


def test_build_structural_design_uses_psi_rows_from_parameter_table() -> None:
    spec = parse_model(
        "eta1 =~ x1 + x2 + x3\n"
        "eta2 =~ y1 + y2 + y3\n"
        "eta2 ~ eta1"
    )
    parameter_table = (
        {
            "relation_index": 3,
            "term_index": 1,
            "lhs": "eta2",
            "operator": "~",
            "rhs": "eta1",
            "is_free": True,
            "parameter": "b1",
            "parameter_index": 5,
            "vector_position": 2,
            "fixed_value": None,
        },
        {
            "relation_index": 4,
            "term_index": 1,
            "lhs": "eta2",
            "operator": "~~",
            "rhs": "eta2",
            "is_free": True,
            "parameter": "psi_eta2",
            "parameter_index": 9,
            "vector_position": 8,
            "fixed_value": None,
        },
    )
    design = build_structural_design(spec, parameter_table=parameter_table)
    assert int(design.gamma_parameter_index.loc["eta2", "eta1"]) == 5
    assert int(design.psi_parameter_index.loc["eta2", "eta2"]) == 9
    assert len(design.disturbance_parameters) == 1
    assert design.disturbance_parameters[0].vector_position == 8


def test_check_structural_validity_reports_empty_paths() -> None:
    spec = parse_model(
        "eta1 =~ x1 + x2 + x3\n"
        "eta2 =~ y1 + y2 + y3\n"
        "eta2 ~ eta1"
    )
    design = build_structural_design(spec)
    warnings = check_structural_validity(design)
    assert isinstance(warnings, tuple)
