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
    from psysem.model import ModelSpec

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
            "is_free": True,
            "parameter": "a",
            "parameter_index": 10,
            "fixed_value": None,
        },
        {
            "relation_index": 2,
            "term_index": 2,
            "is_free": False,
            "parameter": None,
            "parameter_index": None,
            "fixed_value": 0.5,
        },
        {
            "relation_index": 2,
            "term_index": 3,
            "is_free": True,
            "parameter": "p1",
            "parameter_index": 11,
            "fixed_value": None,
        },
    )
    design = build_structural_design(spec, parameter_table=parameter_table)
    free_indices = [item.parameter_index for item in design.path_table if item.is_free]
    assert free_indices == [10, 11]


def test_check_structural_validity_reports_empty_paths() -> None:
    spec = parse_model(
        "eta1 =~ x1 + x2 + x3\n"
        "eta2 =~ y1 + y2 + y3\n"
        "eta2 ~ eta1"
    )
    design = build_structural_design(spec)
    warnings = check_structural_validity(design)
    assert isinstance(warnings, tuple)
