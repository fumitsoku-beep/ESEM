import pandas as pd
import pytest

from psysem import SEMModel, esem_spec_from_dict, parse_model, sem, to_markdown


def test_parse_model_rejects_empty() -> None:
    with pytest.raises(ValueError):
        parse_model("   ")


def test_parse_model_builds_structured_relations() -> None:
    spec = parse_model("eta =~ x1 + x2\ny ~ eta + x1")
    assert spec.source == "syntax"
    assert len(spec.relations) == 2
    assert spec.relations[0].operator == "=~"
    assert spec.relations[1].operator == "~"
    assert "eta" in spec.latent_variables
    assert "x1" in spec.observed_variables
    assert "y" in spec.observed_variables


def test_parse_model_supports_term_modifiers_and_constraints() -> None:
    spec = parse_model("eta =~ l1*x1 + 1.0*x2 + x3\ny ~ b1*eta + 0.5*x1 + x2\nb1 == l1")
    assert len(spec.relations) == 2
    measurement = spec.relations[0]
    structural = spec.relations[1]
    assert measurement.terms[0].label == "l1"
    assert measurement.terms[1].coefficient == pytest.approx(1.0)
    assert measurement.terms[2].coefficient is None
    assert structural.terms[0].label == "b1"
    assert structural.terms[1].coefficient == pytest.approx(0.5)
    assert spec.constraints == ("b1 == l1",)


def test_parse_model_supports_inequality_constraints() -> None:
    spec = parse_model("eta =~ x1 + x2\na >= 0\nb <= a")
    assert spec.constraints == ("a >= 0", "b <= a")
    assert spec.parsed_constraints[0].operator == ">="
    assert spec.parsed_constraints[1].operator == "<="


def test_parse_model_rejects_duplicate_path() -> None:
    with pytest.raises(ValueError, match="Duplicate path"):
        parse_model("y ~ x1\ny ~ x1")


def test_parse_model_rejects_invalid_variable_token() -> None:
    with pytest.raises(ValueError, match="Invalid variable token"):
        parse_model("y ~ 1x")


def test_parse_model_rejects_multiple_operators() -> None:
    with pytest.raises(ValueError, match="Multiple operators"):
        parse_model("y ~ x1 =~ x2")


def test_parse_model_rejects_invalid_term_modifier() -> None:
    with pytest.raises(ValueError, match="statement #1, term #1"):
        parse_model("y ~ x1*2")


def test_parse_model_requires_relation_when_only_constraints_given() -> None:
    with pytest.raises(ValueError, match="No relation expressions found"):
        parse_model("b1 == b2")


def test_parse_model_rejects_invalid_constraint_tokens() -> None:
    with pytest.raises(ValueError, match="Invalid token"):
        parse_model("y ~ x1\nb1 >= unknown-token")


def test_parse_model_rejects_malformed_constraint() -> None:
    with pytest.raises(ValueError, match="Invalid constraint"):
        parse_model("y ~ x1\nb1 ==")


def test_parse_model_error_includes_statement_index() -> None:
    with pytest.raises(ValueError, match="statement #2"):
        parse_model("y ~ x1\ny ~ x1 + ")


def test_smoke_fit() -> None:
    data = pd.DataFrame({"x1": [1.0, 2.0, 3.0], "x2": [1.0, 1.5, 2.5], "y": [1.1, 2.2, 3.1]})
    model = SEMModel("y ~ x1 + x2")
    result = model.fit(data)
    assert result.converged is True
    assert result.n_obs == 3
    assert result.estimator == "ml"
    assert result.model_spec is not None
    assert result.model_spec.source == "syntax"


def test_sem_function() -> None:
    data = pd.DataFrame({"x1": [1.0, 2.0], "y": [1.2, 2.4]})
    result = sem("y ~ x1", data)
    assert result.n_obs == 2


def test_fit_with_spec_when_model_initialized_without_syntax() -> None:
    spec = esem_spec_from_dict(_esem_payload())
    model = SEMModel()
    result = model.fit(_spec_data(), spec=spec)
    assert result.converged is True
    assert result.estimator == "mlr"
    assert result.model_spec is not None
    assert result.model_spec.source == "spec"
    assert "internalizing_f1" in result.model_spec.latent_variables


def test_fit_rejects_mixed_syntax_and_spec_inputs() -> None:
    spec = esem_spec_from_dict(_esem_payload())
    model = SEMModel("y ~ x1")
    with pytest.raises(ValueError, match="either syntax"):
        model.fit(_spec_data(), spec=spec)


def test_fit_requires_model_definition() -> None:
    model = SEMModel()
    with pytest.raises(ValueError, match="No model definition"):
        model.fit(_spec_data())


def test_fit_rejects_non_regression_structural_expression_in_spec() -> None:
    payload = _esem_payload()
    payload["structural"] = ["y ~~ i1"]
    spec = esem_spec_from_dict(payload)
    model = SEMModel()
    with pytest.raises(ValueError, match="only supports regression expressions"):
        model.fit(_spec_data(), spec=spec)


def test_summary_and_markdown_include_phase1_fields() -> None:
    data = pd.DataFrame({"x1": [1.0, 2.0, 3.0], "y": [1.2, 2.3, 3.1]})
    result = SEMModel("y ~ x1").fit(data)
    summary = result.summary()
    report = to_markdown(result)
    assert "Estimator: ml" in summary
    assert "Model source: syntax" in summary
    assert "Optimization:" in summary
    assert "n_free_parameters" in summary
    assert "Estimator: `ml`" in report
    assert "## Optimization" in report


def test_fit_builds_parameter_table_and_parameter_placeholders() -> None:
    data = pd.DataFrame(
        {
            "x1": [1.0, 2.0, 3.0],
            "x2": [1.5, 2.5, 3.5],
            "x3": [0.8, 1.8, 2.8],
            "y": [2.1, 3.2, 4.3],
        }
    )
    syntax = "eta =~ l1*x1 + 1.0*x2 + x3\ny ~ b1*eta + 0.5*x1 + x2\nb1 == l1"
    result = SEMModel(syntax).fit(data)
    assert len(result.parameter_table) == 6
    assert len(result.parameters) == 4
    assert set(result.parameters) == {"l1", "b1", "p1", "p2"}
    fixed_rows = [row for row in result.parameter_table if row["fixed_value"] is not None]
    assert len(fixed_rows) == 2
    assert any(row["fixed_value"] == pytest.approx(1.0) for row in fixed_rows)
    assert any(row["fixed_value"] == pytest.approx(0.5) for row in fixed_rows)


def test_fit_attaches_measurement_design() -> None:
    data = pd.DataFrame(
        {
            "x1": [1.0, 2.0, 3.0],
            "x2": [1.5, 2.5, 3.5],
            "x3": [0.8, 1.8, 2.8],
        }
    )
    result = SEMModel("eta =~ 1*x1 + x2 + x3").fit(data)
    assert result.measurement_design is not None
    assert result.measurement_design.lambda_matrix.shape == (3, 1)
    assert any("n_measurement_latent" == key for key in result.optimization_info)


def _esem_payload() -> dict[str, object]:
    return {
        "blocks": [
            {
                "name": "internalizing",
                "items": ["i1", "i2", "i3"],
                "n_factors": 1,
            }
        ],
        "estimator": "MLR",
        "variable_types": {
            "i1": "continuous",
            "i2": "continuous",
            "i3": "continuous",
            "y": "continuous",
        },
        "structural": ["y ~ internalizing_f1"],
    }


def _spec_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "i1": [1.0, 2.0, 3.0],
            "i2": [1.1, 2.1, 3.1],
            "i3": [0.9, 1.8, 2.7],
            "y": [2.0, 2.8, 3.7],
        }
    )
