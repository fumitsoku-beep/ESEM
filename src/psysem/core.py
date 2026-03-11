from __future__ import annotations

from typing import Any

from .data import ESEMSpec
from .model import ModelSpec, parse_model
from .model import model_spec_from_esem_spec
from .measurement import MeasurementDesign, build_measurement_design, check_measurement_identification
from .structural import StructuralDesign, build_structural_design, check_structural_validity
from .result import SEMResult


class SEMModel:
    """Minimal SEM model entry point.

    This class intentionally provides a small stable API first.
    Numerical estimation will be added in later milestones.
    """

    def __init__(self, syntax: str | None = None):
        self.spec: ModelSpec | None = parse_model(syntax) if syntax is not None else None

    def fit(self, data: Any, *, spec: ESEMSpec | None = None) -> SEMResult:
        """Fit model to data and return placeholder result.

        Parameters
        ----------
        data:
            Any tabular object with a reliable ``len(data)``.
        spec:
            Optional ESEMSpec model definition. Use this when SEMModel is
            initialized without syntax.
        """
        n_obs = _resolve_n_obs(data)
        model_spec = self._resolve_model_spec(spec)
        estimator = (model_spec.estimator or "ml").lower()
        parameter_table = _build_parameter_table(model_spec)
        parameters = _build_parameters(parameter_table)
        measurement_design = _try_build_measurement_design(model_spec, parameter_table)
        structural_design = _try_build_structural_design(model_spec, parameter_table)
        warnings: list[str] = []
        if measurement_design is not None:
            warnings.extend(check_measurement_identification(measurement_design))
        if structural_design is not None:
            warnings.extend(check_structural_validity(structural_design))

        optimization_info: dict[str, Any] = {
            "status": "placeholder",
            "n_iter": 0,
            "objective": float("nan"),
            "n_free_parameters": len(parameters),
            "n_constraints": len(model_spec.constraints),
        }
        if measurement_design is not None:
            optimization_info["n_measurement_latent"] = len(measurement_design.latent_variables)
            optimization_info["n_measurement_observed"] = len(measurement_design.observed_variables)
        if structural_design is not None:
            optimization_info["n_structural_paths"] = len(structural_design.path_table)
            optimization_info["n_structural_endogenous_latent"] = len(
                structural_design.endogenous_latent_variables
            )

        return SEMResult(
            converged=True,
            n_obs=n_obs,
            parameters=parameters,
            fit_indices={},
            parameter_table=parameter_table,
            warnings=tuple(dict.fromkeys(warnings)),
            optimization_info=optimization_info,
            estimator=estimator,
            model_spec=model_spec,
            measurement_design=measurement_design,
            structural_design=structural_design,
        )

    def _resolve_model_spec(self, spec: ESEMSpec | None) -> ModelSpec:
        if spec is not None:
            if self.spec is not None:
                raise ValueError(
                    "Model is already defined by syntax. Provide either syntax in "
                    "`SEMModel(...)` or `spec` in `fit(...)`, not both."
                )
            return model_spec_from_esem_spec(spec)
        if self.spec is None:
            raise ValueError(
                "No model definition found. Provide syntax in `SEMModel(...)` "
                "or pass `spec` to `fit(...)`."
            )
        return self.spec


def sem(syntax: str, data: Any) -> SEMResult:
    """Convenience API for one-off model fitting."""
    return SEMModel(syntax).fit(data)


def _resolve_n_obs(data: Any) -> int:
    try:
        return int(len(data))
    except TypeError as exc:
        raise TypeError("`data` must be a sized tabular object.") from exc


def _build_parameter_table(model_spec: ModelSpec) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    next_parameter_index = 1
    unnamed_counter = 1
    label_registry: dict[str, int] = {}
    for relation_index, relation in enumerate(model_spec.relations, start=1):
        for term_index, term in enumerate(relation.terms, start=1):
            fixed_value = term.coefficient
            is_free = fixed_value is None
            parameter_name: str | None
            parameter_index: int | None
            if not is_free:
                parameter_name = None
                parameter_index = None
            elif term.label is not None:
                parameter_index = label_registry.get(term.label)
                if parameter_index is None:
                    parameter_index = next_parameter_index
                    label_registry[term.label] = parameter_index
                    next_parameter_index += 1
                parameter_name = term.label
            else:
                parameter_index = next_parameter_index
                parameter_name = f"p{unnamed_counter}"
                next_parameter_index += 1
                unnamed_counter += 1

            rows.append(
                {
                    "relation_index": relation_index,
                    "term_index": term_index,
                    "lhs": relation.lhs,
                    "operator": relation.operator,
                    "rhs": term.variable,
                    "label": term.label,
                    "fixed_value": fixed_value,
                    "is_free": is_free,
                    "parameter": parameter_name,
                    "parameter_index": parameter_index,
                }
            )
    return tuple(rows)


def _build_parameters(parameter_table: tuple[dict[str, Any], ...]) -> dict[str, float]:
    parameters: dict[str, float] = {}
    for row in parameter_table:
        if not bool(row["is_free"]):
            continue
        parameter_name = row["parameter"]
        if isinstance(parameter_name, str):
            parameters.setdefault(parameter_name, float("nan"))
    return parameters


def _try_build_measurement_design(
    model_spec: ModelSpec,
    parameter_table: tuple[dict[str, Any], ...],
) -> MeasurementDesign | None:
    has_measurement = any(relation.operator == "=~" for relation in model_spec.relations)
    if not has_measurement:
        return None
    return build_measurement_design(model_spec, parameter_table=parameter_table)


def _try_build_structural_design(
    model_spec: ModelSpec,
    parameter_table: tuple[dict[str, Any], ...],
) -> StructuralDesign | None:
    has_structural = any(relation.operator == "~" for relation in model_spec.relations)
    if not has_structural:
        return None
    return build_structural_design(model_spec, parameter_table=parameter_table)
