from __future__ import annotations

from typing import Any

import math

from .data import ESEMSpec
from .estimation import build_ml_context, optimize_ml_parameters
from .model import ModelSpec, parse_model
from .model import model_spec_from_esem_spec
from .measurement import MeasurementDesign, build_measurement_design, check_measurement_identification
from .parameter_index import ParameterIndexMap, build_parameter_index_map
from .result import SEMResult
from .structural import StructuralDesign, build_structural_design, check_structural_validity


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
        raw_parameter_table = _build_parameter_table(model_spec)
        parameter_index_map = build_parameter_index_map(raw_parameter_table)
        parameter_table = _attach_vector_positions(raw_parameter_table, parameter_index_map)
        parameters = _build_parameters(parameter_index_map)
        measurement_design = _try_build_measurement_design(model_spec, parameter_table)
        structural_design = _try_build_structural_design(model_spec, parameter_table)
        converged = True
        warnings: list[str] = []
        if measurement_design is not None:
            warnings.extend(check_measurement_identification(measurement_design))
        if structural_design is not None:
            warnings.extend(check_structural_validity(structural_design))

        optimization_info: dict[str, Any] = {
            "status": "placeholder",
            "n_iter": 0,
            "objective": float("nan"),
            "n_free_parameters": parameter_index_map.n_free,
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
            optimization_info["n_structural_disturbance_parameters"] = len(
                structural_design.disturbance_parameters
            )

        if estimator in {"ml", "mlr"}:
            ml_context = build_ml_context(data, observed_variables=model_spec.observed_variables)
            warnings.extend(ml_context.warnings)
            optimization_info["ml_n_sample_observed"] = len(ml_context.observed_variables)
            optimization_info["ml_has_sample_covariance"] = ml_context.sample_covariance is not None
            optimization_info["ml_objective_at_sample_cov"] = (
                ml_context.objective_at_sample_cov
                if ml_context.objective_at_sample_cov is not None
                else float("nan")
            )
            optimization_info["ml_optimized"] = False

            should_optimize_ml = (
                measurement_design is not None
                and parameter_index_map.n_free > 0
                and n_obs >= _minimum_ml_n_obs(measurement_design)
            )
            if should_optimize_ml and measurement_design is not None:
                ml_optimization = optimize_ml_parameters(
                    data,
                    measurement_design=measurement_design,
                    structural_design=structural_design,
                    parameter_index_map=parameter_index_map,
                    parameter_table=parameter_table,
                )
                warnings.extend(ml_optimization.warnings)
                optimization_info["ml_optimized"] = True
                optimization_info["ml_optimization_success"] = ml_optimization.success
                optimization_info["status"] = ml_optimization.status
                optimization_info["n_iter"] = ml_optimization.n_iter
                optimization_info["objective"] = (
                    ml_optimization.objective
                    if ml_optimization.objective is not None and math.isfinite(ml_optimization.objective)
                    else float("nan")
                )
                optimization_info["ml_n_optimized_observed"] = len(ml_optimization.observed_variables)
                if ml_optimization.parameter_values:
                    parameters = {
                        name: ml_optimization.parameter_values.get(name, value)
                        for name, value in parameters.items()
                    }
                converged = ml_optimization.success
            elif measurement_design is None:
                warnings.append("ML optimization skipped: measurement design is required in prototype.")
            elif n_obs < _minimum_ml_n_obs(measurement_design):
                warnings.append(
                    "ML optimization skipped: sample size below prototype threshold "
                    f"(n={n_obs}, required>={_minimum_ml_n_obs(measurement_design)})."
                )

        return SEMResult(
            converged=converged,
            n_obs=n_obs,
            parameters=parameters,
            fit_indices={},
            parameter_table=parameter_table,
            warnings=tuple(dict.fromkeys(warnings)),
            optimization_info=optimization_info,
            estimator=estimator,
            model_spec=model_spec,
            parameter_index_map=parameter_index_map,
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

    relation_count = len(model_spec.relations)
    for offset, latent in enumerate(_collect_endogenous_latent_for_disturbance(model_spec), start=1):
        parameter_index = next_parameter_index
        parameter_name = f"p{unnamed_counter}"
        next_parameter_index += 1
        unnamed_counter += 1
        rows.append(
            {
                "relation_index": relation_count + offset,
                "term_index": 1,
                "lhs": latent,
                "operator": "~~",
                "rhs": latent,
                "label": None,
                "fixed_value": None,
                "is_free": True,
                "parameter": parameter_name,
                "parameter_index": parameter_index,
            }
        )
    return tuple(rows)


def _build_parameters(parameter_index_map: ParameterIndexMap) -> dict[str, float]:
    return {
        entry.parameter: float("nan")
        for entry in parameter_index_map.entries
    }


def _attach_vector_positions(
    parameter_table: tuple[dict[str, Any], ...],
    parameter_index_map: ParameterIndexMap,
) -> tuple[dict[str, Any], ...]:
    index_to_position = parameter_index_map.index_to_position()
    rows: list[dict[str, Any]] = []
    for row in parameter_table:
        row_with_position = dict(row)
        if bool(row_with_position["is_free"]):
            parameter_index = row_with_position["parameter_index"]
            if isinstance(parameter_index, int):
                row_with_position["vector_position"] = index_to_position.get(parameter_index)
            else:
                row_with_position["vector_position"] = None
        else:
            row_with_position["vector_position"] = None
        rows.append(row_with_position)
    return tuple(rows)


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


def _collect_endogenous_latent_for_disturbance(model_spec: ModelSpec) -> tuple[str, ...]:
    latent_set = set(model_spec.latent_variables)
    endogenous_latent: list[str] = []
    seen: set[str] = set()
    for relation in model_spec.relations:
        if relation.operator != "~":
            continue
        if relation.lhs not in latent_set:
            continue
        if relation.lhs in seen:
            continue
        seen.add(relation.lhs)
        endogenous_latent.append(relation.lhs)
    return tuple(endogenous_latent)


def _minimum_ml_n_obs(measurement_design: MeasurementDesign) -> int:
    return max(20, len(measurement_design.observed_variables) * 3)
