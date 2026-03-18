from __future__ import annotations

from typing import Any

import math
import numpy as np

from ..data import ESEMSpec
from .estimation import (
	SEMFitConfig,
	build_implied_covariance,
	build_ml_context,
	gaussian_ml_discrepancy,
	optimize_ml_parameters,
)
from .inference import estimate_parameter_inference
from .measurement import MeasurementDesign, build_measurement_design, check_measurement_identification
from .structural import StructuralDesign, build_structural_design, check_structural_validity
from .fit_indices import compute_basic_fit_indices, compute_fit_indices
from .model import ModelSpec, model_spec_from_esem_spec, parse_model
from .parameter_index import ParameterIndexMap, build_parameter_index_map
from .result import SEMResult


class SEMModel:
	"""Minimal SEM model entry point.

	This class intentionally provides a small stable API first.
	Numerical estimation will be added in later milestones.
	"""

	def __init__(self, syntax: str | None = None):
		self.spec: ModelSpec | None = parse_model(syntax) if syntax is not None else None

	def fit(
		self,
		data: Any,
		*,
		spec: ESEMSpec | None = None,
		fit_config: SEMFitConfig | None = None,
	) -> SEMResult:
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
		if fit_config is not None and not isinstance(fit_config, SEMFitConfig):
			raise TypeError("`fit_config` must be a SEMFitConfig instance or None.")
		resolved_fit_config = SEMFitConfig() if fit_config is None else fit_config
		estimator = (model_spec.estimator or "ml").lower()
		raw_parameter_table = _build_parameter_table(model_spec)
		parameter_index_map = build_parameter_index_map(raw_parameter_table)
		parameter_table = _attach_vector_positions(raw_parameter_table, parameter_index_map)
		parameters = _build_parameters(parameter_index_map)
		measurement_design = _try_build_measurement_design(model_spec, parameter_table)
		structural_design = _try_build_structural_design(model_spec, parameter_table)
		converged = True
		parameter_inference: tuple[dict[str, Any], ...] = ()
		fit_indices = compute_basic_fit_indices()
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
			"fit_method": resolved_fit_config.method,
			"fit_max_iter": resolved_fit_config.max_iter,
			"fit_tol": resolved_fit_config.tol,
			"fit_restarts": resolved_fit_config.restarts,
			"fit_random_start_scale": resolved_fit_config.random_start_scale,
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
					fit_config=resolved_fit_config,
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
				optimization_info["ml_n_attempts"] = ml_optimization.n_attempts
				optimization_info["ml_best_attempt"] = (
					ml_optimization.best_attempt if ml_optimization.best_attempt is not None else -1
				)
				optimization_info["ml_method"] = ml_optimization.method or resolved_fit_config.method
				if ml_optimization.failure_category is not None:
					optimization_info["ml_failure_category"] = ml_optimization.failure_category
				if ml_optimization.failure_reason is not None:
					optimization_info["ml_failure_reason"] = ml_optimization.failure_reason
				if ml_optimization.parameter_values:
					parameters = {
						name: ml_optimization.parameter_values.get(name, value)
						for name, value in parameters.items()
					}
				if (
					ml_optimization.sample_covariance is not None
					and ml_optimization.implied_covariance is not None
				):
					fit_result = compute_fit_indices(
						sample_covariance=ml_optimization.sample_covariance.to_numpy(dtype=float),
						implied_covariance=ml_optimization.implied_covariance.to_numpy(dtype=float),
						n_obs=n_obs,
						n_free_parameters=parameter_index_map.n_free,
						objective=ml_optimization.objective,
					)
					fit_indices = fit_result.indices
					warnings.extend(fit_result.warnings)
					if fit_result.status == "failed":
						warnings.append(
							"SEM fit indices are unavailable for the current solution."
						)
					elif fit_result.status == "partial":
						warnings.append(
							"SEM fit indices are partially available; some fit measures could not be computed."
						)
					optimization_info["fit_status"] = fit_result.status
					optimization_info["n_fit_indices_available"] = fit_result.n_available_indices
					optimization_info["n_fit_indices_unavailable"] = fit_result.n_unavailable_indices
					if fit_result.failure_reason is not None:
						optimization_info["fit_failure_reason"] = fit_result.failure_reason
					if fit_result.chi_square is not None:
						optimization_info["chi_square"] = fit_result.chi_square
					if fit_result.df_model is not None:
						optimization_info["df_model"] = fit_result.df_model
					if fit_result.chi_square_baseline is not None:
						optimization_info["chi_square_baseline"] = fit_result.chi_square_baseline
					if fit_result.df_baseline is not None:
						optimization_info["df_baseline"] = fit_result.df_baseline
				if (
					ml_optimization.success
					and ml_optimization.sample_covariance is not None
					and measurement_design is not None
					and len(ml_optimization.parameter_vector) == parameter_index_map.n_free
				):
					sample_cov_array = ml_optimization.sample_covariance.to_numpy(dtype=float)

					def objective_for_inference(vector) -> float:
						implied = build_implied_covariance(
							measurement_design,
							vector,
							parameter_index_map,
							structural_design=structural_design,
							observed_variables=ml_optimization.observed_variables,
						)
						return gaussian_ml_discrepancy(
							sample_cov_array,
							implied.to_numpy(dtype=float),
						)

					inference_result = estimate_parameter_inference(
						objective_fn=objective_for_inference,
						parameter_vector=np.array(ml_optimization.parameter_vector, dtype=float),
						parameter_index_map=parameter_index_map,
					)
					warnings.extend(inference_result.warnings)
					if inference_result.status == "failed":
						warnings.append(
							"Parameter inference could not compute standard errors for the current SEM solution."
						)
					elif inference_result.status == "partial":
						warnings.append(
							"Parameter inference is partially available; some SEM standard errors could not be computed."
						)
					parameter_inference = tuple(
						{
							"parameter": entry.parameter,
							"parameter_index": entry.parameter_index,
							"vector_position": entry.vector_position,
							"estimate": entry.estimate,
							"standard_error": entry.standard_error,
							"z_value": entry.z_value,
							"p_value": entry.p_value,
							"ci_lower": entry.ci_lower,
							"ci_upper": entry.ci_upper,
						}
						for entry in inference_result.entries
					)
					optimization_info["inference_status"] = inference_result.status
					optimization_info["inference_covariance_method"] = (
						inference_result.covariance_method or "none"
					)
					optimization_info["n_inference_warnings"] = len(inference_result.warnings)
					optimization_info["n_inference_parameters"] = len(parameter_inference)
					optimization_info["n_inference_with_se"] = inference_result.n_with_standard_error
					optimization_info["n_inference_without_se"] = inference_result.n_without_standard_error
					if inference_result.failure_reason is not None:
						optimization_info["inference_failure_reason"] = inference_result.failure_reason
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
			fit_indices=fit_indices,
			parameter_table=parameter_table,
			warnings=tuple(dict.fromkeys(warnings)),
			optimization_info=optimization_info,
			estimator=estimator,
			model_spec=model_spec,
			parameter_index_map=parameter_index_map,
			parameter_inference=parameter_inference,
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


def sem(
	syntax: str,
	data: Any,
	*,
	fit_config: SEMFitConfig | None = None,
) -> SEMResult:
	"""Convenience API for one-off model fitting."""
	return SEMModel(syntax).fit(data, fit_config=fit_config)


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

	relation_cursor = len(model_spec.relations)
	for observed in _collect_measurement_observed_for_residual(model_spec):
		parameter_index = next_parameter_index
		parameter_name = f"p{unnamed_counter}"
		next_parameter_index += 1
		unnamed_counter += 1
		relation_cursor += 1
		rows.append(
			{
				"relation_index": relation_cursor,
				"term_index": 1,
				"lhs": observed,
				"operator": "~~",
				"rhs": observed,
				"label": None,
				"fixed_value": None,
				"is_free": True,
				"parameter": parameter_name,
				"parameter_index": parameter_index,
			}
		)

	for latent in _collect_endogenous_latent_for_disturbance(model_spec):
		parameter_index = next_parameter_index
		parameter_name = f"p{unnamed_counter}"
		next_parameter_index += 1
		unnamed_counter += 1
		relation_cursor += 1
		rows.append(
			{
				"relation_index": relation_cursor,
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


def _collect_measurement_observed_for_residual(model_spec: ModelSpec) -> tuple[str, ...]:
	observed: list[str] = []
	seen: set[str] = set()
	for relation in model_spec.relations:
		if relation.operator != "=~":
			continue
		for term in relation.terms:
			name = term.variable
			if name in seen:
				continue
			seen.add(name)
			observed.append(name)
	return tuple(observed)


def _minimum_ml_n_obs(measurement_design: MeasurementDesign) -> int:
	return max(20, len(measurement_design.observed_variables) * 3)


__all__ = ["SEMModel", "sem"]
