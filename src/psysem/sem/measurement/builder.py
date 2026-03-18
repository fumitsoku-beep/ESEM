from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..model import ModelSpec
from .contracts import LoadingParameter, MeasurementDesign


def build_measurement_design(
	model_spec: ModelSpec,
	*,
	parameter_table: tuple[dict[str, Any], ...] | None = None,
) -> MeasurementDesign:
	"""Build Lambda/Theta placeholders from measurement relations."""
	measurement_relations = tuple(
		(relation_index, relation)
		for relation_index, relation in enumerate(model_spec.relations, start=1)
		if relation.operator == "=~"
	)
	if not measurement_relations:
		raise ValueError("No measurement relations (`=~`) found in model.")

	parameter_lookup = _parameter_lookup(parameter_table)
	theta_lookup = _theta_parameter_lookup(parameter_table)
	theta_cov_lookup = _theta_covariance_lookup(parameter_table)
	phi_lookup = _phi_parameter_lookup(parameter_table)
	latent_order: list[str] = []
	latent_seen: set[str] = set()
	observed_order: list[str] = []
	observed_seen: set[str] = set()
	indicator_counts: dict[str, int] = {}
	has_fixed_marker: dict[str, bool] = {}
	free_loadings: list[tuple[str, str]] = []
	fixed_loadings: list[tuple[str, str, float]] = []
	loading_parameters: list[LoadingParameter] = []
	block_latent_pairs: list[tuple[str, str]] = []
	block_latent_seen: set[tuple[str, str]] = set()
	fallback_label_index: dict[str, int] = {}
	fallback_next_index = 1
	indicator_fixed_count: dict[str, int] = {}
	latent_fixed_count: dict[str, int] = {}
	latent_free_count: dict[str, int] = {}

	for relation_index, relation in measurement_relations:
		latent = relation.lhs
		if latent not in latent_seen:
			latent_seen.add(latent)
			latent_order.append(latent)
		indicator_counts.setdefault(latent, 0)
		has_fixed_marker.setdefault(latent, False)
		latent_fixed_count.setdefault(latent, 0)
		latent_free_count.setdefault(latent, 0)

		block_name = _resolve_block_name(latent, model_spec.block_names)
		if block_name is not None:
			pair = (block_name, latent)
			if pair not in block_latent_seen:
				block_latent_seen.add(pair)
				block_latent_pairs.append(pair)

		for term_index, term in enumerate(relation.terms, start=1):
			observed = term.variable
			if observed not in observed_seen:
				observed_seen.add(observed)
				observed_order.append(observed)

			indicator_counts[latent] += 1
			param_meta = parameter_lookup.get((relation_index, term_index))
			if param_meta is None:
				(
					is_free,
					parameter_name,
					parameter_index,
					vector_position,
					fixed_value,
					fallback_next_index,
				) = _fallback_parameter_meta(
					term=term,
					fallback_label_index=fallback_label_index,
					next_parameter_index=fallback_next_index,
				)
			else:
				is_free, parameter_name, parameter_index, vector_position, fixed_value = param_meta
				if term.coefficient is not None and fixed_value is None:
					raise ValueError(
						"Parameter table/meta mismatch: fixed loading term has no fixed value."
					)

			if is_free:
				free_loadings.append((observed, latent))
				latent_free_count[latent] += 1
			else:
				assert fixed_value is not None
				fixed_loadings.append((observed, latent, fixed_value))
				has_fixed_marker[latent] = True
				latent_fixed_count[latent] += 1
				indicator_fixed_count[observed] = indicator_fixed_count.get(observed, 0) + 1

			loading_parameters.append(
				LoadingParameter(
					observed=observed,
					latent=latent,
					is_free=is_free,
					parameter=parameter_name,
					parameter_index=parameter_index,
					vector_position=vector_position,
					fixed_value=fixed_value,
					relation_index=relation_index,
					term_index=term_index,
					block_name=block_name,
				)
			)

	for latent, count in indicator_counts.items():
		if count < 2:
			raise ValueError(f"Latent `{latent}` has fewer than 2 indicators.")

	lambda_matrix = pd.DataFrame(
		0.0,
		index=observed_order,
		columns=latent_order,
	)
	lambda_parameter_index = pd.DataFrame(
		0,
		index=observed_order,
		columns=latent_order,
		dtype=int,
	)
	for observed, latent in free_loadings:
		lambda_matrix.loc[observed, latent] = np.nan
	for item in loading_parameters:
		if item.is_free:
			if item.parameter_index is not None:
				lambda_parameter_index.loc[item.observed, item.latent] = item.parameter_index
			continue
		assert item.fixed_value is not None
		lambda_matrix.loc[item.observed, item.latent] = item.fixed_value

	theta_matrix = pd.DataFrame(
		0.0,
		index=observed_order,
		columns=observed_order,
	)
	theta_parameter_index = pd.DataFrame(
		0,
		index=observed_order,
		columns=observed_order,
		dtype=int,
	)
	theta_next_index = _infer_next_parameter_index(parameter_table, default=fallback_next_index)
	for observed in observed_order:
		theta_meta = theta_lookup.get(observed)
		if theta_meta is None:
			theta_matrix.loc[observed, observed] = np.nan
			theta_parameter_index.loc[observed, observed] = theta_next_index
			theta_next_index += 1
			continue

		(
			is_free,
			_parameter_name,
			parameter_index,
			_vector_position,
			fixed_value,
		) = theta_meta
		if is_free:
			if parameter_index is None:
				raise ValueError(
					f"Free Theta parameter for observed `{observed}` has no `parameter_index`."
				)
			theta_matrix.loc[observed, observed] = np.nan
			theta_parameter_index.loc[observed, observed] = parameter_index
		else:
			if fixed_value is None:
				raise ValueError(
					f"Fixed Theta parameter for observed `{observed}` has no fixed value."
				)
			theta_matrix.loc[observed, observed] = float(fixed_value)

	for (lhs, rhs), theta_meta in theta_cov_lookup.items():
		if lhs not in observed_seen or rhs not in observed_seen:
			continue
		(
			is_free,
			_parameter_name,
			parameter_index,
			_vector_position,
			fixed_value,
		) = theta_meta
		if is_free:
			if parameter_index is None:
				raise ValueError(
					"Free Theta covariance parameter has no `parameter_index`: "
					f"{lhs} ~~ {rhs}."
				)
			theta_matrix.loc[lhs, rhs] = np.nan
			theta_matrix.loc[rhs, lhs] = np.nan
			theta_parameter_index.loc[lhs, rhs] = parameter_index
			theta_parameter_index.loc[rhs, lhs] = parameter_index
		else:
			if fixed_value is None:
				raise ValueError(
					"Fixed Theta covariance parameter has no fixed value: "
					f"{lhs} ~~ {rhs}."
				)
			theta_matrix.loc[lhs, rhs] = float(fixed_value)
			theta_matrix.loc[rhs, lhs] = float(fixed_value)

	phi_matrix = pd.DataFrame(
		0.0,
		index=latent_order,
		columns=latent_order,
	)
	phi_parameter_index = pd.DataFrame(
		0,
		index=latent_order,
		columns=latent_order,
		dtype=int,
	)
	phi_next_index = _infer_next_parameter_index(parameter_table, default=theta_next_index)
	for latent in latent_order:
		phi_meta = phi_lookup.get((latent, latent))
		if phi_meta is None:
			phi_matrix.loc[latent, latent] = np.nan
			phi_parameter_index.loc[latent, latent] = phi_next_index
			phi_next_index += 1
			continue
		(
			is_free,
			_parameter_name,
			parameter_index,
			_vector_position,
			fixed_value,
		) = phi_meta
		if is_free:
			if parameter_index is None:
				raise ValueError(f"Free Phi variance for latent `{latent}` has no `parameter_index`.")
			phi_matrix.loc[latent, latent] = np.nan
			phi_parameter_index.loc[latent, latent] = parameter_index
		else:
			if fixed_value is None:
				raise ValueError(f"Fixed Phi variance for latent `{latent}` has no fixed value.")
			phi_matrix.loc[latent, latent] = float(fixed_value)

	for (lhs, rhs), phi_meta in phi_lookup.items():
		if lhs == rhs or lhs not in latent_seen or rhs not in latent_seen:
			continue
		(
			is_free,
			_parameter_name,
			parameter_index,
			_vector_position,
			fixed_value,
		) = phi_meta
		if is_free:
			if parameter_index is None:
				raise ValueError(
					f"Free Phi covariance parameter has no `parameter_index`: {lhs} ~~ {rhs}."
				)
			phi_matrix.loc[lhs, rhs] = np.nan
			phi_matrix.loc[rhs, lhs] = np.nan
			phi_parameter_index.loc[lhs, rhs] = parameter_index
			phi_parameter_index.loc[rhs, lhs] = parameter_index
		else:
			if fixed_value is None:
				raise ValueError(
					f"Fixed Phi covariance parameter has no fixed value: {lhs} ~~ {rhs}."
				)
			phi_matrix.loc[lhs, rhs] = float(fixed_value)
			phi_matrix.loc[rhs, lhs] = float(fixed_value)

	warnings: list[str] = []
	for latent in latent_order:
		if not has_fixed_marker.get(latent, False):
			warnings.append(
				f"Latent `{latent}` has no fixed loading marker; scale may be under-identified."
			)
		if latent_fixed_count.get(latent, 0) > 1:
			warnings.append(
				f"Latent `{latent}` has multiple fixed loadings; check marker strategy."
			)
		if latent_free_count.get(latent, 0) == 0:
			warnings.append(f"Latent `{latent}` has no free loadings; model may be over-constrained.")

	for observed, count in indicator_fixed_count.items():
		if count > 1:
			warnings.append(
				f"Observed `{observed}` has fixed loadings on multiple latents; check identification."
			)
	if not free_loadings:
		warnings.append("No free loadings found in measurement design.")

	return MeasurementDesign(
		observed_variables=tuple(observed_order),
		latent_variables=tuple(latent_order),
		lambda_matrix=lambda_matrix,
		lambda_parameter_index=lambda_parameter_index,
		theta_matrix=theta_matrix,
		theta_parameter_index=theta_parameter_index,
		phi_matrix=phi_matrix,
		phi_parameter_index=phi_parameter_index,
		loading_parameters=tuple(loading_parameters),
		block_latent_pairs=tuple(block_latent_pairs),
		free_loadings=tuple(free_loadings),
		fixed_loadings=tuple(fixed_loadings),
		warnings=tuple(dict.fromkeys(warnings)),
	)


def _parameter_lookup(
	parameter_table: tuple[dict[str, Any], ...] | None,
) -> dict[tuple[int, int], tuple[bool, str | None, int | None, int | None, float | None]]:
	if parameter_table is None:
		return {}
	lookup: dict[
		tuple[int, int],
		tuple[bool, str | None, int | None, int | None, float | None],
	] = {}
	for row in parameter_table:
		relation_index = int(row["relation_index"])
		term_index = int(row["term_index"])
		key = (relation_index, term_index)
		fixed_raw = row["fixed_value"]
		vector_position = row.get("vector_position")
		lookup[key] = (
			bool(row["is_free"]),
			row["parameter"] if isinstance(row["parameter"], str) else None,
			int(row["parameter_index"]) if isinstance(row["parameter_index"], int) else None,
			int(vector_position) if isinstance(vector_position, int) else None,
			float(fixed_raw) if isinstance(fixed_raw, (int, float)) else None,
		)
	return lookup


def _fallback_parameter_meta(
	*,
	term,
	fallback_label_index: dict[str, int],
	next_parameter_index: int,
) -> tuple[bool, str | None, int | None, int | None, float | None, int]:
	fixed_value = term.coefficient
	if fixed_value is not None:
		return False, None, None, None, float(fixed_value), next_parameter_index

	if term.label is not None:
		index = fallback_label_index.get(term.label)
		if index is None:
			index = next_parameter_index
			fallback_label_index[term.label] = index
			next_parameter_index += 1
		return True, term.label, index, index - 1, None, next_parameter_index

	index = next_parameter_index
	parameter_name = f"p{index}"
	return True, parameter_name, index, index - 1, None, next_parameter_index + 1


def _theta_parameter_lookup(
	parameter_table: tuple[dict[str, Any], ...] | None,
) -> dict[str, tuple[bool, str | None, int | None, int | None, float | None]]:
	if parameter_table is None:
		return {}
	lookup: dict[str, tuple[bool, str | None, int | None, int | None, float | None]] = {}
	for row in parameter_table:
		operator = row.get("operator")
		lhs = row.get("lhs")
		rhs = row.get("rhs")
		if operator != "~~":
			continue
		if not isinstance(lhs, str) or not isinstance(rhs, str) or lhs != rhs:
			continue
		if lhs in lookup:
			raise ValueError(f"Duplicate Theta parameter row found for observed `{lhs}`.")
		fixed_raw = row["fixed_value"]
		vector_position = row.get("vector_position")
		lookup[lhs] = (
			bool(row["is_free"]),
			row["parameter"] if isinstance(row["parameter"], str) else None,
			int(row["parameter_index"]) if isinstance(row["parameter_index"], int) else None,
			int(vector_position) if isinstance(vector_position, int) else None,
			float(fixed_raw) if isinstance(fixed_raw, (int, float)) else None,
		)
	return lookup


def _theta_covariance_lookup(
	parameter_table: tuple[dict[str, Any], ...] | None,
) -> dict[tuple[str, str], tuple[bool, str | None, int | None, int | None, float | None]]:
	if parameter_table is None:
		return {}
	lookup: dict[
		tuple[str, str], tuple[bool, str | None, int | None, int | None, float | None]
	] = {}
	for row in parameter_table:
		operator = row.get("operator")
		lhs = row.get("lhs")
		rhs = row.get("rhs")
		if operator != "~~":
			continue
		if not isinstance(lhs, str) or not isinstance(rhs, str) or lhs == rhs:
			continue
		key = tuple(sorted((lhs, rhs)))
		if key in lookup:
			raise ValueError(
				"Duplicate Theta covariance parameter row found for observed pair "
				f"`{key[0]} ~~ {key[1]}`."
			)
		fixed_raw = row["fixed_value"]
		vector_position = row.get("vector_position")
		lookup[key] = (
			bool(row["is_free"]),
			row["parameter"] if isinstance(row["parameter"], str) else None,
			int(row["parameter_index"]) if isinstance(row["parameter_index"], int) else None,
			int(vector_position) if isinstance(vector_position, int) else None,
			float(fixed_raw) if isinstance(fixed_raw, (int, float)) else None,
		)
	return lookup


def _phi_parameter_lookup(
	parameter_table: tuple[dict[str, Any], ...] | None,
) -> dict[tuple[str, str], tuple[bool, str | None, int | None, int | None, float | None]]:
	if parameter_table is None:
		return {}
	lookup: dict[
		tuple[str, str], tuple[bool, str | None, int | None, int | None, float | None]
	] = {}
	for row in parameter_table:
		operator = row.get("operator")
		lhs = row.get("lhs")
		rhs = row.get("rhs")
		if operator != "~~":
			continue
		if not isinstance(lhs, str) or not isinstance(rhs, str):
			continue
		key = tuple(sorted((lhs, rhs)))
		fixed_raw = row["fixed_value"]
		vector_position = row.get("vector_position")
		lookup[key] = (
			bool(row["is_free"]),
			row["parameter"] if isinstance(row["parameter"], str) else None,
			int(row["parameter_index"]) if isinstance(row["parameter_index"], int) else None,
			int(vector_position) if isinstance(vector_position, int) else None,
			float(fixed_raw) if isinstance(fixed_raw, (int, float)) else None,
		)
	return lookup


def _infer_next_parameter_index(
	parameter_table: tuple[dict[str, Any], ...] | None,
	*,
	default: int,
) -> int:
	if parameter_table is None:
		return default
	max_index = 0
	for row in parameter_table:
		parameter_index = row.get("parameter_index")
		if isinstance(parameter_index, int):
			max_index = max(max_index, parameter_index)
	return max(default, max_index + 1)


def _resolve_block_name(latent: str, block_names: tuple[str, ...]) -> str | None:
	if not block_names:
		return None
	if "_f" not in latent:
		return None
	candidate, _, suffix = latent.rpartition("_f")
	if not suffix.isdigit():
		return None
	if candidate in set(block_names):
		return candidate
	return None


__all__ = ["build_measurement_design"]
