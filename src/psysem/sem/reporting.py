from __future__ import annotations

from .result import SEMResult, _count_inference_rows_with_se, format_result_value


def to_markdown(result: SEMResult) -> str:
	"""Serialize a fit result into a simple markdown report block."""
	lines = ["# SEM Result", "", f"- Converged: `{result.converged}`", f"- N: `{result.n_obs}`"]
	if result.estimator:
		lines.append(f"- Estimator: `{result.estimator}`")
	if result.model_spec is not None:
		lines.append(f"- Model source: `{result.model_spec.source}`")
		lines.append(f"- Relations: `{len(result.model_spec.relations)}`")
	if result.parameter_index_map is not None:
		lines.append(f"- Free parameters: `{result.parameter_index_map.n_free}`")
	if result.parameter_inference:
		n_with_se = _count_inference_rows_with_se(result.parameter_inference)
		n_without_se = len(result.parameter_inference) - n_with_se
		lines.append(
			f"- Inference rows: `{len(result.parameter_inference)}` (`{n_with_se}` with SE, `{n_without_se}` without SE)"
		)
		inference_status = result.optimization_info.get("inference_status")
		if inference_status is not None:
			lines.append(f"- Inference status: `{inference_status}`")
		inference_failure_reason = result.optimization_info.get("inference_failure_reason")
		if inference_failure_reason is not None:
			lines.append(f"- Inference issue: `{inference_failure_reason}`")
	fit_status = result.optimization_info.get("fit_status")
	if fit_status is not None:
		lines.append(f"- Fit status: `{fit_status}`")
		n_fit_available = result.optimization_info.get("n_fit_indices_available")
		n_fit_unavailable = result.optimization_info.get("n_fit_indices_unavailable")
		if n_fit_available is not None and n_fit_unavailable is not None:
			lines.append(
				f"- Fit indices availability: `{n_fit_available}` available / `{n_fit_unavailable}` unavailable"
			)
		fit_failure_reason = result.optimization_info.get("fit_failure_reason")
		if fit_failure_reason is not None:
			lines.append(f"- Fit issue: `{fit_failure_reason}`")
	if result.measurement_design is not None:
		lines.append(
			"- Measurement design: "
			f"`{len(result.measurement_design.observed_variables)} observed / "
			f"{len(result.measurement_design.latent_variables)} latent`"
		)
	if result.structural_design is not None:
		lines.append(
			"- Structural design: "
			f"`{len(result.structural_design.path_table)} paths / "
			f"{len(result.structural_design.endogenous_latent_variables)} latent endogenous / "
			f"{len(result.structural_design.disturbance_parameters)} disturbances`"
		)

	if result.optimization_info:
		lines.extend(["", "## Optimization"])
		for key, value in result.optimization_info.items():
			lines.append(f"- {key}: `{format_result_value(value)}`")

	if result.fit_indices:
		lines.extend(["", "## Fit indices"])
		for key, value in result.fit_indices.items():
			lines.append(f"- {key}: `{format_result_value(value)}`")
	if result.warnings:
		lines.extend(["", "## Warnings"])
		for warning in result.warnings:
			lines.append(f"- {warning}")
	return "\n".join(lines)


__all__ = ["to_markdown"]
