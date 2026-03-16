from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
	from .measurement import MeasurementDesign
	from .model import ModelSpec
	from .parameter_index import ParameterIndexMap
	from .structural import StructuralDesign


@dataclass
class SEMResult:
	converged: bool
	n_obs: int
	parameters: dict[str, float] = field(default_factory=dict)
	parameter_inference: tuple[dict[str, Any], ...] = field(default_factory=tuple)
	fit_indices: dict[str, float] = field(default_factory=dict)
	parameter_table: tuple[dict[str, Any], ...] = field(default_factory=tuple)
	warnings: tuple[str, ...] = field(default_factory=tuple)
	optimization_info: dict[str, Any] = field(default_factory=dict)
	estimator: str | None = None
	model_spec: ModelSpec | None = None
	parameter_index_map: ParameterIndexMap | None = None
	measurement_design: MeasurementDesign | None = None
	structural_design: StructuralDesign | None = None

	def summary(self) -> str:
		lines = [
			"SEM Fit Summary",
			f"Converged: {self.converged}",
			f"N: {self.n_obs}",
		]
		if self.estimator:
			lines.append(f"Estimator: {self.estimator}")
		if self.model_spec is not None:
			lines.append(f"Model source: {self.model_spec.source}")
			lines.append(f"Relations: {len(self.model_spec.relations)}")
		if self.parameter_index_map is not None:
			lines.append(f"Free parameters: {self.parameter_index_map.n_free}")
		if self.parameter_inference:
			n_with_se = _count_inference_rows_with_se(self.parameter_inference)
			n_without_se = len(self.parameter_inference) - n_with_se
			lines.append(
				f"Inference rows: {len(self.parameter_inference)} ({n_with_se} with SE, {n_without_se} without SE)"
			)
			inference_status = self.optimization_info.get("inference_status")
			if inference_status is not None:
				lines.append(f"Inference status: {inference_status}")
			inference_failure_reason = self.optimization_info.get("inference_failure_reason")
			if inference_failure_reason is not None:
				lines.append(f"Inference issue: {inference_failure_reason}")
		fit_status = self.optimization_info.get("fit_status")
		if fit_status is not None:
			lines.append(f"Fit status: {fit_status}")
			n_fit_available = self.optimization_info.get("n_fit_indices_available")
			n_fit_unavailable = self.optimization_info.get("n_fit_indices_unavailable")
			if n_fit_available is not None and n_fit_unavailable is not None:
				lines.append(
					f"Fit indices availability: {n_fit_available} available / {n_fit_unavailable} unavailable"
				)
			fit_failure_reason = self.optimization_info.get("fit_failure_reason")
			if fit_failure_reason is not None:
				lines.append(f"Fit issue: {fit_failure_reason}")
		if self.measurement_design is not None:
			lines.append(
				"Measurement design: "
				f"{len(self.measurement_design.observed_variables)} observed / "
				f"{len(self.measurement_design.latent_variables)} latent"
			)
		if self.structural_design is not None:
			lines.append(
				"Structural design: "
				f"{len(self.structural_design.path_table)} paths / "
				f"{len(self.structural_design.endogenous_latent_variables)} latent endogenous / "
				f"{len(self.structural_design.disturbance_parameters)} disturbances"
			)

		if self.optimization_info:
			lines.append("Optimization:")
			for key, value in self.optimization_info.items():
				lines.append(f"  {key}: {format_result_value(value)}")

		if self.fit_indices:
			lines.append("Fit indices:")
			for key, value in self.fit_indices.items():
				lines.append(f"  {key}: {format_result_value(value)}")
		if self.warnings:
			lines.append("Warnings:")
			for warning in self.warnings:
				lines.append(f"  - {warning}")
		return "\n".join(lines)


def _count_inference_rows_with_se(parameter_inference: tuple[dict[str, Any], ...]) -> int:
	return sum(1 for row in parameter_inference if isinstance(row.get("standard_error"), float))


def format_result_value(value: Any) -> str:
	if isinstance(value, float):
		if math.isnan(value):
			return "nan"
		return f"{value:.4f}"
	if isinstance(value, (int, bool)):
		return str(value)
	return str(value)


__all__ = [
	"SEMResult",
	"_count_inference_rows_with_se",
	"format_result_value",
]
