from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .measurement import MeasurementDesign
    from .model import ModelSpec
    from .structural import StructuralDesign


@dataclass
class SEMResult:
    converged: bool
    n_obs: int
    parameters: dict[str, float] = field(default_factory=dict)
    fit_indices: dict[str, float] = field(default_factory=dict)
    parameter_table: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    optimization_info: dict[str, Any] = field(default_factory=dict)
    estimator: str | None = None
    model_spec: ModelSpec | None = None
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
                f"{len(self.structural_design.endogenous_latent_variables)} latent endogenous"
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


def format_result_value(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.4f}"
    if isinstance(value, (int, bool)):
        return str(value)
    return str(value)
