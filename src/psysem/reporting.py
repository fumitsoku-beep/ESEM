from __future__ import annotations

from .result import SEMResult, format_result_value


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
            f"{len(result.structural_design.endogenous_latent_variables)} latent endogenous`"
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
