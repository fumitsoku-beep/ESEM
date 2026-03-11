from __future__ import annotations

from .contracts import MeasurementDesign


def check_measurement_identification(design: MeasurementDesign) -> tuple[str, ...]:
    """Run lightweight identification checks for measurement design."""
    warnings = list(design.warnings)
    n_obs = len(design.observed_variables)
    n_latent = len(design.latent_variables)
    if n_latent == 0:
        warnings.append("No latent variables found in measurement design.")
    if n_obs <= n_latent:
        warnings.append(
            f"Observed variables ({n_obs}) are not greater than latent variables ({n_latent})."
        )
    free_count = sum(1 for item in design.loading_parameters if item.is_free)
    if free_count == 0:
        warnings.append("Measurement design has no free loadings.")
    fixed_by_observed: dict[str, int] = {}
    for item in design.loading_parameters:
        if item.is_free:
            continue
        fixed_by_observed[item.observed] = fixed_by_observed.get(item.observed, 0) + 1
    for observed, count in fixed_by_observed.items():
        if count > 1:
            warnings.append(
                f"Observed `{observed}` has {count} fixed loadings across latents."
            )
    return tuple(dict.fromkeys(warnings))
