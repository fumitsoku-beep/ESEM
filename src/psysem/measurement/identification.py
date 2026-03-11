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
    return tuple(dict.fromkeys(warnings))
