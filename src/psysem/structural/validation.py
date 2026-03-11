from __future__ import annotations

from .contracts import StructuralDesign


def check_structural_validity(design: StructuralDesign) -> tuple[str, ...]:
    """Run lightweight validity checks for structural design."""
    warnings = list(design.warnings)
    if not design.path_table:
        warnings.append("No structural paths found.")
    if design.beta_matrix.shape[0] != design.beta_matrix.shape[1]:
        warnings.append("Beta matrix is not square.")
    if design.gamma_matrix.shape[0] != len(design.endogenous_latent_variables):
        warnings.append("Gamma matrix row count does not match endogenous latent variables.")
    return tuple(dict.fromkeys(warnings))
