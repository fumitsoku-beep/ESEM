from __future__ import annotations

import numpy as np
import pandas as pd

from ..model import ModelSpec
from .contracts import MeasurementDesign


def build_measurement_design(model_spec: ModelSpec) -> MeasurementDesign:
    """Build Lambda/Theta placeholders from measurement relations."""
    measurement_relations = tuple(rel for rel in model_spec.relations if rel.operator == "=~")
    if not measurement_relations:
        raise ValueError("No measurement relations (`=~`) found in model.")

    latent_order: list[str] = []
    latent_seen: set[str] = set()
    observed_order: list[str] = []
    observed_seen: set[str] = set()
    indicator_counts: dict[str, int] = {}
    has_fixed_marker: dict[str, bool] = {}
    free_loadings: list[tuple[str, str]] = []
    fixed_loadings: list[tuple[str, str, float]] = []

    for relation in measurement_relations:
        latent = relation.lhs
        if latent not in latent_seen:
            latent_seen.add(latent)
            latent_order.append(latent)
        indicator_counts.setdefault(latent, 0)
        has_fixed_marker.setdefault(latent, False)

        for term in relation.terms:
            observed = term.variable
            if observed not in observed_seen:
                observed_seen.add(observed)
                observed_order.append(observed)

            indicator_counts[latent] += 1
            if term.coefficient is None:
                free_loadings.append((observed, latent))
            else:
                fixed_value = float(term.coefficient)
                fixed_loadings.append((observed, latent, fixed_value))
                has_fixed_marker[latent] = True

    for latent, count in indicator_counts.items():
        if count < 2:
            raise ValueError(f"Latent `{latent}` has fewer than 2 indicators.")

    lambda_matrix = pd.DataFrame(
        0.0,
        index=observed_order,
        columns=latent_order,
    )
    for observed, latent in free_loadings:
        lambda_matrix.loc[observed, latent] = np.nan
    for observed, latent, value in fixed_loadings:
        lambda_matrix.loc[observed, latent] = value

    theta_matrix = pd.DataFrame(
        0.0,
        index=observed_order,
        columns=observed_order,
    )
    for observed in observed_order:
        theta_matrix.loc[observed, observed] = np.nan

    warnings: list[str] = []
    for latent in latent_order:
        if not has_fixed_marker.get(latent, False):
            warnings.append(
                f"Latent `{latent}` has no fixed loading marker; scale may be under-identified."
            )

    return MeasurementDesign(
        observed_variables=tuple(observed_order),
        latent_variables=tuple(latent_order),
        lambda_matrix=lambda_matrix,
        theta_matrix=theta_matrix,
        free_loadings=tuple(free_loadings),
        fixed_loadings=tuple(fixed_loadings),
        warnings=tuple(warnings),
    )
