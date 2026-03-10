from __future__ import annotations

import pandas as pd

from .constants import SUPPORTED_ESTIMATORS
from .contracts import ESEMSpec, SpecValidationError
from .structural import observed_structural_variables
from .validators import (
    validate_block_rules,
    validate_data_reference_columns,
    validate_ordinal_columns,
    validate_variable_types,
)


def validate_esem_spec(spec: ESEMSpec, data: pd.DataFrame | None = None) -> None:
    """Validate ESEM spec shape and optional data compatibility."""
    errors: list[str] = []

    if not spec.blocks:
        errors.append("At least one block is required.")

    if spec.estimator not in SUPPORTED_ESTIMATORS:
        supported = ", ".join(sorted(SUPPORTED_ESTIMATORS))
        errors.append(f"Unsupported estimator `{spec.estimator}`. Supported: {supported}.")

    item_set = validate_block_rules(spec, errors)
    validate_variable_types(spec.variable_types, errors)

    for item in item_set:
        if item not in spec.variable_types:
            errors.append(f"Missing variable type for item `{item}`.")

    structural_vars = observed_structural_variables(spec.structural)
    for var_name in structural_vars:
        if var_name not in spec.variable_types:
            errors.append(
                f"Missing variable type for observed structural variable `{var_name}`."
            )

    if data is not None:
        validate_data_reference_columns(spec, data, item_set, structural_vars, errors)
        validate_ordinal_columns(spec, data, errors)

    if errors:
        joined = "\n".join(f"- {message}" for message in errors)
        raise SpecValidationError(f"ESEM spec validation failed:\n{joined}")
