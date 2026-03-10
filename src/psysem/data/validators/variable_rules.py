from __future__ import annotations

from typing import Mapping

from ..constants import SUPPORTED_VARIABLE_TYPES


def validate_variable_types(variable_types: Mapping[str, str], errors: list[str]) -> None:
    for name, var_type in variable_types.items():
        if var_type not in SUPPORTED_VARIABLE_TYPES:
            supported = ", ".join(sorted(SUPPORTED_VARIABLE_TYPES))
            errors.append(
                f"Unsupported variable type `{var_type}` for `{name}`. Supported: {supported}."
            )
