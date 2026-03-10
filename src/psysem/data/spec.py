"""Backward-compatible facade for spec-related APIs.

The implementation now lives in:
- contracts.py
- parser.py
- validator.py
"""

from .contracts import ESEMBlockSpec, ESEMSpec, RotationSpec, SpecValidationError
from .parser import esem_spec_from_dict
from .validator import validate_esem_spec

__all__ = [
    "ESEMBlockSpec",
    "ESEMSpec",
    "RotationSpec",
    "SpecValidationError",
    "esem_spec_from_dict",
    "validate_esem_spec",
]
