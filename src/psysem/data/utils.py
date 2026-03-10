from __future__ import annotations

from typing import Any

from .contracts import SpecValidationError


def parse_str_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise SpecValidationError(f"`{field_name}` must be a list.")
    parsed: list[str] = []
    for idx, entry in enumerate(value):
        text = require_str(entry, f"{field_name}[{idx}]")
        parsed.append(text)
    return parsed


def require_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise SpecValidationError(f"`{field_name}` must be a string.")
    stripped = value.strip()
    if not stripped:
        raise SpecValidationError(f"`{field_name}` cannot be empty.")
    return stripped


def optional_str(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return require_str(value, field_name)
