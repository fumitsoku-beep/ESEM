from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParameterIndexEntry:
    """One free parameter entry in the global optimization vector."""

    parameter_index: int
    parameter: str
    vector_position: int


@dataclass(frozen=True)
class ParameterIndexMap:
    """Global free-parameter index mapping for SEM estimation."""

    entries: tuple[ParameterIndexEntry, ...]

    @property
    def n_free(self) -> int:
        return len(self.entries)

    def index_to_position(self) -> dict[int, int]:
        return {
            entry.parameter_index: entry.vector_position
            for entry in self.entries
        }

    def index_to_name(self) -> dict[int, str]:
        return {
            entry.parameter_index: entry.parameter
            for entry in self.entries
        }

    def name_to_position(self) -> dict[str, int]:
        return {
            entry.parameter: entry.vector_position
            for entry in self.entries
        }


def build_parameter_index_map(
    parameter_table: tuple[dict[str, Any], ...],
) -> ParameterIndexMap:
    """Build deterministic free-parameter ordering from ``parameter_table``."""
    by_index: dict[int, str] = {}
    for row in parameter_table:
        if not bool(row["is_free"]):
            continue
        parameter_index = row["parameter_index"]
        parameter_name = row["parameter"]
        if not isinstance(parameter_index, int):
            raise ValueError("Free parameter row must include integer `parameter_index`.")
        if not isinstance(parameter_name, str):
            raise ValueError("Free parameter row must include string `parameter` name.")
        existing_name = by_index.get(parameter_index)
        if existing_name is not None and existing_name != parameter_name:
            raise ValueError(
                "Inconsistent parameter name for shared `parameter_index` "
                f"{parameter_index}: `{existing_name}` vs `{parameter_name}`."
            )
        by_index[parameter_index] = parameter_name

    entries = tuple(
        ParameterIndexEntry(
            parameter_index=parameter_index,
            parameter=by_index[parameter_index],
            vector_position=position,
        )
        for position, parameter_index in enumerate(sorted(by_index))
    )
    return ParameterIndexMap(entries=entries)

