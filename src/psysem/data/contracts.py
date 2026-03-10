from __future__ import annotations

from dataclasses import dataclass, field


class SpecValidationError(ValueError):
    """Raised when an ESEM spec payload fails validation."""


@dataclass(frozen=True)
class RotationSpec:
    method: str = "geomin"
    oblique: bool = True


@dataclass(frozen=True)
class ESEMBlockSpec:
    name: str
    items: tuple[str, ...]
    n_factors: int
    rotation: RotationSpec | None = None


@dataclass(frozen=True)
class ESEMSpec:
    blocks: tuple[ESEMBlockSpec, ...]
    estimator: str
    variable_types: dict[str, str]
    rotation: RotationSpec | None = None
    structural: tuple[str, ...] = field(default_factory=tuple)
    group: str | None = None
    weight: str | None = None
    cluster: str | None = None
    id: str | None = None
    allow_item_overlap: bool = False
