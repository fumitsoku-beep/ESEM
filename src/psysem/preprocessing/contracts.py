from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class AssociationMatrixConfig:
    items: tuple[str, ...]
    missing_strategy: str = "pairwise"
    correlation_method: str = "pearson"
    variable_types: dict[str, str] | None = None
    stabilize: bool = True
    min_eigenvalue: float = 1e-8
    include_pairwise_counts: bool = True


@dataclass(frozen=True)
class AssociationMatrixResult:
    matrix: pd.DataFrame
    item_names: tuple[str, ...]
    correlation_method: str
    missing_strategy: str
    resolved_variable_types: dict[str, str]
    pairwise_n: pd.DataFrame | None = None
    n_complete_rows: int | None = None
    dropped_rows: int = 0
    stabilization_applied: bool = False
    warnings: tuple[str, ...] = field(default_factory=tuple)
