from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class NetworkConfig:
    items: tuple[str, ...]
    estimator: str = "ggm"
    missing_strategy: str = "pairwise"
    correlation_method: str = "pearson"
    variable_types: dict[str, str] | None = None
    ridge: float = 0.0
    min_abs_edge: float = 0.0


@dataclass
class NetworkResult:
    item_names: tuple[str, ...]
    estimator: str
    inversion_method: str
    correlation_method: str
    missing_strategy: str
    resolved_variable_types: dict[str, str]
    association_matrix: pd.DataFrame
    precision_matrix: pd.DataFrame
    partial_correlation_matrix: pd.DataFrame
    adjacency_matrix: pd.DataFrame
    edge_table: pd.DataFrame
    node_table: pd.DataFrame
    pairwise_n: pd.DataFrame | None = None
    n_complete_rows: int | None = None
    dropped_rows: int = 0
    stabilization_applied: bool = False
    warnings: tuple[str, ...] = field(default_factory=tuple)
