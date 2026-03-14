from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from ..preprocessing import (
    AssociationMatrixConfig,
    build_association_matrix,
)

if TYPE_CHECKING:
    from .fit import EFAConfig


@dataclass(frozen=True)
class _EFAInputMatrix:
    """Compatibility wrapper over the shared preprocessing result."""

    corr: NDArray[np.float64]
    item_names: tuple[str, ...]
    warnings: tuple[str, ...] = ()


def normalize_missing_strategy(strategy: str) -> str:
    """Normalize missing-data strategy names for EFA input preparation."""

    return strategy.strip().lower()


def normalize_correlation_method(method: str) -> str:
    """Normalize correlation-method names for EFA input preparation."""

    return method.strip().lower()


def build_efa_input_matrix(data: pd.DataFrame, config: EFAConfig) -> _EFAInputMatrix:
    """Build the EFA input matrix via the shared preprocessing layer."""

    prepared = build_association_matrix(
        data,
        AssociationMatrixConfig(
            items=tuple(config.items),
            missing_strategy=normalize_missing_strategy(config.missing_strategy),
            correlation_method=normalize_correlation_method(config.correlation_method),
            variable_types=config.variable_types,
            stabilize=True,
            min_eigenvalue=1e-8,
            include_pairwise_counts=False,
        ),
    )
    return _EFAInputMatrix(
        corr=prepared.matrix.to_numpy(dtype=float),
        item_names=prepared.item_names,
        warnings=prepared.warnings,
    )
