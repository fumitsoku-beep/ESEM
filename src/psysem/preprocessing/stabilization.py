from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def stabilize_association_matrix(
    matrix: NDArray[np.float64],
    *,
    min_eigenvalue: float = 1e-8,
) -> tuple[NDArray[np.float64], bool]:
    if min_eigenvalue <= 0:
        raise ValueError("`min_eigenvalue` must be > 0.")

    stabilized = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    stabilized = (stabilized + stabilized.T) / 2.0
    np.fill_diagonal(stabilized, 1.0)

    eigenvalues, eigenvectors = np.linalg.eigh(stabilized)
    minimum = float(np.min(eigenvalues))
    if minimum >= min_eigenvalue:
        return stabilized, False

    clipped = np.clip(eigenvalues, min_eigenvalue, None)
    stabilized = eigenvectors @ np.diag(clipped) @ eigenvectors.T
    scales = np.sqrt(np.clip(np.diag(stabilized), 1e-12, None))
    stabilized = stabilized / np.outer(scales, scales)
    stabilized = (stabilized + stabilized.T) / 2.0
    np.fill_diagonal(stabilized, 1.0)
    return stabilized, True
