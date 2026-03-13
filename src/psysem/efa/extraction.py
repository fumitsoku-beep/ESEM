from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

if TYPE_CHECKING:
    from .fit import EFAConfig


def _extract_pca_method(corr: NDArray[np.float64], config: EFAConfig):
    return _extract_pca(corr=corr, n_factors=config.n_factors)


def _extract_paf_method(corr: NDArray[np.float64], config: EFAConfig):
    return _extract_paf(
        corr=corr,
        n_factors=config.n_factors,
        max_iter=config.max_iter,
        tol=config.tol,
        min_uniqueness=config.min_uniqueness,
    )


def _extract_minres_method(corr: NDArray[np.float64], config: EFAConfig):
    return _extract_minres(
        corr=corr,
        n_factors=config.n_factors,
        max_iter=config.max_iter,
        tol=config.tol,
        min_uniqueness=config.min_uniqueness,
    )


def _extract_pca(
    corr: NDArray[np.float64],
    n_factors: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], int, bool]:
    eigvals, eigvecs = np.linalg.eigh(corr)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    kept_vals = np.clip(eigvals[:n_factors], 0.0, None)
    kept_vecs = eigvecs[:, :n_factors]
    loadings = kept_vecs * np.sqrt(kept_vals)
    communalities = np.sum(loadings * loadings, axis=1)
    return loadings, communalities, 1, True


def _extract_paf(
    corr: NDArray[np.float64],
    n_factors: int,
    max_iter: int,
    tol: float,
    min_uniqueness: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], int, bool]:
    p = corr.shape[0]
    # Initialize communalities with SMC as the standard PAF starting point.
    smc = _squared_multiple_correlations(corr)
    communalities = np.clip(smc, min_uniqueness, 1.0 - min_uniqueness)
    converged = False
    n_iter = 0
    loadings = np.zeros((p, n_factors), dtype=float)

    for n_iter in range(1, max_iter + 1):
        reduced = corr.copy()
        # PAF repeatedly replaces correlation diagonal with current communalities.
        np.fill_diagonal(reduced, communalities)
        eigvals, eigvecs = np.linalg.eigh(reduced)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        kept_vals = np.clip(eigvals[:n_factors], 0.0, None)
        kept_vecs = eigvecs[:, :n_factors]
        loadings = kept_vecs * np.sqrt(kept_vals)
        updated = np.sum(loadings * loadings, axis=1)
        updated = np.clip(updated, min_uniqueness, 1.0 - min_uniqueness)

        delta = np.max(np.abs(updated - communalities))
        communalities = updated
        if delta < tol:
            converged = True
            break

    return loadings, communalities, n_iter, converged


def _extract_minres(
    corr: NDArray[np.float64],
    n_factors: int,
    max_iter: int,
    tol: float,
    min_uniqueness: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], int, bool]:
    """Estimate common-factor loadings via MINRES.

    MINRES minimizes the sum of squared off-diagonal residuals between the
    observed correlation matrix and the reproduced correlation matrix implied by
    a low-rank common-factor solution plus unique variances.

    The optimization variable is the uniqueness vector. For a candidate
    uniqueness vector, communalities are computed as ``1 - uniqueness`` and used
    as the diagonal of the reduced correlation matrix. The top ``n_factors``
    eigencomponents of that reduced matrix define the current loading solution.
    """

    p = corr.shape[0]
    # Start from the classical SMC initialization used by many factor methods.
    smc = _squared_multiple_correlations(corr)
    initial_uniquenesses = np.clip(1.0 - smc, min_uniqueness, 1.0 - min_uniqueness)
    bounds = [(min_uniqueness, 1.0 - min_uniqueness) for _ in range(p)]
    upper = np.triu_indices(p, k=1)

    def objective(uniquenesses: NDArray[np.float64]) -> float:
        loadings, _ = _loadings_from_uniquenesses(
            corr=corr,
            uniquenesses=uniquenesses,
            n_factors=n_factors,
        )
        # MINRES is defined on off-diagonal residuals; the diagonal is handled by
        # the uniqueness terms and is not directly penalized here.
        reproduced_offdiag = loadings @ loadings.T
        residual_offdiag = corr - reproduced_offdiag
        return float(np.sum(residual_offdiag[upper] ** 2))

    optimization = minimize(
        objective,
        x0=initial_uniquenesses,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": max_iter, "ftol": tol},
    )

    optimized_uniquenesses = np.clip(
        np.asarray(optimization.x, dtype=float),
        min_uniqueness,
        1.0 - min_uniqueness,
    )
    loadings, communalities = _loadings_from_uniquenesses(
        corr=corr,
        uniquenesses=optimized_uniquenesses,
        n_factors=n_factors,
    )
    n_iter = int(getattr(optimization, "nit", 0))
    converged = bool(optimization.success)
    return loadings, communalities, n_iter, converged


def _loadings_from_uniquenesses(
    *,
    corr: NDArray[np.float64],
    uniquenesses: NDArray[np.float64],
    n_factors: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build a common-factor loading solution from a uniqueness vector.

    The diagonal of the reduced correlation matrix is replaced by the implied
    communalities. The leading eigencomponents then define the current common
    loading matrix.
    """

    reduced = corr.copy()
    communalities = np.clip(1.0 - uniquenesses, 0.0, 1.0)
    np.fill_diagonal(reduced, communalities)
    eigvals, eigvecs = np.linalg.eigh(reduced)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    kept_vals = np.clip(eigvals[:n_factors], 0.0, None)
    kept_vecs = eigvecs[:, :n_factors]
    loadings = kept_vecs * np.sqrt(kept_vals)
    updated_communalities = np.clip(np.sum(loadings * loadings, axis=1), 0.0, 1.0)
    return loadings, updated_communalities


def _squared_multiple_correlations(corr: NDArray[np.float64]) -> NDArray[np.float64]:
    inv_corr = np.linalg.pinv(corr)
    diag = np.clip(np.diag(inv_corr), 1e-12, None)
    smc = 1.0 - (1.0 / diag)
    return np.clip(smc, 0.0, 1.0)