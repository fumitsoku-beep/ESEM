from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass(frozen=True)
class EFAConfig:
    items: tuple[str, ...]
    n_factors: int
    extraction: str = "paf"
    rotation: str = "varimax"
    max_iter: int = 200
    tol: float = 1e-6
    min_uniqueness: float = 0.005


@dataclass
class EFAResult:
    loadings: pd.DataFrame
    communalities: pd.Series
    uniquenesses: pd.Series
    explained_variance: pd.Series
    correlation_matrix: pd.DataFrame
    extraction: str
    rotation: str
    n_iter: int
    converged: bool


ExtractionOutput = tuple[NDArray[np.float64], NDArray[np.float64], int, bool]
ExtractionMethod = Callable[[NDArray[np.float64], EFAConfig], ExtractionOutput]
RotationMethod = Callable[[NDArray[np.float64], EFAConfig], NDArray[np.float64]]

_EXTRACTION_REGISTRY: dict[str, ExtractionMethod] = {}
_ROTATION_REGISTRY: dict[str, RotationMethod] = {}
_DEFAULTS_LOADED = False


def fit_efa(data: pd.DataFrame, config: EFAConfig) -> EFAResult:
    """Fit exploratory factor analysis to selected item columns.

    Parameters
    ----------
    data:
        Wide-format data frame containing observed item columns.
    config:
        EFA configuration (items, number of factors, extraction, rotation).
    """
    _validate_inputs(data, config)
    extraction = config.extraction.lower()
    rotation = config.rotation.lower()
    extraction_method = get_extraction_method(extraction)
    rotation_method = get_rotation_method(rotation)

    # EFA is run on the item correlation matrix (standardized scale).
    corr = data.loc[:, list(config.items)].corr().to_numpy(dtype=float)
    corr = _stabilize_correlation(corr)

    unrotated_loadings, communalities, n_iter, converged = extraction_method(corr, config)

    # Rotation is applied after extraction to improve interpretability.
    rotated_loadings = rotation_method(unrotated_loadings, config)
    communalities = np.sum(rotated_loadings * rotated_loadings, axis=1)
    uniquenesses = np.clip(1.0 - communalities, config.min_uniqueness, 1.0)
    explained = np.sum(rotated_loadings * rotated_loadings, axis=0)

    factor_names = [f"F{i + 1}" for i in range(config.n_factors)]
    item_names = list(config.items)
    return EFAResult(
        loadings=pd.DataFrame(rotated_loadings, index=item_names, columns=factor_names),
        communalities=pd.Series(communalities, index=item_names, name="communality"),
        uniquenesses=pd.Series(uniquenesses, index=item_names, name="uniqueness"),
        explained_variance=pd.Series(explained, index=factor_names, name="explained_variance"),
        correlation_matrix=pd.DataFrame(corr, index=item_names, columns=item_names),
        extraction=extraction,
        rotation=rotation,
        n_iter=n_iter,
        converged=converged,
    )


def _validate_inputs(data: pd.DataFrame, config: EFAConfig) -> None:
    _ensure_default_methods()
    if not isinstance(data, pd.DataFrame):
        raise TypeError("`data` must be a pandas.DataFrame.")
    if not config.items:
        raise ValueError("`items` cannot be empty.")
    if config.n_factors <= 0:
        raise ValueError("`n_factors` must be > 0.")
    if config.n_factors >= len(config.items):
        raise ValueError("`n_factors` must be smaller than number of items.")
    if config.max_iter <= 0:
        raise ValueError("`max_iter` must be > 0.")
    if config.tol <= 0:
        raise ValueError("`tol` must be > 0.")
    if not (0.0 < config.min_uniqueness < 1.0):
        raise ValueError("`min_uniqueness` must be between 0 and 1.")

    missing = [column for column in config.items if column not in data.columns]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Missing item columns: {joined}.")

    extraction = config.extraction.lower()
    if extraction not in _EXTRACTION_REGISTRY:
        choices = ", ".join(list_extraction_methods())
        raise ValueError(f"Unsupported extraction method `{extraction}`. Available: {choices}.")

    rotation = config.rotation.lower()
    if rotation not in _ROTATION_REGISTRY:
        choices = ", ".join(list_rotation_methods())
        raise ValueError(f"Unsupported rotation method `{rotation}`. Available: {choices}.")


def register_extraction_method(
    name: str,
    method: ExtractionMethod,
    *,
    overwrite: bool = False,
) -> None:
    """Register a custom extraction method."""
    key = _normalize_method_name(name, "extraction")
    if not overwrite and key in _EXTRACTION_REGISTRY:
        raise ValueError(f"Extraction method `{key}` is already registered.")
    _EXTRACTION_REGISTRY[key] = method


def register_rotation_method(
    name: str,
    method: RotationMethod,
    *,
    overwrite: bool = False,
) -> None:
    """Register a custom rotation method."""
    key = _normalize_method_name(name, "rotation")
    if not overwrite and key in _ROTATION_REGISTRY:
        raise ValueError(f"Rotation method `{key}` is already registered.")
    _ROTATION_REGISTRY[key] = method


def list_extraction_methods() -> tuple[str, ...]:
    """Return registered extraction method names."""
    _ensure_default_methods()
    return tuple(sorted(_EXTRACTION_REGISTRY))


def list_rotation_methods() -> tuple[str, ...]:
    """Return registered rotation method names."""
    _ensure_default_methods()
    return tuple(sorted(_ROTATION_REGISTRY))


def get_extraction_method(name: str) -> ExtractionMethod:
    """Resolve extraction method by name."""
    _ensure_default_methods()
    key = _normalize_method_name(name, "extraction")
    try:
        return _EXTRACTION_REGISTRY[key]
    except KeyError as exc:
        choices = ", ".join(list_extraction_methods())
        raise ValueError(f"Unknown extraction method `{key}`. Available: {choices}.") from exc


def get_rotation_method(name: str) -> RotationMethod:
    """Resolve rotation method by name."""
    _ensure_default_methods()
    key = _normalize_method_name(name, "rotation")
    try:
        return _ROTATION_REGISTRY[key]
    except KeyError as exc:
        choices = ", ".join(list_rotation_methods())
        raise ValueError(f"Unknown rotation method `{key}`. Available: {choices}.") from exc


def _ensure_default_methods() -> None:
    global _DEFAULTS_LOADED
    if _DEFAULTS_LOADED:
        return

    _EXTRACTION_REGISTRY.setdefault("paf", _extract_paf_method)
    _EXTRACTION_REGISTRY.setdefault("pca", _extract_pca_method)
    _ROTATION_REGISTRY.setdefault("none", _rotate_none)
    _ROTATION_REGISTRY.setdefault("varimax", _rotate_varimax)
    _DEFAULTS_LOADED = True


def _normalize_method_name(name: str, kind: str) -> str:
    if not isinstance(name, str):
        raise TypeError(f"{kind} method name must be a string.")
    normalized = name.strip().lower()
    if not normalized:
        raise ValueError(f"{kind} method name cannot be empty.")
    return normalized


def _extract_pca_method(corr: NDArray[np.float64], config: EFAConfig) -> ExtractionOutput:
    return _extract_pca(corr=corr, n_factors=config.n_factors)


def _extract_paf_method(corr: NDArray[np.float64], config: EFAConfig) -> ExtractionOutput:
    return _extract_paf(
        corr=corr,
        n_factors=config.n_factors,
        max_iter=config.max_iter,
        tol=config.tol,
        min_uniqueness=config.min_uniqueness,
    )


def _rotate_none(loadings: NDArray[np.float64], _: EFAConfig) -> NDArray[np.float64]:
    return loadings


def _rotate_varimax(loadings: NDArray[np.float64], _: EFAConfig) -> NDArray[np.float64]:
    return _varimax(loadings)


def _extract_pca(corr: NDArray[np.float64], n_factors: int) -> tuple[NDArray[np.float64], NDArray[np.float64], int, bool]:
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


def _squared_multiple_correlations(corr: NDArray[np.float64]) -> NDArray[np.float64]:
    inv_corr = np.linalg.pinv(corr)
    diag = np.clip(np.diag(inv_corr), 1e-12, None)
    smc = 1.0 - (1.0 / diag)
    return np.clip(smc, 0.0, 1.0)


def _varimax(
    loadings: NDArray[np.float64],
    gamma: float = 1.0,
    q: int = 50,
    tol: float = 1e-6,
) -> NDArray[np.float64]:
    p, k = loadings.shape
    rotation = np.eye(k)
    previous = 0.0
    rotated = loadings.copy()

    for _ in range(q):
        rotated = loadings @ rotation
        gram = rotated.T @ rotated
        diagonal = np.diag(np.diag(gram))
        # Classical orthogonal varimax criterion update step.
        transformed = loadings.T @ (rotated**3 - (gamma / p) * rotated @ diagonal)
        u, s, vh = np.linalg.svd(transformed)
        rotation = u @ vh
        current = np.sum(s)
        if previous > 0 and current - previous < tol:
            break
        previous = current

    return loadings @ rotation


def _stabilize_correlation(corr: NDArray[np.float64]) -> NDArray[np.float64]:
    # Ensure symmetry and positive numeric stability for eigendecomposition.
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr, 1.0)
    return corr
