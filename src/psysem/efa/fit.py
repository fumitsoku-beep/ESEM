from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .extraction import _extract_minres_method, _extract_paf_method, _extract_pca_method
from .rotation import (
    _coerce_rotation_target,
    _coerce_rotation_target_weights,
    _rotate_geomin,
    _rotate_none,
    _rotate_oblimin,
    _rotate_promax,
    _rotate_target,
    _rotate_varimax,
)
from .rotation import _stabilize_factor_correlation


@dataclass(frozen=True)
class EFAConfig:
    items: tuple[str, ...]
    n_factors: int
    extraction: str = "paf"
    rotation: str = "varimax"
    rotation_target: pd.DataFrame | NDArray[np.float64] | None = None
    rotation_target_weights: pd.DataFrame | NDArray[np.float64] | None = None
    rotation_restarts: int = 0
    random_state: int | None = None
    max_iter: int = 200
    tol: float = 1e-6
    min_uniqueness: float = 0.005


@dataclass
class EFAResult:
    loadings: pd.DataFrame
    communalities: pd.Series
    uniquenesses: pd.Series
    complexity: pd.Series
    explained_variance: pd.Series
    correlation_matrix: pd.DataFrame
    residual_matrix: pd.DataFrame
    residual_summary: dict[str, float]
    factor_correlation: pd.DataFrame
    extraction: str
    rotation: str
    n_iter: int
    converged: bool
    cross_loaded_items: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class _ExtractionResult:
    """Private normalized extraction result used inside ``fit_efa``.

    Public extraction methods may still return the legacy tuple form for
    backward compatibility. This internal structure exists so future methods can
    provide richer diagnostics without changing the public API.
    """

    loadings: NDArray[np.float64]
    communalities: NDArray[np.float64]
    n_iter: int
    converged: bool
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class _RotationResult:
    """Private normalized rotation result.

    Orthogonal rotations simply return an identity factor-correlation matrix.
    Oblique rotations may provide a non-identity factor-correlation matrix and
    optional rotation diagnostics in the future.
    """

    loadings: NDArray[np.float64]
    factor_correlation: NDArray[np.float64]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class _EFAModelComponents:
    """Private bundle for post-extraction and post-rotation derived quantities."""

    communalities: NDArray[np.float64]
    uniquenesses: NDArray[np.float64]
    explained_variance: NDArray[np.float64]
    complexity: NDArray[np.float64]
    residual_matrix: NDArray[np.float64]
    residual_summary: dict[str, float]


ExtractionOutput = tuple[NDArray[np.float64], NDArray[np.float64], int, bool] | _ExtractionResult
ExtractionMethod = Callable[[NDArray[np.float64], EFAConfig], ExtractionOutput]
RotationOutput = (
    NDArray[np.float64]
    | tuple[NDArray[np.float64], NDArray[np.float64]]
    | tuple[NDArray[np.float64], NDArray[np.float64], tuple[str, ...]]
    | _RotationResult
)
RotationMethod = Callable[[NDArray[np.float64], EFAConfig], RotationOutput]

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

    extraction_result = _normalize_extraction_output(extraction_method(corr, config))

    # Rotation is applied after extraction to improve interpretability.
    rotation_result = _normalize_rotation_output(
        rotation_method(extraction_result.loadings, config),
        n_factors=config.n_factors,
    )
    components = _build_model_components(
        corr=corr,
        loadings=rotation_result.loadings,
        factor_corr=rotation_result.factor_correlation,
        min_uniqueness=config.min_uniqueness,
    )
    cross_loaded = _detect_cross_loaded_items(
        rotation_result.loadings,
        item_names=list(config.items),
        threshold=0.30,
    )
    warnings = _build_interpretation_warnings(
        uniquenesses=components.uniquenesses,
        communalities=components.communalities,
        cross_loaded=cross_loaded,
        residual_summary=components.residual_summary,
        min_uniqueness=config.min_uniqueness,
    )
    warnings = list(extraction_result.warnings) + list(rotation_result.warnings) + warnings

    factor_names = [f"F{i + 1}" for i in range(config.n_factors)]
    item_names = list(config.items)
    return EFAResult(
        loadings=pd.DataFrame(rotation_result.loadings, index=item_names, columns=factor_names),
        communalities=pd.Series(components.communalities, index=item_names, name="communality"),
        uniquenesses=pd.Series(components.uniquenesses, index=item_names, name="uniqueness"),
        complexity=pd.Series(components.complexity, index=item_names, name="complexity"),
        explained_variance=pd.Series(
            components.explained_variance, index=factor_names, name="explained_variance"
        ),
        correlation_matrix=pd.DataFrame(corr, index=item_names, columns=item_names),
        residual_matrix=pd.DataFrame(components.residual_matrix, index=item_names, columns=item_names),
        residual_summary=components.residual_summary,
        factor_correlation=pd.DataFrame(
            rotation_result.factor_correlation,
            index=factor_names,
            columns=factor_names,
        ),
        extraction=extraction,
        rotation=rotation,
        n_iter=extraction_result.n_iter,
        converged=extraction_result.converged,
        cross_loaded_items=tuple(cross_loaded),
        warnings=tuple(dict.fromkeys(warnings)),
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
    if config.rotation_restarts < 0:
        raise ValueError("`rotation_restarts` must be >= 0.")
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
    if rotation == "target":
        target = _coerce_rotation_target(
            config.rotation_target,
            item_names=list(config.items),
            n_factors=config.n_factors,
        )
        _coerce_rotation_target_weights(
            config.rotation_target_weights,
            item_names=list(config.items),
            n_factors=config.n_factors,
            target=target,
        )


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
    _EXTRACTION_REGISTRY.setdefault("minres", _extract_minres_method)
    _EXTRACTION_REGISTRY.setdefault("pca", _extract_pca_method)
    _ROTATION_REGISTRY.setdefault("none", _rotate_none)
    _ROTATION_REGISTRY.setdefault("geomin", _rotate_geomin)
    _ROTATION_REGISTRY.setdefault("oblimin", _rotate_oblimin)
    _ROTATION_REGISTRY.setdefault("promax", _rotate_promax)
    _ROTATION_REGISTRY.setdefault("target", _rotate_target)
    _ROTATION_REGISTRY.setdefault("varimax", _rotate_varimax)
    _DEFAULTS_LOADED = True


def _normalize_method_name(name: str, kind: str) -> str:
    if not isinstance(name, str):
        raise TypeError(f"{kind} method name must be a string.")
    normalized = name.strip().lower()
    if not normalized:
        raise ValueError(f"{kind} method name cannot be empty.")
    return normalized


def _normalize_extraction_output(output: ExtractionOutput) -> _ExtractionResult:
    """Normalize legacy and future extraction outputs into one private contract."""

    if isinstance(output, _ExtractionResult):
        return output

    loadings, communalities, n_iter, converged = output
    return _ExtractionResult(
        loadings=np.asarray(loadings, dtype=float),
        communalities=np.asarray(communalities, dtype=float),
        n_iter=int(n_iter),
        converged=bool(converged),
    )


def _normalize_rotation_output(
    output: RotationOutput,
    *,
    n_factors: int,
) -> _RotationResult:
    """Normalize orthogonal and oblique rotation results into one private contract."""

    if isinstance(output, _RotationResult):
        return _RotationResult(
            loadings=np.asarray(output.loadings, dtype=float),
            factor_correlation=_stabilize_factor_correlation(output.factor_correlation),
            warnings=tuple(output.warnings),
        )

    if isinstance(output, tuple):
        if len(output) == 3:
            loadings, factor_corr, warnings = output
            return _RotationResult(
                loadings=np.asarray(loadings, dtype=float),
                factor_correlation=_stabilize_factor_correlation(factor_corr),
                warnings=tuple(warnings),
            )

        loadings, factor_corr = output
        return _RotationResult(
            loadings=np.asarray(loadings, dtype=float),
            factor_correlation=_stabilize_factor_correlation(factor_corr),
        )

    return _RotationResult(
        loadings=np.asarray(output, dtype=float),
        factor_correlation=np.eye(n_factors, dtype=float),
    )


def _build_model_components(
    *,
    corr: NDArray[np.float64],
    loadings: NDArray[np.float64],
    factor_corr: NDArray[np.float64],
    min_uniqueness: float,
) -> _EFAModelComponents:
    """Compute derived EFA quantities from a normalized extraction+rotation solution."""

    communalities = _compute_communalities(loadings, factor_corr)
    uniquenesses = np.clip(1.0 - communalities, min_uniqueness, 1.0)
    explained_variance = np.sum(loadings * loadings, axis=0)
    complexity = _compute_item_complexity(loadings)
    residual_matrix = _residual_correlation(corr, loadings, uniquenesses, factor_corr)
    residual_summary = _summarize_residuals(residual_matrix)
    return _EFAModelComponents(
        communalities=communalities,
        uniquenesses=uniquenesses,
        explained_variance=explained_variance,
        complexity=complexity,
        residual_matrix=residual_matrix,
        residual_summary=residual_summary,
    )


def _compute_communalities(
    loadings: NDArray[np.float64],
    factor_corr: NDArray[np.float64],
) -> NDArray[np.float64]:
    reproduced_common = loadings @ factor_corr @ loadings.T
    communalities = np.diag(reproduced_common)
    return np.clip(communalities, 0.0, 1.0)


def _stabilize_correlation(corr: NDArray[np.float64]) -> NDArray[np.float64]:
    # Ensure symmetry and positive numeric stability for eigendecomposition.
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr, 1.0)
    return corr


def _compute_item_complexity(loadings: NDArray[np.float64]) -> NDArray[np.float64]:
    abs_loadings = np.abs(loadings)
    sum_sq = np.sum(abs_loadings * abs_loadings, axis=1)
    sum_fourth = np.sum(abs_loadings**4, axis=1)
    return np.divide(
        sum_sq * sum_sq,
        sum_fourth,
        out=np.ones_like(sum_sq),
        where=sum_fourth > 1e-12,
    )


def _residual_correlation(
    corr: NDArray[np.float64],
    loadings: NDArray[np.float64],
    uniquenesses: NDArray[np.float64],
    factor_corr: NDArray[np.float64],
) -> NDArray[np.float64]:
    reproduced = loadings @ factor_corr @ loadings.T + np.diag(uniquenesses)
    reproduced = (reproduced + reproduced.T) / 2.0
    np.fill_diagonal(reproduced, 1.0)
    residual = corr - reproduced
    residual = (residual + residual.T) / 2.0
    np.fill_diagonal(residual, 0.0)
    return residual


def _summarize_residuals(residual: NDArray[np.float64]) -> dict[str, float]:
    p = residual.shape[0]
    upper = np.triu_indices(p, k=1)
    values = residual[upper]
    abs_values = np.abs(values)
    if values.size == 0:
        return {
            "rmsr": 0.0,
            "mean_abs_residual": 0.0,
            "max_abs_residual": 0.0,
            "n_abs_gt_0_05": 0.0,
            "n_abs_gt_0_10": 0.0,
        }
    return {
        "rmsr": float(np.sqrt(np.mean(values * values))),
        "mean_abs_residual": float(np.mean(abs_values)),
        "max_abs_residual": float(np.max(abs_values)),
        "n_abs_gt_0_05": float(np.sum(abs_values > 0.05)),
        "n_abs_gt_0_10": float(np.sum(abs_values > 0.10)),
    }


def _detect_cross_loaded_items(
    loadings: NDArray[np.float64],
    *,
    item_names: list[str],
    threshold: float,
) -> list[str]:
    abs_loadings = np.abs(loadings)
    mask = np.sum(abs_loadings >= threshold, axis=1) >= 2
    return [name for name, flagged in zip(item_names, mask) if flagged]


def _build_interpretation_warnings(
    *,
    uniquenesses: NDArray[np.float64],
    communalities: NDArray[np.float64],
    cross_loaded: list[str],
    residual_summary: dict[str, float],
    min_uniqueness: float,
) -> list[str]:
    warnings: list[str] = []
    if np.any(uniquenesses <= min_uniqueness + 1e-9):
        warnings.append("Boundary uniqueness detected; review possible Heywood-adjacent solution.")
    if np.any(communalities < 0.20):
        warnings.append("Low communality items detected (h2 < 0.20).")
    if cross_loaded:
        warnings.append(f"Cross-loaded items detected: {', '.join(cross_loaded)}.")
    if residual_summary.get("rmsr", 0.0) > 0.08:
        warnings.append("Residual RMSR is relatively high (> 0.08).")
    return warnings
