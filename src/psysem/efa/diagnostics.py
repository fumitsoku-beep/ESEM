from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import chi2

from .contracts import EFADiagnosticsConfig, EFADiagnosticsResult


def run_efa_diagnostics(data: pd.DataFrame, config: EFADiagnosticsConfig) -> EFADiagnosticsResult:
    """Run KMO/Bartlett diagnostics before EFA fitting."""
    corr, items, n_obs, warnings = build_efa_correlation_matrix(
        data=data,
        items=config.items,
        dropna=config.dropna,
    )
    n_items = len(items)
    sample_ratio = float(n_obs / max(n_items, 1))
    warning_list = list(warnings)
    if sample_ratio < config.min_sample_ratio:
        warning_list.append(
            f"Low sample ratio detected ({sample_ratio:.2f} < {config.min_sample_ratio:.2f})."
        )

    kmo_total, kmo_per_item = _compute_kmo(corr)
    kmo_label = _label_kmo(kmo_total)
    bartlett_chi2, bartlett_df, bartlett_p, bartlett_warnings = _bartlett_sphericity_test(
        corr, n_obs
    )
    warning_list.extend(bartlett_warnings)

    return EFADiagnosticsResult(
        items=items,
        n_obs=n_obs,
        n_items=n_items,
        sample_ratio=sample_ratio,
        kmo_total=kmo_total,
        kmo_per_item=pd.Series(kmo_per_item, index=list(items), name="kmo_msa"),
        kmo_label=kmo_label,
        bartlett_chi2=bartlett_chi2,
        bartlett_df=bartlett_df,
        bartlett_p=bartlett_p,
        warnings=tuple(warning_list),
        correlation_matrix=pd.DataFrame(corr, index=list(items), columns=list(items)),
    )


def build_efa_correlation_matrix(
    data: pd.DataFrame,
    items: Iterable[str],
    *,
    dropna: bool = True,
) -> tuple[np.ndarray, tuple[str, ...], int, tuple[str, ...]]:
    """Validate inputs and construct a stabilized item correlation matrix."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("`data` must be a pandas.DataFrame.")

    item_names = _normalize_items(items)
    missing = [name for name in item_names if name not in data.columns]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Missing item columns: {joined}.")

    selected = data.loc[:, list(item_names)].apply(pd.to_numeric, errors="coerce")
    warnings: list[str] = []
    if dropna:
        selected = selected.dropna(axis=0, how="any")
    elif selected.isna().any().any():
        raise ValueError("Missing values detected in selected items. Set `dropna=True` to continue.")

    n_obs = int(selected.shape[0])
    if n_obs < 3:
        raise ValueError("At least 3 complete observations are required for EFA diagnostics.")

    std = selected.std(axis=0, ddof=0)
    constant_items = [name for name in item_names if float(std[name]) == 0.0]
    if constant_items:
        joined = ", ".join(constant_items)
        warnings.append(f"Constant item(s) detected: {joined}.")

    corr_df = selected.corr()
    corr = _stabilize_correlation(corr_df.to_numpy(dtype=float))
    return corr, item_names, n_obs, tuple(warnings)


def _normalize_items(items: Iterable[str]) -> tuple[str, ...]:
    item_names = tuple(items)
    if not item_names:
        raise ValueError("`items` cannot be empty.")
    if any(not isinstance(name, str) or not name.strip() for name in item_names):
        raise ValueError("`items` must contain non-empty strings.")
    if len(set(item_names)) != len(item_names):
        raise ValueError("`items` contains duplicated names.")
    return item_names


def _stabilize_correlation(corr: np.ndarray) -> np.ndarray:
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr, 1.0)
    return corr


def _compute_kmo(corr: np.ndarray) -> tuple[float, np.ndarray]:
    partial_corr = _partial_correlation(corr)
    corr_sq = corr * corr
    partial_sq = partial_corr * partial_corr
    np.fill_diagonal(corr_sq, 0.0)
    np.fill_diagonal(partial_sq, 0.0)

    corr_sum = float(np.sum(corr_sq))
    partial_sum = float(np.sum(partial_sq))
    denom = corr_sum + partial_sum
    if denom <= 0:
        raise ValueError("KMO cannot be computed: correlation matrix has no off-diagonal information.")

    kmo_total = corr_sum / denom
    item_corr = np.sum(corr_sq, axis=0)
    item_partial = np.sum(partial_sq, axis=0)
    item_denom = item_corr + item_partial
    kmo_per_item = np.divide(
        item_corr,
        item_denom,
        out=np.zeros_like(item_corr),
        where=item_denom > 0,
    )
    return float(kmo_total), kmo_per_item


def _partial_correlation(corr: np.ndarray) -> np.ndarray:
    inv_corr = np.linalg.pinv(corr)
    diag = np.clip(np.diag(inv_corr), 1e-12, None)
    scale = np.sqrt(np.outer(diag, diag))
    partial = -inv_corr / scale
    np.fill_diagonal(partial, 0.0)
    return np.clip(partial, -1.0, 1.0)


def _label_kmo(kmo_total: float) -> str:
    if kmo_total >= 0.90:
        return "marvelous"
    if kmo_total >= 0.80:
        return "meritorious"
    if kmo_total >= 0.70:
        return "middling"
    if kmo_total >= 0.60:
        return "mediocre"
    if kmo_total >= 0.50:
        return "miserable"
    return "unacceptable"


def _bartlett_sphericity_test(
    corr: np.ndarray,
    n_obs: int,
) -> tuple[float, int, float, tuple[str, ...]]:
    warnings: list[str] = []
    p = corr.shape[0]
    sign, log_det = np.linalg.slogdet(corr)

    if sign <= 0:
        jitter = 1e-6
        corrected = corr * (1.0 - jitter) + np.eye(p) * jitter
        sign, log_det = np.linalg.slogdet(corrected)
        warnings.append("Correlation matrix was adjusted for Bartlett test due to non-positive determinant.")

    if sign <= 0:
        raise ValueError("Bartlett test failed: determinant of correlation matrix is non-positive.")

    df = int(p * (p - 1) / 2)
    scale = n_obs - 1 - (2 * p + 5) / 6
    chi_square = -scale * log_det
    if chi_square < 0:
        chi_square = 0.0
        warnings.append("Bartlett chi-square was clipped at 0 due to numerical instability.")

    p_value = float(chi2.sf(chi_square, df))
    return float(chi_square), df, p_value, tuple(warnings)
