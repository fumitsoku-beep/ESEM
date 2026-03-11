from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import EFAInterpretationConfig, EFAInterpretationResult
from .fit import EFAResult


def interpret_efa(
    result: EFAResult,
    config: EFAInterpretationConfig = EFAInterpretationConfig(),
) -> EFAInterpretationResult:
    """Build R-style interpretation tables and warnings from one EFA solution."""
    _validate_interpretation_config(config)

    item_table = _build_item_table(result, config)
    factor_table = _build_factor_table(result, config)
    residual_top_pairs = _build_residual_top_pairs(result, config.residual_top_n)
    warnings = _build_interpretation_warnings(
        result=result,
        item_table=item_table,
        factor_table=factor_table,
        config=config,
    )
    summary = _build_summary(result, item_table, config)
    return EFAInterpretationResult(
        item_table=item_table,
        factor_table=factor_table,
        residual_top_pairs=residual_top_pairs,
        warnings=tuple(warnings),
        summary=summary,
    )


def _build_item_table(result: EFAResult, config: EFAInterpretationConfig) -> pd.DataFrame:
    abs_loadings = result.loadings.abs()
    n_salient = (abs_loadings >= config.salient_loading).sum(axis=1)
    n_cross = (abs_loadings >= config.cross_loading).sum(axis=1)
    item_table = pd.DataFrame(index=result.loadings.index)
    item_table["primary_factor"] = abs_loadings.idxmax(axis=1)
    item_table["primary_loading"] = abs_loadings.max(axis=1)
    item_table["h2"] = result.communalities
    item_table["u2"] = result.uniquenesses
    item_table["com"] = result.complexity
    item_table["n_salient_loadings"] = n_salient.astype(int)
    item_table["n_cross_loadings"] = n_cross.astype(int)
    item_table["is_cross_loaded"] = n_cross >= 2
    item_table["is_low_h2"] = result.communalities < config.min_h2
    return item_table


def _build_factor_table(result: EFAResult, config: EFAInterpretationConfig) -> pd.DataFrame:
    abs_loadings = result.loadings.abs()
    ss_loadings = result.explained_variance.astype(float)
    n_items = max(float(result.loadings.shape[0]), 1.0)
    proportion = ss_loadings / n_items
    factor_table = pd.DataFrame(index=result.loadings.columns)
    factor_table["ss_loadings"] = ss_loadings
    factor_table["proportion_var"] = proportion
    factor_table["cumulative_var"] = proportion.cumsum()
    factor_table["n_salient_items"] = (abs_loadings >= config.salient_loading).sum(axis=0).astype(int)
    factor_table["mean_abs_loading"] = abs_loadings.mean(axis=0)
    return factor_table


def _build_residual_top_pairs(result: EFAResult, top_n: int) -> pd.DataFrame:
    residual = result.residual_matrix.to_numpy(dtype=float)
    p = residual.shape[0]
    upper = np.triu_indices(p, k=1)
    values = residual[upper]
    if values.size == 0:
        return pd.DataFrame(columns=["item_i", "item_j", "residual", "abs_residual"])

    abs_values = np.abs(values)
    order = np.argsort(abs_values)[::-1]
    if top_n > 0:
        order = order[:top_n]

    item_names = list(result.residual_matrix.index)
    rows = []
    for idx in order:
        i = int(upper[0][idx])
        j = int(upper[1][idx])
        rows.append(
            {
                "item_i": item_names[i],
                "item_j": item_names[j],
                "residual": float(values[idx]),
                "abs_residual": float(abs_values[idx]),
            }
        )
    return pd.DataFrame(rows, columns=["item_i", "item_j", "residual", "abs_residual"])


def _build_interpretation_warnings(
    *,
    result: EFAResult,
    item_table: pd.DataFrame,
    factor_table: pd.DataFrame,
    config: EFAInterpretationConfig,
) -> list[str]:
    warnings = list(result.warnings)
    n_low_h2 = int(item_table["is_low_h2"].sum())
    if n_low_h2 > 0:
        warnings.append(f"Low communality items: {n_low_h2} item(s) with h2 < {config.min_h2:.2f}.")

    cross_items = item_table.index[item_table["is_cross_loaded"]].tolist()
    if cross_items:
        warnings.append(f"Cross-loaded items: {', '.join(cross_items)}.")

    low_salient_factors = factor_table.index[
        factor_table["n_salient_items"] < config.min_salient_items_per_factor
    ].tolist()
    if low_salient_factors:
        warnings.append(
            "Factors with too few salient items: "
            + ", ".join(low_salient_factors)
            + f" (< {config.min_salient_items_per_factor})."
        )

    rmsr = float(result.residual_summary.get("rmsr", 0.0))
    if rmsr > config.rmsr_warning:
        warnings.append(f"RMSR is high ({rmsr:.4f} > {config.rmsr_warning:.4f}).")

    max_abs_resid = float(result.residual_summary.get("max_abs_residual", 0.0))
    if max_abs_resid > config.max_abs_residual_warning:
        warnings.append(
            f"Max absolute residual is high ({max_abs_resid:.4f} > "
            f"{config.max_abs_residual_warning:.4f})."
        )
    return list(dict.fromkeys(warnings))


def _build_summary(
    result: EFAResult,
    item_table: pd.DataFrame,
    config: EFAInterpretationConfig,
) -> dict[str, float]:
    return {
        "n_items": float(result.loadings.shape[0]),
        "n_factors": float(result.loadings.shape[1]),
        "n_cross_loaded_items": float(item_table["is_cross_loaded"].sum()),
        "n_low_h2_items": float(item_table["is_low_h2"].sum()),
        "rmsr": float(result.residual_summary.get("rmsr", 0.0)),
        "max_abs_residual": float(result.residual_summary.get("max_abs_residual", 0.0)),
        "min_h2_threshold": float(config.min_h2),
    }


def _validate_interpretation_config(config: EFAInterpretationConfig) -> None:
    if config.salient_loading <= 0:
        raise ValueError("`salient_loading` must be > 0.")
    if config.cross_loading < config.salient_loading:
        raise ValueError("`cross_loading` must be >= `salient_loading`.")
    if not (0.0 <= config.min_h2 <= 1.0):
        raise ValueError("`min_h2` must be between 0 and 1.")
    if config.min_salient_items_per_factor <= 0:
        raise ValueError("`min_salient_items_per_factor` must be > 0.")
    if config.rmsr_warning < 0:
        raise ValueError("`rmsr_warning` must be >= 0.")
    if config.max_abs_residual_warning < 0:
        raise ValueError("`max_abs_residual_warning` must be >= 0.")
    if config.residual_top_n <= 0:
        raise ValueError("`residual_top_n` must be > 0.")
