import json
import math
from functools import lru_cache
from pathlib import Path

import pandas as pd
import pytest

from psysem import SEMModel


DATA_DIR = Path(__file__).parent / "data"
HS1939_COLUMNS = ("x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9")
HS1939_SYNTAX = "\n".join(
    (
        "visual =~ x1 + x2 + x3",
        "textual =~ x4 + x5 + x6",
        "speed =~ x7 + x8 + x9",
    )
)


@lru_cache(maxsize=1)
def _load_hs1939_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_DIR / "benchmark_hs1939_raw.csv")
    return data.loc[:, list(HS1939_COLUMNS)]


@lru_cache(maxsize=1)
def _load_hs1939_reference() -> dict:
    return json.loads((DATA_DIR / "benchmark_hs1939_reference.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _fit_hs1939() -> object:
    return SEMModel(HS1939_SYNTAX).fit(_load_hs1939_data())


def _loading_estimate(result: object, *, latent: str, indicator: str) -> float:
    row = next(
        row
        for row in result.parameter_table
        if row["operator"] == "=~" and row["lhs"] == latent and row["rhs"] == indicator
    )
    return float(result.parameters[str(row["parameter"])])


def test_hs1939_benchmark_assets_include_provenance_metadata() -> None:
    reference = _load_hs1939_reference()
    assert reference["dataset_name"] == "HolzingerSwineford1939"
    assert reference["local_csv"] == "tests/data/benchmark_hs1939_raw.csv"
    assert reference["retrieved_via"]["package_name"] == "lavaan"
    assert reference["retrieved_via"]["package_license"] == "GPL-2 | GPL-3"
    assert reference["row_count"] == 301
    assert reference["column_count"] == 15
    assert reference["copyright_and_reuse"]["publicly_available"] is True
    assert reference["copyright_and_reuse"]["public_domain"] is False
    assert reference["copyright_and_reuse"]["citation_recommended"] is True
    assert "public domain" in reference["copyright_and_reuse"]["public_domain_note"].lower()


def test_hs1939_benchmark_level_a_fit_contract() -> None:
    result = _fit_hs1939()
    assert result.n_obs == 301
    assert result.converged is True
    assert result.parameter_inference
    assert result.optimization_info["fit_status"] in {"ok", "partial"}
    assert result.optimization_info["df_model"] > 0
    for key in ("cfi", "tli", "rmsea", "srmr", "aic", "bic"):
        assert key in result.fit_indices
        assert math.isfinite(result.fit_indices[key])


def test_hs1939_benchmark_level_b_loading_direction_and_order() -> None:
    result = _fit_hs1939()
    loadings = {
        (latent, indicator): _loading_estimate(result, latent=latent, indicator=indicator)
        for latent, indicator in (
            ("visual", "x1"),
            ("visual", "x2"),
            ("visual", "x3"),
            ("textual", "x4"),
            ("textual", "x5"),
            ("textual", "x6"),
            ("speed", "x7"),
            ("speed", "x8"),
            ("speed", "x9"),
        )
    }
    assert all(value > 0.0 for value in loadings.values())
    assert loadings[("visual", "x3")] > loadings[("visual", "x2")]
    assert loadings[("textual", "x5")] > loadings[("textual", "x6")]
    assert loadings[("speed", "x8")] > loadings[("speed", "x9")]


def test_hs1939_benchmark_level_c_partial_reference_alignment() -> None:
    reference = _load_hs1939_reference()
    result = _fit_hs1939()
    targets = reference["reference_statistics"]
    loading_tol = reference["comparison_policy"]["selected_loading_abs_tol"]
    fit_tol = reference["comparison_policy"]["fit_index_abs_tol"]

    assert _loading_estimate(result, latent="visual", indicator="x2") == pytest.approx(
        targets["selected_loadings"]["visual_x2"], abs=loading_tol
    )
    assert _loading_estimate(result, latent="visual", indicator="x3") == pytest.approx(
        targets["selected_loadings"]["visual_x3"], abs=loading_tol
    )
    assert _loading_estimate(result, latent="textual", indicator="x5") == pytest.approx(
        targets["selected_loadings"]["textual_x5"], abs=loading_tol
    )
    assert _loading_estimate(result, latent="textual", indicator="x6") == pytest.approx(
        targets["selected_loadings"]["textual_x6"], abs=loading_tol
    )
    assert result.fit_indices["cfi"] == pytest.approx(targets["fit_indices"]["cfi"], abs=fit_tol)
    assert result.fit_indices["tli"] == pytest.approx(targets["fit_indices"]["tli"], abs=fit_tol)
    assert result.fit_indices["rmsea"] == pytest.approx(targets["fit_indices"]["rmsea"], abs=0.05)
    assert result.fit_indices["srmr"] < 0.20