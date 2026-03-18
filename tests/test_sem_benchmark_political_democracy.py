import json
import math
from functools import lru_cache
from pathlib import Path

import pandas as pd
import pytest

from psysem import SEMModel


DATA_DIR = Path(__file__).parent / "data"
PD_COLUMNS = ("x1", "x2", "x3", "y1", "y2", "y3", "y4", "y5", "y6", "y7", "y8")
PD_SYNTAX = "\n".join(
    (
        "ind60 =~ 1*x1 + x2 + x3",
        "dem60 =~ 1*y1 + y2 + y3 + y4",
        "dem65 =~ 1*y5 + y6 + y7 + y8",
        "",
        "dem60 ~ ind60",
        "dem65 ~ ind60 + dem60",
        "",
        "y1 ~~ y5",
        "y2 ~~ y4 + y6",
        "y3 ~~ y7",
        "y4 ~~ y8",
        "y6 ~~ y8",
    )
)


@lru_cache(maxsize=1)
def _load_pd_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_DIR / "benchmark_political_democracy_raw.csv")
    return data.loc[:, list(PD_COLUMNS)]


@lru_cache(maxsize=1)
def _load_pd_reference() -> dict:
    return json.loads((DATA_DIR / "benchmark_political_democracy_reference.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _fit_pd() -> object:
    return SEMModel(PD_SYNTAX).fit(_load_pd_data())


def _parameter_estimate(result: object, *, lhs: str, operator: str, rhs: str) -> float:
    row = next(
        row
        for row in result.parameter_table
        if row["lhs"] == lhs and row["operator"] == operator and row["rhs"] == rhs
    )
    return float(result.parameters[str(row["parameter"])])


def test_political_democracy_benchmark_assets_include_provenance_metadata() -> None:
    reference = _load_pd_reference()
    assert reference["dataset_name"] == "PoliticalDemocracy"
    assert reference["local_csv"] == "tests/data/benchmark_political_democracy_raw.csv"
    assert reference["retrieved_via"]["package_name"] == "lavaan"
    assert reference["row_count"] == 75
    assert reference["column_count"] == 11
    assert reference["copyright_and_reuse"]["publicly_available"] is True
    assert reference["copyright_and_reuse"]["public_domain"] is False
    assert reference["copyright_and_reuse"]["citation_recommended"] is True


def test_political_democracy_benchmark_level_a_fit_contract() -> None:
    result = _fit_pd()
    assert result.n_obs == 75
    assert result.converged is True
    assert result.parameter_inference
    assert result.optimization_info["fit_status"] == "ok"
    assert result.optimization_info["df_model"] > 0
    assert result.optimization_info.get("inference_status") == "ok"
    assert not any("no fixed loading marker" in warning.lower() for warning in result.warnings)
    for key in ("cfi", "tli", "rmsea", "srmr", "aic", "bic"):
        assert key in result.fit_indices
        assert math.isfinite(result.fit_indices[key])


def test_political_democracy_benchmark_level_b_regression_direction_and_order() -> None:
    result = _fit_pd()
    dem60_on_ind60 = _parameter_estimate(result, lhs="dem60", operator="~", rhs="ind60")
    dem65_on_ind60 = _parameter_estimate(result, lhs="dem65", operator="~", rhs="ind60")
    dem65_on_dem60 = _parameter_estimate(result, lhs="dem65", operator="~", rhs="dem60")

    assert dem60_on_ind60 > 0.0
    assert dem65_on_ind60 > 0.0
    assert dem65_on_dem60 > 0.0
    assert dem65_on_dem60 > dem65_on_ind60


def test_political_democracy_benchmark_level_b_selected_fit_profile() -> None:
    result = _fit_pd()
    assert result.fit_indices["cfi"] > 0.90
    assert result.fit_indices["tli"] > 0.85
    assert result.fit_indices["srmr"] < 0.12
    assert result.fit_indices["rmsea"] < 0.15


def test_political_democracy_benchmark_level_c_selected_reference_alignment() -> None:
    reference = _load_pd_reference()
    result = _fit_pd()
    targets = reference["reference_statistics"]
    loading_tol = reference["comparison_policy"]["selected_loading_abs_tol"]
    regression_tol = reference["comparison_policy"]["selected_regression_abs_tol"]
    residual_variance_tol = reference["comparison_policy"]["selected_residual_variance_abs_tol"]

    assert _parameter_estimate(result, lhs="ind60", operator="=~", rhs="x2") == pytest.approx(
        targets["selected_loadings"]["ind60_x2"], abs=loading_tol
    )
    assert _parameter_estimate(result, lhs="ind60", operator="=~", rhs="x3") == pytest.approx(
        targets["selected_loadings"]["ind60_x3"], abs=loading_tol
    )
    assert _parameter_estimate(result, lhs="dem60", operator="=~", rhs="y2") == pytest.approx(
        targets["selected_loadings"]["dem60_y2"], abs=loading_tol
    )
    assert _parameter_estimate(result, lhs="dem60", operator="=~", rhs="y3") == pytest.approx(
        targets["selected_loadings"]["dem60_y3"], abs=loading_tol
    )
    assert _parameter_estimate(result, lhs="dem60", operator="=~", rhs="y4") == pytest.approx(
        targets["selected_loadings"]["dem60_y4"], abs=loading_tol
    )
    assert _parameter_estimate(result, lhs="dem65", operator="=~", rhs="y6") == pytest.approx(
        targets["selected_loadings"]["dem65_y6"], abs=loading_tol
    )
    assert _parameter_estimate(result, lhs="dem65", operator="=~", rhs="y7") == pytest.approx(
        targets["selected_loadings"]["dem65_y7"], abs=loading_tol
    )
    assert _parameter_estimate(result, lhs="dem65", operator="=~", rhs="y8") == pytest.approx(
        targets["selected_loadings"]["dem65_y8"], abs=loading_tol
    )

    assert _parameter_estimate(result, lhs="dem60", operator="~", rhs="ind60") == pytest.approx(
        targets["selected_regressions"]["dem60_on_ind60"], abs=regression_tol
    )
    assert _parameter_estimate(result, lhs="dem65", operator="~", rhs="ind60") == pytest.approx(
        targets["selected_regressions"]["dem65_on_ind60"], abs=regression_tol
    )
    assert _parameter_estimate(result, lhs="dem65", operator="~", rhs="dem60") == pytest.approx(
        targets["selected_regressions"]["dem65_on_dem60"], abs=regression_tol
    )

    for variable, expected in targets["selected_residual_variances"].items():
        assert _parameter_estimate(result, lhs=variable, operator="~~", rhs=variable) == pytest.approx(
            expected,
            abs=residual_variance_tol,
        )


def test_political_democracy_residual_covariance_reference_alignment() -> None:
    reference = _load_pd_reference()
    result = _fit_pd()
    targets = reference["reference_statistics"]["selected_residual_covariances"]

    assert _parameter_estimate(result, lhs="y1", operator="~~", rhs="y5") == pytest.approx(
        targets["y1~~y5"], abs=0.5
    )
    assert _parameter_estimate(result, lhs="y2", operator="~~", rhs="y4") == pytest.approx(
        targets["y2~~y4"], abs=0.5
    )
    assert _parameter_estimate(result, lhs="y2", operator="~~", rhs="y6") == pytest.approx(
        targets["y2~~y6"], abs=0.5
    )
    assert _parameter_estimate(result, lhs="y3", operator="~~", rhs="y7") == pytest.approx(
        targets["y3~~y7"], abs=0.5
    )
    assert _parameter_estimate(result, lhs="y4", operator="~~", rhs="y8") == pytest.approx(
        targets["y4~~y8"], abs=0.5
    )
    assert _parameter_estimate(result, lhs="y6", operator="~~", rhs="y8") == pytest.approx(
        targets["y6~~y8"], abs=0.5
    )
