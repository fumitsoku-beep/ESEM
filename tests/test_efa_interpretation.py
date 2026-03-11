import numpy as np
import pandas as pd
import pytest
from typing import Any

from psysem import EFAConfig, EFAInterpretationConfig, fit_efa, interpret_efa


def _synthetic_efa_data(n: int = 420, seed: int = 2026) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(n, 2))
    loadings = np.array(
        [
            [0.84, 0.08],
            [0.80, 0.12],
            [0.77, 0.10],
            [0.08, 0.83],
            [0.11, 0.79],
            [0.09, 0.76],
        ]
    )
    noise = rng.normal(scale=0.40, size=(n, 6))
    observed = factors @ loadings.T + noise
    return pd.DataFrame(observed, columns=[f"i{i}" for i in range(1, 7)])


def test_interpret_efa_smoke() -> None:
    data = _synthetic_efa_data()
    result = fit_efa(
        data,
        EFAConfig(items=("i1", "i2", "i3", "i4", "i5", "i6"), n_factors=2),
    )
    interpretation = interpret_efa(result)

    assert not interpretation.item_table.empty
    assert not interpretation.factor_table.empty
    assert set(interpretation.item_table.columns) >= {
        "primary_factor",
        "primary_loading",
        "h2",
        "u2",
        "com",
        "n_salient_loadings",
        "n_cross_loadings",
        "is_cross_loaded",
        "is_low_h2",
    }
    assert set(interpretation.factor_table.columns) >= {
        "ss_loadings",
        "proportion_var",
        "cumulative_var",
        "n_salient_items",
        "mean_abs_loading",
    }
    assert set(interpretation.summary) >= {
        "n_items",
        "n_factors",
        "n_cross_loaded_items",
        "n_low_h2_items",
        "rmsr",
        "max_abs_residual",
        "min_h2_threshold",
    }


def test_interpret_efa_residual_top_pairs_has_expected_columns() -> None:
    data = _synthetic_efa_data()
    result = fit_efa(
        data,
        EFAConfig(items=("i1", "i2", "i3", "i4", "i5", "i6"), n_factors=2),
    )
    interpretation = interpret_efa(
        result,
        EFAInterpretationConfig(residual_top_n=5),
    )
    assert interpretation.residual_top_pairs.shape[0] <= 5
    assert list(interpretation.residual_top_pairs.columns) == [
        "item_i",
        "item_j",
        "residual",
        "abs_residual",
    ]


def test_interpret_efa_rejects_invalid_threshold_relation() -> None:
    data = _synthetic_efa_data()
    result = fit_efa(
        data,
        EFAConfig(items=("i1", "i2", "i3", "i4", "i5", "i6"), n_factors=2),
    )
    with pytest.raises(ValueError, match="cross_loading"):
        interpret_efa(
            result,
            EFAInterpretationConfig(salient_loading=0.40, cross_loading=0.30),
        )


def test_interpret_efa_warnings_trigger_with_strict_thresholds() -> None:
    data = _synthetic_efa_data()
    result = fit_efa(
        data,
        EFAConfig(items=("i1", "i2", "i3", "i4", "i5", "i6"), n_factors=2),
    )
    interpretation = interpret_efa(
        result,
        EFAInterpretationConfig(
            min_h2=0.99,
            max_abs_residual_warning=0.0,
            rmsr_warning=0.0,
            min_salient_items_per_factor=10,
            residual_top_n=5,
        ),
    )
    assert interpretation.warnings


def test_interpret_efa_residual_top_pairs_sorted_by_abs_desc() -> None:
    data = _synthetic_efa_data()
    result = fit_efa(
        data,
        EFAConfig(items=("i1", "i2", "i3", "i4", "i5", "i6"), n_factors=2),
    )
    interpretation = interpret_efa(result, EFAInterpretationConfig(residual_top_n=10))
    abs_values = interpretation.residual_top_pairs["abs_residual"].to_numpy(dtype=float)
    assert np.all(abs_values[:-1] >= abs_values[1:])


def test_interpret_efa_primary_loading_matches_result_max_abs_loading() -> None:
    data = _synthetic_efa_data()
    result = fit_efa(
        data,
        EFAConfig(items=("i1", "i2", "i3", "i4", "i5", "i6"), n_factors=2),
    )
    interpretation = interpret_efa(result)
    expected = result.loadings.abs().max(axis=1)
    observed = interpretation.item_table["primary_loading"]
    assert np.allclose(expected.to_numpy(dtype=float), observed.to_numpy(dtype=float))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"salient_loading": 0.0}, "salient_loading"),
        ({"min_h2": -0.1}, "min_h2"),
        ({"min_h2": 1.1}, "min_h2"),
        ({"min_salient_items_per_factor": 0}, "min_salient_items_per_factor"),
        ({"rmsr_warning": -1.0}, "rmsr_warning"),
        ({"max_abs_residual_warning": -1.0}, "max_abs_residual_warning"),
        ({"residual_top_n": 0}, "residual_top_n"),
    ],
)
def test_interpret_efa_rejects_invalid_config_values(
    kwargs: dict[str, object],
    match: str,
) -> None:
    data = _synthetic_efa_data()
    result = fit_efa(
        data,
        EFAConfig(items=("i1", "i2", "i3", "i4", "i5", "i6"), n_factors=2),
    )
    with pytest.raises(ValueError, match=match):
        config_kwargs: dict[str, Any] = kwargs
        interpret_efa(result, EFAInterpretationConfig(**config_kwargs))
