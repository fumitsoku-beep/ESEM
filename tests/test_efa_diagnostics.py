import numpy as np
import pandas as pd
import pytest

from psysem import EFADiagnosticsConfig, run_efa_diagnostics


def _synthetic_efa_data(n: int = 400, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(n, 2))
    loadings = np.array(
        [
            [0.80, 0.10],
            [0.75, 0.15],
            [0.70, 0.05],
            [0.10, 0.80],
            [0.15, 0.75],
            [0.05, 0.70],
        ]
    )
    noise = rng.normal(scale=0.45, size=(n, 6))
    observed = factors @ loadings.T + noise
    columns = [f"i{i}" for i in range(1, 7)]
    return pd.DataFrame(observed, columns=columns)


def test_run_efa_diagnostics_smoke() -> None:
    data = _synthetic_efa_data()
    config = EFADiagnosticsConfig(items=("i1", "i2", "i3", "i4", "i5", "i6"))
    result = run_efa_diagnostics(data, config)

    assert result.n_obs == 400
    assert result.n_items == 6
    assert 0.0 <= result.kmo_total <= 1.0
    assert result.kmo_per_item.shape[0] == 6
    assert result.bartlett_df == 15
    assert 0.0 <= result.bartlett_p <= 1.0
    assert result.correlation_matrix.shape == (6, 6)
    assert result.kmo_label in {
        "unacceptable",
        "miserable",
        "mediocre",
        "middling",
        "meritorious",
        "marvelous",
    }


def test_run_efa_diagnostics_rejects_missing_items() -> None:
    data = _synthetic_efa_data()
    config = EFADiagnosticsConfig(items=("i1", "i2", "i3", "i4", "i5", "i7"))
    with pytest.raises(ValueError, match="Missing item columns"):
        run_efa_diagnostics(data, config)


def test_run_efa_diagnostics_warns_low_sample_ratio() -> None:
    data = _synthetic_efa_data(n=20)
    config = EFADiagnosticsConfig(
        items=("i1", "i2", "i3", "i4", "i5", "i6"),
        min_sample_ratio=10.0,
    )
    result = run_efa_diagnostics(data, config)
    assert any("sample ratio" in msg.lower() for msg in result.warnings)


def test_run_efa_diagnostics_dropna_true_updates_n_obs() -> None:
    data = _synthetic_efa_data()
    data.loc[0, "i1"] = np.nan
    data.loc[1, "i2"] = np.nan
    config = EFADiagnosticsConfig(items=("i1", "i2", "i3", "i4", "i5", "i6"), dropna=True)
    result = run_efa_diagnostics(data, config)
    assert result.n_obs == 398


def test_run_efa_diagnostics_dropna_false_rejects_missing_values() -> None:
    data = _synthetic_efa_data()
    data.loc[0, "i1"] = np.nan
    config = EFADiagnosticsConfig(items=("i1", "i2", "i3", "i4", "i5", "i6"), dropna=False)
    with pytest.raises(ValueError, match="Missing values detected"):
        run_efa_diagnostics(data, config)


def test_run_efa_diagnostics_accepts_explicit_pairwise_missing_strategy() -> None:
    data = _synthetic_efa_data()
    data.loc[0, "i1"] = np.nan
    config = EFADiagnosticsConfig(
        items=("i1", "i2", "i3", "i4", "i5", "i6"),
        dropna=False,
        missing_strategy="pairwise",
    )
    result = run_efa_diagnostics(data, config)
    assert result.n_obs == 399
    assert any("pairwise" in msg.lower() for msg in result.warnings)


def test_run_efa_diagnostics_warns_for_constant_item() -> None:
    data = _synthetic_efa_data()
    data["i6"] = 1.0
    config = EFADiagnosticsConfig(items=("i1", "i2", "i3", "i4", "i5", "i6"))
    result = run_efa_diagnostics(data, config)
    assert any("constant item" in msg.lower() for msg in result.warnings)


def test_run_efa_diagnostics_handles_singular_corr_for_bartlett() -> None:
    data = _synthetic_efa_data()
    data["i2"] = data["i1"]
    config = EFADiagnosticsConfig(items=("i1", "i2", "i3", "i4", "i5", "i6"))
    result = run_efa_diagnostics(data, config)
    assert any("bartlett test" in msg.lower() for msg in result.warnings)


def test_run_efa_diagnostics_rejects_too_few_observations() -> None:
    data = _synthetic_efa_data(n=2)
    config = EFADiagnosticsConfig(items=("i1", "i2", "i3", "i4", "i5", "i6"))
    with pytest.raises(ValueError, match="At least 3 complete observations"):
        run_efa_diagnostics(data, config)


def test_run_efa_diagnostics_rejects_duplicate_items() -> None:
    data = _synthetic_efa_data()
    config = EFADiagnosticsConfig(items=("i1", "i1", "i2"))
    with pytest.raises(ValueError, match="duplicated"):
        run_efa_diagnostics(data, config)


def test_run_efa_diagnostics_rejects_blank_item_name() -> None:
    data = _synthetic_efa_data()
    config = EFADiagnosticsConfig(items=("i1", ""))
    with pytest.raises(ValueError, match="non-empty strings"):
        run_efa_diagnostics(data, config)


def test_run_efa_diagnostics_rejects_non_dataframe_input() -> None:
    config = EFADiagnosticsConfig(items=("i1", "i2"))
    with pytest.raises(TypeError, match="pandas.DataFrame"):
        run_efa_diagnostics(data=[1, 2, 3], config=config)  # type: ignore[arg-type]
