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
