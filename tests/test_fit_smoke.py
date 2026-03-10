import pandas as pd
import pytest

from psysem import SEMModel, parse_model, sem


def test_parse_model_rejects_empty() -> None:
    with pytest.raises(ValueError):
        parse_model("   ")


def test_smoke_fit() -> None:
    data = pd.DataFrame({"x1": [1.0, 2.0, 3.0], "x2": [1.0, 1.5, 2.5], "y": [1.1, 2.2, 3.1]})
    model = SEMModel("y ~ x1 + x2")
    result = model.fit(data)
    assert result.converged is True
    assert result.n_obs == 3


def test_sem_function() -> None:
    data = pd.DataFrame({"x1": [1.0, 2.0], "y": [1.2, 2.4]})
    result = sem("y ~ x1", data)
    assert result.n_obs == 2
