from psysem import compute_basic_fit_indices


def test_fit_indices_shape() -> None:
    indices = compute_basic_fit_indices()
    for key in ["cfi", "tli", "rmsea", "srmr", "aic", "bic"]:
        assert key in indices
