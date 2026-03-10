from __future__ import annotations

from typing import Any

from .model import ModelSpec, parse_model
from .result import SEMResult


class SEMModel:
    """Minimal SEM model entry point.

    This class intentionally provides a small stable API first.
    Numerical estimation will be added in later milestones.
    """

    def __init__(self, syntax: str):
        self.spec: ModelSpec = parse_model(syntax)

    def fit(self, data: Any) -> SEMResult:
        """Fit model to data and return placeholder result.

        Parameters
        ----------
        data:
            Any tabular object with a reliable ``len(data)``.
        """
        n_obs = len(data)
        return SEMResult(
            converged=True,
            n_obs=n_obs,
            parameters={},
            fit_indices={},
        )


def sem(syntax: str, data: Any) -> SEMResult:
    """Convenience API for one-off model fitting."""
    return SEMModel(syntax).fit(data)
