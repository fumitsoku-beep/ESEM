from __future__ import annotations

from typing import Any

from .data import ESEMSpec
from .model import ModelSpec, parse_model
from .model import model_spec_from_esem_spec
from .result import SEMResult


class SEMModel:
    """Minimal SEM model entry point.

    This class intentionally provides a small stable API first.
    Numerical estimation will be added in later milestones.
    """

    def __init__(self, syntax: str | None = None):
        self.spec: ModelSpec | None = parse_model(syntax) if syntax is not None else None

    def fit(self, data: Any, *, spec: ESEMSpec | None = None) -> SEMResult:
        """Fit model to data and return placeholder result.

        Parameters
        ----------
        data:
            Any tabular object with a reliable ``len(data)``.
        spec:
            Optional ESEMSpec model definition. Use this when SEMModel is
            initialized without syntax.
        """
        n_obs = _resolve_n_obs(data)
        model_spec = self._resolve_model_spec(spec)
        estimator = (model_spec.estimator or "ml").lower()

        return SEMResult(
            converged=True,
            n_obs=n_obs,
            parameters={},
            fit_indices={},
            parameter_table=(),
            warnings=(),
            optimization_info={
                "status": "placeholder",
                "n_iter": 0,
                "objective": float("nan"),
            },
            estimator=estimator,
            model_spec=model_spec,
        )

    def _resolve_model_spec(self, spec: ESEMSpec | None) -> ModelSpec:
        if spec is not None:
            if self.spec is not None:
                raise ValueError(
                    "Model is already defined by syntax. Provide either syntax in "
                    "`SEMModel(...)` or `spec` in `fit(...)`, not both."
                )
            return model_spec_from_esem_spec(spec)
        if self.spec is None:
            raise ValueError(
                "No model definition found. Provide syntax in `SEMModel(...)` "
                "or pass `spec` to `fit(...)`."
            )
        return self.spec


def sem(syntax: str, data: Any) -> SEMResult:
    """Convenience API for one-off model fitting."""
    return SEMModel(syntax).fit(data)


def _resolve_n_obs(data: Any) -> int:
    try:
        return int(len(data))
    except TypeError as exc:
        raise TypeError("`data` must be a sized tabular object.") from exc
