from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class MLEstimationContext:
	"""Prepared input state for ML estimation."""

	observed_variables: tuple[str, ...]
	sample_covariance: pd.DataFrame | None
	objective_at_sample_cov: float | None
	warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ParameterBoundsConfig:
	"""Rule-based bounds for SEM free parameters."""

	default_lower: float | None = None
	default_upper: float | None = None
	loading_lower: float | None = None
	loading_upper: float | None = None
	regression_lower: float | None = None
	regression_upper: float | None = None
	variance_lower: float | None = 1e-6
	variance_upper: float | None = None

	def __post_init__(self) -> None:
		_validate_bounds_pair(self.default_lower, self.default_upper, name="default")
		_validate_bounds_pair(self.loading_lower, self.loading_upper, name="loading")
		_validate_bounds_pair(self.regression_lower, self.regression_upper, name="regression")
		_validate_bounds_pair(self.variance_lower, self.variance_upper, name="variance")

	def select(
		self,
		*,
		operator: str,
		lhs: str | None,
		rhs: str | None,
	) -> tuple[float | None, float | None]:
		"""Select bounds for one parameter row based on operator/role."""
		lower = self.default_lower
		upper = self.default_upper
		if operator == "=~":
			lower = _merge_lower(lower, self.loading_lower)
			upper = _merge_upper(upper, self.loading_upper)
		elif operator == "~":
			lower = _merge_lower(lower, self.regression_lower)
			upper = _merge_upper(upper, self.regression_upper)
		elif operator == "~~" and lhs is not None and rhs is not None and lhs == rhs:
			lower = _merge_lower(lower, self.variance_lower)
			upper = _merge_upper(upper, self.variance_upper)
		return lower, upper


@dataclass(frozen=True)
class SEMFitConfig:
	"""Configuration for SEM ML optimization prototype."""

	max_iter: int = 200
	method: str = "L-BFGS-B"
	tol: float = 1e-8
	restarts: int = 0
	random_seed: int | None = None
	random_start_scale: float = 0.15
	bounds: ParameterBoundsConfig = field(default_factory=ParameterBoundsConfig)

	def __post_init__(self) -> None:
		if self.max_iter < 1:
			raise ValueError("`max_iter` must be >= 1.")
		if self.tol <= 0:
			raise ValueError("`tol` must be > 0.")
		if self.restarts < 0:
			raise ValueError("`restarts` must be >= 0.")
		if self.random_start_scale < 0:
			raise ValueError("`random_start_scale` must be >= 0.")
		if self.method != "L-BFGS-B":
			raise ValueError("Prototype currently supports only `method='L-BFGS-B'`.")


@dataclass(frozen=True)
class MLOptimizationResult:
	"""Output bundle for ML prototype optimization."""

	success: bool
	status: str
	n_iter: int
	objective: float | None
	observed_variables: tuple[str, ...]
	parameter_vector: tuple[float, ...]
	parameter_values: dict[str, float] = field(default_factory=dict)
	sample_covariance: pd.DataFrame | None = None
	implied_covariance: pd.DataFrame | None = None
	n_attempts: int = 1
	best_attempt: int | None = None
	method: str | None = None
	failure_category: str | None = None
	failure_reason: str | None = None
	attempt_objectives: tuple[float | None, ...] = field(default_factory=tuple)
	warnings: tuple[str, ...] = field(default_factory=tuple)


def _validate_bounds_pair(
	lower: float | None,
	upper: float | None,
	*,
	name: str,
) -> None:
	if lower is not None and upper is not None and lower > upper:
		raise ValueError(f"`{name}` bounds require lower <= upper.")


def _merge_lower(current: float | None, candidate: float | None) -> float | None:
	if current is None:
		return candidate
	if candidate is None:
		return current
	return max(current, candidate)


def _merge_upper(current: float | None, candidate: float | None) -> float | None:
	if current is None:
		return candidate
	if candidate is None:
		return current
	return min(current, candidate)


__all__ = [
	"MLEstimationContext",
	"MLOptimizationResult",
	"ParameterBoundsConfig",
	"SEMFitConfig",
]
