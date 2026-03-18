from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from .estimation import gaussian_ml_discrepancy


FIT_INDEX_KEYS: tuple[str, ...] = ("cfi", "tli", "rmsea", "srmr", "aic", "bic")


@dataclass(frozen=True)
class FitIndicesResult:
	"""Computed SEM fit indices plus diagnostics."""

	indices: dict[str, float]
	chi_square: float | None = None
	df_model: int | None = None
	chi_square_baseline: float | None = None
	df_baseline: int | None = None
	status: str = "ok"
	failure_reason: str | None = None
	n_available_indices: int = 0
	n_unavailable_indices: int = 0
	warnings: tuple[str, ...] = field(default_factory=tuple)


def compute_basic_fit_indices(
	*,
	sample_covariance: np.ndarray | None = None,
	implied_covariance: np.ndarray | None = None,
	n_obs: int | None = None,
	n_free_parameters: int | None = None,
	objective: float | None = None,
) -> dict[str, float]:
	"""Return core fit indices.

	With no inputs, this keeps backward-compatible placeholder behavior by
	returning `nan` values.
	"""
	return compute_fit_indices(
		sample_covariance=sample_covariance,
		implied_covariance=implied_covariance,
		n_obs=n_obs,
		n_free_parameters=n_free_parameters,
		objective=objective,
	).indices


def compute_fit_indices(
	*,
	sample_covariance: np.ndarray | None,
	implied_covariance: np.ndarray | None,
	n_obs: int | None,
	n_free_parameters: int | None,
	objective: float | None = None,
) -> FitIndicesResult:
	"""Compute SEM fit indices using Gaussian ML discrepancy inputs."""
	warnings: list[str] = []
	indices = _placeholder_indices()
	if sample_covariance is None or implied_covariance is None:
		warnings.append("Fit indices unavailable: sample/implied covariance is missing.")
		return _finalize_fit_result(
			indices=indices,
			warnings=warnings,
			failure_reason="missing_covariance_input",
		)

	try:
		sample = _coerce_covariance(sample_covariance, name="sample_covariance")
		implied = _coerce_covariance(implied_covariance, name="implied_covariance")
	except ValueError as exc:
		warnings.append(f"Fit indices unavailable: {exc}")
		return _finalize_fit_result(
			indices=indices,
			warnings=warnings,
			failure_reason="invalid_covariance_input",
		)

	if sample.shape != implied.shape:
		warnings.append("Fit indices unavailable: sample/implied covariance shape mismatch.")
		return _finalize_fit_result(
			indices=indices,
			warnings=warnings,
			failure_reason="covariance_shape_mismatch",
		)

	p = int(sample.shape[0])
	if p == 0:
		warnings.append("Fit indices unavailable: empty covariance matrices.")
		return _finalize_fit_result(
			indices=indices,
			warnings=warnings,
			failure_reason="empty_covariance_matrices",
		)

	failure_reason: str | None = None

	def note_failure(reason: str) -> None:
		nonlocal failure_reason
		if failure_reason is None:
			failure_reason = reason

	if n_obs is None or n_obs <= 1:
		warnings.append("Fit indices unavailable: `n_obs` must be greater than 1.")
		note_failure("invalid_sample_size")
	if n_free_parameters is None or n_free_parameters < 0:
		warnings.append("Fit indices unavailable: `n_free_parameters` must be >= 0.")
		note_failure("invalid_n_free_parameters")

	objective_value: float | None = objective
	if objective_value is None or not math.isfinite(objective_value):
		try:
			objective_value = gaussian_ml_discrepancy(sample, implied)
		except ValueError as exc:
			warnings.append(f"Failed to compute model discrepancy for fit indices: {exc}")
			objective_value = None
			note_failure("model_discrepancy_unavailable")

	chi_square: float | None = None
	df_model: int | None = None
	if (
		objective_value is not None
		and math.isfinite(objective_value)
		and n_obs is not None
		and n_obs > 1
		and n_free_parameters is not None
		and n_free_parameters >= 0
	):
		chi_square = max(float((n_obs - 1) * objective_value), 0.0)
		df_model = int((p * (p + 1)) // 2 - n_free_parameters)
		indices["aic"] = chi_square + 2.0 * float(n_free_parameters)
		indices["bic"] = chi_square + math.log(float(n_obs)) * float(n_free_parameters)
	else:
		warnings.append("AIC/BIC unavailable due to missing objective or invalid n_obs/n_free.")
		note_failure("aic_bic_unavailable")

	try:
		sample_corr = _covariance_to_correlation(sample)
		implied_corr = _covariance_to_correlation(implied)
		triu = np.triu_indices(p, k=1)
		if triu[0].size == 0:
			indices["srmr"] = 0.0
		else:
			diff = sample_corr[triu] - implied_corr[triu]
			indices["srmr"] = float(np.sqrt(np.mean(diff * diff)))
	except ValueError as exc:
		warnings.append(f"SRMR unavailable: {exc}")
		note_failure("srmr_unavailable")

	chi_square_baseline: float | None = None
	df_baseline: int | None = None
	if n_obs is not None and n_obs > 1:
		baseline = np.diag(np.diag(sample))
		try:
			baseline_objective = gaussian_ml_discrepancy(sample, baseline)
			chi_square_baseline = max(float((n_obs - 1) * baseline_objective), 0.0)
			df_baseline = int((p * (p - 1)) // 2)
		except ValueError as exc:
			warnings.append(f"Baseline model discrepancy unavailable: {exc}")
			note_failure("baseline_discrepancy_unavailable")

	if (
		chi_square is None
		or df_model is None
		or chi_square_baseline is None
		or df_baseline is None
		or n_obs is None
		or n_obs <= 1
		or df_model <= 0
		or df_baseline <= 0
	):
		warnings.append("CFI/TLI/RMSEA unavailable because model/baseline degrees of freedom are invalid.")
		note_failure("invalid_model_degrees_of_freedom")
		return _finalize_fit_result(
			indices=indices,
			chi_square=chi_square,
			df_model=df_model,
			chi_square_baseline=chi_square_baseline,
			df_baseline=df_baseline,
			warnings=warnings,
			failure_reason=failure_reason,
		)

	denominator = chi_square_baseline - float(df_baseline)
	if denominator <= 0.0:
		warnings.append("CFI unavailable: baseline denominator is non-positive.")
		note_failure("cfi_baseline_denominator_non_positive")
	else:
		cfi_raw = 1.0 - max(chi_square - float(df_model), 0.0) / denominator
		indices["cfi"] = float(min(1.0, max(0.0, cfi_raw)))

	baseline_ratio = chi_square_baseline / float(df_baseline)
	model_ratio = chi_square / float(df_model)
	tli_denominator = baseline_ratio - 1.0
	if math.isclose(tli_denominator, 0.0, rel_tol=0.0, abs_tol=1e-12):
		warnings.append("TLI unavailable: baseline ratio denominator is near zero.")
		note_failure("tli_baseline_ratio_denominator_near_zero")
	else:
		tli_raw = (baseline_ratio - model_ratio) / tli_denominator
		indices["tli"] = float(min(1.0, max(0.0, tli_raw)))

	rmsea_term = (chi_square - float(df_model)) / (float(df_model) * float(n_obs - 1))
	indices["rmsea"] = float(math.sqrt(max(rmsea_term, 0.0)))

	return _finalize_fit_result(
		indices=indices,
		chi_square=chi_square,
		df_model=df_model,
		chi_square_baseline=chi_square_baseline,
		df_baseline=df_baseline,
		warnings=warnings,
		failure_reason=failure_reason,
	)


def _placeholder_indices() -> dict[str, float]:
	return {key: float("nan") for key in FIT_INDEX_KEYS}


def _finalize_fit_result(
	*,
	indices: dict[str, float],
	warnings: list[str],
	chi_square: float | None = None,
	df_model: int | None = None,
	chi_square_baseline: float | None = None,
	df_baseline: int | None = None,
	failure_reason: str | None = None,
) -> FitIndicesResult:
	n_available_indices = sum(1 for value in indices.values() if math.isfinite(value))
	n_unavailable_indices = len(indices) - n_available_indices
	if n_available_indices == len(indices):
		status = "ok"
		failure_reason = None
	elif n_available_indices == 0:
		status = "failed"
		failure_reason = failure_reason or "fit_indices_unavailable"
	else:
		status = "partial"
		failure_reason = failure_reason or "fit_indices_partially_unavailable"

	return FitIndicesResult(
		indices=indices,
		chi_square=chi_square,
		df_model=df_model,
		chi_square_baseline=chi_square_baseline,
		df_baseline=df_baseline,
		status=status,
		failure_reason=failure_reason,
		n_available_indices=n_available_indices,
		n_unavailable_indices=n_unavailable_indices,
		warnings=tuple(dict.fromkeys(warnings)),
	)


def _coerce_covariance(covariance: np.ndarray, *, name: str) -> np.ndarray:
	array = np.asarray(covariance, dtype=float)
	if array.ndim != 2:
		raise ValueError(f"`{name}` must be 2D.")
	if array.shape[0] != array.shape[1]:
		raise ValueError(f"`{name}` must be square.")
	return array


def _covariance_to_correlation(covariance: np.ndarray) -> np.ndarray:
	diag = np.diag(covariance)
	if np.any(diag <= 0):
		raise ValueError("covariance diagonal contains non-positive values.")
	scale = np.sqrt(diag)
	denominator = np.outer(scale, scale)
	correlation = covariance / denominator
	correlation = (correlation + correlation.T) / 2.0
	return correlation


__all__ = [
	"FIT_INDEX_KEYS",
	"FitIndicesResult",
	"compute_basic_fit_indices",
	"compute_fit_indices",
]
