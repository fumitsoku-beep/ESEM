from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class StructuralPath:
	"""One structural path entry with parameter metadata."""

	source: str
	target: str
	source_is_latent: bool
	target_is_latent: bool
	is_free: bool
	parameter: str | None
	parameter_index: int | None
	vector_position: int | None
	fixed_value: float | None
	relation_index: int
	term_index: int


@dataclass(frozen=True)
class StructuralDisturbance:
	"""One latent disturbance variance parameter entry."""

	latent: str
	is_free: bool
	parameter: str | None
	parameter_index: int | None
	vector_position: int | None
	fixed_value: float | None
	relation_index: int
	term_index: int


@dataclass(frozen=True)
class StructuralDesign:
	"""Structural-layer matrix representation for SEM."""

	path_table: tuple[StructuralPath, ...]
	endogenous_latent_variables: tuple[str, ...]
	exogenous_latent_variables: tuple[str, ...]
	observed_predictor_variables: tuple[str, ...]
	observed_endogenous_variables: tuple[str, ...]
	beta_matrix: pd.DataFrame
	beta_parameter_index: pd.DataFrame
	gamma_matrix: pd.DataFrame
	gamma_parameter_index: pd.DataFrame
	psi_matrix: pd.DataFrame
	psi_parameter_index: pd.DataFrame
	disturbance_parameters: tuple[StructuralDisturbance, ...] = field(default_factory=tuple)
	warnings: tuple[str, ...] = field(default_factory=tuple)


__all__ = ["StructuralDesign", "StructuralDisturbance", "StructuralPath"]
