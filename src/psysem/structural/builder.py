from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..model import ModelSpec, RelationTerm
from .contracts import StructuralDesign, StructuralPath


def build_structural_design(
    model_spec: ModelSpec,
    *,
    parameter_table: tuple[dict[str, Any], ...] | None = None,
) -> StructuralDesign:
    """Build structural Beta/Gamma placeholders from regression relations."""
    structural_relations = tuple(
        (relation_index, relation)
        for relation_index, relation in enumerate(model_spec.relations, start=1)
        if relation.operator == "~"
    )
    if not structural_relations:
        raise ValueError("No structural relations (`~`) found in model.")

    latent_set = set(model_spec.latent_variables)
    observed_set = set(model_spec.observed_variables)
    allowed = latent_set | observed_set
    parameter_lookup = _parameter_lookup(parameter_table)
    fallback_label_index: dict[str, int] = {}
    fallback_next_index = 1

    endogenous_latent: list[str] = []
    endogenous_latent_seen: set[str] = set()
    observed_endogenous: list[str] = []
    observed_endogenous_seen: set[str] = set()
    observed_predictors: list[str] = []
    observed_predictors_seen: set[str] = set()
    path_table: list[StructuralPath] = []

    latent_edges: list[tuple[str, str]] = []
    for relation_index, relation in structural_relations:
        target = relation.lhs
        if target not in allowed:
            raise ValueError(f"Unknown structural target `{target}`.")
        target_is_latent = target in latent_set
        if target_is_latent:
            if target not in endogenous_latent_seen:
                endogenous_latent_seen.add(target)
                endogenous_latent.append(target)
        else:
            if target not in observed_endogenous_seen:
                observed_endogenous_seen.add(target)
                observed_endogenous.append(target)

        for term_index, term in enumerate(relation.terms, start=1):
            source = term.variable
            if source not in allowed:
                raise ValueError(
                    f"Unknown structural predictor `{source}` in `{target} ~ ...`."
                )
            source_is_latent = source in latent_set
            if not source_is_latent:
                if source not in observed_predictors_seen:
                    observed_predictors_seen.add(source)
                    observed_predictors.append(source)
            if target_is_latent and source_is_latent:
                latent_edges.append((source, target))

            param_meta = parameter_lookup.get((relation_index, term_index))
            if param_meta is None:
                is_free, parameter_name, parameter_index, fixed_value, fallback_next_index = (
                    _fallback_parameter_meta(
                        term=term,
                        fallback_label_index=fallback_label_index,
                        next_parameter_index=fallback_next_index,
                    )
                )
            else:
                is_free, parameter_name, parameter_index, fixed_value = param_meta

            path_table.append(
                StructuralPath(
                    source=source,
                    target=target,
                    source_is_latent=source_is_latent,
                    target_is_latent=target_is_latent,
                    is_free=is_free,
                    parameter=parameter_name,
                    parameter_index=parameter_index,
                    fixed_value=fixed_value,
                    relation_index=relation_index,
                    term_index=term_index,
                )
            )

    exogenous_latent = [name for name in model_spec.latent_variables if name not in endogenous_latent_seen]
    beta_matrix = _build_beta_matrix(endogenous_latent, path_table)
    gamma_columns = exogenous_latent + observed_predictors
    gamma_matrix = _build_gamma_matrix(endogenous_latent, gamma_columns, path_table)
    warnings = _build_structural_warnings(
        path_table=path_table,
        endogenous_latent=endogenous_latent,
        latent_edges=latent_edges,
    )

    return StructuralDesign(
        path_table=tuple(path_table),
        endogenous_latent_variables=tuple(endogenous_latent),
        exogenous_latent_variables=tuple(exogenous_latent),
        observed_predictor_variables=tuple(observed_predictors),
        observed_endogenous_variables=tuple(observed_endogenous),
        beta_matrix=beta_matrix,
        gamma_matrix=gamma_matrix,
        warnings=tuple(warnings),
    )


def _build_beta_matrix(
    endogenous_latent: list[str],
    path_table: list[StructuralPath],
) -> pd.DataFrame:
    beta = pd.DataFrame(
        0.0,
        index=endogenous_latent,
        columns=endogenous_latent,
    )
    for path in path_table:
        if not (path.target_is_latent and path.source_is_latent):
            continue
        if path.target not in beta.index:
            continue
        if path.source not in beta.columns:
            continue
        if path.is_free:
            beta.loc[path.target, path.source] = np.nan
        else:
            beta.loc[path.target, path.source] = float(path.fixed_value or 0.0)
    return beta


def _build_gamma_matrix(
    endogenous_latent: list[str],
    gamma_columns: list[str],
    path_table: list[StructuralPath],
) -> pd.DataFrame:
    gamma = pd.DataFrame(
        0.0,
        index=endogenous_latent,
        columns=gamma_columns,
    )
    for path in path_table:
        if not path.target_is_latent:
            continue
        if path.target not in gamma.index:
            continue
        if path.source not in gamma.columns:
            continue
        if path.is_free:
            gamma.loc[path.target, path.source] = np.nan
        else:
            gamma.loc[path.target, path.source] = float(path.fixed_value or 0.0)
    return gamma


def _build_structural_warnings(
    *,
    path_table: list[StructuralPath],
    endogenous_latent: list[str],
    latent_edges: list[tuple[str, str]],
) -> list[str]:
    warnings: list[str] = []
    if not path_table:
        warnings.append("Structural design has no paths.")
    if not endogenous_latent:
        warnings.append("No endogenous latent variables found in structural design.")
    if _has_direct_self_loop(latent_edges):
        warnings.append("Latent self-loop detected in structural relations.")
    if _has_cycle(latent_edges):
        warnings.append("Latent cycle detected in structural relations.")
    return list(dict.fromkeys(warnings))


def _has_direct_self_loop(edges: list[tuple[str, str]]) -> bool:
    return any(source == target for source, target in edges)


def _has_cycle(edges: list[tuple[str, str]]) -> bool:
    adjacency: dict[str, set[str]] = {}
    nodes: set[str] = set()
    for source, target in edges:
        adjacency.setdefault(source, set()).add(target)
        nodes.add(source)
        nodes.add(target)

    visited: set[str] = set()
    active: set[str] = set()

    def dfs(node: str) -> bool:
        if node in active:
            return True
        if node in visited:
            return False
        visited.add(node)
        active.add(node)
        for neighbor in adjacency.get(node, ()):
            if dfs(neighbor):
                return True
        active.remove(node)
        return False

    return any(dfs(node) for node in nodes)


def _parameter_lookup(
    parameter_table: tuple[dict[str, Any], ...] | None,
) -> dict[tuple[int, int], tuple[bool, str | None, int | None, float | None]]:
    if parameter_table is None:
        return {}
    lookup: dict[tuple[int, int], tuple[bool, str | None, int | None, float | None]] = {}
    for row in parameter_table:
        key = (int(row["relation_index"]), int(row["term_index"]))
        fixed_raw = row["fixed_value"]
        lookup[key] = (
            bool(row["is_free"]),
            row["parameter"] if isinstance(row["parameter"], str) else None,
            int(row["parameter_index"]) if isinstance(row["parameter_index"], int) else None,
            float(fixed_raw) if isinstance(fixed_raw, (int, float)) else None,
        )
    return lookup


def _fallback_parameter_meta(
    *,
    term: RelationTerm,
    fallback_label_index: dict[str, int],
    next_parameter_index: int,
) -> tuple[bool, str | None, int | None, float | None, int]:
    fixed_value = term.coefficient
    if fixed_value is not None:
        return False, None, None, float(fixed_value), next_parameter_index

    if term.label is not None:
        index = fallback_label_index.get(term.label)
        if index is None:
            index = next_parameter_index
            fallback_label_index[term.label] = index
            next_parameter_index += 1
        return True, term.label, index, None, next_parameter_index

    index = next_parameter_index
    return True, f"p{index}", index, None, next_parameter_index + 1
