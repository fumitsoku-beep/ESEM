from __future__ import annotations

from dataclasses import dataclass, field
import re

from .data import ESEMSpec

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_OPERATORS = ("=~", "~~", "~")


class ModelSyntaxError(ValueError):
    """Raised when model syntax parsing fails with location details."""


@dataclass(frozen=True)
class RelationTerm:
    """One right-hand side term with optional parameter modifier."""

    variable: str
    coefficient: float | None = None
    label: str | None = None


@dataclass(frozen=True)
class ModelRelation:
    """One parsed relation expression in SEM syntax."""

    operator: str
    lhs: str
    rhs: tuple[str, ...]
    terms: tuple[RelationTerm, ...]


@dataclass(frozen=True)
class ModelSpec:
    """Normalized SEM model specification used by the fitting pipeline."""

    source: str
    syntax: str
    relations: tuple[ModelRelation, ...]
    observed_variables: tuple[str, ...]
    latent_variables: tuple[str, ...]
    estimator: str | None = None
    block_names: tuple[str, ...] = ()
    warnings: tuple[str, ...] = field(default_factory=tuple)
    constraints: tuple[str, ...] = field(default_factory=tuple)


def parse_model(syntax: str) -> ModelSpec:
    """Parse SEM syntax into a normalized :class:`ModelSpec`."""
    if not isinstance(syntax, str):
        raise TypeError("`syntax` must be a string.")
    cleaned = syntax.strip()
    if not cleaned:
        raise ValueError("Model syntax cannot be empty.")

    statements = _split_statements(cleaned)
    relation_statements: list[tuple[int, str]] = []
    constraints: list[str] = []
    for index, statement in enumerate(statements, start=1):
        if "==" in statement:
            constraints.append(_parse_constraint(statement, statement_index=index))
        else:
            relation_statements.append((index, statement))
    if not relation_statements:
        raise ModelSyntaxError("No relation expressions found in syntax.")

    relations = _parse_relations(tuple(relation_statements))
    observed, latent = _infer_variable_roles(relations)
    return ModelSpec(
        source="syntax",
        syntax=cleaned,
        relations=relations,
        observed_variables=tuple(sorted(observed)),
        latent_variables=tuple(sorted(latent)),
        constraints=tuple(constraints),
    )


def model_spec_from_esem_spec(spec: ESEMSpec) -> ModelSpec:
    """Convert validated ESEMSpec into unified SEM ModelSpec."""
    if not isinstance(spec, ESEMSpec):
        raise TypeError("`spec` must be an ESEMSpec.")

    block_names = tuple(block.name for block in spec.blocks)
    latent_names: list[str] = []
    measurement_relations: list[ModelRelation] = []
    for block in spec.blocks:
        for factor_idx in range(1, block.n_factors + 1):
            latent_name = f"{block.name}_f{factor_idx}"
            latent_names.append(latent_name)
            measurement_relations.append(
                ModelRelation(
                    operator="=~",
                    lhs=latent_name,
                    rhs=tuple(block.items),
                    terms=tuple(RelationTerm(variable=item) for item in block.items),
                )
            )

    structural_relations = (
        _parse_relations(tuple((index, expr) for index, expr in enumerate(spec.structural, start=1)))
        if spec.structural
        else ()
    )
    _validate_structural_relations(
        structural_relations=structural_relations,
        observed_names=set(spec.variable_types.keys()),
        latent_names=set(latent_names),
    )

    all_relations = tuple(measurement_relations) + structural_relations
    syntax_lines = [f"{rel.lhs} {rel.operator} {' + '.join(rel.rhs)}" for rel in all_relations]
    observed = set(spec.variable_types.keys())
    for block in spec.blocks:
        observed.update(block.items)
    latent_set = set(latent_names)
    observed -= latent_set

    return ModelSpec(
        source="spec",
        syntax="\n".join(syntax_lines),
        relations=all_relations,
        observed_variables=tuple(sorted(observed)),
        latent_variables=tuple(sorted(latent_set)),
        estimator=spec.estimator.lower(),
        block_names=block_names,
    )


def _split_statements(syntax: str) -> tuple[str, ...]:
    statements = [segment.strip() for segment in re.split(r"[;\n]+", syntax) if segment.strip()]
    if not statements:
        raise ValueError("Model syntax cannot be empty.")
    return tuple(statements)


def _parse_relations(statements: tuple[tuple[int, str], ...]) -> tuple[ModelRelation, ...]:
    relations: list[ModelRelation] = []
    seen_paths: set[tuple[str, str, str]] = set()
    for statement_index, statement in statements:
        operator = _detect_operator(statement, statement_index=statement_index)
        lhs_raw, rhs_raw = statement.split(operator, 1)
        lhs = lhs_raw.strip()
        rhs_expr = rhs_raw.strip()
        _validate_identifier(lhs, context="left-hand side", statement_index=statement_index)
        if not rhs_expr:
            raise _syntax_error(
                f"Right-hand side is empty in expression `{statement}`.",
                statement_index=statement_index,
            )

        rhs_chunks = tuple(term.strip() for term in rhs_expr.split("+"))
        if any(not term for term in rhs_chunks):
            raise _syntax_error(
                f"Empty term detected on right-hand side in `{statement}`.",
                statement_index=statement_index,
            )
        parsed_terms = tuple(
            _parse_relation_term(chunk, statement_index=statement_index, term_index=term_index)
            for term_index, chunk in enumerate(rhs_chunks, start=1)
        )
        rhs_terms = tuple(term.variable for term in parsed_terms)

        if len(set(rhs_terms)) != len(rhs_terms):
            raise _syntax_error(
                f"Duplicate term detected in expression `{statement}`.",
                statement_index=statement_index,
            )

        for term in rhs_terms:
            key = (lhs, operator, term)
            if key in seen_paths:
                raise _syntax_error(
                    f"Duplicate path detected: `{lhs} {operator} {term}`.",
                    statement_index=statement_index,
                )
            seen_paths.add(key)

        relations.append(ModelRelation(operator=operator, lhs=lhs, rhs=rhs_terms, terms=parsed_terms))
    return tuple(relations)


def _parse_relation_term(chunk: str, *, statement_index: int, term_index: int) -> RelationTerm:
    if "*" not in chunk:
        _validate_identifier(
            chunk,
            context=f"right-hand side term #{term_index}",
            statement_index=statement_index,
            term_index=term_index,
        )
        return RelationTerm(variable=chunk)

    parts = [part.strip() for part in chunk.split("*")]
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise _syntax_error(
            f"Invalid term modifier syntax `{chunk}`.",
            statement_index=statement_index,
            term_index=term_index,
        )
    modifier, variable = parts
    _validate_identifier(
        variable,
        context=f"right-hand side term #{term_index}",
        statement_index=statement_index,
        term_index=term_index,
    )
    try:
        coefficient = float(modifier)
        return RelationTerm(variable=variable, coefficient=coefficient)
    except ValueError:
        _validate_identifier(
            modifier,
            context=f"parameter modifier in term #{term_index}",
            statement_index=statement_index,
            term_index=term_index,
        )
        return RelationTerm(variable=variable, label=modifier)


def _parse_constraint(statement: str, *, statement_index: int) -> str:
    left_raw, right_raw = statement.split("==", 1)
    left = left_raw.strip()
    right = right_raw.strip()
    if not left or not right:
        raise _syntax_error(
            f"Invalid constraint `{statement}`.",
            statement_index=statement_index,
        )
    _validate_identifier(left, context="constraint left-hand side", statement_index=statement_index)
    _validate_identifier(right, context="constraint right-hand side", statement_index=statement_index)
    return f"{left} == {right}"


def _detect_operator(statement: str, *, statement_index: int) -> str:
    for operator in _OPERATORS:
        if operator in statement:
            lhs_raw, rhs_raw = statement.split(operator, 1)
            # Disallow chaining multiple operators in one statement.
            if any(op in lhs_raw or op in rhs_raw for op in _OPERATORS):
                raise _syntax_error(
                    f"Multiple operators detected in expression `{statement}`.",
                    statement_index=statement_index,
                )
            return operator
    raise _syntax_error(
        f"Invalid expression `{statement}`. A relation must contain one of: "
        "`=~`, `~`, `~~`.",
        statement_index=statement_index,
    )


def _validate_identifier(
    token: str,
    *,
    context: str,
    statement_index: int,
    term_index: int | None = None,
) -> None:
    if not _IDENTIFIER_PATTERN.fullmatch(token):
        raise _syntax_error(
            f"Invalid variable token `{token}` in {context}.",
            statement_index=statement_index,
            term_index=term_index,
        )


def _syntax_error(
    message: str,
    *,
    statement_index: int,
    term_index: int | None = None,
) -> ModelSyntaxError:
    if term_index is None:
        return ModelSyntaxError(f"[statement #{statement_index}] {message}")
    return ModelSyntaxError(f"[statement #{statement_index}, term #{term_index}] {message}")


def _infer_variable_roles(relations: tuple[ModelRelation, ...]) -> tuple[set[str], set[str]]:
    latent = {relation.lhs for relation in relations if relation.operator == "=~"}
    observed: set[str] = set()
    for relation in relations:
        if relation.lhs not in latent:
            observed.add(relation.lhs)
        for term in relation.rhs:
            if term not in latent:
                observed.add(term)
    observed -= latent
    return observed, latent


def _validate_structural_relations(
    *,
    structural_relations: tuple[ModelRelation, ...],
    observed_names: set[str],
    latent_names: set[str],
) -> None:
    allowed = observed_names | latent_names
    for relation in structural_relations:
        if relation.operator != "~":
            raise ValueError(
                "`spec.structural` only supports regression expressions (`lhs ~ rhs`) "
                "in Phase 1."
            )
        terms = (relation.lhs, *relation.rhs)
        for token in terms:
            if token not in allowed:
                raise ValueError(
                    f"Unknown variable `{token}` in structural expression "
                    f"`{relation.lhs} ~ {' + '.join(relation.rhs)}`."
                )
