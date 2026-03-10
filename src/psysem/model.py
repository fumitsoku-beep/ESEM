from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """Parsed SEM model definition."""

    syntax: str


def parse_model(syntax: str) -> ModelSpec:
    """Validate and wrap model syntax.

    A full grammar parser will be added in later milestones.
    """
    cleaned = syntax.strip()
    if not cleaned:
        raise ValueError("Model syntax cannot be empty.")
    return ModelSpec(syntax=cleaned)
