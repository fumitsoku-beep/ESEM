from __future__ import annotations

from .result import SEMResult


def to_markdown(result: SEMResult) -> str:
    """Serialize a fit result into a simple markdown report block."""
    lines = ["# SEM Result", "", f"- Converged: `{result.converged}`", f"- N: `{result.n_obs}`"]
    if result.fit_indices:
        lines.extend(["", "## Fit indices"])
        for key, value in result.fit_indices.items():
            lines.append(f"- {key}: `{value:.4f}`")
    return "\n".join(lines)
