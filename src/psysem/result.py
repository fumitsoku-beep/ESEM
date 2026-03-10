from dataclasses import dataclass, field


@dataclass
class SEMResult:
    converged: bool
    n_obs: int
    parameters: dict[str, float] = field(default_factory=dict)
    fit_indices: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "SEM Fit Summary",
            f"Converged: {self.converged}",
            f"N: {self.n_obs}",
        ]
        if self.fit_indices:
            lines.append("Fit indices:")
            for key, value in self.fit_indices.items():
                lines.append(f"  {key}: {value:.4f}")
        return "\n".join(lines)
