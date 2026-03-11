from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from psysem import (
        EFADiagnosticsConfig,
        EFAEvaluationConfig,
        EFAWorkflowConfig,
        FactorSelectionConfig,
        run_efa_workflow,
    )
except ModuleNotFoundError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
    from psysem import (
        EFADiagnosticsConfig,
        EFAEvaluationConfig,
        EFAWorkflowConfig,
        FactorSelectionConfig,
        run_efa_workflow,
    )


DATA_PATH = Path(__file__).with_name("data") / "efa_demo_input.csv"


def generate_demo_data(n_obs: int = 400, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic two-factor dataset for EFA smoke testing."""
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(n_obs, 2))
    loadings = np.array(
        [
            [0.80, 0.10],
            [0.75, 0.15],
            [0.70, 0.05],
            [0.10, 0.80],
            [0.15, 0.75],
            [0.05, 0.70],
        ]
    )
    noise = rng.normal(scale=0.45, size=(n_obs, 6))
    observed = factors @ loadings.T + noise
    columns = [f"i{i}" for i in range(1, 7)]
    return pd.DataFrame(observed, columns=columns)


def load_or_create_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Use existing CSV input if present; otherwise generate and save one."""
    if path.exists():
        return pd.read_csv(path)

    path.parent.mkdir(parents=True, exist_ok=True)
    data = generate_demo_data()
    data.to_csv(path, index=False)
    return data


def main() -> None:
    data = load_or_create_data()
    items = tuple(data.columns)
    workflow = run_efa_workflow(
        data,
        EFAWorkflowConfig(
            items=items,
            diagnostics=EFADiagnosticsConfig(items=()),
            selection=FactorSelectionConfig(
                items=(),
                n_min=1,
                n_max=4,
                pa_iter=200,
                random_state=42,
            ),
            evaluation=EFAEvaluationConfig(),
            candidate_strategy="selection_union",
            include_consensus=True,
            extraction="paf",
            rotation="varimax",
        ),
    )
    diag = workflow.diagnostics
    selection = workflow.selection
    result = workflow.best_model

    print(f"Input data: {DATA_PATH}")
    print(f"Shape: {data.shape}")
    print(f"KMO total: {diag.kmo_total:.4f} ({diag.kmo_label})")
    print(
        "Bartlett: "
        f"chi2={diag.bartlett_chi2:.4f}, df={diag.bartlett_df}, p={diag.bartlett_p:.6f}"
    )
    print(f"Factor suggestions: {selection.suggestions_by_method}")
    print(f"Consensus n_factors: {selection.consensus_n_factors}")
    print(f"Best n_factors from workflow: {workflow.best_n_factors}")
    print(f"Best score: {workflow.best_evaluation.score:.4f}")
    print(f"Converged: {result.converged}")
    print(f"Iterations: {result.n_iter}")
    print("\nCandidate comparison:")
    print(workflow.comparison_table.to_string(index=False))
    print("\nExplained variance:")
    print(result.explained_variance.round(4).to_string())
    print("\nLoadings:")
    print(result.loadings.round(4).to_string())


if __name__ == "__main__":
    main()
