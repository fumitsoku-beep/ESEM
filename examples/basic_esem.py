from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from psysem import ESEMWorkflowConfig, SEMFitConfig, run_esem_workflow
except ModuleNotFoundError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
    from psysem import ESEMWorkflowConfig, SEMFitConfig, run_esem_workflow


DATA_PATH = Path(__file__).with_name("data") / "efa_demo_input.csv"


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing demo data at {DATA_PATH}. Run `python examples/basic_efa.py` first."
        )
    return pd.read_csv(DATA_PATH)


def main() -> None:
    data = load_data()
    items = list(data.columns)
    spec_payload = {
        "blocks": [
            {
                "name": "demo",
                "items": items,
                "n_factors": 2,
            }
        ],
        "estimator": "ML",
        "variable_types": {item: "continuous" for item in items},
        "rotation": {"method": "geomin", "oblique": True},
    }
    workflow = run_esem_workflow(
        data,
        spec_payload,
        ESEMWorkflowConfig(
            fit_config=SEMFitConfig(max_iter=200, restarts=1, random_seed=42),
        ),
    )
    candidate = workflow.best_candidate
    assert candidate.sem_result is not None

    print("Best candidate:", workflow.best_candidate_id)
    print("Comparison table:")
    print(workflow.comparison_table.to_string(index=False))

    print("\nSEM summary:")
    print(candidate.sem_result.summary())

    if candidate.block_efa_results:
        block_name, block_result = next(iter(candidate.block_efa_results.items()))
        print(f"\nBlock EFA preview ({block_name}):")
        print(block_result.loadings.round(3).to_string())


if __name__ == "__main__":
    main()
