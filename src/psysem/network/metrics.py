from __future__ import annotations

import numpy as np
import pandas as pd


def build_edge_table(
    adjacency: np.ndarray,
    *,
    item_names: tuple[str, ...],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    n_items = len(item_names)
    for i in range(n_items):
        for j in range(i + 1, n_items):
            weight = float(adjacency[i, j])
            if np.isclose(weight, 0.0):
                continue
            rows.append(
                {
                    "source": item_names[i],
                    "target": item_names[j],
                    "weight": weight,
                    "abs_weight": abs(weight),
                    "sign": "positive" if weight > 0 else "negative",
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=["source", "target", "weight", "abs_weight", "sign"],
        )
    edge_table = pd.DataFrame(rows)
    return edge_table.sort_values(
        by=["abs_weight", "source", "target"],
        ascending=[False, True, True],
        kind="stable",
    ).reset_index(drop=True)


def build_node_table(
    adjacency: np.ndarray,
    *,
    item_names: tuple[str, ...],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for idx, item_name in enumerate(item_names):
        weights = adjacency[idx, :]
        nonzero = weights[~np.isclose(weights, 0.0)]
        positive = nonzero[nonzero > 0]
        negative = nonzero[nonzero < 0]
        rows.append(
            {
                "node": item_name,
                "degree": int(nonzero.shape[0]),
                "strength": float(np.sum(np.abs(nonzero))),
                "expected_influence": float(np.sum(nonzero)),
                "positive_strength": float(np.sum(positive)),
                "negative_strength": float(np.sum(np.abs(negative))),
            }
        )
    node_table = pd.DataFrame(rows)
    return node_table.sort_values(
        by=["strength", "node"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)
