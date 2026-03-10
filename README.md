# psysem

`psysem` is a Python package skeleton for psychology-oriented structural equation modeling (SEM).

This repository currently provides:

- A minimal package API (`SEMModel`, `parse_model`, `SEMResult`)
- Placeholder extension APIs (`compute_basic_fit_indices`, markdown reporting, invariance stub)
- A smoke-testable fit workflow placeholder
- Test, lint, and type-check project wiring
- GitHub Actions CI template

## Quick start

```bash
pip install -e .[dev]
pytest
```

## Example

```python
import pandas as pd
from psysem import SEMModel

data = pd.DataFrame(
    {
        "x1": [1.0, 2.0, 3.0, 4.0],
        "x2": [1.2, 1.9, 3.2, 3.9],
        "y": [0.9, 2.1, 2.8, 4.2],
    }
)

model = SEMModel(
    """
    y ~ x1 + x2
    """
)
result = model.fit(data)
print(result.summary())
```

## Planned roadmap

- CFA/SEM parser aligned with familiar `lavaan`-style syntax
- Estimation backends (ML, robust variants, bootstrap)
- Fit diagnostics and psychology-focused reporting utilities
- Measurement invariance and mediation workflow helpers
