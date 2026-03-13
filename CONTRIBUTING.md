# Contributing

## Development setup

```bash
pip install -e .[dev]
```

## Checks

```bash
ruff check .
mypy src
pytest
```

## Branching

- Keep `main` as the integrated and relatively stable branch.
- Create a topic branch from `main` for each workstream.
- Recommended branch patterns:
  - `feat/efa-*` for EFA methods, tests, and related implementation updates
  - `feat/esem-*` for ESEM workflow and estimator development
  - `docs/*` for README, MkDocs, templates, and documentation-only updates
- Keep each branch focused on one scope. Avoid mixing EFA, ESEM, and documentation changes in one branch unless they must ship together.
- Sync topic branches with `main` regularly to reduce merge conflicts.
- Merge back into `main` only after the relevant checks and tests pass.
- Add tests for behavioral changes.
- Open a pull request with a short scope summary.
