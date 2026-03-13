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

- Create a feature branch from `main`.
- Keep `main` for integrated, stable changes.
- Use topic branches for larger workstreams, for example:
	- `feat/efa-*` for EFA methods and testing changes
	- `docs/*` for README, MkDocs, templates, and documentation-only updates
- Add tests for behavioral changes.
- Open a pull request with a short scope summary.
