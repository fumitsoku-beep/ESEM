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
- Add tests for behavioral changes.
- Open a pull request with a short scope summary.
