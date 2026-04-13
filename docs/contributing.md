# Contributing to QUEST

## Code style

We use `black` (formatting) and `isort` (import ordering). Run before committing:

```bash
black src/ scripts/ tests/
isort src/ scripts/ tests/
flake8 src/ scripts/ tests/
```

## Docstring convention

All public functions use Google-style docstrings:

```python
def my_function(arg1: str, arg2: int) -> float:
    """
    One-line summary.

    Longer description if needed.

    Args:
        arg1: Description of arg1.
        arg2: Description of arg2.

    Returns:
        Description of return value.

    Example:
        >>> my_function("hello", 3)
        9.0
    """
```

## Adding a new benchmark

1. Create `src/evaluation/benchmarks/your_benchmark.py` with a Dataset class
2. Add a config override file `configs/your_benchmark.yaml`
3. Add a test in `tests/` with a small sample annotation file
4. Update the results table in README.md

## Pull request checklist

- [ ] Tests pass: `pytest tests/ -v`
- [ ] Code formatted: `black` + `isort`
- [ ] New functions have docstrings
- [ ] Config changes have corresponding README updates
