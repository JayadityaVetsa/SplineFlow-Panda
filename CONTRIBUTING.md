# Contributing

Use Python 3.11 or newer and create the environment with `uv sync --all-extras`.
Before opening a pull request, run:

```powershell
ruff check src tests
pytest -m "not slow"
uv build
```

Keep robotics logic outside Streamlit, add tests for numerical behavior, and never
add benchmark values that are not derived from a saved rollout ledger. Large frame
tensors, downloaded models, checkpoints, and raw experiment directories must remain
outside Git history.
