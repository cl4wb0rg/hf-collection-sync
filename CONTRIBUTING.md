# Contributing

Contributions are welcome. Please follow these guidelines:

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install ruff mypy
```

## Code style

This project uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting,
and [mypy](https://mypy.readthedocs.io/) for type checking.

```bash
ruff check hf_sync.py
ruff format hf_sync.py
mypy hf_sync.py
```

CI runs both checks on every push and pull request.

## Pull requests

- Keep changes focused — one concern per PR
- Do not commit `.env`, `.sync_state.json`, log files, or model directories
- Update `README.md` if behaviour or configuration changes
