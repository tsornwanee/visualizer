# Contributing

## Setup

Create a virtual environment, then install the package in editable mode with the development extras:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

## Local Checks

Run the lightweight checks before opening a PR:

```bash
python -m compileall src
python -m build
python -m twine check dist/*
```

If you change notebook examples, make sure they still execute from a clean kernel.

## Repository Conventions

- Keep public examples aligned with the current API names.
- Prefer short, direct notebooks over repetitive galleries.
- When adding transitions, update:
  - `src/visualizer/__init__.py`
  - `README.md`
  - at least one demo notebook if the feature is user-facing

## Releases

Release steps live in [`RELEASING.md`](RELEASING.md).
