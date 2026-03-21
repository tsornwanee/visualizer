# Releasing

## Before the First PyPI Release

1. Confirm the distribution name you want to publish.
2. Check that the name is available on PyPI.
3. If needed, update `[project].name` in `pyproject.toml`.
4. Choose and add a license if you want to grant reuse rights publicly.

Note: as of 2026-03-21, the name `visualizer` is already in use on PyPI at <https://pypi.org/project/visualizer/>, so publishing under that exact distribution name will fail.

## One-Time PyPI Setup

1. Create the project on PyPI or TestPyPI.
2. Configure trusted publishing for this GitHub repository.
3. In GitHub, make sure the `pypi` environment exists if you want to keep the current workflow protections.
4. If you change the PyPI project name, update any PyPI URLs in:
   - `README.md`
   - `.github/workflows/python-publish.yml`

## Local Release Validation

Install the publish tools:

```bash
python -m pip install -e .[publish]
```

Build the artifacts:

```bash
python -m build
```

Validate the package metadata and long description:

```bash
python -m twine check dist/*
```

## Publishing

The repository already contains a GitHub Actions workflow at `.github/workflows/python-publish.yml` that publishes to PyPI when a GitHub Release is published.

Typical flow:

1. Update `CHANGELOG.md`.
2. Bump `version` in `pyproject.toml` and `src/visualizer/__init__.py`.
3. Re-run:

```bash
python -m compileall src
python -m build
python -m twine check dist/*
```

4. Commit and tag the release.
5. Publish a GitHub Release.
6. Let the GitHub Actions publish workflow upload to PyPI.

## Optional Preflight to TestPyPI

If you want a dry run before the first real release, publish to TestPyPI first and install from there in a fresh environment.
