# BMC — Biomechanics and Motor Control

Educational repository (Jupyter Book) of biomechanics and motor-control tutorials.
Work here is mostly pure-Python plus **marimo** and **Jupyter** notebooks.

## Environment
- Python ≥ 3.14, managed with **uv**. Run everything via `uv run …` (venv is `.venv/`).
- Lint/format with **Ruff** (default config): `uv run ruff format <path>`, `uv run ruff check --fix <path>`.

## Notebooks
- `notebooks_marimo/*.py` — **marimo** notebooks, the active source-of-truth set (91).
  They are *reactive*: edit with `uv run marimo edit <file>`. Do **not** run them as plain
  scripts — `python file.py` only hits the `app.run()` guard, never the cell bodies.
- `notebooks/*.ipynb` — the paired Jupyter versions.
- A few marimo notebooks carry committed output caches in
  `notebooks_marimo/__marimo__/session/<name>.py.json`, keyed by a per-cell `code_hash`.
  **After editing such a notebook, re-run all cells in `marimo edit` so the cache matches
  before committing** — otherwise the committed cache is stale. (A PostToolUse hook warns
  when you edit one of these.)

## Verifying a notebook headlessly
marimo files don't run top-to-bottom, so flatten then run to confirm a notebook executes:
```bash
uv run marimo export script notebooks_marimo/<name>.py -o /tmp/flat.py
MPLBACKEND=Agg uv run python /tmp/flat.py   # clean = exit 0, only Agg "cannot be shown" warnings
```

## Tests
- Run with either: `uv run pytest tests/` or `uv run python -m unittest discover -s tests`.
  Tests are written with stdlib **unittest** `TestCase`s; pytest discovers and runs them too.
- Tests import a marimo notebook as a module and exercise its `@app.function` definitions
  (see `tests/test_ordinary_differential_equation.py`).

## Layout
- `functions/*.py` — shared helper modules (signal processing, biomechanics math).
- `data/`, `images/`, `courses/` — assets and course material.

## Conventions
- Tutorial notebooks follow a prose-first teaching structure:
  title/author → "How to use" → guiding questions / challenges → worked examples → references.
- Routine edits are committed directly to `master` (no PR workflow).
