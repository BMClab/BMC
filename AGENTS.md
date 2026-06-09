# Repository Instructions

This repository is a **Jupyter Book** of biomechanics and motor-control tutorials — mostly pure
Python plus **marimo** and **Jupyter** notebooks. Prefer small, focused changes that preserve
notebook content and examples.

## Environment

- Python ≥ 3.14, managed with **uv**. Run everything via `uv run …` so dependencies come from
  `pyproject.toml` and `uv.lock` (the venv is `.venv/`).
- Treat `pyproject.toml` and `uv.lock` as the source of truth for the Python environment.
  `requirements.txt` is legacy compatibility metadata.
- In restricted Codex sandboxes, use `env UV_CACHE_DIR=/tmp/uv-cache uv run …` if `uv` cannot
  write to its default cache.
- Lint/format with **Ruff** (default config): `uv run ruff format <path>`,
  `uv run ruff check --fix <path>`. Run ruff only on files you intentionally changed, or on a
  scoped subset — a broad repository lint currently includes legacy Python utilities that need
  separate modernization work. Do not broadly format or lint converted marimo notebooks unless
  explicitly asked.

## Notebooks

- `notebooks_marimo/*.py` — **marimo** notebooks, the active source-of-truth set (91). They are
  *reactive*: edit with `uv run marimo edit <file>`. Do **not** run them as plain scripts —
  `python file.py` only hits the `app.run()` guard, never the cell bodies.
- `notebooks/*.ipynb` — the paired classic Jupyter versions, legacy source material until
  migration is complete. The long-term direction is marimo-only notebook development: prefer
  marimo for new work and avoid adding new `.ipynb` notebooks unless explicitly asked. Do not
  remove Jupyter files or dependencies as part of unrelated work.
- A few marimo notebooks carry committed output caches in
  `notebooks_marimo/__marimo__/session/<name>.py.json`, keyed by a per-cell `code_hash`. After
  editing such a notebook, re-run all cells in `marimo edit` so the cache matches before
  committing — otherwise the committed cache is stale.
- Avoid committing generated marimo session state under `notebooks_marimo/__marimo__/`.
- For notebook output hygiene, install the local git filter with
  `uv run nbstripout --install --attributes .gitattributes`.

## Verifying a notebook headlessly

marimo files don't run top-to-bottom, so flatten then run to confirm a notebook executes:

```bash
uv run marimo export script notebooks_marimo/<name>.py -o /tmp/flat.py
MPLBACKEND=Agg uv run python /tmp/flat.py   # clean = exit 0, only Agg "cannot be shown" warnings
```

## Tests

- Run the focused suite with `uv run pytest -q` (or `uv run pytest tests/`, or
  `uv run python -m unittest discover -s tests`).
- Tests are written with stdlib **unittest** `TestCase`s; pytest discovers and runs them too.
- Tests import a marimo notebook as a module and exercise its `@app.function` definitions
  (see `tests/test_ordinary_differential_equation.py`).

## Layout

- `functions/*.py` — shared helper modules (signal processing, biomechanics math).
- `data/`, `images/`, `courses/` — assets and course material.

## Conventions

- Tutorial notebooks follow a prose-first teaching structure:
  title/author → "How to use" → guiding questions / challenges → worked examples → references.
- Routine edits are committed directly to `master` (no PR workflow).
