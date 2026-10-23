# Repository Instructions

This repository is a collection of biomechanics and motor-control tutorials — mostly pure
Python plus **marimo** notebooks. Prefer small, focused changes that preserve
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

- `notebooks/*.py` — **marimo** notebooks, the active source-of-truth set (87), with snake_case filenames. They are
  *reactive*: edit with `uv run marimo edit <file>`. Do **not** run them as plain scripts —
  `python file.py` only hits the `app.run()` guard, never the cell bodies.
- The Jupyter migration is complete: do not add `.ipynb` notebooks or Jupyter dependencies
  unless explicitly requested.
- Generated marimo session state and output caches under any `__marimo__/` directory are
  ignored and must not be committed. The three formerly committed session caches were retired
  during migration. Run cells to regenerate outputs; sources do not ship stored outputs.
- `nbs_read/` contains intentional static HTML snapshots with saved outputs and interactive
  launch badges. Update them with `uv run python scripts/export_notebooks.py` after editing
  a notebook; `--check` detects stale snapshots without executing or writing. See
  `scripts/README.md`. Commit the affected sources, HTML and manifest together.
- `nbs/notebooks.json` uses `{"include": "all"}` to discover every marimo notebook in
  `notebooks/` and `courses/`. GitHub Actions generates their editable WASM HTML/assets
  under `nbs/` in the deployed site; do not commit these generated assets. New notebooks
  are included automatically on push. `scripts/export_editable.py` adds them after the
  Cayman/Jekyll build, keeps rendered reading links local without altering Python source,
  and preserves earlier `/html/`, `/notebooks_read/` and `/notebooks_edit/` URLs by redirects.

## Verifying a notebook headlessly

marimo files don't run top-to-bottom, so flatten then run to confirm a notebook executes:

```bash
uv run marimo export script notebooks/<name>.py -o /tmp/flat.py
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
- Place `repo_file` after the opening title/authors, How to use and Python setup cells;
  it must not be the first registered cell.
- Keep figure captions outside raw HTML blocks so Markdown links and LaTeX are rendered.
  Use a blank line after the closed image/figure block, then an inline
  `<span class="figure-caption">…</span>` in `mo.md`; raw `<figcaption>` suppresses math parsing.
- Routine edits are committed directly to `master` (no PR workflow).
