# Repository Instructions

This repository is a Python notebook project for biomechanics and motor control teaching material. Prefer small, focused changes that preserve notebook content and examples.

## Environment

- Use `uv run ...` for Python commands so dependencies come from `pyproject.toml` and `uv.lock`.
- Treat `pyproject.toml` and `uv.lock` as the source of truth for the Python environment. `requirements.txt` is legacy compatibility metadata.
- In restricted Codex sandboxes, use `env UV_CACHE_DIR=/tmp/uv-cache uv run ...` if `uv` cannot write to its default cache.

## Notebook Layout

- Classic Jupyter notebooks live in `notebooks/`.
- Marimo notebooks live in `notebooks_marimo/` and are normal Python files.
- The long-term direction is marimo-only notebook development. Prefer marimo for new notebook work and avoid adding new `.ipynb` notebooks unless the user explicitly asks.
- Existing Jupyter notebooks are legacy source material until migration is complete; do not remove Jupyter files or dependencies as part of unrelated work.
- Do not broadly format or lint converted marimo notebooks unless the user explicitly asks for that cleanup.
- Avoid committing generated marimo session state under `notebooks_marimo/__marimo__/`.

## Checks

- Run the focused test suite with `uv run pytest -q`.
- Use `ruff` only on files you intentionally changed, or on a scoped subset. A broad repository lint currently includes legacy Python utilities that need separate modernization work.
- For notebook output hygiene, install the local git filter with `uv run nbstripout --install --attributes .gitattributes`.
