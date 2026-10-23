# Maintaining the static notebooks

Run commands from the repository root using the locked environment:

```bash
uv sync
uv run python scripts/export_notebooks.py
```

The updater executes every new or changed guide in `notebooks/` and every course
notebook in `courses/`. It writes saved outputs and source code under `html/`,
adds source-specific molab/marimo.app launch badges, and refreshes the static
catalog at `html/index.html`. Reading links to other guides point to their local
HTML pages. Explicit interactive launch links and external references are kept.

Update one notebook:

```bash
uv run python scripts/export_notebooks.py notebooks/reference_frame.py
```

Check freshness without executing notebooks or writing files:

```bash
uv run python scripts/export_notebooks.py --check
```

A stale or missing export makes this command exit with a nonzero status. The
manifest records source, shared-input and HTML hashes, so an edited or removed
HTML file is also detected. Notebook edits rebuild the affected page. Changes
to data, helpers, images, `pyproject.toml`, `uv.lock`, the exporter, Python or
marimo invalidate the selected exports. Run the command without notebook paths
after a shared-input change to refresh the entire collection.

Force a fresh execution, for example after an external dataset changes:

```bash
uv run python scripts/export_notebooks.py --force
uv run python scripts/export_notebooks.py notebooks/reference_frame.py --force
```

Remote data changes cannot be detected by local hashes. Unseeded random examples
may also produce different results when re-executed. The default is four parallel
processes with a 300-second limit per notebook; `--jobs 2` and `--timeout 600`
adjust these settings. A failed or timed-out export leaves the previous HTML
intact, records the failure, and returns a nonzero status. Fix the problem and
rerun the command before publishing.

Commit the changed notebook sources, generated HTML and `html/manifest.json`
together. Publishing to `master` updates the repository's existing GitHub Pages
site; the updater itself does not commit, push, or change hosting settings.
New notebook titles can be added to the README's existing bullet list manually.

## What the HTML contains

The updater uses `marimo export html`: these pages contain saved outputs rather
than a Python runtime. Expandable solutions work, while controls that require
Python cannot recompute results. The launch badges open the same notebook in
interactive mode. Reading the static pages needs neither execution service;
the marimo display assets from a CDN and linked external media still need an
Internet connection. This workflow does not create WebAssembly `html-wasm`
exports.

The export process selects marimo's headless Matplotlib backend so `plt.show()`
outputs are captured. A plot that is only created, or an accordion followed by
another expression, can still disappear. Finish such a cell with the intended
Figure/Axes or accordion object, as in the body-segment example. Review changed
plots and expandable answers after exporting; an execution-success status alone
does not prove that every intended display is visible.
