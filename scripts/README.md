# Maintaining the static notebooks

Run commands from the repository root using the locked environment:

```bash
uv sync
uv run python scripts/export_notebooks.py
```

The updater executes every new or changed guide in `notebooks/` and every course
notebook in `courses/`. It writes saved outputs and source code under `notebooks_read/`,
adds source-specific molab/marimo.app launch badges, and refreshes the static
catalog at `notebooks_read/index.html`. Reading links to other guides point to their local
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

Commit the changed notebook sources, generated HTML and `notebooks_read/manifest.json`
together. Publishing to `master` updates the repository's existing GitHub Pages
site; the updater itself does not commit, push, or change hosting settings.
New notebook titles can be added to the README's existing bullet list manually.

## What the HTML contains

The updater uses `marimo export html`: these pages contain saved outputs rather
than a Python runtime. Expandable solutions work, while controls that require
Python cannot recompute results. The launch badges open the same notebook in
interactive mode. Reading the static pages needs neither execution service;
the marimo display assets from a CDN and linked external media still need an
Internet connection. These saved pages are distinct from the editable WASM exports below.

The export process selects marimo's headless Matplotlib backend so `plt.show()`
outputs are captured. A plot that is only created, or an accordion followed by
another expression, can still disappear. Finish such a cell with the intended
Figure/Axes or accordion object, as in the body-segment example. Review changed
plots and expandable answers after exporting; an execution-success status alone
does not prove that every intended display is visible.

## Editable GitHub Pages pilot and automatic publication

Only `notebooks/body_segment_parameters.py` is currently approved for editing.
The explicit list is `notebooks_edit/notebooks.json`; the builder never exports
other notebooks automatically. The README title for this pilot opens the editor,
while other guide titles open saved versions.

`.github/workflows/pages.yml` runs on every push to `master` and supports manual
runs from the Actions tab. It installs the locked environment, refreshes saved
outputs, builds the existing Cayman site with Jekyll, then adds editable pages
using `marimo export html-wasm --mode edit`. The workflow first restores runner
ownership of Jekyll’s container-created output so the Python exporter can write
its additional files. The complete site is uploaded and
deployed only after successful build steps. Generated editor assets live in the
Pages artifact, not Git. Pages must use **GitHub Actions** as its publishing source.

To build/test the editable part locally without installing Jekyll:

```bash
uv run python scripts/export_notebooks.py
uv run python -c "import shutil; shutil.copytree('notebooks_read', '_site/notebooks_read', dirs_exist_ok=True)"
uv run python scripts/export_editable.py --site _site
uv run python -m http.server --directory _site 8000
```

Open `http://localhost:8000/notebooks_edit/body_segment_parameters.html`.
This local shortcut contains the notebook pages; the workflow also renders the
full Cayman homepage. The native WASM export includes its frontend assets, so
serve the whole `_site/` folder over HTTP. The Python runtime, packages and
notebook data are downloaded in the browser; a first run can take longer.
No notebook needs to import the browser runtime.

Click **Run all** after opening the editor. Changes run locally in the browser
and do not update GitHub. Download the Python file to keep a modified copy.
The editor and saved pilot have direct links to switch between modes. Older
published `/html/` paths redirect to `notebooks_read/` and preserve query strings
and section anchors. A failed editable build preserves previous local editor
files and prevents a workflow deployment.
