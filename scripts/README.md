# Maintaining the static notebooks

Run commands from the repository root using the locked environment:

```bash
uv sync
uv run python scripts/export_notebooks.py
```

The updater executes every new or changed guide in `notebooks/` and every course
notebook in `courses/`. It writes saved outputs and source code under `notebooks_html/`,
adds source-specific links for browser editing, molab and marimo.app, and
refreshes the static catalog at `notebooks_html/index.html`. Reading links to other guides point to their local
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

Commit the changed notebook sources, generated HTML and `notebooks_html/manifest.json`
together. Publishing to `master` updates the repository's existing GitHub Pages
site; the updater itself does not commit, push, or change hosting settings.
New notebook titles can be added to the README's existing bullet list manually.

## What the HTML contains

The updater uses `marimo export html`: these pages contain saved outputs rather
than a Python runtime. Expandable solutions work, while controls that require
Python cannot recompute results. The **Edit in browser**, **Open in molab** and **Open in marimo.app** links
open the same notebook in interactive mode. **Download .py** saves the exact Python
source included with the static page, including when opened locally. The header
also offers **Download helper** for notebooks that import `notebook_resources`.
It saves the exact shared `notebook_resources.py`; place it beside a downloaded
notebook and rerun the import cell if the notebook reports a missing helper.
The helper's error message includes a direct source URL as another way to obtain
the file. The header
centers all navigation links and uses the notebook's active theme colors and text font. Reading the static pages needs neither execution service;
the marimo display assets from a CDN and linked external media still need an
Internet connection. These saved pages are distinct from the editable WASM exports below.

The export process selects marimo's headless Matplotlib backend so `plt.show()`
outputs are captured. A plot that is only created, or an accordion followed by
another expression, can still disappear. Finish such a cell with the intended
Figure/Axes or accordion object, as in the body-segment example. Review changed
plots and expandable answers after exporting; an execution-success status alone
does not prove that every intended display is visible.

### Shared repository resources

`functions/notebook_resources.py` supplies `repo_file` through an ordinary Python
import. `[tool.marimo.runtime]` in `pyproject.toml` adds `functions` to marimo's
module search path. Keep the implementation in that file; do not synchronize
copies into notebooks. Its local lookup and cached remote downloads also work
when the helper is saved beside a notebook opened outside the repository.

The native directory WASM exporter bundles the imported module into a wheel.
Preserve `public/wheels` alongside the frontend assets when publishing the
editable export. Changing the shared helper invalidates the saved snapshots;
run the exporter without notebook arguments to refresh the collection.

### Expandable content

Use `/// details` for fixed written answers and explanations, keeping Markdown
and LaTeX together:

```python
mo.md(r"""
/// details | Solution

The spring force is $F=kx$.
///
""")
```

Use `mo.accordion` when the content is assembled in Python, such as a computed
answer, generated table, plot, or widget:

```python
mo.accordion(
    {"Results": mo.vstack([mo.md("Computed force:"), mo.as_html(figure)])},
    expanded=False,
)
```

Both forms should start collapsed for solutions. Keep their contents eagerly
rendered so the saved page contains the complete answer without a Python
backend. Avoid raw HTML `<details>` wrappers around Markdown or LaTeX. The
repository-wide rules are in [AGENTS.md](../AGENTS.md#conventions).

## Editable GitHub Pages and automatic publication

All 88 notebooks are exported in editable mode. The configuration
`scripts/notebooks.json` contains `{"include": "all"}`, so new marimo notebooks added under
`notebooks/` or `courses/` are discovered automatically. Every README guide title opens
its saved HTML page; the page header offers browser editing, molab and marimo.app.

`.github/workflows/pages.yml` runs on every push to `master` and supports manual
runs from the Actions tab. It installs the locked environment, refreshes saved
outputs, builds the existing Cayman site with Jekyll, then adds editable pages
using `marimo export html-wasm --mode edit` directly at the published site root
(`/BMC/<name>.html`). The workflow first restores runner
ownership of Jekyll’s container-created output so the Python exporter can write
its additional files. The complete site is uploaded and
deployed only after successful build steps. Generated editor assets live in the
Pages artifact, not Git. The exporter stages all native exports before merging
them into the site, preserving the Cayman home page, theme assets, course material
and saved pages. A notebook named `notebooks/index.py` is rejected because
`index.html` is reserved for the home page. Pages must use **GitHub Actions** as
its publishing source.

To build/test the editable part locally without installing Jekyll:

```bash
uv run python scripts/export_notebooks.py
uv run python -c "import shutil; shutil.copytree('notebooks_html', '_site/notebooks_html', dirs_exist_ok=True)"
uv run python scripts/export_editable.py --site _site
uv run python -m http.server --directory _site 8000
```

Open `http://localhost:8000/body_segment_parameters.html`.
This local shortcut contains the notebook pages; the workflow also renders the
full Cayman homepage. If no home page exists, the exporter creates a simple
catalog for local use and refreshes that generated catalog on later builds.
An existing Cayman or custom home page is preserved. The native WASM export
includes its frontend assets, so
serve the whole `_site/` folder over HTTP. The Python runtime, packages and
notebook data are downloaded in the browser; a first run can take longer.
No notebook needs to import the browser runtime.

Click **Run all** after opening the editor. Changes run locally in the browser
and do not update GitHub. Download the Python file to keep a modified copy.
The editor and saved pages have direct links to switch between modes. Older
published `/html/`, `/notebooks_read/` and `/nbs_read/` paths redirect to `notebooks_html/`;
`/notebooks_edit/` and `/nbs/` redirect to editors at the site root. Redirects
preserve query strings and section anchors. The `nbs/` directory is no longer
present in the source repository; only compatibility redirects use that path
in the deployment. A failed native notebook export preserves previous local
editor files and prevents a workflow deployment.

Links between rendered guides resolve to their editable Pages versions, including
links added when an answer panel opens. Original Python source and external source
references are retained. The home page at `/BMC/` links to the saved guides;
the saved catalog is at `/BMC/notebooks_html/`. Rebuilding an existing local site removes
retired editor HTML recorded in the previous export manifest, leaving unrelated
pages intact.
