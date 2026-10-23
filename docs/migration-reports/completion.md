# marimo migration completion

Migration verification: 2026-10-22; final cleanup: 2026-10-23. Baseline comparison commit: `5366ffa7e12fc61b3c2410c173128e468a9fe524`. The user authorized final cleanup and publication to `origin/master` on 2026-10-23. Commit and push results are recorded in Git history and the final handoff; no new GitHub release or Zenodo publication was made.

## Result

The main collection contains **87 marimo guides in `notebooks/`**, with snake_case filenames and no parallel legacy directory. There are **zero `.ipynb` files in the repository working tree**. Jupyter, JupyterLab, ipykernel, ipywidgets, notebook extensions, RISE and nbstripout were removed from dependency metadata; no Jupyter packages remain in `uv.lock`. IPython remains for rich display helpers used by some marimo examples. `pyproject.toml` and `uv.lock` are authoritative; `requirements.txt` records the compatible direct dependencies.

Actual notebook contents were compared against the baseline; file dates were not used. Useful existing conversational rewrites were preserved. Every main guide includes author/laboratory attribution, How to use, an opening question or challenge, Python setup, guided examples, distributed Exercises with collapsed worked solutions, Checkpoint questions, Problems with collapsed final answers only, Go deeper, Internet video lectures and References. Equivalent Portuguese headings are retained where appropriate. All bylines credit both Marcos Duarte and Renato Naville Watanabe and render the same author/laboratory/university layout. References render real list items.

The requested detection notebooks, TimeNormalization, Untitled and duplicate Portuguese viscoelastic chapter were retired. The two Jupyter course labs were converted and retained in the main collection; the entire requested `courses/biomechanicsI_2022/` directory was deleted. Their default data loader retrieves the identical original table from the immutable baseline commit, with optional uploads for new experiments. `fig2_16.png` now resides in `images/` and its notebook reference was repaired.

README now uses **84 guide bullets**, with three links each: an editable GitHub Pages version on the title, a compact molab WebAssembly badge, and a compact marimo.app badge. Badge widths are 43 and 53 pixels respectively. The two Portuguese labs and the anatomical coordinate-system guide are omitted from the README index at the user's request; their notebooks are retained. Change detection and Time normalization of data each have one external repository link to detecta and tnorma. Colab, Jupyter and Classic Notebook launch links were removed from active navigation. The existing 2026 course marimo lecture remains in place with its reference links updated.

The [filename map](notebook-filenames.json) relates baseline marimo names to final filenames. The [inventory](../notebook-migration-inventory.csv) preserves baseline paths/hashes and adds final paths/hashes. Group reports intentionally retain the names used during comparison.

## Generated output-cache policy

Migration retired the three committed session caches for `MatrixFormalism.py`, `OrdinaryDifferentialEquation.py`, and `newtonLawForParticles.py` from `notebooks_marimo/__marimo__/session/`. The ignore rule is now `**/__marimo__/`, applying to generated session state throughout the repository. This is an explicit change from the earlier policy of maintaining those committed outputs: notebook sources no longer ship saved session caches, and users run cells to generate new results. Static cloud previews should not be assumed to include precomputed outputs. The separately maintained HTML snapshots now intentionally ship saved outputs in `html/`; they do not restore the old session-cache policy. README and AGENTS.md state this distinction.

Review follow-up removed **67 obsolete local session caches under old filenames** from `notebooks/__marimo__/session/`. The 87 caches whose names match current notebooks were retained as ignored local artifacts. These deletions remove generated state, not notebook source or experimental data.

## Scientific and runtime repairs

Per-guide details and primary references are in the group/review reports below. Material corrections include pendulum/gravity torque dimensions; stationary-action wording; vertical-jump work/force interpretation; full-design covariance for polynomial confidence/prediction intervals; digital prewarping for repeated-pass filter cutoff correction; independent complete-trial ensemble normalization; missing-value interpolation and input preservation; time-origin and effective-rate preservation in resampling and asynchronous EMG/IMU readers; force-plate numeric IDs, moment signs and calibration coordinates; C3D analog scaling, label counts and valid zero coordinates; TRC marker-group handling; Visual3D alignment on actual common timestamps; and nonuniform timestamps in the original course data.

The C3D examples use the pure-Python `c3d` package. Original examples and problem sets were retained, including Fourier-series problems and optional IEP group-comparison plots. Unavailable historical gymnastics media were replaced with an explicitly labeled, attributed explanatory schematic; it is not presented as a reconstruction of the original performance. Python/programming guides now introduce basic concepts through marimo and current local/cloud workflows.

A final real reactive export found a cell-local acceleration function escaping its marimo cell in `lagrangian_mechanics.py`. Nesting the pendulum default inside `euler_method` repaired the scope error; the complete notebook then passed reactive and flat execution. Dependency review also caught and corrected transitive pandas/SciPy omissions.

## File loading and cloud compatibility

The canonical resolver is `functions/notebook_resources.py`; its function is embedded in 41 notebooks, identical apart from the notebook-only `@app.function` decorator, so they can be opened as individual files. It searches beside the notebook and through parent directories, then fetches public assets into a temporary cache, preserving helper-directory structure. Base URLs isolate caches; downloads are written atomically; invalid repository-relative paths are rejected; parsing errors remain visible. External datasets and archived course inputs have explicit source URLs.

`urllib3` selects its browser transport automatically. **No notebook imports or declares pyodide.** Every main notebook has standalone PEP 723 metadata. This separates ordinary Python execution in local/molab servers from the browser runtime supplied by marimo.app or molab's `/wasm` route.

Cloud links and remote helper fallback use the repository's published `master` branch. They expose the new filenames/helpers once the migration commit reaches that branch. Publishing the source does not create a new GitHub release or Zenodo archive.

## Verification

Current source SHA-256, per-stage executed source SHA-256, and terminal outcomes are recorded in [execution-results.json](execution-results.json); rendered section/list results are in [render-results.json](render-results.json).

| Check | Result |
| --- | --- |
| Full local suite, `uv run pytest -q` | **115 tests and 235 subtests passed**; two expected warnings from synthetic C3D fixtures without analog data/labels |
| `uv lock --check` | Passed |
| `uv run marimo check --strict notebooks/*.py` | **87 passed** |
| `uv run marimo check --strict --select MW notebooks/*.py` | **87 passed** |
| Native `marimo export script`, reactive `marimo export html`, execution of exported scripts | **87/87 passed in all three stages** for the calculations; latest static collection reran reactive HTML for all 87 guides plus the course lecture |
| Real WebAssembly Python execution of exported scripts | **87/87 passed** for the earlier source hashes; later presentation changes were not rerun in WebAssembly |
| Rendered bylines, References lists and teaching-section audit | **87 passed**, no flagged guide |
| README/index paths and names | 84 guides, 168 badges; all 87 filenames mapped and snake_case |
| Static HTML freshness and browser checks | **88/88 current and rendered** with molab and marimo.app blocked; 414 local reading links resolve, including section anchors |
| `.zenodo.json` | Valid JSON; passes the official legacy deposit JSON Schema |

Native verification used Python 3.14.5 and marimo 0.25.1. Earlier script execution used Agg; the static HTML collection uses marimo's headless Matplotlib backend to capture `plt.show()` outputs. HTML export executed the actual reactive graph; running only the notebook's `app.run()` guard was not treated as verification. The earlier native 87-guide run took about 163 seconds with four workers.

The WebAssembly test used Pyodide 314.0.7/Python 3.14.2 in Node.js 22.22.0 with JavaScript Promise Integration enabled, and marimo 0.25.1. It installed the union of required packages, mounted repository assets as fixtures and executed each exported script. External BDS inputs were supplied as fixtures; the two archived course-data reads exercised real urllib3 HTTP in WebAssembly. Only `marimo.notebook_dir` was stubbed to locate the fixture directory. An initial run without Node's required stack-switching flag rejected those two HTTP calls; enabling the documented flag resolved both without notebook changes.

**Limits:** no browser surface was connected, so neither marimo.app nor molab was launched through its website UI. WebAssembly numerical execution, dependency review and native reactive execution provide substantial compatibility evidence, but do not verify service-specific package installation, browser CORS, cloud UI behavior, or every optional user-upload/hardware branch. External media availability can also change; previously unavailable/throttled resources are documented in the portability review. Full execution checks the supplied examples, not every possible experimental input or scientific assumption.

## Zenodo metadata

Added `.zenodo.json` after confirmation that BMC is already enabled in Zenodo. Both creators, ORCIDs, and the Lesson classification follow the [existing BMC record](https://zenodo.org/records/4599319). The record-level CC BY 4.0 license is retained, and the description explicitly preserves the separate MIT software license. Version, date and DOI are omitted so future release metadata can be assigned by Zenodo. Validation used the [official schema](https://github.com/zenodo/zenodo/blob/master/zenodo/modules/deposit/jsonschemas/deposits/records/legacyrecord.json) linked from [Zenodo's documentation](https://developers.zenodo.org/#add-metadata-to-your-github-repository-release). No remote record was changed.

## Comparison and independent review reports

- [Foundations](foundations.md) and [signal foundations](foundations-signals.md)
- [Kinematics and geometric examples](kinematics.md)
- [Dynamics and models](dynamics.md)
- [Portuguese course labs](course-labs.md)
- [Root analysis guides](root-guides-review.md) and [file readers](root-file-guides-review.md)
- [Portability, loader and media](portability-review.md)
- [Final independent integration review](final-integration-review.md)

## Review and deleted-helper follow-up — 2026-10-23

The user removed six superseded local helpers: `detect_cusum.py`, `detect_onset.py`, `detect_peaks.py`, `detect_seq.py`, `residual_analysis.py`, and `tnorm.py`. One obsolete acquisition remained: `read_trc_forces_files.py` fetched `functions/tnorm.py` despite no longer importing it. That fetch was removed. The notebook then passed native reactive export, strict checks, exported execution with HTTP disabled, and a focused WebAssembly execution. The full local suite still passes 94 tests and 113 subtests. A source/import audit finds no further calls to the deleted helpers; existing Python 2 utility files were checked by import-text inspection when Python 3 AST parsing was unavailable. Detection, temporal normalization, and cutoff selection use the installed detecta, tnorma, and optcutfreq packages where needed.

The resolver docstring now describes automatic browser transport without naming a browser-runtime package. Its 41 embedded copies were synchronized and compared with the canonical function after removing only the notebook decorator. The filtering lab's setup now frames the shared experiment around fair filter comparisons, delay, attenuation, endpoint effects and sampling intervals; the gravity lab retains its gravity-estimation framing. Only explanatory text changed in this review: computational AST and standalone dependency metadata remain unchanged in all 87 guides. Per-stage hashes distinguish the latest native reactive execution from the earlier flat-script and WebAssembly executions; all current sources passed reactive execution after dependency cleanup. The completion report now uses the project's lowercase marimo spelling and records the cache policy explicitly.

Final publication cleanup also retired the obsolete `postBuild` hook, which only installed/enabled Jupyter extensions. All 87 guides passed strict marimo checks and the rendered Markdown audit again after the prose changes; the final local suite passed 94 tests and 113 subtests. Generated session state, virtual environments and tool caches remain excluded from Git.

The final dependency cleanup removed unused direct dependencies `openai`, `pydantic-ai-slim`, and `pyversions` after confirming that no repository code imports them. The lockfile now resolves 79 packages, and the compatibility requirements were regenerated from `pyproject.toml`. The course lecture still uses `watermark`, which was retained.

The final reactive HTML execution passed **87/87 current guides** in the trimmed 79-package environment. Exact source hashes are recorded per verification stage; four whitespace-only source cleanups were checked against the final files. The computational AST and standalone metadata still match the previously verified calculations. Publication cleanup also removed 17 orphan bytecode caches and resolved all 12 staged whitespace diagnostics.

Presentation follow-up: `body_segment_parameters.py` now opens with its title and authors, followed by How to use and Python setup. The unchanged file resolver follows those opening cells with `hide_code=True`. Registered cell order and hidden-source configuration were checked directly; strict checks and native reactive HTML export passed. Function bodies and dependency metadata were preserved.

## Caption, image and index follow-up — 2026-10-23

Moved **127 captions in 34 guides** outside raw HTML figure blocks into Markdown-enabled caption spans, retaining centered italic styling, source links and image associations. This fixes all **18 math-bearing captions** and also allows Markdown links in captions to render. Four missing figure closing tags and two malformed polar-basis captions were repaired. The regression test reproduced 18 failures before the fix and now passes all 119 caption-cell subtests using marimo's real Markdown renderer. Source comparisons confirm that scientific computations and standalone dependency metadata are unchanged.

The broken Vitruvian Man thumbnail in `body_segment_parameters.py` now uses the credited, unmodified `images/vitruvian_man.jpg` with a plain HTML image tag linking to the repository (`?raw=1`), matching the other illustrations. The previous Python embedding passed native export but failed to display for the user, so it was removed. The file resolver remains in use for the data tables. The [Wikimedia image audit](wikimedia-image-audit.json) checked every external Wikipedia/Wikimedia image URL in the notebooks and courses. Two further broken thumbnails in `lagrangian_mechanics.py` were replaced with the original SVG URLs, including the file moved from English Wikipedia to Commons; author/license credits were added. All **22 current external URLs return HTTP 200 and valid image data**; SVGs were parsed and raster formats checked with Pillow, including animated GIFs. Availability is verified at the recorded time and may change.

Removed the requested anatomical coordinate-system entry from README while retaining `notebooks/sistema_coordenadas.py`. The index now contains 84 guide bullets and 168 compact badges. All 35 changed guides passed native reactive HTML execution; all 87 passed strict checks and the rendered teaching-section/byline/References audit. The full suite passes 95 tests and 232 subtests with the same two expected C3D-fixture warnings. Earlier flat-script and WebAssembly hashes are retained separately; those stages and the cloud website UI were not rerun for these presentation changes.

Image simplification follow-up: the GitHub Vitruvian Man URL and the suggested 250-pixel Wikimedia alternative both return HTTP 200 JPEGs. The GitHub response matches the repository image bytes. All 22 Wikimedia image URLs in 12 other guides were checked again; they already use direct HTML image tags and marimo retains each image in its rendered Markdown. No equivalent embedding code needs removal there. The revised body-segment guide passed strict checks and native reactive HTML export; the caption regression passes 119 subtests. This does not establish a browser/cloud UI result.

## Static HTML demonstration — 2026-10-23

At the pilot stage, only `body_segment_parameters.py` was exported for review in `html/`. Its README title linked to the rendered GitHub Pages file; other notebook title links remained unchanged. The export contained saved outputs and an added launch bar with matching molab and marimo.app badges. Both services were optional for reading; CDN display assets and linked images still required Internet access. GitHub Pages was already enabled on `master` at the repository root; no hosting settings were changed.

Browser inspection caught a missing plot: its cell created `hax` but did not display it. Adding `hax` as the final expression exposes the existing plot without changing calculations. The new native export and strict check passed. Chromium tests with both execution-service domains blocked showed six tables, one saved Matplotlib plot, rendered math, all four images loaded, and a working expandable solution. Desktop and mobile layouts were inspected; there was no mobile horizontal overflow and no browser page error. The existing caption regression passed one test and 119 subtests. Historical pilot evidence is preserved in commit `d05435c` and the `static_html_pilot` entry in execution-results.json; the current manifest covers the full collection below.

## Full static HTML collection — 2026-10-23

After approval of the demonstration, all **87 guides and the 2026 course lecture** were exported to `html/`. Each page includes saved text, tables and figures, an All guides link and two source-specific interactive launch badges. All 84 README titles now link to static GitHub Pages versions; the 168 compact interactive badges and the requested index omissions remain. Internal reading links stay within the HTML collection, including expandable solutions and section anchors. The audit verifies **414 local reading links**, with no missing page or unresolved section target. Ten outdated source/section links in seven guides were corrected using the actual current names and headings.

The reusable [exporter](../../scripts/export_notebooks.py) and [instructions](../../scripts/README.md) support full, single-notebook, incremental, forced and read-only freshness-check runs. It records source, shared-input and output hashes in [the manifest](../../nbs_read/manifest.json). Shared asset/helper/dependency/exporter changes invalidate cached outputs. Failed execution or an invalid export preserves the previous good page and returns failure. Remote input changes require `--force`. No production dependency was added.

The display audit repaired **16 cells in nine guides** that created plots without displaying them, or discarded an accordion by closing its figure as the final expression. These fixes expose existing figures and final-answer panels. Comparison against `d05435c` confirms that scientific computations and standalone dependency metadata are unchanged in all 87 guides. The collection includes **319 cells with saved figures**. Earlier flat-script and WebAssembly execution hashes are retained; this follow-up reran native reactive HTML, not the cloud execution UI or WebAssembly computations.

All **88 pages passed actual Chromium rendering checks** with molab and marimo.app blocked: headings rendered, source-specific launch targets matched, and no JavaScript page errors or execution-service requests occurred. Seven representative mobile layouts fit without horizontal overflow; all five repaired final-plot accordion panels opened and displayed their images. At initial publication, a stricter image-loading pass identified four unavailable historical images on `bmclab.pesquisa.ufabc.edu.br`, in the laboratory, projectile-motion and spatial/temporal guides. The user confirmed that host is down; the original image references were initially retained. These images have now been restored from repository assets, as described below. A transient GitHub image failure passed the targeted retry. Exact current source/output hashes and per-page results are recorded in [static-html-browser-results.json](static-html-browser-results.json). The full local suite passes **106 tests and 232 subtests**; strict marimo checks pass for all 88 notebooks. The updater's freshness check reports **88/88 current**, and an incremental rerun skips all 88.

These are **read-only snapshots** made with `marimo export html`. Expandable answers work, but Python-dependent controls cannot recompute results. They require no Python kernel or execution service; CDN display assets and external media still need Internet access, as the user accepted. The distinct `html-wasm` run/edit modes are not part of this export workflow.

The GitHub Pages configuration now selects the officially supported **Cayman** theme, with a site title and description and download buttons disabled. This updates Markdown pages; the exported notebooks retain their approved marimo layout.

## Laboratory media restoration — 2026-10-23

The user supplied copies of the three missing laboratory media files. Added
`images/bmclab_lab.png`, `images/jadel.gif` and `images/walking_animation.gif`,
preserving the supplied bytes exactly. The two GIFs retain all **54 and 214
frames**, respectively; every frame was decoded successfully. The fourth missing
image, the laboratory banner, now uses the existing `images/BMClab_header.png`.
The laboratory overview, projectile-motion and spatial/temporal guides use plain
repository image links in the established format. Website/article references
remain for provenance; image display no longer depends on that down host.

All **88 native HTML exports** were refreshed because shared-image changes
invalidate the collection. Browser checks passed for all 88 pages with both
execution services and the down laboratory host blocked, including all four
restored image elements. Before publication, repository image requests for the
three edited guides were fulfilled from the actual local image files; notebook
outputs and frontend assets were not mocked. Other linked media used the network.
There were no broken images or JavaScript page errors. The current browser report
records exact source/output hashes and the restoration details. All 414 internal
reading links and section anchors resolve.

Strict checks pass for the three changed guides, and the caption regression
passes **one test and 119 subtests**. Comparison confirms that computations and
standalone dependency metadata are unchanged. The inventory and native execution
hashes were updated; historical flat-script/WebAssembly results remain separate.

## Editable GitHub Pages pilot — 2026-10-23

At the user's request, saved HTML has moved from `html/` to `notebooks_read/`.
The body-segment guide is the only editable pilot: its README title points to
`notebooks_edit/body_segment_parameters.html`. The other 83 README title links
point to saved versions; the 168 existing service badges and index omissions
are preserved. Python notebook sources and their computations are unchanged.
Earlier full-static browser results identify their historical snapshot commit.

The new Pages workflow refreshes read-only outputs in the locked environment,
builds Cayman with the official Jekyll action, exports only the explicit approved
list in edit mode, then uploads and deploys the combined site. Editable HTML and
browser assets are generated in the Pages artifact rather than committed. Legacy
`/html/` reading URLs redirect to the saved pages, preserving query/section values.
The updater, failure cases and native editable export are covered by tests.

Local browser execution loads all six data tables and the plot. Actual keyboard
editing and cell execution change a computed result from 42 to 48, with both
execution-service domains blocked. Mode navigation and project-prefix assets
are included in the pilot verification. The full local suite passes 112 tests
and 235 subtests with the same two expected C3D fixture warnings. Publication
through Actions and live verification follow the prepared commit; their results
are recorded in the final handoff.

## Full editable rollout and short URLs — 2026-10-23

After approving the working pilot, the user requested every remaining editable
notebook and the names `nbs/` and `nbs_read/`. All **87 guides and the course lecture**
now have native editable exports. Read-only snapshots moved to `nbs_read/`, and all
84 README guide titles target editors under `nbs/`. The 168 service badges and
requested omissions are preserved. Both modes have direct navigation links.

The existing successful push workflow discovers every marimo notebook automatically
using `nbs/notebooks.json` with `{"include": "all"}`. It refreshes saved outputs,
builds Cayman, normalizes output ownership and generates the editable collection
before uploading/deploying the complete site. Generated editors/assets remain in
the deployment artifact. Old `/html/`, `/notebooks_read/` and `/notebooks_edit/`
pages redirect to the appropriate current mode, preserving queries and fragments.
Rendered reading links stay local without changing Python code or source links.

All **88 editors passed actual browser Python execution** with molab and marimo.app
blocked. Additional real-DOM checks cover lazy and shadow-root links, course paths,
code/source links, execution badges and section anchors. Current source/output
hashes, states and limits are in [editable-pages-results.json](editable-pages-results.json).
Native exports passed in both modes; the full suite passes **115 tests and 235
subtests**, with the two expected C3D fixture warnings. Independent review found
no material issue. Publication and live checks follow the prepared commit.

One actual optional-branch defect was corrected: IEP's `_plot` helper became
cell-local after native WASM serialization, so `cogve(..., show=True)` raised
NameError. Renaming its declaration/call to `plot_cogve` retains the arithmetic
and makes the browser example work. The failure and successful corrected plot
were both observed in the actual editor. No other notebook source or dependency
metadata changed. The native serializer may update generated-version/return
metadata in downloadable source; this is distinct from scientific code changes.


## Opening cells and molab links — 2026-10-23

Removed `/wasm` from all 84 molab links in README, as requested; marimo.app and
GitHub Pages remain the browser-execution choices. The original service-badge
counts and guide omissions are unchanged.

Audited all 88 notebooks for a resource helper as the first registered cell.
Moved `repo_file` below the opening guide/Python setup cells in seven guides:
`data_filtering`, `fourier_transform`, `kinematics_angular_2d`, `residual_analysis`,
`signal_basic_properties`, `sistema_coordenadas` and `vertical_jump`.
A full AST multiset comparison proves these are cell-order changes only:
helper bodies, decorators, Markdown and calculations are identical. No notebook
now starts with an app.function helper. All seven pass strict checks, native
HTML export and actual browser execution. Current hashes and browser records
were refreshed. The full suite still passes 115 tests and 235 subtests.
