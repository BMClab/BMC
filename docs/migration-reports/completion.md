# marimo migration completion

Migration verification: 2026-10-22; final cleanup: 2026-10-23. Baseline comparison commit: `5366ffa7e12fc61b3c2410c173128e468a9fe524`. The user authorized final cleanup and publication to `origin/master` on 2026-10-23. Commit and push results are recorded in Git history and the final handoff; no new GitHub release or Zenodo publication was made.

## Result

The main collection contains **87 marimo guides in `notebooks/`**, with snake_case filenames and no parallel legacy directory. There are **zero `.ipynb` files in the repository working tree**. Jupyter, JupyterLab, ipykernel, ipywidgets, notebook extensions, RISE and nbstripout were removed from dependency metadata; no Jupyter packages remain in `uv.lock`. IPython remains for rich display helpers used by some marimo examples. `pyproject.toml` and `uv.lock` are authoritative; `requirements.txt` records the compatible direct dependencies.

Actual notebook contents were compared against the baseline; file dates were not used. Useful existing conversational rewrites were preserved. Every main guide includes author/laboratory attribution, How to use, an opening question or challenge, Python setup, guided examples, distributed Exercises with collapsed worked solutions, Checkpoint questions, Problems with collapsed final answers only, Go deeper, Internet video lectures and References. Equivalent Portuguese headings are retained where appropriate. All bylines credit both Marcos Duarte and Renato Naville Watanabe and render the same author/laboratory/university layout. References render real list items.

The requested detection notebooks, TimeNormalization, Untitled and duplicate Portuguese viscoelastic chapter were retired. The two Jupyter course labs were converted and retained in the main collection; the entire requested `courses/biomechanicsI_2022/` directory was deleted. Their default data loader retrieves the identical original table from the immutable baseline commit, with optional uploads for new experiments. `fig2_16.png` now resides in `images/` and its notebook reference was repaired.

README now uses **85 guide bullets**, with three links each: a molab preview on the title, a compact molab WebAssembly badge, and a compact marimo.app badge. Badge widths are 43 and 53 pixels respectively. The two Portuguese labs are omitted from the README index at the user's request. Change detection and Time normalization of data each have one external repository link to detecta and tnorma. Colab, Jupyter and Classic Notebook launch links were removed from active navigation. The existing 2026 course marimo lecture remains in place with its reference links updated.

The [filename map](notebook-filenames.json) relates baseline marimo names to final filenames. The [inventory](../notebook-migration-inventory.csv) preserves baseline paths/hashes and adds final paths/hashes. Group reports intentionally retain the names used during comparison.

## Generated output-cache policy

Migration retired the three committed session caches for `MatrixFormalism.py`, `OrdinaryDifferentialEquation.py`, and `newtonLawForParticles.py` from `notebooks_marimo/__marimo__/session/`. The ignore rule is now `**/__marimo__/`, applying to generated session state throughout the repository. This is an explicit change from the earlier policy of maintaining those committed outputs: notebook sources no longer ship saved session outputs, and users run cells to generate plots and numerical results. Static cloud previews should not be assumed to include precomputed outputs. README and AGENTS.md now state this policy.

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
| Full local suite, `uv run pytest -q` | **94 tests and 113 subtests passed**; two expected warnings from synthetic C3D fixtures without analog data/labels |
| `uv lock --check` | Passed |
| `uv run marimo check --strict notebooks/*.py` | **87 passed** |
| `uv run marimo check --strict --select MW notebooks/*.py` | **87 passed** |
| Native `marimo export script`, reactive `marimo export html`, execution of exported scripts | **87/87 passed in all three stages** |
| Real WebAssembly Python execution of exported scripts | **87/87 passed** |
| Rendered bylines, References lists and teaching-section audit | **87 passed**, no flagged guide |
| README/index paths and names | 85 guides, 170 badges; all 87 filenames mapped and snake_case |
| `.zenodo.json` | Valid JSON; passes the official legacy deposit JSON Schema |

Native verification used Python 3.14.5 and marimo 0.25.1, with the Agg backend for headless plots. HTML export executed the actual reactive graph; running only the notebook's `app.run()` guard was not treated as verification. The full native 87-guide run took about 163 seconds with four workers.

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
