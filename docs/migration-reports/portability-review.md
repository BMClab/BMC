# Independent portability review

Date: 2026-10-22. Scope: the 87 active `notebooks_marimo/*.py` guides present during this review, `functions/notebook_resources.py`, and selected imported reader/helper modules. The review was initially read-only; the coordinator subsequently authorized only the media repairs listed below. This worker did not change dependency metadata, shared helpers, directory layout or source datasets during this audit. The coordinator applied the reported dependency/loader fixes concurrently.

## Findings and disposition

| Finding | Evidence and remediation | Final status in this review |
|---|---|---|
| Missing standalone dependency blocks | `pandas_data`, `read_trc`, `read_trc_forces_files` and `EnsembleAverage` initially lacked PEP 723 metadata. Their installed repository environment masked this omission. `read_trc` actually exercises its xarray branch. | Coordinator added metadata, including xarray where used. Refreshed AST-import versus metadata scan reports no missing direct package dependencies across 87 files. `mpl_toolkits` is supplied by Matplotlib; optional `dltx` has an embedded fallback. |
| Missing transitive reader dependency | `read_trc_forces_files` loads `functions/io_cortexmac.py`, which imports `detecta` at module import time. | Coordinator added `detecta>=0.0.5`; verified in the final source. |
| Undeclared legacy normalizer | `read_mac_ascii_files.write_v3dtxt` initially imported `tnorm` although no such helper was fetched and only tnorma was declared. | Reported to coordinator. Final function was subsequently rewritten to interpolate from actual timestamps and no longer imports `tnorm`; this audit does not claim independent validation of that separate algorithm rewrite. |
| Direct remote pandas reads outside this repository | `IP_Model` and `Stabilography` read the external demotu/datasets BDS file directly with pandas, bypassing the portable transport. | Coordinator routed both through `repo_file('BDS/data/BDS00013.txt', base_url='https://raw.githubusercontent.com/demotu/datasets/master/')`. The source returned HTTP 200, 515067 bytes and `Access-Control-Allow-Origin: *`. |
| Nine insecure image sources | Nine HTTP image sources remained in seven notebooks, including two unavailable UFABC hosts. | Repaired as detailed below; final HTML/Markdown media-source scan finds no HTTP or relative local image paths. |
| Unnecessary working-directory assumption | `ReadZebrisPressurePlatformASCIIfiles` still inserts `./../functions` into `sys.path`. Its reader is embedded and it does not import a helper through that path. | Low-priority cleanup only, not a demonstrated execution blocker. Remove the unused insertion in a later scoped cleanup. |

No unresolved high-confidence blocker for the default local execution paths was found after the reported fixes. This is a source/transport audit with scoped execution, not proof that every declared package installs in every browser runtime.

## Loader and browser assessment

The canonical helper uses `urllib3>=2.2`, without importing or declaring pyodide. Context7 resolved urllib3 and supplied the official [Emscripten transport documentation](https://urllib3.readthedocs.io/en/stable/reference/contrib/emscripten.html): browser support is selected automatically on Emscripten, beginning in urllib3 2.2, and uses browser Fetch or XMLHttpRequest. The notebook does not need to install or import pyodide to select it. Browser CORS and the runtime's supported network behavior still apply.

The helper looks beside the notebook and through its ancestors first, then downloads into a temporary cache keyed by the base URL. Preserving the repository-relative subdirectories keeps fetched helper siblings importable. It rejects absolute paths, `..` path components and backslashes. Downloading uses a temporary file followed by replacement; parsing happens in the caller, so a format error is not silently retried as a network fallback.

An isolated local probe of the actual canonical function passed all of these checks:

- An existing local asset was returned while network access was deliberately disabled for that branch.
- With `mo.notebook_dir()` returning `None`, real HTTPS downloads of `images/torque2.png` and `images/torque.png` matched the checked-in bytes exactly and shared the expected cache directory.
- A repeat read succeeded with network access disabled, proving cache reuse.
- Absolute, traversal and backslash paths raised `ValueError` before a fetch.

The probe used its own temporary directory and removed it afterward. It did not modify shared helpers. A flat export placed under `/tmp` can have a notebook location outside the checkout, so it may exercise the remote/cache branch instead of freshly edited local helpers. Collection integration should therefore distinguish those two branches; successful export alone does not prove that an edited helper was imported locally.

Context7 also supplied the official [WebAssembly constraints](https://pyodide.org/en/stable/usage/wasm-constraints.html). No notebook directly imports pyodide, ipywidgets, ipympl, Tk/Qt, ezc3d, multiprocessing or subprocess. The existing IPython imports provide rich display objects, mathematics or video frames, not widget communication channels. The C3D examples use the Python `c3d` reader. The AMTI helper includes an optional Windows NetForce named shared-memory branch (`filename='shfile'`, `mmap(..., tagname='shfile')`); that hardware/Windows branch is outside browser capability and is not called by the supplied file-reading example. This review does not infer failure of ordinary BSF-file reading merely from that unused branch.

## Media repairs and source preservation

| Notebook | Repair and verification |
|---|---|
| `KinematicsAngular2D.py` | Replaced the HTTP radian thumbnail with the original HTTPS Wikimedia SVG. The old thumbnail path returns HTTP 400 under the current thumbnail rules; the original SVG returned HTTP 200, `image/svg+xml`, 2077 bytes. |
| `ScalarVector.py` | Addition, subtraction and cross-product parallelogram images now use HTTPS. Their actual SVG responses returned HTTP 200 with the expected media type (25289, 9382 and 923 bytes). |
| `KineticsFundamentalConcepts.py` | Torque animation now uses its existing Wikimedia HTTPS URL. Initial verification returned HTTP 200; a later repeated batch request was throttled with HTTP 429. |
| `FourierSeries.py` | Fourier animation now uses its existing Wikimedia HTTPS URL. A direct GET returned HTTP 200, `image/gif`, 278023 bytes; a later repeated batch request was throttled. The [Commons source page](https://commons.wikimedia.org/wiki/File:Fourier_series_and_transform.gif) was also inspected. |
| `Matrix.py` | Sarrus illustration now uses HTTPS. Initial verification returned HTTP 200 and CORS `*`; a later repeated batch request was throttled. |
| `FreeBodyDiagram.py` | Replaced the unavailable `ebm.ufabc.edu.br/.../torque2.jpg` with the existing repository `images/torque2.png`. Visual inspection confirms the same 4 m / 2 m bracket, 600 N force and 40-degree angle. The existing `images/torque.png` contains the five original solution methods and now has an explicit link instead of a punctuation-only link. Both raw repository URLs returned HTTP 200 and the expected PNG bytes. |
| `CenterOfMassAndMomentOfInertia.py` | The original `daiane.gif` and alternative UFABC host paths were unavailable during verification. The [BMClab analysis article](https://bmclab.pesquisa.ufabc.edu.br/the-dos-santos-movement/) opened successfully earlier in the review, but a later retry timed out; its original animation links remained unavailable. Preserved Daiane dos Santos/BMClab attribution, the Dos Santos I description and original oblique-view context. Added an accessible self-contained SVG explaining parabolic center-of-mass flight and the relation between mass distribution and rotation. It is explicitly labeled a schematic, not a reconstruction or photograph of her performance. |

The Daiane text also now states the actual model assumptions: uniform gravity and negligible drag for a parabolic center-of-mass trajectory, and negligible external torque with a single/principal-axis model for `H=Iω`. A twisting gymnast requires a three-dimensional analysis. Thus the replacement retains both teaching points while avoiding the old unconditional “always a parabola” wording. The missing original animation remains a documented loss of historical media; no invented performance image was substituted.

The new SVG was rendered to PNG and visually inspected for clipping, label overlap and agreement between the marked center-of-mass positions and the plotted parabola. All labels fit. It uses a viewBox and automatic width and requires no external asset. The `http://www.w3.org/2000/svg` XML namespace is an identifier, not an insecure fetched image source.

No relative local Markdown or HTML image references remained when the authorized media pass started, so that portion required no additional changes. Existing repository HTTPS image references already work without a local asset directory. Final scanning also checked constant paths passed to `mo.image`/video/audio during the earlier audit and found no standalone local-file media requirement.

## Side effects, private data and execution evidence

The refreshed AST scan finds no absolute private file paths in executable string constants and no literal-path `open`/pandas/NumPy loading calls bypassing the portable input scheme. Matches such as `/content/` inside scholarly URLs were inspected and excluded from private-path findings.

Actual default writes were traced rather than treating every writer definition as an automatic write:

- `CodeStructure` creates its demonstration trials/results under a unique temporary directory. Its deliberately missing trial is removed only inside that generated directory.
- `read_delsys_cortex_file` writes uploaded bytes or its explicitly synthetic format example into a unique temporary directory; default reader calls do not save converted output.
- The MAC/TRC examples use empty optional output names for source reads. The round-trip demonstration writes a separate temporary copy and offers a download.
- `IEP` serializes CSV to in-memory bytes for `mo.download`, not to the source dataset.
- `repo_file` writes only its temporary download cache. Writer functions retain explicit output parameters for users who deliberately call them.

This review found no default cell that overwrites a source dataset. It did not execute arbitrary optional writer or uploaded-recording branches.

All seven media-edited notebooks passed `uv run marimo check --strict`. Each then executed its real cells successfully using `MPLBACKEND=Agg uv run marimo export html ...`, with seven exit codes of zero. This execution checks notebook code and rendered output construction; external `<img>` fetches were checked separately and can still be affected by rate limiting. The new schematic was additionally rendered and visually inspected. No execution on marimo.app or molab is claimed.

Temporary evidence is under `/tmp/bmc-dynamics/`: `portability-final.json` (87-file import/metadata/path/write inventory), `loader-results.json`, `media-results.json`, `media-check.json`, `media-check-final.json`, the seven `*.media.html`/`*.media.log` pairs, and `com-portability-schematic.svg`/`.png`. These are verification artifacts, not repository deliverables. The repository report is the durable record; the coordinator remains responsible for final collection-wide checks after its central edits and directory consolidation.
