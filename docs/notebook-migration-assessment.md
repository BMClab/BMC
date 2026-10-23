# marimo migration: assessment and review pilot

Date: 2026-10-22<br>
Baseline: `5366ffa7e12fc61b3c2410c173128e468a9fe524`<br>
Stage: migration, revision and consolidation completed. See the [completion report](migration-reports/completion.md) for the final 87-guide inventory, execution evidence and cloud-site verification limits. The assessment below records the original baseline and pilot decisions.

## Scope and inventory

The goal is a marimo-only repository with conversational teaching, adequate exercises with expandable solutions, and execution locally, at <https://marimo.app/>, and at <https://molab.marimo.io/notebooks>.

At the assessment baseline there were **92 Jupyter notebooks in `notebooks/` and 95 marimo notebooks in `notebooks_marimo/`**. Every Jupyter notebook in the former has a counterpart in the latter. Three pairs have different names:

| Jupyter stem | marimo stem |
| --- | --- |
| `MuscleModeling` | `muscle_modeling` |
| `MuscleSimulation` | `muscle_simulation` |
| `MusculoskeletaModelingSimulation` | `musculoskeletal_modeling_simulation` |

At the assessment baseline, the three additional marimo notebooks were `BasicMechanicalElementsMuscleModel.py`, `cap2_modelos_viscoelasticos.py`, and `ch2_viscoelastic_models.py`. The user subsequently authorized consolidating the two viscoelastic translations into `ch2_viscoelastic_models.py`: a cell-by-cell comparison found the same 38 cells, models, exercises, solutions, figures and references, with no additional Portuguese-only section or example. Retain the English chapter and `BasicMechanicalElementsMuscleModel.py`; the Portuguese duplicate is removed.

Three further Jupyter files are outside the main directory:

- `README.ipynb`: one Markdown cell; reconcile its navigation/content with `README.md` before retiring the duplicate notebook index.
- `courses/biomechanicsI_2022/lab_acel_gravidade.ipynb`: 28 cells.
- `courses/biomechanicsI_2022/lab_filtros.ipynb`: 32 cells.

The two course notebooks also need marimo versions, preserving their Portuguese teaching material and course context. Removing only the main Jupyter directory would not achieve the repository-wide goal.

## Content comparison

[The inventory](notebook-migration-inventory.csv) has 98 rows: 92 pairs, three marimo-only files, and three additional Jupyter files. It records source paths, SHA-256 hashes of baseline contents, Markdown word counts and similarity, teaching markers, expandable-block counts, and current stage. Dates were not used.

Jupyter Markdown was read from cell sources. marimo Markdown was extracted from literal arguments to `mo.md`, including static portions of formatted strings. Similarity compares lowercase word-token sequences with `difflib.SequenceMatcher(autojunk=False)`; it is a triage signal, not proof of mathematical or content equivalence. SVG text, dynamically assembled prose, and executable code are excluded from this metric.

English and Portuguese teaching prompts and expandable containers were inspected. The pilot and `ReferenceFrame.py` received closer reading, and representative guides were inspected to distinguish teaching revisions from simple format conversions. This is an inventory and content triage, **not an exhaustive scientific review or execution of every notebook**. Low similarity can mean additions, restructuring, or lost material; later section-by-section review must distinguish those cases.

Baseline categories, before the pilot edit:

| Category | Count | Follow-up |
| --- | ---: | --- |
| Conversational guide with expandable solutions | 8 | Preserve rewrite; check coverage, original content, solutions, and execution |
| Conversational guide without expandable solution containers | 15 | Preserve rewrite; complete practice and solutions |
| Other conversion/revision | 72 | Compare both versions; develop or complete guide and practice |
| Jupyter outside main directory | 3 | Resolve index and convert course notebooks |

The eight guides with expandable practice are `AngularVelocity3D`, `CoordinateSystem`, `KinematicChain`, `KinematicsOfRigidBody`, `ReferenceFrame`, `Transformation2D`, `Transformation3D`, and `sistema_coordenadas`. Counts include both `mo.accordion` and Markdown/HTML details blocks. Their presence does not establish that all exercises have solutions.

The fifteen guides needing expandable-solution work are `Biomechanics`, `DataFiltering`, `FourierTransform`, `KinematicsAngular2D`, `KinematicsParticle`, `muscle_modeling`, `muscle_simulation`, `musculoskeletal_modeling_simulation`, `OrdinaryDifferentialEquation`, `PathFrame`, `ProjectileMotion`, `ResidualAnalysis`, `ScalarVector`, `SignalBasicProperties`, and the marimo-only `BasicMechanicalElementsMuscleModel`.

The remaining 72 are not necessarily untouched conversions. Some have substantial code/prose changes or are short reference notebooks. Preserve useful changes rather than mechanically converting the old Jupyter file over them.

## Pilot: Basic trigonometry

Changed file: [TrigonometryBasics.py](../notebooks/trigonometry_basics.py).

Its original marimo version closely retained the Jupyter prose, without exercises or conversational structure. It is small enough to review as a whole and contains errors that a format-only conversion would miss.

The pilot retains similar triangles, radians, all five rows of common values, all five identities, author attribution, and the original reference resources. It adds a motivating forearm example, six distributed exercises with worked solutions, five checkpoint questions with an answer panel, three independent problems with final answers only, one reactive angle slider, and a numerical identity check. All ten solution/answer panels start collapsed. Exercises and problems apply existing concepts rather than introduce a new mathematical topic. Inline SVG diagrams replace the two remotely loaded illustrations.

Corrections:

1. `tan(30°)` is `1/sqrt(3)`, not `1*sqrt(3)`.
2. `tan(90°)` is undefined; infinity describes a one-sided limiting behavior, not its value at that angle.
3. The last term of `cos(alpha ± beta)` is `∓ sin(alpha) sin(beta)`, not `∓ sin(alpha) cos(beta)`.

Only marimo and Python's standard library are needed. PEP 723 metadata declares `marimo>=0.23.9` and Python `>=3.11` for standalone/cloud use. This does not change the repository's Python >=3.14 environment or lockfile. There are no working-directory assumptions, network reads, repository imports, or external image requests.

Review locally from the repository root:

```sh
uv run marimo edit notebooks_marimo/TrigonometryBasics.py
```

Review the tone, pacing, exercise difficulty, solution placement, and amount of explanation before applying the pattern more widely.

## Shared teaching structure

The user specified `notebooks_marimo/ReferenceFrame.py` as the model for the overall contents of every notebook. Use the same sequence of teaching elements across the collection, with topic-appropriate depth and equivalent headings in Portuguese notebooks:

| Element | Purpose |
| --- | --- |
| Title and author attribution | Identify the subject and preserve authorship |
| How to use this guide | Explain prerequisites, progression, and how to use exercises and answers |
| Opening challenge or motivating example | Ask for a prediction that the guide later revisits |
| Python setup | Explain the packages and data needed, where applicable |
| Guided topic sections | Conversational explanations, guiding questions, worked examples, and distributed Exercises |
| Checkpoint questions | Recall and interpretation questions before independent practice |
| Problems | Independent practice combining the notebook's existing concepts |
| Go deeper | Annotated optional reading and links to related notebooks |
| Video lectures on the Internet | Verified, relevant video resources; a subsection of Go deeper, as in ReferenceFrame |
| References | Sources used in the notebook, retaining original attribution |

Keep these as distinct teaching sections/cells, not merely a list of labels. Resources should have a short explanation of their relevance. Use marimo notebook targets for internal follow-up links and update those targets during the final directory rename. Resource lists must fit the subject; do not add unrelated material to fill a template.

**User-approved answer policy:** inline **Exercises** retain collapsed worked solutions. End-of-notebook **Problems** provide collapsed **final answers only, without derivations**. Conceptual problems receive a concise direct answer; numerical problems include the result and units. Checkpoint questions can have a separate collapsed answer panel. Do not expose answers in always-visible output cells beneath the prompts.

## Portable files and dependencies

The user-provided reference is `/home/marcos/github/demotu/optcutfreq/docs/optcutfreq_notebook.py`, especially its inline dependency block and local-file/raw-GitHub fallback. Retain that approach, with one refinement: prefer `mo.notebook_dir()` when seeking actual files. `mo.notebook_location()` can represent a web URL, whereas the playground can mount a GitHub repository into its filesystem. See [marimo's playground guide](https://docs.marimo.io/guides/publishing/playground/) and [WebAssembly guide](https://docs.marimo.io/guides/wasm/).

For a notebook in either `notebooks_marimo/` or the future `notebooks/`, a reader supporting paths and URLs can use this pattern:

```python
from pathlib import Path

_url = "https://raw.githubusercontent.com/BMClab/BMC/master/data/Pezzack.txt"
_directory = mo.notebook_dir()
_local = (
    Path(_directory).parent / "data" / "Pezzack.txt"
    if _directory is not None
    else None
)
_source = _local if _local is not None and _local.is_file() else _url
_data = pd.read_csv(_source, sep=r"\s+", skiprows=6, header=None)
```

This is a proposed pattern for later data-bearing notebooks, not an exercised feature of the trigonometry pilot. Adapt the relative path for course notebooks. Prefer an explicitly named repository asset over searching arbitrary working directories. Keep parsing outside fallback handling: a corrupt local file should report its parsing error, not silently load different remote data.

Acceptance criteria for files and packages:

- Test local data access with the working directory outside the repository, the fallback with only the notebook available, and GitHub-mounted files in marimo.app.
- Check filename case, URL encoding, and the actual remote resource. Verify browser HTTP/CORS behavior in the real runtime.
- Use a URL for compatible text readers, fetched bytes and an in-memory stream for compatible binary readers, and a temporary file only when the API requires it. Keep download/parsing failures visible; do not substitute synthetic experimental data.
- Declare each standalone notebook's packages in PEP 723 metadata. Verify Pyodide availability instead of assuming native wheels work in WebAssembly.
- Resolve imports from `functions/` deliberately. A notebook uploaded alone cannot assume sibling modules exist. Prefer an appropriate maintained package, or preserve attribution and behavior when embedding a small necessary helper.
- Apply the same policy to images and downloadable examples. Preserve or update relative references during the rename.
- For immutable releases, raw URLs can use a validated commit/tag instead of `master`; keep the data version consistent with the published notebook.

A shared loader could reduce repetition later, but requiring an uninstalled repository helper would reintroduce the standalone-cloud problem. Start with the small explicit pattern; add shared infrastructure only where it remains available in all three environments.

## Original stages and acceptance (completed)

1. **Initial stage: assessment and one pilot.** Review trigonometry. No other notebook conversion, Jupyter removal, directory rename, commit, or publication belonged to that initial stage. The user subsequently approved full migration and, on 2026-10-23, final cleanup and publication.
2. **After approval: revise in topic groups.** Start with foundational material, then mechanics, signals/statistics, and specialized readers/simulations. Preserve existing conversational revisions. Compare original sections, equations, examples, figures, references, and exercises before editing each notebook.
3. **Teaching acceptance.** Apply the shared structure above, including Checkpoint questions, Problems, Go deeper, Video lectures on the Internet, and References. Each substantial topic needs meaningful practice: conceptual prediction, calculation/code where applicable, and interpretation/error-checking. Choose counts by coverage, not a fixed quota. Preserve original exercises/problems and provide the appropriate collapsed response: worked solutions for Exercises, final answers without derivations for Problems.
4. **Execution acceptance.** Check marimo structure and headless execution, inspect mathematics/figures, exercise controls, and verify local, marimo.app, and molab operation. Record results per notebook. Missing data, incompatible dependencies, or cloud failures remain explicit unfinished items.
5. **Final consolidation.** After content/runtime acceptance, retire the Jupyter sources and rename `notebooks_marimo/` to `notebooks/`. Resolve the three Jupyter files elsewhere. Update README/navigation, course links, encoded cloud launch URLs, notebook-importing tests, `AGENTS.md`, `.gitignore`, output filters, and obsolete Jupyter-specific environment metadata. Recheck paths and launches after renaming.

The rename comes last: README primarily targets `.ipynb`/nbviewer, four test modules import `notebooks_marimo`, and notebook links contain both old paths and URL-encoded paths. Merely renaming directories would break those entry points.

Tracked session caches exist for `MatrixFormalism`, `OrdinaryDifferentialEquation`, and `newtonLawForParticles`. Follow the repository's refresh instructions when revising those notebooks; avoid adding new generated session state. Resolve final cache handling during consolidation.

## Verification record

Observed environment: Python 3.14.5, marimo 0.25.1.

A read-only check of the 95 main marimo files reported 15 existing diagnostics in 11 files: 11 formatting diagnostics and four runtime self-import warnings in `read_c3d.py`, `read_trc.py`, `rep_missing.py`, and `resample.py`. These are assessment findings, not repairs made by this pilot. No broad execution or broad formatting was performed.

Pilot results:

| Check | Result |
| --- | --- |
| `uv run marimo check --strict notebooks_marimo/TrigonometryBasics.py` | Passed |
| `uv run marimo check --strict --select MW notebooks_marimo/TrigonometryBasics.py` | Passed WebAssembly compatibility rules |
| `uv run ruff check --ignore PLR1711,B018 notebooks_marimo/TrigonometryBasics.py` | Passed; exemptions retain marimo-generated returns and the final expression displaying the slider |
| Export to a flat script, then `MPLBACKEND=Agg uv run python /tmp/bmc-trigonometry-pilot.py` | Passed; numerical outputs agree with exact expressions |
| `uv run marimo export html notebooks_marimo/TrigonometryBasics.py -o /tmp/bmc-trigonometry-pilot.html` | Passed; all cells executed |
| Programmatic marimo graph execution with slider definitions at 0, 30, 60, and 90 degrees | Passed after shared-structure revision: expected values, ten closed solution/answer containers, two SVG outputs, and all shared sections at each setting |
| Independent arithmetic checks of exercise answers, all three problem answers, and all five identities | Passed |
| Parse and render both SVGs with Inkscape; inspect PNGs | Passed; labels and geometry legible |
| `uv run marimo export html-wasm notebooks_marimo/TrigonometryBasics.py -o /tmp/bmc-trigonometry-pilot-wasm.html --mode edit --single-file` | Export passed; browser execution not verified |
| `git diff --check` | Passed |
| Inventory coverage and baseline hashes | All 98 rows covered; all 190 source hashes match the stated commit |
| Independent review of pilot and assessment | No remaining findings; focused recheck confirms the shared structure, correct Problem answers without derivations, checkpoint answers, and pilot-only scope |

The default Ruff run flags generated returns (`PLR1711`) and the slider display expression (`B018`); the scoped command above excludes those two marimo conventions. No lint configuration was changed. Parameter-override checks exercise the dependency graph, not browser interaction. The full repository test suite was not run: shared helpers and existing tested notebook functions were not changed.

Website execution and browser visual inspection remain unverified because this session has no connected browser. Static WebAssembly checks and an export do not substitute for launching the notebook on both requested websites. The local preview files are temporary review artifacts, not committed notebook state.
