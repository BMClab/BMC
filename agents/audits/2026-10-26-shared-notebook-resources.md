# Shared notebook resource resolver

Date: 2026-10-26. Scope: the approved shared-module migration and the user's
chosen recovery flow for a notebook downloaded without its helper.

## Implementation

`functions/notebook_resources.py` is the single resolver implementation.
Removed its 41 notebook definitions: 38 notebooks now import it through an
ordinary marimo cell, and three unused definitions were removed from
`force_plate_calibration.py`, `polynomial_fitting.py`, and `similarity.py`.
The obsolete embedding script was retired.

The import cells catch only a `ModuleNotFoundError` naming `notebook_resources`.
Their replacement error includes the direct helper source URL, says to save
`notebook_resources.py` in the same folder as the downloaded notebook, and tells
the reader to rerun the cell. Missing dependencies inside an existing helper
retain their original exceptions. No helper code is fetched or installed
automatically.

Marimo's `pythonpath` is configured once in `pyproject.toml`. The 76 cells that
consume the resolver now receive it explicitly through their cell arguments.
The resolver function's signature, docstring, local asset lookup, remote cache,
atomic downloads, and error behavior are unchanged.

The existing static exporter adds **Download helper** only to pages whose
source imports `notebook_resources`. The notebook and helper downloads contain
their exact source bytes, including when opened from a local saved HTML file.
Active instructions in `AGENTS.md`, `README.md`, and `scripts/README.md` describe
the import and recovery workflow.

## Verification

- New tests failed before implementation for the absent shared imports and
  helper download; the focused suite now passes:
  **25 tests, 45 subtests**.
- Full suite: **124 tests, 277 subtests passed**. Two warnings come from the
  existing C3D fixture with absent analog data/labels.
- Scoped Ruff checks: **passed**. Changed test files pass the formatter check.
- `uv lock --check`: **passed**; no dependency or lockfile change was needed.
- Native static export: **88 exported, 0 failed**.
- Freshness check: **88/88 selected exports are current**.
- Compared all 88 notebook ASTs with exact pre-task working-tree baselines:
  **only the intended resolver definitions/import cells and required cell
  argument changes differ**. Earlier local table and expandable-content changes
  are preserved.
- Compared every earlier embedded resolver body with the canonical module:
  **all 41 match**, excluding their notebook decorator.
- Decoded saved download links: **88 exact notebook downloads and 38 exact
  helper downloads**, with no helper offered on the other 50 pages.
- Saved figure presence: **all 322 figure-containing cells retained**.
- Executed a real detached `optimization.py` through native HTML export:
  **without the helper, it fails with the intended recovery instructions;
  placing the shared helper beside it restores full notebook execution**.
  Its data files were acquired through the existing resolver.
- Native editable WASM export of `optimization.py`: **passed**. Its generated
  wheel contains `notebook_resources.py` byte-identical to the shared source,
  confirming discovery of the import inside the `try` block.
- Complete native editable-site build: **88 editable notebooks exported
  successfully** through `scripts/export_editable.py` into a temporary site.
  The final merged site retains the exact helper wheel, referenced by the 38
  notebooks that need it.
- Independent read-only review: **PASS, no concrete issues found**. The reviewer
  independently checked focused tests, freshness, all download payloads, the
  generated helper wheel, path configuration, and notebook AST equivalence.
- `git diff --check`: **passed**.

Browser appearance, download clicks, and execution in the live browser Python
runtime were unavailable and are not claimed. Native notebook execution,
generated export payloads, and exact source downloads were checked.

Per-notebook saved-output checks and detached/editable export results are
recorded in [the JSON evidence](2026-10-26-shared-notebook-resources.json).
