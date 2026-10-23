# marimo Collection Completion Implementation Plan

> **For agentic workers:** use the executing-plans workflow for integration and the dispatching-parallel-agents workflow for independent topic groups. Track completion below and in per-group reports. The user approved the pilot and explicitly requested full execution; no further planning approval is needed.

**Goal:** Complete the conversational marimo conversion and leave no Jupyter notebooks in the repository, with the main collection under `notebooks/`.

**Architecture:** Preserve and revise existing marimo notebooks after comparison with their Jupyter counterparts. Independent topic groups own disjoint files; the coordinator owns shared portability, verification, navigation, and the final directory move. Keep progress and verification evidence in `docs/migration-reports/`.

**Tech Stack:** Python >=3.14 locally via uv; marimo with standalone PEP 723 metadata and Pyodide-compatible dependencies for browser use.

**Spec:** `docs/notebook-migration-assessment.md`, including the user-approved shared teaching structure and final-answer-only Problems policy; approved pilot `notebooks/trigonometry_basics.py`.

## Global constraints

- Work in the current checkout, preserving the approved pilot and assessment. Repository convention is direct work on master. The user authorized final cleanup, commit and push on 2026-10-23; no new GitHub release or Zenodo publication is requested.
- Content ownership is recorded in `docs/migration-reports/ownership.json`. Do not edit another group's files or rename directories during concurrent work.
- Preserve authorship, language, original topics, equations, examples, media, references, and exercises. Correct errors with evidence. Do not overwrite useful previous rewrites.
- Every guide has How to use, a motivating challenge, guided examples and distributed Exercises, Checkpoint questions, Problems, Go deeper, Video lectures on the Internet, and References. Translate headings consistently in Portuguese material.
- Exercises have collapsed worked solutions; Problems have collapsed final answers only, without derivations. Keep answers out of visible outputs. Cover each substantial topic; do not substitute generic boilerplate for teaching.
- Execute actual cells, not just the app.run guard. Record failures and fix them; do not claim a notebook passes because it parses.
- Local, marimo.app, and molab are targets. Distinguish tested environments from static compatibility evidence. Keep real experimental data and failure visibility.

## Review focus

1. Imported standalone notebooks without sibling files: dependency declarations and portable asset loading.
2. Native-only scientific packages: preserve examples using browser-supported alternatives where possible; no silent skipped demonstrations.
3. Existing problems and equations: preserve them and give correct final answers, including figures needed to interpret a problem.
4. Notebook/module filename collisions and reactive variable reuse: marimo checks plus real execution.
5. Final rename: encoded and ordinary links, course assets, tests, generated state, and all remaining tracked Jupyter files.

## Tasks

### 1. Revise topic groups

Files: assigned notebooks and each group's own `docs/migration-reports/<group>.md` only.

- [x] Compare both versions and record discrepancies for every assigned notebook.
- [x] Preserve existing conversational material; rewrite remaining exposition and supply topic-specific practice/resources.
- [x] Fix notebook-local runtime errors, declare dependencies, and identify shared-helper fixes for integration.
- [x] Run scoped marimo checks and execution; record per-file results and unresolved external inputs.
- [x] Independently review the revised content and numerical answers in integration.

### 2. Portable execution and course notebooks

Files: shared asset-loader source/template and verification tooling as needed, `tests/`, two course marimo notebooks, integration-owned notebooks and helper fixes.

- [x] Make asset loading independent of cwd, support GitHub fallback and browser HTTP, and retain downloaded data integrity.
- [x] Fix self-import collisions and undeclared or native-only packages without dropping examples.
- [x] Convert both Portuguese course notebooks and preserve all original teaching/data.
- [x] Add meaningful tests for shared infrastructure and known runtime defects; run red/green checks where behavior changes.
- [x] Execute every notebook, capture terminal results, repair failures, and verify standalone/browser compatibility as far as available runtimes permit.

### 3. Consolidate and verify

Files: main and course notebook directories, README, conversion documentation, AGENTS, tests, ignore/filter config, environment metadata, migration inventory/status.

- [x] Reconcile README.ipynb contents with the landing page.
- [x] Remove old Jupyter files only after replacement coverage is established; move main marimo collection to `notebooks/` and preserve non-notebook assets.
- [x] Update all active paths, internal links, cloud launch URLs, notebook-importing tests, instructions, and obsolete notebook environment metadata.
- [x] Refresh or retire tracked generated caches in favor of reproducible source as part of the authorized consolidation.
- [x] Run the test suite, all-notebook structural checks, content/answer coverage audit, and execution from final paths.
- [x] Request independent final review, resolve findings, and record exact verification limits in the final report.

## Progress ledger

- Pilot and shared structure approved by user; full migration authorized.
- Worktree decision: work in place to preserve the approved uncommitted pilot and follow repository master workflow; disjoint file ownership prevents worker overlap.
- Cloud limitation at start: no connected browser surface. Seek executable Pyodide checks where useful and report actual website verification separately.

- User amendments: Python tutorials limited to marimo basics; all notebook title blocks credit Marcos Duarte and Renato Naville Watanabe, preserving other credits and leaving BMClab.py authorship unchanged; no explicit pyodide imports/dependencies; fig2_16.png moved to images; seven detect/Detect notebook pairs removed in favor of demotu/detecta.

- User amendment: compare the Portuguese viscoelastic chapter with the English version, retain any extra content in English, then remove `cap2_modelos_viscoelasticos.py`. Completed after verifying equivalent 38-cell content; see the dynamics report.

- Completed integration: 87 snake_case marimo guides in `notebooks/`; zero repository `.ipynb` files and no Jupyter packages in the environment metadata/lockfile. The two converted Portuguese lab guides remain, while the user-requested `courses/biomechanicsI_2022/` folder was deleted. The labs read the unchanged public source data from the baseline commit when no upload is supplied.
- README now has 85 guide bullets, each with a molab preview and two compact execution badges. The Portuguese lab section is omitted at the user's request. Detection and time normalization each have one external package-repository entry. Active notebook/course launch links use the final snake_case paths; legacy Colab/Jupyter/classic notebook links were removed.
- Added `.zenodo.json` after the user confirmed the existing Zenodo integration. Both creators/ORCIDs and the Lesson classification match the existing record; the description preserves the separate content/software licensing. Official schema validation passes; no release or publication was made.
- Verification: 94 tests and 113 subtests passed; all 87 strict and WebAssembly checks passed; all 87 native reactive HTML exports, flat exports and executions passed; all 87 exported computations passed in real WebAssembly Python. Actual cloud websites remain untested because no browser surface was connected. Final evidence and limits: `docs/migration-reports/completion.md` and `execution-results.json`.

- Publication cleanup (2026-10-23): retired the Jupyter-only postBuild hook, synchronized README/AGENTS cache instructions, removed 67 stale local session caches and three unused direct dependencies, and refreshed final verification/source hashes before the requested commit and push.
