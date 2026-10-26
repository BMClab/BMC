# Shared Notebook Resources Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement inline. Steps use checkbox syntax for tracking.

**Goal:** Replace embedded `repo_file` definitions with a shared import while
giving readers of individual downloaded notebooks clear recovery instructions.

**Architecture:** Keep `functions/notebook_resources.py` as the implementation.
Configure marimo's source path once. Import from a short notebook cell that
explains how to obtain a missing helper, while preserving unrelated import
errors. Saved pages offer an exact helper download beside the notebook download.

**Tech Stack:** Python, marimo, existing HTML/WASM exporters and unittest tests.

**Spec:** User approved the shared-module recommendation and confirmed that a
reader opening a single downloaded notebook may download the missing helper.

## Global Constraints

- Preserve earlier local changes, tutorial content, and scientific calculations.
- Keep individual `.py` downloads byte-identical to the notebook source.
- Do not introduce a published package, automatic code download, or dependency.
- Keep the resolver's local lookup, cache, download, and error behavior intact.
- Use `uv run`; refresh shared-input-dependent snapshots and their manifest.
- Commits and publication remain separate from this local implementation.

## Review Focus

- Missing helper errors name the file, URL, destination, and rerun action.
- Missing dependencies inside an existing helper retain their original errors.
- A downloaded helper beside a detached notebook is importable and resolves data.
- Notebook cells reference the imported resolver through explicit dependencies.
- Native WASM exports bundle the actual helper, including imports inside `try`.

## Task 1: Shared import and recovery

**Files:** `functions/notebook_resources.py`, `notebooks/*.py`, `pyproject.toml`,
`tests/test_notebook_resources.py`, `scripts/sync_notebook_resources.py`.
**Interface:** `from notebook_resources import repo_file` with unchanged function
signature and behavior.

- [x] Add and run failing tests for missing-helper guidance, sidecar recovery,
      and preservation of nested dependency errors.
- [x] Replace 38 used definitions with import cells; remove three unused ones.
- [x] Add explicit cell dependencies; configure marimo's `functions` search path.
- [x] Test the resolver directly and retire the obsolete embedding script.

## Task 2: Downloads, documentation, and verification

**Files:** `scripts/export_notebooks.py`, `tests/test_export_notebooks.py`,
`README.md`, `scripts/README.md`, `AGENTS.md`, `notebooks_html/`, audit evidence.
**Interface:** Existing exact notebook download plus an exact optional helper
download for sources importing `notebook_resources`.

- [x] Add a failing exporter test for the helper download and exact bytes.
- [x] Implement the optional helper download and update active documentation.
- [x] Verify real detached-notebook failure/recovery and WASM wheel payloads.
- [x] Export all 88 notebooks after shared-input changes; check saved artifacts
      and both download payloads.
- [x] Run the full test suite, scoped checks, freshness check, and read-only review.

## Completion record

All 38 shared imports and 76 dependent cells are verified. All 88 native saved
exports and 88 native editable exports succeeded. All notebook/helper downloads
match current source bytes. Detached failure/recovery, exact helper wheel
payloads, 124 tests with 277 subtests, scoped Ruff, lock consistency, and
freshness checks passed. Independent read-only review: PASS with no findings.
Browser interaction and live browser-runtime execution were unavailable.
