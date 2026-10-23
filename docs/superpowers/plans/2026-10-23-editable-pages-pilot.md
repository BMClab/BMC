# Editable GitHub Pages pilot implementation plan

> **For agentic workers:** Use superpowers:executing-plans to implement this bounded pilot inline, with one fresh final review.

**Goal:** Publish one editable browser notebook on GitHub Pages and rebuild it on pushes, retaining all existing read-only notebooks.

**Architecture:** Keep Python sources in `notebooks/`; move saved HTML to `notebooks_read/`. An explicit list in `notebooks_edit/notebooks.json` permits only `notebooks/body_segment_parameters.py`. GitHub Actions builds the existing Cayman/Jekyll site, refreshes saved outputs and adds the WASM editor under `notebooks_edit/`. Generated editable assets belong to the Pages artifact rather than Git.

**Tech stack:** Locked uv/Python environment, native `marimo export html-wasm --mode edit`, official Jekyll and GitHub Pages actions.

**Spec:** The user's 2026-10-23 request for investigation and a single editable pilot, plus the existing notebook/content requirements. No full editable rollout is authorized yet.

## Constraints and decisions

- Preserve all 87 main guides, the course lecture, computations, PEP 723 metadata, author bylines and existing README omissions.
- Only the pilot's README title changes to editable mode; other titles use their relocated read-only pages. Both existing service badges remain.
- Names `notebooks_read/` and `notebooks_edit/` follow the user's proposal.
- Preserve Cayman and old published `/html/` URLs using generated redirects, including section fragments.
- Browser runtime is supplied by the exporter; no notebook imports or declares its runtime engine.
- Browser edits do not write back to GitHub; users download Python to retain a separate copy.
- Routine work stays on master under repository instructions and the established publication authorization.

## Review focus

1. Pages assets and workers must resolve under the `/BMC/` project prefix.
2. A failed export must stop deployment and preserve existing local output.
3. No unapproved second notebook may become an editable page.
4. Existing Jekyll root, read-only pages and old section links must remain usable.
5. A real browser edit followed by cell execution must change the computed result.

## Tasks

- [x] 1. Move read-only exports; update exporter, tests and navigation. Add a validated editable allowlist and a pilot-only local edit link. Verify meaningful failure cases, then all existing exporter tests and actual native exports.
- [x] 2. Implement `scripts/export_editable.py` to add allowlisted native WASM exports, mode-navigation links and legacy redirects to a built site. Test failure preservation, invalid/duplicate sources, correct paths and real native output. Add push/manual Actions workflow using the locked environment and official Pages actions, preserving the Jekyll root. Document local rebuild and deployment commands.
- [ ] 3. Run scoped Ruff, full existing tests, freshness checks and headless browser checks. Test package/data loading, rendered plots/tables, expandable answers, a real source edit and recomputation. Obtain one fresh read-only review, publish the pilot through Actions and verify actual deployed URLs before requesting review of the pilot.

## Evidence / rulings

- Baseline `d58c744`; clean tree. Existing Pages source is legacy/master/root.
- Official marimo docs and template reviewed through Context7 and primary sources. Installed marimo 0.25.1 exports editable mode with local browser assets; the prototype bundle is about 27 MB. Export alone does not demonstrate Python execution; actual browser execution is required.
- User explicitly requested the single-notebook test and corresponding README change. Proceeding with that bounded implementation without another approval gate; further editable notebooks await the pilot review.

- Task 1: tests first reproduced missing new paths, edit navigation and registry validation; all 14 exporter tests plus three subtests now pass. All 88 native read-only exports refreshed and current; 414 reading links and anchors resolve.
- Task 2: three builder tests failed before implementation, then pass including a real native WASM export, selective publication, failure preservation and missing-read-page checks.
- Full suite: 112 tests and 235 subtests pass; two expected synthetic C3D warnings. Scoped Ruff passes. No Python notebook source or dependency metadata changed.
- Browser execution: six tables and one plot load with both execution domains blocked. Actual CodeMirror editing and Ctrl+Enter change a scratch-cell result from 42 to 48.
- Layout regression: the native editor uses an absolute root container. Reusing read-only navigation CSS without a positioning parent left a blank viewport despite successful execution. A one-property prototype (`#root { position: relative; }`) restores visible content and anchors the editor below the mode bar; the final browser pass checks the unmodified generated page.

- Final unmodified generated page: editor visibly fills the viewport below navigation; actual edit 42→48, read-mode switch, old URL query/fragment preservation and mobile navigation fit all pass under `/BMC/`.
- Fresh independent review: Ready; no Critical, Important or Minor findings. Reviewer declined only live Actions/hosted behavior; those are the remaining publication checks below, not omitted scope.
- Final local suite after the layout fix: 112 tests and 235 subtests passed, with two expected C3D warnings. New/changed scripts and tests pass Ruff. README preserves all168servicebadges and has83savedtitlesplus1editabletitle.

- First Actions run 39909126211: all 88 read-only refreshes and the Cayman/Jekyll build passed. The editable step failed creating its staging directory because Docker-created `_site/` was root-owned. Deployment was correctly skipped and the previous live site retained. Added an ownership-normalization step before post-processing; the next actual Actions run is the regression check for this runner-only condition.
