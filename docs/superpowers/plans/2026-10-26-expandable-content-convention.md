# Expandable Content Convention Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan inline. Steps use checkbox syntax for tracking.

**Goal:** Apply the user's approved distinction between written disclosures and
accordions containing Python-generated content throughout BMC.

**Architecture:** Keep fixed Markdown answers inside `mo.md` using `/// details`.
Retain `mo.accordion` for computed text, generated tables, and Python display
objects. Preserve existing cells, answer content, collapsed state, and outputs.

**Tech Stack:** Python, marimo, the existing static HTML exporter.

**Spec:** The user's approval in this conversation: use both methods and
standardize the previously recommended convention.

## Global Constraints

- Preserve tutorial content and all existing unrelated work.
- Use the locked environment through `uv run`.
- Refresh affected HTML snapshots and manifest entries together.
- Do not broadly format notebooks or add dependencies.
- Work locally in the existing checkout; commits and publication are separate.

## Review Focus

- Equations retain their LaTeX and rendered math elements.
- Lists, code blocks, links, and tables retain their rendered structure.
- Display objects and computed Markdown stay in accordions.
- Converted disclosures retain their titles and initially collapsed state.
- Source downloads remain byte-identical to their current notebook sources.

## Task 1: Document and apply the convention

**Files:** `AGENTS.md`, `scripts/README.md`, fixed-answer cells in 40 notebooks.
**Interfaces:** Existing `mo.md`, `mo.accordion`, and static export workflow.

- [x] Record source/output baselines and classify every accordion.
- [x] Convert only single-item accordions containing literal Markdown, with no
      special options beyond the existing collapsed default.
- [x] Verify each replacement's rendered body and title against the original.
- [x] Document fixed Markdown versus generated/display-object content.

## Task 2: Refresh and verify saved outputs

**Files:** Affected `notebooks_html/*.html`, `notebooks_html/manifest.json`,
`agents/audits/2026-10-26-expandable-content-convention.md`.
**Interfaces:** `scripts/export_notebooks.py`; outputs from Task 1.

- [x] Re-export affected notebooks with the existing exporter.
- [x] Check actual saved disclosure bodies, math, source downloads, and retained
      accordion content; inspect any differences before accepting them.
- [x] Run `uv run pytest -q`, exporter `--check`, and `git diff --check`.
- [x] Review the changes and record verified counts and any limitations.

## Completion record

Converted 248 fixed Markdown answers across 40 notebooks. Retained 37 accordions
for generated content or display objects; the collection now has 511 Markdown
details blocks. All 40 exports succeeded. Saved-output equivalence, exact source
downloads, 120 tests and 237 subtests, and the 88/88 freshness check passed.
Independent read-only review: PASS, no findings. Browser appearance/interaction
was unavailable and is not claimed. See
`agents/audits/2026-10-26-expandable-content-convention.md` for the evidence.
