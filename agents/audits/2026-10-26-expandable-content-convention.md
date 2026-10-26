# Expandable content convention

Date: 2026-10-26. Scope: apply the user's approved convention across the 88
marimo notebooks in `notebooks/` and `courses/`.

## Convention

- Fixed written solutions, hints, and explanations use `/// details | Title`
  inside `mo.md`, including equations, lists, code examples, and hand-written
  Markdown tables.
- Computed answers, generated tables, plots, DataFrames, widgets, and combined
  display objects use `mo.accordion`.
- Solutions start collapsed. Saved HTML contains eagerly rendered content.
- Raw HTML `<details>` wrappers are avoided because they can bypass Markdown
  and LaTeX parsing.

The rules are recorded in `AGENTS.md`; `scripts/README.md` includes examples.

## Applied changes

| Measure | Result |
| --- | --- |
| Fixed Markdown accordions converted to details | 248 |
| Notebook sources and saved pages updated | 40 |
| Markdown details blocks after conversion | 511, across 87 notebooks |
| Accordions retained for computed Markdown | 24 |
| Accordions retained for display objects/layouts | 12 |
| Accordion retained for a generated section mapping | 1 |
| Raw HTML details wrappers in notebook sources | 0 |
| Dynamic Markdown details blocks needing the reverse conversion | 0 |

Only single-item accordions with literal Markdown and the existing collapsed
default were converted. All 37 remaining accordion call sites were inspected.
Their contents use runtime values or Python display objects, so they follow
the convention. Existing cells and their code-visibility settings were kept.
The earlier optimization table and body-segment-caption repairs were preserved.

## Verification

Before editing, saved the exact working-tree source and HTML baselines. For
each conversion, compared both the directly rendered Markdown and the actual
saved-session HTML with the earlier accordion body. After removing only the
container wrappers, all 248 saved bodies have identical parsed HTML tokens.
Titles and initially collapsed states also match.

The preserved bodies include **494 `marimo-tex` elements, 72 ordered lists,
298 list items, 30 inline code elements, and one link**. AST comparison confirms
that each affected source differs from its baseline only in the intended
accordion-to-details expressions. All existing details blocks in the affected
pages retain their content.

All 40 saved `Download .py` links decode to their current source bytes exactly.
The affected pages retain all **137 cells containing saved figures** and their
three computed/rich accordions. Figure presence was checked; re-executed plots
with unseeded inputs are not claimed to be byte-identical to earlier images.

- Existing exporter, selecting the 40 changed sources:
  **40 exported, 0 failed**.
- `uv run python scripts/export_notebooks.py --check`:
  **88/88 selected exports are current**.
- `uv run pytest -q`: **120 passed, 237 subtests passed**. Two warnings came from
  the existing C3D fixture with absent analog data/labels.
- `git diff --check`: **passed**.
- Rechecked the earlier optimization table repair: all eight first-column
  formulas still contain math-rendering elements and preserve their data.
- Independent read-only review: **PASS, no findings**. The reviewer independently
  checked all saved answer bodies, source transformations, retained accordion
  types, exact downloads, and the three retained saved accordions.

Browser appearance and interaction could not be checked because no browser was
available. Verification covers executed notebook exports and their saved HTML
structures; it does not claim screenshot or click testing.

The per-notebook counts and retained-accordion inventory are in
[the JSON evidence](2026-10-26-expandable-content-convention.json).
