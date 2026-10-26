# Notebook table LaTeX rendering audit

Date: 2026-10-26. Scope: the reported optimization comparison table and the
same rendering failure in other notebook tables and their captions.

## Findings and changes

- `notebooks/optimization.py`: all eight comparison row labels were plain
  strings in a pandas index. Marimo's automatic DataFrame display does not pass
  these index values through its Markdown/math renderer. The first label also
  contained a Python tab escape (`\t` in `\text`); the other LaTeX commands had
  surplus backslashes. The labels now use raw strings with single LaTeX
  backslashes. The display promotes the index to a named first column and uses
  `mo.ui.table(..., format_mapping={'Quantity / cost function': mo.md})`.
- `notebooks/body_segment_parameters.py`: six adjacent table captions displayed
  literal `\text{...}` commands through `IPython.display.Latex`. These captions
  contain ordinary prose, so they now use `mo.md` without LaTeX wrappers.
- No other instance of raw math in displayed table values or headers was found.
  The remaining `Latex` calls in `confidence_prediction_intervals.py` were also
  inspected: their sigma expressions already contain rendered `marimo-tex`
  elements, and their percent-point-function output is ordinary text.

Both affected HTML snapshots and their manifest entries were regenerated with
the existing exporter. Numerical calculations and all table data values were
preserved. No exporter or environment changes were needed.

## Audit method and results

Inspected all 88 notebook sources under `notebooks/` and `courses/`, including
their DataFrame construction cells and legacy `Latex` display calls. Parsed the
saved sessions from all 88 corresponding HTML snapshots. For each saved table,
decoded HTML attributes and nested JSON rather than searching the escaped HTML
source for dollar signs.

| Saved output check | Result |
| --- | --- |
| Marimo tables inspected, including row labels and column names | 66 |
| Ordinary HTML/Markdown tables inspected | 21 |
| Raw math-bearing table values before the fix | 8, all in optimization |
| Formatted table cells containing `marimo-tex` after the fix | 8 |
| Remaining raw LaTeX/math in displayed tables | 0 |
| Remaining visible `\text{...}` outside math/code output | 0 |

The check examined formatted table data separately from the raw data retained
for filtering/downloads. Raw formula strings in that underlying data are
intentional; the displayed values contain Markdown MIME bundles with math
elements.

## Verification

- A temporary behavioral check executed the actual optimization comparison
  cell with deterministic numerical fixtures and decoded its display payload.
  Before the edit, it failed because all eight labels were plain strings. After
  the edit, it passed: eight math elements, correct `\text` and `\sum` commands,
  no tab/control-character corruption, and all five displayed columns present.
- `uv run python scripts/export_notebooks.py notebooks/optimization.py notebooks/body_segment_parameters.py`:
  **2 exported, 0 failed**. This executes both notebooks through marimo.
- Compared the seven affected saved tables with their versions at `HEAD`:
  **every numerical/data value was unchanged**.
- Decoded each affected page's `Download .py` link:
  **both downloads match the current source bytes exactly**.
- `uv run python scripts/export_notebooks.py --check`:
  **88/88 selected exports are current**.
- `uv run pytest -q`: **120 passed, 237 subtests passed**. Two warnings came from
  the existing C3D fixture with absent analog data/labels.
- `git diff --check`: **passed**.

Browser visual verification was unavailable: the browser connector returned
`No browser is available`. Verification establishes successful notebook
execution and correct saved math-rendering payloads; it does not claim a browser
screenshot inspection.
