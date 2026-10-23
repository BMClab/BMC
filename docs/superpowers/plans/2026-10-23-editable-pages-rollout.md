# Complete editable Pages rollout

Approved scope: the user confirmed the editable pilot works, requested `nbs/` and
`nbs_read/`, and authorized editable generation for all remaining notebooks.

- [x] Rename both published directories, update builders/tests/navigation/docs,
  discover all notebooks on each build, and preserve redirects from previous URLs.
- [x] Export all 88 notebooks in both modes; verify native execution, real browser
  execution, editable reading links, mode switches, and a representative real edit.
- [ ] Run scoped Ruff and full tests, obtain one fresh read-only review, publish
  through the existing push workflow and verify the live site.

Decisions: retain `notebooks/` as Python source; `nbs_read/` contains committed saved
outputs, while Actions generates `nbs/` HTML/assets. Preserve README omissions and
both existing service badges. No scientific code or dependency changes are intended.
The approved full scope replaces the pilot allowlist with `{"include": "all"}` so
new marimo notebooks are published automatically. Reading links in the live editor
must resolve locally without altering the source code users edit/download.

Evidence starts at clean `7aa6082`. The existing Actions workflow already deploys
successfully and normalizes Jekyll container output ownership. Native sources are
87 main guides plus one course lecture. All older URLs remain redirect aliases.

- Red/green: tests first failed for the new names, full discovery and old URL families;
  20 focused tests plus three subtests now pass, including real native exports and
  course paths. The full suite passed 115 tests and 235 subtests.
- Generated 88 read-only and 88 editable exports. Shared runtime assets are deployed
  rather than committed. All 84 README titles target nbs; service badges stay intact.
- Real DOM checks pass for ordinary, cloud-preview, lazy-panel and shadow-root
  reading links, plus source links, explicit execution links, fragments and courses.
- Browser batch runs actual Python with both execution services blocked. Cell-status
  counts vary with frontend rendering, so completion uses sustained idle/error-free
  state rather than comparing counts to source function totals.
- IEP optional plotting reproduced a real NameError: native serialization made
  `_plot` cell-local. Renamed only its declaration/call to `plot_cogve`; original
  math is unchanged. The actual browser then executes show=True without cell errors.
- Independent review: no material findings; full browser pass and deployment are
  the remaining checks. No other notebook source or dependency changes intended.

- Final browser batch: 88/88 pass. IEP was rechecked after the helper fix, including show=True. Current source/output hashes are recorded in editable-pages-results.json. Full suite after the fix:115tests235subtests; publication/live checks follow the prepared commit.
