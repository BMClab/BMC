# Static HTML collection and update script

Approved basis: the published body-segment HTML demonstration and the user's
2026-10-23 instruction to export the remaining notebooks and provide an updater.

## Requirements

- Export all 87 guides and the existing 2026 course lecture into `html/`.
- Preserve the approved launch bar and link its two badges to the exact notebook.
- Keep reading links within the static collection; leave external references intact.
- Change all 84 README guide titles to their GitHub Pages HTML pages. Preserve
  the two interactive badges per guide and the requested omitted entries.
- Provide `scripts/export_notebooks.py` and documented full, single-notebook,
  incremental, forced and freshness-check commands.
- Execute notebooks in the locked environment. Never replace a working page
  with a failed export. Track source, shared-input and output hashes; changes to
  shared assets, helpers, dependencies or the exporter invalidate cached exports.
- Reading must not depend on molab or marimo.app. CDN display assets and external
  media remain permitted, as clarified by the user. Keep notebook computations.

## Execution

- [x] Build/test the updater: correct source-specific badges; internal links and
  anchors; missing/tampered output detection; selective/incremental updates;
  native execution failure preserves previous HTML and reports failure.
- [x] Export the collection. Use marimo's headless plotting backend so
  `plt.show()` is captured. Inspect plot-producing cells with no displayed
  figure and repair missing display expressions without changing calculations.
- [x] Update README/navigation and contributor instructions. Record source and
  execution evidence without changing historical flat/WebAssembly hashes.
- [x] Run scoped Ruff, relevant/full tests and strict notebook checks. Review the
  final change with a fresh read-only reviewer.
- [x] Complete the final image-loading browser pass. Four historical UFABC
  images are unavailable; the user confirmed that host is down. Retain source
  references and record media exceptions separately from page rendering.
- Publication and live Pages verification follow this prepared commit; results
  are recorded in Git history and the final handoff.

## Decisions

- Continue on `master`, following repository instructions and the established
  authorized publication workflow; no hosting settings need changing.
- Main guide output names are `html/<name>.html`; course notebooks preserve
  their repository-relative directories under `html/courses/`.
- No new runtime dependencies. The script uses the standard library and the
  repository's installed marimo. Browser verification is a separate check.
- Source freshness is conservative: a shared local input change rebuilds all
  selected pages; remote data changes require `--force`.

## Evidence ledger

- Preflight: clean tree at `d05435c`; 88 source notebooks discovered. Approved
  pilot already verifies responsive layout and expandable solutions. Native
  plotting backend test captures `plt.show()` media in exported console data.
- Updater: 11 focused tests pass, including a real native `plt.show()` capture.
  Regression tests first exposed and then verified fixes for source edits during
  execution and catalog titles accidentally taken from dependency comments.
- First batch: 88/88 native exports passed. Display audit found 16 missing figure
  or accordion outputs in nine guides; fixed only display expressions and
  re-exported those nine (79 others correctly skipped). Remaining no-figure
  candidate is IEP's explicitly disabled optional group-comparison branch.
- User steering: configured the officially supported Cayman theme in
  `_config.yml`, with a site title/description and downloads disabled. The
  notebook HTML retains the approved marimo layout.
- Static link audit found ten outdated source/section targets. Repaired them
  using the actual current guide names/headings; the next audit checks all links.
- Local suite: 106 tests and 232 subtests passed; two expected C3D fixture
  warnings. Strict check passed for all 88 notebooks. New script/tests pass Ruff.
- Final local browser run: 88/88 rendered with both execution domains blocked,
  zero page errors/service requests. Seven mobile checks have no
  horizontal overflow; five repaired final-plot panels expand and display images.
- All 414 local reading links and section anchors resolve. Presentation-normalized
  AST and PEP 723 comparisons against `d05435c` pass for all 87 main guides.
- Freshness: 88/88 current. Default incremental rerun executes none and skips 88.
  Current source/output hashes and browser evidence are recorded in the migration
  reports; earlier flat-script/WebAssembly hashes remain historical.
- Final independent review: Ready, with no Critical, Important or Minor finding.
  The reviewer independently checked all source/output hashes, exact-notebook
  launch URLs, README targets/badges, figure counts and failure preservation.
- Final rulings on review boundaries: retain the accepted network/display-only
  scope (offline use and Python recomputation remain outside these snapshots);
  retain documented `--force` for remote inputs (otherwise changed remote data
  can remain old); preserve historical execution hashes (they do not verify
  current presentation in WebAssembly); preserve inventory CRLF (default Git
  whitespace checks need the documented override); support one updater invocation
  at a time with internal `--jobs` concurrency (overlapping independent runs can
  race on the manifest). These are existing requirements/workflow decisions, not
  new implementation findings. No minor findings were deferred.
- Stricter media pass: wait for every image request, rather than relying on
  initial rendering. Four UFABC-hosted images time out; user confirmed the host
  is down. The transient GitHub image failure passes its retry. These media
  exceptions remain explicit in the browser evidence; notebook rendering and
  saved plots/tables pass.
