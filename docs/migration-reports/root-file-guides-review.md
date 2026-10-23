# Independent review: applied motion-file guides

Reviewed 2026-10-22. Scope: `OpenC3Dfile`, `read_c3d`, `pandas_data`, `read_trc`, `read_trc_forces_files`, `read_mac_ascii_files`, and `functions/c3d_data.py`. Compared the six paired Jupyter sources with current marimo content. Global author/reference/loader normalization was coordinated by root and excluded from this review. Root subsequently authorized the reader repairs below; this report distinguishes repaired findings from the remaining content item.

## Remaining content-preservation item

- **P2 — Retain the original walking animation figure.** `notebooks_marimo/OpenC3Dfile.py:334` explains the replacement of the old separate-window animation with a slider, but the original notebook's embedded `images/walking.gif` is absent from the new guide. The existing 2.1 MB repository GIF is still available. Preserve it as an optional/collapsed historical illustration next to the slider or under Go deeper. Automatic animation-file writes need not return. Root was notified; no notebook edits for this item were made by this reviewer.

## Reproduced findings and authorized repairs

Locations below refer to the corrected sources before the planned directory consolidation.

| Finding | Reproduction / consequence | Correction and locations |
|---|---|---|
| **C3D labels incorrectly determined data width** | A real synthetic C3D containing two stored points and either one or three `POINT:LABELS` entries raised a DataFrame shape error. A point-only file raised `AttributeError` while retrieving absent analog labels. This is permitted format variation, not necessarily corrupt data. | `functions/c3d_data.py:16` normalizes labels to `POINT:USED`/`ANALOG:USED`, collects available LABELS2/etc., fills missing labels, truncates unused labels and avoids suffix collisions. Same implementation embedded at `OpenC3Dfile.py:115` and `read_c3d.py:115`. Channels with no analog samples no longer require analog labels. |
| **Dropping individual TRC coordinates corrupted marker groups** | With marker A's Y column entirely missing, B valid and C empty, `dropna=True` removed A's Y as well as C. MultiIndex/xarray output failed; flat output had five coordinate columns but reported only one marker. | Whole XYZ groups are removed only when every coordinate is missing. Partial missingness is preserved. `functions/read_trc.py:136`, `functions/io_cortexmac.py:132`, embedded `read_trc.py:207` and `read_mac_ascii_files.py:196`. The notebook explanation was synchronized by root. |
| **Plate 1 selection also selected plate 10** | A `.forces` fixture with plates 1 and 10 returned 14 channels for `forcepla=[1]`, while reporting one plate. | Exact channel-prefix/numeric-suffix parsing and explicit unavailable/incomplete-plate errors: `functions/io_cortexmac.py:403`, embedded `read_mac_ascii_files.py:439`. Requested plate order is preserved. |
| **Calibration plate IDs were truncated** | A `.cal` record for plate 10 returned `[1]`. | Parse the complete ID, removing only an optional K suffix: `functions/io_cortexmac.py:314`, embedded `read_mac_ascii_files.py:346`. Whitespace-separated rows also tolerate repeated spaces. |
| **Integer-valued data failed unit conversion under pandas 3** | Integer text coordinates such as 3 mm raised `TypeError` when fractional meter values were assigned through `.iloc`. | Coordinate columns are converted to float before TRC unit conversion; `.forces` numeric data are read as float. Both shared and embedded copies are covered by tests. Source force/COP/free-moment units and conversions are documented as N/mm/N mm → N/m/N m for these Cortex exports. |
| **Visual3D export silently misaligned equally sampled streams** | Two TRC rows with a RangeIndex and two force rows with a seconds index became three output rows with missing values. The legacy resampling path inferred one frequency from sample-count ratios and discarded nonzero time origins. | `functions/io_cortexmac.py:913` and embedded `read_mac_ascii_files.py:505` use actual timestamps, validate origin and recording intervals, and linearly interpolate onto the common measured span at the higher native/requested rate. Data are assembled positionally only after alignment. No extrapolation or gap-filling is performed. A force RangeIndex requires an explicit frequency. |
| **Visual3D filename input exported millimeters as if MKS; target components had distinct signal names** | Legacy filename loading used `units=''`; a 1 mm source coordinate remained 1 in the output. Headers used Ax/Ay/Az as separate target names. | Filename TRC input is converted to meters; DataFrame input is explicitly documented as already MKS. The writer emits a TIME signal, one repeated marker name per XYZ group, and the corresponding component fields. Both corrected writer copies are tested. |
| **Zero-padding distorted the plotted heel trajectory at recording boundaries** | For the supplied walk2 R.Heely, the final measured height was 0.187873 m; default polyphase zero-padding produced a final interpolated value of 0.058366 m. Line extension produced 0.188767 m. | Root applied `padtype='line'` in both `read_trc_forces_files.py` resampling cells (current lines 214 and 304), after this review supplied the reproduction. |
| **C3D header demonstration rejected the supplied DEC file** | The initial exported script raised `ValueError: This teaching header example expects Intel representation`; an undefined `_byteorder` was also visible in that cell. | Root completed processor validation and integer byte-order selection. The final guide executes successfully on the bundled DEC fixture. |

The Visual3D writer intentionally limits output to times actually supported by both streams. For bundled walk1 this yields 1,090 rows through 2.42 s; walk2 yields 751 rows through 1.666667 s. Final analog samples beyond the last measured marker time are omitted, as documented. This prevents silently extrapolating marker position. Interval checks establish compatible timestamps, not independent proof that acquisition devices were synchronized.

## Original-content and conceptual coverage

| Guide | Review result |
|---|---|
| OpenC3Dfile | Retains binary-header interpretation, C3D structure, metadata, marker and analog arrays, force traces and marker-motion visualization. Obsolete BTK installation and separate-window execution are replaced with the current pure-Python reader and slider. Original GIF preservation remains noted above. |
| read_c3d | Retains the original practical task of reading synchronized analog/marker tables. Replaces an unavailable private trial and binary EZC3D dependency with the supplied public C3D file and pure-Python reader. The table indexing and channel-unit discussion match the fixture. |
| pandas_data | Retains flat and hierarchical columns, marker/component selection, level operations, raw arrays and plots. The public TRC fixture replaces the original private path. No high-confidence additional conceptual/runtime defect found. |
| read_trc | Preserves flat, MultiIndex and xarray representations, alternate empty-trajectory policies, units, and writing support. Whole-marker missing-data behavior is now consistent across representations. |
| read_trc_forces_files | Preserves the two-trial comparison, distinct sample rates, marker resampling, heel-height/force plots and foot-contact interpretation. The bundled trials start at zero and have a 3:1 sample-rate ratio. For walk2, the lower heel at mid-contact supports right foot on plate 6 then left on plate 5; no contrary conclusion was inferred from plate numbers alone. The resampling example remains specific to the demonstrated integer ratio. |
| read_mac_ascii_files | Retains TRC, ANC, calibration, forces and Visual3D helper definitions and the worked data examples. Additional formats remain explained as optional formats. No original numbered Problems with embedded statements were discarded in these six notebooks. |

The bundled C3D has 487 point frames at 100 Hz, 4,870 analog samples at 1,000 Hz, 33 markers and 28 analog channels. Marker units are mm; force channels declare N and moment channels N mm. `c3d.Reader` already applies analog calibration, so the loader correctly avoids scaling a second time. Negative residuals indicate invalid points; a valid zero coordinate is preserved.

## Verification evidence

Temporary reproductions and logs: `/tmp/root-file-review/`. No input fixtures were overwritten.

- `uv run pytest -q tests/test_file_guide_readers.py tests/test_io_cortexmac.py`: **21 passed, 45 subtests passed**. The two warnings are expected py-c3d warnings from deliberately absent analog data/labels. The initial red run reproduced the reader failures before production fixes; fixture-construction compatibility issues in py-c3d's writer were corrected separately.
- The new `tests/test_file_guide_readers.py` executes every repaired shared function and its embedded notebook counterpart. It checks C3D shorter/longer/missing/colliding labels, valid zero coordinates, partial/empty XYZ groups in all supported representations, plate IDs, integer unit conversion, aligned/offset/different-rate Visual3D data, and filename units.
- All six guides: **strict marimo check PASS → script export PASS → actual exported cell execution PASS**, using `MPLBACKEND=Agg`. Results are recorded in `/tmp/root-file-review/execution.json`; per-file logs are beside it. The first OpenC3D execution failed before root's header correction; the final run passed.
- `uv run ruff check tests/test_file_guide_readers.py functions/c3d_data.py`: **PASS**.
- Bundled walk1/walk2 Visual3D exports executed into `/tmp/root-file-review/`, with the expected meter-valued first marker coordinates and aligned output row counts above.

No cloud/editor execution or import into the proprietary Visual3D application was performed. These checks establish Python behavior and documented ASCII structure, not a claim of a live Visual3D import. The optional ANC reader was retained and read during review but no ANC fixture was available for independent hardware-conversion verification.

## Documentation evidence

- [C3D POINT:LABELS specification](https://www.c3d.org/HTML/Documents/pointlabels.htm): label count can differ from stored trajectory count; POINT:USED defines stored points, with additional label parameters supported.
- [SciPy resample_poly reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html): default zero extension and the alternative line-extension boundary policy. Context7 `/scipy/scipy` also confirmed the output-count formula.
- [NumPy interp reference](https://numpy.org/doc/2.4/reference/generated/numpy.interp.html): monotone interpolation coordinates and explicit out-of-range behavior; fetched through Context7 `/websites/numpy_doc_2_4` before the writer change.
- [HAS-Motion Visual3D ASCII format](https://wiki.has-motion.com/doku.php?id=visual3d%3Adocumentation%3Adefinitions%3Afile_formats%3Avisual3d_ascii_format): time-signal/header fields, component structure and MKS units. No live application import is claimed.
- Context7 searches for `c3d` and `py-c3d` returned unrelated packages. The review therefore inspected installed py-c3d 0.6.0 source and the primary C3D format specification for those details.

## Reviewer-owned changes in this follow-up

`functions/c3d_data.py`; `functions/read_trc.py`; the `read_trc`, `read_cal`, `read_forces`, and `write_v3dtxt` functions in `functions/io_cortexmac.py`; corresponding embedded functions in `OpenC3Dfile.py`, `read_c3d.py`, `read_trc.py`, and `read_mac_ascii_files.py`; `tests/test_file_guide_readers.py`; this report. Other shared io_cortexmac functions were not edited in this follow-up. No commits, directory moves, input-data writes, or dependency/loader edits were made by this reviewer. `tnorma` is no longer used by the revised Visual3D helper; root was informed for its dependency pass.
