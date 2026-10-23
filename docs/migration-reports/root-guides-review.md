# Independent review of six root-owned guides

Review date: 2026-10-22. Scope: `EnsembleAverage`, `rep_missing`, `resample`, `IEP`, `Text and binary files`, and `read_delsys_cortex_file`, plus `functions/critic_damp.py`. The guides were reviewed read-only. A subsequent explicit assignment authorized the shared filter repair and its focused regression tests. This worker did not edit the six guides. Author/reference/asset-loader normalization was in progress and was excluded from this review.

The review compared actual current source with paired Jupyter Markdown/code and the marimo blobs at `5366ffa7e12fc61b3c2410c173128e468a9fe524`. Line numbers below describe the files during this review; central formatting may move them. No cloud-site execution is claimed.

## Material findings and disposition

### P1 — Ensemble normalization discarded longer trials' endpoints (corrected by coordinator)

Location: `notebooks_marimo/EnsembleAverage.py:112–118`, originally the single `tnorma(y)` call. The prose and Exercise 1 promise that each complete valid trial maps to 0–100% cycle. Passing the whole padded matrix to `tnorma` instead removes every boundary row containing any NaN, thereby trimming every trial at the shortest trial's endpoint.

Reproduction: pad a five-sample ramp from 0 to 1 beside an eight-sample ramp from 0 to 1. The former matrix call returns final values `[1, 0.57142857]`, with retained original rows `[0, 4]`. The seeded tutorial has valid trial lengths `[109, 117, 81, 90, 117, 92, 88, 106, 87, 110]` but retained only rows `[0, 80]`; some trials lose 36 samples before normalization. This inherited defect changes the ensemble's movement phases.

Correction: independently normalize each column's complete finite interval and stack the results. The coordinator applied this at lines 115–118. The focused column-wise reproducer now reaches `[1, 1]`; inspection confirms the notebook uses that correction. Duration remains a separate measurement and cannot be recovered from normalized percent-cycle time alone.

### P1 — Delsys reader erased asynchronous time origins (corrected by coordinator)

Location: `notebooks_marimo/read_delsys_cortex_file.py:283,308` and the matching shared `read_delsys` implementation in `functions/io_cortexmac.py`. Both signals previously received an index starting at zero, regardless of their recorded Cortex timestamps. That can make EMG and IMU signals with different acquisition origins appear synchronized.

Reproduction: shift the provided synthetic example's EMG Cortex timestamps by 150 frames and IMU timestamps by 300 frames; read with `freq_trc=150`, `resample=[0, 0]`, and output writing disabled. Before correction, both indexes began at 0 seconds instead of 1 and 2 seconds.

The coordinator restored each signal's original time origin. The same notebook reproducer now returns 1.0 and 2.0 seconds, and the shared-helper `DelsysOffsetTests` regression passes. This establishes preservation of distinct starts. Full handling of irregular timestamps or asynchronous origins across multiple individual sensors was not established by this one-sensor synthetic example.

### P2 — Interpolation skipped NaNs when the custom sentinel was absent (corrected by coordinator)

Location: `notebooks_marimo/rep_missing.py:105–107,120–131` and the matching shared helper. The interpolation contract says that both the configured missing-data sentinel and NaNs are filled. Previously the outer `idx.any()` branch tested only sentinel occurrences, so an absent custom sentinel prevented NaN interpolation altogether.

Reproduction: `rep_missing([1, np.nan, 3], value=0, new_value='interp')` returned `[1, nan, 3]`. The coordinator combined the sentinel and NaN masks before deciding whether replacement is needed. The current notebook reproducer returns `[1, 2, 3]`.

### P2 — RMS envelope documentation describes a different operation (corrected by coordinator)

Location: `notebooks_marimo/read_delsys_cortex_file.py:105–112,121–124,146–147`; `moving_rms` is at lines 80–86. The docstring says the RMS filter is applied twice and calls `fc_lp` a low-pass cutoff. The implementation calls `moving_rms` once: one centered convolution of squared samples with a normalized rectangular window, followed by a square root. Its window length is `round(freq/fc_lp)`; `fc_lp` therefore controls inverse window duration and is not the RMS operation's Butterworth half-power cutoff.

A 60-sample signal with 20 zeros, 20 ones and 20 zeros, using a 10-sample window, differs by up to 0.387298 when the RMS operation is applied twice rather than once. The difference is substantive, not merely wording about an equivalent implementation.

Recommended correction: document one moving-RMS operation, the inverse-window-duration parameter, zero-padding/endpoint limitations, and distinguish it from the rectification path's forward/backward Butterworth filtering. Preserve the current calculation unless there is a separate scientific reason to change it. The obsolete embedded docstring link is also a suitable target for the coordinator's link normalization.

### P2 — IEP's group-comparison visualization examples were omitted (restored by coordinator)

Location: `notebooks_marimo/IEP.py:449–486`. Subject-level summaries and group count/mean/SD tables are retained, but both legacy visualization examples were removed: COP/Rambling/Trembling versus Vision, faceted by Surface and separated by AgeGroup; and COP/COGv/COP−COGv with the same grouping. The paired `notebooks/IEP.ipynb` contains these as `plot_summary` and `plot_summary2`.

Recommended correction: retain an optional plot of selected subject summaries for each component set. Individual subject values with explicit sample counts would preserve the comparison example without reviving automatic whole-dataset processing, unjustified exclusions, or treating repeated trials as independent people. The default single trial must continue to be identified as insufficient for a population comparison. This is a preservation gap, not a flaw in the current numerical summaries.

## Other reviewed content

| Guide | Review result and preservation assessment |
| --- | --- |
| EnsembleAverage | Original trial-generation, normalization, ensemble SD, and display alternatives remain. The distinction between variability and uncertainty of the mean is explained. Main material issue was the normalization defect above. |
| rep_missing | Numeric replacement, deletion, mean/median and interpolation examples remain. Endpoint extension and uniformly spaced sample assumptions are explicit. Main material issue was the absent-sentinel branch above. |
| resample | No additional high-confidence material issue found. Preserves rational polyphase resampling, time origin and effective output rate, and explains anti-aliasing and uniform-input assumptions. Integer-input behavior was tested against float input with the installed SciPy and matched; no speculative legacy integer-input issue is reported. |
| IEP | Preserves the equilibrium-point model, illustrative media, force-zero interpolation, rambling/trembling, COG comparison, group summaries and component covariance discussion. Correctly distinguishes a model interpretation from proof of neural control. The current metadata uses recorded height in centimeters and mass in kilograms; each trial has its own valid interval. The original plotting gap is noted above. |
| Text and binary files | No high-confidence material issue found. Preserves the original text/binary comparison and makes encoding, representation, portability and size/speed qualifications more accurate. The runnable byte/array examples and final answers are consistent. |
| read_delsys_cortex_file | Preserves reader functionality and adds a synthetic runnable file without requiring private equipment data. The offset bug and RMS documentation mismatch are described above. Synthetic parsing does not establish compatibility with every proprietary Cortex/Delsys export variation. |

## Authorized shared filter repair

The original shared `functions/critic_damp.py` multiplied the requested digital frequency in hertz by an analog correction factor before prewarping. That is only a low-frequency approximation. For two passes at a desired 10 Hz with a 100 Hz sampling rate, the original combined power was approximately 0.52255 for Butterworth and 0.60045 for the critical filter instead of 0.5. At 35 Hz the critical-filter branch fell back to an uncorrected design, giving combined power 0.0625.

With explicit coordinator authorization, the shared function now matches the already corrected notebook implementation at `functions/critic_damp.py:56–64`:

```python
corr = (2**(1/(2*npass)) - 1)**(-0.5)  # critical filter
corr = (2**(1/npass) - 1)**(-0.25)     # Butterworth filter
wc = np.tan(np.pi * fcut / freq) * corr
fc = freq / np.pi * np.arctan(wc)
```

Each correction line is selected only for its corresponding filter family. The coefficient calculation uses `wc`. Correcting the prewarped frequency makes the desired combined half-power frequency exact for this bilinear design, including cutoffs near Nyquist. The function rejects invalid cutoff/sample-rate/pass-count/filter inputs instead of warning and silently falling back. `fcorr=False` preserves uncorrected design semantics. Documentation distinguishes per-pass order, digital design frequency, and the analog critical step-response property.

`tests/test_critic_damp.py` checks the physical response with SciPy `freqz`, rather than repeating the coefficient formula as its expected result: both families, cutoffs 2/10/35 Hz, and 1/2/4 passes yield combined power 0.5. It also verifies unity DC gain, stable poles, coefficient agreement with SciPy's second-order Butterworth design at the returned frequency, uncorrected behavior, and invalid-input rejection. At desired 10 Hz/two passes/100 Hz sampling, the returned Butterworth design frequency is 12.249240607627653 Hz.

## Verification and evidence

- `MPLBACKEND=Agg uv run pytest -q tests/test_critic_damp.py tests/test_signal_processing_notebooks.py tests/test_io_cortexmac.py`: **27 passed, 56 subtests passed**. An earlier run overlapped the coordinator's Delsys repair and failed only the new offset regression; the rerun above passed after that repair landed.
- Current notebook function reproducers confirm the Delsys native start times and missing-data interpolation corrections. A separate normalization reproducer confirms both complete ramp endpoints are preserved by the new column-wise operation.
- Scoped Ruff on the changed filter/helper test passed, with the pre-existing executable-file-mode `EXE002` warning excluded for the legacy helper. No broad notebook formatting was run by this reviewer.
- Temporary extracted legacy content and reproducers are under `/tmp/root-guides-review/`, `/tmp/root_review_repro.py`, and `/tmp/root_review_corrections.py`. They write only temporary synthetic data, never original measurement data.
- The review did not run all six notebooks in a browser or on marimo.app/molab. Actual-cell exports and central render/cloud checks remain the coordinator's separately recorded verification work.

Primary documentation/source consulted: [tnorma](https://github.com/demotu/tnorma), including installed 0.0.3 implementation; [BDS dataset description](https://raw.githubusercontent.com/demotu/datasets/master/BDS/README.md) and [metadata](https://raw.githubusercontent.com/demotu/datasets/master/BDS/BDSinfo.txt); current SciPy documentation obtained through Context7 for [filtfilt](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html), [freqz](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.freqz.html), and [bilinear](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.bilinear.html). The BDS publication page itself returned an access error; no claim depends on having read that inaccessible page.

Coordinator resolution: corrected RMS documentation in both shared/embedded helpers and restored both optional IEP subject-level comparison figure sets. Final execution and toggle checks are recorded in the integration report.
