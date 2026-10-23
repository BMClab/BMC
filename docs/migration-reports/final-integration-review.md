# Final independent integration review

Date: 2026-10-22. This was a focused, read-only review of the consolidated `notebooks/` collection, README navigation, rendering structure, standalone metadata, teaching-section coverage and active test/course paths. Only this report was added by the reviewer. Full native and WebAssembly execution remains the coordinator's separate verification campaign.

**Disposition: no material current issue remains in this review's scope.** Two dependency omissions, one missing setup section and stale directory instructions were reported during the review, corrected by the coordinator, and rechecked below.

## Final inventory and navigation

- `notebooks/` contains **87 Python guides**, all with snake_case filenames. The 87 values in `notebook-filenames.json` match the actual file set exactly, with no missing targets or collisions.
- README contains **85 guide bullets**, covering every guide except `lab_acel_gravidade.py` and `lab_filtros.py`. Those two omissions follow the latest explicit user instruction.
- Each guide title points to its molab preview. Each bullet has a molab `/wasm` badge and a marimo.app badge pointing to the same final filename. There are **170 rendered guide badge images**. README renders no tables.
- README has one Change detection entry linking to `https://github.com/demotu/detecta` and one Time normalization of data entry linking to `https://github.com/demotu/tnorma`; neither destination is duplicated in README.
- **739 active notebook-path occurrences** across README, notebook source, tests, maintenance scripts and course Markdown/Python were checked against the final file set. There are no unresolved notebook targets or remaining `notebooks_marimo` references in those active surfaces. The documented `notebooks/*.py` shell glob is valid and was not treated as a literal filename.
- The two Portuguese lab asset URLs intentionally retain `courses/biomechanicsI_2022/massa_A.txt` under immutable commit `5366ffa7e12fc61b3c2410c173128e468a9fe524`. These are archived external data references, not dependencies on the removed working-tree directory. Integration follow-up by the coordinator: the remaining older-course Jupyter assignment link was subsequently replaced with the current marimo muscle-modeling guide.

## Rendered bylines and references

The review extracted the actual Markdown arguments and rendered them with the installed marimo Markdown renderer; it did not infer layout solely from line breaks in Python source. HTML inspection covered **87 byline blocks and 88 References blocks** (one guide has two reference blocks), with a temporary combined rendering available at `/tmp/bmc-final-static-review/rendered.html`.

- Every byline renders one blockquote with exactly two explicit `<br>` elements: authors, laboratory, university. Both Marcos Duarte and Renato Watanabe are credited throughout; original author order and English/Portuguese wording are retained. Each block links the laboratory address.
- Every References block renders real list items, ranging from 1 to 15 items per block. None renders the references as one undifferentiated paragraph.
- README was independently rendered to `/tmp/bmc-final-static-review/readme.html` to verify bullet-list and badge structure.

This establishes rendered HTML structure. It is not a claim of browser screenshot inspection, complete mathematical typesetting review, or cloud execution.

## Dependencies and teaching structure

- All **87** guides have valid PEP 723 TOML blocks, a Python requirement and a marimo dependency. AST import inspection and dependency-block inspection found **no pyodide imports or pyodide dependencies**. Explanatory mentions of the word are not imports.
- Direct imports were compared with metadata. Local helper imports, Matplotlib's `mpl_toolkits` namespace, and the explicitly optional `dltx` import with an embedded fallback were distinguished from missing distribution dependencies. Loaded helper dependencies were also inspected.
- The initial review found missing `pandas` in the angular-velocity guide, whose imported TRC reader requires pandas at module import, and missing `scipy` in the inverted-pendulum guide, whose executed COG helper imports SciPy. The final `angular_velocity_3d.py:3` declares pandas and `ip_model.py:3` declares SciPy. Both corrections were rechecked.
- All guides have How to use, an opening question or challenge, setup, distributed exercises with worked solutions, checkpoint questions, independent problems with collapsed final-answer panels, further reading, Internet video lectures and references. Wording-aware inspection accepted the Portuguese headings and the section-wise exercise groups in `ch2_viscoelastic_models.py`; opening questions do not need a literal “Challenge” label to serve that role.
- `ensemble_average.py` initially lacked explicit setup prose. Its new Python setup section identifies NumPy, Matplotlib, SciPy and tnorma and was rechecked after correction.
- The Problems-region scan found collapsed answer markup and final-answer labels throughout the collection. This integration pass did not independently recompute every mathematical answer; those content checks belong to the per-group migration and scientific-review reports.

## Active launch and test paths

The review found three local launch commands in `python_installation.py` that still used the deleted `notebooks_marimo/` directory, and one stale directory description in `biomechanics.py`. The coordinator corrected these to `notebooks/`; the final active-surface scan returns no stale occurrences.

All renamed notebook references in test loaders point to existing final files, including constant joined paths and the dynamic `load_notebook(...)` names. `scripts/sync_notebook_resources.py` targets `notebooks/`. `uv run pytest --collect-only -q` completed successfully with **94 tests collected**. Collection is a path/import check, not a claim that the tests themselves ran in this review.

## Evidence and limits

- Temporary review scripts: `/tmp/bmc_final_static_review.py`, `/tmp/bmc_final_paths_review.py`.
- Per-guide SHA-256, metadata, rendered-block counts and scan details: `/tmp/bmc-final-static-review/audit.json`.
- Final inventory/navigation/path result: `/tmp/bmc-final-static-review/paths.json` — 87 guides, 87 mapping entries, 85 guide bullets, 170 badges, 739 path occurrences, empty issue list.
- Review source changes were limited to this report. No notebook, README, course, test or helper edits were made by the reviewer during this integration task.
- Cloud-link shapes were checked against the requested final routes and local filenames. Since links target the published `master` branch, this review does not claim that unpublished renamed files are already reachable there or that either cloud service executed them. The coordinator owns final native/WebAssembly/cloud evidence and report refreshes.
