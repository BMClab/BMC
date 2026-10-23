# Static HTML notebooks

[Browse the static collection](https://bmclab.github.io/BMC/notebooks_read/)
or open `index.html` locally in a browser.

These pages contain saved text, tables and plots. They can be read without
Python, molab or marimo.app. Expandable exercise solutions remain available.
The two badges at the top open the corresponding notebook in molab or marimo.app
when interactive execution is wanted.

The HTML uses marimo's display assets from a CDN and the original image links,
so an Internet connection is still needed for those resources. Neither notebook
execution service is needed to read the page.

The Python sources remain in [notebooks/](../notebooks/) and [courses/](../courses/).
All 87 guides and the 2026 course lecture are included. Main guide exports use
the source filename with an `.html` extension; course exports preserve their
directory structure under `notebooks_read/courses/`.

Use [scripts/export_notebooks.py](../scripts/export_notebooks.py) to update the
snapshots. See [the update instructions](../scripts/README.md) for incremental,
single-notebook, forced and freshness-check commands. `manifest.json` records
the source and output hashes for each page.

The body-segment guide also has an [editable GitHub Pages pilot](https://bmclab.github.io/BMC/notebooks_edit/body_segment_parameters.html). Other guides remain read-only until the pilot is reviewed.
