# Editable notebooks

[Browse the editable collection](https://bmclab.github.io/BMC/nbs/).

Python sources stay in `notebooks/` and `courses/`. `notebooks.json` uses
`{"include": "all"}` so every marimo notebook is exported automatically on push.
GitHub Actions generates the HTML and browser assets under `nbs/` in the deployed
site; these generated files are not committed to Git.

Click **Run all**, then edit and rerun cells. Download the Python file to keep
your edits. Every editor links to its saved version in
[nbs_read/](https://bmclab.github.io/BMC/nbs_read/).

See [the build instructions](../scripts/README.md) for the local procedure.
