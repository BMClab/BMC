# Editable notebooks

[Open the Body segment parameters pilot](https://bmclab.github.io/BMC/notebooks_edit/body_segment_parameters.html).

Python source stays in `notebooks/`. `notebooks.json` is the explicit list of
notebooks approved for editable publication; it contains only the pilot for now.
GitHub Actions generates the HTML and browser assets under `notebooks_edit/` in
the deployed site. These generated files are not committed to Git.

Click **Run all** to compute results, then edit and rerun cells. Edits do not write
to the repository; download the Python file to retain your own copy. The editor
links to the corresponding [saved version](https://bmclab.github.io/BMC/notebooks_read/body_segment_parameters.html).

See [the build instructions](../scripts/README.md) for the local procedure.
