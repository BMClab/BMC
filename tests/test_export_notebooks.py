"""Static exports must remain readable, current, and safe to update."""

import contextlib
import importlib.util
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path, PurePosixPath
from unittest.mock import patch

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/export_notebooks.py"
_SPEC = importlib.util.spec_from_file_location("export_notebooks", _SCRIPT)
exporter = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(exporter)


def native_html(content="Saved output"):
    session = {
        "cells": [{"outputs": [{"data": {"text/html": content}}], "console": []}]
    }
    return (
        "<!DOCTYPE html><html><head><title>Example</title></head><body>"
        '<div id="root"></div><script>\n'
        f'  "session": {json.dumps(session)},\n'
        '</script><script id="source">Original source links are untouched</script></body></html>'
    )


class StaticPageTests(unittest.TestCase):
    def setUp(self):
        self.paths = [
            PurePosixPath("notebooks/example.py"),
            PurePosixPath("notebooks/reference_frame.py"),
            PurePosixPath("courses/2026/example.py"),
        ]

    def test_reading_links_stay_static_and_launch_links_run_the_same_source(self):
        preview = "https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks/reference_frame.py"
        content = (
            f'<a href="{preview}#axes">Frames</a>'
            '<a href="reference_frame.py#vectors">Vectors</a>'
            f'<a href="{preview}/wasm">Run interactively</a>'
            '<a href="https://en.wikipedia.org/wiki/Frame_of_reference">Reference</a>'
            '<marimo-accordion data-content="&lt;a href=&quot;reference_frame.py#axes&quot;&gt;Answer&lt;/a&gt;"></marimo-accordion>'
        )
        original = native_html(content).replace(
            "Original source links are untouched", preview
        )
        page, _ = exporter.decorate_export(original, self.paths[0], self.paths)
        self.assertIn('href="reference_frame.html#axes"', page.replace('\\"', '"'))
        self.assertIn("reference_frame.html#vectors", page)
        self.assertIn("reference_frame.html#axes", page)
        self.assertIn(preview + "/wasm", page)
        self.assertIn("https://en.wikipedia.org/wiki/Frame_of_reference", page)
        self.assertIn('<script id="source">' + preview, page)
        self.assertIn("/master/notebooks/example.py/wasm", page)
        self.assertIn(
            "https://marimo.app/github.com/BMClab/BMC/blob/master/notebooks/example.py",
            page,
        )

    def test_course_launch_and_catalog_links_use_the_course_path(self):
        page, _ = exporter.decorate_export(native_html(), self.paths[2], self.paths)
        self.assertIn("/master/courses/2026/example.py/wasm", page)
        self.assertIn('href="../../index.html"', page)
        self.assertEqual(
            exporter.output_path(self.paths[2]),
            PurePosixPath("courses/2026/example.html"),
        )

    def test_output_strings_cannot_terminate_the_export_script(self):
        page, _ = exporter.decorate_export(
            native_html("</script><p>Saved text</p>"), self.paths[0], self.paths
        )
        self.assertEqual(page.count("</script>"), 2)
        self.assertIn("Saved text", page)


class ExportUpdateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "notebooks").mkdir()
        (self.root / "data").mkdir()
        (self.root / "pyproject.toml").write_text('[project]\nname="fixture"\n')
        (self.root / "uv.lock").write_text("locked environment")
        self.source = self.root / "notebooks/example.py"
        self.source.write_text("import marimo\napp = marimo.App()\n")
        self.asset = self.root / "data/example.txt"
        self.asset.write_text("first input")
        self.page = self.root / "html/example.html"

    def run_exporter(self, *args):
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            return exporter.main(list(args), root=self.root)

    def fake_export(self, args, **kwargs):
        self.assertEqual(kwargs["env"]["MPLBACKEND"], "module://marimo._output.mpl")
        Path(args[args.index("-o") + 1]).write_text(native_html())
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    def test_incremental_updates_detect_source_shared_input_and_output_changes(self):
        with patch.object(
            exporter.subprocess, "run", side_effect=self.fake_export
        ) as execute:
            self.assertEqual(self.run_exporter(), 0)
            self.assertEqual(self.run_exporter(), 0)
            self.assertEqual(execute.call_count, 1)
            self.source.write_text(
                self.source.read_text() + "# a changed explanation\n"
            )
            self.assertEqual(self.run_exporter("--check"), 1)
            self.assertEqual(self.run_exporter(), 0)
            self.asset.write_text("updated input")
            self.assertEqual(self.run_exporter("--check"), 1)
            self.assertEqual(self.run_exporter(), 0)
            self.page.write_text("tampered output")
            self.assertEqual(self.run_exporter("--check"), 1)
            self.assertEqual(self.run_exporter(), 0)
            self.assertEqual(execute.call_count, 4)
            self.assertEqual(self.run_exporter("--check"), 0)

    def test_failed_native_export_preserves_the_previous_page_and_reports_failure(self):
        with patch.object(exporter.subprocess, "run", side_effect=self.fake_export):
            self.assertEqual(self.run_exporter(), 0)
        original = self.page.read_bytes()
        with patch.object(
            exporter.subprocess,
            "run",
            return_value=subprocess.CompletedProcess(
                [], 1, stdout="", stderr="A cell failed"
            ),
        ):
            self.assertEqual(self.run_exporter("--force"), 1)
        self.assertEqual(self.page.read_bytes(), original)
        self.assertEqual(self.run_exporter("--check"), 1)

    def test_single_notebook_update_keeps_other_manifest_entries_and_pages(self):
        second = self.root / "notebooks/other.py"
        second.write_text("import marimo\napp = marimo.App()\n")
        with patch.object(
            exporter.subprocess, "run", side_effect=self.fake_export
        ) as execute:
            self.assertEqual(self.run_exporter(), 0)
            other_page = self.root / "html/other.html"
            previous = other_page.read_bytes()
            self.assertEqual(self.run_exporter("notebooks/example.py", "--force"), 0)
            self.assertEqual(execute.call_count, 3)
        self.assertEqual(other_page.read_bytes(), previous)
        manifest = json.loads((self.root / "html/manifest.json").read_text())
        self.assertEqual(len(manifest["notebooks"]), 2)

    def test_check_is_read_only_even_when_no_export_exists(self):
        self.assertEqual(self.run_exporter("--check"), 1)
        self.assertFalse((self.root / "html").exists())

    def test_bad_export_format_does_not_overwrite_a_previous_good_page(self):
        with patch.object(exporter.subprocess, "run", side_effect=self.fake_export):
            self.assertEqual(self.run_exporter(), 0)
        previous = self.page.read_bytes()

        def invalid_export(args, **kwargs):
            Path(args[args.index("-o") + 1]).write_text(
                "<html>No saved notebook data</html>"
            )
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

        with patch.object(exporter.subprocess, "run", side_effect=invalid_export):
            self.assertEqual(self.run_exporter("--force"), 1)
        self.assertEqual(self.page.read_bytes(), previous)

    def test_edit_during_export_is_not_recorded_as_a_current_snapshot(self):
        with patch.object(exporter.subprocess, "run", side_effect=self.fake_export):
            self.assertEqual(self.run_exporter(), 0)
        previous = self.page.read_bytes()

        def edit_during_export(args, **kwargs):
            result = self.fake_export(args, **kwargs)
            self.source.write_text(
                self.source.read_text() + "# edited while exporting\n"
            )
            return result

        with patch.object(exporter.subprocess, "run", side_effect=edit_during_export):
            self.assertEqual(self.run_exporter("--force"), 1)
        self.assertEqual(self.page.read_bytes(), previous)
        self.assertEqual(self.run_exporter("--check"), 1)

    def test_catalog_uses_the_notebook_heading_instead_of_dependency_comments(self):
        self.source.write_text("""# /// script
# dependencies = ["marimo"]
# ///
import marimo
app = marimo.App()
@app.cell
def _(mo):
    mo.md("# A readable guide title")
""")
        with patch.object(exporter.subprocess, "run", side_effect=self.fake_export):
            self.assertEqual(self.run_exporter(), 0)
        catalog = (self.root / "html/index.html").read_text()
        self.assertIn("A readable guide title", catalog)
        self.assertNotIn("/// script", catalog)

    def test_native_execution_saves_printed_results_and_show_figures(self):
        self.source.write_text("""import marimo
app = marimo.App()
@app.cell
def _():
    import matplotlib.pyplot as plt
    return (plt,)
@app.cell
def _(plt):
    plt.plot([0, 1], [1, 0])
    plt.show()
    print("Result: 2")
    return
""")
        self.assertEqual(self.run_exporter(), 0)
        manifest = json.loads((self.root / "html/manifest.json").read_text())
        self.assertGreater(manifest["notebooks"][0]["cells_with_saved_figures"], 0)
        self.assertIn("Result: 2", self.page.read_text())
