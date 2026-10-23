"""Publishing must retain saved pages and export only the approved notebooks."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


class EditablePagesTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location(
            "export_editable", SCRIPTS / "export_editable.py"
        )
        self.builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.builder)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root / "notebooks").mkdir()
        (self.root / "scripts").mkdir()
        for name in ("pilot", "other"):
            (self.root / "notebooks" / f"{name}.py").write_text(
                "import marimo\napp = marimo.App()\n@app.cell\ndef _():\n    2 + 2\n"
            )
        self.registry = self.root / "scripts/notebooks.json"
        self.registry.write_text('["notebooks/pilot.py"]')
        self.site = self.root / "_site"
        (self.site / "notebooks_html").mkdir(parents=True)
        (self.site / "notebooks_html/index.html").write_text("Saved catalog")
        (self.site / "notebooks_html/pilot.html").write_text("Saved pilot")
        (self.site / "index.html").write_text("Existing Cayman site")
        self.theme = self.site / "assets/css/style.css"
        self.theme.parent.mkdir(parents=True)
        self.theme.write_text("Existing Cayman theme")
        self.syllabus = self.site / "courses/2026/syllabus.html"
        self.syllabus.parent.mkdir(parents=True)
        self.syllabus.write_text("Existing course material")

    def test_real_native_export_is_editable_and_preserves_the_existing_site(self):
        result = self.builder.build(self.site, root=self.root)
        page = self.site / "pilot.html"
        self.assertEqual(len(result["notebooks"]), 1)
        self.assertIn('"mode": "edit"', page.read_text())
        self.assertIn('href="notebooks_html/pilot.html"', page.read_text())
        self.assertIn('href="index.html"', page.read_text())
        self.assertTrue(any((self.site / "assets").glob("*.js")))
        self.assertFalse((self.site / "other.html").exists())
        self.assertEqual((self.site / "index.html").read_text(), "Existing Cayman site")
        self.assertEqual(self.theme.read_text(), "Existing Cayman theme")
        self.assertEqual(self.syllabus.read_text(), "Existing course material")
        self.assertEqual(
            (self.site / "notebooks_html/pilot.html").read_text(), "Saved pilot"
        )
        redirect = (self.site / "html/pilot.html").read_text()
        self.assertIn("../notebooks_html/pilot.html", redirect)
        self.assertIn("location.search", redirect)
        self.assertIn("location.hash", redirect)
        self.assertIn("../pilot.html", (self.site / "nbs/pilot.html").read_text())
        self.assertIn("../index.html", (self.site / "nbs/index.html").read_text())

    def test_native_failure_preserves_previous_editable_output(self):
        target = self.site / "pilot.html"
        target.write_text("Previous working editor")
        with (
            patch.object(
                self.builder.subprocess,
                "run",
                return_value=subprocess.CompletedProcess(
                    [], 1, stdout="", stderr="Export failed"
                ),
            ),
            self.assertRaises(RuntimeError),
        ):
            self.builder.build(self.site, root=self.root)
        self.assertEqual(target.read_text(), "Previous working editor")
        self.assertEqual((self.site / "index.html").read_text(), "Existing Cayman site")
        self.assertEqual(self.theme.read_text(), "Existing Cayman theme")
        self.assertFalse((self.site / "html").exists())

    def test_missing_readonly_page_or_empty_registry_stops_build(self):
        (self.site / "notebooks_html/pilot.html").unlink()
        with self.assertRaises(ValueError):
            self.builder.build(self.site, root=self.root)
        self.registry.write_text("[]")
        with self.assertRaises(ValueError):
            self.builder.build(self.site, root=self.root)

    def test_full_collection_and_course_paths_have_all_previous_redirects(self):
        self.registry.write_text('{"include": "all"}')
        (self.site / "notebooks_html/other.html").write_text("Saved other")
        source = self.root / "courses/2026/lecture.py"
        source.parent.mkdir(parents=True)
        source.write_text("import marimo\napp = marimo.App()\n")
        saved = self.site / "notebooks_html/courses/2026/lecture.html"
        saved.parent.mkdir(parents=True)
        saved.write_text("Saved lecture")
        result = self.builder.build(self.site, root=self.root)
        self.assertEqual(len(result["notebooks"]), 3)
        page = (self.site / "courses/2026/lecture.html").read_text()
        self.assertIn('href="../../notebooks_html/courses/2026/lecture.html"', page)
        self.assertIn('href="../../index.html"', page)
        self.assertEqual(self.syllabus.read_text(), "Existing course material")
        for prefix in ("html", "notebooks_read", "nbs_read"):
            self.assertIn(
                "../notebooks_html/pilot.html",
                (self.site / prefix / "pilot.html").read_text(),
            )
            self.assertIn(
                "../../../notebooks_html/courses/2026/lecture.html",
                (self.site / prefix / "courses/2026/lecture.html").read_text(),
            )
            self.assertIn(
                "../notebooks_html/index.html",
                (self.site / prefix / "index.html").read_text(),
            )
        for prefix in ("nbs", "notebooks_edit"):
            self.assertIn(
                "../pilot.html", (self.site / prefix / "pilot.html").read_text()
            )
            self.assertIn(
                "../../../courses/2026/lecture.html",
                (self.site / prefix / "courses/2026/lecture.html").read_text(),
            )

    def test_local_build_without_a_home_page_creates_a_root_catalog(self):
        (self.site / "index.html").unlink()
        self.builder.build(self.site, root=self.root)
        catalog = (self.site / "index.html").read_text()
        self.assertIn('href="pilot.html"', catalog)
        self.assertIn('href="notebooks_html/index.html"', catalog)

    def test_reserved_index_notebook_cannot_replace_the_home_page(self):
        (self.root / "notebooks/index.py").write_text(
            "import marimo\napp = marimo.App()\n"
        )
        self.registry.write_text('["notebooks/index.py"]')
        (self.site / "notebooks_html/index.html").write_text("Saved index notebook")
        with self.assertRaisesRegex(ValueError, "index.html"):
            self.builder.build(self.site, root=self.root)
        self.assertEqual((self.site / "index.html").read_text(), "Existing Cayman site")

    def test_rebuild_removes_retired_editor_without_removing_unrelated_pages(self):
        (self.site / "index.html").unlink()
        self.registry.write_text('{"include": "all"}')
        (self.site / "notebooks_html/other.html").write_text("Saved other")
        self.builder.build(self.site, root=self.root)
        self.assertTrue((self.site / "other.html").is_file())
        self.assertIn('href="other.html"', (self.site / "index.html").read_text())
        self.registry.write_text('["notebooks/pilot.py"]')
        self.builder.build(self.site, root=self.root)
        self.assertFalse((self.site / "other.html").exists())
        self.assertFalse((self.site / "nbs/other.html").exists())
        self.assertNotIn('href="other.html"', (self.site / "index.html").read_text())
        self.assertIn('href="pilot.html"', (self.site / "index.html").read_text())
        self.assertEqual(self.syllabus.read_text(), "Existing course material")
        self.assertEqual(self.theme.read_text(), "Existing Cayman theme")
        self.assertEqual(
            (self.site / "notebooks_html/other.html").read_text(), "Saved other"
        )

    def test_repository_root_is_rejected_as_a_build_destination(self):
        (self.root / "notebooks_html").mkdir()
        (self.root / "notebooks_html/index.html").write_text("Catalog")
        (self.root / "notebooks_html/pilot.html").write_text("Saved pilot")
        with self.assertRaises(ValueError):
            self.builder.build(self.root, root=self.root)
        self.assertEqual(json.loads(self.registry.read_text()), ["notebooks/pilot.py"])
