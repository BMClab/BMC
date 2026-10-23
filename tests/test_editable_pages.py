"""Publishing must retain saved pages and export only the approved notebooks."""

import importlib.util
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
        (self.root / "notebooks_edit").mkdir()
        for name in ("pilot", "other"):
            (self.root / "notebooks" / f"{name}.py").write_text(
                "import marimo\napp = marimo.App()\n@app.cell\ndef _():\n    2 + 2\n"
            )
        self.registry = self.root / "notebooks_edit/notebooks.json"
        self.registry.write_text('["notebooks/pilot.py"]')
        self.site = self.root / "_site"
        (self.site / "notebooks_read").mkdir(parents=True)
        (self.site / "notebooks_read/index.html").write_text("Saved catalog")
        (self.site / "notebooks_read/pilot.html").write_text("Saved pilot")
        (self.site / "index.html").write_text("Existing Cayman site")

    def test_real_native_export_is_editable_and_preserves_the_existing_site(self):
        result = self.builder.build(self.site, root=self.root)
        page = self.site / "notebooks_edit/pilot.html"
        self.assertEqual(len(result["notebooks"]), 1)
        self.assertIn('"mode": "edit"', page.read_text())
        self.assertIn('href="../notebooks_read/pilot.html"', page.read_text())
        self.assertTrue((self.site / "notebooks_edit/assets").is_dir())
        self.assertFalse((self.site / "notebooks_edit/other.html").exists())
        self.assertEqual((self.site / "index.html").read_text(), "Existing Cayman site")
        self.assertEqual(
            (self.site / "notebooks_read/pilot.html").read_text(), "Saved pilot"
        )
        redirect = (self.site / "html/pilot.html").read_text()
        self.assertIn("../notebooks_read/pilot.html", redirect)
        self.assertIn("location.search", redirect)
        self.assertIn("location.hash", redirect)

    def test_native_failure_preserves_previous_editable_output(self):
        target = self.site / "notebooks_edit/pilot.html"
        target.parent.mkdir()
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
        self.assertFalse((self.site / "html").exists())

    def test_missing_readonly_page_or_empty_registry_stops_build(self):
        (self.site / "notebooks_read/pilot.html").unlink()
        with self.assertRaises(ValueError):
            self.builder.build(self.site, root=self.root)
        self.registry.write_text("[]")
        with self.assertRaises(ValueError):
            self.builder.build(self.site, root=self.root)
