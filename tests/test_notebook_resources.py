"""Files resolve relative to the notebook; failed downloads never poison caches."""

import ast
import builtins
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "resource_example", ROOT / "functions" / "notebook_resources.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
repo_file = module.repo_file


def resource_import_cells():
    """Load actual notebook import cells without running their other examples."""
    for path in sorted((ROOT / "notebooks").glob("*.py")):
        tree = ast.parse(path.read_text())
        if not any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "repo_file"
            for node in ast.walk(tree)
        ):
            continue
        cells = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_"
            and any(
                isinstance(item, ast.ImportFrom) and item.module == "notebook_resources"
                for item in ast.walk(node)
            )
        ]
        if len(cells) != 1:
            raise AssertionError(
                f"{path.name} must import the shared resolver in one cell"
            )
        cell = cells[0]
        cell.decorator_list = []
        yield path, ast.unparse(cell)


class NotebookResourceImportTests(unittest.TestCase):
    def test_missing_helper_explains_download_destination_and_retry(self):
        original_import = builtins.__import__

        def missing_helper(name, *args, **kwargs):
            if name == "notebook_resources":
                raise ModuleNotFoundError(name=name)
            return original_import(name, *args, **kwargs)

        checked = 0
        for path, code in resource_import_cells():
            with self.subTest(notebook=path.name):
                namespace = {}
                exec(code, namespace)  # noqa: S102 - Exercise the actual repository notebook cell.
                with (
                    patch("builtins.__import__", side_effect=missing_helper),
                    self.assertRaises(ModuleNotFoundError) as caught,
                ):
                    namespace["_"]()
                error = caught.exception
                self.assertEqual(error.name, "notebook_resources")
                self.assertIn(
                    "https://raw.githubusercontent.com/BMClab/BMC/master/functions/notebook_resources.py",
                    str(error),
                )
                self.assertIn("same folder", str(error))
                self.assertIn("rerun", str(error).lower())
                checked += 1
        self.assertGreater(checked, 0)

    def test_downloaded_helper_beside_detached_notebook_resolves_local_data(self):
        _, code = next(resource_import_cells())
        with tempfile.TemporaryDirectory() as folder:
            location = Path(folder)
            (location / "notebook_resources.py").write_bytes(
                (ROOT / "functions/notebook_resources.py").read_bytes()
            )
            (location / "data").mkdir()
            (location / "data/example.txt").write_text("measured data")
            script = (
                "import sys\n"
                f"sys.path.insert(0, {folder!r})\n"
                f"exec({code!r})\n"
                "resolver, = _()\n"
                'assert resolver("data/example.txt").read_text() == "measured data"\n'
            )
            result = subprocess.run(
                [sys.executable, "-I", "-c", script],
                cwd=folder,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_existing_helper_missing_dependency_is_not_reported_as_missing_helper(self):
        _, code = next(resource_import_cells())
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "notebook_resources.py").write_text(
                "import missing_dependency_of_existing_helper\n"
            )
            script = (
                f"import sys\nsys.path.insert(0, {folder!r})\nexec({code!r})\n_()\n"
            )
            result = subprocess.run(
                [sys.executable, "-I", "-c", script],
                cwd=folder,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("missing_dependency_of_existing_helper", result.stderr)
            self.assertNotIn("https://raw.githubusercontent.com", result.stderr)


class NotebookResourceTests(unittest.TestCase):
    def test_course_notebook_finds_repository_data_without_cwd(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "data").mkdir()
            wanted = root / "data" / "example.txt"
            wanted.write_text("measured data")
            course = root / "courses" / "example"
            course.mkdir(parents=True)
            with patch("marimo.notebook_dir", return_value=course):
                self.assertEqual(repo_file("data/example.txt"), wanted)

    def test_rejects_escape_from_repository(self):
        with self.assertRaises(ValueError):
            repo_file("../outside.txt")

    def test_missing_notebook_location_does_not_prevent_remote_fallback(self):
        from types import SimpleNamespace

        response = SimpleNamespace(status=200, data=b"1,2\n")
        with (
            tempfile.TemporaryDirectory() as folder,
            patch("marimo.notebook_dir", return_value=None),
            patch("tempfile.gettempdir", return_value=folder),
            patch("urllib3.request", return_value=response),
        ):
            result = repo_file("data/example with spaces.csv")
            self.assertEqual(result.read_bytes(), b"1,2\n")

    def test_failed_download_does_not_create_cached_file(self):
        from types import SimpleNamespace

        response = SimpleNamespace(status=404, data=b"not found")
        with tempfile.TemporaryDirectory() as folder:
            with (
                patch("marimo.notebook_dir", return_value=None),
                patch("tempfile.gettempdir", return_value=folder),
                patch("urllib3.request", return_value=response),
                self.assertRaises(OSError),
            ):
                repo_file("data/nonexistent.csv")
            self.assertFalse(list(Path(folder).rglob("*.csv")))
