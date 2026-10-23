"""Files resolve relative to the notebook; failed downloads never poison caches."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('resource_example', ROOT / 'notebooks' / 'amti_bsf_file.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
repo_file = module.repo_file


class NotebookResourceTests(unittest.TestCase):
    def test_course_notebook_finds_repository_data_without_cwd(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'data').mkdir()
            wanted = root / 'data' / 'example.txt'
            wanted.write_text('measured data')
            course = root / 'courses' / 'example'
            course.mkdir(parents=True)
            with patch('marimo.notebook_dir', return_value=course):
                self.assertEqual(repo_file('data/example.txt'), wanted)

    def test_rejects_escape_from_repository(self):
        with self.assertRaises(ValueError):
            repo_file('../outside.txt')

    def test_missing_notebook_location_does_not_prevent_remote_fallback(self):
        from types import SimpleNamespace
        response = SimpleNamespace(status=200, data=b'1,2\n')
        with tempfile.TemporaryDirectory() as folder:
            with patch('marimo.notebook_dir', return_value=None), patch('tempfile.gettempdir', return_value=folder), patch('urllib3.request', return_value=response):
                result = repo_file('data/example with spaces.csv')
                self.assertEqual(result.read_bytes(), b'1,2\n')

    def test_failed_download_does_not_create_cached_file(self):
        from types import SimpleNamespace
        response = SimpleNamespace(status=404, data=b'not found')
        with tempfile.TemporaryDirectory() as folder:
            with patch('marimo.notebook_dir', return_value=None), patch('tempfile.gettempdir', return_value=folder), patch('urllib3.request', return_value=response):
                with self.assertRaises(OSError):
                    repo_file('data/nonexistent.csv')
            self.assertFalse(list(Path(folder).rglob('*.csv')))
