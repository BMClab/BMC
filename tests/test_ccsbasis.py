import ast
import unittest
import warnings
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "functions" / "CCSbasis.py"

LABELS = [
    r"$\hat{e_1}$",
    r"$\hat{e_2}$",
    r"$\hat{e_3}$",
    r"$\hat{i}$",
    r"$\hat{j}$",
    r"$\hat{k}$",
]


class CCSbasisSourceTests(unittest.TestCase):
    def test_source_has_no_invalid_escape_sequences(self):
        source = MODULE_PATH.read_text(encoding="utf-8")

        with warnings.catch_warnings():
            warnings.simplefilter("error", SyntaxWarning)
            compile(source, str(MODULE_PATH), "exec")

    def test_axis_labels_are_unchanged(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
        strings = [
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and "hat" in node.value
        ]

        self.assertEqual(strings, LABELS)


if __name__ == "__main__":
    unittest.main()
