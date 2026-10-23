"""Embed the common asset resolver while keeping each notebook standalone.

Run with explicit notebook paths, or no arguments to synchronize the collection.
"""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / "functions" / "notebook_resources.py").read_text()
node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef))
helper = ast.get_source_segment(source, node)
paths = [Path(p) for p in sys.argv[1:]] or list((ROOT / "notebooks").glob("*.py"))
for path in paths:
    text = path.read_text()
    tree = ast.parse(text)
    matches = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "repo_file"]
    if not matches:
        continue
    lines = text.splitlines(keepends=True)
    n = matches[0]
    lines[n.lineno - 1 : n.end_lineno] = [helper + "\n"]
    path.write_text("".join(lines))
