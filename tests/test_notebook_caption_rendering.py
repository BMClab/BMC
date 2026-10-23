"""Figure captions must pass their mathematics through marimo's renderer."""

import ast
import re
import textwrap
import unittest
from html.parser import HTMLParser
from pathlib import Path

import marimo as mo

ROOT = Path(__file__).resolve().parents[1]


class CaptionCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.captions = []
        self.stack = []
        self.current = None
        self.root_depth = None

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        is_caption = tag == "figcaption" or (
            tag == "span" and "figure-caption" in attributes.get("class", "").split()
        )
        if self.current is None and is_caption:
            self.current = {"text": "", "math": 0}
            self.root_depth = len(self.stack)
        if self.current is not None and tag == "marimo-tex":
            self.current["math"] += 1
        if tag not in {"img", "br", "hr", "input", "meta", "link", "source"}:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag not in self.stack:
            return
        index = len(self.stack) - 1 - self.stack[::-1].index(tag)
        if self.current is not None and index <= self.root_depth:
            self.captions.append(self.current)
            self.current = None
        del self.stack[index:]

    def handle_data(self, data):
        if self.current is not None:
            self.current["text"] += data


class NotebookCaptionRenderingTests(unittest.TestCase):
    def test_math_is_typeset_within_every_notebook_caption(self):
        captions_checked = 0
        math_captions_checked = 0
        paths = [
            *(ROOT / "notebooks").glob("*.py"),
            *(ROOT / "courses").rglob("*.py"),
        ]
        for path in sorted(paths):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "md"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                ):
                    continue
                content = textwrap.dedent(node.args[0].value)
                if "<figcaption" not in content and "figure-caption" not in content:
                    continue
                source = CaptionCollector()
                source.feed(content)
                rendered = CaptionCollector()
                rendered.feed(mo.md(content).text)
                expected = [
                    len(
                        re.findall(
                            r"(?<!\\)\$(?!\$).*?(?<!\\)\$(?!\$)", c["text"], re.DOTALL
                        )
                    )
                    for c in source.captions
                ]
                with self.subTest(notebook=path.name, line=node.lineno):
                    self.assertTrue(source.captions, "Caption was not recognized")
                    self.assertIsNone(source.current, "Caption was not closed")
                    self.assertEqual(len(rendered.captions), len(source.captions))
                    self.assertEqual([c["math"] for c in rendered.captions], expected)
                captions_checked += len(expected)
                math_captions_checked += sum(count > 0 for count in expected)
        self.assertGreater(captions_checked, 0)
        self.assertGreater(math_captions_checked, 0)
