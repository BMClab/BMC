"""Add approved editable notebooks to an existing Pages build. See scripts/README.md."""

from __future__ import annotations

import argparse
import json
import posixpath
import subprocess
import sys
import tempfile
from html import escape
from pathlib import Path, PurePosixPath

import marimo
from export_notebooks import (
    READ_DIR,
    ROOT,
    STYLE,
    atomic_write,
    digest,
    editable_sources,
    notebook_title,
    output_path,
)


def decorate(document: str, source: PurePosixPath) -> str:
    if document.count("<body>") != 1 or document.count("</head>") != 1:
        raise ValueError("Unsupported marimo WASM HTML document")
    parent = output_path(source).parent
    saved = posixpath.relpath(
        str(PurePosixPath("..") / READ_DIR / output_path(source)), str(parent)
    )
    catalog = posixpath.relpath(
        str(PurePosixPath("..") / READ_DIR / "index.html"), str(parent)
    )
    nav = f'''
<nav id="bmc-static-navigation" aria-label="Notebook viewing and execution modes">
  <p><a href="{catalog}">All guides</a> · <strong>Editable notebook</strong> · Python runs in your browser</p>
  <div class="bmc-launches"><a href="{saved}">Read saved outputs</a><span>Run all cells to start. Download your Python file to keep edits.</span></div>
</nav>
'''
    # The WASM editor fills an absolutely positioned container. Anchor it below
    # the navigation bar rather than to the viewport.
    style = STYLE + "<style>#root { position: relative; }</style>\n"
    return document.replace("</head>", style + "</head>", 1).replace(
        "<body>", "<body>" + nav, 1
    )


def legacy_redirects(site: Path) -> None:
    for saved in (site / READ_DIR).rglob("*.html"):
        relative = saved.relative_to(site / READ_DIR)
        old = site / "html" / relative
        target = posixpath.relpath(
            str(PurePosixPath(READ_DIR) / relative.as_posix()),
            str(PurePosixPath("html") / relative.parent.as_posix()),
        )
        document = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta http-equiv="refresh" content="0; url={escape(target, quote=True)}"><title>Notebook moved</title>
<script>location.replace({json.dumps(target)} + location.search + location.hash);</script></head>
<body><a href="{escape(target, quote=True)}">Open the saved notebook</a></body></html>
'''
        atomic_write(old, document)


def build(site: Path, *, root: Path = ROOT) -> dict:
    site = site.resolve()
    sources = editable_sources(root)
    if not sources:
        raise ValueError("No notebooks approved in notebooks_edit/notebooks.json")
    for source in sources:
        if not (site / READ_DIR / output_path(source)).is_file():
            raise ValueError(
                f"Build the read-only site first: missing {READ_DIR}/{output_path(source)}"
            )
    if not (site / READ_DIR / "index.html").is_file():
        raise ValueError("Build the read-only catalog before exporting editable pages")
    entries = []
    with tempfile.TemporaryDirectory(dir=site, prefix=".bmc-edit-") as temporary:
        staging = Path(temporary) / "notebooks_edit"
        staging.mkdir()
        for source in sources:
            before = digest(root / source)
            target = staging / output_path(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "marimo",
                    "export",
                    "html-wasm",
                    str(root / source),
                    "--mode",
                    "edit",
                    "-o",
                    str(target),
                ],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
            if result.returncode:
                raise RuntimeError((result.stdout + result.stderr)[-6000:])
            if not target.is_file() or not (target.parent / "assets").is_dir():
                raise ValueError(f"Incomplete WASM export for {source}")
            if digest(root / source) != before:
                raise RuntimeError(f"{source} changed during export; rerun the build")
            atomic_write(target, decorate(target.read_text(), source))
            entries.append(
                {
                    "source": str(source),
                    "source_sha256": before,
                    "html": str(output_path(source)),
                    "html_sha256": digest(target),
                    "mode": "edit",
                }
            )
        links = "\n".join(
            f'<li><a href="{output_path(source)}">{escape(notebook_title(root / source))}</a></li>'
            for source in sources
        )
        atomic_write(
            staging / "index.html",
            f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Editable BMC notebooks</title></head><body>
<h1>Editable BMC notebooks</h1><p>Run Python and experiment in your browser. Edits do not update the GitHub repository; download your Python file to keep a copy.</p><ul>{links}</ul><p><a href="../{READ_DIR}/index.html">Read saved outputs for all guides</a></p></body></html>\n""",
        )
        manifest = {"marimo": marimo.__version__, "mode": "edit", "notebooks": entries}
        atomic_write(
            staging / "export-manifest.json", json.dumps(manifest, indent=2) + "\n"
        )
        destination = site / "notebooks_edit"
        previous = Path(temporary) / "previous"
        if destination.exists():
            destination.replace(previous)
        try:
            staging.replace(destination)
        except OSError:
            if previous.exists():
                previous.replace(destination)
            raise
    legacy_redirects(site)
    (site / ".nojekyll").touch()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--site",
        type=Path,
        default=ROOT / "_site",
        help="Existing Jekyll/site build (default: _site)",
    )
    args = parser.parse_args()
    manifest = build(args.site)
    print(
        f"Exported {len(manifest['notebooks'])} editable notebook(s) to {args.site / 'notebooks_edit'}"
    )


if __name__ == "__main__":
    main()
