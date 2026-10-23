"""Add approved editable notebooks to an existing Pages build. See scripts/README.md."""

from __future__ import annotations

import argparse
import json
import posixpath
import shutil
import subprocess
import sys
import tempfile
from html import escape
from pathlib import Path, PurePosixPath

import marimo
from export_notebooks import (
    EDIT_DIR,
    EDITABLE_LIST,
    READ_DIR,
    ROOT,
    STYLE,
    atomic_write,
    digest,
    editable_sources,
    notebook_title,
    output_path,
)


def decorate(document: str, source: PurePosixPath, sources=()) -> str:
    if document.count("<body>") != 1 or document.count("</head>") != 1:
        raise ValueError("Unsupported marimo WASM HTML document")
    parent = output_path(source).parent
    saved = posixpath.relpath(
        str(PurePosixPath("..") / READ_DIR / output_path(source)), str(parent)
    )
    catalog = posixpath.relpath("index.html", str(parent))
    nav = f'''
<nav id="bmc-static-navigation" aria-label="Notebook viewing and execution modes">
  <p><a href="{catalog}">All guides</a> · <strong>Editable notebook</strong> · Python runs in your browser</p>
  <div class="bmc-launches"><a href="{saved}">Read saved outputs</a><span>Run all cells to start. Download your Python file to keep edits.</span></div>
</nav>
'''
    # The WASM editor fills an absolutely positioned container. Anchor it below
    # the navigation bar rather than to the viewport.
    style = STYLE + "<style>#root { position: relative; }</style>\n"
    return (
        document.replace("</head>", style + "</head>", 1)
        .replace("<body>", "<body>" + nav, 1)
        .replace("</body>", reading_links(source, sources) + "</body>", 1)
    )


def reading_links(source: PurePosixPath, sources) -> str:
    """Keep rendered guide links local while preserving editable Python source."""
    parent = str(output_path(source).parent)
    targets = {
        str(path): posixpath.relpath(str(output_path(path)), parent) for path in sources
    }
    script = r"""<script id="bmc-reading-links">
(() => {
  const targets = __BMC_TARGETS__;
  const virtual = new URL(__BMC_SOURCE__, 'https://bmc.invalid/');
  function rewrite(anchor) {
    if (anchor.closest('.cm-editor')) return;
    const raw = anchor.getAttribute('href');
    if (!raw || raw.startsWith('#') || raw.startsWith('?')) return;
    try {
      let url, key;
      if (/^[a-z][a-z0-9+.-]*:|^\/\//i.test(raw)) {
        url = new URL(raw);
        const prefix = '/github/BMClab/BMC/blob/master/';
        if (url.origin !== 'https://molab.marimo.io' || !url.pathname.startsWith(prefix) || !url.pathname.endsWith('.py')) return;
        key = decodeURIComponent(url.pathname.slice(prefix.length));
      } else {
        url = new URL(raw, virtual);
        if (!url.pathname.endsWith('.py')) return;
        key = decodeURIComponent(url.pathname.slice(1));
        if (!targets[key]) key = decodeURIComponent(new URL(raw, 'https://bmc.invalid/').pathname.slice(1));
      }
      if (targets[key]) anchor.setAttribute('href', targets[key] + url.search + url.hash);
    } catch (_) { /* External or malformed references remain untouched. */ }
  }
  function scan(node) {
    if (node.nodeType !== 1) return;
    if (node.matches('a[href]')) rewrite(node);
    node.querySelectorAll('a[href]').forEach(rewrite);
  }
  const root = document.getElementById('root');
  new MutationObserver(records => records.forEach(record => {
    if (record.type === 'attributes') rewrite(record.target);
    else record.addedNodes.forEach(scan);
  })).observe(root, {subtree: true, childList: true, attributes: true, attributeFilter: ['href']});
  document.addEventListener('click', event => {
    const anchor = event.composedPath().find(node => node.matches?.('a[href]'));
    if (anchor) rewrite(anchor);
  }, true);
  scan(root);
})();
</script>
"""
    return script.replace(
        "__BMC_TARGETS__", json.dumps(targets).replace("<", r"\u003c")
    ).replace("__BMC_SOURCE__", json.dumps(str(source)))


def legacy_redirects(site: Path) -> None:
    for alias, current in (
        ("html", READ_DIR),
        ("notebooks_read", READ_DIR),
        ("notebooks_edit", EDIT_DIR),
    ):
        old_root = site / alias
        if old_root.exists():
            shutil.rmtree(old_root)
        for page in (site / current).rglob("*.html"):
            relative = page.relative_to(site / current)
            old = old_root / relative
            target = posixpath.relpath(
                str(PurePosixPath(current) / relative.as_posix()),
                str(PurePosixPath(alias) / relative.parent.as_posix()),
            )
            document = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta http-equiv="refresh" content="0; url={escape(target, quote=True)}"><title>Notebook moved</title>
<script>location.replace({json.dumps(target)} + location.search + location.hash);</script></head>
<body><a href="{escape(target, quote=True)}">Open the notebook</a></body></html>
'''
            atomic_write(old, document)


def build(site: Path, *, root: Path = ROOT) -> dict:
    site = site.resolve()
    if site == root.resolve():
        raise ValueError("Use a separate site build directory, not the repository root")
    sources = editable_sources(root)
    if not sources:
        raise ValueError(f"No notebooks selected in {EDITABLE_LIST}")
    for source in sources:
        if not (site / READ_DIR / output_path(source)).is_file():
            raise ValueError(
                f"Build the read-only site first: missing {READ_DIR}/{output_path(source)}"
            )
    if not (site / READ_DIR / "index.html").is_file():
        raise ValueError("Build the read-only catalog before exporting editable pages")
    entries = []
    with tempfile.TemporaryDirectory(dir=site, prefix=".bmc-edit-") as temporary:
        staging = Path(temporary) / EDIT_DIR
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
            atomic_write(target, decorate(target.read_text(), source, sources))
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
            f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Editable BMC notebooks</title><style>body{{max-width:65rem;margin:3rem auto;padding:0 1.2rem;font:17px/1.6 system-ui,sans-serif;color:#172b3a}}a{{color:#176d52}}li{{margin:.4rem 0}}</style></head><body>
<h1>Editable BMC notebooks</h1><p>Run Python and experiment in your browser. Edits do not update the GitHub repository; download your Python file to keep a copy.</p><ul>{links}</ul><p><a href="../{READ_DIR}/index.html">Read saved outputs for all guides</a></p></body></html>\n""",
        )
        manifest = {"marimo": marimo.__version__, "mode": "edit", "notebooks": entries}
        atomic_write(
            staging / "export-manifest.json", json.dumps(manifest, indent=2) + "\n"
        )
        destination = site / EDIT_DIR
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
        f"Exported {len(manifest['notebooks'])} editable notebook(s) to {args.site / EDIT_DIR}"
    )


if __name__ == "__main__":
    main()
