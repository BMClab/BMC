"""Execute notebooks and maintain static HTML snapshots. See scripts/README.md."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import posixpath
import re
import subprocess
import sys
import tempfile
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from html import escape, unescape
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit, urlunsplit

import marimo

ROOT = Path(__file__).resolve().parents[1]
READ_DIR = "notebooks_html"
EDITABLE_LIST = "scripts/notebooks.json"
SESSION = re.compile(r'(?m)^(\s*"session": )(.+)(,\s*)$')
PREVIEW = re.compile(
    r"https://molab\.marimo\.io/github/BMClab/BMC/blob/master/"
    r"(?P<source>(?:notebooks|courses)/[\w/.-]+\.py)(?P<wasm>/wasm)?"
)
HREF = re.compile(
    r'(?P<prefix>\bhref=)(?P<quote>\\?["\x27]|&quot;|&#x27;)(?P<url>.*?)(?P=quote)'
)
STYLE = """
<style id="bmc-static-navigation-style">
  body { display: flex; flex-direction: column; height: 100vh; margin: 0; }
  #root { flex: 1; min-height: 0; height: 0; }
  #bmc-static-navigation {
    --bmc-launch-icon-height: 16px;
    --bmc-control-height: 28px;
    position: relative; z-index: 100; flex: none;
    display: flex; align-items: center; justify-content: space-between;
    flex-wrap: wrap; gap: 8px 24px; padding: 12px 24px;
    background: var(--background, Canvas); color: var(--foreground, CanvasText);
    border-bottom: 1px solid var(--border, GrayText);
    font-family: var(--marimo-text-font, var(--text-font, sans-serif));
    font-size: 14px; line-height: 1.4;
  }
  #bmc-static-navigation p { margin: 0; }
  #bmc-static-navigation .bmc-launches { display: flex; align-items: center; flex-wrap: wrap; gap: 10px; }
  #bmc-static-navigation a { display: inline-flex; border-radius: 4px; color: var(--link, LinkText); }
  #bmc-static-navigation a:focus-visible { outline: 3px solid var(--link, Highlight); outline-offset: 3px; }
  #bmc-static-navigation .bmc-launches > a { min-height: var(--bmc-control-height); align-items: center; }
  #bmc-static-navigation .bmc-launches img { display: block; height: var(--bmc-control-height); width: auto; }
  #bmc-static-navigation .bmc-edit-button {
    align-items: center; justify-content: center; gap: 6px;
    box-sizing: border-box; min-height: var(--bmc-control-height); padding: 0 12px;
    border: 1px solid var(--input, ButtonBorder);
    background: var(--secondary, ButtonFace); color: var(--secondary-foreground, ButtonText);
    font-family: inherit;
    font-weight: 600; text-decoration: none;
  }
  #bmc-static-navigation .bmc-edit-button:hover {
    background: var(--accent, ButtonFace); color: var(--accent-foreground, ButtonText);
  }
  #bmc-static-navigation .bmc-edit-button:active {
    background: var(--muted, ButtonFace); color: var(--foreground, ButtonText);
  }
  #bmc-static-navigation .bmc-edit-button svg { flex: none; width: var(--bmc-launch-icon-height); height: var(--bmc-launch-icon-height); }
  @media (max-width: 540px) {
    #bmc-static-navigation { padding: 10px 14px; }
  }
  @media print { #bmc-static-navigation { display: none; } }
</style>
"""


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def output_path(source: PurePosixPath) -> PurePosixPath:
    relative = (
        source.relative_to("notebooks") if source.parts[0] == "notebooks" else source
    )
    return relative.with_suffix(".html")


def discover(root: Path) -> list[PurePosixPath]:
    paths = sorted(
        [*(root / "notebooks").glob("*.py"), *(root / "courses").rglob("*.py")]
    )
    return [
        PurePosixPath(path.relative_to(root).as_posix())
        for path in paths
        if re.search(r"\bmarimo\.App\s*\(", path.read_text())
    ]


def editable_sources(root: Path) -> list[PurePosixPath]:
    registry = root / EDITABLE_LIST
    if not registry.is_file():
        return []
    values = json.loads(registry.read_text())
    if values == {"include": "all"}:
        return discover(root)
    if not isinstance(values, list) or any(
        not isinstance(value, str) for value in values
    ):
        raise ValueError(
            f"{EDITABLE_LIST} must contain a list of notebook source paths"
        )
    known = set(discover(root))
    paths = [PurePosixPath(value) for value in values]
    if len(paths) != len(set(paths)) or any(path not in known for path in paths):
        raise ValueError(
            f"{EDITABLE_LIST} contains duplicate or unknown notebook sources"
        )
    return paths


def badge(label: str, url: str, image_url: str) -> str:
    return f'''<a href="{escape(url, quote=True)}" target="_blank" rel="noopener noreferrer" aria-label="Run this notebook in {label}">
      <img src="{escape(image_url, quote=True)}" alt="Open in {label}" height="28"/></a>'''


def decorate_export(
    document: str,
    source: PurePosixPath,
    sources: list[PurePosixPath],
    *,
    editable_sources=(),
) -> tuple[str, dict]:
    """Rewrite saved reading links, preserving original code and live launch URLs."""
    match = SESSION.search(document)
    if match is None or document.count("<body>") != 1 or document.count("</head>") != 1:
        raise ValueError(
            "Unsupported marimo HTML format: expected saved session and HTML document"
        )
    session = json.loads(match[2])
    targets = {str(path): output_path(path) for path in sources}
    parent = str(output_path(source).parent)

    def local(target):
        return posixpath.relpath(str(target), parent)

    def rewrite(value):
        if isinstance(value, dict):
            return {key: rewrite(item) for key, item in value.items()}
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if not isinstance(value, str):
            return value

        def preview(found):
            target = targets.get(found["source"])
            return (
                local(target) if target is not None and not found["wasm"] else found[0]
            )

        def relative(found):
            parts = urlsplit(unescape(found["url"]))
            if parts.scheme or parts.netloc or not parts.path.endswith(".py"):
                return found[0]
            candidate = posixpath.normpath(str(source.parent / parts.path))
            target = targets.get(candidate, targets.get(parts.path))
            if target is None:
                return found[0]
            url = urlunsplit(("", "", local(target), parts.query, parts.fragment))
            return found["prefix"] + found["quote"] + url + found["quote"]

        return HREF.sub(relative, PREVIEW.sub(preview, value))

    session = rewrite(session)
    encoded = json.dumps(session, ensure_ascii=True)
    for character, replacement in [
        ("<", r"\u003c"),
        (">", r"\u003e"),
        ("&", r"\u0026"),
        ("'", r"\u0027"),
    ]:
        encoded = encoded.replace(character, replacement)
    document = document[: match.start(2)] + encoded + document[match.end(2) :]
    path = str(source)
    launches = badge(
        "molab",
        f"https://molab.marimo.io/github/BMClab/BMC/blob/master/{path}/wasm",
        "https://marimo.io/molab-shield.svg",
    )
    launches += badge(
        "marimo.app",
        f"https://marimo.app/github.com/BMClab/BMC/blob/master/{path}",
        "https://marimo.io/shield.svg",
    )
    edit_link = ""
    if source in editable_sources:
        target = local(PurePosixPath("..") / output_path(source))
        edit_link = f'''<a class="bmc-edit-button" href="{target}">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false">
        <path d="M12 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/>
        <path d="m16 3 5 5-10 10-5 1 1-5Z"/>
      </svg><span>Edit in browser</span></a>'''
    nav = f'''
<nav id="bmc-static-navigation" aria-label="Notebook viewing and execution modes">
  <p><a href="{local("index.html")}">All guides</a> · <strong>Static version</strong> · Saved text, tables and plots</p>
  <div class="bmc-launches">{edit_link}{launches}</div>
</nav>
'''
    document = document.replace("</head>", STYLE + "</head>", 1).replace(
        "<body>", "<body>" + nav, 1
    )
    document = "\n".join(line.rstrip() for line in document.splitlines()) + "\n"
    cells = session.get("cells", [])
    media = sum(
        "image/png" in json.dumps(cell) or "image/svg+xml" in json.dumps(cell)
        for cell in cells
    )
    return document, {"cells": len(cells), "cells_with_saved_figures": media}


def shared_digest(root: Path, sources: list[PurePosixPath]) -> str:
    """Conservatively invalidate exports after changes to any shared local input."""
    hasher = hashlib.sha256()
    hasher.update(
        f"{sys.version}\n{marimo.__version__}\n{digest(Path(__file__))}\n".encode()
    )
    hasher.update("\n".join(map(str, sources)).encode())
    paths = [root / name for name in ("pyproject.toml", "uv.lock", EDITABLE_LIST)]
    for directory in ("data", "functions", "images"):
        paths += [
            path
            for path in (root / directory).rglob("*")
            if path.is_file()
            and not any(
                part.startswith(".") or part == "__pycache__"
                for part in path.relative_to(root).parts
            )
            and path.suffix != ".pyc"
        ]
    for path in sorted(paths):
        if path.is_file():
            hasher.update(path.relative_to(root).as_posix().encode())
            hasher.update(digest(path).encode())
    return hasher.hexdigest()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(text)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def export_one(
    root: Path,
    source: PurePosixPath,
    sources: list[PurePosixPath],
    shared: str,
    timeout: int,
) -> dict:
    source_sha = digest(root / source)
    with tempfile.TemporaryDirectory(prefix="bmc-html-") as directory:
        native = Path(directory) / "notebook.html"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "marimo",
                "export",
                "html",
                str(root / source),
                "-o",
                str(native),
                "--force",
            ],
            cwd=root,
            env=os.environ | {"MPLBACKEND": "module://marimo._output.mpl"},
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if result.returncode:
            raise RuntimeError((result.stdout + result.stderr)[-6000:])
        page, statistics = decorate_export(
            native.read_text(), source, sources, editable_sources=editable_sources(root)
        )
    if digest(root / source) != source_sha:
        raise RuntimeError("Notebook changed during execution; rerun the exporter")
    destination = root / READ_DIR / output_path(source)
    atomic_write(destination, page)
    return {
        "source": str(source),
        "source_sha256": source_sha,
        "shared_inputs_sha256": shared,
        "html": str(output_path(source)),
        "html_sha256": digest(destination),
        "native_export": "passed",
        **statistics,
    }


def is_current(root: Path, source: PurePosixPath, entry: dict, shared: str) -> bool:
    destination = root / READ_DIR / output_path(source)
    return bool(
        not entry.get("last_error")
        and destination.is_file()
        and entry.get("source_sha256") == digest(root / source)
        and entry.get("shared_inputs_sha256") == shared
        and entry.get("html_sha256") == digest(destination)
    )


def notebook_title(path: Path) -> str:
    for node in ast.walk(ast.parse(path.read_text())):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "md"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            match = re.search(
                r"^# (.+)$", textwrap.dedent(node.args[0].value), re.MULTILINE
            )
            if match:
                return match[1].strip()
    return path.stem.replace("_", " ")


def write_catalog(root: Path, sources: list[PurePosixPath]) -> None:
    items = []
    for source in sources:
        if not (root / READ_DIR / output_path(source)).is_file():
            continue
        label = notebook_title(root / source)
        items.append(f'<li><a href="{output_path(source)}">{escape(label)}</a></li>')
    page = (
        """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>BMC static notebooks</title>
<style>body{max-width:65rem;margin:3rem auto;padding:0 1.2rem;font:17px/1.6 system-ui,sans-serif;color:#172b3a}a{color:#176d52}li{margin:.4rem 0}p{max-width:52rem}</style></head><body>
<h1>Biomechanics and Motor Control</h1><p>Read saved notebook outputs without molab or marimo.app. Each page has optional badges for interactive execution. Display assets and external media still need Internet access.</p><ul>
"""
        + "\n".join(items)
        + "\n</ul></body></html>\n"
    )
    atomic_write(root / READ_DIR / "index.html", page)


def main(argv: list[str] | None = None, *, root: Path = ROOT) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "notebooks",
        nargs="*",
        help="Source paths; default: every guide and course notebook",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-execute even if inputs and output are unchanged",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report stale/missing exports without writing or executing",
    )
    parser.add_argument(
        "--jobs", type=int, default=4, help="Concurrent notebook processes (default: 4)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Seconds allowed per notebook (default: 300)",
    )
    args = parser.parse_args(argv)
    if args.jobs < 1 or args.timeout < 1:
        parser.error("--jobs and --timeout must be positive")
    sources = discover(root)
    try:
        editable_sources(root)
    except ValueError as error:
        parser.error(str(error))
    try:
        selected = (
            list(
                dict.fromkeys(
                    PurePosixPath(Path(path).resolve().relative_to(root).as_posix())
                    if Path(path).is_absolute()
                    else PurePosixPath(path)
                    for path in args.notebooks
                )
            )
            or sources
        )
    except ValueError:
        parser.error("Notebook paths must be inside this repository")
    for source in selected:
        if source not in sources:
            parser.error(f"Not a repository notebook: {source}")
    manifest_path = root / READ_DIR / "manifest.json"
    try:
        previous = json.loads(manifest_path.read_text())
    except FileNotFoundError, json.JSONDecodeError:
        previous = {}
    entries = {
        item["source"].removeprefix("../"): item
        for item in previous.get("notebooks", [])
    }
    shared = shared_digest(root, sources)
    stale = [
        source
        for source in selected
        if not is_current(root, source, entries.get(str(source), {}), shared)
    ]
    if args.check:
        for source in stale:
            print(f"STALE {source}")
        print(
            f"{len(selected) - len(stale)}/{len(selected)} selected exports are current"
        )
        return int(bool(stale))
    pending = selected if args.force else stale
    failures = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(export_one, root, source, sources, shared, args.timeout): source
            for source in pending
        }
        for future in as_completed(futures):
            source = futures[future]
            try:
                entries[str(source)] = future.result()
                print(f"EXPORTED {source}", flush=True)
            except (
                OSError,
                RuntimeError,
                ValueError,
                subprocess.TimeoutExpired,
            ) as error:
                failures += 1
                entries.setdefault(str(source), {"source": str(source)})[
                    "last_error"
                ] = str(error)
                print(f"FAILED {source}: {error}", file=sys.stderr, flush=True)
    manifest = {
        "exporter": f"marimo {marimo.__version__}; scripts/export_notebooks.py",
        "viewing": "No Python kernel, molab or marimo.app needed. CDN display assets and external images need Internet access.",
        "notebooks": [
            entries[str(source)] for source in sources if str(source) in entries
        ],
    }
    atomic_write(manifest_path, json.dumps(manifest, indent=2) + "\n")
    write_catalog(root, sources)
    print(
        f"{len(pending) - failures} exported, {len(selected) - len(pending)} unchanged, {failures} failed"
    )
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
