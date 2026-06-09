# Introduction to Python for Modeling and Simulation

## "IDE"

- [Marimo](https://marimo.io/)

### Examples

- [Welcome to marimo](https://molab.marimo.io/notebooks/nb_TWVGCgZZK4L8zj5ziUBNVL)
- [Plotting](https://molab.marimo.io/notebooks/nb_vXxD13t2RoMTLjC89qdn6c)

## Learn Python

- [GitHub repo with Python notebooks](https://github.com/marimo-team/learn/tree/main/python)

## Install marimo into a user-local Python environment so you don't need admin/root

The reliable, no-admin approaches:

### Option 1 — `pip --user`

```bash
python -m pip install --user marimo
```

Then run with:

```bash
python -m marimo edit
```

### Option 2 — virtual environment (cleaner, recommended)

```bash
python -m venv ~/marimo-env
# Windows:
%USERPROFILE%\marimo-env\Scripts\activate
# macOS/Linux:
source ~/marimo-env/bin/activate

pip install marimo
marimo edit
```

A venv lives entirely in your home directory and needs no elevated rights.

### Option 3 — `uv` (no system Python config needed)

```bash
# install uv to ~/.local without admin
curl -LsSf https://astral.sh/uv/install.sh | sh   # macOS/Linux
# or on Windows PowerShell:
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

uv tool install marimo
marimo edit
```

### Option 4 — fully browser-based (zero install, WASM)

If even pip is locked down by the corporate proxy, use the in-browser version at the marimo playground: [marimo.new](https://marimo.new/) — it runs via Pyodide/WebAssembly with nothing installed locally. Good fallback when network egress to PyPI is also blocked.

Related links if useful:

- [marimo.app](https://marimo.app/) — same engine, but lets you save and generate shareable permalinks.
- [molab.marimo.io](https://molab.marimo.io/) — free cloud-hosted version backed by a real server (better for heavier ML/AI work that Pyodide can't handle).

## Install marimo on your local computer when you have admin/root access

Use these instructions on a personal computer where you are allowed to install
software. Admin/root access is useful for installing Python or `uv`; after that,
you should still install marimo in an isolated tool environment or virtual
environment instead of modifying the operating system's Python packages.

### Option 1 — `uvx` (recommended for a quick start)

Install `uv` once, then let `uvx` run marimo in an isolated environment.

Windows PowerShell:

```powershell
winget install --id=astral-sh.uv -e
uvx marimo edit
```

macOS with Homebrew:

```bash
brew install uv
uvx marimo edit
```

Linux/macOS with the official installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uvx marimo edit
```

This is the simplest option if you only want to open, edit, and run marimo
notebooks.

### Option 2 — install Python, then use a virtual environment

Install Python first. On Windows, use the installer from
[python.org](https://www.python.org/downloads/) and enable "Add python.exe to
PATH". On macOS, Homebrew is a convenient option. On Ubuntu/Debian, use `apt`.

Windows:

```powershell
py -m venv %USERPROFILE%\marimo-env
%USERPROFILE%\marimo-env\Scripts\activate
python -m pip install --upgrade pip
python -m pip install marimo
python -m marimo edit
```

macOS:

```bash
brew install python
python3 -m venv ~/marimo-env
source ~/marimo-env/bin/activate
python -m pip install --upgrade pip
python -m pip install marimo
python -m marimo edit
```

Ubuntu/Debian:

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
python3 -m venv ~/marimo-env
source ~/marimo-env/bin/activate
python -m pip install --upgrade pip
python -m pip install marimo
python -m marimo edit
```

Do not use `sudo pip install marimo` on Linux or macOS unless you know exactly
why you need it. It can change system Python packages that other programs depend
on. A virtual environment is safer and easier to delete.

### Option 3 — install in an existing Python environment

If you already have a Python environment for the course, activate it and install
marimo there:

```bash
python -m pip install marimo
python -m marimo edit
```

This is fine for a course-specific environment, but avoid mixing unrelated
projects in the same environment.

Common gotchas on locked-down corporate machines:

- If PyPI is blocked, point pip at your company's internal mirror: `pip install --user -i https://your-internal-mirror/simple marimo`
- If `marimo` isn't on your PATH after `--user` install, call it as `python -m marimo edit` or add the user scripts dir (`python -m site --user-base`) to PATH.
- Corporate TLS interception may require `--trusted-host` or your org's CA bundle.
