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

If even pip is locked down by the corporate proxy, use the in-browser version at the marimo playground: [https://marimo.new/](https://marimo.new/) — it runs via Pyodide/WebAssembly with nothing installed locally. Good fallback when network egress to PyPI is also blocked.

Common gotchas on locked-down corporate machines:

- If PyPI is blocked, point pip at your company's internal mirror: `pip install --user -i https://your-internal-mirror/simple marimo`
- If `marimo` isn't on your PATH after `--user` install, call it as `python -m marimo edit` or add the user scripts dir (`python -m site --user-base`) to PATH.
- Corporate TLS interception may require `--trusted-host` or your org's CA bundle.
