---
name: project-kickoff
description: "Initialize a Python data-science project with Git and uv. Use when: starting a new project, setting up a reproducible environment, scaffolding a data analysis repo with uv and Git, creating a bioinformatics project."
argument-hint: "Project name and key dependencies (e.g., 'snRNA-seq project with scanpy')"
---

# Project Kickoff: Git + uv Initialization

## When to Use
- Starting a new Python data-science or bioinformatics project
- Setting up a reproducible environment with uv + Git
- Scaffolding directory structure for multi-notebook analysis

## Procedure

### 1. Initialize Git
- Run `git init` in the project root
- Create `.gitignore` with entries for:
  - Python: `__pycache__/`, `*.pyc`, `.venv/`, `*.egg-info/`
  - Jupyter: `.ipynb_checkpoints/`
  - Data: `data/`, `*.h5ad`, `*.h5`, `*.loom`
  - OS: `.DS_Store`

### 2. Initialize uv
- Run `uv init --no-readme` to create `pyproject.toml`
- Pin a compatible Python version: `uv python install 3.11 && uv python pin 3.11`
- Edit `pyproject.toml`:
  - `[project]` — name, version, description, `requires-python >= "3.11"`
  - `[project.dependencies]` — analysis packages (e.g., scanpy, anndata, matplotlib)
  - `[dependency-groups]` `dev` — development tools (e.g., jupyter, ipykernel, ruff)
- Or use `uv add <pkg>` and `uv add --group dev <pkg>` to add interactively
- Remove the auto-generated `main.py` if present

### 3. Install Dependencies
- Run `uv sync --all-groups` to create virtual environment and lockfile
- Verify: `uv run python -c "import <main_package>; print('OK')"`

### 4. Scaffold Directory Structure

```
project-root/
├── src/                # Local Python modules (processing, plotting, utils)
│   └── __init__.py
├── notebooks/          # Jupyter notebooks (numbered for execution order)
├── data/               # Raw and processed data (gitignored)
├── references/         # External references, papers, companion code notes
├── pyproject.toml
├── uv.lock
├── .gitignore
└── README.md
```

### 5. Create README.md
Include:
- Project description and dataset reference
- Setup instructions (`uv sync --all-groups`)
- Notebook execution order table
- Project structure overview
- Key references (paper DOI, GEO accession, companion websites)

### 6. Register Jupyter Kernel (when ready for notebook execution)

```bash
uv run python -m ipykernel install --user --name <project_name> --display-name "<Display Name>"
```

### 7. Initial Commit

```bash
git add .
git commit -m "chore: initialize project with uv and directory structure"
```

## Notes
- Always gitignore `data/` — raw data should be fetched programmatically for reproducibility
- Use `uv add <pkg>` for new deps; `uv add --group dev <pkg>` for dev tools
- Number notebooks (`01_`, `02_`, ...) to enforce execution order
- Save intermediate results as `.h5ad` checkpoints in `data/` between notebooks
- Pin `requires-python` to the minimum version you support
- Keep `uv.lock` in version control for full reproducibility
