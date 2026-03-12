---
name: notebook-writing
description: "Write Jupyter notebooks in Jupytext percent format with structured markdown narrative. Use when: creating new analysis notebooks, adding cells to existing notebooks, reviewing notebook quality. Enforces percent-format conventions, section numbering, I/O tables, and plot standards."
argument-hint: "Notebook topic and number (e.g., '07 Cell-Type Deconvolution')"
---

# Notebook Writing Standards (Jupytext Percent Format)

## When to Use
- Creating a new analysis notebook in `notebooks/`
- Adding or editing cells in an existing percent-format notebook
- Reviewing notebook structure and narrative quality

## File Format

All notebooks are **Jupytext percent-format `.py` files** — not `.ipynb`.

- Code cells start with `# %%`
- Markdown cells start with `# %% [markdown]` and every line is prefixed with `# `
- Blank markdown lines use a bare `#`
- Files are named `{NN}_{Title_Snake_Case}.py` (e.g., `06_Spatial_GAT.py`)

## Notebook Structure

Every notebook follows this exact skeleton:

### 1. Title Cell (First Cell — Markdown)

```
# %% [markdown]
# # {NN} — {Human-Readable Title}
#
# **Pipeline Step {X} of {Y}**
#
# {1–3 paragraph overview of what this notebook does and why.}
#
# ### Pipeline
# 1. {Step one}
# 2. {Step two}
# ...
#
# ### Inputs
# | File | Description |
# |---|---|
# | `path/to/input.h5ad` | {What this file contains} |
#
# ### Outputs
# | File | Description |
# |---|---|
# | `path/to/output.png` | {What this file contains} |
```

Rules:
- Title format is always `# {NN} — {Title}` (em dash, not hyphen)
- Subtitle is bold: `**Pipeline Step X of Y**` or `**Deep Learning Extension**`
- Pipeline section uses a numbered list
- Inputs / Outputs are markdown tables with backtick-wrapped file paths
- The overview is written in scientific/biomedical tone

### 2. Setup Cell (Second Cell — Code)

```python
# %%
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.module_name import function_a, function_b

DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
CACHE_DIR = PROJECT_ROOT / "cache"
ASSETS_DIR = PROJECT_ROOT / "assets"

print("Imports ready.")
```

Rules:
- `PROJECT_ROOT` boilerplate is always first
- Imports from `src/` come after the path setup
- Path constants are ALL_CAPS, built from `PROJECT_ROOT`
- Cell ends with `print("Imports ready.")`
- If warning suppression is needed, place `warnings.filterwarnings(...)` before other imports
- Global plot defaults are set here if the notebook produces figures:
  `set_plot_defaults(fontsize=12, dpi=300)`

### 3. Section Cells (Alternating Markdown → Code)

Each analysis step is a **markdown cell** (explanation) followed by one or more **code cells** (execution).

#### Section Header Format

```
# %% [markdown]
# ## {N.M} {Section Title}
#
# {1–3 paragraphs explaining what this step does and why it matters.}
```

- `N` = notebook number, `M` = section number (1-indexed)
- Examples: `## 1.1 Download Dataset`, `## 4.7 Translational Drug Target Discovery`
- Markdown text is scientific prose — explain the *why*, not just the *what*
- Use **bold** for emphasis, `backticks` for code/paths/parameters
- Use markdown tables for structured data
- Use ordered lists for sequential procedures
- Use bullet points for unordered enumerations

#### Analytical Notes (Blockquotes)

For nuanced or counter-intuitive results, use blockquote format:

```
# %% [markdown]
# ## {N.M} Analytical Note
#
# > **Why does X happen?**
# >
# > Explanation paragraph 1.
# >
# > Explanation paragraph 2.
```

### 4. Final Cell

The last substantive code cell typically saves outputs and prints a summary.
The notebook may optionally end with a trailing markdown cell (e.g., interpretation).

## Code Cell Conventions

### General
- One logical operation per code cell — keep cells focused
- End cells with `print()` statements summarizing results
- Use `display()` from IPython for DataFrames and styled tables
- Use `from IPython.display import Image, display` for showing saved figures

### Result Reporting Pattern

```python
print(f"\nLoaded: {adata.shape[0]} spots × {adata.shape[1]} genes")
print(f"Samples: {adata.obs['sample_id'].nunique()}")
```

- Use f-strings with descriptive labels
- Use `\n` prefix for visual separation after a computation
- Use `:,` format spec for large numbers
- Use `:.4f` for floats, `:.1f` for file sizes

### Memory Cleanup (When Needed)

```python
del adata.uns["temp_key"]
import gc; gc.collect()
```

### Module Reload (During Development)

```python
from importlib import reload
import src.spatial_pipeline as sp
reload(sp)
from src.spatial_pipeline import updated_function
```

## Plot Conventions

### Standard Plot Recipe

```python
fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor("white")

# ... plotting code ...

ax.spines[["top", "right"]].set_visible(False)
ax.set_xlabel("X Label", fontsize=12)
ax.set_ylabel("Y Label", fontsize=12)
ax.set_title("Title", fontsize=13, pad=10)

plt.tight_layout()
out_path = ASSETS_DIR / "descriptive_name.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(fig)
display(Image(filename=str(out_path), width=650))
print(f"Plot saved → {out_path}")
```

Rules:
- White figure background: `fig.patch.set_facecolor("white")`
- Remove top/right spines: `ax.spines[["top", "right"]].set_visible(False)`
- Save at 300 DPI with `bbox_inches="tight"`
- Always `plt.close(fig)` after saving (prevents inline double-render)
- Display via `Image(filename=str(path), width=N)` — typical widths: 500–900
- Save to `ASSETS_DIR` (publication figures) or `CACHE_DIR` (intermediate)
- Print confirmation with arrow: `f"Plot saved → {out_path}"`

### Scanpy Plot Integration

```python
sc.pl.umap(adata, color="leiden", ax=ax, show=False, title="...", frameon=False)
plt.tight_layout()
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(fig)
display(Image(filename=str(out_path), width=500))
```

- Always pass `show=False` to scanpy plot functions
- Wrap in a `fig, ax` context for full control

## Variable Naming

| Category | Convention | Examples |
|---|---|---|
| Path constants | ALL_CAPS | `PROJECT_ROOT`, `DATA_PROCESSED`, `CACHE_DIR`, `ASSETS_DIR` |
| Data objects | lowercase | `adata`, `graph`, `de_df`, `stabl_result` |
| Figure objects | `fig`, `ax`, `axes` | — |
| Output paths | descriptive snake_case | `vol1_path`, `bench_path`, `out_path` |
| Thresholds | ALL_CAPS | `FC_THR`, `FDR_THR` |

## Import Organization

Within the setup cell, order imports as:
1. Standard library (`sys`, `pathlib`, `warnings`, `time`)
2. Third-party (`scanpy`, `numpy`, `pandas`, `matplotlib`)
3. Local `src/` modules

Additional imports may appear in later code cells when they are only
needed for a specific section (e.g., `from IPython.display import Image, display`).

## Narrative Style

- **Scientific/biomedical tone** — accessible but precise
- Explain the *why* behind each analysis step, not just the mechanics
- Connect each step to its role in the broader pipeline
- Use domain-specific terminology with brief inline definitions when first introduced
- Keep markdown paragraphs to 1–3 sentences for readability
- Use bold for key conclusions or takeaways
- Cite references (paper DOIs, GEO accessions, tool names) where relevant

## Anti-patterns to Avoid

- `.ipynb` files (use percent-format `.py` only)
- Missing section numbers (every `##` must have `N.M` prefix)
- Code cells without summary `print()` at the end
- Inline matplotlib rendering without saving to file first
- Figures not closed with `plt.close(fig)`
- Markdown cells without explanatory prose (no bare headers)
- Notebook logic that should be extracted to `src/` (see function-writing skill)
- Unnumbered notebook filenames
- Missing Inputs/Outputs tables in the title cell
