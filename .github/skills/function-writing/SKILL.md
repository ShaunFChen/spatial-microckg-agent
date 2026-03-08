---
name: function-writing
description: "Coding standards for Python data-science modules. Use when: writing functions in src/, creating analysis utilities, reviewing module code quality. Enforces type hints, Google-style docstrings, and modular design."
argument-hint: "Module or function to create (e.g., 'preprocessing function for QC filtering')"
---

# Function Writing Standards

## When to Use
- Writing or editing any Python function in `src/`
- Creating new analysis utility modules
- Reviewing code for type safety and documentation quality

## Rules

### 1. Type Hints (Mandatory)
Every function must include complete Python type hints for all parameters
and the return value. Use `from __future__ import annotations` at the
top of each module for modern syntax.

```python
from __future__ import annotations

import anndata as ad
import pandas as pd

def filter_cells(
    adata: ad.AnnData,
    min_genes: int = 200,
    max_pct_mt: float = 5.0,
) -> ad.AnnData:
    ...
```

- Use `|` union syntax (e.g. `str | None`) with `from __future__ import annotations`
- Prefer concrete types (`list[str]`) over abstract (`Sequence[str]`) unless the function genuinely accepts any sequence
- Use `Any` sparingly — only for true pass-through kwargs

### 2. Docstrings (Mandatory — Google Style)
Every public function must have a Google-style docstring with:
- One-line summary (imperative mood)
- Extended description if non-obvious behavior exists
- `Args:` section listing every parameter with type and meaning
- `Returns:` section describing the return value
- `Raises:` section if the function raises specific exceptions

```python
def normalize_log(adata: ad.AnnData, target_sum: float = 1e4) -> ad.AnnData:
    """Normalize counts per cell and log-transform.

    Stores the raw counts in ``adata.raw`` before normalization so that
    they remain accessible for differential expression and plotting.

    Args:
        adata: AnnData with filtered raw counts.
        target_sum: Total counts each cell is normalized to.

    Returns:
        The AnnData with normalized, log-transformed values in ``X``
        and raw counts preserved in ``adata.raw``.
    """
```

### 3. Module Organization
- One module per concern: `data_loading.py`, `preprocessing.py`, `clustering.py`, `differential.py`, `plotting.py`
- Module-level docstring explaining purpose
- Constants (e.g. marker gene dicts, risk gene lists) defined at module level with ALL_CAPS naming
- Private helpers prefixed with `_` (no docstring requirement, but type hints still required)
- Public API exposed through `__init__.py` with `__all__`

### 4. Abstraction Principle
- Complex logic in notebooks must be extracted into `src/` functions
- Notebooks should contain only: imports, function calls, brief glue code, and markdown narrative
- Each function should do one thing well
- Prefer returning new objects over mutation; if mutation is used, document it explicitly

### 5. Error Handling
- Validate at module boundaries (e.g. check gene exists before indexing)
- Use descriptive `ValueError` / `KeyError` messages
- Don't add defensive checks for impossible internal states

### 6. Print Statements
- Use `print()` for progress reporting in long-running functions (not `logging`)
- Format: `f"  {description}: {count} items"` (2-space indent for sub-steps)
- Keep output concise — one line per major step

## Anti-patterns to Avoid
- Functions without type hints
- Docstrings with only a summary line (missing Args/Returns)
- NumPy-style docstrings (use Google style for consistency)
- Business logic in notebooks that should be in `src/`
- Overly generic function signatures that hide intent
