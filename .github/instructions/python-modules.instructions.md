---
applyTo: "src/**/*.py"
description: "Enforces coding standards for Python modules in src/: type hints, Google-style docstrings, modular design."
---

# Python Module Standards

All functions in `src/` must follow these rules:

1. **Type hints** on all parameters and return values. Use `from __future__ import annotations`.
2. **Google-style docstrings** with Args, Returns, and Raises sections.
3. **Module-level docstring** explaining the module's purpose.
4. **Constants** in ALL_CAPS at module level.
5. **Private helpers** prefixed with `_`.
6. **Public API** exposed through `__init__.py` with `__all__`.

See `.github/skills/function-writing/SKILL.md` for full details and examples.
