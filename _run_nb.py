"""Execute a notebook programmatically and report errors."""
import sys
from pathlib import Path
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

nb_path = sys.argv[1]
with open(nb_path) as f:
    nb = nbformat.read(f, as_version=4)

print(f"Cells: {len(nb.cells)}")
for i, c in enumerate(nb.cells):
    print(f"  [{i+1}] {c.cell_type}: {len(c.source)} chars")

ep = ExecutePreprocessor(timeout=600, kernel_name="python3", allow_errors=True)
ep.preprocess(nb, {"metadata": {"path": str(Path(nb_path).parent)}})

with open(nb_path, "w") as f:
    nbformat.write(nb, f)

has_errors = False
for i, c in enumerate(nb.cells):
    for o in getattr(c, "outputs", []):
        if o.get("output_type") == "error":
            has_errors = True
            print(f"\n=== ERROR in cell {i+1} ({c.cell_type}) ===")
            print(f"  {o.get('ename')}: {o.get('evalue')}")
            for line in o.get("traceback", [])[:8]:
                # strip ANSI escape codes for readability
                import re
                clean = re.sub(r'\x1b\[[0-9;]*m', '', line)
                print("  ", clean)

if not has_errors:
    print("\nAll cells executed without errors.")
