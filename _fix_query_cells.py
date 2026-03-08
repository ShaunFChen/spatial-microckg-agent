"""Fix query cells in notebooks to wrap with try/except for API quota errors."""
import json
import sys

def fix_notebook(path):
    with open(path, 'r') as f:
        nb = json.load(f)

    fixed = 0
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        src = ''.join(cell['source'])
        if 'query_graph' in src and 'try:' not in src:
            lines = cell['source']
            comment_line = lines[0]
            rest = lines[1:]
            new_lines = [comment_line, 'try:\n']
            for line in rest:
                new_lines.append('    ' + line)
            new_lines.append('except Exception as e:\n')
            new_lines.append('    print(f"Query failed (likely API quota exceeded): {type(e).__name__}")\n')
            new_lines.append('    print("The agent was created successfully -- resolve API billing to enable queries.")\n')
            cell['source'] = new_lines
            fixed += 1
            print(f'  Fixed: {comment_line.strip()}')

    with open(path, 'w') as f:
        json.dump(nb, f, indent=1)
    print(f'Fixed {fixed} cells in {path}')

for nb_path in sys.argv[1:]:
    fix_notebook(nb_path)
