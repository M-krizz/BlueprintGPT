"""Fix the m-squared strings and any other issues after ASCII cleaning."""
with open('visualization/export_svg_blueprint.py', 'r', encoding='utf-8') as f:
    text = f.read()

# The ASCII clean replaced "m²" with "m" (stripping ²).
# Also may have stripped "Â²" to nothing. Fix all area text references.
# Pattern: dim.text = f"{area:.1f} m" should become dim.text = f"{area:.1f} m\u00b2"
import re

# Fix area text in _draw_room
text = re.sub(
    r'dim\.text = f"\{area:\.1f\} m"',
    'dim.text = f"{area:.1f} m\u00b2"',
    text
)

# Fix title block info text  
text = re.sub(
    r'Total Area: \{total_area:\.1f\} m',
    'Total Area: {total_area:.1f} m\u00b2',
    text
)

# Also check for any remaining garbled sequences like Â
text = text.replace('\u00c2\u00b2', '\u00b2')  # cleanup double-encoded ²
text = text.replace(' m  |', ' m\u00b2  |')  # if the ² was fully stripped in title

with open('visualization/export_svg_blueprint.py', 'w', encoding='utf-8', newline='\r\n') as f:
    f.write(text)

# Verify syntax
try:
    compile(text, 'export_svg_blueprint.py', 'exec')
    print("Syntax check PASSED.")
except SyntaxError as e:
    print(f"Syntax check FAILED at line {e.lineno}: {e.msg}")
    # Print the offending line
    lines = text.split('\n')
    if e.lineno and e.lineno <= len(lines):
        start = max(0, e.lineno - 3)
        for i in range(start, min(len(lines), e.lineno + 2)):
            marker = ">>>" if i == e.lineno - 1 else "   "
            print(f"{marker} {i+1}: {lines[i]}")
