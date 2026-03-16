"""Write final targeted patches."""
import re

with open('visualization/export_svg_blueprint.py', 'rb') as f:
    text = f.read().decode('utf-8')

# Use \r\n for windows line endings
CRLF = '\r\n'
LF = '\n'

# Detect line ending
if '\r\n' in text:
    NL = '\r\n'
else:
    NL = '\n'

def r(old, new, t):
    if old in t:
        print(f"  Replaced: {old[:60]!r}...")
        return t.replace(old, new)
    else:
        print(f"  NOT FOUND: {old[:60]!r}...")
        return t

# 1. canvas width -- fix regardless of current title_block_h value
# Just replace the width_px line itself
text = re.sub(
    r'width_px = _px\(bw\) \+ 2 \* MARGIN \+ 60\s*#[^\n\r]*',
    'width_px = _px(bw) + 2 * MARGIN + 100 + LEGEND_W  # extra for dim + legend col',
    text
)
# fix title_block_h to 60
text = re.sub(r'title_block_h = 50(?=\r?\n)', 'title_block_h = 60', text)
print("Width_px and title_block_h done.")

# 2. Room label font 10 -> 13 (in draw_room only - near label.text)
# Match font-size 10 + font-weight + fill #263238 near label.text
text = re.sub(
    r'("font-size": "10",\s*"font-weight": "600",\s*"fill": "#263238",\s*\}[^)]*\)\s*label\.text)',
    lambda m: m.group(0).replace('"font-size": "10",', '"font-size": "13",').replace('"font-weight": "600",', '"font-weight": "700",'),
    text
)
print("Room label font done.")

# 3. Area dim font 8 -> 10 (near dim.text = f"{area}") 
text = re.sub(
    r'("font-size": "8",\s*"fill": "#546e7a",\s*\}[^)]*\)\s*dim\.text)',
    lambda m: m.group(0).replace('"font-size": "8",', '"font-size": "10",'),
    text
)
print("Area dim font done.")

# 4. Scale bar: MARGIN -> LEGEND_W based, 5 -> 3 segments
text = re.sub(
    r'bar_x = width_px - MARGIN - _px\(5\)',
    'bar_x = width_px - LEGEND_W - 20 - _px(3)',
    text
)
text = re.sub(
    r'bar_y = tb_y \+ 20',
    'bar_y = tb_y + 28',
    text
)
text = re.sub(
    r'bar_w = _px\(5\)',
    'bar_w = _px(3)',
    text
)
text = re.sub(r'for i in range\(6\):', 'for i in range(4):', text, count=1)
print("Scale bar done.")

# 5. Title font 14 -> 16
text = text.replace('"font-size": "14", "font-weight": "bold", "fill": "white",', '"font-size": "16", "font-weight": "bold", "fill": "white",')
# Info text font 9 -> 11
# only in title block area (near tb_y)
text = re.sub(
    r'("font-size": "9", "fill": "#b0bec5",\s*\}\)\s*info\.text)',
    lambda m: m.group(0).replace('"font-size": "9"', '"font-size": "11"'),
    text
)
print("Title text sizes done.")

# 6. Legend: replace old legend that still uses MARGIN-100 offset if it exists
text = re.sub(
    r'g_legend = SubElement\(svg, "g", \{"id": "legend",\s*"transform": f"translate\(\{width_px - MARGIN - 100:.0f\}, \{MARGIN:.0f\}\)"\}\)',
    'g_legend = SubElement(svg, "g", {"id": "legend", "transform": f"translate({legend_x:.0f}, {MARGIN:.0f})"})',
    text
)
# Make sure legend_x exists above that reference
if 'legend_x = width_px - LEGEND_W' not in text:
    text = re.sub(
        r'(# Legend.*?)(g_legend = )',
        lambda m: m.group(1) + 'legend_x = width_px - LEGEND_W + 6\n    ' + m.group(2),
        text,
        flags=re.DOTALL,
        count=1
    )
    print("legend_x added.")
print("Legend transform done.")

with open('visualization/export_svg_blueprint.py', 'wb') as f:
    f.write(text.encode('utf-8'))

print("\nFinal patch complete.")
