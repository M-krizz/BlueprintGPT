"""
Bytes-level SVG renderer patch.
Reads the file as raw bytes, applies replacements on byte strings,
writes back as bytes - no encoding conversion needed.
"""

with open('visualization/export_svg_blueprint.py', 'rb') as f:
    data = f.read()

def rep(old_str, new_str, d):
    old_b = old_str.encode('latin-1')
    new_b = new_str.encode('latin-1')
    if old_b in d:
        d = d.replace(old_b, new_b, 1)
        return d, True
    return d, False

# 1. SCALE 40 -> 80
data, ok = rep('SCALE = 40  # pixels per metre', 'SCALE = 80  # pixels per metre', data)
print(f"1. SCALE: {ok}")

# 2. MARGIN 60 -> 70
data, ok = rep('MARGIN = 60  # px margin around boundary', 'MARGIN = 70  # px margin around boundary', data)
print(f"2. MARGIN: {ok}")

# 3. LEGEND_W constant
data, ok = rep('WINDOW_WIDTH_PX = 1.8', 'WINDOW_WIDTH_PX = 1.8\r\nLEGEND_W = 160  # reserved px on right for legend', data)
print(f"3. LEGEND_W: {ok}")

# 4. Room label font
data, ok = rep('"font-size": "10",\r\n        "font-weight": "600",', '"font-size": "13",\r\n        "font-weight": "700",', data)
print(f"4. Label font: {ok}")

# 5. Area dim font (only the one followed by dim.text)
old5 = b'"font-size": "8",\r\n        "fill": "#546e7a",\r\n    })\r\n    dim.text'
new5 = b'"font-size": "10",\r\n        "fill": "#546e7a",\r\n    })\r\n    dim.text'
if old5 in data:
    data = data.replace(old5, new5, 1)
    print("5. Area font: True")
else:
    print("5. Area font: False")

# 6. Title block height
data, ok = rep('    tb_h = 50\r\n', '    tb_h = 60\r\n', data)
print(f"6. tb_h: {ok}")

# 7. Title font 14 -> 16
data, ok = rep('"font-size": "14", "font-weight": "bold"', '"font-size": "16", "font-weight": "bold"', data)
print(f"7. Title font: {ok}")

# 8. Info font 9 -> 11
old8 = b'"font-size": "9", "fill": "#b0bec5",\r\n    })\r\n    info.text'
new8 = b'"font-size": "11", "fill": "#b0bec5",\r\n    })\r\n    info.text'
if old8 in data:
    data = data.replace(old8, new8, 1)
    print("8. Info font: True")
else:
    print("8. Info font: False")

# 9. Scale bar 5m -> 3m
data, ok = rep('bar_x = width_px - MARGIN - _px(5)', 'bar_x = width_px - LEGEND_W - 20 - _px(3)', data)
print(f"9a. Scale bar x: {ok}")
data, ok = rep('bar_y = tb_y + 20', 'bar_y = tb_y + 28', data)
print(f"9b. Scale bar y: {ok}")
data, ok = rep('bar_w = _px(5)', 'bar_w = _px(3)', data)
print(f"9c. Scale bar w: {ok}")
data, ok = rep('for i in range(6):', 'for i in range(4):', data)
print(f"9d. Scale range: {ok}")
data, ok = rep('"font-size": "7", "fill": "#b0bec5"', '"font-size": "9", "fill": "#b0bec5"', data)
print(f"9e. Scale font: {ok}")

# 10. Canvas width
data, ok = rep(
    'width_px = _px(bw) + 2 * MARGIN + 60  # extra for dim lines',
    'width_px = _px(bw) + 2 * MARGIN + 100 + LEGEND_W  # extra for dim + legend',
    data
)
print(f"10. Canvas width: {ok}")

# 11. Compass position
data, ok = rep('_draw_compass(svg, width_px - 40, 40)', '_draw_compass(svg, width_px - LEGEND_W - 40, 40)', data)
print(f"11. Compass: {ok}")

# 12. Legend block
old_leg = (
    b'    # Legend\r\n'
    b'    g_legend = SubElement(svg, "g", {"id": "legend",\r\n'
    b'                                      "transform": f"translate({width_px - MARGIN - 100:.0f}, {MARGIN:.0f})"})\r\n'
    b'    ly = 0\r\n'
    b'    seen_types = set()\r\n'
    b'    for room in building.rooms:\r\n'
    b'        if room.room_type in seen_types:\r\n'
    b'            continue\r\n'
    b'        seen_types.add(room.room_type)\r\n'
    b'        fill = ROOM_FILL.get(room.room_type, DEFAULT_FILL)\r\n'
    b'        SubElement(g_legend, "rect", {\r\n'
    b'            "x": "0", "y": f"{ly}",\r\n'
    b'            "width": "12", "height": "12",\r\n'
    b'            "fill": fill, "stroke": "#37474f", "stroke-width": "0.5",\r\n'
    b'        })\r\n'
    b'        lt = SubElement(g_legend, "text", {\r\n'
    b'            "x": "16", "y": f"{ly + 10}",\r\n'
    b'            "font-size": "8", "fill": "#263238",\r\n'
    b'        })\r\n'
    b'        lt.text = room.room_type\r\n'
    b'        ly += 16'
)

new_leg = (
    b'    # Legend -- top-right, outside the plan\r\n'
    b'    legend_x = width_px - LEGEND_W + 6\r\n'
    b'    g_legend = SubElement(svg, "g", {"id": "legend",\r\n'
    b'                                      "transform": f"translate({legend_x:.0f}, {MARGIN:.0f})"})\r\n'
    b'    hdr = SubElement(g_legend, "text", {\r\n'
    b'        "x": "0", "y": "14",\r\n'
    b'        "font-size": "13", "font-weight": "bold", "fill": "#263238",\r\n'
    b'    })\r\n'
    b'    hdr.text = "Legend"\r\n'
    b'    ly = 28\r\n'
    b'    seen_types = set()\r\n'
    b'    for room in building.rooms:\r\n'
    b'        if room.room_type in seen_types:\r\n'
    b'            continue\r\n'
    b'        seen_types.add(room.room_type)\r\n'
    b'        fill = ROOM_FILL.get(room.room_type, DEFAULT_FILL)\r\n'
    b'        SubElement(g_legend, "rect", {\r\n'
    b'            "x": "0", "y": f"{ly}",\r\n'
    b'            "width": "20", "height": "20",\r\n'
    b'            "fill": fill, "stroke": "#37474f", "stroke-width": "1",\r\n'
    b'        })\r\n'
    b'        lt = SubElement(g_legend, "text", {\r\n'
    b'            "x": "28", "y": f"{ly + 14}",\r\n'
    b'            "font-size": "12", "fill": "#263238",\r\n'
    b'        })\r\n'
    b'        lt.text = room.room_type\r\n'
    b'        ly += 28'
)

if old_leg in data:
    data = data.replace(old_leg, new_leg, 1)
    print("12. Legend: True")
else:
    print("12. Legend: False")

with open('visualization/export_svg_blueprint.py', 'wb') as f:
    f.write(data)

# Verify by trying to import
import subprocess
r = subprocess.run(['python', '-c', 'from visualization.export_svg_blueprint import render_svg_blueprint; print("Import OK")'],
                   capture_output=True, text=True, cwd='.')
print(f"Import test: {r.stdout.strip()}")
if r.returncode != 0:
    print(f"Import error: {r.stderr[-300:]}")

print("\nAll patches applied.")
