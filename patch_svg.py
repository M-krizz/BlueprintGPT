"""Patch export_svg_blueprint.py to scale up and fix legend placement."""
import re

with open('visualization/export_svg_blueprint.py', 'rb') as f:
    raw = f.read()

text = raw.decode('latin-1')

# 1. Scale 40 → 80
text = text.replace('SCALE = 40  # pixels per metre', 'SCALE = 80  # pixels per metre (scaled up for legibility)')

# 2. Margin 60 → 70
text = text.replace('MARGIN = 60  # px margin around boundary', 'MARGIN = 70  # px margin around boundary')

# 3. Thicker walls
text = text.replace('WALL_WIDTH = 3.0', 'WALL_WIDTH = 3.5')
text = text.replace('INNER_WALL_WIDTH = 1.5', 'INNER_WALL_WIDTH = 2.0')
text = text.replace('WINDOW_WIDTH_PX = 1.8', 'WINDOW_WIDTH_PX = 2.0\nLEGEND_W = 160  # reserved px on right for legend')

# 4. Room label font 10 → 13, weight 600 → 700
text = text.replace('"font-size": "10",\n        "font-weight": "600",', '"font-size": "13",\n        "font-weight": "700",')

# 5. Area dim font 8 → 10
text = re.sub(
    r'"font-size": "8",\s*"fill": "#546e7a"',
    '"font-size": "10",\n        "fill": "#546e7a"',
    text
)

# 6. Extra width for legend
text = text.replace(
    'title_block_h = 50\n    width_px = _px(bw) + 2 * MARGIN + 60  # extra for dim lines',
    'title_block_h = 60\n    width_px = _px(bw) + 2 * MARGIN + 100 + LEGEND_W  # extra for dim + legend col'
)

# 7. Move compass left of legend strip
text = text.replace(
    '_draw_compass(svg, width_px - 40, 40)',
    '_draw_compass(svg, width_px - LEGEND_W - 40, 40)'
)

# 8. Title block height and title font
text = text.replace('    tb_h = 50\n    tb_y = height_px - tb_h', '    tb_h = 60\n    tb_y = height_px - tb_h')
text = text.replace(
    '"font-size": "14", "font-weight": "bold", "fill": "white",\n    })\n    t.text = title',
    '"font-size": "16", "font-weight": "bold", "fill": "white",\n    })\n    t.text = title'
)
text = text.replace(
    '"font-size": "9", "fill": "#b0bec5",\n    })\n    info.text',
    '"font-size": "11", "fill": "#b0bec5",\n    })\n    info.text'
)
# scale bar: 5 segments → 3 segments
text = text.replace(
    "bar_x = width_px - MARGIN - _px(5)\n    bar_y = tb_y + 20\n    bar_w = _px(5)\n    SubElement(g, \"line\", {\n        \"x1\": f\"{bar_x:.0f}\", \"y1\": f\"{bar_y:.0f}\",\n        \"x2\": f\"{bar_x + bar_w:.0f}\", \"y2\": f\"{bar_y:.0f}\",\n        \"stroke\": \"white\", \"stroke-width\": \"2\",\n    })\n    for i in range(6):",
    "bar_x = width_px - LEGEND_W - 20 - _px(3)\n    bar_y = tb_y + 25\n    bar_w = _px(3)\n    SubElement(g, \"line\", {\n        \"x1\": f\"{bar_x:.0f}\", \"y1\": f\"{bar_y:.0f}\",\n        \"x2\": f\"{bar_x + bar_w:.0f}\", \"y2\": f\"{bar_y:.0f}\",\n        \"stroke\": \"white\", \"stroke-width\": \"2\",\n    })\n    for i in range(4):"
)
text = text.replace(
    '"text-anchor": "middle", "font-size": "7", "fill": "#b0bec5"',
    '"text-anchor": "middle", "font-size": "9", "fill": "#b0bec5"'
)

# 9. Rebuild legend block with larger swatches + header
old_legend = '''    # Legend
    g_legend = SubElement(svg, "g", {"id": "legend",
                                      "transform": f"translate({width_px - MARGIN - 100:.0f}, {MARGIN:.0f})"})
    ly = 0
    seen_types = set()
    for room in building.rooms:
        if room.room_type in seen_types:
            continue
        seen_types.add(room.room_type)
        fill = ROOM_FILL.get(room.room_type, DEFAULT_FILL)
        SubElement(g_legend, "rect", {
            "x": "0", "y": f"{ly}",
            "width": "12", "height": "12",
            "fill": fill, "stroke": "#37474f", "stroke-width": "0.5",
        })
        lt = SubElement(g_legend, "text", {
            "x": "16", "y": f"{ly + 10}",
            "font-size": "8", "fill": "#263238",
        })
        lt.text = room.room_type
        ly += 16'''

new_legend = '''    # Legend -- top-right outside the plan
    legend_x = width_px - LEGEND_W + 6
    g_legend = SubElement(svg, "g", {"id": "legend",
                                      "transform": f"translate({legend_x:.0f}, {MARGIN:.0f})"})
    hdr = SubElement(g_legend, "text", {
        "x": "0", "y": "14",
        "font-size": "13", "font-weight": "bold", "fill": "#263238",
    })
    hdr.text = "Legend"
    ly = 28
    seen_types = set()
    for room in building.rooms:
        if room.room_type in seen_types:
            continue
        seen_types.add(room.room_type)
        fill = ROOM_FILL.get(room.room_type, DEFAULT_FILL)
        SubElement(g_legend, "rect", {
            "x": "0", "y": f"{ly}",
            "width": "20", "height": "20",
            "fill": fill, "stroke": "#37474f", "stroke-width": "1",
        })
        lt = SubElement(g_legend, "text", {
            "x": "28", "y": f"{ly + 14}",
            "font-size": "12", "fill": "#263238",
        })
        lt.text = room.room_type
        ly += 28'''

if old_legend in text:
    text = text.replace(old_legend, new_legend)
    print("Legend block replaced successfully.")
else:
    print("WARNING: legend block not found verbatim -- check file.")

with open('visualization/export_svg_blueprint.py', 'wb') as f:
    f.write(text.encode('utf-8'))

print("Patch complete.")
