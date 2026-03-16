"""Final pass to fix remaining SVG patches."""
import re

with open('visualization/export_svg_blueprint.py', 'rb') as f:
    text = f.read().decode('utf-8')

# 1. Canvas width - look for the actual pattern in the patched file
old_w = 'title_block_h = 50\n    width_px = _px(bw) + 2 * MARGIN + 60  # extra for dim lines'
new_w = 'title_block_h = 60\n    width_px = _px(bw) + 2 * MARGIN + 100 + LEGEND_W  # extra for dim + legend col'
if old_w in text:
    text = text.replace(old_w, new_w)
    print("Canvas width fixed.")
else:
    # Maybe title_block_h was already changed to 60 (by title-block patch)
    old_w2 = 'title_block_h = 60\n    width_px = _px(bw) + 2 * MARGIN + 60  # extra for dim lines'
    if old_w2 in text:
        text = text.replace(old_w2, 'title_block_h = 60\n    width_px = _px(bw) + 2 * MARGIN + 100 + LEGEND_W  # extra for dim + legend col')
        print("Canvas width fixed (alt path).")
    else:
        print("WARNING: old canvas width pattern not found. Printing neighbourhood...")
        idx = text.find('title_block_h')
        print(repr(text[idx:idx+200]))

# 2. Room label font size: find and fix font-size 10 -> 13 in _draw_room
# Find the room name label section specifically
old_label = '"font-size": "10",\n        "font-weight": "600",\n        "fill": "#263238",\n    })\n    label.text = room.room_type'
new_label = '"font-size": "13",\n        "font-weight": "700",\n        "fill": "#263238",\n    })\n    label.text = room.room_type'
if old_label in text:
    text = text.replace(old_label, new_label)
    print("Room label font fixed.")
else:
    print("WARNING: room label block not found verbatim.")

# 3. Area dim font size 8 -> 10 (there may be other font-size 8 uses)
# Only inside draw_room, near 'dim.text = f"{area'
old_area = '"font-size": "8",\n        "fill": "#546e7a",\n    })\n    dim.text = f"{area:.1f}'
new_area = '"font-size": "10",\n        "fill": "#546e7a",\n    })\n    dim.text = f"{area:.1f}'
if old_area in text:
    text = text.replace(old_area, new_area)
    print("Area dim font fixed.")
else:
    print("WARNING: area dim block not found verbatim.")

# 4. Title block scale bar: raw 5m -> 3m
old_bar = '    bar_x = width_px - MARGIN - _px(5)\n    bar_y = tb_y + 20\n    bar_w = _px(5)\n    SubElement(g, "line", {\n        "x1": f"{bar_x:.0f}", "y1": f"{bar_y:.0f}",\n        "x2": f"{bar_x + bar_w:.0f}", "y2": f"{bar_y:.0f}",\n        "stroke": "white", "stroke-width": "2",\n    })\n    for i in range(6):'
new_bar = '    bar_x = width_px - LEGEND_W - 20 - _px(3)\n    bar_y = tb_y + 30\n    bar_w = _px(3)\n    SubElement(g, "line", {\n        "x1": f"{bar_x:.0f}", "y1": f"{bar_y:.0f}",\n        "x2": f"{bar_x + bar_w:.0f}", "y2": f"{bar_y:.0f}",\n        "stroke": "white", "stroke-width": "2",\n    })\n    for i in range(4):'
if old_bar in text:
    text = text.replace(old_bar, new_bar)
    print("Scale bar fixed.")
else:
    print("WARNING: scale bar pattern not found verbatim.")
    idx = text.find('bar_x = width_px')
    print(repr(text[idx:idx+300]))

# 5. Legend block: check what we have
idx = text.find('# Legend')
leg_block = text[idx:idx+900]
if 'legend_x = width_px - LEGEND_W' not in leg_block:
    print("WARNING: legend_x LEGEND_W not found. Current legend block:")
    print(leg_block)
else:
    print("Legend block good.")

with open('visualization/export_svg_blueprint.py', 'wb') as f:
    f.write(text.encode('utf-8'))

print("\nSecond patch complete.")
