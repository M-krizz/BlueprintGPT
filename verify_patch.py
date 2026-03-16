"""Verify SVG patch status."""
with open('visualization/export_svg_blueprint.py', 'rb') as f:
    text = f.read().decode('utf-8')

checks = {
    'SCALE=80': 'SCALE = 80',
    'LEGEND_W=160': 'LEGEND_W = 160',
    'legend_x = width_px - LEGEND_W': 'legend_x = width_px - LEGEND_W',
    'legend header': 'hdr.text = "Legend"',
    'width_px has LEGEND_W': 'LEGEND_W  # extra for dim + legend col',
    'Compass moved': 'width_px - LEGEND_W - 40, 40',
    'scale bar 3m': 'bar_x = width_px - LEGEND_W - 20',
    'font-size 13 label': '"font-size": "13"',
}
for name, token in checks.items():
    print(f'{name}: {token in text}')
