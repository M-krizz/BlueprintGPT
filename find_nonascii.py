"""Find and show all lines with non-ASCII chars."""
with open('visualization/export_svg_blueprint.py', 'rb') as f:
    raw = f.read()
text = raw.decode('latin-1')
for i, line in enumerate(text.split('\n'), 1):
    for ch in line:
        if ord(ch) > 127 and ch not in '\r':
            # Show only lines with non-standard chars (not m² which is fine)
            print(f"Line {i}: {line.rstrip()}")
            print(f"  Non-ASCII chars: {[(c, hex(ord(c))) for c in line if ord(c) > 127 and c != '\r']}")
            break
