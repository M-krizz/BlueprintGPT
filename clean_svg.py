"""Strip all non-ASCII bytes from export_svg_blueprint.py and rewrite as clean UTF-8."""
import re

with open('visualization/export_svg_blueprint.py', 'rb') as f:
    raw = f.read()

# Decode, replacing any bad bytes
text = raw.decode('utf-8', errors='replace')

# Replace all non-printable / non-ASCII chars (except common ones like newline, tab)
# Keep only ASCII printable + \r \n \t
cleaned = []
for ch in text:
    cp = ord(ch)
    if cp == 0x0A or cp == 0x0D or cp == 0x09:  # newline, CR, tab
        cleaned.append(ch)
    elif 0x20 <= cp <= 0x7E:  # printable ASCII
        cleaned.append(ch)
    elif cp == 0xB2:  # superscript 2 for m^2
        cleaned.append(ch)
    elif ch == '\ufffd':  # replacement character
        pass  # skip
    elif cp > 0x7E:
        pass  # skip non-ASCII
    else:
        pass  # skip other control chars

result = ''.join(cleaned)

# Fix any doubled/mangled comment separators that lost their box-drawing chars
# Replace sequences of spaces where box chars were with simple dashes
result = re.sub(r'#\s{5,}\n', '# ' + '-' * 70 + '\n', result)

with open('visualization/export_svg_blueprint.py', 'w', encoding='utf-8', newline='\r\n') as f:
    f.write(result)

print("File cleaned and saved as UTF-8.")

# Verify it imports
try:
    compile(result, 'export_svg_blueprint.py', 'exec')
    print("Syntax check PASSED.")
except SyntaxError as e:
    print(f"Syntax check FAILED: {e}")
