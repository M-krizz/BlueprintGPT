"""
Properly fix encoding: read as bytes, decode as cp1252 (Windows),
replace non-ASCII comment chars with ASCII equivalents,
but KEEP legitimate chars like m-squared in code strings.
Then write back as clean UTF-8 with encoding cookie.
"""
with open('visualization/export_svg_blueprint.py', 'rb') as f:
    raw = f.read()

# cp1252 is the most correct encoding for Windows-created Python files
text = raw.decode('cp1252')

# Replace all non-ASCII chars in the file EXCEPT the ones we need (² = \u00b2)
cleaned = []
for ch in text:
    cp = ord(ch)
    if cp <= 127:
        cleaned.append(ch)
    elif ch == '\u00b2':  # keep superscript 2
        cleaned.append(ch)
    elif ch in ('\u2013', '\u2014'):  # en-dash, em-dash -> regular dash
        cleaned.append('-')
    elif ch in ('\u2018', '\u2019'):  # curly single quotes -> straight
        cleaned.append("'")
    elif ch in ('\u201c', '\u201d'):  # curly double quotes -> straight
        cleaned.append('"')
    elif ch == '\u2026':  # ellipsis
        cleaned.append('...')
    elif ch == '\u2190':  # left arrow
        cleaned.append('<-')
    elif ch == '\u2192':  # right arrow
        cleaned.append('->')
    elif ch in ('\u2500', '\u2550'):  # box drawing
        cleaned.append('-')
    elif ch == '\u00a0':  # non-breaking space
        cleaned.append(' ')
    elif ch == '\u00bb':  # right-pointing double angle
        cleaned.append('>>')
    elif ch == '\u00ab':  # left-pointing double angle
        cleaned.append('<<')
    elif ch == '\u00a6':  # broken bar
        cleaned.append('|')
    elif ch == '\u00b0':  # degree symbol
        cleaned.append(' degrees')
    elif ch == '\u00c2':  # stray A-circumflex (from UTF-8 double encoding)
        pass  # skip it
    elif ch == '\u0192':  # f with hook
        cleaned.append('f')
    elif ch == '\u00e2':  # a-circumflex
        cleaned.append('-')  # usually part of box drawing
    elif ch == '\u0080' or ch == '\u0081' or ch == '\u0082':
        pass  # control chars, skip
    elif ch == '\u0093' or ch == '\u0094':
        cleaned.append('"')  # curly quotes in cp1252
    elif ch == '\u0096' or ch == '\u0097':
        cleaned.append('-')  # dashes in cp1252
    elif ch == '\u0085':
        cleaned.append('...')  # NEL
    elif ch == '\u0099':
        cleaned.append('(TM)')
    else:
        # For any other non-ASCII, just skip it
        pass

result = ''.join(cleaned)

# Ensure we have the encoding cookie at the top
if '# -*- coding' not in result[:100]:
    result = '# -*- coding: utf-8 -*-\n' + result

with open('visualization/export_svg_blueprint.py', 'w', encoding='utf-8', newline='\r\n') as f:
    f.write(result)

# Verify
try:
    compile(result, 'export_svg_blueprint.py', 'exec')
    print("Syntax check PASSED!")
except SyntaxError as e:
    print(f"Syntax check FAILED: line {e.lineno}, {e.msg}")
    lines = result.split('\n')
    if e.lineno and e.lineno <= len(lines):
        for i in range(max(0, e.lineno-3), min(len(lines), e.lineno+2)):
            m = ">>>" if i == e.lineno-1 else "   "
            print(f"{m} {i+1}: {lines[i]}")

print("Done.")
