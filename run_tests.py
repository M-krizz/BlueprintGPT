import subprocess
with open('pytest_log.txt', 'w', encoding='utf-8') as f:
    result = subprocess.run([r'.\.venv\Scripts\python.exe', '-m', 'pytest', 'tests/', '-v'], capture_output=True, text=True)
    f.write(result.stdout)
    f.write("\n\nSTDERR:\n")
    f.write(result.stderr)
