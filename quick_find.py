
filename = r"src/v10/early_signal_engine.py"
with open(filename, 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "def _generate_signal" in line:
        print(f"Line {i+1}: {line.strip()}")
