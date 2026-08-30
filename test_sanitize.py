import sys
from pathlib import Path
sys.path.append("/home/ojasvkushwah/Documents/Documents/unified")

from backend.adapters.unsloth import UnslothAdapter

adapter = UnslothAdapter()
input_file = Path("factory_output.jsonl")
sanitized_file = Path("factory_sanitized.jsonl")

adapter._sanitize_pii(input_file, sanitized_file)

print("Sanitized Contents:")
print(sanitized_file.read_text())
