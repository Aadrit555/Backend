import sys
from pathlib import Path
sys.path.append("/home/ojasvkushwah/Documents/Documents/unified")

from backend.data_factory import run_pipeline
from backend.adapters.unsloth import UnslothAdapter

txt_file = Path("employee_handbook.txt")
structured_file = Path("employee_structured.jsonl")
sanitized_file = Path("employee_sanitized.jsonl")

# 1. Structure
print("--- RUNNING FACTORY (STRUCTURING) ---")
run_pipeline(txt_file, structured_file)

# 2. Sanitize
print("\n--- RUNNING PRESIDIO (SANITIZING) ---")
adapter = UnslothAdapter()
adapter._sanitize_pii(structured_file, sanitized_file)

print("\n--- FINAL OUTPUT ---")
print(sanitized_file.read_text())
