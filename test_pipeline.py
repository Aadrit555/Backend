import sys
from pathlib import Path
sys.path.append("/home/ojasvkushwah/Documents/Documents/unified")

from backend.adapters.unsloth import UnslothAdapter

adapter = UnslothAdapter()
txt_file = Path("test_pii_data.txt")
structured_file = Path("structured_test.jsonl")
sanitized_file = Path("sanitized_test.jsonl")

print("Structuring...")
adapter._structure_text_to_jsonl(txt_file, structured_file)

print("Sanitizing...")
adapter._sanitize_pii(structured_file, sanitized_file)

print("Done. Contents of sanitized file:")
print(sanitized_file.read_text())
