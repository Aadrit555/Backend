import sys
from pathlib import Path
sys.path.append("/home/ojasvkushwah/Documents/Documents/unified")

from backend.data_factory import run_pipeline

txt_file = Path("test_pii_data.txt")
output_file = Path("factory_output.jsonl")

run_pipeline(txt_file, output_file)

print("Done. Contents of output file:")
print(output_file.read_text())
