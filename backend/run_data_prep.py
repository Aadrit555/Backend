import sys
import traceback
from pathlib import Path

def main():
    if len(sys.argv) < 3:
        print("Usage: python run_data_prep.py <job_id> <raw_path>")
        sys.exit(1)
        
    job_id = sys.argv[1]
    raw_path = Path(sys.argv[2])
    out_path = raw_path.parent / f"{raw_path.stem}_sanitized.jsonl"
    structured_path = raw_path.parent / f"{raw_path.stem}_structured.jsonl"
    
    print(f"[DATA FACTORY] Starting data prep job {job_id} for {raw_path}")
    
    try:
        from backend.data_factory import run_pipeline
        from backend.adapters.unsloth import UnslothAdapter
        
        # 1. Pipeline (Extraction -> Data-Juicer -> Distilabel)
        run_pipeline(raw_path, structured_path)
        
        # 2. Sanitize
        adapter = UnslothAdapter()
        adapter._sanitize_pii(structured_path, out_path)
        
        print(f"[DATA FACTORY] Completed! Output saved to {out_path}")
        print(f"___FINAL_OUTPUT_PATH___:{out_path}")
    except Exception as e:
        print(f"[DATA FACTORY ERROR] {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
