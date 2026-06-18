import os
import sys
import json
import pandas as pd

# Fallback: load KGAT token from kaggle.json if not in environment or is empty
if not os.environ.get('KAGGLE_API_TOKEN'):
    json_path = os.path.expanduser('~/.kaggle/kaggle.json')
    if os.path.exists(json_path):
        try:
            with open(json_path) as f:
                cfg = json.load(f)
            key = cfg.get('key', '')
            if key.startswith('KGAT_'):
                os.environ['KAGGLE_API_TOKEN'] = key
        except Exception:
            pass

def main():
    print("Importing kagglehub...", flush=True)
    import kagglehub
    
    comp = 'stanford-ribonanza-rna-folding'
    file_name = 'sample_submission.csv'
    
    print(f"Downloading '{file_name}' from '{comp}' using kagglehub...", flush=True)
    try:
        # Download a single file from the competition
        file_path = kagglehub.competition_download(comp, path=file_name)
        print(f"Download complete! File saved at: {file_path}", flush=True)
        
        # Verify file exists and read the first 10 records
        if os.path.exists(file_path):
            print("Reading the first 10 records...", flush=True)
            # Read only first 10 rows to be fast and memory efficient
            df = pd.read_csv(file_path, nrows=10)
            print("\n=== SUCCESS: FIRST 10 RECORDS FROM KAGGLE ===", flush=True)
            print(df.to_string(index=False), flush=True)
            print("============================================\n", flush=True)
        else:
            print(f"Error: File not found at {file_path}", file=sys.stderr, flush=True)
            
    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}", file=sys.stderr, flush=True)
        print("\nPrerequisites Check:", file=sys.stderr, flush=True)
        print("1. Ensure you have accepted the competition rules at: https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding/rules", file=sys.stderr, flush=True)
        print("2. Ensure your ~/.kaggle/kaggle.json contains valid credentials.", file=sys.stderr, flush=True)

if __name__ == '__main__':
    main()
