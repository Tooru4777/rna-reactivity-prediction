#!/bin/bash
# =============================================
# Download Stanford Ribonanza RNA Folding data
# =============================================
#
# Prerequisites:
#   1. Accept the competition rules on Kaggle
#   2. Place your kaggle.json in ~/.kaggle/
#   3. Run: chmod 600 ~/.kaggle/kaggle.json
#
# Usage:
#   bash download_data.sh

set -euo pipefail

echo "Checking Kaggle API installation..."
pip install kaggle kagglehub -q

echo "Downloading dataset (this may take several minutes)..."

# Use the KAGGLE_USERNAME and KAGGLE_KEY environment variables
# or the ~/.kaggle/kaggle.json credentials file.
# NEVER hardcode API tokens in scripts.
python -c "
import kagglehub
import shutil
import os
import json
import sys
import zipfile
from pathlib import Path

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

print('Connecting to Kaggle API...')
try:
    path = kagglehub.competition_download('stanford-ribonanza-rna-folding')
    print(f'Download complete. Cache path: {path}')

    # Materialise the Kaggle cache into the repository's ignored dataset/ dir.
    os.makedirs('dataset', exist_ok=True)
    if os.path.isdir(path):
        shutil.copytree(path, 'dataset', dirs_exist_ok=True)
    elif zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            archive.extractall('dataset')
    else:
        raise RuntimeError(f'Unexpected Kaggle download format: {path}')

    candidates = sorted(Path('dataset').rglob('train_data.csv'))
    competition_candidates = [
        candidate for candidate in candidates
        if 'stanford-ribonanza-rna-folding' in candidate.parts
    ]
    if len(competition_candidates) == 1:
        train_csv = competition_candidates[0]
    elif len(candidates) == 1:
        train_csv = candidates[0]
    else:
        found = ', '.join(map(str, candidates)) or 'none'
        raise FileNotFoundError(
            f'Could not uniquely resolve downloaded train_data.csv; found: {found}'
        )
    print(f'Dataset ready: {train_csv}')
except Exception as e:
    print(f'Download failed: {e}')
    print('Make sure your Kaggle credentials are configured.')
    sys.exit(1)
"

echo "Done!"
