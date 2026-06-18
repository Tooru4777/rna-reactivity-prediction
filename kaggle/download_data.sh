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

set -e

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

    # Copy to project dataset directory
    if os.path.exists('dataset'):
        shutil.rmtree('dataset')
    shutil.copytree(path, 'dataset')
    print('Data copied to dataset/')
except Exception as e:
    print(f'Download failed: {e}')
    print('Make sure your Kaggle credentials are configured.')
"

echo "Done!"
