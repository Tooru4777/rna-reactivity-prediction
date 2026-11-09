#!/bin/bash
set -euo pipefail

ENV_PYTHON="${HOME}/miniconda3/envs/rna_rl_env/bin/python"
KERNEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "$#" -eq 0 ]; then
    echo "Usage: bash kaggle/run_gpu_ablation.sh --commit <pushed-40-character-SHA> --mode smoke|development|convergence" >&2
    exit 2
fi
exec "${ENV_PYTHON}" "${KERNEL_DIR}/submit_gpu.py" "$@"
