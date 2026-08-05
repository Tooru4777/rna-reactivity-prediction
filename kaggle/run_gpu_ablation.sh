#!/bin/bash
set -euo pipefail

ENV_PYTHON="${HOME}/miniconda3/envs/rna_rl_env/bin/python"
KAGGLE_BIN="${HOME}/miniconda3/envs/rna_rl_env/bin/kaggle"
KERNEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL_ID="tooru4777/rna-reactivity-grouped-ablation"
OUTPUT_DIR="${KERNEL_DIR}/outputs"
TIMEOUT_SECONDS="${KAGGLE_TIMEOUT_SECONDS:-2400}"

"${ENV_PYTHON}" -m json.tool "${KERNEL_DIR}/kernel-metadata.json" >/dev/null
"${KAGGLE_BIN}" kernels push -p "${KERNEL_DIR}" -t "${TIMEOUT_SECONDS}"

echo "Submitted ${KERNEL_ID}. Check with:"
echo "  ${KAGGLE_BIN} kernels status ${KERNEL_ID}"
echo "Download completed outputs with:"
echo "  ${KAGGLE_BIN} kernels output ${KERNEL_ID} -p ${OUTPUT_DIR} --force"
