"""Kaggle GPU entrypoint for leakage-safe Ribonanza ablation."""

import os
import subprocess
import sys
from pathlib import Path


REPOSITORY = "https://github.com/Tooru4777/rna-reactivity-prediction.git"
BRANCH = "codex/exact-length-results"
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
CHECKOUT = Path("/tmp/rna-reactivity-prediction")
RESULTS = WORKING / "results"


def run(*args, cwd=None):
    print("+", " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=cwd, check=True)


def find_competition_input(root=KAGGLE_INPUT_ROOT):
    """Find train_data.csv even when Kaggle nests it under OLD/."""
    candidates = sorted(root.rglob("train_data.csv")) if root.is_dir() else []
    competition_candidates = [
        path for path in candidates
        if "stanford-ribonanza-rna-folding" in path.parts
    ]
    if competition_candidates:
        return competition_candidates[0]
    if len(candidates) == 1:
        return candidates[0]
    found = ", ".join(map(str, candidates)) or "none"
    raise FileNotFoundError(
        f"Could not uniquely locate Ribonanza train_data.csv under {root}; found: {found}"
    )


def main():
    competition_input = find_competition_input()
    print(f"Using competition data: {competition_input}", flush=True)

    # Kaggle may allocate a Tesla P100 (sm_60). Its current torch 2.10 image
    # starts at sm_70, so pin the last CUDA 11.8 wheel family that supports it.
    run(
        sys.executable, "-m", "pip", "install", "--quiet", "--force-reinstall",
        "torch==2.7.1", "--index-url", "https://download.pytorch.org/whl/cu118",
    )
    run(sys.executable, "-m", "pip", "install", "--quiet", "viennarna>=2.5")
    run(
        sys.executable, "-c",
        "import torch; "
        "assert torch.cuda.is_available(); "
        "caps=[torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())]; "
        "assert all(f'sm_{c[0]}{c[1]}' in torch.cuda.get_arch_list() for c in caps), "
        "(caps, torch.cuda.get_arch_list()); "
        "print(torch.__version__, [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], caps); "
        "print((torch.ones(1, device='cuda') + 1).item())",
    )
    run("git", "clone", "--depth", "1", "--branch", BRANCH, REPOSITORY, CHECKOUT)

    max_sequences = os.environ.get("RNA_MAX_SEQUENCES", "1000")
    epochs = os.environ.get("RNA_EPOCHS", "15")
    validation_seeds = os.environ.get("RNA_VALIDATION_SEEDS", "42 123 2026").split()
    run(
        sys.executable,
        "experiments/run_ablation.py",
        "--data", competition_input,
        "--max-sequences", max_sequences,
        "--epochs", epochs,
        "--output-dir", RESULTS,
        "--require-vienna",
        "--split-method", "both",
        "--seeds", *validation_seeds,
        "--sample-seed", "42",
        cwd=CHECKOUT,
    )
    print(f"Kaggle outputs are ready in {RESULTS}")


if __name__ == "__main__":
    main()
