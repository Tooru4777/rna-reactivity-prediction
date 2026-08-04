"""Kaggle GPU entrypoint for leakage-safe Ribonanza ablation."""

import os
import subprocess
import sys
from pathlib import Path


REPOSITORY = "https://github.com/Tooru4777/rna-reactivity-prediction.git"
BRANCH = "agent/grouped-ribonanza-validation"
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
CHECKOUT = WORKING / "rna-reactivity-prediction"
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

    run(sys.executable, "-m", "pip", "install", "--quiet", "viennarna>=2.5")
    run("git", "clone", "--depth", "1", "--branch", BRANCH, REPOSITORY, CHECKOUT)

    max_samples = os.environ.get("RNA_MAX_SAMPLES", "1000")
    epochs = os.environ.get("RNA_EPOCHS", "15")
    run(
        sys.executable,
        "experiments/run_ablation.py",
        "--data", competition_input,
        "--max-samples", max_samples,
        "--epochs", epochs,
        "--output-dir", RESULTS,
        "--require-vienna",
        cwd=CHECKOUT,
    )
    print(f"Kaggle outputs are ready in {RESULTS}")


if __name__ == "__main__":
    main()
