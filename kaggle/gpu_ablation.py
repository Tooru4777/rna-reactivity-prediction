"""Kaggle GPU entrypoint for leakage-safe Ribonanza ablation."""

import os
import subprocess
import sys
from pathlib import Path


REPOSITORY = "https://github.com/Tooru4777/rna-reactivity-prediction.git"
BRANCH = "agent/grouped-ribonanza-validation"
COMPETITION_INPUT = Path("/kaggle/input/stanford-ribonanza-rna-folding/train_data.csv")
WORKING = Path("/kaggle/working")
CHECKOUT = WORKING / "rna-reactivity-prediction"
RESULTS = WORKING / "results"


def run(*args, cwd=None):
    print("+", " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=cwd, check=True)


def main():
    if not COMPETITION_INPUT.is_file():
        raise FileNotFoundError(f"Competition dataset is not mounted: {COMPETITION_INPUT}")

    run(sys.executable, "-m", "pip", "install", "--quiet", "viennarna>=2.5")
    run("git", "clone", "--depth", "1", "--branch", BRANCH, REPOSITORY, CHECKOUT)

    max_samples = os.environ.get("RNA_MAX_SAMPLES", "1000")
    epochs = os.environ.get("RNA_EPOCHS", "15")
    run(
        sys.executable,
        "experiments/run_ablation.py",
        "--data", COMPETITION_INPUT,
        "--max-samples", max_samples,
        "--epochs", epochs,
        "--output-dir", RESULTS,
        "--require-vienna",
        cwd=CHECKOUT,
    )
    print(f"Kaggle outputs are ready in {RESULTS}")


if __name__ == "__main__":
    main()
