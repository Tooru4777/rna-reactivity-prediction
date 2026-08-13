"""Kaggle GPU entrypoint for leakage-safe Ribonanza ablation."""

import os
import re
import subprocess
import sys
from pathlib import Path


REPOSITORY = "https://github.com/Tooru4777/rna-reactivity-prediction.git"
DEFAULT_REPOSITORY_REF = "main"
REPOSITORY_REF = os.environ.get("RNA_REPOSITORY_REF", DEFAULT_REPOSITORY_REF)
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
CHECKOUT = Path("/tmp/rna-reactivity-prediction")
RESULTS = WORKING / "results"


def run(*args, cwd=None):
    print("+", " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=cwd, check=True)


def checkout_repository(repository=REPOSITORY, repository_ref=REPOSITORY_REF,
                        checkout=CHECKOUT):
    """Checkout a branch/tag or an exact 40-character commit SHA."""
    checkout = Path(checkout)
    print(f"Checking out repository ref: {repository_ref}", flush=True)
    if re.fullmatch(r"[0-9a-fA-F]{40}", repository_ref):
        run("git", "init", checkout)
        run("git", "remote", "add", "origin", repository, cwd=checkout)
        run("git", "fetch", "--depth", "1", "origin", repository_ref, cwd=checkout)
        run("git", "checkout", "--detach", "FETCH_HEAD", cwd=checkout)
    else:
        run(
            "git", "clone", "--depth", "1", "--branch", repository_ref,
            repository, checkout,
        )

    resolved_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
    ).strip()
    if re.fullmatch(r"[0-9a-fA-F]{40}", repository_ref):
        if resolved_commit.lower() != repository_ref.lower():
            raise RuntimeError(
                f"Requested commit {repository_ref}, checked out {resolved_commit}"
            )
    print(f"Checked out commit: {resolved_commit}", flush=True)
    return resolved_commit


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
    run(
        sys.executable, "-m", "pip", "install", "--quiet",
        "pandas==2.3.3", "numpy==2.2.5", "matplotlib==3.10.9",
        "PyYAML==6.0.3", "ViennaRNA==2.7.2",
    )
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
    checkout_repository()

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
        "--split-method", "both",
        "--seeds", *validation_seeds,
        "--sample-seed", "42",
        cwd=CHECKOUT,
    )
    print(f"Kaggle outputs are ready in {RESULTS}")


if __name__ == "__main__":
    main()
