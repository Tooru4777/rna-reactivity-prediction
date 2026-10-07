"""Kaggle GPU entrypoint for leakage-safe Ribonanza ablation."""

import os
import re
import subprocess
import sys
import csv
import json
import math
from pathlib import Path


REPOSITORY = "https://github.com/Tooru4777/rna-reactivity-prediction.git"
DEFAULT_REPOSITORY_REF = "__PIN_EXACT_COMMIT_BEFORE_SUBMISSION__"
REPOSITORY_REF = os.environ.get("RNA_REPOSITORY_REF", DEFAULT_REPOSITORY_REF)
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
CHECKOUT = Path("/tmp/rna-reactivity-prediction")
RESULTS = WORKING / "results"


def audit_smoke_results(results_dir, expected_sequences, expected_variants):
    """Fail the Kaggle job unless every non-reportable smoke invariant holds."""
    results_dir = Path(results_dir)
    with (results_dir / "run_manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("run_mode") != "smoke" or manifest.get("reportable") is not False:
        raise RuntimeError("Smoke outputs must be explicitly non-reportable")
    if int(manifest.get("filtered_unique_sequences", -1)) != int(expected_sequences):
        raise RuntimeError("Smoke cohort does not contain the requested sequence count")
    if set(manifest.get("variants", [])) != set(expected_variants):
        raise RuntimeError("Smoke variant set differs from the requested controls")

    with (results_dir / "split_report.json").open(encoding="utf-8") as handle:
        split_report = json.load(handle)
    if any(part.casefold() == "old" for part in Path(split_report["source"]).parts):
        raise RuntimeError("Smoke run resolved the superseded OLD data file")
    per_seed = split_report["splits_by_seed"]
    if set(per_seed) != {"42"}:
        raise RuntimeError("Smoke run must use validation seed 42 only")
    if set(per_seed["42"]) != {"random", "grouped", "clustered"}:
        raise RuntimeError("Smoke run did not execute all three split strategies")
    if any(per_seed["42"]["grouped"]["sequence_overlap"].values()):
        raise RuntimeError("Exact-grouped smoke split leaked sequences")
    if any(
        per_seed["42"]["clustered"]["similarity_cluster_overlap"].values()
    ):
        raise RuntimeError("Similarity-clustered smoke split leaked clusters")

    with (results_dir / "environment.json").open(encoding="utf-8") as handle:
        environment = json.load(handle)
    gpu_names = environment.get("gpu_names", [])
    if not gpu_names or not all("P100" in name for name in gpu_names):
        raise RuntimeError(f"Smoke run did not use the requested P100: {gpu_names}")

    with (results_dir / "ablation_summary.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    expected_rows = 3 * len(expected_variants)
    if len(rows) != expected_rows:
        raise RuntimeError(
            f"Smoke summary has {len(rows)} rows; expected {expected_rows}"
        )
    expected_summary_keys = {
        ("42", split_method, variant_id)
        for split_method in ("random", "grouped", "clustered")
        for variant_id in expected_variants
    }
    summary_keys = [
        (row.get("seed"), row.get("split_method"), row.get("variant_id"))
        for row in rows
    ]
    if len(set(summary_keys)) != len(summary_keys):
        raise RuntimeError("Smoke summary contains duplicate seed/split/variant rows")
    if set(summary_keys) != expected_summary_keys:
        raise RuntimeError("Smoke summary does not contain the complete split/variant grid")
    for row in rows:
        for column in ("best_cv_loss", "cv_macro_sequence_mae"):
            if not math.isfinite(float(row[column])):
                raise RuntimeError(f"Smoke metric is not finite: {column}")
        for column in ("test_mae", "test_macro_sequence_mae"):
            if column not in row:
                raise RuntimeError(f"Smoke summary is missing locked-test column: {column}")
            test_value = str(row.get(column, "")).strip()
            if test_value and not math.isnan(float(test_value)):
                raise RuntimeError(
                    f"Smoke run must not evaluate the locked test partition: {column}"
                )

    with (results_dir / "test_results.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        test_rows = list(csv.DictReader(handle))
    if test_rows:
        raise RuntimeError("Smoke run wrote locked-test result rows")

    with (results_dir / "paired_bootstrap_ci.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        bootstrap_rows = list(csv.DictReader(handle))
    expected_comparisons = {
        "vienna_real_vs_sequence_only",
        "vienna_real_vs_position_shuffled",
        "vienna_real_vs_zero_channels",
    }
    expected_bootstrap_keys = {
        ("42", split_method, "cv", comparison_id, metric)
        for split_method in ("random", "grouped", "clustered")
        for comparison_id in expected_comparisons
        for metric in ("nucleotide_weighted_mae", "macro_sequence_mae")
    }
    bootstrap_keys = [
        (
            row.get("seed"), row.get("split"), row.get("partition"),
            row.get("comparison_id"), row.get("metric"),
        )
        for row in bootstrap_rows
    ]
    if len(set(bootstrap_keys)) != len(bootstrap_keys):
        raise RuntimeError("Smoke bootstrap output contains duplicate result rows")
    if set(bootstrap_keys) != expected_bootstrap_keys:
        raise RuntimeError("Smoke bootstrap controls or split/metric rows are incomplete")
    for row in bootstrap_rows:
        for column in (
            "point_improvement", "ci_lower", "ci_upper",
            "probability_improvement",
        ):
            if not math.isfinite(float(row[column])):
                raise RuntimeError(f"Smoke bootstrap metric is not finite: {column}")
    print(
        "Smoke audit passed: current data, P100, CV-only splits and controls",
        flush=True,
    )


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
    """Find the corrected train_data.csv while rejecting superseded OLD data."""
    candidates = sorted(root.rglob("train_data.csv")) if root.is_dir() else []
    current_candidates = [
        path for path in candidates
        if not any(part.casefold() == "old" for part in path.parts)
    ]
    competition_candidates = [
        path for path in current_candidates
        if "stanford-ribonanza-rna-folding" in path.parts
    ]
    if len(competition_candidates) == 1:
        return competition_candidates[0]
    if len(current_candidates) == 1:
        return current_candidates[0]
    if not current_candidates and candidates:
        raise FileNotFoundError(
            "Only superseded OLD/train_data.csv was found; attach the corrected "
            "current competition file"
        )
    found = ", ".join(map(str, candidates)) or "none"
    raise FileNotFoundError(
        f"Could not uniquely locate Ribonanza train_data.csv under {root}; found: {found}"
    )


def main():
    competition_input = find_competition_input()
    print(f"Using competition data: {competition_input}", flush=True)

    if not re.fullmatch(r"[0-9a-fA-F]{40}", REPOSITORY_REF):
        raise RuntimeError(
            "Kaggle submissions must pin RNA_REPOSITORY_REF to an exact "
            "40-character commit SHA"
        )

    # Kaggle may allocate a Tesla P100 (sm_60). Its current torch 2.10 image
    gpu_names = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    ).strip().splitlines()
    if not gpu_names or not all("P100" in name for name in gpu_names):
        raise RuntimeError(f"Select P100 before starting this run: {gpu_names}")

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
    run("apt-get", "update", "-qq")
    run("apt-get", "install", "-y", "-qq", "mmseqs2")
    run(
        sys.executable, "-c",
        "import torch; "
        "assert torch.cuda.is_available(); "
        "names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]; "
        "assert names and all('P100' in name for name in names), names; "
        "caps=[torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())]; "
        "assert all(f'sm_{c[0]}{c[1]}' in torch.cuda.get_arch_list() for c in caps), "
        "(caps, torch.cuda.get_arch_list()); "
        "print(torch.__version__, names, caps); "
        "print((torch.ones(1, device='cuda') + 1).item())",
    )
    checkout_repository()

    max_sequences = os.environ.get("RNA_MAX_SEQUENCES", "256")
    epochs = os.environ.get("RNA_EPOCHS", "1")
    validation_seeds = os.environ.get("RNA_VALIDATION_SEEDS", "42").split()
    run_mode = os.environ.get("RNA_RUN_MODE", "smoke")
    split_method = os.environ.get("RNA_SPLIT_METHOD", "all")
    variants = os.environ.get(
        "RNA_VARIANTS",
        "transformer_seq4 transformer_vienna_real7 "
        "transformer_vienna_shuffled7 transformer_vienna_zero7",
    ).split()
    similarity_manifest = WORKING / "similarity_split_manifest.json"
    run(
        sys.executable,
        "experiments/build_similarity_manifest.py",
        "--data", competition_input,
        "--max-sequences", max_sequences,
        "--sample-seed", "42",
        "--output", similarity_manifest,
        "--work-dir", "/tmp/rna-mmseqs-work",
        "--threads", "1",
        cwd=CHECKOUT,
    )
    run(
        sys.executable,
        "experiments/run_ablation.py",
        "--data", competition_input,
        "--max-sequences", max_sequences,
        "--epochs", epochs,
        "--output-dir", RESULTS,
        "--split-method", split_method,
        "--similarity-manifest", similarity_manifest,
        "--variants", *variants,
        "--run-mode", run_mode,
        "--bootstrap-replicates", "10000",
        "--num-workers", "2",
        "--seeds", *validation_seeds,
        "--sample-seed", "42",
        cwd=CHECKOUT,
    )
    if run_mode == "smoke":
        audit_smoke_results(
            RESULTS,
            expected_sequences=int(max_sequences),
            expected_variants=variants,
        )
    print(f"Kaggle outputs are ready in {RESULTS}")


if __name__ == "__main__":
    main()
