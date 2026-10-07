"""Tests for the Kaggle GPU entrypoint."""

from pathlib import Path
import importlib.util
import csv
import json

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "kaggle" / "gpu_ablation.py"
SPEC = importlib.util.spec_from_file_location("gpu_ablation", MODULE_PATH)
gpu_ablation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gpu_ablation)
find_competition_input = gpu_ablation.find_competition_input
checkout_repository = gpu_ablation.checkout_repository
audit_smoke_results = gpu_ablation.audit_smoke_results


def test_runner_requires_exact_ref_and_pins_smoke_dependencies():
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert gpu_ablation.DEFAULT_REPOSITORY_REF == (
        "__PIN_EXACT_COMMIT_BEFORE_SUBMISSION__"
    )
    assert '"ViennaRNA==2.7.2"' in source
    assert '"mmseqs2"' in source
    assert 'os.environ.get("RNA_MAX_SEQUENCES", "256")' in source
    assert 'os.environ.get("RNA_EPOCHS", "1")' in source
    assert 'os.environ.get("RNA_RUN_MODE", "smoke")' in source


def test_rejects_train_data_nested_under_old(tmp_path):
    expected = (
        tmp_path
        / "stanford-ribonanza-rna-folding"
        / "OLD"
        / "train_data.csv"
    )
    expected.parent.mkdir(parents=True)
    expected.touch()

    with pytest.raises(FileNotFoundError, match="Only superseded OLD"):
        find_competition_input(tmp_path)


def test_finds_current_train_data_and_ignores_old(tmp_path):
    root = tmp_path / "stanford-ribonanza-rna-folding"
    current = root / "train_data.csv"
    legacy = root / "OLD" / "train_data.csv"
    legacy.parent.mkdir(parents=True)
    current.touch()
    legacy.touch()

    assert find_competition_input(tmp_path) == current


def test_missing_competition_data_has_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Could not uniquely locate"):
        find_competition_input(tmp_path)


def test_checkout_repository_uses_branch_clone(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        gpu_ablation, "run",
        lambda *args, cwd=None: calls.append((args, cwd)),
    )
    monkeypatch.setattr(
        gpu_ablation.subprocess, "check_output",
        lambda *args, **kwargs: "a" * 40 + "\n",
    )
    checkout = tmp_path / "checkout"

    resolved = checkout_repository("repository", "feature-branch", checkout)

    assert resolved == "a" * 40
    assert calls == [((
        "git", "clone", "--depth", "1", "--branch", "feature-branch",
        "repository", checkout,
    ), None)]


def test_checkout_repository_fetches_exact_commit(monkeypatch, tmp_path):
    calls = []
    commit = "0123456789abcdef" * 2 + "01234567"
    monkeypatch.setattr(
        gpu_ablation, "run",
        lambda *args, cwd=None: calls.append((args, cwd)),
    )
    monkeypatch.setattr(
        gpu_ablation.subprocess, "check_output",
        lambda *args, **kwargs: commit + "\n",
    )
    checkout = tmp_path / "checkout"

    resolved = checkout_repository("repository", commit, checkout)

    assert resolved == commit
    assert calls == [
        (("git", "init", checkout), None),
        (("git", "remote", "add", "origin", "repository"), checkout),
        (("git", "fetch", "--depth", "1", "origin", commit), checkout),
        (("git", "checkout", "--detach", "FETCH_HEAD"), checkout),
    ]


def test_checkout_repository_rejects_wrong_commit(monkeypatch, tmp_path):
    requested = "1" * 40
    monkeypatch.setattr(gpu_ablation, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        gpu_ablation.subprocess, "check_output",
        lambda *args, **kwargs: "2" * 40 + "\n",
    )

    with pytest.raises(RuntimeError, match="Requested commit"):
        checkout_repository("repository", requested, tmp_path / "checkout")


def test_smoke_audit_accepts_current_p100_complete_outputs(tmp_path):
    variants = [
        "transformer_seq4",
        "transformer_vienna_real7",
        "transformer_vienna_shuffled7",
        "transformer_vienna_zero7",
    ]
    (tmp_path / "run_manifest.json").write_text(json.dumps({
        "run_mode": "smoke",
        "reportable": False,
        "filtered_unique_sequences": 256,
        "variants": variants,
    }), encoding="utf-8")
    empty_overlap = {"train_cv": 0, "train_test": 0, "cv_test": 0}
    (tmp_path / "split_report.json").write_text(json.dumps({
        "source": "/kaggle/input/stanford-ribonanza-rna-folding/train_data.csv",
        "splits_by_seed": {"42": {
            "random": {},
            "grouped": {"sequence_overlap": empty_overlap},
            "clustered": {"similarity_cluster_overlap": empty_overlap},
        }},
    }), encoding="utf-8")
    (tmp_path / "environment.json").write_text(json.dumps({
        "gpu_names": ["Tesla P100-PCIE-16GB"],
    }), encoding="utf-8")
    with (tmp_path / "ablation_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "seed", "split_method", "variant_id", "best_cv_loss",
            "test_mae", "cv_macro_sequence_mae", "test_macro_sequence_mae",
        ])
        writer.writeheader()
        writer.writerows([
            {
                "seed": "42",
                "split_method": split_method,
                "variant_id": variant_id,
                "best_cv_loss": "0.2",
                "test_mae": "",
                "cv_macro_sequence_mae": "0.22",
                "test_macro_sequence_mae": "",
            }
            for split_method in ("random", "grouped", "clustered")
            for variant_id in variants
        ])
    with (tmp_path / "test_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "seed", "split_method", "variant_id", "test_mae",
        ])
        writer.writeheader()
    with (tmp_path / "paired_bootstrap_ci.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "comparison_id", "seed", "split", "partition", "metric",
            "point_improvement", "ci_lower", "ci_upper",
            "probability_improvement",
        ])
        writer.writeheader()
        for split_method in ("random", "grouped", "clustered"):
            for comparison_id in (
                "vienna_real_vs_sequence_only",
                "vienna_real_vs_position_shuffled",
                "vienna_real_vs_zero_channels",
            ):
                for metric in ("nucleotide_weighted_mae", "macro_sequence_mae"):
                    writer.writerow({
                        "comparison_id": comparison_id,
                        "seed": "42",
                        "split": split_method,
                        "partition": "cv",
                        "metric": metric,
                        "point_improvement": "0.01",
                        "ci_lower": "0.001",
                        "ci_upper": "0.02",
                        "probability_improvement": "0.95",
                    })

    audit_smoke_results(tmp_path, 256, variants)


def test_smoke_audit_rejects_old_source(tmp_path):
    variants = ["transformer_seq4"]
    (tmp_path / "run_manifest.json").write_text(json.dumps({
        "run_mode": "smoke",
        "reportable": False,
        "filtered_unique_sequences": 256,
        "variants": variants,
    }), encoding="utf-8")
    empty_overlap = {"train_cv": 0, "train_test": 0, "cv_test": 0}
    (tmp_path / "split_report.json").write_text(json.dumps({
        "source": "/kaggle/input/competition/OLD/train_data.csv",
        "splits_by_seed": {"42": {
            "random": {},
            "grouped": {"sequence_overlap": empty_overlap},
            "clustered": {"similarity_cluster_overlap": empty_overlap},
        }},
    }), encoding="utf-8")

    with pytest.raises(RuntimeError, match="OLD"):
        audit_smoke_results(tmp_path, 256, variants)
