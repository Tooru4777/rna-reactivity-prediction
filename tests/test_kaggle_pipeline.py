"""Tests for the Kaggle GPU entrypoint."""

from pathlib import Path
import importlib.util

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "kaggle" / "gpu_ablation.py"
SPEC = importlib.util.spec_from_file_location("gpu_ablation", MODULE_PATH)
gpu_ablation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gpu_ablation)
find_competition_input = gpu_ablation.find_competition_input
checkout_repository = gpu_ablation.checkout_repository


def test_runner_defaults_to_main_and_pins_vienna():
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert gpu_ablation.DEFAULT_REPOSITORY_REF == "main"
    assert '"ViennaRNA==2.7.2"' in source


def test_finds_train_data_nested_under_old(tmp_path):
    expected = (
        tmp_path
        / "stanford-ribonanza-rna-folding"
        / "OLD"
        / "train_data.csv"
    )
    expected.parent.mkdir(parents=True)
    expected.touch()

    assert find_competition_input(tmp_path) == expected


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
