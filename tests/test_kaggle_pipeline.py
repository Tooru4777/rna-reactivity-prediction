"""Tests for the Kaggle GPU entrypoint."""

from pathlib import Path
import importlib.util

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "kaggle" / "gpu_ablation.py"
SPEC = importlib.util.spec_from_file_location("gpu_ablation", MODULE_PATH)
gpu_ablation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gpu_ablation)
find_competition_input = gpu_ablation.find_competition_input


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
