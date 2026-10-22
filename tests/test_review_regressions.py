"""Regression checks for the comprehensive review; fixtures are engineering-only."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.run_ablation import write_artifact_checksums, write_training_curves_figure
from experiments.validate_development_archive import validate_development_archive
from src.data_pipeline_reactivity import RNAReactivityDataset
from src.similarity_splitting import validate_similarity_manifest

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "results/kaggle-v18-df2d87b"


def test_final_evaluation_fails_before_data_access(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "experiments/run_ablation.py"),
                             "--run-mode", "final-eval", "--split-method", "clustered",
                             "--similarity-manifest", str(tmp_path / "absent.json"),
                             "--data", str(tmp_path / "absent.csv")], capture_output=True, text=True)
    assert result.returncode != 0
    assert "final-eval is disabled" in result.stderr
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("sequence,values,max_length,match", [
    ("ACGU", [0.1, 0.2, 0.3, 0.4], 3, "silent truncation"),
    ("AC", [np.inf, 0.2], 2, "Infinite"),
    ("A", [np.nan, 0.2], 2, "beyond"),
    ("AC", [np.nan, np.nan], 2, "finite reactivity"),
])
def test_data_domains_fail_before_training(sequence, values, max_length, match):
    frame = pd.DataFrame({"sequence": [sequence], "experiment_type": ["2A3_MaP"],
                          "SN_filter": [1], **{f"reactivity_{i:04d}": [v]
                          for i, v in enumerate(values, 1)}})
    with pytest.raises(ValueError, match=match):
        RNAReactivityDataset(dataframe=frame, use_structure=False, max_length=max_length)


def test_early_stopping_figure_does_not_label_survivor_mean_as_full_seed_mean(tmp_path):
    frame = pd.DataFrame([{"seed": seed, "split_method": "clustered", "variant": "model",
                           "epoch": epoch, "cv_loss": 0.2 + seed / 100}
                          for seed, last in [(1, 2), (2, 4), (3, 3)]
                          for epoch in range(1, last + 1)])
    result = write_training_curves_figure(frame, tmp_path / "curves.png")
    assert result.seeds.tolist() == [3, 3, 2, 1]
    assert result.complete_seed_set.tolist() == [True, True, False, False]


@pytest.mark.parametrize("parameter,value", [("search_type", 2), ("strand", 2), ("kmer_length", 12)])
def test_manifest_rejects_changed_nucleotide_search_definition(parameter, value):
    manifest = json.loads((ARCHIVE / "similarity_split_manifest.json").read_text())
    manifest["cross_split_search_audit"]["parameters"][parameter] = value
    with pytest.raises(ValueError, match=parameter):
        validate_similarity_manifest(manifest, require_cross_split_audit=True)


@pytest.mark.parametrize("filename,column,value", [
    ("baseline_results.csv", "cv_valid_positions", 1),
    ("baseline_results.csv", "median_test_mae", 0.2),
    ("ablation_summary.csv", "epochs_trained", 999),
    ("prediction_health_initial.csv", "partition", "test"),
    ("evaluation_metrics.csv", "valid_positions", 1),
])
def test_validator_rejects_semantic_corruption_even_with_recomputed_checksums(tmp_path, filename, column, value):
    archive = tmp_path / "archive"
    shutil.copytree(ARCHIVE, archive)
    frame = pd.read_csv(archive / filename)
    frame.loc[0, column] = value
    frame.to_csv(archive / filename, index=False)
    write_artifact_checksums(archive)
    with pytest.raises(ValueError):
        validate_development_archive(archive)


def test_v18_archive_all_cv_groups():
    assert validate_development_archive(ARCHIVE)["validated_rows"] == 36


def test_empty_cv_cannot_be_saved_as_perfect_checkpoint(tmp_path):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from src.model_reactivity import RNAReactivityCNNOnly
    from experiments.run_ablation import train_variant
    train = DataLoader(TensorDataset(torch.ones(1, 4, 4), torch.ones(1, 4, 2), torch.ones(1, 4, 2)))
    empty = DataLoader(TensorDataset(torch.empty(0, 4, 4), torch.empty(0, 4, 2), torch.empty(0, 4, 2)))
    checkpoint = tmp_path / 'model.pth'
    with pytest.raises(ValueError, match='CV requires measured positions'):
        train_variant(RNAReactivityCNNOnly(input_dim=4, cnn_out_dim=8), train, empty,
                      torch.device('cpu'), num_epochs=1, save_path=checkpoint)
    assert not checkpoint.exists()


def test_sharded_uneven_masks_use_global_denominator_and_gradients():
    import torch
    # This exercises real autograd; it is not a CUDA DataParallel hardware test.
    values = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], requires_grad=True)
    weights = torch.tensor([1., 1., 1., 7., 11.])
    global_loss = (values * weights).sum() / weights.sum()
    shards = [(values[:3] * weights[:3]).sum(), (values[3:] * weights[3:]).sum()]
    gathered_loss = torch.stack(shards).sum() / weights.sum()
    a = torch.autograd.grad(global_loss, values, retain_graph=True)[0]
    b = torch.autograd.grad(gathered_loss, values)[0]
    assert torch.equal(a, b)
    assert torch.allclose(global_loss, gathered_loss)
    wrong = torch.stack([shards[0] / weights[:3].sum(), shards[1] / weights[3:].sum()]).mean()
    assert not torch.isclose(global_loss, wrong)
