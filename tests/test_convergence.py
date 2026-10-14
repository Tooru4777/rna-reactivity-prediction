import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.run_ablation import fit_experiment_median_baseline, train_variant
from experiments.validate_development_archive import validate_development_archive
from pathlib import Path


def test_assay_medians_ignore_cv_targets_and_pool_only_measured_training_values():
    frame = pd.DataFrame({
        "sequence": ["AAA", "CCC", "GGG"],
        "experiment_type": ["2A3_MaP", "DMS_MaP", "2A3_MaP"],
        "reactivity_0001": [0.0, 0.2, 1.0],
        "reactivity_0002": [0.0, 0.3, 1.0],
        "reactivity_0003": [1.0, 0.4, 1.0],
    })
    columns = ["reactivity_0001", "reactivity_0002", "reactivity_0003"]
    baseline = fit_experiment_median_baseline(frame, [0, 1], columns)
    assert baseline == {"2A3_MaP": 0.0, "DMS_MaP": pytest.approx(0.3)}
    frame.loc[2, columns] = -99
    assert fit_experiment_median_baseline(frame, [0, 1], columns) == baseline


class ConstantModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(0.25))

    def forward(self, features, padding_mask=None):
        return self.value.expand(features.shape[0], features.shape[1], 2)


def test_early_stopping_preserves_best_checkpoint_and_honours_patience(tmp_path):
    loader = DataLoader(TensorDataset(
        torch.ones(2, 3, 4), torch.full((2, 3, 2), 0.5), torch.ones(2, 3, 2)
    ), batch_size=2)
    checkpoint = tmp_path / "best.pth"
    result = train_variant(
        ConstantModel(), loader, loader, torch.device("cpu"), num_epochs=10,
        lr=0.0, save_path=checkpoint, early_stopping_patience=2,
        early_stopping_min_delta=1e-4,
    )
    assert result["epochs_trained"] == 3
    assert result["best_epoch"] == 1
    assert result["stopped_early"]
    assert torch.load(checkpoint, weights_only=True)["value"].item() == 0.25


def test_v17_compact_archive_recomputes_all_54_cv_scores():
    root = Path(__file__).resolve().parents[1] / "results/kaggle-v17-9d904ff"
    report = validate_development_archive(root)
    assert report["validated_rows"] == 54
    assert report["test_rows"] == 0
    assert report["similarity_repairs"] == 2
