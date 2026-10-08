"""
Unit Tests — Data Pipelines
=============================
Tests for the reportable RNAReactivityDataset pipeline:
  - Correct tensor shapes and dtypes
  - Mask validity (binary, correct padding pattern)
  - Feature dimension handling (4-dim vs 7-dim)
  - Explicit small measured-data fixtures and missing-data failure
  - DataLoader batch collation

Usage:
    python -m pytest tests/test_data_pipelines.py -v
"""

import sys
import os
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data_pipeline_reactivity import RNAReactivityDataset
import src.data_pipeline_reactivity as reactivity_pipeline
from src.splitting import (
    grouped_split_indices,
    random_split_indices,
    sequence_overlap_counts,
    validate_split_integrity,
)
from src.data_loading import load_unique_sequence_subset, resolve_ribonanza_train_data


# =====================================================================
# Reactivity Dataset Tests
# =====================================================================

class TestReactivityDataset:
    """Tests for the RNA reactivity prediction dataset."""

    @pytest.fixture
    def reactivity_frame(self):
        """Small explicit fixture representing measured reactivity profiles."""
        return pd.DataFrame({
            "sequence_id": ["rna1", "rna2"],
            "sequence": ["ACGU", "UGCA"],
            "experiment_type": ["2A3_MaP", "DMS_MaP"],
            "SN_filter": [1.0, 1.0],
            "reactivity_0001": [0.1, 0.2],
            "reactivity_0002": [0.2, 0.3],
            "reactivity_0003": [0.3, 0.4],
            "reactivity_0004": [0.4, 0.5],
        })

    @pytest.fixture
    def dataset_4d(self, reactivity_frame):
        """4-dim dataset from an explicit measured-data fixture."""
        return RNAReactivityDataset(
            dataframe=reactivity_frame,
            max_length=206,
            use_structure=False,
        )

    @pytest.fixture
    def dataset_7d(self, reactivity_frame):
        """Structure-feature dataset from the same explicit fixture."""
        return RNAReactivityDataset(
            dataframe=reactivity_frame,
            max_length=206,
            use_structure=True,
        )

    def test_fixture_length(self, dataset_4d):
        assert len(dataset_4d) == 2

    def test_missing_reactivity_data_fails_fast(self):
        with pytest.raises(FileNotFoundError, match="real measured reactivity"):
            RNAReactivityDataset(
                sequences_csv="nonexistent.csv", use_structure=False
            )

    @pytest.mark.parametrize(
        "column,value,message",
        [
            ("experiment_type", "unknown", "Unexpected experiment_type"),
            ("sequence", "ACGN", "only A, C, G, and U"),
        ],
    )
    def test_research_data_domain_validation(self, column, value, message):
        frame = pd.DataFrame({
            "sequence": ["ACGU"],
            "experiment_type": ["2A3_MaP"],
            "SN_filter": [1.0],
            "reactivity_0001": [0.2],
        })
        frame.loc[0, column] = value
        with pytest.raises(ValueError, match=message):
            RNAReactivityDataset(
                dataframe=frame, use_structure=False
            )

    def test_output_types(self, dataset_4d):
        """Each sample should return 3 tensors."""
        features, targets, mask = dataset_4d[0]
        assert isinstance(features, torch.Tensor)
        assert isinstance(targets, torch.Tensor)
        assert isinstance(mask, torch.Tensor)

    def test_feature_shape_4d(self, dataset_4d):
        """4-dim features: (max_length, 4)."""
        features, targets, mask = dataset_4d[0]
        assert features.shape == (206, 4)

    def test_feature_shape_7d(self, dataset_7d):
        """Structure-feature runs must always expose seven input channels."""
        features, targets, mask = dataset_7d[0]
        assert features.shape == (206, 7)

    def test_structure_request_fails_without_viennarna(
        self, reactivity_frame, monkeypatch
    ):
        monkeypatch.setattr(reactivity_pipeline, "HAS_VIENNA", False)
        with pytest.raises(RuntimeError, match="ViennaRNA is required"):
            RNAReactivityDataset(dataframe=reactivity_frame, use_structure=True)

    def test_reactivity_columns_are_sorted_numerically(self, reactivity_frame):
        reordered = reactivity_frame[[
            "sequence_id", "sequence", "experiment_type", "SN_filter",
            "reactivity_0003", "reactivity_0001", "reactivity_0004", "reactivity_0002",
        ]]
        dataset = RNAReactivityDataset(dataframe=reordered, use_structure=False)
        assert dataset.reactivity_cols == [
            "reactivity_0001", "reactivity_0002", "reactivity_0003", "reactivity_0004"
        ]

    def test_target_shape(self, dataset_4d):
        """Targets: (max_length, 2) for 2A3_MaP and DMS_MaP."""
        features, targets, mask = dataset_4d[0]
        assert targets.shape == (206, 2)

    def test_mask_shape(self, dataset_4d):
        """Mask: (max_length, 2)."""
        features, targets, mask = dataset_4d[0]
        assert mask.shape == (206, 2)

    def test_mask_is_binary(self, dataset_4d):
        """Mask values should be 0.0 or 1.0 only."""
        features, targets, mask = dataset_4d[0]
        unique_vals = torch.unique(mask)
        for v in unique_vals:
            assert v.item() in (0.0, 1.0), f"Unexpected mask value: {v}"

    def test_features_are_one_hot(self, dataset_4d):
        """Each row of features should sum to 0 (padding) or 1 (one-hot)."""
        features, _, _ = dataset_4d[0]
        row_sums = features.sum(dim=1)
        for s in row_sums:
            assert s.item() in (0.0, 1.0), f"Feature row sum {s} is not 0 or 1"

    def test_dataloader_batching(self, dataset_4d):
        """DataLoader should produce valid batched tensors."""
        loader = DataLoader(dataset_4d, batch_size=8, shuffle=False)
        features, targets, mask = next(iter(loader))

        assert features.shape == (len(dataset_4d), 206, 4)
        assert targets.shape == (len(dataset_4d), 206, 2)
        assert mask.shape == (len(dataset_4d), 206, 2)

    def test_feature_dim_property(self, dataset_4d, dataset_7d):
        """feature_dim property should match actual feature tensor dimension."""
        f4, _, _ = dataset_4d[0]
        assert f4.shape[1] == dataset_4d.feature_dim

        f7, _, _ = dataset_7d[0]
        assert f7.shape[1] == dataset_7d.feature_dim

    def test_structure_controls_preserve_targets_and_feature_width(
        self, monkeypatch
    ):
        frame = pd.DataFrame({
            "sequence": ["AACCGGUU", "AACCGGUU"],
            "experiment_type": ["2A3_MaP", "DMS_MaP"],
            "SN_filter": [1.0, 1.0],
            **{
                f"reactivity_{position:04d}": [0.1 * position, 0.05 * position]
                for position in range(1, 9)
            },
        })
        monkeypatch.setattr(
            reactivity_pipeline.RNA,
            "fold",
            lambda sequence: ("((..))..", 0.0),
        )
        real = RNAReactivityDataset(dataframe=frame, structure_mode="real")
        shuffled = RNAReactivityDataset(
            dataframe=frame,
            structure_mode="position_shuffled",
            structure_control_seed=17,
        )
        zero = RNAReactivityDataset(dataframe=frame, structure_mode="zero")

        real_features, real_targets, real_mask = real[0]
        shuffled_features, shuffled_targets, shuffled_mask = shuffled[0]
        zero_features, zero_targets, zero_mask = zero[0]
        assert real_features.shape == shuffled_features.shape == zero_features.shape == (
            206, 7
        )
        assert torch.equal(real_features[:, :4], shuffled_features[:, :4])
        assert torch.equal(real_features[:, :4], zero_features[:, :4])
        assert torch.equal(real_targets, shuffled_targets)
        assert torch.equal(real_targets, zero_targets)
        assert torch.equal(real_mask, shuffled_mask)
        assert torch.equal(real_mask, zero_mask)
        assert torch.equal(
            real_features[:8, 4:].sum(dim=0),
            shuffled_features[:8, 4:].sum(dim=0),
        )
        assert not torch.equal(real_features[:8, 4:], shuffled_features[:8, 4:])
        assert torch.count_nonzero(zero_features[:, 4:]) == 0
        assert torch.equal(shuffled[0][0], shuffled[1][0])

    def test_structure_control_seed_is_reproducible(self, monkeypatch):
        frame = pd.DataFrame({
            "sequence": ["AACCGGUU"],
            "experiment_type": ["2A3_MaP"],
            "SN_filter": [1.0],
            **{
                f"reactivity_{position:04d}": [0.1]
                for position in range(1, 9)
            },
        })
        monkeypatch.setattr(
            reactivity_pipeline.RNA,
            "fold",
            lambda sequence: ("((..))..", 0.0),
        )
        first = RNAReactivityDataset(
            dataframe=frame,
            structure_mode="position_shuffled",
            structure_control_seed=99,
        )
        second = RNAReactivityDataset(
            dataframe=frame,
            structure_mode="position_shuffled",
            structure_control_seed=99,
        )
        assert torch.equal(first[0][0], second[0][0])

    def test_invalid_structure_mode_fails_fast(self, reactivity_frame):
        with pytest.raises(ValueError, match="structure_mode"):
            RNAReactivityDataset(
                dataframe=reactivity_frame, structure_mode="random"
            )


def test_grouped_split_has_no_sequence_overlap():
    import pandas as pd

    frame = pd.DataFrame({
        "sequence": [f"SEQ{i}" for i in range(20) for _ in range(2)],
        "experiment_type": ["2A3_MaP", "DMS_MaP"] * 20,
    })
    train, cv, test = grouped_split_indices(frame, seed=42)
    split_groups = [set(frame.iloc[idx]["sequence"]) for idx in (train, cv, test)]

    assert split_groups[0].isdisjoint(split_groups[1])
    assert split_groups[0].isdisjoint(split_groups[2])
    assert split_groups[1].isdisjoint(split_groups[2])
    assert sorted(train + cv + test) == list(range(len(frame)))


def test_unique_sequence_subset_keeps_all_experiment_rows(tmp_path):
    import pandas as pd

    csv_path = tmp_path / "train_data.csv"
    frame = pd.DataFrame({
        "sequence": ["AAA", "CCC", "GGG", "AAA", "CCC", "GGG"],
        "experiment_type": ["2A3_MaP"] * 3 + ["DMS_MaP"] * 3,
        "SN_filter": [1.0] * 6,
        "reactivity_0001": [0.1] * 6,
    })
    frame.to_csv(csv_path, index=False)

    subset = load_unique_sequence_subset(csv_path, max_sequences=2, chunk_size=2)
    assert subset["sequence"].nunique() == 2
    assert len(subset) == 4
    assert set(subset.groupby("sequence")["experiment_type"].nunique()) == {2}


def test_unique_sequence_subset_samples_only_quality_eligible_sequences(tmp_path):
    import pandas as pd

    csv_path = tmp_path / "train_data.csv"
    frame = pd.DataFrame({
        "sequence": ["LOW", "GOOD1", "GOOD2", "LOW", "GOOD1", "GOOD2"],
        "experiment_type": ["2A3_MaP"] * 3 + ["DMS_MaP"] * 3,
        "SN_filter": [0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        "reactivity_0001": [0.1] * 6,
    })
    frame.to_csv(csv_path, index=False)

    subset = load_unique_sequence_subset(csv_path, max_sequences=2, chunk_size=2)
    assert set(subset["sequence"]) == {"GOOD1", "GOOD2"}


def test_resolve_ribonanza_data_rejects_legacy_only_directory(tmp_path):
    expected = tmp_path / "stanford-ribonanza-rna-folding" / "OLD" / "train_data.csv"
    expected.parent.mkdir(parents=True)
    expected.touch()

    with pytest.raises(FileNotFoundError, match="Only superseded OLD"):
        resolve_ribonanza_train_data(tmp_path)
    assert (
        resolve_ribonanza_train_data(tmp_path, allow_legacy=True)
        == expected.resolve()
    )


def test_resolve_ribonanza_data_prefers_current_over_old(tmp_path):
    root = tmp_path / "stanford-ribonanza-rna-folding"
    current = root / "train_data.csv"
    legacy = root / "OLD" / "train_data.csv"
    legacy.parent.mkdir(parents=True)
    current.touch()
    legacy.touch()

    assert resolve_ribonanza_train_data(tmp_path) == current.resolve()


def test_full_loader_filters_quality_and_excludes_error_columns(tmp_path):
    frame = pd.DataFrame({
        "sequence": ["AAAA", "CCCC"],
        "experiment_type": ["2A3_MaP", "DMS_MaP"],
        "SN_filter": [1.0, 0.0],
        "reactivity_0001": [0.1, 0.2],
        "reactivity_error_0001": [0.01, 0.02],
    })
    csv_path = tmp_path / "train_data.csv"
    frame.to_csv(csv_path, index=False)

    loaded = load_unique_sequence_subset(
        csv_path, max_sequences=0, chunk_size=1
    )

    assert loaded["sequence"].tolist() == ["AAAA"]
    assert "reactivity_error_0001" not in loaded.columns
    assert loaded["reactivity_0001"].dtype.name == "float32"


def test_loader_rejects_negative_sequence_limit(tmp_path):
    csv_path = tmp_path / "train_data.csv"
    pd.DataFrame({
        "sequence": ["AAAA"],
        "experiment_type": ["2A3_MaP"],
        "SN_filter": [1.0],
        "reactivity_0001": [0.1],
    }).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="non-negative"):
        load_unique_sequence_subset(csv_path, max_sequences=-1)


def test_resolve_ribonanza_data_rejects_ambiguous_directory(tmp_path):
    for folder in ("first", "second"):
        path = tmp_path / folder / "train_data.csv"
        path.parent.mkdir()
        path.touch()
    with pytest.raises(FileNotFoundError, match="Could not uniquely resolve"):
        resolve_ribonanza_train_data(tmp_path)


def test_random_split_detects_sequence_overlap():
    import pandas as pd

    frame = pd.DataFrame({"sequence": [f"SEQ{i}" for i in range(30)] * 2})
    splits = random_split_indices(len(frame), seed=42)
    overlap = sequence_overlap_counts(frame, splits)
    assert sum(overlap.values()) > 0


def test_split_integrity_rejects_duplicate_or_missing_rows():
    with pytest.raises(ValueError, match="duplicate row indices"):
        validate_split_integrity(4, ([0, 1], [1], [2]))
    with pytest.raises(ValueError, match="does not match dataset rows"):
        validate_split_integrity(4, ([0], [1], [2]))
