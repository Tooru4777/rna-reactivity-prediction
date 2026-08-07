"""
Unit Tests — Data Pipelines
=============================
Tests for RNAReactivityDataset and RNA3DDataset:
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
from src.data_pipeline_3d import RNA3DDataset
from src.splitting import (
    grouped_split_indices,
    random_split_indices,
    sequence_overlap_counts,
    validate_split_integrity,
)
from src.data_loading import load_unique_sequence_subset


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

    @pytest.fixture
    def dataset_real(self):
        """Dataset from actual CSV if available, else skip."""
        csv_path = os.path.join(PROJECT_ROOT, "train_data_1000.csv")
        if not os.path.exists(csv_path):
            pytest.skip("train_data_1000.csv not found")
        return RNAReactivityDataset(
            sequences_csv=csv_path,
            max_length=206,
            use_structure=False,
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
        """7-dim features: (max_length, 4) when ViennaRNA unavailable, or (max_length, 7)."""
        features, targets, mask = dataset_7d[0]
        # Feature dim depends on whether ViennaRNA is installed
        assert features.shape[0] == 206
        assert features.shape[1] in (4, 7)

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

    def test_real_data_shapes(self, dataset_real):
        """Verify shapes on real data (if available)."""
        features, targets, mask = dataset_real[0]
        assert features.shape == (206, 4)
        assert targets.shape == (206, 2)
        assert mask.shape == (206, 2)

    def test_feature_dim_property(self, dataset_4d, dataset_7d):
        """feature_dim property should match actual feature tensor dimension."""
        f4, _, _ = dataset_4d[0]
        assert f4.shape[1] == dataset_4d.feature_dim

        f7, _, _ = dataset_7d[0]
        assert f7.shape[1] == dataset_7d.feature_dim


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


# =====================================================================
# 3D Dataset Tests
# =====================================================================

class TestRNA3DDataset:
    """Tests for the RNA 3D coordinate prediction dataset."""

    @pytest.fixture
    def dataset(self, tmp_path):
        """3D dataset backed by explicit sequence and coordinate fixtures."""
        sequences_csv = tmp_path / "sequences.csv"
        labels_csv = tmp_path / "labels.csv"
        pd.DataFrame({
            "target_id": ["rna1", "rna2"],
            "sequence": ["ACGU", "UGCA"],
        }).to_csv(sequences_csv, index=False)
        label_rows = []
        for target_id in ("rna1", "rna2"):
            for position in range(4):
                label_rows.append({
                    "ID": f"{target_id}_{position}",
                    "x": float(position),
                    "y": float(position + 1),
                    "z": float(position + 2),
                })
        pd.DataFrame(label_rows).to_csv(labels_csv, index=False)
        return RNA3DDataset(
            sequences_csv=sequences_csv,
            labels_csv=labels_csv,
            max_length=200,
        )

    def test_fixture_length(self, dataset):
        assert len(dataset) == 2

    def test_missing_3d_labels_fail_fast(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="real sequence"):
            RNA3DDataset(
                sequences_csv=tmp_path / "missing_sequences.csv",
                labels_csv=tmp_path / "missing_labels.csv",
            )

    def test_output_types(self, dataset):
        """Each sample should return 3 tensors."""
        one_hot, coords, mask = dataset[0]
        assert isinstance(one_hot, torch.Tensor)
        assert isinstance(coords, torch.Tensor)
        assert isinstance(mask, torch.Tensor)

    def test_one_hot_shape(self, dataset):
        """One-hot features: (max_length, 4)."""
        one_hot, coords, mask = dataset[0]
        assert one_hot.shape == (200, 4)

    def test_coords_shape(self, dataset):
        """Coordinates: (max_length, 3) for x, y, z."""
        one_hot, coords, mask = dataset[0]
        assert coords.shape == (200, 3)

    def test_mask_shape(self, dataset):
        """Mask: (max_length,) — 1D for 3D dataset."""
        one_hot, coords, mask = dataset[0]
        assert mask.shape == (200,)

    def test_mask_is_binary(self, dataset):
        """Mask values should be 0.0 or 1.0."""
        one_hot, coords, mask = dataset[0]
        unique_vals = torch.unique(mask)
        for v in unique_vals:
            assert v.item() in (0.0, 1.0)

    def test_padding_consistency(self, dataset):
        """Padded positions should have zero one-hot features and zero coords."""
        one_hot, coords, mask = dataset[0]

        # Find padded positions (mask == 0)
        padded_positions = (mask == 0).nonzero(as_tuple=True)[0]
        if len(padded_positions) > 0:
            idx = padded_positions[0].item()
            assert one_hot[idx].sum().item() == 0.0, "Padded position has non-zero features"
            assert coords[idx].sum().item() == 0.0, "Padded position has non-zero coords"

    def test_dataloader_batching(self, dataset):
        """DataLoader should produce valid batched tensors."""
        loader = DataLoader(dataset, batch_size=16, shuffle=False)
        one_hot, coords, mask = next(iter(loader))

        assert one_hot.shape == (len(dataset), 200, 4)
        assert coords.shape == (len(dataset), 200, 3)
        assert mask.shape == (len(dataset), 200)
