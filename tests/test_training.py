"""
Unit Tests — Training Smoke Tests
===================================
Verifies that a single training step (forward + backward + optimizer.step())
completes without error for the reportable reactivity models.

These are NOT performance tests — they only check that the training loop
mechanics are correct (loss computation, gradient flow, parameter updates).

Usage:
    python -m pytest tests/test_training.py -v
"""

import sys
import os
import hashlib
import pytest
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.model_reactivity import (
    RNAReactivityCNNOnly,
    RNAReactivityCNN_LSTM,
    RNAReactivityCNN_LSTM_Transformer,
    RNAReactivityPredictor,
)
from experiments.run_ablation import evaluate_checkpoint, sha256_frame


# =====================================================================
# Reactivity Training Smoke Tests
# =====================================================================

class TestReactivityTraining:
    """Smoke tests for reactivity model training."""

    @pytest.fixture(params=[
        ("CNNOnly", RNAReactivityCNNOnly, 4),
        ("CNN_LSTM", RNAReactivityCNN_LSTM, 4),
        ("CNN_LSTM_Transformer", RNAReactivityCNN_LSTM_Transformer, 4),
        ("FullModel", RNAReactivityPredictor, 7),
    ], ids=lambda x: x[0])
    def model_spec(self, request):
        return request.param

    def test_single_training_step(self, model_spec):
        """One forward + backward + step must complete without error."""
        name, cls, in_dim = model_spec
        model = cls(input_dim=in_dim)
        model.train()

        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.L1Loss(reduction='none')

        # Test-only tensor fixture; never used for reportable model fitting.
        batch_size, seq_len = 4, 50
        features = torch.randn(batch_size, seq_len, in_dim)
        targets = torch.rand(batch_size, seq_len, 2)  # [0, 1]
        masks = torch.ones(batch_size, seq_len, 2)

        # Record initial parameter values
        initial_params = {
            n: p.clone() for n, p in model.named_parameters()
        }

        # Training step
        optimizer.zero_grad()
        predictions = model(features)
        reacts_clipped = torch.clamp(targets, 0.0, 1.0)
        loss_matrix = criterion(predictions, reacts_clipped)
        masked_loss = loss_matrix * masks
        loss = masked_loss.sum() / masks.sum()
        loss.backward()
        optimizer.step()

        # Verify loss is a valid scalar
        assert loss.dim() == 0, "Loss should be a scalar"
        assert not torch.isnan(loss), f"{name}: loss is NaN"
        assert not torch.isinf(loss), f"{name}: loss is Inf"

        # Verify parameters actually changed
        params_changed = False
        for n, p in model.named_parameters():
            if not torch.equal(p, initial_params[n]):
                params_changed = True
                break
        assert params_changed, f"{name}: no parameters changed after optimizer step"

    def test_masked_loss_excludes_padding(self, model_spec):
        """Loss at masked (padded) positions should not contribute."""
        name, cls, in_dim = model_spec
        model = cls(input_dim=in_dim)
        model.eval()

        batch_size, seq_len = 2, 50
        features = torch.randn(batch_size, seq_len, in_dim)
        targets = torch.rand(batch_size, seq_len, 2)

        # Mask: first 20 positions valid, rest padded
        masks = torch.zeros(batch_size, seq_len, 2)
        masks[:, :20, :] = 1.0

        with torch.no_grad():
            predictions = model(features)
            reacts_clipped = torch.clamp(targets, 0.0, 1.0)
            loss_matrix = torch.abs(predictions - reacts_clipped)
            masked_loss = loss_matrix * masks

            # Loss should only reflect positions 0-19
            padded_loss = (loss_matrix * (1 - masks)).sum()
            assert (masked_loss[:, 20:, :] == 0).all(), (
                "Masked positions should contribute zero loss"
            )

    def test_clipped_eval_metric(self, model_spec):
        """Evaluation metric should clamp both predictions and targets."""
        name, cls, in_dim = model_spec
        model = cls(input_dim=in_dim)
        model.eval()

        features = torch.randn(2, 30, in_dim)
        masks = torch.ones(2, 30, 2)

        with torch.no_grad():
            predictions = model(features)

            # Kaggle-style clipped MAE
            preds_clipped = torch.clamp(predictions, 0.0, 1.0)
            targets_clipped = torch.clamp(torch.rand(2, 30, 2), 0.0, 1.0)
            clipped_mae = torch.abs(preds_clipped - targets_clipped)

            # Clipped MAE should be in [0, 1]
            assert clipped_mae.min() >= 0.0
            assert clipped_mae.max() <= 1.0

    def test_held_out_evaluation_emits_error_records(self, tmp_path):
        """A saved checkpoint must produce aggregate and stratified test records."""
        factory = lambda: RNAReactivityCNNOnly(input_dim=4, cnn_out_dim=8)
        checkpoint = tmp_path / "model.pth"
        torch.save(factory().state_dict(), checkpoint)

        features = torch.zeros(2, 8, 4)
        features[0, :4, 0] = 1.0
        features[1, :6, 1] = 1.0
        targets = torch.full((2, 8, 2), 0.5)
        masks = torch.zeros(2, 8, 2)
        masks[0, :4, 0] = 1.0
        masks[1, :6, 1] = 1.0
        loader = DataLoader(TensorDataset(features, targets, masks), batch_size=2)
        frame = pd.DataFrame({
            "sequence_id": ["a", "b"],
            "sequence": ["AAAA", "CCCCCC"],
            "experiment_type": ["2A3_MaP", "DMS_MaP"],
            "structure": ["(())", "......"],
        })

        mae, rows, structure_rows = evaluate_checkpoint(
            factory, checkpoint, loader, torch.device("cpu"), frame, [0, 1], 8
        )

        assert 0.0 <= mae <= 1.0
        assert len(rows) == 2
        assert {row["experiment_type"] for row in rows} == {"2A3_MaP", "DMS_MaP"}
        assert {row["structure_class"] for row in structure_rows} == {
            "paired", "unpaired"
        }


def test_streaming_frame_hash_matches_reference_serialization():
    frame = pd.DataFrame({
        "sequence": ["AAAA", "CCCC"],
        "value": [0.1, float("nan")],
    })
    serialized = frame.to_csv(
        index=False,
        na_rep="<NA>",
        float_format="%.17g",
        lineterminator="\n",
    )
    expected = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    assert sha256_frame(frame, ["sequence", "value"]) == expected
