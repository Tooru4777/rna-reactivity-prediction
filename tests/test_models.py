"""
Unit Tests — RNA Model Architectures
======================================
Tests all 5 model variants (4 reactivity + 1 3D) for:
  - Correct output shapes
  - Expected parameter counts
  - Edge cases (single-sample batch, max-length sequence)
  - Gradient flow (parameters receive gradients after backward pass)

Usage:
    python -m pytest tests/test_models.py -v
"""

import sys
import os
import pytest
import torch

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.model_reactivity import (
    RNAReactivityCNNOnly,
    RNAReactivityCNN_LSTM,
    RNAReactivityCNN_LSTM_Transformer,
    RNAReactivityPredictor,
)
from src.model_3d import RNAPredictor3D


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture(params=[
    ("CNNOnly_4d", RNAReactivityCNNOnly, {"input_dim": 4}, 4),
    ("CNNOnly_7d", RNAReactivityCNNOnly, {"input_dim": 7}, 7),
    ("CNN_LSTM", RNAReactivityCNN_LSTM, {"input_dim": 7}, 7),
    ("CNN_LSTM_Transformer", RNAReactivityCNN_LSTM_Transformer, {"input_dim": 4}, 4),
    ("FullModel", RNAReactivityPredictor, {"input_dim": 7}, 7),
], ids=lambda x: x[0])
def reactivity_model_spec(request):
    """Parametrised fixture yielding (name, model_class, kwargs, input_dim)."""
    return request.param


@pytest.fixture
def full_model():
    return RNAReactivityPredictor(input_dim=7)


@pytest.fixture
def model_3d():
    return RNAPredictor3D()


# =====================================================================
# Reactivity Model Tests
# =====================================================================

class TestReactivityModels:
    """Tests for all reactivity model variants."""

    def test_output_shape(self, reactivity_model_spec, device):
        """Output must be (batch, seq_len, 2) for reactivity prediction."""
        name, cls, kwargs, in_dim = reactivity_model_spec
        model = cls(**kwargs).to(device)

        batch_size, seq_len = 4, 206
        x = torch.randn(batch_size, seq_len, in_dim, device=device)
        out = model(x)

        assert out.shape == (batch_size, seq_len, 2), (
            f"{name}: expected shape ({batch_size}, {seq_len}, 2), got {out.shape}"
        )

    def test_single_sample_batch(self, reactivity_model_spec, device):
        """Model must handle batch_size=1 without errors."""
        name, cls, kwargs, in_dim = reactivity_model_spec
        model = cls(**kwargs).to(device)

        x = torch.randn(1, 206, in_dim, device=device)
        out = model(x)

        assert out.shape == (1, 206, 2)

    def test_short_sequence(self, reactivity_model_spec, device):
        """Model must handle very short sequences (seq_len=10)."""
        name, cls, kwargs, in_dim = reactivity_model_spec
        model = cls(**kwargs).to(device)

        x = torch.randn(2, 10, in_dim, device=device)
        out = model(x)

        assert out.shape == (2, 10, 2)

    def test_gradient_flow(self, reactivity_model_spec, device):
        """All parameters must receive non-None gradients after backward."""
        name, cls, kwargs, in_dim = reactivity_model_spec
        model = cls(**kwargs).to(device)
        model.train()

        x = torch.randn(2, 50, in_dim, device=device)
        out = model(x)
        loss = out.sum()
        loss.backward()

        for param_name, param in model.named_parameters():
            assert param.grad is not None, (
                f"{name}: parameter '{param_name}' has no gradient"
            )
            assert not torch.all(param.grad == 0), (
                f"{name}: parameter '{param_name}' has all-zero gradients"
            )

    def test_right_padding_does_not_change_real_positions(
        self, reactivity_model_spec, device
    ):
        """Explicit masking must make predictions invariant to right-padding."""
        name, cls, kwargs, in_dim = reactivity_model_spec
        model = cls(**kwargs).to(device).eval()
        real = torch.randn(1, 12, in_dim, device=device)
        padded = torch.cat([real, torch.zeros(1, 8, in_dim, device=device)], dim=1)

        with torch.no_grad():
            short_out = model(
                real, padding_mask=torch.zeros(1, 12, dtype=torch.bool, device=device)
            )
            long_out = model(
                padded,
                padding_mask=torch.tensor(
                    [[False] * 12 + [True] * 8], device=device
                ),
            )

        assert torch.allclose(short_out, long_out[:, :12], atol=1e-5), name

    def test_training_mode_is_padding_invariant_without_dropout(
        self, reactivity_model_spec, device
    ):
        """Normalization statistics must not depend on the amount of padding."""
        name, cls, kwargs, in_dim = reactivity_model_spec
        short_model = cls(**kwargs, dropout=0.0).to(device).train()
        padded_model = cls(**kwargs, dropout=0.0).to(device).train()
        padded_model.load_state_dict(short_model.state_dict())
        real = torch.randn(2, 12, in_dim, device=device)
        padded = torch.cat([real, torch.zeros(2, 8, in_dim, device=device)], dim=1)

        short_out = short_model(
            real, padding_mask=torch.zeros(2, 12, dtype=torch.bool, device=device)
        )
        padded_out = padded_model(
            padded,
            padding_mask=torch.tensor(
                [[False] * 12 + [True] * 8] * 2, device=device
            ),
        )

        assert torch.allclose(short_out, padded_out[:, :12], atol=1e-5), name


class TestFullModelSpecifics:
    """Additional tests specific to the full RNAReactivityPredictor."""

    def test_residual_connection_effect(self, full_model, device):
        """Verify the model produces different outputs than a zero-init would.

        This is a basic sanity check that the residual connection path
        (lstm_out + transformer_out) is active.
        """
        model = full_model.to(device).eval()
        x = torch.randn(1, 50, 7, device=device)

        with torch.no_grad():
            out = model(x)

        # Output should not be all zeros (residual ensures signal passes through)
        assert not torch.all(out == 0), "Model output is all zeros"

    def test_parameter_count(self, full_model):
        """Full model should have ~2.28M parameters (regression guard)."""
        params = sum(p.numel() for p in full_model.parameters())
        # Allow ±5% tolerance for minor architecture tweaks
        assert 2_100_000 < params < 2_500_000, (
            f"Parameter count {params:,} outside expected range [2.1M, 2.5M]"
        )


# =====================================================================
# 3D Model Tests
# =====================================================================

class TestModel3D:
    """Tests for the RNAPredictor3D model."""

    def test_output_shape(self, model_3d, device):
        """Output must be (batch, seq_len, 3) for x, y, z coordinates."""
        model = model_3d.to(device)
        x = torch.randn(4, 200, 4, device=device)
        out = model(x)

        assert out.shape == (4, 200, 3)

    def test_single_sample(self, model_3d, device):
        """Handle batch_size=1."""
        model = model_3d.to(device)
        x = torch.randn(1, 200, 4, device=device)
        out = model(x)

        assert out.shape == (1, 200, 3)

    def test_short_sequence(self, model_3d, device):
        """Handle short sequences."""
        model = model_3d.to(device)
        x = torch.randn(2, 20, 4, device=device)
        out = model(x)

        assert out.shape == (2, 20, 3)

    def test_gradient_flow(self, model_3d, device):
        """All parameters receive gradients."""
        model = model_3d.to(device)
        model.train()

        x = torch.randn(2, 50, 4, device=device)
        out = model(x)
        loss = out.sum()
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"'{name}' has no gradient"

    def test_parameter_count(self, model_3d):
        """3D model should have a reasonable parameter count (~596K)."""
        params = sum(p.numel() for p in model_3d.parameters())
        # 2-layer Bi-LSTM (hidden=128) + CNN (64 filters) ≈ 596K params
        assert 400_000 < params < 700_000, (
            f"Parameter count {params:,} outside expected range [400K, 700K]"
        )
