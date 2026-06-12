"""
RNA 3D Structure Prediction Model
==================================
A hybrid CNN + Bi-LSTM architecture for predicting per-nucleotide 3D
coordinates (x, y, z) from one-hot encoded RNA sequences.

Architecture rationale:
  - 1D CNN:     Captures local sequence motifs (e.g., hairpin loops, bulges)
                that are known to influence local 3D geometry.
  - Bi-LSTM:    Models long-range dependencies in both 5'→3' and 3'→5'
                directions, reflecting the fact that RNA folding depends on
                base-pairing interactions across the entire sequence.
"""

import torch
import torch.nn as nn


class RNAPredictor3D(nn.Module):
    """
    Predicts 3D spatial coordinates for each nucleotide in an RNA sequence.

    Input:  (batch_size, max_length, 4)  — one-hot encoded sequence
    Output: (batch_size, max_length, 3)  — predicted x, y, z coordinates
    """

    def __init__(self, input_dim=4, cnn_out_dim=64, lstm_hidden_dim=128,
                 output_dim=3, dropout=0.3):
        super(RNAPredictor3D, self).__init__()

        # Local feature extraction via 1D convolution.
        # kernel_size=5 corresponds to a sliding window of 5 nucleotides,
        # capturing short-range structural motifs.
        self.cnn = nn.Conv1d(
            in_channels=input_dim,
            out_channels=cnn_out_dim,
            kernel_size=5,
            padding=2  # same-padding to preserve sequence length
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Bidirectional LSTM for capturing long-range sequence context.
        # RNA secondary structure (e.g., stem-loops) involves base-pairing
        # between distant positions, making bidirectional processing essential.
        self.lstm = nn.LSTM(
            input_size=cnn_out_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        # Output projection: Bi-LSTM produces 2 * hidden_dim features
        # per position, which we map to 3D coordinates (x, y, z).
        self.fc = nn.Linear(lstm_hidden_dim * 2, output_dim)

    def forward(self, x):
        # x shape: (batch, seq_len, 4)

        # CNN expects (batch, channels, seq_len)
        x = x.transpose(1, 2)
        x = self.relu(self.cnn(x))
        x = self.dropout(x)

        # Back to (batch, seq_len, features) for LSTM
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)

        # Per-nucleotide 3D coordinate prediction
        out = self.fc(lstm_out)
        return out


if __name__ == "__main__":
    print("Testing RNAPredictor3D architecture...")
    model = RNAPredictor3D()
    dummy_input = torch.randn(1, 200, 4)
    print(f"  Input shape:  {dummy_input.shape}")
    predictions = model(dummy_input)
    print(f"  Output shape: {predictions.shape}")
    print("Architecture test passed.")
