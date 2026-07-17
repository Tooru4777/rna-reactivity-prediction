"""
RNA Reactivity Prediction — Model Variants for Ablation Study
==============================================================
Four model architectures of increasing complexity, designed for
systematic ablation to quantify the contribution of each component.

Variant hierarchy:
  1. CNNOnly:          CNN baseline (local motifs only)
  2. CNN_LSTM:         + Bi-LSTM (sequential context)
  3. CNN_LSTM_Transformer: + Transformer (self-attention, 4-dim input)
  4. Full Model:       + ViennaRNA 2D structure features (7-dim input)

All variants share the same interface:
  Input:  (batch, seq_len, input_dim)
  Output: (batch, seq_len, 2)  — 2A3_MaP and DMS_MaP reactivity
"""

import torch
import torch.nn as nn


class RNAReactivityCNNOnly(nn.Module):
    """
    Ablation baseline: CNN-only model.

    Two 1D convolution layers capture local sequence motifs (kernel=5 ≈ 5nt
    sliding window), but cannot model long-range base-pairing interactions.
    This establishes the lower bound of what local features alone can achieve.
    """

    def __init__(self, input_dim=7, cnn_out_dim=128, output_dim=2, dropout=0.3):
        super().__init__()
        self.cnn1 = nn.Conv1d(input_dim, cnn_out_dim // 2, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(cnn_out_dim // 2)
        self.cnn2 = nn.Conv1d(cnn_out_dim // 2, cnn_out_dim, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(cnn_out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(cnn_out_dim, output_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = x.transpose(1, 2)
        x = self.relu(self.bn1(self.cnn1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.cnn2(x)))
        x = self.dropout(x)
        x = x.transpose(1, 2)
        return self.fc(x)


class RNAReactivityCNN_LSTM(nn.Module):
    """
    CNN + Bi-LSTM: adds sequential context to local motifs.

    The Bi-LSTM processes CNN features in both 5'→3' and 3'→5' directions,
    capturing dependencies between distant nucleotides (e.g., base-pairing
    in stem-loops). This tests whether sequential context improves over
    local-only features.
    """

    def __init__(self, input_dim=7, cnn_out_dim=128, lstm_hidden_dim=128,
                 output_dim=2, dropout=0.3):
        super().__init__()
        self.cnn1 = nn.Conv1d(input_dim, cnn_out_dim // 2, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(cnn_out_dim // 2)
        self.cnn2 = nn.Conv1d(cnn_out_dim // 2, cnn_out_dim, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(cnn_out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=cnn_out_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.fc = nn.Linear(lstm_hidden_dim * 2, output_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.relu(self.bn1(self.cnn1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.cnn2(x)))
        x = self.dropout(x)
        x = x.transpose(1, 2)

        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        return self.fc(lstm_out)


class RNAReactivityCNN_LSTM_Transformer(nn.Module):
    """
    CNN + Bi-LSTM + Transformer Encoder: adds global self-attention.

    The Transformer's multi-head self-attention can directly attend to any
    pair of positions, capturing long-range base-pairing interactions that
    LSTM may struggle with (especially for sequences >100nt). A residual
    connection from LSTM to Transformer output stabilises early training.

    This variant uses 4-dim input (sequence only, no ViennaRNA features)
    to isolate the architectural contribution from the feature contribution.
    """

    def __init__(self, input_dim=4, cnn_out_dim=128, lstm_hidden_dim=128,
                 transformer_nhead=8, transformer_layers=2, output_dim=2,
                 dropout=0.3):
        super().__init__()
        self.cnn1 = nn.Conv1d(input_dim, cnn_out_dim // 2, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(cnn_out_dim // 2)
        self.cnn2 = nn.Conv1d(cnn_out_dim // 2, cnn_out_dim, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(cnn_out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=cnn_out_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        d_model = lstm_hidden_dim * 2
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=transformer_nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers
        )
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.relu(self.bn1(self.cnn1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.cnn2(x)))
        x = self.dropout(x)
        x = x.transpose(1, 2)

        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)

        transformer_out = self.transformer(lstm_out)
        transformer_out = transformer_out + lstm_out  # residual connection
        return self.fc(transformer_out)


class RNAReactivityPredictor(nn.Module):
    """
    Full model: CNN + Bi-LSTM + Transformer with ViennaRNA 2D structure input.

    This is the complete architecture used for the competition entry.
    It takes 7-dimensional input per nucleotide:
      - 4 dims: one-hot sequence (A, C, G, U)
      - 3 dims: one-hot secondary structure from ViennaRNA MFE ('(', ')', '.')

    The ViennaRNA features provide an explicit structural prior: the model
    knows which nucleotides are paired vs. unpaired before learning reactivity.
    This tests whether domain knowledge (2D structure) improves predictions.

    Architecture:
      1. Dual CNN + BatchNorm:    Local motif extraction
      2. Bi-LSTM (2-layer):       Sequential context (both directions)
      3. Transformer Encoder:     Global self-attention + residual from LSTM
      4. Linear output:           Per-nucleotide reactivity (2A3_MaP, DMS_MaP)
    """

    def __init__(self, input_dim=7, cnn_out_dim=128, lstm_hidden_dim=128,
                 transformer_nhead=8, transformer_layers=2, output_dim=2,
                 dropout=0.3):
        super().__init__()

        # Dual-layer CNN with BatchNorm
        self.cnn1 = nn.Conv1d(input_dim, cnn_out_dim // 2, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(cnn_out_dim // 2)
        self.cnn2 = nn.Conv1d(cnn_out_dim // 2, cnn_out_dim, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(cnn_out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=cnn_out_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        # Transformer Encoder with residual connection
        d_model = lstm_hidden_dim * 2
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=transformer_nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers
        )

        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # x: (batch, seq_len, 7)
        x = x.transpose(1, 2)
        x = self.relu(self.bn1(self.cnn1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.cnn2(x)))
        x = self.dropout(x)
        x = x.transpose(1, 2)

        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)

        transformer_out = self.transformer(lstm_out)
        transformer_out = transformer_out + lstm_out  # residual
        return self.fc(transformer_out)


if __name__ == "__main__":
    print("Testing all model variants...")
    variants = [
        ("CNN Only (7-dim)",           RNAReactivityCNNOnly(input_dim=7), 7),
        ("CNN + LSTM (7-dim)",         RNAReactivityCNN_LSTM(input_dim=7), 7),
        ("CNN + LSTM + Transformer (4-dim)", RNAReactivityCNN_LSTM_Transformer(input_dim=4), 4),
        ("Full Model (7-dim)",         RNAReactivityPredictor(input_dim=7), 7),
    ]

    for name, model, in_dim in variants:
        dummy = torch.randn(2, 206, in_dim)
        out = model(dummy)
        params = sum(p.numel() for p in model.parameters())
        print(f"  {name:45s} | Output: {out.shape} | Params: {params:,}")

    print("\nAll architecture tests passed.")
