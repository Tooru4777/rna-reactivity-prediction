"""
Kaggle Training Script — RNA Reactivity Prediction
====================================================
Self-contained script for the Stanford Ribonanza RNA Folding competition.
Designed to run in a Kaggle Notebook environment with GPU acceleration.

Model: CNN (2-layer + BatchNorm) → Bi-LSTM → Transformer Encoder
Input: 7-dimensional features per nucleotide
       - 4 dims: one-hot encoded sequence (A, C, G, U)
       - 3 dims: one-hot encoded secondary structure from ViennaRNA ( '(', ')', '.' )
Output: 2 reactivity values per nucleotide (2A3_MaP and DMS_MaP)

Key design decisions (with rationale):
  1. Dual CNN layers with BatchNorm for better gradient flow and feature hierarchy
  2. Bi-LSTM as implicit positional encoding (avoids sinusoidal PE)
  3. Transformer with residual connection to LSTM output for training stability
  4. L1 Loss (MAE) instead of MSE — more robust to outlier reactivity values
  5. Only clamp targets (not predictions) during training to avoid gradient vanishing
     — see commit c90c63d for the bug discovery and fix

Evaluation metric: Clipped MAE (predictions and targets clipped to [0, 1])
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import os

try:
    import RNA
    HAS_VIENNA = True
except ImportError:
    print("\n[WARNING] ViennaRNA package not found.")
    print("Install with: !pip install viennarna")
    print("Falling back to dummy structure (all dots). "
          "Model will not learn 2D structural features.\n")
    HAS_VIENNA = False


# =====================================================================
# Model Architecture
# =====================================================================

class RNAReactivityPredictor(nn.Module):
    """
    Hybrid deep learning model for per-nucleotide reactivity prediction.

    The architecture processes RNA features through three stages:
      1. CNN:         Local motif extraction (kernel_size=5 ≈ 5nt window)
      2. Bi-LSTM:     Sequential context in both directions
      3. Transformer: Global self-attention for long-range base interactions

    A residual connection from LSTM to Transformer output stabilises training,
    as the Transformer can initially just pass through the LSTM features.
    """

    def __init__(self, input_dim=7, cnn_out_dim=128, lstm_hidden_dim=128,
                 transformer_nhead=8, transformer_layers=2, output_dim=2,
                 dropout=0.3):
        super(RNAReactivityPredictor, self).__init__()

        # --- Stage 1: Dual-layer CNN with BatchNorm ---
        # Two convolution layers create a feature hierarchy:
        # Layer 1 captures primitive patterns (e.g., dinucleotide frequencies)
        # Layer 2 composes them into higher-order motifs
        self.cnn1 = nn.Conv1d(in_channels=input_dim,
                              out_channels=cnn_out_dim // 2, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(cnn_out_dim // 2)
        self.cnn2 = nn.Conv1d(in_channels=cnn_out_dim // 2,
                              out_channels=cnn_out_dim, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(cnn_out_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # --- Stage 2: Bidirectional LSTM ---
        # Processes sequence in both 5'→3' and 3'→5' directions.
        # Also serves as implicit positional encoding for the Transformer.
        self.lstm = nn.LSTM(input_size=cnn_out_dim,
                            hidden_size=lstm_hidden_dim,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=True,
                            dropout=dropout)

        # --- Stage 3: Transformer Encoder (Self-Attention) ---
        # Captures long-range interactions (e.g., base pairs separated by
        # hundreds of nucleotides) that LSTM may struggle with.
        d_model = lstm_hidden_dim * 2  # 256 (bidirectional output)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=transformer_nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                 num_layers=transformer_layers)

        # --- Output projection ---
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # x: (batch, seq_len, 7)

        # CNN: (batch, 7, seq_len) → (batch, 128, seq_len)
        x = x.transpose(1, 2)
        x = self.relu(self.bn1(self.cnn1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.cnn2(x)))
        x = self.dropout(x)
        x = x.transpose(1, 2)

        # LSTM: (batch, seq_len, 128) → (batch, seq_len, 256)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)

        # Transformer + Residual connection
        # The residual allows the model to fall back to LSTM features
        # if the Transformer hasn't learned useful attention patterns yet.
        transformer_out = self.transformer(lstm_out)
        transformer_out = transformer_out + lstm_out  # residual

        # Per-nucleotide reactivity prediction
        out = self.fc(transformer_out)
        return out


# =====================================================================
# Data Pipeline
# =====================================================================

class RNAReactivityDataset(Dataset):
    """
    Dataset for the Stanford Ribonanza RNA Folding competition.

    Features (7 dimensions per nucleotide):
      - Dims 0-3: Sequence one-hot (A, C, G, U)
      - Dims 4-6: Secondary structure one-hot ('(', ')', '.')
                   computed by ViennaRNA's minimum free energy (MFE) algorithm

    Targets (2 dimensions per nucleotide):
      - Channel 0: 2A3_MaP reactivity
      - Channel 1: DMS_MaP reactivity
    """

    def __init__(self, sequences_csv, max_length=206):
        self.max_length = max_length
        self.char_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3,
                         '(': 4, ')': 5, '.': 6}

        if os.path.exists(sequences_csv):
            print(f"Loading Kaggle dataset: {sequences_csv}")
            self.seq_df = pd.read_csv(sequences_csv)

            # Quality filter: keep only high signal-to-noise samples
            if 'SN_filter' in self.seq_df.columns:
                initial_count = len(self.seq_df)
                self.seq_df = self.seq_df[self.seq_df['SN_filter'] == 1.0]
                print(f"Quality filter (SN_filter == 1.0): "
                      f"{initial_count} -> {len(self.seq_df)} samples")

            self.seq_df = self.seq_df.head(50000).reset_index(drop=True)
            self.mock_data = False

            # Pre-compute RNA secondary structures using ViennaRNA MFE
            print("Computing RNA 2D structures via ViennaRNA (this may take ~1 min)...")
            if HAS_VIENNA:
                self.seq_df['structure'] = self.seq_df['sequence'].apply(
                    lambda seq: RNA.fold(seq)[0]
                )
            else:
                self.seq_df['structure'] = self.seq_df['sequence'].apply(
                    lambda seq: '.' * len(seq)
                )
            print("Structure computation complete.")
            self.num_samples = len(self.seq_df)

            # Identify reactivity columns (exclude error columns)
            self.reactivity_cols = [
                c for c in self.seq_df.columns
                if c.startswith('reactivity_') and 'error' not in c
            ]
            print(f"Found {len(self.reactivity_cols)} reactivity columns")
        else:
            print("[WARNING] Kaggle CSV not found. Using synthetic data.")
            self.mock_data = True
            self.num_samples = 1000
            self.reactivity_cols = []

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.mock_data:
            length = np.random.randint(50, self.max_length)
            seq_str = ''.join(np.random.choice(['A', 'C', 'G', 'U'], size=length))
            reactivities = np.random.randn(length, 2).astype(np.float32)
            valid_mask = np.ones((length, 2), dtype=np.float32)
        else:
            seq_str = self.seq_df.iloc[idx].get('sequence', '')
            if pd.isna(seq_str) or len(seq_str) == 0:
                seq_str = 'A' * 50
            length = len(seq_str)

            if len(self.reactivity_cols) > 0:
                row_reactivities = self.seq_df.iloc[idx][self.reactivity_cols].values
                actual_reactivity_len = min(length, len(row_reactivities))

                reactivities = np.zeros((length, 2), dtype=np.float32)
                valid_mask = np.zeros((length, 2), dtype=np.float32)

                # Route reactivity to the correct channel based on experiment type
                experiment_type = self.seq_df.iloc[idx].get('experiment_type', '2A3_MaP')
                exp_idx = 0 if experiment_type == '2A3_MaP' else 1

                for i in range(actual_reactivity_len):
                    val = row_reactivities[i]
                    if not pd.isna(val):
                        reactivities[i, exp_idx] = float(val)
                        valid_mask[i, exp_idx] = 1.0
            else:
                reactivities = np.zeros((length, 2), dtype=np.float32)
                valid_mask = np.zeros((length, 2), dtype=np.float32)

        seq_str = seq_str[:self.max_length]

        # Retrieve secondary structure string
        if self.mock_data:
            struct_str = '.' * len(seq_str)
        else:
            struct_str = self.seq_df.iloc[idx].get(
                'structure', '.' * len(seq_str)
            )[:self.max_length]

        reactivities = reactivities[:self.max_length]
        valid_mask = valid_mask[:self.max_length]
        actual_length = len(seq_str)

        # 7-dimensional one-hot: 4 for sequence + 3 for structure
        one_hot = np.zeros((self.max_length, 7), dtype=np.float32)
        for i in range(actual_length):
            char_seq = seq_str[i]
            char_struct = struct_str[i]
            if char_seq in self.char_map:
                one_hot[i, self.char_map[char_seq]] = 1.0
            if char_struct in self.char_map:
                one_hot[i, self.char_map[char_struct]] = 1.0

        padded_reactivities = np.zeros((self.max_length, 2), dtype=np.float32)
        padded_reactivities[:actual_length] = reactivities

        mask = np.zeros((self.max_length, 2), dtype=np.float32)
        mask[:actual_length] = valid_mask

        return torch.tensor(one_hot), torch.tensor(padded_reactivities), torch.tensor(mask)


# =====================================================================
# Training Loop
# =====================================================================

def train_model():
    print("=== Stanford Ribonanza — Reactivity Prediction Training ===")

    # Try multiple Kaggle data paths (different Kaggle environments)
    possible_paths = [
        "/kaggle/input/competitions/stanford-ribonanza-rna-folding/OLD/train_data.csv",
        "/kaggle/input/competitions/stanford-ribonanza-rna-folding/train_data.csv",
        "/kaggle/input/stanford-ribonanza-rna-folding/train_data.csv"
    ]

    KAGGLE_CSV_PATH = ""
    for path in possible_paths:
        if os.path.exists(path):
            KAGGLE_CSV_PATH = path
            break

    if not KAGGLE_CSV_PATH:
        # Local fallback: use the 1000-sample subset for testing
        KAGGLE_CSV_PATH = "train_data_1000.csv"

    dataset = RNAReactivityDataset(sequences_csv=KAGGLE_CSV_PATH, max_length=206)

    # Train / CV / Test split (70% / 15% / 15%)
    test_size = int(0.15 * len(dataset))
    cv_size = int(0.15 * len(dataset))
    train_size = len(dataset) - cv_size - test_size
    train_dataset, cv_dataset, test_dataset = random_split(
        dataset, [train_size, cv_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    cv_loader = DataLoader(cv_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print(f"Dataset: {len(dataset)} total | "
          f"Train: {train_size} | CV: {cv_size} | Test: {test_size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Architecture: CNN(2-layer+BN) + Bi-LSTM + Transformer (7-dim input)")
    model = RNAReactivityPredictor(input_dim=7).to(device)

    # L1 Loss (MAE): more robust to outlier reactivity values than MSE
    criterion = nn.L1Loss(reduction='none')

    # Adam with weight decay (L2 regularisation) to prevent overfitting
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # Reduce LR when CV loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    num_epochs = 50
    best_cv_loss = float('inf')
    early_stop_patience = 7
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0

        for batch_idx, (sequences, reactivities, masks) in enumerate(train_loader):
            sequences = sequences.to(device)
            reactivities = reactivities.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            predictions = model(sequences)

            # IMPORTANT: Only clamp targets, NOT predictions.
            # Clamping predictions during training causes gradient vanishing:
            # if pred > 1.0 or pred < 0.0, its gradient becomes exactly 0,
            # so the model cannot learn to pull the prediction back in-range.
            # This bug was identified during training when loss stopped decreasing.
            # Fix: clamp only the target values, compute L1 against raw predictions.
            reacts_clipped = torch.clamp(reactivities, 0.0, 1.0)
            loss_matrix = criterion(predictions, reacts_clipped)
            masked_loss = loss_matrix * masks

            actual_nucleotides_count = masks.sum()
            if actual_nucleotides_count > 0:
                final_loss = masked_loss.sum() / actual_nucleotides_count
            else:
                final_loss = torch.tensor(0.0, requires_grad=True).to(device)

            final_loss.backward()
            optimizer.step()

            total_train_loss += final_loss.item()

        avg_train_loss = (total_train_loss / len(train_loader)
                          if len(train_loader) > 0 else 0.0)

        # --- Cross-Validation ---
        model.eval()
        total_cv_loss = 0.0
        with torch.no_grad():
            for sequences, reactivities, masks in cv_loader:
                sequences = sequences.to(device)
                reactivities = reactivities.to(device)
                masks = masks.to(device)
                predictions = model(sequences)

                # Evaluation uses Kaggle's Clipped MAE:
                # both predictions and targets are clipped to [0, 1]
                preds_clipped = torch.clamp(predictions, 0.0, 1.0)
                reacts_clipped = torch.clamp(reactivities, 0.0, 1.0)
                loss_matrix = torch.abs(preds_clipped - reacts_clipped)
                masked_loss = loss_matrix * masks

                actual_nucleotides_count = masks.sum()
                if actual_nucleotides_count > 0:
                    cv_loss = masked_loss.sum() / actual_nucleotides_count
                else:
                    cv_loss = torch.tensor(0.0).to(device)

                total_cv_loss += cv_loss.item()

        avg_cv_loss = (total_cv_loss / len(cv_loader)
                       if len(cv_loader) > 0 else 0.0)

        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Loss: {avg_train_loss:.4f} | CV Loss: {avg_cv_loss:.4f}")

        # --- Early Stopping & Checkpointing ---
        if avg_cv_loss < best_cv_loss:
            best_cv_loss = avg_cv_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "rna_reactivity_model_weights_v2.pth")
            print(f"  -> New best CV Loss ({best_cv_loss:.4f}). Weights saved.")
        else:
            epochs_no_improve += 1
            print(f"  -> No improvement ({epochs_no_improve}/{early_stop_patience})")
            if epochs_no_improve >= early_stop_patience:
                print("  -> Early stopping triggered.")
                break

        scheduler.step(avg_cv_loss)

    print("[DONE] Training complete. Best weights: rna_reactivity_model_weights_v2.pth")

    # --- Final Test Set Evaluation ---
    print("\n=== Test Set Evaluation ===")
    model.load_state_dict(torch.load("rna_reactivity_model_weights_v2.pth"))
    model.eval()
    total_test_loss = 0.0
    with torch.no_grad():
        for sequences, reactivities, masks in test_loader:
            sequences = sequences.to(device)
            reactivities = reactivities.to(device)
            masks = masks.to(device)
            predictions = model(sequences)

            preds_clipped = torch.clamp(predictions, 0.0, 1.0)
            reacts_clipped = torch.clamp(reactivities, 0.0, 1.0)
            loss_matrix = torch.abs(preds_clipped - reacts_clipped)
            masked_loss = loss_matrix * masks
            actual_nucleotides_count = masks.sum()
            if actual_nucleotides_count > 0:
                test_loss = masked_loss.sum() / actual_nucleotides_count
            else:
                test_loss = torch.tensor(0.0).to(device)
            total_test_loss += test_loss.item()

    avg_test_loss = (total_test_loss / len(test_loader)
                     if len(test_loader) > 0 else 0.0)
    print(f"Final Test Loss (Clipped MAE): {avg_test_loss:.4f}")


if __name__ == "__main__":
    train_model()
