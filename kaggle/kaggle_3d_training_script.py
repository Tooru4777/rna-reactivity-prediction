"""
Kaggle Training Script — RNA 3D Structure Prediction
=====================================================
Self-contained script for predicting per-nucleotide 3D coordinates (x, y, z)
from RNA sequences. Designed to run in a Kaggle Notebook with GPU acceleration.

Merged from:
  - src/model_3d.py          (model architecture)
  - src/data_pipeline_3d.py  (dataset & preprocessing)
  - src/train_3d.py          (training loop)

Model: CNN (1D Conv, kernel=5) → Bi-LSTM (2-layer)
Input:  4-dimensional one-hot encoded sequence (A, C, G, U)
Output: 3 values per nucleotide (x, y, z coordinates)

Training strategy:
  - Loss:           MSE with padding mask (exclude padded positions)
  - Optimiser:      Adam with weight decay (L2 regularisation)
  - LR Scheduler:   ReduceLROnPlateau (halve LR when CV loss plateaus)
  - Early Stopping: patience=7 epochs
  - Data Split:     70% Train / 15% CV / 15% Test
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import os


# =====================================================================
# Model Architecture
# =====================================================================

class RNAPredictor3D(nn.Module):
    """
    Hybrid CNN + Bi-LSTM model for per-nucleotide 3D coordinate prediction.

    Architecture:
      1. 1D CNN:    Local motif extraction (kernel_size=5 ≈ 5nt sliding window)
      2. Bi-LSTM:   Long-range sequential context in both 5'→3' and 3'→5'

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


# =====================================================================
# Data Pipeline
# =====================================================================

class RNA3DDataset(Dataset):
    """
    PyTorch Dataset for RNA sequence → 3D coordinate prediction.

    Each sample returns:
        - one_hot: (max_length, 4) tensor — one-hot encoded RNA sequence
        - coords:  (max_length, 3) tensor — x, y, z atomic coordinates
        - mask:    (max_length,)   tensor — 1.0 for real nucleotides, 0.0 for padding
    """

    def __init__(self, sequences_csv, labels_csv=None, max_length=200):
        self.max_length = max_length
        self.char_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3}

        if os.path.exists(sequences_csv) and (labels_csv is None or os.path.exists(labels_csv)):
            print(f"Loading data from: {sequences_csv}")
            self.seq_df = pd.read_csv(sequences_csv)
            if labels_csv and os.path.exists(labels_csv):
                self.label_df = pd.read_csv(labels_csv)
            else:
                self.label_df = None
            self.mock_data = False
            self.num_samples = len(self.seq_df)
            print(f"Loaded {self.num_samples} samples")
        else:
            print("[WARNING] Real CSV files not found.")
            print("Generating 1000 synthetic RNA 3D samples for pipeline validation.")
            self.mock_data = True
            self.num_samples = 1000

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.mock_data:
            # Generate synthetic data for testing the pipeline end-to-end
            length = np.random.randint(50, self.max_length)
            seq_str = ''.join(np.random.choice(['A', 'C', 'G', 'U'], size=length))
            coords = np.random.randn(length, 3).astype(np.float32)
        else:
            # Load real sequence and coordinates from Kaggle CSV
            seq_str = self.seq_df.iloc[idx]['sequence']
            length = len(seq_str)

            if self.label_df is not None:
                target_id = self.seq_df.iloc[idx]['target_id']
                target_labels = self.label_df[self.label_df['ID'].str.startswith(target_id)]
                coords = target_labels[['x', 'y', 'z']].values.astype(np.float32)
            else:
                coords = np.zeros((length, 3), dtype=np.float32)

        # Truncate to max_length
        seq_str = seq_str[:self.max_length]
        coords = coords[:self.max_length]
        actual_length = len(seq_str)

        # --- One-hot encoding ---
        # Each nucleotide is represented as a 4-dimensional binary vector.
        # This avoids imposing an ordinal relationship between bases.
        one_hot = np.zeros((self.max_length, 4), dtype=np.float32)
        for i, char in enumerate(seq_str):
            if char in self.char_map:
                one_hot[i, self.char_map[char]] = 1.0

        # --- Zero-padded 3D coordinates ---
        padded_coords = np.zeros((self.max_length, 3), dtype=np.float32)
        padded_coords[:actual_length] = coords

        # --- Padding mask ---
        # Essential for masked loss computation: we must NOT penalise
        # the model for predictions at padded positions.
        mask = np.zeros((self.max_length,), dtype=np.float32)
        mask[:actual_length] = 1.0

        return torch.tensor(one_hot), torch.tensor(padded_coords), torch.tensor(mask)


# =====================================================================
# Training Loop
# =====================================================================

def train_model():
    print("=== RNA 3D Structure — Kaggle Training Pipeline ===")

    # --- Locate dataset ---
    # Try multiple Kaggle data paths (different Kaggle environments)
    possible_seq_paths = [
        "/kaggle/input/stanford-ribonanza-rna-folding/train_sequences.csv",
        "/kaggle/input/competitions/stanford-ribonanza-rna-folding/train_sequences.csv",
        "/kaggle/input/competitions/stanford-ribonanza-rna-folding/OLD/train_sequences.csv",
    ]
    possible_label_paths = [
        "/kaggle/input/stanford-ribonanza-rna-folding/train_labels.csv",
        "/kaggle/input/competitions/stanford-ribonanza-rna-folding/train_labels.csv",
        "/kaggle/input/competitions/stanford-ribonanza-rna-folding/OLD/train_labels.csv",
    ]

    seq_csv = ""
    label_csv = ""
    for sp in possible_seq_paths:
        if os.path.exists(sp):
            seq_csv = sp
            break
    for lp in possible_label_paths:
        if os.path.exists(lp):
            label_csv = lp
            break

    if not seq_csv:
        # Local fallback
        seq_csv = "dataset/train_sequences.csv"
        label_csv = "dataset/train_labels.csv"
        print(f"[INFO] Kaggle paths not found. Using local fallback: {seq_csv}")

    print(f"Sequences CSV: {seq_csv}")
    print(f"Labels CSV:    {label_csv}")

    dataset = RNA3DDataset(
        sequences_csv=seq_csv,
        labels_csv=label_csv if label_csv else None,
        max_length=200
    )

    # --- Train / CV / Test split (70% / 15% / 15%) ---
    test_size = int(0.15 * len(dataset))
    cv_size = int(0.15 * len(dataset))
    train_size = len(dataset) - cv_size - test_size
    train_dataset, cv_dataset, test_dataset = random_split(
        dataset, [train_size, cv_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    cv_loader = DataLoader(cv_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"Dataset: {len(dataset)} total | "
          f"Train: {train_size} | CV: {cv_size} | Test: {test_size}")

    # --- Model & Optimiser Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Architecture: CNN (1D Conv) + Bi-LSTM (2-layer)")
    model = RNAPredictor3D(input_dim=4).to(device)

    # reduction='none' so we can apply the padding mask before averaging.
    criterion = nn.MSELoss(reduction='none')

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
        # --- Training ---
        model.train()
        total_train_loss = 0.0

        for batch_idx, (sequences, coords, masks) in enumerate(train_loader):
            sequences = sequences.to(device)
            coords = coords.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            # Forward pass
            predictions = model(sequences)  # (batch, max_len, 3)

            # Compute per-nucleotide MSE across x, y, z dimensions
            loss_matrix = criterion(predictions, coords)   # (batch, max_len, 3)
            loss_per_nt = loss_matrix.sum(dim=2)            # (batch, max_len)

            # Apply padding mask: zero out loss at padded positions
            masked_loss = loss_per_nt * masks

            # Average only over real (non-padded) nucleotides
            real_nt_count = masks.sum()
            if real_nt_count > 0:
                final_loss = masked_loss.sum() / real_nt_count
            else:
                final_loss = torch.tensor(0.0, requires_grad=True).to(device)

            # Backward pass & parameter update
            final_loss.backward()
            optimizer.step()

            total_train_loss += final_loss.item()

        avg_train_loss = (total_train_loss / len(train_loader)
                          if len(train_loader) > 0 else 0.0)

        # --- Cross-Validation ---
        model.eval()
        total_cv_loss = 0.0
        with torch.no_grad():
            for sequences, coords, masks in cv_loader:
                sequences = sequences.to(device)
                coords = coords.to(device)
                masks = masks.to(device)
                predictions = model(sequences)

                loss_matrix = criterion(predictions, coords)
                loss_per_nt = loss_matrix.sum(dim=2)
                masked_loss = loss_per_nt * masks

                real_nt_count = masks.sum()
                if real_nt_count > 0:
                    cv_loss = masked_loss.sum() / real_nt_count
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
            torch.save(model.state_dict(), "rna_3d_model_weights_v2.pth")
            print(f"  -> New best CV Loss ({best_cv_loss:.4f}). Weights saved.")
        else:
            epochs_no_improve += 1
            print(f"  -> No improvement ({epochs_no_improve}/{early_stop_patience})")
            if epochs_no_improve >= early_stop_patience:
                print("  -> Early stopping triggered.")
                break

        scheduler.step(avg_cv_loss)

    print("[DONE] Training complete. Best weights: rna_3d_model_weights_v2.pth")

    # --- Final Test Set Evaluation ---
    print("\n=== Test Set Evaluation ===")
    model.load_state_dict(torch.load("rna_3d_model_weights_v2.pth"))
    model.eval()
    total_test_loss = 0.0
    with torch.no_grad():
        for sequences, coords, masks in test_loader:
            sequences = sequences.to(device)
            coords = coords.to(device)
            masks = masks.to(device)
            predictions = model(sequences)

            loss_matrix = criterion(predictions, coords)
            loss_per_nt = loss_matrix.sum(dim=2)
            masked_loss = loss_per_nt * masks
            real_nt_count = masks.sum()
            if real_nt_count > 0:
                test_loss = masked_loss.sum() / real_nt_count
            else:
                test_loss = torch.tensor(0.0).to(device)
            total_test_loss += test_loss.item()

    avg_test_loss = (total_test_loss / len(test_loader)
                     if len(test_loader) > 0 else 0.0)
    print(f"Final Test Loss (MSE): {avg_test_loss:.4f}")


if __name__ == "__main__":
    train_model()
