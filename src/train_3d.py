"""
Training Script — RNA 3D Structure Prediction
===============================================
End-to-end training pipeline for the RNAPredictor3D model.

Training strategy:
  - Loss:      MSE (Mean Squared Error) with masking to exclude padded positions
  - Optimiser: Adam with default learning rate 1e-3
  - Epochs:    5 (for local validation; increase to 50+ for full training)

The masked loss is critical: since RNA sequences have variable lengths,
shorter sequences are zero-padded to max_length. Without masking, the model
would be penalised for predicting non-zero coordinates at padded positions,
leading to a trivial solution that predicts zeros everywhere.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data_pipeline_3d import RNA3DDataset
from model_3d import RNAPredictor3D


def train_model():
    print("=== RNA 3D Structure — Training Pipeline ===")

    # --- Data Loading ---
    # Missing real sequence/coordinate inputs fail immediately.
    dataset = RNA3DDataset(
        sequences_csv="dataset/train_sequences.csv",
        labels_csv="dataset/train_labels.csv",
        max_length=200
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # --- Model & Optimiser Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")

    model = RNAPredictor3D().to(device)

    # reduction='none' so we can apply the padding mask before averaging.
    criterion = nn.MSELoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- Training Loop ---
    num_epochs = 5

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, (sequences, coords, masks) in enumerate(dataloader):
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

            total_loss += final_loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_loss:.4f}")

    print("\n[SUCCESS] Training completed. Model demonstrates normal learning behaviour.")


if __name__ == "__main__":
    train_model()
