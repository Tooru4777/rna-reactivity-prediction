"""
Training Script — RNA Reactivity Prediction
=============================================
Standalone training pipeline for the full RNAReactivityPredictor model.

This script trains the full ablation model (CNN + Bi-LSTM + Transformer)
with ViennaRNA secondary structure features (7-dim input).

Training strategy:
  - Loss:       L1 (MAE) with target-only clamping (see README for rationale)
  - Evaluation: Clipped MAE matching the Kaggle competition metric
  - Optimiser:  Adam with L2 regularisation (weight_decay)
  - Scheduler:  ReduceLROnPlateau for adaptive learning rate
  - Split:      70% train / 15% CV / 15% test

Usage:
    # From project root
    python src/train_reactivity.py

    # With custom data path
    python src/train_reactivity.py --data path/to/train.csv --epochs 50

Output:
    - Saves model weights to rna_reactivity_model_weights.pth
    - Prints per-epoch train/CV loss
"""

import sys
import os
import argparse
import time

# Add project root to path when run as script
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np

from src.model_reactivity import RNAReactivityPredictor
from src.data_pipeline_reactivity import RNAReactivityDataset
from src.splitting import (
    grouped_split_indices,
    sequence_overlap_counts,
    validate_split_integrity,
)


def padding_mask_from_features(features):
    return features[..., :4].sum(dim=-1).eq(0)


def set_seed(seed):
    """Fix all random seeds for reproducible training."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Run one training epoch. Returns average training loss."""
    model.train()
    total_abs = 0.0
    total_valid = 0.0

    for features, reactivities, masks in train_loader:
        features = features.to(device)
        reactivities = reactivities.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        predictions = model(features, padding_mask=padding_mask_from_features(features))

        # Only clamp targets — keep raw predictions for gradient flow
        # See README § "Debugging: Gradient Vanishing" for full explanation
        reacts_clipped = torch.clamp(reactivities, 0.0, 1.0)
        loss_matrix = criterion(predictions, reacts_clipped)
        masked_loss = loss_matrix * masks

        nt_count = masks.sum()
        if nt_count > 0:
            final_loss = masked_loss.sum() / nt_count
        else:
            final_loss = torch.tensor(0.0, requires_grad=True, device=device)

        final_loss.backward()
        optimizer.step()
        total_abs += masked_loss.sum().item()
        total_valid += nt_count.item()

    return total_abs / max(total_valid, 1.0)


def evaluate(model, data_loader, device):
    """Evaluate using Kaggle's Clipped MAE metric. Returns average CV loss."""
    model.eval()
    total_abs = 0.0
    total_valid = 0.0

    with torch.no_grad():
        for features, reactivities, masks in data_loader:
            features = features.to(device)
            reactivities = reactivities.to(device)
            masks = masks.to(device)

            predictions = model(
                features, padding_mask=padding_mask_from_features(features)
            )

            # Clipped MAE: clamp BOTH predictions and targets to [0, 1]
            preds_clipped = torch.clamp(predictions, 0.0, 1.0)
            reacts_clipped = torch.clamp(reactivities, 0.0, 1.0)
            loss_matrix = torch.abs(preds_clipped - reacts_clipped)
            masked_loss = loss_matrix * masks

            nt_count = masks.sum()
            if nt_count > 0:
                cv_loss = masked_loss.sum() / nt_count
            else:
                cv_loss = torch.tensor(0.0, device=device)
            total_abs += masked_loss.sum().item()
            total_valid += nt_count.item()

    return total_abs / max(total_valid, 1.0)


def main():
    parser = argparse.ArgumentParser(
        description="Train the RNA Reactivity Prediction model"
    )
    parser.add_argument(
        "--data", type=str,
        default=os.path.join(PROJECT_ROOT, "train_data_1000.csv"),
        help="Path to training CSV"
    )
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="L2 regularisation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-length", type=int, default=206, help="Max sequence length")
    parser.add_argument(
        "--save-path", type=str,
        default=os.path.join(PROJECT_ROOT, "rna_reactivity_model_weights.pth"),
        help="Path to save model weights"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  RNA Reactivity Prediction — Training Pipeline")
    print("=" * 60)

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Seed:   {args.seed}")

    # --- Data Loading ---
    dataset = RNAReactivityDataset(
        sequences_csv=args.data,
        max_length=args.max_length,
        use_structure=True,  # Use ViennaRNA features if available
    )

    # Leakage-safe 70% train / 15% CV / 15% test by unique sequence.
    n = len(dataset)
    train_idx, cv_idx, test_idx = grouped_split_indices(
        dataset.seq_df, group_col="sequence", seed=args.seed
    )
    validate_split_integrity(n, (train_idx, cv_idx, test_idx))
    train_set, cv_set, test_set = (
        Subset(dataset, train_idx), Subset(dataset, cv_idx), Subset(dataset, test_idx)
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    cv_loader = DataLoader(cv_set, batch_size=args.batch_size, shuffle=False)

    print(f"\nDataset: {n} total | Train: {len(train_idx)} | "
          f"CV: {len(cv_idx)} | Test: {len(test_idx)}")
    print(f"Sequence overlap: {sequence_overlap_counts(dataset.seq_df, (train_idx, cv_idx, test_idx))}")
    print(f"Feature dim: {dataset.feature_dim}")

    # --- Model Setup ---
    model = RNAReactivityPredictor(input_dim=dataset.feature_dim).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params:,}")

    criterion = nn.L1Loss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    # --- Training Loop ---
    best_cv_loss = float('inf')
    best_epoch = 0
    start_time = time.time()

    print(f"\n{'Epoch':>6} | {'Train Loss':>12} | {'CV Loss':>12} | {'LR':>10} |")
    print("-" * 52)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        cv_loss = evaluate(model, cv_loader, device)

        current_lr = optimizer.param_groups[0]['lr']
        marker = "  ★ best" if cv_loss < best_cv_loss else ""

        if cv_loss < best_cv_loss:
            best_cv_loss = cv_loss
            best_epoch = epoch
            # Save best model weights
            torch.save(model.state_dict(), args.save_path)

        scheduler.step(cv_loss)

        print(f"{epoch:>6} | {train_loss:>12.4f} | {cv_loss:>12.4f} | {current_lr:>10.6f} |{marker}")

    elapsed = time.time() - start_time

    # --- Summary ---
    print(f"\n{'=' * 52}")
    print(f"Training complete in {elapsed:.1f}s")
    print(f"Best CV Loss: {best_cv_loss:.4f} (epoch {best_epoch})")
    print(f"Weights saved: {args.save_path}")

    # --- Final test set evaluation ---
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    # Reload best weights
    model.load_state_dict(
        torch.load(args.save_path, map_location=device, weights_only=True)
    )
    test_loss = evaluate(model, test_loader, device)
    print(f"Test Loss (Clipped MAE): {test_loss:.4f}")
    print("=" * 52)


if __name__ == "__main__":
    main()
