"""
Ablation Study — RNA Reactivity Prediction
============================================
Systematically trains 4 model variants of increasing complexity
to quantify the contribution of each architectural component.

Variants:
  1. CNN Only              — local motifs only
  2. CNN + Bi-LSTM         — + sequential context
  3. CNN + LSTM + Transformer (4-dim) — + self-attention, no structural features
  4. Full Model (7-dim)    — + ViennaRNA secondary structure features

Each variant is trained with identical hyperparameters, random seed,
and data split for fair comparison.

Output:
  - experiments/ablation_results.csv  — per-epoch train/CV loss for all variants
  - experiments/ablation_summary.csv  — final best CV loss per variant
  - stdout summary table

Usage:
    python experiments/run_ablation.py
"""

import sys
import os

# Add project root to path so we can import from src/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import pandas as pd
import time
import yaml

from src.model_reactivity import (
    RNAReactivityCNNOnly,
    RNAReactivityCNN_LSTM,
    RNAReactivityCNN_LSTM_Transformer,
    RNAReactivityPredictor,
)
from src.data_pipeline_reactivity import RNAReactivityDataset


# =====================================================================
# Reproducibility
# =====================================================================

def set_seed(seed):
    """Fix all random seeds for reproducible experiments."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =====================================================================
# Training Function
# =====================================================================

def train_variant(model, train_loader, cv_loader, device, num_epochs=15,
                  lr=0.001, weight_decay=1e-5, patience=2, factor=0.5, variant_name="model"):
    """
    Train a single model variant and return per-epoch loss history.

    Uses:
      - L1 Loss (MAE) with padding mask — only clamp targets, not predictions
      - Adam with weight decay (L2 regularisation)
      - ReduceLROnPlateau scheduler

    Returns:
        dict with keys: train_losses, cv_losses, best_cv_loss, best_epoch, elapsed_sec
    """
    model = model.to(device)
    criterion = nn.L1Loss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=factor, patience=patience
    )

    train_losses = []
    cv_losses = []
    best_cv_loss = float('inf')
    best_epoch = 0

    start_time = time.time()

    for epoch in range(num_epochs):
        # --- Training ---
        model.train()
        total_train_loss = 0.0

        for features, reactivities, masks in train_loader:
            features = features.to(device)
            reactivities = reactivities.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            predictions = model(features)

            # Only clamp targets — see README for gradient vanishing explanation
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
            total_train_loss += final_loss.item()

        avg_train = total_train_loss / max(len(train_loader), 1)

        # --- Cross-Validation (Clipped MAE — Kaggle metric) ---
        model.eval()
        total_cv_loss = 0.0
        with torch.no_grad():
            for features, reactivities, masks in cv_loader:
                features = features.to(device)
                reactivities = reactivities.to(device)
                masks = masks.to(device)
                predictions = model(features)

                preds_clipped = torch.clamp(predictions, 0.0, 1.0)
                reacts_clipped = torch.clamp(reactivities, 0.0, 1.0)
                loss_matrix = torch.abs(preds_clipped - reacts_clipped)
                masked_loss = loss_matrix * masks

                nt_count = masks.sum()
                if nt_count > 0:
                    cv_loss = masked_loss.sum() / nt_count
                else:
                    cv_loss = torch.tensor(0.0, device=device)
                total_cv_loss += cv_loss.item()

        avg_cv = total_cv_loss / max(len(cv_loader), 1)

        train_losses.append(avg_train)
        cv_losses.append(avg_cv)

        if avg_cv < best_cv_loss:
            best_cv_loss = avg_cv
            best_epoch = epoch + 1

        scheduler.step(avg_cv)

        print(f"  Epoch [{epoch+1:2d}/{num_epochs}] | "
              f"Train: {avg_train:.4f} | CV: {avg_cv:.4f}"
              f"{'  ★ best' if avg_cv <= best_cv_loss else ''}")

    elapsed = time.time() - start_time

    return {
        'train_losses': train_losses,
        'cv_losses': cv_losses,
        'best_cv_loss': best_cv_loss,
        'best_epoch': best_epoch,
        'elapsed_sec': elapsed,
    }


# =====================================================================
# Main Ablation
# =====================================================================

def main():
    # Load config
    config_path = os.path.join(PROJECT_ROOT, "configs", "ablation_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seed = config.get("seed", 42)
    batch_size = config.get("batch_size", 64)
    num_epochs = config.get("num_epochs", 15)
    lr = config.get("learning_rate", 0.001)
    weight_decay = float(config.get("weight_decay", 1e-5))
    patience = config.get("patience", 2)
    factor = config.get("factor", 0.5)

    print("=" * 60)
    print("  ABLATION STUDY — RNA Reactivity Prediction")
    print("=" * 60)

    set_seed(seed)
    print(f"\nRandom seed: {seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load data (4-dim: no ViennaRNA, for first 3 variants) ---
    data_path = os.path.join(PROJECT_ROOT, "train_data_1000.csv")

    dataset_4dim = RNAReactivityDataset(
        sequences_csv=data_path, max_length=206, use_structure=False
    )

    # Split: 70% train / 15% CV / 15% test
    n = len(dataset_4dim)
    test_size = int(0.15 * n)
    cv_size = int(0.15 * n)
    train_size = n - cv_size - test_size

    # Use same split for all variants
    set_seed(seed)
    train_set_4, cv_set_4, test_set_4 = random_split(
        dataset_4dim, [train_size, cv_size, test_size]
    )

    train_loader_4 = DataLoader(train_set_4, batch_size=batch_size, shuffle=True)
    cv_loader_4 = DataLoader(cv_set_4, batch_size=batch_size, shuffle=False)

    # --- Load data (7-dim: with ViennaRNA fallback, for full model) ---
    # For the full model, we try to use ViennaRNA features.
    # If ViennaRNA is not installed, the dataset falls back to 4-dim.
    dataset_7dim = RNAReactivityDataset(
        sequences_csv=data_path, max_length=206, use_structure=True
    )
    set_seed(seed)
    train_set_7, cv_set_7, test_set_7 = random_split(
        dataset_7dim, [train_size, cv_size, test_size]
    )
    train_loader_7 = DataLoader(train_set_7, batch_size=batch_size, shuffle=True)
    cv_loader_7 = DataLoader(cv_set_7, batch_size=batch_size, shuffle=False)

    # Determine actual feature dim for full model
    full_model_dim = dataset_7dim.feature_dim

    print(f"\nDataset: {n} total | Train: {train_size} | CV: {cv_size} | Test: {test_size}")

    # --- Define variants ---

    variants = [
        {
            'name': 'CNN Only',
            'model': RNAReactivityCNNOnly(input_dim=4),
            'train_loader': train_loader_4,
            'cv_loader': cv_loader_4,
        },
        {
            'name': 'CNN + Bi-LSTM',
            'model': RNAReactivityCNN_LSTM(input_dim=4),
            'train_loader': train_loader_4,
            'cv_loader': cv_loader_4,
        },
        {
            'name': 'CNN + LSTM + Transformer',
            'model': RNAReactivityCNN_LSTM_Transformer(input_dim=4),
            'train_loader': train_loader_4,
            'cv_loader': cv_loader_4,
        },
        {
            'name': f'Full Model (+ViennaRNA {full_model_dim}d)',
            'model': RNAReactivityPredictor(input_dim=full_model_dim),
            'train_loader': train_loader_7,
            'cv_loader': cv_loader_7,
        },
    ]

    # --- Run ablation ---
    all_results = {}
    epoch_records = []

    for variant in variants:
        name = variant['name']
        model = variant['model']
        params = sum(p.numel() for p in model.parameters())

        print(f"\n{'─' * 60}")
        print(f"  Training: {name}")
        print(f"  Parameters: {params:,}")
        print(f"{'─' * 60}")

        set_seed(seed)
        result = train_variant(
            model=model,
            train_loader=variant['train_loader'],
            cv_loader=variant['cv_loader'],
            device=device,
            num_epochs=num_epochs,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
            factor=factor,
            variant_name=name,
        )
        result['params'] = params
        all_results[name] = result

        # Record per-epoch data
        for ep in range(num_epochs):
            if ep < len(result['train_losses']):
                epoch_records.append({
                    'variant': name,
                    'epoch': ep + 1,
                    'train_loss': result['train_losses'][ep],
                    'cv_loss': result['cv_losses'][ep],
                })

    # --- Save per-epoch results ---
    results_dir = os.path.join(PROJECT_ROOT, "experiments")
    os.makedirs(results_dir, exist_ok=True)

    epoch_df = pd.DataFrame(epoch_records)
    epoch_csv = os.path.join(results_dir, "ablation_results.csv")
    epoch_df.to_csv(epoch_csv, index=False)
    print(f"\nPer-epoch results saved: {epoch_csv}")

    # --- Save summary ---
    summary_records = []
    for name, res in all_results.items():
        summary_records.append({
            'variant': name,
            'params': res['params'],
            'best_cv_loss': res['best_cv_loss'],
            'best_epoch': res['best_epoch'],
            'final_train_loss': res['train_losses'][-1],
            'elapsed_sec': res['elapsed_sec'],
        })

    summary_df = pd.DataFrame(summary_records)
    summary_csv = os.path.join(results_dir, "ablation_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    # --- Print summary table ---
    print(f"\n{'=' * 70}")
    print("  ABLATION STUDY RESULTS")
    print(f"{'=' * 70}")
    print(f"{'Variant':<35} {'Params':>10} {'Best CV':>10} {'Epoch':>6} {'Time':>8}")
    print(f"{'─' * 70}")
    for _, row in summary_df.iterrows():
        print(f"{row['variant']:<35} {row['params']:>10,} "
              f"{row['best_cv_loss']:>10.4f} {row['best_epoch']:>6} "
              f"{row['elapsed_sec']:>7.1f}s")
    print(f"{'─' * 70}")
    print(f"\nSummary saved: {summary_csv}")


if __name__ == "__main__":
    main()
