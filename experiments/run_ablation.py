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
import argparse
import json
import platform
import re

# Add project root to path so we can import from src/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
import time
import yaml
import matplotlib.pyplot as plt

from src.model_reactivity import (
    RNAReactivityCNNOnly,
    RNAReactivityCNN_LSTM,
    RNAReactivityCNN_LSTM_Transformer,
    RNAReactivityPredictor,
)
from src.data_pipeline_reactivity import RNAReactivityDataset
from src.data_loading import load_unique_sequence_subset
from src.splitting import (
    grouped_split_indices,
    random_split_indices,
    sequence_overlap_counts,
)


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
                  lr=0.001, weight_decay=1e-5, patience=2, factor=0.5,
                  variant_name="model", save_path=None):
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
        total_train_abs = 0.0
        total_train_nt = 0.0

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
            total_train_abs += masked_loss.sum().item()
            total_train_nt += nt_count.item()

        avg_train = total_train_abs / max(total_train_nt, 1.0)

        # --- Cross-Validation (Clipped MAE — Kaggle metric) ---
        model.eval()
        total_cv_abs = 0.0
        total_cv_nt = 0.0
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
                total_cv_abs += masked_loss.sum().item()
                total_cv_nt += nt_count.item()

        avg_cv = total_cv_abs / max(total_cv_nt, 1.0)

        train_losses.append(avg_train)
        cv_losses.append(avg_cv)

        if avg_cv < best_cv_loss:
            best_cv_loss = avg_cv
            best_epoch = epoch + 1
            if save_path:
                torch.save(model.state_dict(), save_path)

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

def parse_args():
    parser = argparse.ArgumentParser(description="Run the RNA reactivity ablation study")
    parser.add_argument(
        "--data",
        default=os.path.join(PROJECT_ROOT, "dataset", "train_data.csv"),
        help="Ribonanza train_data.csv path",
    )
    parser.add_argument(
        "--max-sequences", "--max-samples", dest="max_sequences", type=int, default=1000,
        help="Unique sequences to load with all experiment rows; use 0 for the full dataset",
    )
    parser.add_argument("--epochs", type=int, help="Override config epoch count")
    parser.add_argument(
        "--output-dir", default=os.path.join(PROJECT_ROOT, "experiments"),
        help="Directory for CSV, JSON, plots, and model weights",
    )
    parser.add_argument(
        "--require-vienna", action="store_true",
        help="Fail instead of silently using 4D inputs when ViennaRNA is unavailable",
    )
    parser.add_argument(
        "--split-method", choices=("grouped", "random", "both"), default="grouped",
        help="Validation strategy; 'both' quantifies row-level leakage",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # Load config
    config_path = os.path.join(PROJECT_ROOT, "configs", "ablation_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seed = config.get("seed", 42)
    batch_size = config.get("batch_size", 64)
    num_epochs = args.epochs or config.get("num_epochs", 15)
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

    # Select unique sequences first, then retain every experiment row for them.
    data_path = os.path.abspath(args.data)
    if not os.path.isfile(data_path):
        raise FileNotFoundError(
            f"Ribonanza data not found at {data_path}. Run kaggle/download_data.sh first."
        )
    max_sequences = args.max_sequences or None
    sequence_frame = load_unique_sequence_subset(
        data_path, max_sequences=max_sequences, seed=seed
    )

    dataset_4dim = RNAReactivityDataset(
        dataframe=sequence_frame, max_length=206, use_structure=False,
    )
    n = len(dataset_4dim)

    dataset_7dim = RNAReactivityDataset(
        dataframe=sequence_frame, max_length=206, use_structure=True,
    )
    if args.require_vienna and dataset_7dim.feature_dim != 7:
        raise RuntimeError("ViennaRNA is required, but the RNA Python package is unavailable")
    full_model_dim = dataset_7dim.feature_dim

    split_indices = {}
    if args.split_method in ("random", "both"):
        split_indices["random"] = random_split_indices(n, seed=seed)
    if args.split_method in ("grouped", "both"):
        split_indices["grouped"] = grouped_split_indices(
            dataset_4dim.seq_df, group_col="sequence", seed=seed
        )

    results_dir = os.path.abspath(args.output_dir)
    os.makedirs(results_dir, exist_ok=True)

    split_report = {
        "seed": seed,
        "source": data_path,
        "requested_unique_sequences": max_sequences,
        "filtered_rows": n,
        "filtered_unique_sequences": int(dataset_4dim.seq_df["sequence"].nunique()),
        "splits": {},
    }
    for method, indices in split_indices.items():
        train_idx, cv_idx, test_idx = indices
        split_report["splits"][method] = {
            "rows": {"train": len(train_idx), "cv": len(cv_idx), "test": len(test_idx)},
            "unique_sequences": {
                "train": int(dataset_4dim.seq_df.iloc[train_idx]["sequence"].nunique()),
                "cv": int(dataset_4dim.seq_df.iloc[cv_idx]["sequence"].nunique()),
                "test": int(dataset_4dim.seq_df.iloc[test_idx]["sequence"].nunique()),
            },
            "sequence_overlap": sequence_overlap_counts(dataset_4dim.seq_df, indices),
        }
    with open(os.path.join(results_dir, "split_report.json"), "w") as f:
        json.dump(split_report, f, indent=2)

    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "vienna_features": dataset_7dim.feature_dim == 7,
    }
    with open(os.path.join(results_dir, "environment.json"), "w") as f:
        json.dump(environment, f, indent=2)

    all_results = {}
    epoch_records = []
    variant_specs = [
        ("CNN Only", lambda: RNAReactivityCNNOnly(input_dim=4), dataset_4dim),
        ("CNN + Bi-LSTM", lambda: RNAReactivityCNN_LSTM(input_dim=4), dataset_4dim),
        ("CNN + LSTM + Transformer", lambda: RNAReactivityCNN_LSTM_Transformer(input_dim=4), dataset_4dim),
        (f"Full Model (+ViennaRNA {full_model_dim}d)", lambda: RNAReactivityPredictor(input_dim=full_model_dim), dataset_7dim),
    ]

    for split_method, indices in split_indices.items():
        train_idx, cv_idx, _ = indices
        print(f"\nDataset: {n} rows | unique sequences: "
              f"{dataset_4dim.seq_df['sequence'].nunique()} | split: {split_method}")
        for name, model_factory, dataset in variant_specs:
            train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
            cv_loader = DataLoader(Subset(dataset, cv_idx), batch_size=batch_size, shuffle=False)
            set_seed(seed)
            model = model_factory()
            params = sum(p.numel() for p in model.parameters())
            print(f"\n{'─' * 60}\n  Training: {name} [{split_method}]\n  Parameters: {params:,}\n{'─' * 60}")
            slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
            result = train_variant(
                model=model, train_loader=train_loader, cv_loader=cv_loader,
                device=device, num_epochs=num_epochs, lr=lr,
                weight_decay=weight_decay, patience=patience, factor=factor,
                variant_name=name,
                save_path=os.path.join(results_dir, f"{split_method}_{slug}_best.pth"),
            )
            result["params"] = params
            all_results[(split_method, name)] = result
            for ep, (train_loss, cv_loss) in enumerate(
                zip(result["train_losses"], result["cv_losses"]), start=1
            ):
                epoch_records.append({
                    "split_method": split_method, "variant": name, "epoch": ep,
                    "train_loss": train_loss, "cv_loss": cv_loss,
                })

    # --- Save per-epoch results ---
    epoch_df = pd.DataFrame(epoch_records)
    epoch_csv = os.path.join(results_dir, "ablation_results.csv")
    epoch_df.to_csv(epoch_csv, index=False)
    print(f"\nPer-epoch results saved: {epoch_csv}")

    # --- Save summary ---
    summary_records = []
    for (split_method, name), res in all_results.items():
        summary_records.append({
            'split_method': split_method,
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

    if set(split_indices) == {"random", "grouped"}:
        comparison = summary_df.pivot(index="variant", columns="split_method", values="best_cv_loss")
        comparison["grouped_minus_random"] = comparison["grouped"] - comparison["random"]
        comparison.reset_index().to_csv(
            os.path.join(results_dir, "leakage_comparison.csv"), index=False
        )

    fig, ax = plt.subplots(figsize=(10, 6))
    for (split_method, name), history in all_results.items():
        ax.plot(range(1, len(history["cv_losses"]) + 1), history["cv_losses"], label=f"{name} [{split_method}]")
    ax.set(xlabel="Epoch", ylabel="Clipped MAE", title="RNA reactivity validation curves")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "training_curves.png"), dpi=160)
    plt.close(fig)

    # --- Print summary table ---
    print(f"\n{'=' * 70}")
    print("  ABLATION STUDY RESULTS")
    print(f"{'=' * 70}")
    print(f"{'Split':<9} {'Variant':<35} {'Params':>10} {'Best CV':>10} {'Epoch':>6} {'Time':>8}")
    print(f"{'─' * 70}")
    for _, row in summary_df.iterrows():
        print(f"{row['split_method']:<9} {row['variant']:<35} {row['params']:>10,} "
              f"{row['best_cv_loss']:>10.4f} {row['best_epoch']:>6} "
              f"{row['elapsed_sec']:>7.1f}s")
    print(f"{'─' * 70}")
    print(f"\nSummary saved: {summary_csv}")


if __name__ == "__main__":
    main()
