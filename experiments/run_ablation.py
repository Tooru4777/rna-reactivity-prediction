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
  - ablation_results.csv       — per-epoch train/CV loss for all variants
  - ablation_summary.csv       — CV-selected checkpoint and held-out test MAE
  - model_selection.json       — explicit CV selection rule and final test result
  - error_analysis_*.csv       — per-sequence and stratified test errors
  - sequence_length_distribution.csv — exact-length cohort counts
  - sequence_length_histogram.png    — 1-nt unique-sequence histogram
  - data_quality_report.json   — cohort schema, coverage, and target checks
  - split_report.json          — sequence overlap audit
  - run_manifest.json          — code/data cohort fingerprint and hyperparameters

Usage:
    python experiments/run_ablation.py
"""

import sys
import os
import argparse
import hashlib
import json
import platform
import re
import subprocess

# Add project root to path so we can import from src/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
import time
import yaml
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

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
    validate_split_integrity,
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
    torch.use_deterministic_algorithms(True, warn_only=True)


# =====================================================================
# Training Function
# =====================================================================

def padding_mask_from_features(features):
    """Return True at right-padded positions using the four sequence channels."""
    return features[..., :4].sum(dim=-1).eq(0)


def forward_padding_safe(model, features):
    """Run a reactivity model with an explicit padding mask."""
    return model(features, padding_mask=padding_mask_from_features(features))


def format_mean_std(mean, std):
    """Format repeated-run estimates without printing NaN for one-seed smoke tests."""
    return f"{mean:.4f}" if pd.isna(std) else f"{mean:.4f}±{std:.4f}"


def write_sequence_length_artifacts(frame, results_dir):
    """Write exact-length counts and a 1-nt histogram for the retained cohort.

    Sequence counts are computed after deduplicating the experiment-profile
    rows by exact sequence. The CSV includes every integer length from the
    cohort minimum through maximum so absent lengths are explicit zeros.
    """
    if frame.empty:
        raise ValueError("Cannot summarize sequence lengths for an empty cohort")
    if "sequence" not in frame or frame["sequence"].isna().any():
        raise ValueError("Every cohort row must contain an RNA sequence")

    sequences = frame[["sequence"]].drop_duplicates().copy()
    sequences["sequence_length"] = sequences["sequence"].astype(str).str.len()
    profile_lengths = frame["sequence"].astype(str).str.len()
    minimum = int(sequences["sequence_length"].min())
    maximum = int(sequences["sequence_length"].max())
    lengths = pd.Index(range(minimum, maximum + 1), name="sequence_length")

    unique_counts = sequences["sequence_length"].value_counts().reindex(lengths, fill_value=0)
    profile_counts = profile_lengths.value_counts().reindex(lengths, fill_value=0)
    distribution = pd.DataFrame({
        "sequence_length": lengths.astype(int),
        "unique_sequences": unique_counts.astype(int).to_numpy(),
        "experiment_profiles": profile_counts.astype(int).to_numpy(),
    })
    total_sequences = int(distribution["unique_sequences"].sum())
    distribution["share_of_sequences"] = (
        distribution["unique_sequences"] / total_sequences
    )
    distribution["cumulative_unique_sequences"] = (
        distribution["unique_sequences"].cumsum()
    )

    os.makedirs(results_dir, exist_ok=True)
    distribution.to_csv(
        os.path.join(results_dir, "sequence_length_distribution.csv"), index=False
    )

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(
        distribution["sequence_length"],
        distribution["unique_sequences"],
        width=0.9,
        color="#3973ac",
        edgecolor="#244a70",
        linewidth=0.6,
    )
    ax.set_title("RNA sequence length distribution", loc="left", fontsize=15, pad=28)
    ax.text(
        0.0, 1.015,
        f"Quality-filtered cohort • unique sequences (n={total_sequences:,}) • exact 1-nt bins",
        transform=ax.transAxes, color="#4a5560", fontsize=10, va="bottom",
    )
    ax.set_xlabel("Sequence length (nt)")
    ax.set_ylabel("Unique sequences")
    ax.set_xlim(minimum - 1, maximum + 1)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=12))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    observed = distribution["unique_sequences"] > 0
    if int(observed.sum()) <= 20:
        for bar, count in zip(bars, distribution["unique_sequences"]):
            if count:
                ax.annotate(
                    f"{int(count):,}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, color="#24313d",
                )

    fig.tight_layout()
    fig.savefig(
        os.path.join(results_dir, "sequence_length_histogram.png"), dpi=160
    )
    plt.close(fig)
    return distribution


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
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"  Using DataParallel across {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
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
            predictions = forward_padding_safe(model, features)

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
                predictions = forward_padding_safe(model, features)

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
                model_to_save = model.module if isinstance(model, nn.DataParallel) else model
                torch.save(model_to_save.state_dict(), save_path)

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


def evaluate_checkpoint(model_factory, checkpoint_path, data_loader, device,
                        dataframe, row_indices, max_length=206):
    """Evaluate a CV-selected checkpoint on a held-out partition.

    Returns the nucleotide-weighted clipped MAE plus per-row and
    paired/unpaired records used by the error-analysis reports.
    """
    model = model_factory().to(device)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.eval()

    total_abs = 0.0
    total_valid = 0.0
    per_row = []
    structure_rows = []
    cursor = 0

    with torch.no_grad():
        for features, targets, masks in data_loader:
            features = features.to(device)
            targets = targets.to(device)
            masks = masks.to(device)
            predictions = torch.clamp(forward_padding_safe(model, features), 0.0, 1.0)
            targets = torch.clamp(targets, 0.0, 1.0)
            errors = torch.abs(predictions - targets) * masks

            total_abs += errors.sum().item()
            total_valid += masks.sum().item()
            batch_size = features.size(0)
            batch_indices = row_indices[cursor:cursor + batch_size]
            cursor += batch_size

            for batch_pos, row_idx in enumerate(batch_indices):
                row = dataframe.iloc[int(row_idx)]
                sequence = str(row.get("sequence", ""))[:max_length]
                valid = masks[batch_pos]
                sample_errors = errors[batch_pos]
                valid_count = valid.sum().item()
                sequence_id = row.get("sequence_id", None)
                if sequence_id is None or pd.isna(sequence_id):
                    sequence_id = "missing"
                sequence_hash = hashlib.sha1(sequence.encode("utf-8")).hexdigest()[:12]
                per_row.append({
                    "row_index": int(row_idx),
                    "sequence_id": str(sequence_id),
                    "sequence_hash": sequence_hash,
                    "sequence_length": len(sequence),
                    "experiment_type": str(row.get("experiment_type", "unknown")),
                    "valid_positions": int(valid_count),
                    "absolute_error_sum": sample_errors.sum().item(),
                    "mae": sample_errors.sum().item() / max(valid_count, 1.0),
                })

                structure = str(row.get("structure", "." * len(sequence)))[:len(sequence)]
                for label, selected in (
                    ("paired", torch.tensor(
                        [char in "()" for char in structure], device=device, dtype=torch.bool
                    )),
                    ("unpaired", torch.tensor(
                        [char == "." for char in structure], device=device, dtype=torch.bool
                    )),
                ):
                    if selected.numel() == 0:
                        continue
                    selected_mask = valid[:len(sequence)] * selected[:, None]
                    selected_count = selected_mask.sum().item()
                    if selected_count:
                        structure_rows.append({
                            "row_index": int(row_idx),
                            "structure_class": label,
                            "valid_positions": int(selected_count),
                            "absolute_error_sum": (
                                sample_errors[:len(sequence)] * selected[:, None]
                            ).sum().item(),
                        })

    return total_abs / max(total_valid, 1.0), per_row, structure_rows


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
        help="Unique sequences to load with all quality-eligible experiment rows; use 0 for full data",
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
    parser.add_argument(
        "--seeds", nargs="+", type=int,
        help="Validation seeds; defaults to the single seed in the config",
    )
    parser.add_argument(
        "--sample-seed", type=int, default=42,
        help="Seed used once to select the quality-filtered sequence cohort",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # Load config
    config_path = os.path.join(PROJECT_ROOT, "configs", "ablation_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seeds = args.seeds or [config.get("seed", 42)]
    batch_size = config.get("batch_size", 64)
    num_epochs = args.epochs or config.get("num_epochs", 15)
    lr = config.get("learning_rate", 0.001)
    weight_decay = float(config.get("weight_decay", 1e-5))
    patience = config.get("patience", 2)
    factor = config.get("factor", 0.5)

    print("=" * 60)
    print("  ABLATION STUDY — RNA Reactivity Prediction")
    print("=" * 60)

    set_seed(seeds[0])
    print(f"\nValidation seeds: {seeds} | sample seed: {args.sample_seed}")

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
        data_path, max_sequences=max_sequences, seed=args.sample_seed
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

    quality_frame = dataset_4dim.seq_df
    reactivity_cols = dataset_4dim.reactivity_cols
    target_values = quality_frame[reactivity_cols]
    cohort_lengths = (
        quality_frame["sequence"].astype(str).drop_duplicates().str.len()
    )
    experiments_per_sequence = quality_frame.groupby("sequence")[
        "experiment_type"
    ].nunique()
    data_quality_report = {
        "grain": "one quality-eligible experiment profile per retained CSV row",
        "rows": int(len(quality_frame)),
        "columns": int(len(quality_frame.columns)),
        "unique_sequences": int(quality_frame["sequence"].nunique()),
        "experiment_type_rows": {
            str(key): int(value) for key, value in
            quality_frame["experiment_type"].value_counts().sort_index().items()
        },
        "sequences_with_both_experiments": int((experiments_per_sequence == 2).sum()),
        "sequences_with_one_experiment": int((experiments_per_sequence == 1).sum()),
        "duplicate_full_rows": int(quality_frame.duplicated().sum()),
        "duplicate_sequence_experiment_pairs": int(
            quality_frame.duplicated(["sequence", "experiment_type"]).sum()
        ),
        "reactivity_columns": len(reactivity_cols),
        "valid_reactivity_targets": int(target_values.notna().sum().sum()),
        "reactivity_targets_below_zero": int((target_values < 0).sum().sum()),
        "reactivity_targets_above_one": int((target_values > 1).sum().sum()),
        "sequence_length": {
            "grain": "unique exact sequence",
            "min": int(cohort_lengths.min()),
            "median": float(cohort_lengths.median()),
            "max": int(cohort_lengths.max()),
        },
        "notes": [
            "Out-of-range targets are retained in source data and clipped only for the competition MAE.",
            "Duplicate sequence/experiment pairs are reported rather than dropped because distinct experimental profiles may be legitimate.",
        ],
    }

    splits_by_seed = {}
    for seed in seeds:
        split_indices = {}
        if args.split_method in ("random", "both"):
            split_indices["random"] = random_split_indices(n, seed=seed)
        if args.split_method in ("grouped", "both"):
            split_indices["grouped"] = grouped_split_indices(
                dataset_4dim.seq_df, group_col="sequence", seed=seed
            )
        splits_by_seed[seed] = split_indices

    results_dir = os.path.abspath(args.output_dir)
    os.makedirs(results_dir, exist_ok=True)
    write_sequence_length_artifacts(quality_frame, results_dir)
    with open(os.path.join(results_dir, "data_quality_report.json"), "w") as f:
        json.dump(data_quality_report, f, indent=2)

    split_report = {
        "sample_seed": args.sample_seed,
        "validation_seeds": seeds,
        "source": data_path,
        "requested_unique_sequences": max_sequences,
        "filtered_rows": n,
        "filtered_unique_sequences": int(dataset_4dim.seq_df["sequence"].nunique()),
        "splits_by_seed": {},
    }
    for seed, split_indices in splits_by_seed.items():
        split_report["splits_by_seed"][str(seed)] = {}
        for method, indices in split_indices.items():
            train_idx, cv_idx, test_idx = indices
            validate_split_integrity(n, indices)
            overlap = sequence_overlap_counts(dataset_4dim.seq_df, indices)
            if method == "grouped" and any(overlap.values()):
                raise RuntimeError(f"Grouped split leaked RNA sequences: {overlap}")
            split_report["splits_by_seed"][str(seed)][method] = {
                "rows": {"train": len(train_idx), "cv": len(cv_idx), "test": len(test_idx)},
                "unique_sequences": {
                    "train": int(dataset_4dim.seq_df.iloc[train_idx]["sequence"].nunique()),
                    "cv": int(dataset_4dim.seq_df.iloc[cv_idx]["sequence"].nunique()),
                    "test": int(dataset_4dim.seq_df.iloc[test_idx]["sequence"].nunique()),
                },
                "sequence_overlap": overlap,
            }
    with open(os.path.join(results_dir, "split_report.json"), "w") as f:
        json.dump(split_report, f, indent=2)

    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_names": [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ] if torch.cuda.is_available() else [],
        "data_parallel": torch.cuda.is_available() and torch.cuda.device_count() > 1,
        "vienna_features": dataset_7dim.feature_dim == 7,
    }
    with open(os.path.join(results_dir, "environment.json"), "w") as f:
        json.dump(environment, f, indent=2)

    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None
    cohort_sequences = sorted(dataset_4dim.seq_df["sequence"].astype(str).unique())
    run_manifest = {
        "artifact_schema_version": 2,
        "git_commit": git_commit,
        "data_file": os.path.basename(data_path),
        "cohort_sha256": hashlib.sha256(
            "\n".join(cohort_sequences).encode("utf-8")
        ).hexdigest(),
        "filtered_rows": n,
        "filtered_unique_sequences": len(cohort_sequences),
        "sample_seed": args.sample_seed,
        "validation_seeds": seeds,
        "split_method": args.split_method,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "scheduler_patience": patience,
        "scheduler_factor": factor,
        "metric": "nucleotide-weighted MAE after clipping predictions and targets to [0, 1]",
        "determinism": "fixed seeds and deterministic algorithms with warn_only=True",
    }
    with open(os.path.join(results_dir, "run_manifest.json"), "w") as f:
        json.dump(run_manifest, f, indent=2)

    all_results = {}
    epoch_records = []
    test_records = []
    error_records = []
    structure_error_records = []
    variant_specs = [
        ("CNN Only", lambda: RNAReactivityCNNOnly(input_dim=4), dataset_4dim),
        ("CNN + Bi-LSTM", lambda: RNAReactivityCNN_LSTM(input_dim=4), dataset_4dim),
        ("CNN + LSTM + Transformer", lambda: RNAReactivityCNN_LSTM_Transformer(input_dim=4), dataset_4dim),
        (f"Full Model (+ViennaRNA {full_model_dim}d)", lambda: RNAReactivityPredictor(input_dim=full_model_dim), dataset_7dim),
    ]

    for seed, split_indices in splits_by_seed.items():
        for split_method, indices in split_indices.items():
            train_idx, cv_idx, test_idx = indices
            print(f"\nDataset: {n} rows | unique sequences: "
                  f"{dataset_4dim.seq_df['sequence'].nunique()} | "
                  f"seed: {seed} | split: {split_method}")
            for name, model_factory, dataset in variant_specs:
                train_loader = DataLoader(
                    Subset(dataset, train_idx), batch_size=batch_size, shuffle=True
                )
                cv_loader = DataLoader(
                    Subset(dataset, cv_idx), batch_size=batch_size, shuffle=False
                )
                set_seed(seed)
                model = model_factory()
                params = sum(p.numel() for p in model.parameters())
                print(f"\n{'─' * 60}\n  Training: {name} "
                      f"[{split_method}, seed={seed}]\n  Parameters: {params:,}\n{'─' * 60}")
                slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
                checkpoint_path = os.path.join(
                    results_dir, f"seed_{seed}_{split_method}_{slug}_best.pth"
                )
                result = train_variant(
                    model=model, train_loader=train_loader, cv_loader=cv_loader,
                    device=device, num_epochs=num_epochs, lr=lr,
                    weight_decay=weight_decay, patience=patience, factor=factor,
                    variant_name=name,
                    save_path=checkpoint_path,
                )
                test_loader = DataLoader(
                    Subset(dataset, test_idx), batch_size=batch_size, shuffle=False
                )
                test_mae, row_errors, structure_errors = evaluate_checkpoint(
                    model_factory=model_factory,
                    checkpoint_path=checkpoint_path,
                    data_loader=test_loader,
                    device=device,
                    dataframe=dataset_7dim.seq_df,
                    row_indices=list(test_idx),
                    max_length=dataset.max_length,
                )
                result["test_mae"] = test_mae
                test_records.append({
                    "seed": seed,
                    "split_method": split_method,
                    "variant": name,
                    "test_mae": test_mae,
                })
                for record in row_errors:
                    error_records.append({
                        "seed": seed, "split_method": split_method,
                        "variant": name, **record,
                    })
                for record in structure_errors:
                    structure_error_records.append({
                        "seed": seed, "split_method": split_method,
                        "variant": name, **record,
                    })
                result["params"] = params
                all_results[(seed, split_method, name)] = result
                for ep, (train_loss, cv_loss) in enumerate(
                    zip(result["train_losses"], result["cv_losses"]), start=1
                ):
                    epoch_records.append({
                        "seed": seed, "split_method": split_method,
                        "variant": name, "epoch": ep,
                        "train_loss": train_loss, "cv_loss": cv_loss,
                    })

    # --- Save per-epoch results ---
    epoch_df = pd.DataFrame(epoch_records)
    epoch_csv = os.path.join(results_dir, "ablation_results.csv")
    epoch_df.to_csv(epoch_csv, index=False)
    print(f"\nPer-epoch results saved: {epoch_csv}")

    # --- Save summary ---
    summary_records = []
    for (seed, split_method, name), res in all_results.items():
        summary_records.append({
            'seed': seed,
            'split_method': split_method,
            'variant': name,
            'params': res['params'],
            'best_cv_loss': res['best_cv_loss'],
            'best_epoch': res['best_epoch'],
            'test_mae': res['test_mae'],
            'final_train_loss': res['train_losses'][-1],
            'elapsed_sec': res['elapsed_sec'],
        })

    summary_df = pd.DataFrame(summary_records)
    summary_csv = os.path.join(results_dir, "ablation_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    aggregate_df = (
        summary_df.groupby(["split_method", "variant"], as_index=False)
        .agg(
            mean_cv_loss=("best_cv_loss", "mean"),
            std_cv_loss=("best_cv_loss", "std"),
            min_cv_loss=("best_cv_loss", "min"),
            max_cv_loss=("best_cv_loss", "max"),
            seeds=("seed", "nunique"),
            mean_test_mae=("test_mae", "mean"),
            std_test_mae=("test_mae", "std"),
        )
    )
    aggregate_df.to_csv(os.path.join(results_dir, "ablation_aggregate.csv"), index=False)

    grouped_candidates = aggregate_df[aggregate_df["split_method"] == "grouped"]
    selection_pool = grouped_candidates if not grouped_candidates.empty else aggregate_df
    selected_row = selection_pool.sort_values("mean_cv_loss").iloc[0]
    model_selection = {
        "selection_rule": "lowest mean CV MAE across validation seeds",
        "selected_split_method": str(selected_row["split_method"]),
        "selected_variant": str(selected_row["variant"]),
        "mean_cv_mae": float(selected_row["mean_cv_loss"]),
        "std_cv_mae": (
            None if pd.isna(selected_row["std_cv_loss"])
            else float(selected_row["std_cv_loss"])
        ),
        "mean_held_out_test_mae": float(selected_row["mean_test_mae"]),
        "std_held_out_test_mae": (
            None if pd.isna(selected_row["std_test_mae"])
            else float(selected_row["std_test_mae"])
        ),
        "seeds": int(selected_row["seeds"]),
        "caveat": (
            "Each seed defines a different held-out partition; test values are a "
            "repeated holdout estimate, not a Kaggle leaderboard score."
        ),
    }
    with open(os.path.join(results_dir, "model_selection.json"), "w") as f:
        json.dump(model_selection, f, indent=2)

    test_df = pd.DataFrame(test_records)
    test_df.to_csv(os.path.join(results_dir, "test_results.csv"), index=False)
    error_df = pd.DataFrame(error_records)
    error_df["length_bin"] = pd.cut(
        error_df["sequence_length"], bins=[0, 100, 150, 206],
        labels=["<=100", "101-150", "151-206"], include_lowest=True,
    )
    error_df.to_csv(os.path.join(results_dir, "error_analysis_per_profile.csv"), index=False)
    sequence_error_df = error_df.groupby(
        ["seed", "split_method", "variant", "sequence_hash", "sequence_length"],
        as_index=False,
    ).agg(
        experiment_profiles=("row_index", "size"),
        valid_positions=("valid_positions", "sum"),
        absolute_error_sum=("absolute_error_sum", "sum"),
    )
    sequence_error_df["mae"] = (
        sequence_error_df["absolute_error_sum"] / sequence_error_df["valid_positions"]
    )
    sequence_error_df.to_csv(
        os.path.join(results_dir, "error_analysis_per_sequence.csv"), index=False
    )

    error_by_length = error_df.groupby(
        ["seed", "split_method", "variant", "length_bin"],
        observed=True, as_index=False,
    ).agg(
        sequences=("sequence_hash", "nunique"),
        rows=("row_index", "size"),
        valid_positions=("valid_positions", "sum"),
        absolute_error_sum=("absolute_error_sum", "sum"),
    )
    error_by_length["weighted_mae"] = (
        error_by_length["absolute_error_sum"] / error_by_length["valid_positions"]
    )
    error_by_length.to_csv(os.path.join(results_dir, "error_by_length.csv"), index=False)

    error_by_experiment = error_df.groupby(
        ["seed", "split_method", "variant", "experiment_type"], as_index=False
    ).agg(
        rows=("row_index", "size"),
        valid_positions=("valid_positions", "sum"),
        absolute_error_sum=("absolute_error_sum", "sum"),
    )
    error_by_experiment["weighted_mae"] = (
        error_by_experiment["absolute_error_sum"]
        / error_by_experiment["valid_positions"]
    )
    error_by_experiment.to_csv(
        os.path.join(results_dir, "error_by_experiment.csv"), index=False
    )

    structure_error_df = pd.DataFrame(structure_error_records)
    structure_summary = (
        structure_error_df.groupby(
            ["seed", "split_method", "variant", "structure_class"], as_index=False
        )
        .agg(
            valid_positions=("valid_positions", "sum"),
            absolute_error_sum=("absolute_error_sum", "sum"),
        )
    )
    structure_summary["mae"] = (
        structure_summary["absolute_error_sum"] / structure_summary["valid_positions"]
    )
    structure_summary.to_csv(os.path.join(results_dir, "error_by_structure.csv"), index=False)

    split_methods = set(summary_df["split_method"])
    if split_methods == {"random", "grouped"}:
        comparison = summary_df.pivot(
            index=["seed", "variant"], columns="split_method", values="best_cv_loss"
        )
        comparison["grouped_minus_random"] = comparison["grouped"] - comparison["random"]
        comparison = comparison.reset_index()
        comparison.to_csv(os.path.join(results_dir, "leakage_comparison.csv"), index=False)
        leakage_aggregate = (
            comparison.groupby("variant", as_index=False)
            .agg(
                mean_random=("random", "mean"),
                mean_grouped=("grouped", "mean"),
                mean_grouped_minus_random=("grouped_minus_random", "mean"),
                std_grouped_minus_random=("grouped_minus_random", "std"),
                seeds=("seed", "nunique"),
            )
        )
        leakage_aggregate.to_csv(
            os.path.join(results_dir, "leakage_aggregate.csv"), index=False
        )

    transformer_name = "CNN + LSTM + Transformer"
    full_name = f"Full Model (+ViennaRNA {full_model_dim}d)"
    model_scores = summary_df.pivot(
        index=["seed", "split_method"], columns="variant", values="best_cv_loss"
    )
    vienna_comparison = model_scores[[transformer_name, full_name]].copy()
    vienna_comparison.columns = ["transformer", "with_vienna"]
    vienna_comparison["absolute_improvement"] = (
        vienna_comparison["transformer"] - vienna_comparison["with_vienna"]
    )
    vienna_comparison["relative_improvement"] = (
        vienna_comparison["absolute_improvement"] / vienna_comparison["transformer"]
    )
    vienna_comparison = vienna_comparison.reset_index()
    vienna_comparison.to_csv(
        os.path.join(results_dir, "vienna_comparison.csv"), index=False
    )
    vienna_aggregate = (
        vienna_comparison.groupby("split_method", as_index=False)
        .agg(
            mean_transformer=("transformer", "mean"),
            mean_with_vienna=("with_vienna", "mean"),
            mean_absolute_improvement=("absolute_improvement", "mean"),
            std_absolute_improvement=("absolute_improvement", "std"),
            mean_relative_improvement=("relative_improvement", "mean"),
            seeds=("seed", "nunique"),
        )
    )
    vienna_aggregate.to_csv(
        os.path.join(results_dir, "vienna_aggregate.csv"), index=False
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    mean_curves = (
        epoch_df.groupby(["split_method", "variant", "epoch"], as_index=False)
        .agg(mean_cv_loss=("cv_loss", "mean"))
    )
    for (split_method, name), curve in mean_curves.groupby(["split_method", "variant"]):
        ax.plot(curve["epoch"], curve["mean_cv_loss"], label=f"{name} [{split_method}]")
    ax.set(xlabel="Epoch", ylabel="Clipped MAE", title="RNA reactivity validation curves")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "training_curves.png"), dpi=160)
    plt.close(fig)

    selected_errors = error_by_length[
        (error_by_length["split_method"] == model_selection["selected_split_method"])
        & (error_by_length["variant"] == model_selection["selected_variant"])
    ]
    length_plot = selected_errors.groupby("length_bin", observed=True, as_index=False).agg(
        valid_positions=("valid_positions", "sum"),
        absolute_error_sum=("absolute_error_sum", "sum"),
    )
    length_plot["mae"] = (
        length_plot["absolute_error_sum"] / length_plot["valid_positions"]
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(length_plot["length_bin"].astype(str), length_plot["mae"], color="#3973ac")
    ax.set(
        xlabel="Sequence length (nt)", ylabel="Held-out clipped MAE",
        title=f"Error by sequence length: {model_selection['selected_variant']}",
    )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "error_by_length.png"), dpi=160)
    plt.close(fig)

    # --- Print summary table ---
    print(f"\n{'=' * 70}")
    print("  ABLATION STUDY RESULTS")
    print(f"{'=' * 70}")
    print(f"{'Split':<9} {'Variant':<35} {'CV MAE':>14} {'Test MAE':>14}")
    print(f"{'─' * 70}")
    for _, row in aggregate_df.iterrows():
        print(f"{row['split_method']:<9} {row['variant']:<35} "
              f"{format_mean_std(row['mean_cv_loss'], row['std_cv_loss']):>14} "
              f"{format_mean_std(row['mean_test_mae'], row['std_test_mae']):>14}")
    print(f"{'─' * 70}")
    print(f"\nSummary saved: {summary_csv}")
    test_std = model_selection["std_held_out_test_mae"]
    test_display = f"{model_selection['mean_held_out_test_mae']:.4f}"
    if test_std is not None:
        test_display += f"±{test_std:.4f}"
    print(f"Selected model: {model_selection['selected_variant']} | "
          f"held-out test MAE: {test_display}")


if __name__ == "__main__":
    main()
