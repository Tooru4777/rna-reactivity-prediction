"""
Exploratory Data Analysis — Stanford Ribonanza RNA Folding
===========================================================
This script analyses the training dataset to understand:
  1. Sequence length distributions
  2. Nucleotide composition across samples
  3. Reactivity value distributions (2A3_MaP vs DMS_MaP)
  4. Correlation between sequence features and reactivity

Run as a script:
    python 01_data_exploration.py

Or convert to a Jupyter notebook:
    pip install jupytext
    jupytext --to notebook 01_data_exploration.py
"""

# %% [markdown]
# # RNA Reactivity Data Exploration
# Analysing the Stanford Ribonanza RNA Folding competition dataset.

# %%
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for headless environments
import matplotlib.pyplot as plt
import os
import sys

# Try to load the dataset
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'train_data_1000.csv')
if not os.path.exists(DATA_PATH):
    print(f"[ERROR] Dataset not found at: {DATA_PATH}")
    print("Please ensure train_data_1000.csv is in the project root.")
    sys.exit(1)

print(f"Loading dataset from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns[:15])}...")

# %% [markdown]
# ## 1. Basic Dataset Overview

# %%
print("\n=== Dataset Info ===")
print(f"Total samples: {len(df)}")
print(f"Number of columns: {len(df.columns)}")

# Identify key columns
if 'sequence' in df.columns:
    print(f"\nSample sequence: {df['sequence'].iloc[0][:50]}...")
    print(f"Sequence lengths: min={df['sequence'].str.len().min()}, "
          f"max={df['sequence'].str.len().max()}, "
          f"mean={df['sequence'].str.len().mean():.1f}")

if 'experiment_type' in df.columns:
    print(f"\nExperiment types:\n{df['experiment_type'].value_counts()}")

if 'SN_filter' in df.columns:
    print(f"\nSignal-to-Noise filter:\n{df['SN_filter'].value_counts()}")

# %% [markdown]
# ## 2. Sequence Length Distribution

# %%
if 'sequence' in df.columns:
    seq_lengths = df['sequence'].str.len()

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.hist(seq_lengths, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    ax.set_xlabel('Sequence Length (nucleotides)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of RNA Sequence Lengths', fontsize=14)
    ax.axvline(seq_lengths.mean(), color='red', linestyle='--',
               label=f'Mean = {seq_lengths.mean():.0f}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'results',
                             'seq_length_distribution.png'), dpi=150)
    plt.close()
    print("Saved: results/seq_length_distribution.png")

# %% [markdown]
# ## 3. Nucleotide Composition

# %%
if 'sequence' in df.columns:
    # Count nucleotide frequencies across all sequences
    all_seqs = ''.join(df['sequence'].values)
    nt_counts = {nt: all_seqs.count(nt) for nt in ['A', 'C', 'G', 'U']}
    total = sum(nt_counts.values())
    nt_fractions = {nt: count / total for nt, count in nt_counts.items()}

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    bars = ax.bar(nt_fractions.keys(), nt_fractions.values(),
                  color=colors, edgecolor='white', width=0.6)
    ax.set_ylabel('Fraction', fontsize=12)
    ax.set_title('Nucleotide Composition', fontsize=14)
    ax.set_ylim(0, 0.4)

    for bar, frac in zip(bars, nt_fractions.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{frac:.3f}', ha='center', fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'results',
                             'nucleotide_composition.png'), dpi=150)
    plt.close()
    print("Saved: results/nucleotide_composition.png")

# %% [markdown]
# ## 4. Reactivity Distribution

# %%
reactivity_cols = [c for c in df.columns
                   if c.startswith('reactivity_') and 'error' not in c]

if len(reactivity_cols) > 0:
    print(f"Found {len(reactivity_cols)} reactivity columns")

    # Gather all reactivity values (excluding NaN)
    all_reactivities = df[reactivity_cols].values.flatten()
    all_reactivities = all_reactivities[~np.isnan(all_reactivities)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Full distribution
    axes[0].hist(all_reactivities, bins=100, color='coral', edgecolor='white', alpha=0.8)
    axes[0].set_xlabel('Reactivity Value', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Reactivity Distribution (All Values)', fontsize=14)
    axes[0].axvline(0, color='black', linestyle='--', alpha=0.5)
    axes[0].axvline(1, color='black', linestyle='--', alpha=0.5)

    # Clipped distribution [0, 1]
    clipped = np.clip(all_reactivities, 0, 1)
    axes[1].hist(clipped, bins=50, color='mediumpurple', edgecolor='white', alpha=0.8)
    axes[1].set_xlabel('Reactivity Value (Clipped to [0, 1])', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('Clipped Reactivity Distribution', fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'results',
                             'reactivity_distribution.png'), dpi=150)
    plt.close()
    print("Saved: results/reactivity_distribution.png")

    # Statistics
    print(f"\nReactivity statistics:")
    print(f"  Total non-NaN values: {len(all_reactivities):,}")
    print(f"  Mean:   {np.mean(all_reactivities):.4f}")
    print(f"  Median: {np.median(all_reactivities):.4f}")
    print(f"  Std:    {np.std(all_reactivities):.4f}")
    print(f"  Min:    {np.min(all_reactivities):.4f}")
    print(f"  Max:    {np.max(all_reactivities):.4f}")
    print(f"  % in [0,1]: {np.mean((all_reactivities >= 0) & (all_reactivities <= 1))*100:.1f}%")
else:
    print("No reactivity columns found in dataset.")

# %% [markdown]
# ## 5. NaN Pattern Analysis

# %%
if len(reactivity_cols) > 0:
    nan_per_col = df[reactivity_cols].isna().sum()
    nan_fractions = nan_per_col / len(df)

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    positions = range(len(nan_fractions))
    ax.bar(positions, nan_fractions.values, color='salmon', width=1.0)
    ax.set_xlabel('Nucleotide Position', fontsize=12)
    ax.set_ylabel('Fraction of NaN', fontsize=12)
    ax.set_title('Missing Reactivity Values by Position', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'results',
                             'nan_pattern.png'), dpi=150)
    plt.close()
    print("Saved: results/nan_pattern.png")

# %% [markdown]
# ## 6. Summary
#
# Key findings:
# - Sequences have variable lengths requiring padding for batched training
# - Reactivity values can be negative or > 1 (competition clips to [0, 1] for eval)
# - NaN values are common at later positions (shorter sequences)
# - Nucleotide composition is roughly balanced

# %%
print("\n=== EDA Complete ===")
print("Generated plots saved to: results/")
