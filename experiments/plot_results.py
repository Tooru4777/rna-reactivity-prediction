"""
Visualization — Training Curves & Attention Heatmap
=====================================================
Generates two key visualizations for the ablation study:

1. Training Curves: Loss vs. epoch for all model variants
   (train + CV on the same plot for each variant)

2. 2D Attention Heatmap: Transformer self-attention weights
   for a sample RNA sequence, showing which nucleotide positions
   attend to which other positions.

Prerequisites:
    Run `experiments/run_ablation.py` first to generate ablation_results.csv.

Output:
    - results/training_curves.png
    - results/attention_heatmap.png

Usage:
    python experiments/plot_results.py
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.model_reactivity import RNAReactivityPredictor


# =====================================================================
# 1. Training Curves
# =====================================================================

def plot_training_curves(csv_path, save_path):
    """
    Plot train/CV loss curves for all model variants on a single figure.

    Layout: 2x2 subplot grid, one per variant, with shared y-axis for
    easy comparison. Each subplot shows train (blue) and CV (orange) loss.
    """
    print("Generating training curves...")

    df = pd.read_csv(csv_path)
    variants = df['variant'].unique()

    # Use a clean, publication-ready style
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'legend.fontsize': 9,
        'figure.facecolor': 'white',
    })

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    axes = axes.flatten()

    colors = {
        'train': '#2196F3',  # blue
        'cv': '#FF5722',     # deep orange
    }

    for i, variant in enumerate(variants):
        ax = axes[i]
        vdf = df[df['variant'] == variant]

        ax.plot(vdf['epoch'], vdf['train_loss'],
                color=colors['train'], linewidth=2, marker='o',
                markersize=4, label='Train Loss (MAE)')
        ax.plot(vdf['epoch'], vdf['cv_loss'],
                color=colors['cv'], linewidth=2, marker='s',
                markersize=4, label='CV Loss (Clipped MAE)')

        # Mark best CV epoch
        best_idx = vdf['cv_loss'].idxmin()
        best_epoch = vdf.loc[best_idx, 'epoch']
        best_val = vdf.loc[best_idx, 'cv_loss']
        ax.axvline(best_epoch, color='#4CAF50', linestyle='--',
                   alpha=0.5, linewidth=1)
        ax.annotate(f'Best: {best_val:.4f}',
                    xy=(best_epoch, best_val),
                    xytext=(best_epoch + 0.5, best_val + 0.02),
                    fontsize=8, color='#4CAF50', fontweight='bold')

        ax.set_title(variant, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.5, vdf['epoch'].max() + 0.5)

    fig.suptitle('Ablation Study — Training Curves\n'
                 'RNA Reactivity Prediction (1000 samples, seed=42)',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# =====================================================================
# 2. Attention Heatmap
# =====================================================================

def extract_attention_weights(model, input_tensor):
    """
    Extract attention weights from the Transformer encoder layer.

    Registers a forward hook on the first TransformerEncoderLayer's
    self-attention module to capture the attention weight matrix.

    Returns:
        attention_weights: (num_heads, seq_len, seq_len) numpy array
    """
    attention_weights = {}

    def hook_fn(module, input_args, output):
        # nn.MultiheadAttention returns (attn_output, attn_weights)
        # when need_weights=True
        pass

    # We need to modify the forward to get attention weights
    # Use a different approach: manually call the attention layer
    model.eval()

    # Get intermediate representations up to the transformer input
    with torch.no_grad():
        x = input_tensor
        x = x.transpose(1, 2)
        x = model.relu(model.bn1(model.cnn1(x)))
        x = model.dropout(x)
        x = model.relu(model.bn2(model.cnn2(x)))
        x = model.dropout(x)
        x = x.transpose(1, 2)

        lstm_out, _ = model.lstm(x)
        lstm_out = model.dropout(lstm_out)

        # Now manually get attention from the first transformer layer
        first_layer = model.transformer.layers[0]
        attn_module = first_layer.self_attn

        # Call multihead attention directly with need_weights=True
        attn_output, attn_weights_tensor = attn_module(
            lstm_out, lstm_out, lstm_out,
            need_weights=True,
            average_attn_weights=False  # get per-head weights
        )

    return attn_weights_tensor.squeeze(0).numpy()


def plot_attention_heatmap(save_path, seq_length=50):
    """
    Generate a 2D heatmap of Transformer attention weights for a sample
    RNA sequence.

    The heatmap shows nucleotide-to-nucleotide attention patterns,
    which can reveal learned structural relationships (e.g., base-pairing).

    We show attention from 4 different heads to illustrate how different
    heads may specialise in different structural patterns.
    """
    print("Generating attention heatmap...")

    # Create a sample RNA sequence
    test_sequence = "AUGGCUACGGUCGAAUGCGCUAGCUAGCUAGCUAGCUAGCGGAAUUCCGG"
    actual_len = min(len(test_sequence), seq_length)
    test_sequence = test_sequence[:actual_len]

    # One-hot encode (7-dim, but we use 4-dim model for simplicity)
    char_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
    max_length = 206
    features = np.zeros((max_length, 7), dtype=np.float32)
    for i, c in enumerate(test_sequence):
        if c in char_map:
            features[i, char_map[c]] = 1.0
        features[i, 6] = 1.0  # dummy structure: all dots

    input_tensor = torch.tensor(features).unsqueeze(0)  # (1, 206, 7)

    # Load model (random weights — we're showing the visualisation technique)
    model = RNAReactivityPredictor(input_dim=7)

    # Try to load trained weights if available
    weights_path = os.path.join(PROJECT_ROOT, "rna_reactivity_model_weights.pth")
    if os.path.exists(weights_path):
        try:
            model.load_state_dict(
                torch.load(weights_path, map_location='cpu', weights_only=True)
            )
            print(f"  Loaded trained weights: {weights_path}")
            weight_source = "Trained Model"
        except Exception:
            print("  Using randomly initialised weights (demo mode)")
            weight_source = "Random Initialisation (Demo)"
    else:
        print("  Using randomly initialised weights (demo mode)")
        weight_source = "Random Initialisation (Demo)"

    # Extract attention weights
    attn_weights = extract_attention_weights(model, input_tensor)
    # attn_weights shape: (num_heads, seq_len, seq_len)

    # Crop to actual sequence length
    attn_weights = attn_weights[:, :actual_len, :actual_len]
    num_heads = attn_weights.shape[0]

    # Plot 4 attention heads (or fewer if model has fewer)
    heads_to_show = min(4, num_heads)

    plt.rcParams.update({
        'font.size': 10,
        'figure.facecolor': 'white',
    })

    fig, axes = plt.subplots(1, heads_to_show, figsize=(4 * heads_to_show, 4.5))
    if heads_to_show == 1:
        axes = [axes]

    seq_labels = list(test_sequence)

    for h in range(heads_to_show):
        ax = axes[h]
        im = ax.imshow(
            attn_weights[h],
            cmap='viridis',
            aspect='equal',
            interpolation='nearest'
        )

        # Label every 5th nucleotide for readability
        tick_positions = list(range(0, actual_len, 5))
        tick_labels = [seq_labels[i] + str(i+1) for i in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=7, rotation=90)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels, fontsize=7)

        ax.set_title(f'Head {h+1}', fontweight='bold')
        ax.set_xlabel('Key Position (attended to)')
        if h == 0:
            ax.set_ylabel('Query Position (attending from)')

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f'Transformer Self-Attention Heatmap\n'
        f'Sequence: {test_sequence[:30]}... (len={actual_len})  |  {weight_source}',
        fontsize=12, fontweight='bold', y=1.02
    )
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# =====================================================================
# Main
# =====================================================================

def main():
    results_dir = os.path.join(PROJECT_ROOT, "results")

    # 1. Training curves
    ablation_csv = os.path.join(PROJECT_ROOT, "experiments", "ablation_results.csv")
    if os.path.exists(ablation_csv):
        plot_training_curves(
            csv_path=ablation_csv,
            save_path=os.path.join(results_dir, "training_curves.png")
        )
    else:
        print(f"[SKIP] {ablation_csv} not found. Run run_ablation.py first.")

    # 2. Attention heatmap
    plot_attention_heatmap(
        save_path=os.path.join(results_dir, "attention_heatmap.png")
    )

    print("\nAll visualizations complete.")


if __name__ == "__main__":
    main()
