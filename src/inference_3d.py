"""
Inference & Visualisation — RNA 3D Structure
=============================================
Loads a trained RNAPredictor3D model and generates a 3D backbone visualisation
for a given RNA sequence.

Usage:
    python inference_3d.py

The script:
  1. Converts an input RNA sequence to a one-hot tensor
  2. Loads pre-trained weights (.pth file) — works on CPU without a GPU
  3. Runs a forward pass to predict per-nucleotide (x, y, z) coordinates
  4. Renders a 3D matplotlib plot with 5' (green) and 3' (red) endpoints
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

from model_3d import RNAPredictor3D


def sequence_to_onehot(seq_str, max_length=200):
    """Convert an RNA sequence string to a one-hot encoded tensor."""
    char_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
    seq_str = seq_str[:max_length]
    one_hot = np.zeros((max_length, 4), dtype=np.float32)
    for i, char in enumerate(seq_str.upper()):
        if char in char_map:
            one_hot[i, char_map[char]] = 1.0

    # Add batch dimension: (1, max_length, 4)
    return torch.tensor(one_hot).unsqueeze(0)


def visualize_rna_3d(sequence, weights_path="rna_3d_model_weights.pth",
                     save_path="../results/predicted_rna_3d.png",
                     allow_random_demo=False):
    """Predict and visualise the 3D backbone of an RNA sequence."""
    print("=== RNA 3D Structure Inference ===")
    model = RNAPredictor3D()

    if os.path.exists(weights_path):
        print(f"Loaded trained weights from: {weights_path}")
        model.load_state_dict(
            torch.load(weights_path, map_location=torch.device('cpu'))
        )
    else:
        if not allow_random_demo:
            raise FileNotFoundError(
                f"Weights file not found: {weights_path}. "
                "Use allow_random_demo=True only for a visualisation smoke test."
            )
        print("[SMOKE TEST] Using random parameters; this is not a prediction.")

    model.eval()

    print(f"\nInput RNA sequence (length {len(sequence)}): {sequence}")
    input_tensor = sequence_to_onehot(sequence)

    # Inference: no gradient computation needed
    with torch.no_grad():
        predictions = model(input_tensor)  # (1, max_length, 3)

    # Extract coordinates for the actual sequence length
    seq_length = len(sequence)
    coords = predictions[0, :seq_length, :].numpy()
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    # --- 3D Visualisation ---
    print("Rendering 3D structure...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # RNA backbone trace
    ax.plot(x, y, z, marker='o', linestyle='-', color='royalblue',
            markersize=5, alpha=0.8, label="RNA Backbone")

    # 5' and 3' terminal markers
    ax.scatter(x[0], y[0], z[0], color='lime', s=150,
               edgecolor='black', label="5' End (Start)")
    ax.scatter(x[-1], y[-1], z[-1], color='red', s=150,
               edgecolor='black', label="3' End (Stop)")

    title = (
        "Random-output visualisation (smoke test)"
        if allow_random_demo
        else "Model-predicted RNA 3D coordinates"
    )
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel("X Coordinate (Å)")
    ax.set_ylabel("Y Coordinate (Å)")
    ax.set_zlabel("Z Coordinate (Å)")
    ax.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"Visualisation saved to: {save_path}")


if __name__ == "__main__":
    test_sequence = "AUGGCUACGGUCGAAUGCGCUAGCUAGCUAGCUAGCUAGCGGAAUUCCGG"
    visualize_rna_3d(test_sequence)
