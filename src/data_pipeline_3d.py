"""
RNA 3D Structure Data Pipeline
==============================
Custom PyTorch Dataset for loading RNA sequences and their corresponding
3D atomic coordinates from the Stanford Ribonanza RNA Folding competition.

Key preprocessing steps:
  1. One-hot encoding of nucleotide sequences (A, C, G, U → 4-dim vectors)
  2. Zero-padding to a fixed max_length for batched GPU training
  3. Binary mask generation to exclude padded positions from loss computation
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os


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
            if labels_csv:
                self.label_df = pd.read_csv(labels_csv)
            else:
                self.label_df = None
            self.mock_data = False
            self.num_samples = len(self.seq_df)
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
        # Sequences shorter than max_length are padded with zeros.
        # The mask tensor below indicates which positions are real vs. padded.
        padded_coords = np.zeros((self.max_length, 3), dtype=np.float32)
        padded_coords[:actual_length] = coords

        # --- Padding mask ---
        # Essential for masked loss computation: we must NOT penalise
        # the model for predictions at padded positions.
        mask = np.zeros((self.max_length,), dtype=np.float32)
        mask[:actual_length] = 1.0

        return torch.tensor(one_hot), torch.tensor(padded_coords), torch.tensor(mask)


if __name__ == "__main__":
    print("Initialising RNA 3D Dataset...")
    dataset = RNA3DDataset(
        sequences_csv="dataset/train_sequences.csv",
        labels_csv="dataset/train_labels.csv"
    )

    # Mini-batch DataLoader (batch_size=32)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for sequences, coordinates, masks in dataloader:
        print("\nSuccessfully loaded one batch:")
        print(f"  Sequences shape  (B, L, 4): {sequences.shape}")
        print(f"  Coordinates shape(B, L, 3): {coordinates.shape}")
        print(f"  Masks shape      (B, L):    {masks.shape}")
        break
