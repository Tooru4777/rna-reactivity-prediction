"""
RNA Reactivity Data Pipeline
=============================
PyTorch Dataset for the Stanford Ribonanza RNA Folding competition.

Supports two input modes:
  - 4-dim: Sequence-only one-hot (A, C, G, U) — for architecture ablation
  - 7-dim: Sequence + ViennaRNA 2D structure — for full model

Falls back to synthetic data if CSVs are not found (for pipeline testing).
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os

# ViennaRNA is optional — falls back to dummy structure if not installed
try:
    import RNA
    HAS_VIENNA = True
except ImportError:
    HAS_VIENNA = False


class RNAReactivityDataset(Dataset):
    """
    Dataset for RNA reactivity prediction.

    Each sample returns:
        - features:  (max_length, feature_dim) tensor — one-hot features
        - targets:   (max_length, 2) tensor — 2A3_MaP and DMS_MaP reactivity
        - mask:      (max_length, 2) tensor — 1.0 for valid positions, 0.0 for padding/NaN

    Args:
        sequences_csv: Path to CSV with 'sequence', 'experiment_type', reactivity columns
        max_length:    Pad/truncate all sequences to this length
        use_structure: If True, compute ViennaRNA 2D structure (7-dim input).
                       If False, use sequence-only features (4-dim input).
    """

    def __init__(self, sequences_csv, max_length=206, use_structure=True,
                 allow_synthetic=False, synthetic_seed=42):
        self.max_length = max_length
        self.use_structure = use_structure and HAS_VIENNA
        self.synthetic_seed = synthetic_seed

        # Feature dimension: 4 (seq only) or 7 (seq + structure)
        self.feature_dim = 7 if self.use_structure else 4

        self.seq_char_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
        self.struct_char_map = {'(': 4, ')': 5, '.': 6}

        if os.path.exists(sequences_csv):
            print(f"Loading dataset: {sequences_csv}")
            self.seq_df = pd.read_csv(sequences_csv)

            # Quality filter
            if 'SN_filter' in self.seq_df.columns:
                initial = len(self.seq_df)
                self.seq_df = self.seq_df[self.seq_df['SN_filter'] == 1.0]
                print(f"Quality filter (SN_filter=1.0): {initial} -> {len(self.seq_df)}")

            self.seq_df = self.seq_df.reset_index(drop=True)
            self.mock_data = False
            self.num_samples = len(self.seq_df)

            # Compute secondary structures if requested
            if self.use_structure:
                print("Computing ViennaRNA 2D structures...")
                self.seq_df['structure'] = self.seq_df['sequence'].apply(
                    lambda seq: RNA.fold(seq)[0]
                )
                print("Structure computation complete.")

            # Identify reactivity columns
            self.reactivity_cols = [
                c for c in self.seq_df.columns
                if c.startswith('reactivity_') and 'error' not in c
            ]
            print(f"Loaded {self.num_samples} samples, "
                  f"{len(self.reactivity_cols)} reactivity columns, "
                  f"feature_dim={self.feature_dim}")
        elif allow_synthetic:
            print(f"[SMOKE TEST] {sequences_csv} not found. Using synthetic data.")
            self.mock_data = True
            self.num_samples = 500
            self.reactivity_cols = []
        else:
            raise FileNotFoundError(
                f"Training data not found: {sequences_csv}. "
                "Synthetic data are disabled for research runs. "
                "Pass allow_synthetic=True only for a pipeline smoke test."
            )

    @property
    def sample_groups(self):
        """Group labels used to keep identical sequences in one data split."""
        if self.mock_data:
            return np.array([f"synthetic_{i}" for i in range(self.num_samples)])
        return self.seq_df["sequence"].fillna("").astype(str).to_numpy()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.mock_data:
            # Per-index RNG keeps smoke-test samples stable across epochs.
            rng = np.random.default_rng(self.synthetic_seed + idx)
            length = rng.integers(50, self.max_length)
            seq_str = ''.join(rng.choice(['A', 'C', 'G', 'U'], size=length))
            struct_str = '.' * length
            reactivities = rng.normal(size=(length, 2)).astype(np.float32)
            valid_mask = np.ones((length, 2), dtype=np.float32)
        else:
            seq_str = self.seq_df.iloc[idx].get('sequence', '')
            if pd.isna(seq_str) or len(seq_str) == 0:
                seq_str = 'A' * 50
            length = len(seq_str)

            # Get structure string
            if self.use_structure:
                struct_str = self.seq_df.iloc[idx].get(
                    'structure', '.' * length
                )[:self.max_length]
            else:
                struct_str = '.' * length

            # Extract reactivity values
            if len(self.reactivity_cols) > 0:
                row_vals = self.seq_df.iloc[idx][self.reactivity_cols].values
                actual_len = min(length, len(row_vals))
                reactivities = np.zeros((length, 2), dtype=np.float32)
                valid_mask = np.zeros((length, 2), dtype=np.float32)

                experiment_type = self.seq_df.iloc[idx].get('experiment_type', '2A3_MaP')
                exp_idx = 0 if experiment_type == '2A3_MaP' else 1

                for i in range(actual_len):
                    val = row_vals[i]
                    if not pd.isna(val):
                        reactivities[i, exp_idx] = float(val)
                        valid_mask[i, exp_idx] = 1.0
            else:
                reactivities = np.zeros((length, 2), dtype=np.float32)
                valid_mask = np.zeros((length, 2), dtype=np.float32)

        # Truncate
        seq_str = seq_str[:self.max_length]
        struct_str = struct_str[:self.max_length]
        reactivities = reactivities[:self.max_length]
        valid_mask = valid_mask[:self.max_length]
        actual_length = len(seq_str)

        # Build feature tensor
        features = np.zeros((self.max_length, self.feature_dim), dtype=np.float32)
        for i in range(actual_length):
            c = seq_str[i]
            if c in self.seq_char_map:
                features[i, self.seq_char_map[c]] = 1.0
            if self.use_structure and i < len(struct_str):
                s = struct_str[i]
                if s in self.struct_char_map:
                    features[i, self.struct_char_map[s]] = 1.0

        # Pad targets and mask
        padded_reactivities = np.zeros((self.max_length, 2), dtype=np.float32)
        padded_reactivities[:actual_length] = reactivities

        mask = np.zeros((self.max_length, 2), dtype=np.float32)
        mask[:actual_length] = valid_mask

        return (
            torch.tensor(features),
            torch.tensor(padded_reactivities),
            torch.tensor(mask)
        )
