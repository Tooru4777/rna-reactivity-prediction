"""
RNA Reactivity Data Pipeline
=============================
PyTorch Dataset for the Stanford Ribonanza RNA Folding competition.

Supports two input modes:
  - 4-dim: Sequence-only one-hot (A, C, G, U) — for architecture ablation
  - 7-dim: Sequence + ViennaRNA 2D structure — for full model

Missing inputs fail fast; this pipeline requires measured targets.
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
        max_samples:   Optional maximum number of CSV rows to load.
    """

    def __init__(self, sequences_csv=None, max_length=206, use_structure=True,
                 max_samples=None, dataframe=None):
        self.max_length = max_length
        self.use_structure = use_structure and HAS_VIENNA

        # Feature dimension: 4 (seq only) or 7 (seq + structure)
        self.feature_dim = 7 if self.use_structure else 4

        self.seq_char_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
        self.struct_char_map = {'(': 4, ')': 5, '.': 6}

        if dataframe is not None or (sequences_csv and os.path.exists(sequences_csv)):
            if dataframe is not None:
                print("Loading dataset from in-memory sequence subset")
                self.seq_df = dataframe.copy()
            else:
                print(f"Loading dataset: {sequences_csv}")
                self.seq_df = pd.read_csv(sequences_csv, nrows=max_samples)

            # Quality filter
            if 'SN_filter' in self.seq_df.columns:
                initial = len(self.seq_df)
                self.seq_df = self.seq_df[self.seq_df['SN_filter'] == 1.0]
                print(f"Quality filter (SN_filter=1.0): {initial} -> {len(self.seq_df)}")

            self.seq_df = self.seq_df.reset_index(drop=True)
            self._validate_research_frame()
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
        else:
            raise FileNotFoundError(
                f"Training data not found: {sequences_csv}. This pipeline requires "
                "real measured reactivity targets."
            )

    def _validate_research_frame(self):
        """Fail fast on schema/domain problems that could corrupt labels."""
        required = {"sequence", "experiment_type"}
        missing = sorted(required - set(self.seq_df.columns))
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        if self.seq_df.empty:
            raise ValueError("No quality-eligible RNA rows remain after filtering")
        sequences = self.seq_df["sequence"]
        if sequences.isna().any() or sequences.astype(str).str.len().eq(0).any():
            raise ValueError("RNA sequences must be non-empty and non-null")
        invalid_sequence = ~sequences.astype(str).str.fullmatch(r"[ACGU]+")
        if invalid_sequence.any():
            raise ValueError("RNA sequences may contain only A, C, G, and U")
        allowed_experiments = {"2A3_MaP", "DMS_MaP"}
        observed = set(self.seq_df["experiment_type"].dropna().astype(str))
        unexpected = sorted(observed - allowed_experiments)
        if unexpected or self.seq_df["experiment_type"].isna().any():
            raise ValueError(f"Unexpected experiment_type values: {unexpected}")

        reactivity_cols = [
            c for c in self.seq_df.columns
            if c.startswith("reactivity_") and "error" not in c
        ]
        if not reactivity_cols:
            raise ValueError("No reactivity target columns were found")
        if self.seq_df[reactivity_cols].notna().sum(axis=1).eq(0).any():
            raise ValueError("Each retained experiment row must contain a reactivity target")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        seq_str = self.seq_df.iloc[idx].get('sequence', '')
        length = len(seq_str)

        # Get structure string
        if self.use_structure:
            struct_str = self.seq_df.iloc[idx].get(
                'structure', '.' * length
            )[:self.max_length]
        else:
            struct_str = '.' * length

        row_vals = self.seq_df.iloc[idx][self.reactivity_cols].values
        actual_len = min(length, len(row_vals))
        reactivities = np.zeros((length, 2), dtype=np.float32)
        valid_mask = np.zeros((length, 2), dtype=np.float32)

        experiment_type = self.seq_df.iloc[idx]['experiment_type']
        exp_idx = 0 if experiment_type == '2A3_MaP' else 1

        for i in range(actual_len):
            val = row_vals[i]
            if not pd.isna(val):
                reactivities[i, exp_idx] = float(val)
                valid_mask[i, exp_idx] = 1.0

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
