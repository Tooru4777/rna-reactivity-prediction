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
import re
import hashlib

# Sequence-only ablations do not need ViennaRNA; structure-feature runs do.
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

    STRUCTURE_MODES = {"none", "real", "position_shuffled", "zero"}

    def __init__(self, sequences_csv=None, max_length=206, use_structure=True,
                 max_samples=None, dataframe=None, structure_mode=None,
                 structure_control_seed=314159, precomputed_structures=None):
        self.max_length = max_length
        if structure_mode is None:
            structure_mode = "real" if use_structure else "none"
        if structure_mode not in self.STRUCTURE_MODES:
            raise ValueError(
                f"structure_mode must be one of {sorted(self.STRUCTURE_MODES)}"
            )
        self.structure_mode = structure_mode
        self.structure_control_seed = int(structure_control_seed)
        self.use_structure = structure_mode != "none"
        if self.use_structure and not HAS_VIENNA:
            raise RuntimeError(
                "ViennaRNA is required when use_structure=True; install the "
                "ViennaRNA Python package or use sequence-only features explicitly"
            )

        # Feature dimension: 4 (seq only) or 7 (seq + structure)
        self.feature_dim = 7 if self.use_structure else 4

        self.seq_char_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
        self.struct_char_map = {'(': 4, ')': 5, '.': 6}

        if dataframe is not None or (sequences_csv and os.path.exists(sequences_csv)):
            if dataframe is not None:
                print("Loading dataset from in-memory sequence subset")
                self.seq_df = dataframe.copy(deep=False)
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
                unique_sequences = self.seq_df["sequence"].astype(str).unique()
                if precomputed_structures is None:
                    structures = {
                        sequence: RNA.fold(sequence)[0]
                        for sequence in unique_sequences
                    }
                else:
                    missing = sorted(set(unique_sequences) - set(precomputed_structures))
                    if missing:
                        raise ValueError(
                            "precomputed_structures is missing retained RNA sequences"
                        )
                    structures = {
                        sequence: str(precomputed_structures[sequence])
                        for sequence in unique_sequences
                    }
                    invalid = [
                        sequence for sequence, structure in structures.items()
                        if len(structure) != len(sequence)
                        or not set(structure).issubset(set("()."))
                    ]
                    if invalid:
                        raise ValueError(
                            "precomputed_structures contains invalid dot-bracket values"
                        )
                self.seq_df = self.seq_df.assign(
                    structure=self.seq_df["sequence"].astype(str).map(structures)
                )
                if self.structure_mode == "position_shuffled":
                    controlled = {
                        sequence: self._shuffle_structure(sequence, structures[sequence])
                        for sequence in unique_sequences
                    }
                    self.seq_df = self.seq_df.assign(
                        feature_structure=(
                            self.seq_df["sequence"].astype(str).map(controlled)
                        )
                    )
                elif self.structure_mode == "real":
                    self.seq_df = self.seq_df.assign(
                        feature_structure=self.seq_df["structure"]
                    )
                else:
                    self.seq_df = self.seq_df.assign(feature_structure="")
                print("Structure computation complete.")

            # Identify reactivity columns
            self.reactivity_cols = self._reactivity_columns()
            self._sequences = self.seq_df["sequence"].astype(str).to_numpy()
            self._experiments = self.seq_df["experiment_type"].astype(str).to_numpy()
            self._reactivities = self.seq_df[self.reactivity_cols].to_numpy(
                dtype=np.float32, na_value=np.nan
            )
            if self.use_structure:
                self._feature_structures = (
                    self.seq_df["feature_structure"].astype(str).to_numpy()
                )
            else:
                self._feature_structures = None
            print(f"Loaded {self.num_samples} samples, "
                  f"{len(self.reactivity_cols)} reactivity columns, "
                  f"feature_dim={self.feature_dim}")
        else:
            raise FileNotFoundError(
                f"Training data not found: {sequences_csv}. This pipeline requires "
                "real measured reactivity targets."
            )

    def _shuffle_structure(self, sequence, structure):
        """Return a deterministic position-permuted structure control."""
        if len(structure) < 2:
            return structure
        seed_material = f"{self.structure_control_seed}:{sequence}".encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
        permutation = np.random.default_rng(seed).permutation(len(structure))
        if np.array_equal(permutation, np.arange(len(structure))):
            permutation = np.roll(permutation, 1)
        chars = np.asarray(list(structure), dtype="U1")
        return "".join(chars[permutation].tolist())

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

        reactivity_cols = self._reactivity_columns()
        if not reactivity_cols:
            raise ValueError("No reactivity target columns were found")
        observed_positions = [int(column.rsplit("_", 1)[-1]) for column in reactivity_cols]
        expected_positions = list(range(1, len(reactivity_cols) + 1))
        if observed_positions != expected_positions:
            raise ValueError(
                "Reactivity columns must cover continuous positions starting at 0001"
            )
        if self.seq_df[reactivity_cols].notna().sum(axis=1).eq(0).any():
            raise ValueError("Each retained experiment row must contain a reactivity target")

    def _reactivity_columns(self):
        """Return target columns in numeric nucleotide order."""
        indexed = []
        for column in self.seq_df.columns:
            match = re.fullmatch(r"reactivity_(\d+)", str(column))
            if match:
                indexed.append((int(match.group(1)), str(column)))
        return [column for _, column in sorted(indexed)]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        seq_str = self._sequences[idx]
        length = len(seq_str)

        # Get structure string
        if self.use_structure:
            struct_str = self._feature_structures[idx][:self.max_length]
        else:
            struct_str = '.' * length

        row_vals = self._reactivities[idx]
        actual_len = min(length, len(row_vals))
        reactivities = np.zeros((length, 2), dtype=np.float32)
        valid_mask = np.zeros((length, 2), dtype=np.float32)

        experiment_type = self._experiments[idx]
        exp_idx = 0 if experiment_type == '2A3_MaP' else 1
        finite = np.isfinite(row_vals[:actual_len])
        reactivities[:actual_len, exp_idx] = np.where(
            finite, row_vals[:actual_len], 0.0
        )
        valid_mask[:actual_len, exp_idx] = finite.astype(np.float32)

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
            if (
                self.use_structure
                and self.structure_mode != "zero"
                and i < len(struct_str)
            ):
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
