"""Leakage-safe dataset splitting utilities."""

import numpy as np


def grouped_split_indices(dataframe, group_col="sequence", fractions=(0.70, 0.15, 0.15), seed=42):
    """Return train/CV/test row indices without sharing groups across splits."""
    if group_col not in dataframe.columns:
        raise ValueError(f"Required grouping column not found: {group_col}")
    if len(fractions) != 3 or not np.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must contain three values that sum to 1")

    groups = dataframe[group_col].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    if len(unique_groups) < 3:
        raise ValueError("At least three unique groups are required")

    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)

    n_groups = len(unique_groups)
    n_train = max(1, int(fractions[0] * n_groups))
    n_cv = max(1, int(fractions[1] * n_groups))
    if n_train + n_cv >= n_groups:
        n_train = n_groups - 2
        n_cv = 1

    train_groups = set(unique_groups[:n_train])
    cv_groups = set(unique_groups[n_train:n_train + n_cv])
    test_groups = set(unique_groups[n_train + n_cv:])

    def indices_for(selected):
        return np.flatnonzero(np.isin(groups, list(selected))).tolist()

    return (
        indices_for(train_groups),
        indices_for(cv_groups),
        indices_for(test_groups),
    )


def random_split_indices(n_rows, fractions=(0.70, 0.15, 0.15), seed=42):
    """Return deterministic row-level train/CV/test indices."""
    if n_rows < 3:
        raise ValueError("At least three rows are required")
    if len(fractions) != 3 or not np.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must contain three values that sum to 1")
    indices = np.random.default_rng(seed).permutation(n_rows)
    n_train = max(1, int(fractions[0] * n_rows))
    n_cv = max(1, int(fractions[1] * n_rows))
    if n_train + n_cv >= n_rows:
        n_train = n_rows - 2
        n_cv = 1
    return (
        indices[:n_train].tolist(),
        indices[n_train:n_train + n_cv].tolist(),
        indices[n_train + n_cv:].tolist(),
    )


def sequence_overlap_counts(dataframe, splits, group_col="sequence"):
    """Count unique sequences shared by each pair of row-index splits."""
    groups = [set(dataframe.iloc[idx][group_col].astype(str)) for idx in splits]
    return {
        "train_cv": len(groups[0] & groups[1]),
        "train_test": len(groups[0] & groups[2]),
        "cv_test": len(groups[1] & groups[2]),
    }


def validate_split_integrity(n_rows, splits):
    """Raise if partitions overlap, omit rows, or contain out-of-range indices."""
    flattened = [int(index) for split in splits for index in split]
    if len(flattened) != n_rows:
        raise ValueError(
            f"Split row count {len(flattened)} does not match dataset rows {n_rows}"
        )
    if len(set(flattened)) != n_rows:
        raise ValueError("Split partitions contain duplicate row indices")
    if set(flattened) != set(range(n_rows)):
        raise ValueError("Split partitions do not cover exactly the dataset row indices")
