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
