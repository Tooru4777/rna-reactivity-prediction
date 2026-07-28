"""Leakage-aware train/CV/test splitting helpers."""

from dataclasses import dataclass

import numpy as np
from torch.utils.data import Subset


@dataclass(frozen=True)
class GroupedSplits:
    train: Subset
    cv: Subset
    test: Subset


def grouped_train_cv_test_split(dataset, groups, seed=42,
                                cv_fraction=0.15, test_fraction=0.15):
    """Split samples by group so one RNA sequence cannot cross partitions."""
    groups = np.asarray(groups)
    if len(groups) != len(dataset):
        raise ValueError("groups must contain one label per dataset row")
    if not 0 < cv_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("cv_fraction and test_fraction must be between 0 and 1")
    if cv_fraction + test_fraction >= 1:
        raise ValueError("cv_fraction + test_fraction must be less than 1")

    unique_groups = np.unique(groups)
    if len(unique_groups) < 3:
        raise ValueError("At least three unique sequence groups are required")

    rng = np.random.default_rng(seed)
    shuffled = unique_groups.copy()
    rng.shuffle(shuffled)

    n_test = max(1, int(round(len(shuffled) * test_fraction)))
    n_cv = max(1, int(round(len(shuffled) * cv_fraction)))
    if n_test + n_cv >= len(shuffled):
        n_test = 1
        n_cv = 1

    test_groups = set(shuffled[:n_test])
    cv_groups = set(shuffled[n_test:n_test + n_cv])
    train_groups = set(shuffled[n_test + n_cv:])

    def indices_for(selected):
        return [i for i, group in enumerate(groups) if group in selected]

    return GroupedSplits(
        train=Subset(dataset, indices_for(train_groups)),
        cv=Subset(dataset, indices_for(cv_groups)),
        test=Subset(dataset, indices_for(test_groups)),
    )
