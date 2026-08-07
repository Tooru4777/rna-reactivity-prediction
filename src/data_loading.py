"""Memory-bounded loading helpers for the wide Ribonanza CSV."""

import random

import pandas as pd


def load_unique_sequence_subset(csv_path, max_sequences=None, chunk_size=4096, seed=42):
    """Load all quality-filtered experiment rows for sampled unique RNAs.

    Ribonanza stores experiment types in separate rows that may be far apart in
    the CSV. Reading the first N rows can therefore discard paired experiments
    and hide row-level split leakage. This two-pass loader first selects unique
    quality-eligible sequence strings using narrow metadata columns, then scans
    the wide file in bounded chunks and retains every row for those sequences.
    """
    if not max_sequences:
        return pd.read_csv(csv_path)
    all_sequences = []
    seen = set()
    for chunk in pd.read_csv(
        csv_path, usecols=["sequence", "SN_filter"], chunksize=chunk_size
    ):
        eligible = chunk[chunk["SN_filter"] == 1.0]
        for sequence in eligible["sequence"].dropna().astype(str).unique():
            if sequence not in seen:
                all_sequences.append(sequence)
                seen.add(sequence)

    sample_size = min(max_sequences, len(all_sequences))
    selected_set = set(random.Random(seed).sample(all_sequences, sample_size))

    frames = []
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        matched = chunk[chunk["sequence"].astype(str).isin(selected_set)]
        if not matched.empty:
            frames.append(matched)
    if not frames:
        raise ValueError("No rows matched the selected RNA sequences")
    return pd.concat(frames, ignore_index=True)
