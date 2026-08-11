"""Memory-bounded loading helpers for the wide Ribonanza CSV."""

import random
from pathlib import Path

import pandas as pd


def resolve_ribonanza_train_data(path):
    """Resolve the official train CSV from a file or extracted data directory."""
    requested = Path(path).expanduser()
    if requested.is_file():
        return requested.resolve()

    search_root = requested if requested.is_dir() else requested.parent
    candidates = sorted(search_root.rglob("train_data.csv")) if search_root.is_dir() else []
    if len(candidates) == 1:
        return candidates[0].resolve()

    competition_candidates = [
        candidate for candidate in candidates
        if "stanford-ribonanza-rna-folding" in candidate.parts
    ]
    if len(competition_candidates) == 1:
        return competition_candidates[0].resolve()

    old_candidates = [candidate for candidate in candidates if "OLD" in candidate.parts]
    if len(old_candidates) == 1:
        return old_candidates[0].resolve()

    found = ", ".join(str(candidate) for candidate in candidates) or "none"
    raise FileNotFoundError(
        f"Could not uniquely resolve Ribonanza train_data.csv from {requested}; "
        f"found: {found}"
    )


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
