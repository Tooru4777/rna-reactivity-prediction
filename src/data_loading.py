"""Memory-bounded loading helpers for the wide Ribonanza CSV."""

import random
from pathlib import Path
import re

import pandas as pd


def _is_legacy_ribonanza_path(path):
    """Return whether a path points into the superseded Kaggle ``OLD`` tree."""
    return any(part.casefold() == "old" for part in Path(path).parts)


def resolve_ribonanza_train_data(path, allow_legacy=False):
    """Resolve the current official train CSV from a file or directory.

    Kaggle retains the original competition file under ``OLD/train_data.csv``.
    That file predates corrections to ``reactivity_error_*`` and a small number
    of ``SN_filter`` values, so research runs reject it unless a caller opts in
    explicitly for historical archive reproduction.
    """
    requested = Path(path).expanduser()
    if requested.is_file():
        if _is_legacy_ribonanza_path(requested) and not allow_legacy:
            raise FileNotFoundError(
                "Refusing superseded OLD/train_data.csv; provide the current "
                "competition train_data.csv or pass allow_legacy=True only to "
                "reproduce a historical archive"
            )
        return requested.resolve()

    search_root = requested if requested.is_dir() else requested.parent
    candidates = sorted(search_root.rglob("train_data.csv")) if search_root.is_dir() else []
    current_candidates = [
        candidate for candidate in candidates
        if not _is_legacy_ribonanza_path(candidate)
    ]
    competition_candidates = [
        candidate for candidate in current_candidates
        if "stanford-ribonanza-rna-folding" in candidate.parts
    ]
    if len(competition_candidates) == 1:
        return competition_candidates[0].resolve()
    if len(current_candidates) == 1:
        return current_candidates[0].resolve()

    legacy_candidates = [
        candidate for candidate in candidates if _is_legacy_ribonanza_path(candidate)
    ]
    if not current_candidates and allow_legacy and len(legacy_candidates) == 1:
        return legacy_candidates[0].resolve()
    if not current_candidates and legacy_candidates and not allow_legacy:
        found = ", ".join(str(candidate) for candidate in legacy_candidates)
        raise FileNotFoundError(
            "Only superseded OLD/train_data.csv candidates were found: " + found
        )

    found = ", ".join(str(candidate) for candidate in candidates) or "none"
    raise FileNotFoundError(
        f"Could not uniquely resolve Ribonanza train_data.csv from {requested}; "
        f"found: {found}"
    )


def _training_columns(csv_path):
    """Return the columns required for modeling and provenance reports.

    The corrected competition file is very wide because it includes one
    ``reactivity_error_*`` column per nucleotide.  The current models do not
    consume those error estimates, so excluding them roughly halves parsing
    and resident-memory costs without changing any model input or target.
    """
    columns = list(pd.read_csv(csv_path, nrows=0).columns)
    metadata = {
        "sequence_id", "dataset_name", "sequence", "experiment_type",
        "reads", "signal_to_noise", "SN_filter",
    }
    selected = [
        column for column in columns
        if column in metadata or re.fullmatch(r"reactivity_\d+", str(column))
    ]
    required = {"sequence", "experiment_type", "SN_filter"}
    missing = sorted(required - set(selected))
    if missing:
        raise ValueError(f"Missing required Ribonanza columns: {missing}")
    if not any(re.fullmatch(r"reactivity_\d+", str(column)) for column in selected):
        raise ValueError("No reactivity target columns were found")
    return selected


def _read_dtypes(columns):
    """Use float32 for numeric model inputs to bound full-cohort memory."""
    dtypes = {}
    for column in columns:
        if column == "SN_filter" or re.fullmatch(r"reactivity_\d+", str(column)):
            dtypes[column] = "float32"
    return dtypes


def load_unique_sequence_subset(csv_path, max_sequences=None, chunk_size=4096, seed=42):
    """Load all quality-filtered experiment rows for sampled unique RNAs.

    Ribonanza stores experiment types in separate rows that may be far apart in
    the CSV. Reading the first N rows can therefore discard paired experiments
    and hide row-level split leakage. This two-pass loader first selects unique
    quality-eligible sequence strings using narrow metadata columns, then scans
    the wide file in bounded chunks and retains every row for those sequences.
    """
    if max_sequences is not None and int(max_sequences) < 0:
        raise ValueError("max_sequences must be non-negative; use 0 for full data")
    if int(chunk_size) <= 0:
        raise ValueError("chunk_size must be positive")

    selected_columns = _training_columns(csv_path)
    dtypes = _read_dtypes(selected_columns)

    if max_sequences in (None, 0):
        frames = []
        for chunk in pd.read_csv(
            csv_path,
            usecols=selected_columns,
            dtype=dtypes,
            chunksize=chunk_size,
        ):
            eligible = chunk[chunk["SN_filter"] == 1.0]
            if not eligible.empty:
                frames.append(eligible)
        if not frames:
            raise ValueError("No quality-eligible RNA rows were found")
        return pd.concat(frames, ignore_index=True)

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
    for chunk in pd.read_csv(
        csv_path,
        usecols=selected_columns,
        dtype=dtypes,
        chunksize=chunk_size,
    ):
        matched = chunk[
            (chunk["SN_filter"] == 1.0)
            & chunk["sequence"].astype(str).isin(selected_set)
        ]
        if not matched.empty:
            frames.append(matched)
    if not frames:
        raise ValueError("No rows matched the selected RNA sequences")
    return pd.concat(frames, ignore_index=True)
