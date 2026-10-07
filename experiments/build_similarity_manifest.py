"""Build a frozen MMseqs2 split manifest for one Ribonanza cohort."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(PROJECT_ROOT))

from src.data_loading import load_unique_sequence_subset, resolve_ribonanza_train_data
from src.mmseqs_clustering import (
    audit_cross_partition_similarity,
    build_mmseqs_manifest,
    write_manifest_atomic,
)
from src.similarity_splitting import validate_similarity_manifest


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_target_weights(frame, block_rows=4096):
    """Return per-sequence valid-target counts with bounded temporary memory."""
    target_columns = [
        column for column in frame.columns
        if re.fullmatch(r"reactivity_\d+", str(column))
    ]
    if not target_columns:
        raise ValueError("No reactivity target columns were found")
    row_weights = np.zeros(len(frame), dtype=np.int64)
    for start in range(0, len(frame), block_rows):
        stop = min(start + block_rows, len(frame))
        values = frame.iloc[start:stop][target_columns].to_numpy(
            dtype=np.float32, copy=False
        )
        row_weights[start:stop] = np.isfinite(values).sum(axis=1)
    weighted = frame[["sequence"]].copy()
    weighted["valid_targets"] = row_weights
    result = weighted.groupby("sequence", sort=True)["valid_targets"].sum()
    if (result <= 0).any():
        raise ValueError("Every retained sequence must have at least one valid target")
    return {str(sequence): int(weight) for sequence, weight in result.items()}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--max-sequences", type=int, default=256)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--mmseqs", default="mmseqs")
    parser.add_argument("--identity", type=float, default=0.8)
    parser.add_argument("--coverage", type=float, default=0.8)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--allocation-seed", type=int, default=20261001)
    parser.add_argument("--allocation-tolerance", type=float, default=0.03)
    return parser.parse_args()


def main():
    args = parse_args()
    source = resolve_ribonanza_train_data(args.data)
    source_sha256 = sha256_file(source)
    cohort = load_unique_sequence_subset(
        source,
        max_sequences=args.max_sequences or None,
        seed=args.sample_seed,
    )
    sequence_weights = valid_target_weights(cohort)
    sequences = sorted(sequence_weights)
    manifest = build_mmseqs_manifest(
        sequences,
        source_sha256=source_sha256,
        sequence_weights=sequence_weights,
        work_dir=args.work_dir,
        executable=args.mmseqs,
        identity=args.identity,
        coverage=args.coverage,
        threads=args.threads,
        allocation_seed=args.allocation_seed,
        allocation_tolerance=args.allocation_tolerance,
    )
    manifest["preprocessing"] = {
        "data_file": source.name,
        "source_size_bytes": source.stat().st_size,
        "requested_unique_sequences": args.max_sequences or None,
        "filtered_rows": len(cohort),
        "filtered_unique_sequences": len(sequences),
        "sample_seed": args.sample_seed,
        "filter": "SN_filter == 1.0",
        "weight": "finite reactivity targets across all retained profiles",
    }
    manifest["cross_split_search_audit"] = audit_cross_partition_similarity(
        sequences,
        assignments=manifest["assignments"],
        cohort_sha256=manifest["cohort_sha256"],
        assignment_sha256=manifest["assignment_sha256"],
        work_dir=Path(args.work_dir) / "cross_split_search_audit",
        executable=args.mmseqs,
        identity=args.identity,
        coverage=args.coverage,
        threads=args.threads,
    )
    validate_similarity_manifest(
        manifest,
        expected_source_sha256=source_sha256,
        require_cross_split_audit=True,
    )
    output = write_manifest_atomic(manifest, args.output)
    print(json.dumps({
        "manifest": os.fspath(output),
        "source_sha256": source_sha256,
        "sequences": len(sequences),
        "clusters": manifest["allocation"]["clusters"],
        "assignment_sha256": manifest["assignment_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
