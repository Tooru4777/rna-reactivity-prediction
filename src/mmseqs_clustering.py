"""Deterministic MMseqs2 clustering for similarity-aware data splits."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
import subprocess
import tempfile
from typing import Any

from .similarity_splitting import (
    allocate_cluster_partitions,
    canonical_sequence,
    canonicalize_cluster_assignments,
    sequence_digest,
)


_FRACTIONS = (0.70, 0.15, 0.15)


def _sha256_lines(lines: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _normalise_sequences(sequences: Iterable[str]) -> tuple[list[str], dict[str, str]]:
    supplied = list(sequences)
    if not supplied:
        raise ValueError("at least one sequence is required")

    digest_to_sequence: dict[str, str] = {}
    ordered_digests: list[str] = []
    for sequence in supplied:
        canonical = canonical_sequence(sequence)
        digest = sequence_digest(canonical)
        if digest in digest_to_sequence:
            if digest_to_sequence[digest] == canonical:
                raise ValueError("sequences must be unique after canonicalisation")
            raise ValueError(f"SHA-256 collision for sequence digest {digest}")
        digest_to_sequence[digest] = canonical
        ordered_digests.append(digest)

    return ordered_digests, digest_to_sequence


def _positive_weight(value: object, *, digest: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"weight for {digest} must be a positive finite number")
    try:
        weight = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"weight for {digest} must be a positive finite number"
        ) from exc
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError(f"weight for {digest} must be a positive finite number")
    return weight


def _normalise_weights(
    sequence_weights: Mapping[str, float] | Sequence[float],
    *,
    ordered_digests: Sequence[str],
    digest_to_sequence: Mapping[str, str],
) -> dict[str, float]:
    expected = set(ordered_digests)

    if not isinstance(sequence_weights, Mapping):
        values = list(sequence_weights)
        if len(values) != len(ordered_digests):
            raise ValueError("sequence_weights must contain one weight per sequence")
        return {
            digest: _positive_weight(value, digest=digest)
            for digest, value in zip(ordered_digests, values, strict=True)
        }

    resolved: dict[str, float] = {}
    for key, value in sequence_weights.items():
        if not isinstance(key, str):
            raise ValueError("sequence_weights keys must be sequence digests or sequences")

        digest: str | None = key if key in expected else None
        if digest is None:
            try:
                candidate_sequence = canonical_sequence(key)
                candidate_digest = sequence_digest(candidate_sequence)
            except (TypeError, ValueError):
                candidate_digest = ""
                candidate_sequence = ""
            if (
                candidate_digest in expected
                and digest_to_sequence[candidate_digest] == candidate_sequence
            ):
                digest = candidate_digest

        if digest is None:
            raise ValueError(f"sequence_weights contains an unknown key: {key!r}")
        if digest in resolved:
            raise ValueError(f"sequence_weights contains duplicate keys for {digest}")
        resolved[digest] = _positive_weight(value, digest=digest)

    missing = sorted(expected.difference(resolved))
    if missing:
        raise ValueError(
            "sequence_weights is missing weights for: " + ", ".join(missing)
        )
    return resolved


def _write_digest_fasta(path: Path, digest_to_sequence: Mapping[str, str]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for digest in sorted(digest_to_sequence):
            handle.write(f">{digest}\n{digest_to_sequence[digest]}\n")


def _run_mmseqs_version(executable: str) -> str:
    completed = subprocess.run(
        [executable, "version"],
        check=True,
        capture_output=True,
        text=True,
    )
    output = (getattr(completed, "stdout", "") or "").strip()
    if not output:
        output = (getattr(completed, "stderr", "") or "").strip()
    if not output:
        raise RuntimeError("MMseqs2 returned an empty version string")
    return output.splitlines()[0].strip()


def _parse_cluster_tsv(
    path: Path, expected_digests: set[str]
) -> dict[str, str]:
    raw_assignments: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.rstrip("\r\n")
            if not stripped:
                continue
            fields = stripped.split("\t")
            if len(fields) != 2 or not all(fields):
                raise ValueError(
                    f"invalid MMseqs2 cluster row at line {line_number}: "
                    "expected representative and member"
                )
            representative, member = fields
            if representative not in expected_digests:
                raise ValueError(
                    f"MMseqs2 returned an unknown representative: {representative}"
                )
            if member not in expected_digests:
                raise ValueError(f"MMseqs2 returned an unknown member: {member}")
            if member in raw_assignments:
                previous = raw_assignments[member]
                if previous != representative:
                    raise ValueError(
                        f"MMseqs2 member {member} has conflicting representatives: "
                        f"{previous} and {representative}"
                    )
                raise ValueError(f"MMseqs2 member {member} appears more than once")
            raw_assignments[member] = representative

    missing = sorted(expected_digests.difference(raw_assignments))
    if missing:
        raise ValueError(
            "MMseqs2 output is missing members: " + ", ".join(missing)
        )
    return raw_assignments


def audit_cross_partition_similarity(
    sequences: Iterable[str],
    *,
    assignments: Mapping[str, Mapping[str, str]],
    cohort_sha256: str,
    assignment_sha256: str,
    work_dir: str | os.PathLike[str],
    executable: str | os.PathLike[str] = "mmseqs",
    identity: float = 0.8,
    coverage: float = 0.8,
    sensitivity: float = 7.5,
    threads: int = 1,
    split_memory_limit: str = "4G",
) -> dict[str, Any]:
    """Fail if a sensitive, independent search finds a cross-partition hit.

    The clustering pass uses ``easy-cluster`` with single-step clustering.  This
    audit deliberately creates fresh databases through six directed
    ``easy-search`` runs at higher sensitivity.  It therefore checks for
    qualifying edges that the clustering heuristic may have missed instead of
    merely re-checking that cluster identifiers do not overlap.
    """

    if not math.isfinite(float(identity)) or not 0 < float(identity) <= 1:
        raise ValueError("identity must be in (0, 1]")
    if not math.isfinite(float(coverage)) or not 0 < float(coverage) <= 1:
        raise ValueError("coverage must be in (0, 1]")
    if not math.isfinite(float(sensitivity)) or float(sensitivity) <= 0:
        raise ValueError("sensitivity must be a positive finite number")
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("threads must be a positive integer")

    _, digest_to_sequence = _normalise_sequences(sequences)
    expected_digests = set(digest_to_sequence)
    if set(assignments) != expected_digests:
        raise ValueError("audit assignments must exactly match the sequence cohort")

    partition_sequences: dict[str, dict[str, str]] = {
        partition: {} for partition in ("train", "cv", "test")
    }
    for digest in sorted(expected_digests):
        record = assignments[digest]
        if not isinstance(record, Mapping):
            raise ValueError("each audit assignment must be a mapping")
        partition = record.get("partition")
        if partition not in partition_sequences:
            raise ValueError("audit assignment contains an invalid partition")
        partition_sequences[str(partition)][digest] = digest_to_sequence[digest]
    if any(not members for members in partition_sequences.values()):
        raise ValueError("cross-partition audit requires non-empty partitions")

    executable_string = os.fspath(executable)
    version = _run_mmseqs_version(executable_string)
    directions = [
        (query, target)
        for query in ("train", "cv", "test")
        for target in ("train", "cv", "test")
        if query != target
    ]
    direction_results: list[dict[str, Any]] = []
    directory = Path(work_dir)
    directory.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix="mmseqs_cross_partition_audit_", dir=directory
    ) as temporary_name:
        temporary_root = Path(temporary_name)
        fasta_paths: dict[str, Path] = {}
        for partition, members in partition_sequences.items():
            fasta_path = temporary_root / f"{partition}.fasta"
            _write_digest_fasta(fasta_path, members)
            fasta_paths[partition] = fasta_path

        for query_partition, target_partition in directions:
            label = f"{query_partition}_to_{target_partition}"
            result_path = temporary_root / f"{label}.tsv"
            search_tmp = temporary_root / f"{label}_tmp"
            command = [
                executable_string,
                "easy-search",
                os.fspath(fasta_paths[query_partition]),
                os.fspath(fasta_paths[target_partition]),
                os.fspath(result_path),
                os.fspath(search_tmp),
                "--search-type",
                "3",
                "--strand",
                "1",
                "-k",
                "8",
                "--min-seq-id",
                format(float(identity), "g"),
                "-c",
                format(float(coverage), "g"),
                "--cov-mode",
                "0",
                "--alignment-mode",
                "3",
                "--seq-id-mode",
                "0",
                "-s",
                format(float(sensitivity), "g"),
                "--threads",
                str(threads),
                "--mask",
                "0",
                "--split-memory-limit",
                split_memory_limit,
                "--format-output",
                "query,target,fident,qcov,tcov,alnlen,qlen,tlen",
            ]
            try:
                subprocess.run(command, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    f"MMseqs2 cross-partition search failed for {label}: "
                    f"{exc.stderr}\n{exc.stdout}"
                ) from exc
            if not result_path.is_file():
                raise RuntimeError(
                    f"MMseqs2 did not create the audit result for {label}"
                )

            hits: list[tuple[str, str, float, float, float]] = []
            with result_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    stripped = line.rstrip("\r\n")
                    if not stripped:
                        continue
                    fields = stripped.split("\t")
                    if len(fields) != 8:
                        raise ValueError(
                            f"invalid MMseqs2 audit row at {label}:{line_number}"
                        )
                    query_digest, target_digest = fields[:2]
                    if query_digest not in partition_sequences[query_partition]:
                        raise ValueError("MMseqs2 audit returned an unknown query digest")
                    if target_digest not in partition_sequences[target_partition]:
                        raise ValueError("MMseqs2 audit returned an unknown target digest")
                    try:
                        fident, qcov, tcov = map(float, fields[2:5])
                        tuple(map(float, fields[5:]))
                    except ValueError as exc:
                        raise ValueError(
                            f"invalid numeric MMseqs2 audit row at {label}:{line_number}"
                        ) from exc
                    if not all(math.isfinite(value) for value in (fident, qcov, tcov)):
                        raise ValueError("MMseqs2 audit returned a non-finite metric")
                    hits.append((query_digest, target_digest, fident, qcov, tcov))

            if hits:
                first = hits[0]
                raise RuntimeError(
                    "Independent MMseqs2 audit found cross-partition similarity "
                    f"hits for {label} (hits={len(hits)}, "
                    f"first={first[0][:12]}->{first[1][:12]})"
                )
            direction_results.append({
                "query_partition": query_partition,
                "target_partition": target_partition,
                "qualifying_hits": 0,
            })

    return {
        "status": "passed",
        "backend": "mmseqs",
        "search": "easy-search",
        "version": version,
        "cohort_sha256": cohort_sha256,
        "assignment_sha256": assignment_sha256,
        "qualifying_hit_count": 0,
        "max_qualifying_fident": None,
        "max_qualifying_qcov": None,
        "max_qualifying_tcov": None,
        "direction_results": direction_results,
        "parameters": {
            "min_seq_id": float(identity),
            "coverage": float(coverage),
            "cov_mode": 0,
            "alignment_mode": 3,
            "seq_id_mode": 0,
            "sensitivity": float(sensitivity),
            "search_type": 3,
            "strand": 1,
            "kmer_length": 8,
            "threads": threads,
            "mask": 0,
            "split_memory_limit": split_memory_limit,
            "format_output": (
                "query,target,fident,qcov,tcov,alnlen,qlen,tlen"
            ),
        },
    }


def build_mmseqs_manifest(
    sequences: Iterable[str],
    *,
    source_sha256: str,
    sequence_weights: Mapping[str, float] | Sequence[float],
    work_dir: str | os.PathLike[str],
    executable: str | os.PathLike[str] = "mmseqs",
    identity: float = 0.8,
    coverage: float = 0.8,
    threads: int = 1,
    allocation_seed: int = 20261001,
    allocation_tolerance: float = 0.03,
    split_memory_limit: str = "4G",
    kmer_length: int = 8,
) -> dict[str, Any]:
    """Cluster unique RNA sequences and return a validated-schema manifest.

    ``sequence_weights`` may be keyed by sequence digest or by sequence text. A
    sequence of weights is also accepted and is matched to the input order.
    """

    if not isinstance(source_sha256, str) or len(source_sha256) != 64:
        raise ValueError("source_sha256 must be a 64-character SHA-256 digest")
    try:
        int(source_sha256, 16)
    except ValueError as exc:
        raise ValueError("source_sha256 must be a hexadecimal SHA-256 digest") from exc
    if not math.isfinite(float(identity)) or not 0 < float(identity) <= 1:
        raise ValueError("identity must be in (0, 1]")
    if not math.isfinite(float(coverage)) or not 0 < float(coverage) <= 1:
        raise ValueError("coverage must be in (0, 1]")
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("threads must be a positive integer")
    if not isinstance(split_memory_limit, str) or not split_memory_limit.strip():
        raise ValueError("split_memory_limit must be a non-empty MMseqs2 size")
    if isinstance(kmer_length, bool) or not isinstance(kmer_length, int):
        raise ValueError("kmer_length must be a positive integer")
    if kmer_length <= 0:
        raise ValueError("kmer_length must be a positive integer")

    ordered_digests, digest_to_sequence = _normalise_sequences(sequences)
    weights = _normalise_weights(
        sequence_weights,
        ordered_digests=ordered_digests,
        digest_to_sequence=digest_to_sequence,
    )

    directory = Path(work_dir)
    directory.mkdir(parents=True, exist_ok=True)
    input_fasta = directory / "mmseqs_input.fasta"
    output_prefix = directory / "mmseqs_clusters"
    temporary_directory = directory / "mmseqs_tmp"
    _write_digest_fasta(input_fasta, digest_to_sequence)

    executable_string = os.fspath(executable)
    version = _run_mmseqs_version(executable_string)
    command = [
        executable_string,
        "easy-cluster",
        os.fspath(input_fasta),
        os.fspath(output_prefix),
        os.fspath(temporary_directory),
        "--min-seq-id",
        format(float(identity), "g"),
        "-c",
        format(float(coverage), "g"),
        "--cov-mode",
        "0",
        "--cluster-mode",
        "1",
        "--single-step-clustering",
        "1",
        "--threads",
        str(threads),
        "--mask",
        "0",
        "--split-memory-limit",
        split_memory_limit,
        "-k",
        str(kmer_length),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)

    cluster_path = Path(f"{output_prefix}_cluster.tsv")
    raw_assignments = _parse_cluster_tsv(cluster_path, set(digest_to_sequence))
    cluster_assignments = canonicalize_cluster_assignments(raw_assignments)

    cluster_weights: dict[str, float] = {}
    for digest, cluster_id in cluster_assignments.items():
        cluster_weights[cluster_id] = cluster_weights.get(cluster_id, 0.0) + weights[digest]
    cluster_partitions = allocate_cluster_partitions(
        cluster_weights,
        fractions=_FRACTIONS,
        seed=allocation_seed,
        tolerance=allocation_tolerance,
    )

    assignments = {
        digest: {
            "cluster_id": cluster_assignments[digest],
            "partition": cluster_partitions[cluster_assignments[digest]],
        }
        for digest in sorted(cluster_assignments)
    }
    cohort_sha256 = _sha256_lines(sorted(assignments))
    assignment_sha256 = _sha256_lines(
        f"{digest}\t{entry['cluster_id']}\t{entry['partition']}"
        for digest, entry in assignments.items()
    )

    return {
        "schema_version": 1,
        "source_sha256": source_sha256.lower(),
        "cohort_sha256": cohort_sha256,
        "assignment_sha256": assignment_sha256,
        "clustering": {
            "backend": "mmseqs",
            "version": version,
            "parameters": {
                "min_seq_id": float(identity),
                "coverage": float(coverage),
                "cov_mode": 0,
                "cluster_mode": 1,
                "single_step_clustering": 1,
                "threads": threads,
                "mask": 0,
                "split_memory_limit": split_memory_limit,
                "kmer_length": kmer_length,
            },
        },
        "allocation": {
            "fractions": {
                "train": _FRACTIONS[0],
                "cv": _FRACTIONS[1],
                "test": _FRACTIONS[2],
            },
            "seed": allocation_seed,
            "tolerance": allocation_tolerance,
            "weight_definition": (
                "caller-supplied valid target counts summed within each "
                "similarity cluster"
            ),
            "sequences": len(weights),
            "clusters": len(cluster_weights),
            "total_weight": float(sum(weights.values())),
            "partition_weight": {
                partition: float(sum(
                    cluster_weights[cluster_id]
                    for cluster_id, assigned in cluster_partitions.items()
                    if assigned == partition
                ))
                for partition in ("train", "cv", "test")
            },
        },
        "assignments": assignments,
    }


def write_manifest_atomic(
    manifest: Mapping[str, Any], path: str | os.PathLike[str]
) -> Path:
    """Atomically write a manifest as deterministic, UTF-8 JSON."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    except BaseException:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
        raise
    return destination


__all__ = ["build_mmseqs_manifest", "write_manifest_atomic"]
