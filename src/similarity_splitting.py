"""Deterministic helpers for similarity-clustered dataset splits.

This module deliberately does not perform sequence clustering.  A separate,
version-pinned preprocessing step is expected to turn sequence digests into raw
cluster labels.  The helpers below canonicalise those labels, allocate complete
clusters to partitions, validate a frozen manifest, and map the manifest back to
dataframe row indices without exposing raw sequences in the manifest.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import math
import re
from typing import Any, Iterable, Mapping, Sequence


PARTITIONS = ("train", "cv", "test")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def canonical_sequence(sequence: str) -> str:
    """Return a canonical RNA sequence, rejecting ambiguous or non-RNA input."""
    if not isinstance(sequence, str):
        raise ValueError("RNA sequence must be a string")
    canonical = sequence.strip().upper()
    if not canonical:
        raise ValueError("RNA sequence must not be empty")
    invalid = sorted(set(canonical) - set("ACGU"))
    if invalid:
        raise ValueError(
            "RNA sequence contains unsupported symbols: " + ", ".join(invalid)
        )
    return canonical


def sequence_digest(sequence: str) -> str:
    """Return the SHA-256 digest of a canonical RNA sequence."""
    return hashlib.sha256(canonical_sequence(sequence).encode("utf-8")).hexdigest()


def _normalise_digest(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a hexadecimal SHA-256 string")
    digest = value.strip().lower()
    if not _SHA256_RE.fullmatch(digest):
        raise ValueError(f"{field_name} must be a hexadecimal SHA-256 string")
    return digest


def _cluster_id(member_digests: Iterable[str]) -> str:
    members = sorted(member_digests)
    if not members:
        raise ValueError("Similarity clusters must not be empty")
    payload = "\n".join(members).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonicalize_cluster_assignments(
    raw_assignments: Mapping[str, Any],
) -> dict[str, str]:
    """Replace backend-specific cluster labels with stable content hashes.

    Parameters
    ----------
    raw_assignments:
        Mapping from sequence SHA-256 digest to an arbitrary, hashable raw
        cluster label (for example an MMseqs2 representative identifier).

    Returns
    -------
    dict
        Mapping from normalised sequence digest to a canonical cluster ID.  A
        cluster ID is the SHA-256 hash of its sorted member digests, so it does
        not depend on input order or on the backend's chosen representative.
    """
    if not isinstance(raw_assignments, Mapping) or not raw_assignments:
        raise ValueError("raw_assignments must be a non-empty mapping")

    clusters: dict[Any, list[str]] = defaultdict(list)
    seen: dict[str, Any] = {}
    for member_digest, raw_label in raw_assignments.items():
        digest = _normalise_digest(member_digest, "sequence digest")
        try:
            hash(raw_label)
        except TypeError as exc:
            raise ValueError("Raw cluster labels must be hashable") from exc
        if raw_label is None or (isinstance(raw_label, str) and not raw_label.strip()):
            raise ValueError("Raw cluster labels must not be empty")
        if digest in seen and seen[digest] != raw_label:
            raise ValueError("A sequence digest belongs to multiple raw clusters")
        seen[digest] = raw_label
        clusters[raw_label].append(digest)

    canonical: dict[str, str] = {}
    for members in clusters.values():
        unique_members = sorted(set(members))
        cluster_id = _cluster_id(unique_members)
        for digest in unique_members:
            if digest in canonical:
                raise ValueError("A sequence digest belongs to multiple clusters")
            canonical[digest] = cluster_id
    return canonical


def _validate_fractions(fractions: Sequence[float]) -> tuple[float, float, float]:
    if len(fractions) != 3:
        raise ValueError("fractions must contain train, CV, and test values")
    try:
        values = tuple(float(value) for value in fractions)
    except (TypeError, ValueError) as exc:
        raise ValueError("fractions must be finite numeric values") from exc
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("fractions must be finite and greater than zero")
    if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("fractions must sum to 1")
    return values  # type: ignore[return-value]


def _stable_tie_break(seed: int, cluster_id: str, partition: str = "") -> str:
    payload = f"{int(seed)}\0{cluster_id}\0{partition}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def allocate_cluster_partitions(
    cluster_weights: Mapping[str, float],
    fractions: Sequence[float] = (0.70, 0.15, 0.15),
    seed: int = 20261001,
    tolerance: float = 0.03,
) -> dict[str, str]:
    """Allocate whole clusters to train/CV/test with deterministic balancing.

    Clusters are considered in descending weight order.  At each step, the
    partition that minimises total squared relative deviation from the target
    weights is chosen; SHA-256 tie breakers make the result independent of
    mapping insertion order.  The function fails closed when the achieved
    fractions are infeasible within ``tolerance``.
    """
    target_fractions = _validate_fractions(fractions)
    if not isinstance(cluster_weights, Mapping) or len(cluster_weights) < 3:
        raise ValueError("At least three similarity clusters are required")
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    try:
        tolerance_value = float(tolerance)
    except (TypeError, ValueError) as exc:
        raise ValueError("tolerance must be a finite number") from exc
    if not math.isfinite(tolerance_value) or not 0.0 <= tolerance_value <= 1.0:
        raise ValueError("tolerance must be between 0 and 1")

    weights: dict[str, float] = {}
    for cluster_id, raw_weight in cluster_weights.items():
        if not isinstance(cluster_id, str) or not cluster_id.strip():
            raise ValueError("Cluster IDs must be non-empty strings")
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError) as exc:
            raise ValueError("Cluster weights must be finite and positive") from exc
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("Cluster weights must be finite and positive")
        if cluster_id in weights:
            raise ValueError(f"Duplicate cluster ID: {cluster_id}")
        weights[cluster_id] = weight

    total_weight = sum(weights.values())
    targets = {
        partition: total_weight * fraction
        for partition, fraction in zip(PARTITIONS, target_fractions)
    }
    totals = {partition: 0.0 for partition in PARTITIONS}
    counts = {partition: 0 for partition in PARTITIONS}
    assignments: dict[str, str] = {}
    ordered = sorted(
        weights.items(),
        key=lambda item: (-item[1], _stable_tie_break(seed, item[0])),
    )

    for position, (cluster_id, weight) in enumerate(ordered):
        empty_partitions = [name for name in PARTITIONS if counts[name] == 0]
        remaining_including_current = len(ordered) - position
        candidates = (
            empty_partitions
            if empty_partitions and remaining_including_current == len(empty_partitions)
            else list(PARTITIONS)
        )

        scored_candidates = []
        for partition in candidates:
            prospective = dict(totals)
            prospective[partition] += weight
            relative_squared_error = sum(
                ((prospective[name] - targets[name]) / targets[name]) ** 2
                for name in PARTITIONS
            )
            scored_candidates.append(
                (
                    relative_squared_error,
                    _stable_tie_break(seed, cluster_id, partition),
                    partition,
                )
            )
        _, _, selected = min(scored_candidates)
        assignments[cluster_id] = selected
        totals[selected] += weight
        counts[selected] += 1

    if any(counts[partition] == 0 for partition in PARTITIONS):
        raise ValueError("Cluster allocation produced an empty partition")

    achieved = {name: totals[name] / total_weight for name in PARTITIONS}
    deviations = {
        name: abs(achieved[name] - target)
        for name, target in zip(PARTITIONS, target_fractions)
    }
    if max(deviations.values()) > tolerance_value + 1e-12:
        formatted = ", ".join(
            f"{name}={achieved[name]:.4f}" for name in PARTITIONS
        )
        raise ValueError(
            "Similarity-cluster weights cannot satisfy the requested fractions "
            f"within tolerance {tolerance_value:.4f}; achieved {formatted}"
        )
    return assignments


def _assignment_checksum(assignments: Mapping[str, Mapping[str, str]]) -> str:
    lines = [
        f"{digest}\t{record['cluster_id']}\t{record['partition']}"
        for digest, record in sorted(assignments.items())
    ]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _cohort_checksum(sequence_digests: Iterable[str]) -> str:
    return hashlib.sha256(
        "\n".join(sorted(sequence_digests)).encode("utf-8")
    ).hexdigest()


def _normalise_manifest_assignments(
    manifest: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    raw_assignments = manifest.get("assignments")
    if not isinstance(raw_assignments, Mapping) or not raw_assignments:
        raise ValueError("Manifest assignments must be a non-empty mapping")

    assignments: dict[str, dict[str, str]] = {}
    for raw_digest, raw_record in raw_assignments.items():
        digest = _normalise_digest(raw_digest, "manifest sequence digest")
        if digest in assignments:
            raise ValueError("Manifest contains duplicate sequence digests")
        if not isinstance(raw_record, Mapping):
            raise ValueError("Each manifest assignment must be a mapping")
        cluster_id = _normalise_digest(
            raw_record.get("cluster_id"), "manifest cluster ID"
        )
        partition = raw_record.get("partition")
        if partition not in PARTITIONS:
            raise ValueError(
                "Manifest partition must be one of: " + ", ".join(PARTITIONS)
            )
        assignments[digest] = {
            "cluster_id": cluster_id,
            "partition": str(partition),
        }
    return assignments


def _validate_cross_split_search_audit(manifest: Mapping[str, Any]) -> None:
    audit = manifest.get("cross_split_search_audit")
    if not isinstance(audit, Mapping):
        raise ValueError("Manifest is missing the independent cross-split search audit")
    if audit.get("status") != "passed":
        raise ValueError("Cross-split search audit did not pass")
    if audit.get("backend") != "mmseqs" or audit.get("search") != "easy-search":
        raise ValueError("Cross-split audit must use an independent MMseqs2 easy-search")
    if not isinstance(audit.get("version"), str) or not audit["version"].strip():
        raise ValueError("Cross-split audit must record the MMseqs2 version")

    for checksum_name in ("cohort_sha256", "assignment_sha256"):
        audit_checksum = _normalise_digest(
            audit.get(checksum_name), f"audit {checksum_name}"
        )
        manifest_checksum = _normalise_digest(
            manifest.get(checksum_name), f"manifest {checksum_name}"
        )
        if audit_checksum != manifest_checksum:
            raise ValueError(f"Cross-split audit {checksum_name} does not match manifest")

    hit_count = audit.get("qualifying_hit_count")
    if isinstance(hit_count, bool) or hit_count != 0:
        raise ValueError("Cross-split audit contains qualifying similarity hits")
    for field in (
        "max_qualifying_fident",
        "max_qualifying_qcov",
        "max_qualifying_tcov",
    ):
        if audit.get(field) is not None:
            raise ValueError(f"Passed cross-split audit must record {field} as null")

    clustering = manifest.get("clustering")
    clustering_parameters = (
        clustering.get("parameters") if isinstance(clustering, Mapping) else None
    )
    audit_parameters = audit.get("parameters")
    if not isinstance(clustering_parameters, Mapping):
        raise ValueError("Manifest clustering parameters are missing")
    if not isinstance(audit_parameters, Mapping):
        raise ValueError("Cross-split audit parameters are missing")
    for parameter in ("min_seq_id", "coverage"):
        try:
            audit_value = float(audit_parameters.get(parameter))
            clustering_value = float(clustering_parameters.get(parameter))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Cross-split audit parameter is invalid: {parameter}"
            ) from exc
        if not math.isfinite(audit_value) or not math.isclose(
            audit_value, clustering_value, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                f"Cross-split audit {parameter} must match the clustering threshold"
            )
    fixed_parameters = {
        "cov_mode": 0,
        "alignment_mode": 3,
        "seq_id_mode": 0,
        "mask": 0,
    }
    for parameter, expected in fixed_parameters.items():
        if audit_parameters.get(parameter) != expected:
            raise ValueError(
                f"Cross-split audit parameter {parameter} must equal {expected}"
            )
    try:
        sensitivity = float(audit_parameters.get("sensitivity"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Cross-split audit sensitivity is invalid") from exc
    if not math.isfinite(sensitivity) or sensitivity < 7.5:
        raise ValueError("Cross-split audit sensitivity must be at least 7.5")

    expected_directions = {
        (query, target)
        for query in PARTITIONS
        for target in PARTITIONS
        if query != target
    }
    raw_direction_results = audit.get("direction_results")
    if not isinstance(raw_direction_results, list):
        raise ValueError("Cross-split audit direction results must be a list")
    observed_directions: list[tuple[Any, Any]] = []
    for result in raw_direction_results:
        if not isinstance(result, Mapping):
            raise ValueError("Each cross-split audit direction must be a mapping")
        observed_directions.append(
            (result.get("query_partition"), result.get("target_partition"))
        )
        direction_hits = result.get("qualifying_hits")
        if isinstance(direction_hits, bool) or direction_hits != 0:
            raise ValueError("Cross-split audit direction contains qualifying hits")
    if len(set(observed_directions)) != len(observed_directions):
        raise ValueError("Cross-split audit contains duplicate directed searches")
    if set(observed_directions) != expected_directions:
        raise ValueError("Cross-split audit did not run all six directed searches")


def validate_similarity_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_sequence_digests: Iterable[str] | None = None,
    expected_source_sha256: str | None = None,
    expected_cohort_sha256: str | None = None,
    require_cross_split_audit: bool = False,
) -> None:
    """Validate a frozen similarity split manifest, raising on any mismatch."""
    if not isinstance(manifest, Mapping):
        raise ValueError("Similarity split manifest must be a mapping")
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported similarity split manifest schema_version")

    source_sha256 = _normalise_digest(
        manifest.get("source_sha256"), "manifest source_sha256"
    )
    cohort_sha256 = _normalise_digest(
        manifest.get("cohort_sha256"), "manifest cohort_sha256"
    )
    assignments = _normalise_manifest_assignments(manifest)

    # Verify the frozen bytes-equivalent content before reporting downstream
    # semantic failures.  This makes any edit to a committed manifest surface
    # unambiguously as tampering rather than as a secondary allocation error.
    assignment_sha256 = _normalise_digest(
        manifest.get("assignment_sha256"), "manifest assignment_sha256"
    )
    if assignment_sha256 != _assignment_checksum(assignments):
        raise ValueError("Manifest assignment checksum does not match its assignments")
    computed_cohort_sha256 = _cohort_checksum(assignments)
    if cohort_sha256 != computed_cohort_sha256:
        raise ValueError("Manifest cohort_sha256 does not match its assignments")

    members_by_cluster: dict[str, list[str]] = defaultdict(list)
    partitions_by_cluster: dict[str, set[str]] = defaultdict(set)
    partition_counts = {name: 0 for name in PARTITIONS}
    for digest, record in assignments.items():
        cluster_id = record["cluster_id"]
        partition = record["partition"]
        members_by_cluster[cluster_id].append(digest)
        partitions_by_cluster[cluster_id].add(partition)
        partition_counts[partition] += 1

    for cluster_id, members in members_by_cluster.items():
        if _cluster_id(members) != cluster_id:
            raise ValueError(
                "Manifest cluster ID does not match its canonical member digest hash"
            )
        if len(partitions_by_cluster[cluster_id]) != 1:
            raise ValueError("A similarity cluster spans multiple partitions")
    if any(partition_counts[name] == 0 for name in PARTITIONS):
        raise ValueError("Manifest must contain non-empty train, CV, and test partitions")

    if expected_sequence_digests is not None:
        expected = {
            _normalise_digest(value, "expected sequence digest")
            for value in expected_sequence_digests
        }
        observed = set(assignments)
        if expected != observed:
            missing = len(expected - observed)
            extra = len(observed - expected)
            raise ValueError(
                "Manifest cohort differs from the expected sequence digests "
                f"(missing={missing}, extra={extra})"
            )
    if expected_source_sha256 is not None:
        expected_source = _normalise_digest(
            expected_source_sha256, "expected source SHA-256"
        )
        if source_sha256 != expected_source:
            raise ValueError("Manifest source SHA-256 does not match the data source")
    if expected_cohort_sha256 is not None:
        expected_cohort = _normalise_digest(
            expected_cohort_sha256, "expected cohort SHA-256"
        )
        if cohort_sha256 != expected_cohort:
            raise ValueError("Manifest cohort SHA-256 does not match the expected cohort")
    if require_cross_split_audit:
        _validate_cross_split_search_audit(manifest)


def _row_assignment_records(
    dataframe: Any,
    manifest: Mapping[str, Any],
    sequence_col: str,
) -> list[dict[str, str]]:
    if sequence_col not in dataframe.columns:
        raise ValueError(f"Required sequence column not found: {sequence_col}")
    validate_similarity_manifest(manifest)
    assignments = _normalise_manifest_assignments(manifest)

    records: list[dict[str, str]] = []
    observed: set[str] = set()
    for sequence in dataframe[sequence_col].tolist():
        digest = sequence_digest(sequence)
        record = assignments.get(digest)
        if record is None:
            raise ValueError(
                "Dataframe contains a sequence absent from the similarity manifest: "
                f"{digest[:12]}"
            )
        observed.add(digest)
        records.append(record)

    extra = set(assignments) - observed
    if extra:
        raise ValueError(
            "Similarity manifest contains sequences absent from the dataframe "
            f"(extra={len(extra)})"
        )
    return records


def split_indices_from_manifest(
    dataframe: Any,
    manifest: Mapping[str, Any],
    sequence_col: str = "sequence",
) -> tuple[list[int], list[int], list[int]]:
    """Map a validated frozen manifest to positional dataframe row indices."""
    records = _row_assignment_records(dataframe, manifest, sequence_col)
    by_partition = {name: [] for name in PARTITIONS}
    for row_index, record in enumerate(records):
        by_partition[record["partition"]].append(row_index)
    if any(not by_partition[name] for name in PARTITIONS):
        raise ValueError("Manifest mapping produced an empty dataframe partition")
    return tuple(by_partition[name] for name in PARTITIONS)  # type: ignore[return-value]


def similarity_cluster_overlap_counts(
    dataframe: Any,
    splits: Sequence[Sequence[int]],
    manifest: Mapping[str, Any],
    sequence_col: str = "sequence",
) -> dict[str, int]:
    """Count similarity clusters shared by each pair of row-index splits."""
    if len(splits) != 3:
        raise ValueError("splits must contain train, CV, and test indices")
    records = _row_assignment_records(dataframe, manifest, sequence_col)
    flattened = [int(index) for split in splits for index in split]
    if len(flattened) != len(records) or len(set(flattened)) != len(records):
        raise ValueError("Split partitions must contain every dataframe row exactly once")
    if set(flattened) != set(range(len(records))):
        raise ValueError("Split partitions contain out-of-range or missing row indices")

    cluster_sets = [
        {records[int(index)]["cluster_id"] for index in split}
        for split in splits
    ]
    return {
        "train_cv": len(cluster_sets[0] & cluster_sets[1]),
        "train_test": len(cluster_sets[0] & cluster_sets[2]),
        "cv_test": len(cluster_sets[1] & cluster_sets[2]),
    }
