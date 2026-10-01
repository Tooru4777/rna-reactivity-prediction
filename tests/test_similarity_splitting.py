"""Tests for deterministic similarity-cluster manifests and split mapping."""

import hashlib

import pandas as pd
import pytest

from src.similarity_splitting import (
    allocate_cluster_partitions,
    canonical_sequence,
    canonicalize_cluster_assignments,
    sequence_digest,
    similarity_cluster_overlap_counts,
    split_indices_from_manifest,
    validate_similarity_manifest,
)


def _checksum_assignments(assignments):
    lines = [
        f"{digest}\t{record['cluster_id']}\t{record['partition']}"
        for digest, record in sorted(assignments.items())
    ]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _cohort_checksum(assignments):
    return hashlib.sha256(
        "\n".join(sorted(assignments)).encode("utf-8")
    ).hexdigest()


def _manifest(sequence_to_raw_cluster, raw_cluster_partitions):
    raw = {
        sequence_digest(sequence): raw_cluster
        for sequence, raw_cluster in sequence_to_raw_cluster.items()
    }
    canonical = canonicalize_cluster_assignments(raw)
    assignments = {
        digest: {
            "cluster_id": cluster_id,
            "partition": raw_cluster_partitions[raw[digest]],
        }
        for digest, cluster_id in canonical.items()
    }
    return {
        "schema_version": 1,
        "source_sha256": "a" * 64,
        "cohort_sha256": _cohort_checksum(assignments),
        "assignment_sha256": _checksum_assignments(assignments),
        "assignments": assignments,
    }


def _partition_by_sequence(frame, splits):
    result = {}
    for partition, indices in zip(("train", "cv", "test"), splits):
        for index in indices:
            result[canonical_sequence(frame.iloc[index]["sequence"])] = partition
    return result


def test_canonical_sequence_and_digest_are_strict_and_stable():
    assert canonical_sequence("  acgu\n") == "ACGU"
    assert sequence_digest("acgu") == sequence_digest(" ACGU ")
    with pytest.raises(ValueError, match="must not be empty"):
        canonical_sequence("  ")
    with pytest.raises(ValueError, match="unsupported symbols"):
        canonical_sequence("ACGT")
    with pytest.raises(ValueError, match="must be a string"):
        canonical_sequence(None)


def test_cluster_ids_depend_only_on_sorted_member_digests():
    digest_a = sequence_digest("AAAA")
    digest_b = sequence_digest("AAAC")
    digest_c = sequence_digest("CCCC")
    first = canonicalize_cluster_assignments(
        {digest_a: "representative-a", digest_b: "representative-a", digest_c: "c"}
    )
    second = canonicalize_cluster_assignments(
        {digest_c: 99, digest_b: "different-label", digest_a: "different-label"}
    )
    assert first == second
    assert first[digest_a] == first[digest_b]
    assert first[digest_a] != first[digest_c]


def test_cluster_assignment_rejects_conflicting_normalised_digest():
    digest = sequence_digest("AAAA")
    with pytest.raises(ValueError, match="multiple raw clusters"):
        canonicalize_cluster_assignments({digest: "one", digest.upper(): "two"})


def test_weighted_cluster_allocation_is_deterministic_and_balanced():
    weights = {
        hashlib.sha256(f"cluster-{index}".encode()).hexdigest(): 1.0
        for index in range(100)
    }
    forward = allocate_cluster_partitions(weights, seed=17)
    reverse = allocate_cluster_partitions(dict(reversed(list(weights.items()))), seed=17)
    assert forward == reverse
    assert set(forward.values()) == {"train", "cv", "test"}
    counts = {
        partition: sum(value == partition for value in forward.values())
        for partition in ("train", "cv", "test")
    }
    assert counts == {"train": 70, "cv": 15, "test": 15}


def test_weighted_cluster_allocation_fails_when_requested_ratio_is_infeasible():
    weights = {
        hashlib.sha256(name.encode()).hexdigest(): weight
        for name, weight in (("giant", 98.0), ("small-a", 1.0), ("small-b", 1.0))
    }
    with pytest.raises(ValueError, match="cannot satisfy"):
        allocate_cluster_partitions(weights, tolerance=0.03)
    with pytest.raises(ValueError, match="At least three"):
        allocate_cluster_partitions(dict(list(weights.items())[:2]))


def test_manifest_maps_all_profiles_of_a_sequence_to_one_partition():
    manifest = _manifest(
        {"AAAA": "a", "CCCC": "c", "GGGG": "g"},
        {"a": "train", "c": "cv", "g": "test"},
    )
    frame = pd.DataFrame(
        {"sequence": ["AAAA", "AAAA", "CCCC", "GGGG", "CCCC"]}
    )
    splits = split_indices_from_manifest(frame, manifest)
    assert splits == ([0, 1], [2, 4], [3])
    assert similarity_cluster_overlap_counts(frame, splits, manifest) == {
        "train_cv": 0,
        "train_test": 0,
        "cv_test": 0,
    }


def test_manifest_partition_membership_is_invariant_to_dataframe_row_order():
    manifest = _manifest(
        {"AAAA": "a", "CCCC": "c", "GGGG": "g"},
        {"a": "train", "c": "cv", "g": "test"},
    )
    original = pd.DataFrame({"sequence": ["AAAA", "CCCC", "GGGG"]})
    shuffled = original.iloc[[2, 0, 1]].reset_index(drop=True)
    assert _partition_by_sequence(
        original, split_indices_from_manifest(original, manifest)
    ) == _partition_by_sequence(
        shuffled, split_indices_from_manifest(shuffled, manifest)
    )


def test_cluster_overlap_audit_detects_a_cluster_split_across_partitions():
    manifest = _manifest(
        {"AAAA": "shared", "AAAC": "shared", "CCCC": "c", "GGGG": "g"},
        {"shared": "train", "c": "cv", "g": "test"},
    )
    frame = pd.DataFrame({"sequence": ["AAAA", "AAAC", "CCCC", "GGGG"]})
    overlaps = similarity_cluster_overlap_counts(
        frame,
        ([0, 2], [1], [3]),
        manifest,
    )
    assert overlaps == {"train_cv": 1, "train_test": 0, "cv_test": 0}


def test_manifest_validation_fails_closed_on_hash_and_membership_mismatches():
    manifest = _manifest(
        {"AAAA": "a", "CCCC": "c", "GGGG": "g"},
        {"a": "train", "c": "cv", "g": "test"},
    )
    expected = [sequence_digest(value) for value in ("AAAA", "CCCC", "GGGG")]
    validate_similarity_manifest(
        manifest,
        expected_sequence_digests=expected,
        expected_source_sha256="a" * 64,
        expected_cohort_sha256=manifest["cohort_sha256"],
    )

    with pytest.raises(ValueError, match="data source"):
        validate_similarity_manifest(manifest, expected_source_sha256="b" * 64)
    with pytest.raises(ValueError, match="missing=1"):
        validate_similarity_manifest(
            manifest,
            expected_sequence_digests=expected + [sequence_digest("UUUU")],
        )

    tampered = {
        **manifest,
        "assignments": {
            digest: dict(record) for digest, record in manifest["assignments"].items()
        },
    }
    train_digest = next(
        digest for digest, record in tampered["assignments"].items()
        if record["partition"] == "train"
    )
    cv_digest = next(
        digest for digest, record in tampered["assignments"].items()
        if record["partition"] == "cv"
    )
    tampered["assignments"][train_digest]["partition"] = "cv"
    tampered["assignments"][cv_digest]["partition"] = "train"
    with pytest.raises(ValueError, match="assignment checksum"):
        validate_similarity_manifest(tampered)


def test_manifest_rejects_noncanonical_cluster_id_and_extra_sequences():
    manifest = _manifest(
        {"AAAA": "a", "CCCC": "c", "GGGG": "g"},
        {"a": "train", "c": "cv", "g": "test"},
    )
    broken = {
        **manifest,
        "assignments": {
            digest: dict(record) for digest, record in manifest["assignments"].items()
        },
    }
    first_digest = next(iter(broken["assignments"]))
    broken["assignments"][first_digest]["cluster_id"] = "f" * 64
    broken["assignment_sha256"] = _checksum_assignments(broken["assignments"])
    with pytest.raises(ValueError, match="canonical member digest hash"):
        validate_similarity_manifest(broken)

    incomplete_frame = pd.DataFrame({"sequence": ["AAAA", "CCCC"]})
    with pytest.raises(ValueError, match="absent from the dataframe"):
        split_indices_from_manifest(incomplete_frame, manifest)


def test_overlap_audit_rejects_invalid_row_partitions():
    manifest = _manifest(
        {"AAAA": "a", "CCCC": "c", "GGGG": "g"},
        {"a": "train", "c": "cv", "g": "test"},
    )
    frame = pd.DataFrame({"sequence": ["AAAA", "CCCC", "GGGG"]})
    with pytest.raises(ValueError, match="every dataframe row exactly once"):
        similarity_cluster_overlap_counts(frame, ([0], [0], [2]), manifest)
