"""Tests for deterministic MMseqs2 similarity-manifest construction."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src import mmseqs_clustering
from src.similarity_splitting import sequence_digest, validate_similarity_manifest


def _sequences(count: int) -> list[str]:
    return [
        "".join(bases)
        for bases in itertools.islice(itertools.product("ACGU", repeat=3), count)
    ]


def _mock_mmseqs(monkeypatch: pytest.MonkeyPatch, rows: list[tuple[str, str]]):
    calls: list[list[str]] = []

    def fake_run(command, *, check, capture_output, text):
        assert check is True
        assert capture_output is True
        assert text is True
        calls.append(command)
        if command[1] == "version":
            return SimpleNamespace(stdout="15.6f452\n", stderr="")
        assert command[1] == "easy-cluster"
        output_path = Path(f"{command[3]}_cluster.tsv")
        output_path.write_text(
            "".join(f"{representative}\t{member}\n" for representative, member in rows),
            encoding="utf-8",
        )
        return SimpleNamespace(stdout="", stderr="")

    monkeypatch.setattr(mmseqs_clustering.subprocess, "run", fake_run)
    return calls


def test_build_manifest_writes_sorted_digest_fasta_and_fixed_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sequences = _sequences(20)
    digests = [sequence_digest(sequence) for sequence in sequences]
    calls = _mock_mmseqs(monkeypatch, [(digest, digest) for digest in digests])

    manifest = mmseqs_clustering.build_mmseqs_manifest(
        reversed(sequences),
        source_sha256="a" * 64,
        sequence_weights={digest: 1.0 for digest in digests},
        work_dir=tmp_path,
        executable="custom-mmseqs",
    )

    assert calls == [
        ["custom-mmseqs", "version"],
        [
            "custom-mmseqs",
            "easy-cluster",
            str(tmp_path / "mmseqs_input.fasta"),
            str(tmp_path / "mmseqs_clusters"),
            str(tmp_path / "mmseqs_tmp"),
            "--min-seq-id",
            "0.8",
            "-c",
            "0.8",
            "--cov-mode",
            "0",
            "--cluster-mode",
            "1",
            "--single-step-clustering",
            "1",
            "--threads",
            "1",
            "--mask",
            "0",
            "--split-memory-limit",
            "4G",
            "-k",
            "8",
        ],
    ]

    fasta_lines = (tmp_path / "mmseqs_input.fasta").read_text(
        encoding="utf-8"
    ).splitlines()
    assert fasta_lines[::2] == [f">{digest}" for digest in sorted(digests)]
    digest_to_sequence = {
        sequence_digest(sequence): sequence for sequence in sequences
    }
    assert fasta_lines[1::2] == [
        digest_to_sequence[digest] for digest in sorted(digests)
    ]

    assert manifest["clustering"] == {
        "backend": "mmseqs",
        "version": "15.6f452",
        "parameters": {
            "min_seq_id": 0.8,
            "coverage": 0.8,
            "cov_mode": 0,
            "cluster_mode": 1,
            "single_step_clustering": 1,
            "threads": 1,
            "mask": 0,
            "split_memory_limit": "4G",
            "kmer_length": 8,
        },
    }


def test_build_manifest_rejects_missing_member(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sequences = _sequences(3)
    digests = [sequence_digest(sequence) for sequence in sequences]
    _mock_mmseqs(monkeypatch, [(digests[0], digests[0]), (digests[1], digests[1])])

    with pytest.raises(ValueError, match="missing members"):
        mmseqs_clustering.build_mmseqs_manifest(
            sequences,
            source_sha256="b" * 64,
            sequence_weights={digest: 1.0 for digest in digests},
            work_dir=tmp_path,
        )


def test_build_manifest_rejects_conflicting_duplicate_member(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sequences = _sequences(3)
    digests = [sequence_digest(sequence) for sequence in sequences]
    _mock_mmseqs(
        monkeypatch,
        [
            (digests[0], digests[0]),
            (digests[0], digests[1]),
            (digests[2], digests[1]),
            (digests[2], digests[2]),
        ],
    )

    with pytest.raises(ValueError, match="conflicting representatives"):
        mmseqs_clustering.build_mmseqs_manifest(
            sequences,
            source_sha256="c" * 64,
            sequence_weights={digest: 1.0 for digest in digests},
            work_dir=tmp_path,
        )


def test_manifest_passes_similarity_manifest_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sequences = _sequences(20)
    digests = [sequence_digest(sequence) for sequence in sequences]
    _mock_mmseqs(monkeypatch, [(digest, digest) for digest in digests])

    manifest = mmseqs_clustering.build_mmseqs_manifest(
        sequences,
        source_sha256="d" * 64,
        sequence_weights=tuple(1.0 for _ in sequences),
        work_dir=tmp_path,
    )

    validated = validate_similarity_manifest(
        manifest,
        expected_sequence_digests=digests,
        expected_source_sha256="d" * 64,
        expected_cohort_sha256=manifest["cohort_sha256"],
    )
    assert validated is not False


def test_write_manifest_atomic_replaces_destination(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "manifest.json"
    destination.parent.mkdir()
    destination.write_text("old content", encoding="utf-8")
    manifest = {"schema_version": 1, "text": "RNA"}

    result = mmseqs_clustering.write_manifest_atomic(manifest, destination)

    assert result == destination
    assert json.loads(destination.read_text(encoding="utf-8")) == manifest
    assert not list(destination.parent.glob(f".{destination.name}.*.tmp"))
