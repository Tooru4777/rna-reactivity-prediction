"""Tests for cohort-level result tables and figures."""

import json
from pathlib import Path

import pandas as pd
import pytest

from experiments.validate_archive import (
    ArchiveValidationError,
    validate_artifact_checksums,
    validate_sequence_length_distribution,
)
from experiments.run_ablation import (
    evaluate_experiment_mean_baseline,
    fit_experiment_mean_baseline,
    write_artifact_checksums,
    write_length_error_figure,
    write_sequence_length_artifacts,
    write_training_curves_figure,
)


ARCHIVE = Path(__file__).resolve().parents[1] / "results" / "kaggle-v10-fec86dc"


def test_sequence_length_artifacts_count_unique_sequences_and_fill_gaps(tmp_path):
    frame = pd.DataFrame({
        "sequence": ["AAA", "AAA", "CCCCC", "GGGGG", "GGGGG"],
        "experiment_type": ["2A3_MaP", "DMS_MaP", "2A3_MaP", "2A3_MaP", "DMS_MaP"],
    })

    distribution = write_sequence_length_artifacts(frame, tmp_path)

    assert distribution["sequence_length"].tolist() == [3, 4, 5]
    assert distribution["unique_sequences"].tolist() == [1, 0, 2]
    assert distribution["experiment_profiles"].tolist() == [2, 0, 3]
    assert distribution["cumulative_unique_sequences"].tolist() == [1, 1, 3]
    assert distribution["share_of_sequences"].tolist() == [1 / 3, 0, 2 / 3]

    csv_frame = pd.read_csv(tmp_path / "sequence_length_distribution.csv")
    pd.testing.assert_frame_equal(csv_frame, distribution, check_dtype=False)
    with (tmp_path / "sequence_length_histogram.png").open("rb") as handle:
        assert handle.read(8) == b"\x89PNG\r\n\x1a\n"

    validate_sequence_length_distribution(
        tmp_path / "sequence_length_distribution.csv",
        {"filtered_unique_sequences": 3, "filtered_rows": 5},
        {"sequence_length": {"min": 3, "median": 5.0, "max": 5}},
    )


def test_experiment_mean_baseline_uses_training_rows_only():
    frame = pd.DataFrame({
        "sequence": ["AA", "CC", "GG", "UU"],
        "experiment_type": ["2A3_MaP", "2A3_MaP", "DMS_MaP", "DMS_MaP"],
        "reactivity_0001": [0.0, 0.25, 0.2, 0.1],
        "reactivity_0002": [1.0, 0.75, 0.4, 0.5],
    })
    columns = ["reactivity_0001", "reactivity_0002"]

    means = fit_experiment_mean_baseline(frame, [0, 2], columns)
    mae, valid_positions = evaluate_experiment_mean_baseline(
        frame, [1, 3], columns, means
    )

    assert means == {"2A3_MaP": pytest.approx(0.5), "DMS_MaP": pytest.approx(0.3)}
    assert mae == pytest.approx(0.225)
    assert valid_positions == 4


def test_artifact_checksums_cover_only_compact_outputs(tmp_path):
    (tmp_path / "table.csv").write_text("a\n1\n", encoding="utf-8")
    (tmp_path / "report.json").write_text('{"ok": true}\n', encoding="utf-8")
    (tmp_path / "plot.png").write_bytes(b"not-a-real-png")
    (tmp_path / "error_analysis_per_profile.csv").write_text("large\n", encoding="utf-8")
    (tmp_path / "weights.pth").write_bytes(b"weights")

    write_artifact_checksums(tmp_path)
    present = {path.name for path in tmp_path.iterdir() if path.is_file()}
    validate_artifact_checksums(tmp_path, present)
    inventory = (tmp_path / "artifact_checksums.sha256").read_text(encoding="utf-8")
    assert "table.csv" in inventory
    assert "report.json" in inventory
    assert "plot.png" in inventory
    assert "error_analysis_per_profile.csv" not in inventory
    assert "weights.pth" not in inventory

    (tmp_path / "table.csv").write_text("a\n2\n", encoding="utf-8")
    with pytest.raises(ArchiveValidationError, match="Checksum mismatch"):
        validate_artifact_checksums(tmp_path, present)


def test_revised_figures_render_from_real_archived_results(tmp_path):
    epoch_frame = pd.read_csv(ARCHIVE / "ablation_results.csv")
    length_frame = pd.read_csv(ARCHIVE / "error_by_length.csv")
    selection = json.loads((ARCHIVE / "model_selection.json").read_text(encoding="utf-8"))

    curves = write_training_curves_figure(epoch_frame, tmp_path / "training.png")
    lengths = write_length_error_figure(
        length_frame, selection, tmp_path / "length.png"
    )

    assert curves["seeds"].eq(3).all()
    assert curves["std_cv_loss"].notna().all()
    assert lengths["seeds"].eq(3).all()
    assert lengths["min_sequences_per_seed"].min() >= 1
    for filename in ("training.png", "length.png"):
        with (tmp_path / filename).open("rb") as handle:
            assert handle.read(8) == b"\x89PNG\r\n\x1a\n"
