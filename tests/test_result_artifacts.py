"""Tests for cohort-level result tables and figures."""

import pandas as pd

from experiments.validate_archive import validate_sequence_length_distribution
from experiments.run_ablation import write_sequence_length_artifacts


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
