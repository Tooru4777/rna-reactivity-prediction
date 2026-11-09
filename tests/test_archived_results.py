"""Integrity checks for the reportable compact Kaggle result archive."""

from pathlib import Path

from experiments.validate_archive import validate_archive


ARCHIVE = Path(__file__).resolve().parents[1] / "results" / "kaggle-v10-fec86dc"


def test_reportable_archive_is_internally_consistent():
    report = validate_archive(ARCHIVE)

    assert report["evaluations"] == 24
    assert report["grouped_overlap_max"] == 0
    assert report["selected_variant"] == "Full Model (+ViennaRNA 7d)"
    assert report["max_error_reconciliation_difference"] < 1e-8
