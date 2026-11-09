import pandas as pd
import pytest

from src.prediction_health import summarize_prediction_health


def test_saturated_predictions_are_compared_to_baseline_with_measured_weights():
    frame = pd.DataFrame([{
        "seed": 42, "split_method": "clustered", "partition": "cv",
        "variant_id": "model", "valid_positions": 2,
        "prediction_sum": -4.0, "prediction_squared_sum": 8.0,
        "prediction_min": -2.0, "prediction_max": -2.0,
        "prediction_low_count": 2, "prediction_high_count": 0,
        "prediction_clipped_count": 2, "absolute_error_sum": 0.6,
        "raw_absolute_error_sum": 4.6, "target_sum": 0.6,
    }])
    baselines = pd.DataFrame([{
        "seed": 42, "split_method": "clustered", "cv_mae": 0.2,
    }])
    row = summarize_prediction_health(frame, baselines).iloc[0]
    assert row["raw_mae"] == pytest.approx(2.3)
    assert row["mae"] == pytest.approx(0.3)
    assert row["improvement_over_baseline"] == pytest.approx(-0.1)
    assert row["saturation_warning"]
    assert row["constant_warning"]
    assert not row["beats_baseline"]
