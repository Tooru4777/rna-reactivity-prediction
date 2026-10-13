"""Measured-position prediction diagnostics and paired baseline comparisons."""

import numpy as np
import pandas as pd


def summarize_prediction_health(errors, baselines):
    keys = ["seed", "split_method", "partition", "variant_id"]
    rows = []
    for key, frame in errors.groupby(keys, sort=True):
        count = float(frame["valid_positions"].sum())
        if count <= 0:
            raise ValueError("Health report requires measured positions")
        mean = frame["prediction_sum"].sum() / count
        variance = max(0.0, frame["prediction_squared_sum"].sum() / count - mean**2)
        record = dict(zip(keys, key))
        record.update(
            valid_positions=int(count),
            raw_prediction_min=frame["prediction_min"].min(),
            raw_prediction_max=frame["prediction_max"].max(),
            raw_prediction_mean=mean,
            raw_prediction_std=np.sqrt(variance),
            fraction_clipped_low=frame["prediction_low_count"].sum() / count,
            fraction_clipped_high=frame["prediction_high_count"].sum() / count,
            fraction_clipped=frame["prediction_clipped_count"].sum() / count,
            mae=frame["absolute_error_sum"].sum() / count,
            raw_mae=frame["raw_absolute_error_sum"].sum() / count,
            zero_prediction_mae=frame["target_sum"].sum() / count,
        )
        rows.append(record)
    health = pd.DataFrame(rows)
    reference = baselines[["seed", "split_method", "cv_mae"]].rename(
        columns={"cv_mae": "train_mean_baseline_cv_mae"}
    )
    health = health.merge(reference, on=["seed", "split_method"], validate="many_to_one")
    health["improvement_over_baseline"] = np.where(
        health["partition"].eq("cv"),
        health["train_mean_baseline_cv_mae"] - health["mae"],
        np.nan,
    )
    health["saturation_warning"] = health["fraction_clipped"] >= 0.95
    health["constant_warning"] = health["raw_prediction_std"] < 1e-6
    health["beats_baseline"] = health["improvement_over_baseline"] > 0
    return health
