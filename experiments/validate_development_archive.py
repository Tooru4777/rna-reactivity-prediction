"""Independently validate a compact schema-v4 CV development archive."""

import argparse
import json
import hashlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.validate_archive import require, validate_artifact_checksums
from src.similarity_splitting import validate_similarity_manifest


def validate_development_archive(directory):
    root = Path(directory)
    present = {p.name for p in root.iterdir() if p.is_file()}
    require(all(Path(n).suffix in {".csv", ".json", ".png", ".md", ".sha256"}
                for n in present), "Non-compact artifact in development archive")
    require("error_analysis_per_profile.csv" not in present, "Profile-level output in archive")
    validate_artifact_checksums(root, present)
    manifest = json.loads((root / "run_manifest.json").read_text())
    require(manifest.get("artifact_schema_version") == 4, "Expected schema v4")
    require(manifest.get("run_mode") == "development" and manifest.get("reportable") is False,
            "Archive must be explicitly development-only")
    split = json.loads((root / "split_report.json").read_text())
    seeds = manifest["validation_seeds"]
    splits = set(split["splits_by_seed"][str(seeds[0])])
    expected = {(s, method, v) for s in seeds for method in splits for v in manifest["variants"]}
    summary = pd.read_csv(root / "ablation_summary.csv")
    keys = list(zip(summary.seed, summary.split_method, summary.variant_id))
    require(len(keys) == len(expected) and set(keys) == expected, "Incomplete or duplicate result grid")
    require(summary.test_mae.isna().all(), "Development test scores must be empty")
    require(pd.read_csv(root / "test_results.csv").empty, "Development test rows must be empty")
    for seed in seeds:
        record = split["splits_by_seed"][str(seed)]
        require(not any(record["grouped"]["sequence_overlap"].values()), "Exact-group overlap")
        require(not any(record["clustered"]["similarity_cluster_overlap"].values()), "Cluster overlap")
    similarity = json.loads((root / "similarity_split_manifest.json").read_text())
    validate_similarity_manifest(
        similarity, expected_source_sha256=manifest["source_file_sha256"],
        require_cross_split_audit=True,
    )
    errors = pd.read_csv(root / "error_analysis_per_sequence.csv")
    require(errors.partition.eq("cv").all(), "Non-CV errors in development archive")
    require(not errors.duplicated(["seed", "split_method", "variant_id", "sequence_hash"]).any(),
            "Duplicate sequence-level error rows")
    groups = ["seed", "split_method", "variant_id"]
    errors["recomputed_sequence_mae"] = errors.absolute_error_sum / errors.valid_positions
    recomputed = errors.groupby(groups).agg(
        error_sum=("absolute_error_sum", "sum"), positions=("valid_positions", "sum"),
        macro=("recomputed_sequence_mae", "mean"),
    ).reset_index()
    recomputed["mae"] = recomputed.error_sum / recomputed.positions
    check = summary.merge(recomputed, on=groups, validate="one_to_one")
    require(len(check) == len(expected), "Missing recomputed scores")
    require(np.allclose(check.best_cv_loss, check.mae, atol=1e-7, rtol=0), "CV score mismatch")
    require(np.allclose(check.cv_macro_sequence_mae, check.macro, atol=1e-7, rtol=0), "Macro score mismatch")
    health = pd.read_csv(root / "prediction_health.csv")
    require(health.partition.eq("cv").all(), "Non-CV health rows")
    check = health.merge(recomputed, on=groups, suffixes=("_reported", "_recomputed"), validate="one_to_one")
    require(len(check) == len(expected), "Health grid incomplete")
    require(np.allclose(check.mae_reported, check.mae_recomputed, atol=1e-7, rtol=0), "Health score mismatch")
    validate_development_details(root, manifest, summary, errors, recomputed, similarity)
    return {
        "validated_rows": len(expected), "unique_sequences": manifest["filtered_unique_sequences"],
        "test_rows": 0, "similarity_repairs": len(similarity.get("similarity_repairs", [])),
    }


def validate_development_details(root, manifest, summary, errors, recomputed, similarity):
    """Reconcile convergence, baselines, membership and all reported CV scores."""
    keys = ["seed", "split_method", "variant_id"]
    expected = set(map(tuple, summary[keys].to_numpy()))

    def complete_grid(frame, columns=keys, expected_keys=expected):
        observed = list(map(tuple, frame[columns].to_numpy()))
        require(len(observed) == len(expected_keys) and set(observed) == expected_keys,
                "Incomplete or duplicate diagnostic grid")

    require(hashlib.sha256((root / "similarity_split_manifest.json").read_bytes()).hexdigest()
            == manifest["similarity_manifest_sha256"], "Similarity file hash mismatch")
    require(similarity["assignment_sha256"] == manifest["similarity_manifest_assignment_sha256"],
            "Run assignment hash mismatch")
    require(summary.test_macro_sequence_mae.isna().all(), "Development macro test must be empty")
    require(np.isfinite(errors[["absolute_error_sum", "valid_positions"]]).all().all(),
            "Non-finite error numerator or denominator")
    require((errors.valid_positions > 0).all() and
            (errors.valid_positions == np.floor(errors.valid_positions)).all(), "Invalid denominator")
    require(errors.absolute_error_sum.between(0, errors.valid_positions + 1e-7).all(),
            "Invalid clipped error numerator")
    assignments = similarity["assignments"]
    require(set(errors.sequence_hash).issubset(assignments), "Unknown sequence in errors")
    for row in errors.itertuples():
        record = assignments[row.sequence_hash]
        require(row.cluster_id == record["cluster_id"], "Error cluster differs from manifest")
        if row.split_method == "clustered":
            require(record["partition"] == "cv", "Clustered CV error outside CV membership")
    expected_cv = {h for h, a in assignments.items() if a["partition"] == "cv"}
    for _, g in errors[errors.split_method.eq("clustered")].groupby(keys):
        require(set(g.sequence_hash) == expected_cv, "Incomplete clustered CV membership")

    epochs = pd.read_csv(root / "ablation_results.csv")
    require(set(map(tuple, epochs[keys].to_numpy())) == expected, "Epoch grid mismatch")
    require(not epochs.duplicated(keys + ["epoch"]).any(), "Duplicate epoch")
    for key, g in epochs.groupby(keys):
        g = g.sort_values("epoch")
        row = summary.set_index(keys).loc[key]
        require(g.epoch.tolist() == list(range(1, len(g) + 1)), "Non-contiguous epochs")
        require(np.isfinite(g[["train_loss", "cv_loss"]]).all().all(), "Non-finite training history")
        require(g.cv_loss.between(0, 1).all(), "Invalid CV loss")
        require(abs(g.cv_loss.min() - row.best_cv_loss) < 1e-7, "Best CV differs from epoch minimum")
        require(int(g.loc[g.cv_loss.idxmin(), "epoch"]) == int(row.best_epoch), "Best epoch mismatch")
        if "epochs_trained" in summary:
            require(int(row.epochs_trained) == len(g), "Epoch count mismatch")
            require(len(g) <= manifest["epochs"], "Epoch budget exceeded")
            patience = manifest.get("early_stopping_patience", 0)
            delta = manifest.get("early_stopping_min_delta", 0.0)
            best, stale, stop = float("inf"), 0, False
            for index, loss in enumerate(g.cv_loss, 1):
                if loss < best - delta:
                    best, stale = loss, 0
                else:
                    stale += 1
                if patience and stale >= patience:
                    require(index == len(g), "History continued after early stopping")
                    stop = True
            require(bool(row.stopped_early) == stop, "Early-stopping flag mismatch")
            require(stop or len(g) == manifest["epochs"], "History ended without stopping or budget cap")

    baselines = pd.read_csv(root / "baseline_results.csv")
    baseline_keys = ["seed", "split_method"]
    complete_grid(baselines, baseline_keys, {key[:2] for key in expected})
    require(baselines.test_mae.isna().all() and baselines.test_valid_positions.eq(0).all(),
            "Development baseline evaluated test")
    require(baselines.cv_mae.between(0, 1).all(), "Invalid baseline score")
    for field in ("mean_2A3_MaP", "mean_DMS_MaP"):
        require(baselines[field].between(0, 1).all(), "Invalid train-only constant")
    if "median_cv_mae" in baselines:
        require(baselines.median_cv_mae.between(0, 1).all() and baselines.median_test_mae.isna().all(),
                "Invalid median baseline or unlocked test")
    b = recomputed.merge(baselines, on=baseline_keys, validate="many_to_one")
    require(b.positions.eq(b.cv_valid_positions).all(), "Baseline/model denominator mismatch")
    for filename in ("prediction_health.csv", "prediction_health_initial.csv"):
        health = pd.read_csv(root / filename)
        complete_grid(health)
        require(health.partition.eq("cv").all(), "Health evaluated non-CV partition")
        for field in ("mae", "raw_mae", "raw_prediction_mean", "raw_prediction_std"):
            require(np.isfinite(health[field]).all(), "Non-finite health diagnostic")
        for field in ("fraction_clipped", "fraction_clipped_low", "fraction_clipped_high"):
            require(health[field].between(0, 1).all(), "Invalid clipped fraction")
        require(np.allclose(health.fraction_clipped, health.fraction_clipped_low +
                            health.fraction_clipped_high, atol=1e-12, rtol=0), "Clipping fractions mismatch")
        h = health.merge(baselines, on=baseline_keys, validate="many_to_one")
        require(h.valid_positions.eq(h.cv_valid_positions).all(), "Health denominator mismatch")
        require(np.allclose(h.improvement_over_baseline, h.cv_mae - h.mae, atol=1e-12, rtol=0),
                "Mean baseline improvement mismatch")
        if "median_cv_mae" in baselines:
            require(np.allclose(h.improvement_over_median_baseline, h.median_cv_mae - h.mae,
                                atol=1e-12, rtol=0), "Median baseline improvement mismatch")

    metrics = pd.read_csv(root / "evaluation_metrics.csv").rename(columns={"split": "split_method"})
    complete_grid(metrics)
    require(metrics.partition.eq("cv").all(), "Metrics evaluated non-CV partition")
    check = metrics.merge(recomputed, on=keys, validate="one_to_one")
    require(np.allclose(check.nucleotide_weighted_mae, check.mae, atol=1e-7, rtol=0), "Metric MAE mismatch")
    require(np.allclose(check.macro_sequence_mae, check.macro, atol=1e-7, rtol=0), "Metric macro mismatch")
    require(check.valid_positions.eq(check.positions).all(), "Metric denominator mismatch")
    agg = pd.read_csv(root / "ablation_aggregate.csv")
    complete_grid(agg, ["split_method", "variant_id"], {key[1:] for key in expected})
    for row in agg.itertuples():
        group = summary[(summary.split_method == row.split_method) & (summary.variant_id == row.variant_id)]
        require(len(group) == row.seeds and abs(group.best_cv_loss.mean() - row.mean_cv_loss) < 1e-7,
                "Aggregate mean or seed count mismatch")
        if len(group) > 1:
            require(abs(group.best_cv_loss.std() - row.std_cv_loss) < 1e-7, "Aggregate sample SD mismatch")
    selection = json.loads((root / "model_selection.json").read_text())
    candidates = agg[agg.split_method.eq("clustered")].sort_values("mean_cv_loss")
    require(selection["selected_split_method"] == "clustered" and
            selection["selected_variant_id"] == candidates.iloc[0].variant_id,
            "Selection must use clustered CV only")
    require(selection["mean_held_out_test_mae"] is None, "Selection exposes test")
    require(abs(selection["mean_cv_mae"] - candidates.iloc[0].mean_cv_loss) < 1e-7,
            "Selected CV score mismatch")
    from src.evaluation_statistics import paired_cluster_bootstrap_ci
    bootstrap = pd.read_csv(root / "paired_bootstrap_ci.csv")
    statistics_input = errors.rename(columns={"split_method": "split"})
    comparisons = [
        ("vienna_real_vs_sequence_only", "transformer_seq4"),
        ("vienna_real_vs_position_shuffled", "transformer_vienna_shuffled7"),
        ("vienna_real_vs_zero_channels", "transformer_vienna_zero7"),
    ]
    recomputed_bootstrap = []
    for name, reference in comparisons:
        if {reference, "transformer_vienna_real7"}.issubset(manifest["variants"]):
            recomputed_bootstrap.append(paired_cluster_bootstrap_ci(
                statistics_input, reference, "transformer_vienna_real7", comparison_id=name,
                n_bootstrap=manifest["bootstrap_replicates"], random_seed=17029,
            ))
    if recomputed_bootstrap:
        expected_bootstrap = pd.concat(recomputed_bootstrap, ignore_index=True)
        bootstrap_keys = ["seed", "split", "partition", "comparison_id", "metric"]
        complete_grid(bootstrap, bootstrap_keys, set(map(tuple, expected_bootstrap[bootstrap_keys].to_numpy())))
        check = bootstrap.merge(expected_bootstrap, on=bootstrap_keys,
                                validate="one_to_one", suffixes=("", "_expected"))
        for field in ("point_improvement", "ci_lower", "ci_upper", "probability_improvement"):
            require(np.allclose(check[field], check[field + "_expected"], atol=1e-12, rtol=0),
                    f"Bootstrap mismatch: {field}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive")
    print(json.dumps(validate_development_archive(parser.parse_args().archive), indent=2))
