"""Independently validate a compact schema-v4 CV development archive."""

import argparse
import json
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
    return {
        "validated_rows": len(expected), "unique_sequences": manifest["filtered_unique_sequences"],
        "test_rows": 0, "similarity_repairs": len(similarity.get("similarity_repairs", [])),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive")
    print(json.dumps(validate_development_archive(parser.parse_args().archive), indent=2))
