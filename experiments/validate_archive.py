"""Validate a compact, versioned Kaggle result archive.

The validator intentionally uses only the Python standard library so it can run
in CI without loading model checkpoints or competition data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


REQUIRED_FILES = {
    "README.md",
    "run_manifest.json",
    "environment.json",
    "data_quality_report.json",
    "split_report.json",
    "model_selection.json",
    "ablation_results.csv",
    "ablation_summary.csv",
    "ablation_aggregate.csv",
    "test_results.csv",
    "leakage_comparison.csv",
    "leakage_aggregate.csv",
    "vienna_comparison.csv",
    "vienna_aggregate.csv",
    "error_analysis_per_sequence.csv",
    "error_by_length.csv",
    "error_by_experiment.csv",
    "error_by_structure.csv",
    "training_curves.png",
    "error_by_length.png",
}
SCHEMA_V2_FILES = {
    "sequence_length_distribution.csv",
    "sequence_length_histogram.png",
}
SCHEMA_V3_FILES = {
    "baseline_results.csv",
    "artifact_checksums.sha256",
}
FORBIDDEN_NAMES = {
    "train_data.csv",
    "error_analysis_per_profile.csv",
}
EXPECTED_VARIANTS = {
    "CNN Only",
    "CNN + Bi-LSTM",
    "CNN + LSTM + Transformer",
    "Full Model (+ViennaRNA 7d)",
}
TRANSFORMER = "CNN + LSTM + Transformer"
WITH_VIENNA = "Full Model (+ViennaRNA 7d)"


class ArchiveValidationError(ValueError):
    """Raised when a reportable result archive is internally inconsistent."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ArchiveValidationError(message)


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def record_key(row: dict[str, str]) -> tuple[int, str, str]:
    return int(row["seed"]), row["split_method"], row["variant"]


def assert_close(actual: float, expected: float, label: str, tolerance: float = 1e-12) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise ArchiveValidationError(
            f"{label}: expected {expected:.16g}, observed {actual:.16g}"
        )


def validate_artifact_checksums(archive: Path, present: set[str]) -> None:
    """Verify a complete checksum inventory of compact data and image artifacts."""
    checksum_path = archive / "artifact_checksums.sha256"
    expected_names = {
        name for name in present
        if Path(name).suffix.lower() in {".csv", ".json", ".png"}
        and name != "error_analysis_per_profile.csv"
    }
    observed = {}
    for line_number, line in enumerate(
        checksum_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        match = re.fullmatch(r"([0-9a-f]{64})  ([^/\\]+)", line)
        require(bool(match), f"Invalid checksum line {line_number}")
        digest, name = match.groups()
        require(name not in observed, f"Duplicate checksum entry: {name}")
        observed[name] = digest
    require(
        set(observed) == expected_names,
        "Checksum inventory does not exactly match compact CSV/JSON/PNG artifacts",
    )
    for name, expected_digest in observed.items():
        digest = hashlib.sha256((archive / name).read_bytes()).hexdigest()
        require(digest == expected_digest, f"Checksum mismatch: {name}")


def validate_baseline_results(path: Path, seeds: set[int]) -> None:
    """Check the training-only statistical baseline coverage and domains."""
    records = load_csv(path)
    expected_keys = {
        (seed, split) for seed in seeds for split in {"random", "grouped"}
    }
    observed_keys = {(int(row["seed"]), row["split_method"]) for row in records}
    require(observed_keys == expected_keys, "Baseline seed/split coverage is incomplete")
    require(len(records) == len(expected_keys), "Baseline rows are duplicated")
    for row in records:
        require(
            row["baseline"] == "Train-only experiment mean",
            "Unexpected baseline definition",
        )
        for column in ("cv_mae", "test_mae", "mean_2A3_MaP", "mean_DMS_MaP"):
            value = float(row[column])
            require(math.isfinite(value), f"Baseline {column} is non-finite")
            require(0.0 <= value <= 1.0, f"Baseline {column} is outside [0, 1]")
        for column in ("cv_valid_positions", "test_valid_positions"):
            require(int(row[column]) > 0, f"Baseline {column} is not positive")


def grouped_values(
    records: list[dict[str, str]], value_column: str
) -> dict[tuple[str, str], list[float]]:
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in records:
        values[(row["split_method"], row["variant"])].append(float(row[value_column]))
    return values


def reconcile_error_file(
    path: Path, expected_test: dict[tuple[int, str, str], float]
) -> float:
    totals: dict[tuple[int, str, str], list[float]] = defaultdict(lambda: [0.0, 0.0])
    for row in load_csv(path):
        key = record_key(row)
        totals[key][0] += float(row["absolute_error_sum"])
        totals[key][1] += float(row["valid_positions"])

    require(set(totals) == set(expected_test), f"{path.name}: evaluation keys do not match test_results.csv")
    max_difference = 0.0
    for key, (error_sum, positions) in totals.items():
        require(positions > 0, f"{path.name}: non-positive denominator for {key}")
        difference = abs(error_sum / positions - expected_test[key])
        max_difference = max(max_difference, difference)
    require(max_difference <= 1e-7, f"{path.name}: MAE reconciliation differs by {max_difference}")
    return max_difference


def validate_sequence_length_distribution(
    path: Path, manifest: dict, quality: dict
) -> None:
    records = load_csv(path)
    require(records, f"{path.name}: no length rows")
    required_columns = {
        "sequence_length",
        "unique_sequences",
        "experiment_profiles",
        "share_of_sequences",
        "cumulative_unique_sequences",
    }
    require(
        required_columns.issubset(records[0]),
        f"{path.name}: missing columns {sorted(required_columns - set(records[0]))}",
    )

    lengths = [int(row["sequence_length"]) for row in records]
    unique_counts = [int(row["unique_sequences"]) for row in records]
    profile_counts = [int(row["experiment_profiles"]) for row in records]
    require(
        lengths == list(range(lengths[0], lengths[-1] + 1)),
        f"{path.name}: lengths must cover a continuous 1-nt range",
    )
    require(
        all(value >= 0 for value in unique_counts),
        f"{path.name}: negative sequence count",
    )
    require(
        all(value >= 0 for value in profile_counts),
        f"{path.name}: negative profile count",
    )
    require(
        sum(unique_counts) == manifest["filtered_unique_sequences"],
        f"{path.name}: unique-sequence total disagrees with manifest",
    )
    require(
        sum(profile_counts) == manifest["filtered_rows"],
        f"{path.name}: experiment-profile total disagrees with manifest",
    )

    cumulative = 0
    expanded_lengths: list[int] = []
    for row, length, count in zip(records, lengths, unique_counts):
        cumulative += count
        assert_close(
            float(row["share_of_sequences"]),
            count / manifest["filtered_unique_sequences"],
            f"{path.name}: sequence share at {length}",
        )
        require(
            int(row["cumulative_unique_sequences"]) == cumulative,
            f"{path.name}: cumulative count mismatch at {length}",
        )
        expanded_lengths.extend([length] * count)

    require(
        lengths[0] == quality["sequence_length"]["min"],
        f"{path.name}: minimum length mismatch",
    )
    require(
        lengths[-1] == quality["sequence_length"]["max"],
        f"{path.name}: maximum length mismatch",
    )
    assert_close(
        statistics.median(expanded_lengths),
        quality["sequence_length"]["median"],
        f"{path.name}: median length",
    )


def validate_archive(archive: str | Path) -> dict[str, object]:
    archive = Path(archive)
    require(archive.is_dir(), f"Archive directory does not exist: {archive}")

    present = {path.name for path in archive.iterdir() if path.is_file()}
    missing = REQUIRED_FILES - present
    require(not missing, f"Missing required artifacts: {sorted(missing)}")
    empty = sorted(path.name for path in archive.iterdir() if path.is_file() and path.stat().st_size == 0)
    require(not empty, f"Empty artifacts: {empty}")
    forbidden = sorted(
        path.name
        for path in archive.iterdir()
        if path.is_file()
        and (path.name in FORBIDDEN_NAMES or path.suffix in {".pth", ".log"})
    )
    require(not forbidden, f"Non-compact artifacts must not be committed: {forbidden}")
    manifest = load_json(archive / "run_manifest.json")
    schema_version = int(manifest.get("artifact_schema_version", 1))
    if schema_version >= 2:
        missing_v2 = SCHEMA_V2_FILES - present
        require(not missing_v2, f"Missing schema v2 artifacts: {sorted(missing_v2)}")
    if schema_version >= 3:
        missing_v3 = SCHEMA_V3_FILES - present
        require(not missing_v3, f"Missing schema v3 artifacts: {sorted(missing_v3)}")
    if "artifact_checksums.sha256" in present:
        validate_artifact_checksums(archive, present)

    image_names = ["training_curves.png", "error_by_length.png"]
    if schema_version >= 2:
        image_names.append("sequence_length_histogram.png")
    for image_name in image_names:
        with (archive / image_name).open("rb") as handle:
            require(handle.read(8) == b"\x89PNG\r\n\x1a\n", f"Invalid PNG file: {image_name}")

    commit = str(manifest["git_commit"])
    require(bool(re.fullmatch(r"[0-9a-f]{40}", commit)), "Manifest git_commit must be a full SHA")
    archive_suffix = archive.name.rsplit("-", maxsplit=1)[-1]
    if re.fullmatch(r"[0-9a-f]{7,12}", archive_suffix):
        require(commit.startswith(archive_suffix), "Archive name and manifest commit disagree")

    seeds = {int(seed) for seed in manifest["validation_seeds"]}
    require(seeds == {42, 123, 2026}, f"Unexpected validation seeds: {sorted(seeds)}")
    require(manifest["filtered_unique_sequences"] == 1000, "Expected 1,000 retained sequences")
    require(manifest["filtered_rows"] == 1820, "Expected 1,820 retained profiles")
    require(manifest["split_method"] == "both", "Both split methods must be present")
    if schema_version >= 3:
        require(
            bool(re.fullmatch(r"[0-9a-f]{64}", manifest["source_file_sha256"])),
            "Source file SHA-256 is missing or invalid",
        )
        require(manifest["source_file_size_bytes"] > 0, "Source file size is invalid")
        require(
            bool(re.fullmatch(r"[0-9a-f]{64}", manifest["cohort_data_sha256"])),
            "Cohort data SHA-256 is missing or invalid",
        )
        require(
            len(manifest["cohort_data_sha256_columns"]) > 200,
            "Cohort hash does not cover the reactivity targets",
        )

    environment = load_json(archive / "environment.json")
    require(environment["cuda_available"] is True, "Archived run did not use CUDA")
    require(environment["gpu_count"] == 1, "Reportable run must use one GPU")
    require(len(environment["gpu_names"]) == 1, "Expected one recorded GPU name")
    require("P100" in environment["gpu_names"][0], "Reportable archive is not the P100 run")
    require(environment["data_parallel"] is False, "Single-GPU run should not use DataParallel")
    if schema_version >= 3:
        dependencies = environment.get("dependencies", {})
        for distribution in (
            "torch", "pandas", "numpy", "matplotlib", "PyYAML", "ViennaRNA"
        ):
            require(dependencies.get(distribution), f"Missing dependency version: {distribution}")
        require(
            dependencies["ViennaRNA"] == "2.7.2",
            "Reportable run must use pinned ViennaRNA 2.7.2",
        )

    quality = load_json(archive / "data_quality_report.json")
    require(quality["rows"] == manifest["filtered_rows"], "Quality and manifest row counts disagree")
    require(
        quality["unique_sequences"] == manifest["filtered_unique_sequences"],
        "Quality and manifest sequence counts disagree",
    )
    require(quality["duplicate_full_rows"] == 0, "Exact duplicate rows are present")
    require(quality["valid_reactivity_targets"] > 0, "No measured reactivity targets were recorded")
    if schema_version >= 2:
        validate_sequence_length_distribution(
            archive / "sequence_length_distribution.csv", manifest, quality
        )
    if schema_version >= 3:
        validate_baseline_results(archive / "baseline_results.csv", seeds)

    summary = load_csv(archive / "ablation_summary.csv")
    tests = load_csv(archive / "test_results.csv")
    require(len(summary) == 24, f"Expected 24 summary rows, found {len(summary)}")
    require(len(tests) == 24, f"Expected 24 test rows, found {len(tests)}")
    expected_keys = {
        (seed, split, variant)
        for seed in seeds
        for split in {"random", "grouped"}
        for variant in EXPECTED_VARIANTS
    }
    summary_by_key = {record_key(row): row for row in summary}
    test_by_key = {record_key(row): float(row["test_mae"]) for row in tests}
    require(set(summary_by_key) == expected_keys, "Ablation summary is missing seed/split/model combinations")
    require(set(test_by_key) == expected_keys, "Test results are missing seed/split/model combinations")
    for key in expected_keys:
        assert_close(float(summary_by_key[key]["test_mae"]), test_by_key[key], f"test score {key}")

    epoch_records = load_csv(archive / "ablation_results.csv")
    epochs_by_key: dict[tuple[int, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in epoch_records:
        epochs_by_key[record_key(row)].append(row)
    require(set(epochs_by_key) == expected_keys, "Per-epoch results are missing evaluation groups")
    for key, records in epochs_by_key.items():
        records.sort(key=lambda row: int(row["epoch"]))
        require(len(records) == manifest["epochs"], f"Unexpected epoch count for {key}")
        require(
            [int(row["epoch"]) for row in records] == list(range(1, manifest["epochs"] + 1)),
            f"Epoch sequence is incomplete for {key}",
        )
        best = min(records, key=lambda row: float(row["cv_loss"]))
        assert_close(float(best["cv_loss"]), float(summary_by_key[key]["best_cv_loss"]), f"best CV {key}")
        require(int(best["epoch"]) == int(summary_by_key[key]["best_epoch"]), f"Best epoch mismatch for {key}")
        assert_close(
            float(records[-1]["train_loss"]),
            float(summary_by_key[key]["final_train_loss"]),
            f"final train loss {key}",
        )

    cv_groups = grouped_values(summary, "best_cv_loss")
    test_groups = grouped_values(summary, "test_mae")
    aggregate = load_csv(archive / "ablation_aggregate.csv")
    require(len(aggregate) == 8, f"Expected eight aggregate rows, found {len(aggregate)}")
    aggregate_by_key = {(row["split_method"], row["variant"]): row for row in aggregate}
    require(set(aggregate_by_key) == set(cv_groups), "Aggregate model groups do not match per-seed results")
    for key, cv_values in cv_groups.items():
        row = aggregate_by_key[key]
        assert_close(float(row["mean_cv_loss"]), statistics.mean(cv_values), f"mean CV {key}")
        assert_close(float(row["std_cv_loss"]), statistics.stdev(cv_values), f"CV std {key}")
        assert_close(float(row["mean_test_mae"]), statistics.mean(test_groups[key]), f"mean test {key}")
        assert_close(float(row["std_test_mae"]), statistics.stdev(test_groups[key]), f"test std {key}")
        require(int(row["seeds"]) == len(seeds), f"Seed count mismatch for {key}")

    grouped_rows = [row for row in aggregate if row["split_method"] == "grouped"]
    selected_aggregate = min(grouped_rows, key=lambda row: float(row["mean_cv_loss"]))
    selection = load_json(archive / "model_selection.json")
    require(selection["selected_split_method"] == "grouped", "Model selection must use grouped CV")
    require(selection["selected_variant"] == selected_aggregate["variant"], "Selected model is not grouped-CV minimum")
    assert_close(selection["mean_cv_mae"], float(selected_aggregate["mean_cv_loss"]), "selected CV mean")
    assert_close(selection["std_cv_mae"], float(selected_aggregate["std_cv_loss"]), "selected CV std")
    assert_close(
        selection["mean_held_out_test_mae"],
        float(selected_aggregate["mean_test_mae"]),
        "selected test mean",
    )
    assert_close(
        selection["std_held_out_test_mae"],
        float(selected_aggregate["std_test_mae"]),
        "selected test std",
    )

    split_report = load_json(archive / "split_report.json")
    require(split_report["filtered_rows"] == manifest["filtered_rows"], "Split and manifest rows disagree")
    require(
        split_report["filtered_unique_sequences"] == manifest["filtered_unique_sequences"],
        "Split and manifest sequence counts disagree",
    )
    grouped_overlaps: list[int] = []
    random_overlaps: list[int] = []
    for seed in seeds:
        methods = split_report["splits_by_seed"][str(seed)]
        grouped = methods["grouped"]
        random = methods["random"]
        grouped_overlaps.extend(int(value) for value in grouped["sequence_overlap"].values())
        random_overlaps.extend(int(value) for value in random["sequence_overlap"].values())
        require(sum(grouped["rows"].values()) == manifest["filtered_rows"], f"Grouped rows incomplete for seed {seed}")
        require(sum(random["rows"].values()) == manifest["filtered_rows"], f"Random rows incomplete for seed {seed}")
        require(
            grouped["unique_sequences"] == {"train": 700, "cv": 150, "test": 150},
            f"Grouped sequence counts are not 700/150/150 for seed {seed}",
        )
    require(max(grouped_overlaps) == 0, "Grouped split contains exact-sequence overlap")
    require(min(random_overlaps) > 0, "Random split unexpectedly has no exact-sequence overlap")

    parameters = defaultdict(set)
    for row in summary:
        parameters[row["variant"]].add(int(row["params"]))
    require(all(len(values) == 1 for values in parameters.values()), "Parameter count changes across seeds/splits")
    require(
        next(iter(parameters[WITH_VIENNA])) - next(iter(parameters[TRANSFORMER])) == 960,
        "ViennaRNA comparison does not isolate the expected input-feature parameter change",
    )

    leakage_rows = load_csv(archive / "leakage_comparison.csv")
    require(len(leakage_rows) == len(seeds) * len(EXPECTED_VARIANTS), "Leakage comparison row count is incomplete")
    leakage_differences: dict[str, list[float]] = defaultdict(list)
    for row in leakage_rows:
        seed = int(row["seed"])
        variant = row["variant"]
        random_cv = float(summary_by_key[(seed, "random", variant)]["best_cv_loss"])
        grouped_cv = float(summary_by_key[(seed, "grouped", variant)]["best_cv_loss"])
        assert_close(float(row["random"]), random_cv, f"leakage random {seed, variant}")
        assert_close(float(row["grouped"]), grouped_cv, f"leakage grouped {seed, variant}")
        difference = grouped_cv - random_cv
        assert_close(float(row["grouped_minus_random"]), difference, f"leakage delta {seed, variant}")
        leakage_differences[variant].append(difference)

    leakage_aggregate = {row["variant"]: row for row in load_csv(archive / "leakage_aggregate.csv")}
    require(set(leakage_aggregate) == EXPECTED_VARIANTS, "Leakage aggregate variants are incomplete")
    for variant, differences in leakage_differences.items():
        row = leakage_aggregate[variant]
        random_values = cv_groups[("random", variant)]
        grouped_cv_values = cv_groups[("grouped", variant)]
        assert_close(float(row["mean_random"]), statistics.mean(random_values), f"leakage mean random {variant}")
        assert_close(float(row["mean_grouped"]), statistics.mean(grouped_cv_values), f"leakage mean grouped {variant}")
        assert_close(
            float(row["mean_grouped_minus_random"]),
            statistics.mean(differences),
            f"leakage mean delta {variant}",
        )
        assert_close(
            float(row["std_grouped_minus_random"]),
            statistics.stdev(differences),
            f"leakage delta std {variant}",
        )

    vienna_rows = load_csv(archive / "vienna_comparison.csv")
    require(len(vienna_rows) == len(seeds) * 2, "ViennaRNA comparison row count is incomplete")
    vienna_by_split: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    for row in vienna_rows:
        seed = int(row["seed"])
        split = row["split_method"]
        transformer = float(summary_by_key[(seed, split, TRANSFORMER)]["best_cv_loss"])
        with_vienna = float(summary_by_key[(seed, split, WITH_VIENNA)]["best_cv_loss"])
        improvement = transformer - with_vienna
        relative = improvement / transformer
        assert_close(float(row["transformer"]), transformer, f"Vienna transformer {seed, split}")
        assert_close(float(row["with_vienna"]), with_vienna, f"Vienna model {seed, split}")
        assert_close(float(row["absolute_improvement"]), improvement, f"Vienna delta {seed, split}")
        assert_close(float(row["relative_improvement"]), relative, f"Vienna relative {seed, split}")
        vienna_by_split[split].append((transformer, with_vienna, improvement, relative))

    vienna_aggregate = {
        row["split_method"]: row for row in load_csv(archive / "vienna_aggregate.csv")
    }
    require(set(vienna_aggregate) == {"random", "grouped"}, "ViennaRNA aggregate splits are incomplete")
    for split, values in vienna_by_split.items():
        row = vienna_aggregate[split]
        transformer_values, vienna_values, improvements, relative_values = map(list, zip(*values))
        assert_close(float(row["mean_transformer"]), statistics.mean(transformer_values), f"Vienna mean transformer {split}")
        assert_close(float(row["mean_with_vienna"]), statistics.mean(vienna_values), f"Vienna mean model {split}")
        assert_close(float(row["mean_absolute_improvement"]), statistics.mean(improvements), f"Vienna mean delta {split}")
        assert_close(float(row["std_absolute_improvement"]), statistics.stdev(improvements), f"Vienna delta std {split}")
        assert_close(float(row["mean_relative_improvement"]), statistics.mean(relative_values), f"Vienna mean relative {split}")

    max_error_difference = 0.0
    for name in (
        "error_analysis_per_sequence.csv",
        "error_by_length.csv",
        "error_by_experiment.csv",
        "error_by_structure.csv",
    ):
        max_error_difference = max(
            max_error_difference,
            reconcile_error_file(archive / name, test_by_key),
        )

    return {
        "archive": str(archive),
        "git_commit": commit,
        "evaluations": len(tests),
        "grouped_overlap_max": max(grouped_overlaps),
        "random_overlap_min": min(random_overlaps),
        "selected_variant": selection["selected_variant"],
        "selected_grouped_cv_mae": selection["mean_cv_mae"],
        "selected_held_out_test_mae": selection["mean_held_out_test_mae"],
        "max_error_reconciliation_difference": max_error_difference,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Versioned result archive directory")
    args = parser.parse_args()
    report = validate_archive(args.archive)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
