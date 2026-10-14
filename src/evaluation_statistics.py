"""Statistical summaries for RNA reactivity evaluation.

The functions in this module operate on error numerators and denominators rather
than model predictions.  This keeps the statistical layer independent of
PyTorch and makes every reported metric exactly reconcilable with the archived
evaluation errors.

The expected input grain may be one or more rows per RNA sequence (for example,
one row per experimental profile).  ``summarize_sequence_errors`` first pools
those rows within a sequence.  ``summarize_evaluation_metrics`` then reports:

* nucleotide-weighted MAE, which gives every measured nucleotide equal weight;
* macro-sequence MAE, which gives every exact RNA sequence equal weight.

``paired_cluster_bootstrap_ci`` compares two variants evaluated on the same
sequences.  It resamples whole similarity clusters, never individual profiles or
nucleotides.  Positive improvements mean that the candidate has lower MAE than
the reference (reference MAE minus candidate MAE).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

import numpy as np
import pandas as pd


ERROR_COLUMNS = (
    "variant_id",
    "seed",
    "split",
    "partition",
    "sequence_hash",
    "cluster_id",
    "absolute_error_sum",
    "valid_positions",
)
EVALUATION_GROUP_COLUMNS = ("variant_id", "seed", "split", "partition")
SEQUENCE_GROUP_COLUMNS = EVALUATION_GROUP_COLUMNS + (
    "sequence_hash",
    "cluster_id",
)
PAIR_COLUMNS = ("seed", "split", "partition", "sequence_hash", "cluster_id")
BOOTSTRAP_GROUP_COLUMNS = ("seed", "split", "partition")


class EvaluationStatisticsError(ValueError):
    """Raised when evaluation errors cannot support a valid comparison."""


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise EvaluationStatisticsError(f"Missing required columns: {missing}")


def _validated_errors(errors: pd.DataFrame) -> pd.DataFrame:
    """Return a validated copy of an evaluation-error table."""
    if not isinstance(errors, pd.DataFrame):
        raise TypeError("errors must be a pandas DataFrame")
    _require_columns(errors, ERROR_COLUMNS)
    if errors.empty:
        raise EvaluationStatisticsError("Evaluation errors must not be empty")

    frame = errors.loc[:, list(ERROR_COLUMNS)].copy()
    identifier_columns = (
        "variant_id",
        "seed",
        "split",
        "partition",
        "sequence_hash",
        "cluster_id",
    )
    null_columns = [column for column in identifier_columns if frame[column].isna().any()]
    if null_columns:
        raise EvaluationStatisticsError(
            f"Identifier columns contain missing values: {null_columns}"
        )

    for column in ("absolute_error_sum", "valid_positions"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        values = frame[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise EvaluationStatisticsError(f"{column} must contain only finite values")

    error_values = frame["absolute_error_sum"].to_numpy(dtype=float)
    position_values = frame["valid_positions"].to_numpy(dtype=float)
    if (error_values < 0.0).any():
        raise EvaluationStatisticsError("absolute_error_sum must be non-negative")
    if (position_values <= 0.0).any():
        raise EvaluationStatisticsError("valid_positions must be positive")
    if not np.equal(position_values, np.floor(position_values)).all():
        raise EvaluationStatisticsError("valid_positions must contain integer counts")
    if (error_values > position_values + 1e-8).any():
        raise EvaluationStatisticsError(
            "absolute_error_sum cannot exceed valid_positions for clipped MAE"
        )

    # Strings make sorting, joining, and deterministic seed derivation stable
    # even when upstream cluster identifiers are numeric.
    for column in ("variant_id", "split", "partition", "sequence_hash", "cluster_id"):
        frame[column] = frame[column].astype(str)
        if frame[column].str.len().eq(0).any():
            raise EvaluationStatisticsError(f"{column} must not contain empty strings")

    return frame


def summarize_sequence_errors(errors: pd.DataFrame) -> pd.DataFrame:
    """Pool profile-level errors within each exact RNA sequence.

    A sequence may have multiple experimental profiles.  Their error sums and
    valid-position counts are added before sequence MAE is calculated, so a
    sequence with many profiles still contributes only one value to the later
    macro-sequence mean.
    """
    frame = _validated_errors(errors)

    # A sequence must not silently move between similarity clusters within the
    # same evaluation.  If this invariant fails, grouping by cluster_id would
    # split one biological sequence into multiple bootstrap observations.
    cluster_counts = frame.groupby(
        list(EVALUATION_GROUP_COLUMNS) + ["sequence_hash"],
        sort=False,
    )["cluster_id"].nunique()
    if (cluster_counts != 1).any():
        raise EvaluationStatisticsError(
            "A sequence maps to multiple cluster_id values within one evaluation"
        )

    sequence_errors = (
        frame.groupby(list(SEQUENCE_GROUP_COLUMNS), as_index=False, sort=True)
        .agg(
            profiles=("absolute_error_sum", "size"),
            absolute_error_sum=("absolute_error_sum", "sum"),
            valid_positions=("valid_positions", "sum"),
        )
    )
    sequence_errors["sequence_mae"] = (
        sequence_errors["absolute_error_sum"]
        / sequence_errors["valid_positions"]
    )
    return sequence_errors


def summarize_evaluation_metrics(errors: pd.DataFrame) -> pd.DataFrame:
    """Report nucleotide-weighted and macro-sequence MAE per evaluation."""
    sequence_errors = summarize_sequence_errors(errors)
    metrics = (
        sequence_errors.groupby(list(EVALUATION_GROUP_COLUMNS), as_index=False, sort=True)
        .agg(
            sequences=("sequence_hash", "nunique"),
            clusters=("cluster_id", "nunique"),
            experiment_profiles=("profiles", "sum"),
            absolute_error_sum=("absolute_error_sum", "sum"),
            valid_positions=("valid_positions", "sum"),
            macro_sequence_mae=("sequence_mae", "mean"),
        )
    )
    metrics["nucleotide_weighted_mae"] = (
        metrics["absolute_error_sum"] / metrics["valid_positions"]
    )
    ordered_columns = list(EVALUATION_GROUP_COLUMNS) + [
        "sequences",
        "clusters",
        "experiment_profiles",
        "valid_positions",
        "absolute_error_sum",
        "nucleotide_weighted_mae",
        "macro_sequence_mae",
    ]
    return metrics.loc[:, ordered_columns]


def _group_seed(base_seed: int, values: tuple[object, ...]) -> int:
    """Derive a stable RNG seed that is independent of other result groups."""
    payload = "\x1f".join([str(int(base_seed)), *(str(value) for value in values)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _paired_sequence_errors(
    sequence_errors: pd.DataFrame,
    reference_variant: str,
    candidate_variant: str,
) -> pd.DataFrame:
    """Return strictly paired reference/candidate sequence errors."""
    reference_variant = str(reference_variant)
    candidate_variant = str(candidate_variant)
    if reference_variant == candidate_variant:
        raise EvaluationStatisticsError(
            "reference_variant and candidate_variant must be different"
        )

    observed = set(sequence_errors["variant_id"])
    absent = sorted({reference_variant, candidate_variant} - observed)
    if absent:
        raise EvaluationStatisticsError(f"Variants are absent from errors: {absent}")

    value_columns = ["absolute_error_sum", "valid_positions", "sequence_mae"]
    reference = sequence_errors[
        sequence_errors["variant_id"] == reference_variant
    ].loc[:, list(PAIR_COLUMNS) + value_columns]
    candidate = sequence_errors[
        sequence_errors["variant_id"] == candidate_variant
    ].loc[:, list(PAIR_COLUMNS) + value_columns]

    paired = reference.merge(
        candidate,
        on=list(PAIR_COLUMNS),
        how="outer",
        suffixes=("_reference", "_candidate"),
        validate="one_to_one",
        indicator=True,
    )
    if not paired["_merge"].eq("both").all():
        missing_reference = int(paired["_merge"].eq("right_only").sum())
        missing_candidate = int(paired["_merge"].eq("left_only").sum())
        raise EvaluationStatisticsError(
            "Variants are not exactly paired on seed/split/partition/sequence/cluster "
            f"(missing reference={missing_reference}, missing candidate={missing_candidate})"
        )
    paired = paired.drop(columns="_merge")

    reference_positions = paired["valid_positions_reference"].to_numpy(dtype=float)
    candidate_positions = paired["valid_positions_candidate"].to_numpy(dtype=float)
    if not np.array_equal(reference_positions, candidate_positions):
        raise EvaluationStatisticsError(
            "Paired variants have different valid_positions for the same sequence"
        )
    return paired.sort_values(list(PAIR_COLUMNS), kind="mergesort").reset_index(drop=True)


def _bootstrap_group(
    group: pd.DataFrame,
    *,
    reference_variant: str,
    candidate_variant: str,
    comparison_id: str,
    group_values: tuple[object, ...],
    n_bootstrap: int,
    confidence_level: float,
    random_seed: int,
    batch_size: int,
) -> list[dict[str, object]]:
    """Bootstrap one seed/split/partition group in bounded memory."""
    cluster_summary = (
        group.groupby("cluster_id", as_index=False, sort=True)
        .agg(
            sequences=("sequence_hash", "size"),
            valid_positions=("valid_positions_reference", "sum"),
            reference_error=("absolute_error_sum_reference", "sum"),
            candidate_error=("absolute_error_sum_candidate", "sum"),
            reference_sequence_mae_sum=("sequence_mae_reference", "sum"),
            candidate_sequence_mae_sum=("sequence_mae_candidate", "sum"),
        )
    )
    n_clusters = len(cluster_summary)
    n_sequences = int(cluster_summary["sequences"].sum())
    if n_clusters == 0 or n_sequences == 0:
        raise EvaluationStatisticsError("A bootstrap group contains no paired sequences")

    valid = cluster_summary["valid_positions"].to_numpy(dtype=float)
    reference_error = cluster_summary["reference_error"].to_numpy(dtype=float)
    candidate_error = cluster_summary["candidate_error"].to_numpy(dtype=float)
    sequence_counts = cluster_summary["sequences"].to_numpy(dtype=float)
    reference_macro_numerator = cluster_summary[
        "reference_sequence_mae_sum"
    ].to_numpy(dtype=float)
    candidate_macro_numerator = cluster_summary[
        "candidate_sequence_mae_sum"
    ].to_numpy(dtype=float)

    point_weighted = (
        reference_error.sum() / valid.sum()
        - candidate_error.sum() / valid.sum()
    )
    point_macro = (
        reference_macro_numerator.sum() / sequence_counts.sum()
        - candidate_macro_numerator.sum() / sequence_counts.sum()
    )

    rng = np.random.default_rng(_group_seed(random_seed, group_values))
    weighted_samples = np.empty(n_bootstrap, dtype=np.float64)
    macro_samples = np.empty(n_bootstrap, dtype=np.float64)
    for start in range(0, n_bootstrap, batch_size):
        stop = min(start + batch_size, n_bootstrap)
        draws = rng.integers(0, n_clusters, size=(stop - start, n_clusters))

        sampled_valid = valid[draws].sum(axis=1)
        weighted_samples[start:stop] = (
            reference_error[draws].sum(axis=1) / sampled_valid
            - candidate_error[draws].sum(axis=1) / sampled_valid
        )

        sampled_sequences = sequence_counts[draws].sum(axis=1)
        macro_samples[start:stop] = (
            reference_macro_numerator[draws].sum(axis=1) / sampled_sequences
            - candidate_macro_numerator[draws].sum(axis=1) / sampled_sequences
        )

    alpha = (1.0 - confidence_level) / 2.0
    seed, split, partition = group_values
    common = {
        "comparison_id": comparison_id,
        "reference_variant": reference_variant,
        "candidate_variant": candidate_variant,
        "seed": seed,
        "split": split,
        "partition": partition,
        "confidence_level": confidence_level,
        "bootstrap_replicates": n_bootstrap,
        "bootstrap_seed": int(random_seed),
        "bootstrap_unit": "cluster_id",
        "clusters": n_clusters,
        "sequences": n_sequences,
        "improvement_definition": "reference_mae_minus_candidate_mae",
    }
    records = []
    for metric, point, samples in (
        ("nucleotide_weighted_mae", point_weighted, weighted_samples),
        ("macro_sequence_mae", point_macro, macro_samples),
    ):
        records.append({
            **common,
            "metric": metric,
            "point_improvement": float(point),
            "ci_lower": float(np.quantile(samples, alpha)),
            "ci_upper": float(np.quantile(samples, 1.0 - alpha)),
            "probability_improvement": float(np.mean(samples > 0.0)),
        })
    return records


def paired_cluster_bootstrap_ci(
    errors: pd.DataFrame,
    reference_variant: str,
    candidate_variant: str,
    *,
    comparison_id: str | None = None,
    n_bootstrap: int = 10_000,
    confidence_level: float = 0.95,
    random_seed: int = 17_029,
    batch_size: int = 256,
) -> pd.DataFrame:
    """Return deterministic paired cluster-bootstrap confidence intervals.

    Confidence intervals are produced separately for each
    ``seed``/``split``/``partition`` evaluation.  Within an evaluation, whole
    ``cluster_id`` groups are sampled with replacement and all sequences in a
    sampled cluster travel together.  This avoids treating correlated profiles,
    nucleotides, or homologous sequences as independent observations.

    The result contains one row for nucleotide-weighted MAE and one for
    macro-sequence MAE.  A positive ``point_improvement`` means that the
    candidate has lower error than the reference.
    """
    if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, (int, np.integer)):
        raise TypeError("n_bootstrap must be an integer")
    if int(n_bootstrap) <= 0:
        raise EvaluationStatisticsError("n_bootstrap must be positive")
    if not 0.0 < float(confidence_level) < 1.0:
        raise EvaluationStatisticsError("confidence_level must be between 0 and 1")
    if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)):
        raise TypeError("batch_size must be an integer")
    if int(batch_size) <= 0:
        raise EvaluationStatisticsError("batch_size must be positive")

    reference_variant = str(reference_variant)
    candidate_variant = str(candidate_variant)
    comparison_id = (
        str(comparison_id)
        if comparison_id is not None
        else f"{candidate_variant}_vs_{reference_variant}"
    )
    if not comparison_id:
        raise EvaluationStatisticsError("comparison_id must not be empty")

    sequence_errors = summarize_sequence_errors(errors)
    paired = _paired_sequence_errors(
        sequence_errors,
        reference_variant=reference_variant,
        candidate_variant=candidate_variant,
    )

    records: list[dict[str, object]] = []
    grouped = paired.groupby(list(BOOTSTRAP_GROUP_COLUMNS), sort=True)
    for group_values, group in grouped:
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        records.extend(
            _bootstrap_group(
                group,
                reference_variant=reference_variant,
                candidate_variant=candidate_variant,
                comparison_id=comparison_id,
                group_values=group_values,
                n_bootstrap=int(n_bootstrap),
                confidence_level=float(confidence_level),
                random_seed=int(random_seed),
                batch_size=int(batch_size),
            )
        )

    return pd.DataFrame.from_records(records).sort_values(
        ["seed", "split", "partition", "metric"], kind="mergesort"
    ).reset_index(drop=True)


__all__ = [
    "EvaluationStatisticsError",
    "paired_cluster_bootstrap_ci",
    "summarize_evaluation_metrics",
    "summarize_sequence_errors",
]
