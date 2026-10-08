"""Tests for sequence-level metrics and paired cluster bootstrap intervals."""

import pandas as pd
import pytest

from src.evaluation_statistics import (
    EvaluationStatisticsError,
    paired_cluster_bootstrap_ci,
    summarize_evaluation_metrics,
    summarize_sequence_errors,
)


def _row(
    variant_id,
    sequence_hash,
    cluster_id,
    absolute_error_sum,
    valid_positions,
    *,
    seed=42,
    split="similarity_grouped",
    partition="test",
):
    return {
        "variant_id": variant_id,
        "seed": seed,
        "split": split,
        "partition": partition,
        "sequence_hash": sequence_hash,
        "cluster_id": cluster_id,
        "absolute_error_sum": absolute_error_sum,
        "valid_positions": valid_positions,
    }


def test_sequence_summary_pools_profiles_before_calculating_mae():
    errors = pd.DataFrame([
        _row("model", "seq-a", "cluster-a", 1.0, 2),
        _row("model", "seq-a", "cluster-a", 1.0, 2),
        _row("model", "seq-b", "cluster-b", 0.1, 1),
    ])

    sequence_errors = summarize_sequence_errors(errors)
    metrics = summarize_evaluation_metrics(errors).iloc[0]

    assert len(sequence_errors) == 2
    seq_a = sequence_errors.set_index("sequence_hash").loc["seq-a"]
    assert seq_a["profiles"] == 2
    assert seq_a["absolute_error_sum"] == pytest.approx(2.0)
    assert seq_a["valid_positions"] == 4
    assert seq_a["sequence_mae"] == pytest.approx(0.5)

    # Nucleotide weighted: (2.0 + 0.1) / (4 + 1) = 0.42.
    assert metrics["nucleotide_weighted_mae"] == pytest.approx(0.42)
    # Macro sequence: mean(0.5, 0.1) = 0.30, not a profile-level mean.
    assert metrics["macro_sequence_mae"] == pytest.approx(0.30)
    assert metrics["sequences"] == 2
    assert metrics["clusters"] == 2
    assert metrics["experiment_profiles"] == 3


def test_sequence_summary_does_not_mutate_input():
    errors = pd.DataFrame([
        _row("model", "seq-a", 7, 0.4, 2),
    ])
    original = errors.copy(deep=True)

    summarize_sequence_errors(errors)

    pd.testing.assert_frame_equal(errors, original)


@pytest.mark.parametrize(
    "column,value,message",
    [
        ("absolute_error_sum", -0.1, "non-negative"),
        ("absolute_error_sum", 3.0, "cannot exceed"),
        ("valid_positions", 0, "must be positive"),
        ("valid_positions", 1.5, "integer counts"),
        ("cluster_id", None, "missing values"),
    ],
)
def test_invalid_error_rows_fail_fast(column, value, message):
    row = _row("model", "seq-a", "cluster-a", 0.2, 2)
    row[column] = value
    with pytest.raises(EvaluationStatisticsError, match=message):
        summarize_sequence_errors(pd.DataFrame([row]))


def test_sequence_cannot_map_to_multiple_clusters_within_evaluation():
    errors = pd.DataFrame([
        _row("model", "seq-a", "cluster-a", 0.2, 2),
        _row("model", "seq-a", "cluster-b", 0.3, 2),
    ])

    with pytest.raises(EvaluationStatisticsError, match="multiple cluster_id"):
        summarize_sequence_errors(errors)


def _constant_improvement_errors():
    records = []
    for sequence, cluster, reference_error in (
        ("seq-a", "cluster-1", 4.0),
        ("seq-b", "cluster-1", 5.0),
        ("seq-c", "cluster-2", 6.0),
        ("seq-d", "cluster-3", 7.0),
    ):
        records.append(_row("reference", sequence, cluster, reference_error, 10))
        records.append(_row("candidate", sequence, cluster, reference_error - 1.0, 10))
    return pd.DataFrame(records)


def test_constant_paired_improvement_has_exact_bootstrap_interval():
    result = paired_cluster_bootstrap_ci(
        _constant_improvement_errors(),
        "reference",
        "candidate",
        comparison_id="candidate_vs_reference",
        n_bootstrap=500,
        random_seed=123,
        batch_size=31,
    )

    assert set(result["metric"]) == {
        "nucleotide_weighted_mae",
        "macro_sequence_mae",
    }
    assert result["point_improvement"].tolist() == pytest.approx([0.1, 0.1])
    assert result["ci_lower"].tolist() == pytest.approx([0.1, 0.1])
    assert result["ci_upper"].tolist() == pytest.approx([0.1, 0.1])
    assert result["probability_improvement"].tolist() == [1.0, 1.0]
    assert result["clusters"].tolist() == [3, 3]
    assert result["sequences"].tolist() == [4, 4]
    assert result["improvement_definition"].eq(
        "reference_mae_minus_candidate_mae"
    ).all()


def test_bootstrap_is_reproducible_and_independent_of_batch_size():
    errors = _constant_improvement_errors()
    # Introduce heterogeneous paired effects so the RNG stream matters.
    candidate_rows = errors["variant_id"].eq("candidate")
    errors.loc[candidate_rows, "absolute_error_sum"] = [4.0, 3.0, 7.0, 4.0]

    first = paired_cluster_bootstrap_ci(
        errors, "reference", "candidate", n_bootstrap=701,
        random_seed=9876, batch_size=37,
    )
    second = paired_cluster_bootstrap_ci(
        errors, "reference", "candidate", n_bootstrap=701,
        random_seed=9876, batch_size=113,
    )

    pd.testing.assert_frame_equal(first, second)


def test_identical_models_have_zero_improvement_and_interval():
    errors = _constant_improvement_errors()
    reference_errors = errors.loc[
        errors["variant_id"].eq("reference"), "absolute_error_sum"
    ].to_numpy()
    errors.loc[
        errors["variant_id"].eq("candidate"), "absolute_error_sum"
    ] = reference_errors

    result = paired_cluster_bootstrap_ci(
        errors, "reference", "candidate", n_bootstrap=100, random_seed=1
    )

    for column in ("point_improvement", "ci_lower", "ci_upper"):
        assert result[column].tolist() == pytest.approx([0.0, 0.0])
    assert result["probability_improvement"].tolist() == [0.0, 0.0]


def test_bootstrap_outputs_each_seed_split_and_partition_separately():
    first = _constant_improvement_errors()
    second = first.copy()
    second["seed"] = 123
    second["partition"] = "cv"

    result = paired_cluster_bootstrap_ci(
        pd.concat([first, second], ignore_index=True),
        "reference",
        "candidate",
        n_bootstrap=50,
    )

    assert len(result) == 4
    assert set(zip(result["seed"], result["partition"])) == {
        (42, "test"),
        (123, "cv"),
    }


def test_bootstrap_rejects_missing_variant_pair():
    errors = _constant_improvement_errors()
    errors = errors[
        ~(
            errors["variant_id"].eq("candidate")
            & errors["sequence_hash"].eq("seq-d")
        )
    ]

    with pytest.raises(EvaluationStatisticsError, match="not exactly paired"):
        paired_cluster_bootstrap_ci(
            errors, "reference", "candidate", n_bootstrap=10
        )


def test_bootstrap_rejects_different_valid_position_denominators():
    errors = _constant_improvement_errors()
    mask = (
        errors["variant_id"].eq("candidate")
        & errors["sequence_hash"].eq("seq-a")
    )
    errors.loc[mask, "valid_positions"] = 11

    with pytest.raises(EvaluationStatisticsError, match="different valid_positions"):
        paired_cluster_bootstrap_ci(
            errors, "reference", "candidate", n_bootstrap=10
        )


def test_bootstrap_validates_configuration():
    errors = _constant_improvement_errors()
    with pytest.raises(EvaluationStatisticsError, match="n_bootstrap must be positive"):
        paired_cluster_bootstrap_ci(
            errors, "reference", "candidate", n_bootstrap=0
        )
    with pytest.raises(EvaluationStatisticsError, match="between 0 and 1"):
        paired_cluster_bootstrap_ci(
            errors, "reference", "candidate", confidence_level=1.0
        )
    with pytest.raises(EvaluationStatisticsError, match="batch_size must be positive"):
        paired_cluster_bootstrap_ci(
            errors, "reference", "candidate", batch_size=0
        )
