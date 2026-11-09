# V18: current-data exploratory convergence evidence

Kernel: `tooru4777/rna-reactivity-grouped-ablation`, API version 18, COMPLETE.
Training commit: `df2d87bd98ce8ad5b4e27766a98ad73cd5085ada`.
Retrieved and reviewed on 2026-10-22 (Asia/Taipei).

This archive preserves all 26 original checksum-listed compact artifacts byte
for byte. No source data, profile-level outputs or checkpoints are committed.
The full downloaded outputs and logs are retained in the review task's work
folder. The API metadata reports version 18 and lastRunTime
2026-10-19T08:22:58.631Z; output listing timestamps are not used to infer when
training began. The log's first event is at 10248.26 elapsed seconds, and outputs
are ready at 14436.02; the earlier interval has no explanatory events and must
not all be attributed to queueing. Recorded model training totals 3927.63 s.

The same V17 manifest/cohort was used: 1,000 unique RNAs, 1,781 eligible profiles,
176,351 measured targets, seeds 42/123/2026, four variants and three splits.
Training used two Tesla T4 GPUs with DataParallel, at most 60 epochs, patience 8,
and min_delta 0.0001. Twenty-five runs stopped early and eleven exhausted the
budget without an early-stopping trigger. A best checkpoint may occur fewer
than eight epochs before stopping because tiny absolute improvements do not
reset material-improvement patience.

Clustered CV: real ViennaRNA 0.2042589064 ± 0.0006469717, Bi-LSTM
0.2201505432, zero 0.2282369311, shuffled 0.2301838175 (use CSV for full precision).
Train-only assay mean is 0.293335; median is 0.279594. All test fields remain
empty. The entire cohort has been exposed in development; no independent final
test is claimed. Confidence intervals are paired cluster bootstrap intervals
within each seed, conditional on the selected CV checkpoints, and do not
correct model-selection bias.

The original `training_curves.png` has a presentation limitation: after early
stopping the available-seed average changes composition. The review provides
`v18_training_curves_reviewed.png` with individual runs and means only while
all seeds are present. Original image and checksums are intentionally unchanged.
The selected-model length figure has only five short CV sequences per seed;
those are the same five RNAs across seeds and do not support a subgroup claim.

Verified: original checksums; all 36 weighted/macro CV metrics; all 36 bootstrap
rows; train-only mean/median constants; raw profile/sequence denominators;
all 36 saved checkpoints replayed on CPU (maximum GPU/CPU MAE difference
1.45e-8); independently downloaded source and cohort hashes; six fresh directed
MMseqs2 searches on real cohort sequences. The local search version was
18-8cc5c+ds-1 versus archived 13-45111+ds-2; it validates the frozen assignment
under the recorded thresholds, not bitwise regeneration of original clustering.

Run `python experiments/validate_development_archive.py results/kaggle-v18-df2d87b`.
See [the full review](../../docs/comprehensive_review_2026-10-22.md).
