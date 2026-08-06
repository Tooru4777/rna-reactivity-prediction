# Reproducibility artifacts

Each reportable Kaggle run is archived in a versioned subdirectory. The archive
contains compact CSV, JSON, and PNG artifacts only; competition data, logs, and
model checkpoints are intentionally excluded.

Required files for a final run:

- `run_manifest.json`: Git commit, cohort fingerprint, seeds, and hyperparameters
- `environment.json`: Python, PyTorch, CUDA, and GPU details
- `data_quality_report.json`: cohort grain, experiment coverage, duplicates, and target ranges
- `split_report.json`: row counts, unique sequences, and overlap audit
- `ablation_aggregate.csv`: multi-seed CV and held-out test MAE
- `test_results.csv`: held-out test MAE for every seed and model
- `model_selection.json`: selection rule and selected model result
- `leakage_aggregate.csv`: random versus grouped validation comparison
- `vienna_aggregate.csv`: controlled sequence-only versus ViennaRNA comparison
- `error_by_length.csv`, `error_by_experiment.csv`, `error_by_structure.csv`
- `error_analysis_per_sequence.csv` (optional detailed archive; profile-level raw output remains on Kaggle)
- `training_curves.png` and `error_by_length.png`

The final archive is added only after the padding-safe Kaggle run completes and
the headline values have been independently checked against the per-seed files.
