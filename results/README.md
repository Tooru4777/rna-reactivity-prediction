# Reproducibility artifacts

Each reportable Kaggle run is archived in a versioned subdirectory. Archives
contain compact CSV, JSON, and PNG artifacts only; competition data, logs,
profile-level predictions, and model checkpoints are intentionally excluded.

## Reportable run

[`kaggle-v9-bff599f`](kaggle-v9-bff599f/README.md) is the current validated run.
It used one Tesla P100, real competition measurements, three validation seeds,
and training commit `bff599f5bfa59ccf6a2d202a35231a76384b843a`.

Required files for a final run:

- `run_manifest.json`: Git commit, cohort fingerprint, seeds, and hyperparameters
- `environment.json`: Python, PyTorch, CUDA, and GPU details
- `data_quality_report.json`: cohort grain, experiment coverage, duplicates, and target ranges
- `split_report.json`: row counts, unique sequences, and overlap audit
- `ablation_results.csv`: per-epoch training and CV curves
- `ablation_summary.csv`: per-seed checkpoint-selection and test results
- `ablation_aggregate.csv`: multi-seed CV and held-out test MAE
- `test_results.csv`: held-out test MAE for every seed and model
- `model_selection.json`: selection rule and selected model result
- `leakage_comparison.csv` and `leakage_aggregate.csv`: per-seed and aggregate random/grouped comparisons
- `vienna_comparison.csv` and `vienna_aggregate.csv`: per-seed and aggregate controlled ViennaRNA comparisons
- `error_by_length.csv`, `error_by_experiment.csv`, `error_by_structure.csv`
- `error_analysis_per_sequence.csv` (profile-level raw output remains on Kaggle)
- `training_curves.png` and `error_by_length.png`

Validate the archive with:

```bash
python experiments/validate_archive.py results/kaggle-v9-bff599f
```

The validator recomputes aggregate metrics, checks grouped sequence overlap,
reconciles error numerators and denominators with every held-out score, and
rejects data, logs, checkpoints, or profile-level outputs in the compact archive.
